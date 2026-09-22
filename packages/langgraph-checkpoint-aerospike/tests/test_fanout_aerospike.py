import operator
import os
from typing import Annotated, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.aerospike import AerospikeSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

# Dedicated sets so this fan-out workload doesn't intermix with other tests'
# data — useful when poking around in `aql` between failures.
_FANOUT_SETS = ("lg_cp_fanout", "lg_cp_fanout_w", "lg_cp_fanout_meta")

# ---------- Graph definitions (copied from Mongo benchmark, sync only) ----------


class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]


class JokeInput(TypedDict):
    subject: str


class JokeOutput(TypedDict):
    jokes: list[str]


class JokeState(JokeInput, JokeOutput): ...


N_SUBJECTS = 10


def fanout_to_subgraph() -> StateGraph:
    # Subgraph nodes create a joke.
    def edit(state: JokeOutput) -> JokeOutput:
        return {"jokes": [f"{state['jokes'][0]}... and cats!"]}

    def generate(state: JokeInput) -> JokeOutput:
        return {"jokes": [f"Joke about the year {state['subject']}"]}

    def bump(state: JokeOutput) -> dict[str, list[str]]:
        return {"jokes": [state["jokes"][0] + " and the year before"]}

    def bump_loop(state: JokeOutput) -> str:
        # Repeat bump 3 times, then go to edit
        return "edit" if state["jokes"][0].endswith(" and the year before" * 3) else "bump"

    subgraph = StateGraph(JokeState)
    subgraph.add_node("edit", edit)
    subgraph.add_node("generate", generate)
    subgraph.add_node("bump", bump)
    subgraph.set_entry_point("generate")
    subgraph.add_edge("generate", "bump")
    subgraph.add_node("bump_loop", bump_loop)
    subgraph.add_conditional_edges("bump", bump_loop)
    subgraph.set_finish_point("edit")
    subgraphc = subgraph.compile()

    # Parent graph maps the joke-generating subgraph.
    def fanout(state: OverallState) -> list:
        return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

    parentgraph = StateGraph(OverallState)
    parentgraph.add_node("generate_joke", subgraphc)
    parentgraph.add_conditional_edges(START, fanout)
    parentgraph.add_edge("generate_joke", END)
    return parentgraph


# ---------- Fixtures ----------


@pytest.fixture
def joke_subjects() -> OverallState:
    years = [str(2025 - 10 * i) for i in range(N_SUBJECTS)]
    return {"subjects": years}


@pytest.fixture()
def aerospike_saver(session, aerospike_namespace, truncate_sets):
    """Yield an `AerospikeSaver` on dedicated sets, truncated each test."""
    truncate_sets(_FANOUT_SETS)
    saver = AerospikeSaver(
        session=session,
        namespace=aerospike_namespace,
        set_cp=_FANOUT_SETS[0],
        set_writes=_FANOUT_SETS[1],
        set_meta=_FANOUT_SETS[2],
    )
    try:
        yield saver
    finally:
        truncate_sets(_FANOUT_SETS)


@pytest.fixture(autouse=True)
def disable_langsmith() -> None:
    """Disable LangSmith tracing for all tests."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_API_KEY"] = ""


# ---------- Tests (sync only) ----------


def test_fanout_aerospike(joke_subjects: OverallState, aerospike_saver: AerospikeSaver) -> None:
    assert isinstance(aerospike_saver, BaseCheckpointSaver)

    graph = fanout_to_subgraph()
    graphc = graph.compile(checkpointer=aerospike_saver)

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "fanout_aerospike",
            "checkpoint_ns": "demo_fanout",
        }
    }

    # Sync streaming version
    out = list(graphc.stream(joke_subjects, config=config))

    # We expect one result per subject
    assert len(out) == N_SUBJECTS
    assert isinstance(out[0], dict)
    assert out[0].keys() == {"generate_joke"}
    assert set(out[0]["generate_joke"].keys()) == {"jokes"}

    # Every joke should end with the triple "year before" + "... and cats!"
    assert all(
        res["generate_joke"]["jokes"][0].endswith(f"{' and the year before' * 3}... and cats!")
        for res in out
    )


def test_custom_properties_aerospike(aerospike_saver: AerospikeSaver) -> None:
    state_graph = fanout_to_subgraph()

    assistant_id = "456"
    user_id = "789"
    # `assistant_id` is a known LangGraph-forwarded `configurable` key, so it
    # flows into checkpoint metadata automatically. Arbitrary extras like
    # `user_id` must be passed via `config["metadata"]`, which AerospikeSaver
    # merges into the checkpoint metadata inside `put()`.
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "custom_props_aerospike",
            "checkpoint_ns": "demo_fanout",
            "assistant_id": assistant_id,
        },
        "metadata": {"user_id": user_id},
    }

    compiled = state_graph.compile(checkpointer=aerospike_saver)

    # We don’t care about actual jokes here, just that a checkpoint is written.
    compiled.invoke(
        input={"subjects": [], "jokes": []},  # type: ignore[arg-type]
        config=config,
        stream_mode="values",
        debug=False,
    )

    checkpoint_tuple = aerospike_saver.get_tuple(config)
    assert checkpoint_tuple is not None

    assert checkpoint_tuple.metadata.get("assistant_id") == assistant_id
    assert checkpoint_tuple.metadata.get("user_id") == user_id
