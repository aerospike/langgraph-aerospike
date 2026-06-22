"""Fork from a checkpoint, end to end.

Run this against a local Aerospike to watch a LangGraph thread build up context
(it figures out the customer's order id from what they describe), resolve a
ticket, and then *rewind*. We read the thread's checkpoint history straight out
of Aerospike, rehydrate the checkpoint from *after* the order id was derived,
and resume from there with a corrected request. The new path reuses that
already-derived order id -- it is never looked up again. Then we use the saved
refund checkpoint to write a handoff note explaining what changed and what to do
now.

    uv run python cookbooks/fork-from-checkpoint/demo.py

The walkthrough pauses for you to press Enter between phases.

Phases:
    1. Run the original ticket: identify the order, then select refund.
    2. The customer changes their mind -- they want a replacement instead.
    3. Find the reuse point in Aerospike: list the history and rehydrate the
       checkpoint where the order id is known but nothing is resolved yet.
    4. Fork from it with the corrected request -- reusing the saved order id.
    5. Use the original refund checkpoint to write a handoff note.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

import aerospike
from agent import build_support_graph
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.aerospike import AerospikeSaver
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.graph.state import CompiledStateGraph

# Edit these values for your environment.
AEROSPIKE_HOST: str = "127.0.0.1"
AEROSPIKE_PORT: int = 3000
AEROSPIKE_NAMESPACE: str = "test"

# One customer conversation. The fork resumes from a prior checkpoint in this
# same thread, so the earlier checkpoints remain readable by id.
THREAD_ID: str = "support-thread-7842"
CHECKPOINT_NS: str = "production"

# The customer names the product ("headphones") -- that is the context the
# agent turns into an order id. The correction deliberately does NOT mention
# the product again, to prove the order id is remembered, not re-derived.
ORIGINAL_REQUEST: str = (
    "I bought a pair of AeroPro headphones that arrived broken. I'd like a refund."
)
CORRECTED_REQUEST: str = "On second thought, please send a replacement instead."


def _hr(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def _pause() -> None:
    """Wait for the user to press Enter before the next phase.

    Degrades gracefully to a no-op when stdin isn't interactive (e.g. piped),
    so the demo never hangs in CI.
    """
    with suppress(EOFError):
        input("\n  ... press Enter for the next step ...")


# === Step 6: Connect to Aerospike ===
@contextmanager
def _connect() -> Iterator[aerospike.Client]:
    """Open an Aerospike client, guaranteeing it is closed afterwards."""
    try:
        client = aerospike.client({"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}).connect()
    except aerospike.exception.AerospikeError as exc:
        raise SystemExit(
            f"Could not connect to Aerospike at {AEROSPIKE_HOST}:{AEROSPIKE_PORT}. "
            "Confirm your Aerospike server is running and update the constants "
            "at the top of demo.py if needed."
        ) from exc
    try:
        yield client
    finally:
        client.close()


# === Step 7: Build the checkpointer ===
def _build_checkpointer(client: aerospike.Client) -> AerospikeSaver:
    """Create an AerospikeSaver. No TTL here -- we want the history to persist."""
    return AerospikeSaver(client=client, namespace=AEROSPIKE_NAMESPACE)


def _values(tpl: CheckpointTuple | None) -> dict[str, Any]:
    """The graph state stored inside a checkpoint."""
    return tpl.checkpoint.get("channel_values", {}) if tpl else {}


def _short(checkpoint_id: str | None) -> str:
    """Last 12 chars of a checkpoint id.

    LangGraph ids are time-ordered, so they share a long *leading* prefix
    within one run -- the tail is what actually distinguishes them.
    """
    return checkpoint_id[-12:] if checkpoint_id else "-"


def _stage(step: int | None) -> str:
    """Human-readable labels for LangGraph's checkpoint step metadata."""
    return {
        -1: "thread start",
        0: "request received",
        1: "order identified",
        2: "intent classified",
        3: "decision selected",
    }.get(step, f"step {step}")


class SupportOutcome:
    """Tiny value object so the phases read cleanly."""

    def __init__(self, order_id: str | None, intent: str | None, resolution: str | None) -> None:
        self.order_id = order_id
        self.intent = intent
        self.resolution = resolution


# === Step 8: Run one turn of the thread to a resolution ===
def _resolve(graph: CompiledStateGraph, config: RunnableConfig, text: str) -> SupportOutcome:
    """Send one user turn and return the resulting order id, intent, resolution."""
    result = graph.invoke({"messages": [HumanMessage(text)]}, config)
    print(f"  user        > {text}")
    print(f"  assistant   > {result['messages'][-1].content}")
    return SupportOutcome(
        order_id=result["order_id"],
        intent=result["intent"],
        resolution=result["resolution"],
    )


def main() -> int:
    prod_config: RunnableConfig = {
        "configurable": {"thread_id": THREAD_ID, "checkpoint_ns": CHECKPOINT_NS}
    }

    with _connect() as client:
        saver = _build_checkpointer(client)
        graph = build_support_graph(saver)

        # Start clean so re-runs are reproducible regardless of prior state.
        saver.delete_thread(THREAD_ID)

        # === Phase 1: original run -- start blank, derive the order id, refund ===
        # Show the empty starting state first so it's clear nothing is known yet:
        # the agent has to derive the order id from what the customer describes.
        _hr("Phase 1 - The original ticket")
        print(
            "  starting state : order_id=None  intent=None  resolution=None  (nothing derived yet)"
        )
        print("  A ticket arrives. The agent works the order out from scratch:")
        original = _resolve(graph, prod_config, ORIGINAL_REQUEST)
        print(f"  order_id    : {original.order_id}  (derived from 'headphones')")
        print(f"  intent      : {original.intent}")
        print(f"  resolution  : {original.resolution}")

        # Remember exactly which checkpoint the original run ended on, so we can
        # use it later as the before-state for the handoff note.
        original_final = saver.get_tuple(prod_config)
        assert original_final is not None
        original_final_config = original_final.config

        # === Phase 2: the customer changes their mind ===
        # Introduce the new request *before* we touch the history, so the rewind
        # that follows has a clear motivation: reuse the order lookup we already
        # did instead of starting a brand-new conversation.
        _pause()
        _hr("Phase 2 - The customer changes their mind")
        print(f"  customer    > {CORRECTED_REQUEST}")
        print("  This request does not mention headphones, so a fresh run would not know")
        print("  which order to use. You need a saved checkpoint that already has order_id.")

        # === Step 9 / Phase 3: read history + rehydrate the order-id checkpoint ===
        # Now that we know what we want, look at what Aerospike saved and find
        # the moment worth rewinding to: the earliest checkpoint where the order
        # id is already known but nothing has been resolved yet.
        _pause()
        _hr("Phase 3 - Find the reuse point in Aerospike")
        print(f'  pending request: "{CORRECTED_REQUEST}"')
        print("  Need: order_id set, but no decision selected yet.")
        print()
        history: list[CheckpointTuple] = list(saver.list(prod_config))
        header = (
            f"  {'seq':<4}{'checkpoint':<14}{'saved after':<18}"
            f"{'order_id':<11}{'intent':<13}resolution"
        )
        print(header)
        for i, tpl in enumerate(reversed(history), start=1):
            cid = tpl.config["configurable"]["checkpoint_id"]
            values = _values(tpl)
            step = tpl.metadata.get("step")
            print(
                f"  {i:<4}{_short(cid):<14}{_stage(step):<18}"
                f"{str(values.get('order_id')):<11}{str(values.get('intent')):<13}"
                f"{values.get('resolution')}"
            )
        print(f"  -> {len(history)} checkpoints saved for this thread.")

        # === Step 10: rehydrate the post-lookup checkpoint ===
        fork_point = next(
            (
                tpl
                for tpl in reversed(history)
                if _values(tpl).get("order_id") and not _values(tpl).get("resolution")
            ),
            history[-1],
        )
        rehydrated = saver.get_tuple(fork_point.config)  # the sub-ms Aerospike read
        values = _values(rehydrated)
        target_id = fork_point.config["configurable"]["checkpoint_id"]
        print()
        print(f"  reuse point : {_short(target_id)} (order identified, not yet resolved)")
        print(
            f"  order_id    : {values.get('order_id')}  <- already derived, sitting in the checkpoint"
        )
        print("  -> Restored the checkpoint state; the order lookup node did not run again.")

        # === Step 11 / Phase 4: fork from it with the corrected request ===
        # The historical checkpoint_id in fork_point.config tells LangGraph to
        # resume from here. `identify_order` already ran, so it does NOT run
        # again -- the order id is reused from the checkpoint. Only `classify`
        # and the resolution node re-run, on the new message.
        _pause()
        _hr("Phase 4 - Fork from that checkpoint")
        fork_config = fork_point.config
        forked = _resolve(graph, fork_config, CORRECTED_REQUEST)
        reused = forked.order_id == original.order_id
        print(
            f"  order_id    : {forked.order_id}  ({'reused from the checkpoint' if reused else 'CHANGED!'})"
        )
        print(f"  intent      : {forked.intent}")
        print(f"  resolution  : {forked.resolution}")
        print("  -> The correction never mentioned the product, yet the order id carried over.")

        # === Step 12 / Phase 5: use the original decision to explain the change ===
        # Forking resumed from an earlier checkpoint, so the refund checkpoints
        # were never touched. Read the old checkpoint back out of Aerospike by
        # id and compare it with the corrected run. This makes the kept history
        # useful while keeping the corrected run as the outcome to use.
        _pause()
        _hr("Phase 5 - Use history to write the handoff note")
        original_after = _values(saver.get_tuple(original_final_config))
        messages = original_after.get("messages") or []
        based_on = messages[0].content if messages else "(unknown)"
        print("  Internal note from saved state:")
        print("    Customer first asked for a refund, then corrected the request to a replacement.")
        print(f"    Order id stayed {forked.order_id}; no second order lookup was needed.")
        print(f"    Use current outcome: {forked.resolution}")
        print()
        print("  Source fields read from Aerospike:")
        print(f'    original request : "{based_on}"')
        print(f"    original decision: {original_after.get('resolution')}")
        print(f'    corrected request: "{CORRECTED_REQUEST}"')
        print("  -> The kept checkpoint supplies the before-state for this handoff note.")

        # Guardrails: the old checkpoint must be untouched, the fork must have
        # rerouted, and the order id must have survived the rewind.
        if original_after.get("resolution") != original.resolution:
            print("  ! Historical checkpoint changed -- forking should not have touched it.")
            return 1
        if forked.resolution == original.resolution:
            print("  ! Fork produced the same resolution -- the new request did not reroute.")
            return 1
        if not reused:
            print("  ! Order id was not reused -- the rewind failed to preserve context.")
            return 1

        _pause()
        _hr("Result")
        print("  - Rewound to a saved checkpoint that already knew the order id and")
        print("    resumed down a new path -- reusing that context, not recomputing it.")
        print("  - The old final checkpoint is useful context for a handoff note:")
        print("    what the customer first asked for, what changed, and what to do now.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
