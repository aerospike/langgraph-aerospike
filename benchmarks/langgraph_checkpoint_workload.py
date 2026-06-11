"""Benchmark the LangGraph checkpointer IO patterns and graph runtime against every backend.

Defines a :class:`LangGraphIoWorkload` that exercises the LangGraph
checkpointer at two levels:

* **Raw IO** -- the four ``BaseCheckpointSaver`` calls a graph invocation
  triggers per node: ``put`` (write the new checkpoint), ``put_writes``
  (record pending channel writes), ``get_tuple`` (load the latest state on
  resume), and ``list`` (timeline / debug UIs).
* **Graph runtime** -- end-to-end ``graph.invoke`` against a fake LLM, using
  the saver as the production checkpointer. This stacks the langgraph
  orchestrator's overhead on top of the raw IO so you can isolate what's
  saver cost vs. what's graph cost. Four graph shapes:

  - ``<backend>_graph_sequential`` -- linear ``_SEQUENTIAL_NODE_COUNT``-node
    chain. One ``put``/``get`` cycle per node, serialized.
  - ``<backend>_graph_fanout`` -- entry -> ``_FANOUT_BRANCHES`` parallel
    branches -> join. ``put_writes`` contention from concurrent branches
    under one thread_id.
  - ``<backend>_graph_resume`` -- ``pre -> review -> post`` linear chain
    where ``review`` calls ``interrupt()``. Each measured call is two
    ``graph.invoke`` round-trips on the same ``thread_id``: the first
    pauses (persisting an ``__interrupt__`` checkpoint), the second
    ``Command(resume=...)``s and runs to completion. Exercises the
    suspended-state path (HITL) that ``sequential`` / ``fanout`` skip.
  - ``<backend>_graph_cyclic`` -- two-node cyclic loop
    (``agent <-> tool``) running ``_CYCLIC_ITERATIONS`` iterations.
    Models the IO shape of a canonical ReAct agent loop, though both
    nodes are the same fake-LLM stub so no real reasoning happens --
    the value is in the shape, not the semantics. Unlike the
    fixed-topology shapes above, per-invoke saver round-trips and
    final state size scale with the iteration count, so this is the
    shape that surfaces backends whose per-step cost grows with
    thread-history depth.

Each backend uses its production saver implementation:

- ``aerospike`` -> this repo's :class:`AerospikeSaver`
- ``postgres``  -> ``langgraph.checkpoint.postgres.PostgresSaver``
- ``redis``     -> ``langgraph.checkpoint.redis.RedisSaver``

Run with the default backend URIs and benchmark settings::

    uv run python benchmarks/langgraph_checkpoint_workload.py

Override connection URIs, QPS, worker/scheduler counts, runtime, and the
per-backend connection limits with CLI flags. Pass ``none`` for any URI to
disable that backend; the framework will simply skip it.
"""

from __future__ import annotations

import argparse
import copy
import urllib.parse
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from itertools import count
from typing import TYPE_CHECKING, Any

from ai_ecosystem_benchmark import BaseBenchmarkWorkload, BenchmarkRunner
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, Send, interrupt

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

# Number of distinct ``thread_id`` values the workload pre-populates per
# backend for the raw ops. Read/list calls round-robin over this pool so
# concurrent workers touch different rows/keys (avoids artificial
# single-row contention) while still hitting data that was seeded in
# ``setup``.
_THREAD_POOL_SIZE = 1024
# Fixed ``checkpoint_ns`` for the raw-op corpus. Non-empty so this
# benchmark's data is namespace-isolated from anything else that might
# already live in the database (langgraph defaults to ``""``).
_CHECKPOINT_NS = "bench"
# A single fake-LLM response is enough: every node returns the same canned
# AIMessage, so per-call latency is dominated by checkpoint I/O instead of
# LLM serialization. Keep it short so message-state payloads stay tiny.
_FAKE_LLM_RESPONSE = "benchmark"
# Linear-graph length. Each node triggers a full saver write cycle, so the
# per-call cost of ``<backend>_graph_sequential`` scales with this.
_SEQUENTIAL_NODE_COUNT = 5
# Fanout fan-width. Three concurrent branches per invocation is the
# smallest number that meaningfully exercises the saver's ``put_writes``
# contention path while keeping the graph at five executed nodes
# (entry + 3 branches + join).
_FANOUT_BRANCHES = 3
# Number of ``agent <-> tool`` round-trips the cyclic graph runs before
# terminating (modelled on the ReAct agent loop). Each iteration
# produces two checkpoint writes (one after ``agent``, one after
# ``tool``) and grows the message state by two entries, so per-invoke
# saver round-trips and final state size both scale linearly with this.
# Five matches the short end of real production agent loops; bump it to
# specifically probe deep-thread scaling.
_CYCLIC_ITERATIONS = 5

# Default backend URIs and runner settings. All overridable via CLI flags
# (see ``_parse_args``); edit these if you'd rather not pass flags.
_DEFAULT_AEROSPIKE_URI = "aerospike://10.100.0.4:3000/test"
_DEFAULT_POSTGRES_URI = "postgresql://bench:benchpassword@10.100.0.2:5432/bench"
_DEFAULT_REDIS_URI = "redis://10.100.0.5:6379/0"
_DEFAULT_QPS = 1000
_DEFAULT_WORKER_THREAD_COUNT = 8192
_DEFAULT_SCHEDULER_THREAD_COUNT = 8
_DEFAULT_RUNTIME_PER_FUNCTION = 30
_DEFAULT_AEROSPIKE_MAX_CONNS = 512
_DEFAULT_POSTGRES_POOL_SIZE = 512
_DEFAULT_REDIS_POOL_SIZE = 512


def _optional_uri(value: str) -> str | None:
    stripped = value.strip()
    if stripped.lower() in {"", "none", "null", "disabled"}:
        return None
    return stripped


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LangGraph checkpointer IO patterns and graph runtime "
            "against configured backends."
        )
    )
    parser.add_argument(
        "--aerospike-uri",
        type=_optional_uri,
        default=_DEFAULT_AEROSPIKE_URI,
        help=(
            "Aerospike URI: host:port[/namespace] or aerospike://host:port[/namespace]. "
            "Pass 'none' to disable. Defaults to %(default)s."
        ),
    )
    parser.add_argument(
        "--postgres-uri",
        type=_optional_uri,
        default=_DEFAULT_POSTGRES_URI,
        help="Postgres libpq URI. Pass 'none' to disable. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--redis-uri",
        type=_optional_uri,
        default=_DEFAULT_REDIS_URI,
        help=(
            "Redis URI. Requires Redis Stack/RediSearch + RedisJSON. "
            "Pass 'none' to disable. Defaults to %(default)s."
        ),
    )
    parser.add_argument(
        "--qps",
        type=_positive_int,
        default=_DEFAULT_QPS,
        help="Target queries per second. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--worker-thread-count",
        type=_positive_int,
        default=_DEFAULT_WORKER_THREAD_COUNT,
        help="Worker thread count used by the benchmark runner. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--scheduler-thread-count",
        type=_positive_int,
        default=_DEFAULT_SCHEDULER_THREAD_COUNT,
        help="Scheduler thread count used by the benchmark runner. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--runtime-per-function",
        type=_positive_int,
        default=_DEFAULT_RUNTIME_PER_FUNCTION,
        help="Seconds to run each benchmark method. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--aerospike-max-conns",
        type=_positive_int,
        default=_DEFAULT_AEROSPIKE_MAX_CONNS,
        help="Aerospike max_conns_per_node for the benchmark client. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--postgres-pool-size",
        type=_positive_int,
        default=_DEFAULT_POSTGRES_POOL_SIZE,
        help="Pre-warmed psycopg connection pool size. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--redis-pool-size",
        type=_positive_int,
        default=_DEFAULT_REDIS_POOL_SIZE,
        help="Redis BlockingConnectionPool max connection count. Defaults to %(default)s.",
    )
    return parser.parse_args()


# ----- medium-sized payload template (~3 KB serialized) -----
#
# Roughly mirrors a typical conversational-agent step: short chat history
# with one tool round-trip + a couple of retrieved documents + agent
# scratchpad + metadata. Sized to land between the historic toy payload
# (~150 B) and a heavy RAG step (~15-20 KB); 3 KB is large enough to
# exercise serialization paths and stress per-record size on Aerospike's
# map bins, while staying well within the per-call budget at high QPS.
#
# Built once at module scope and deep-copied per call so per-call cost is
# bounded by ``deepcopy`` + a UUID/ts swap rather than dict construction.
# Content is deterministic (no per-call randomness) so the serialized
# size is stable across runs and across backends -- payload-size variance
# otherwise leaks into latency variance.
_MEDIUM_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a helpful assistant. " * 4},
    {"role": "user", "content": "I need help analyzing my Q3 sales data. " * 2},
    {
        "role": "assistant",
        "content": "Looking up the figures now. " * 3,
        "tool_calls": [
            {
                "id": "call_1",
                "name": "fetch_sales",
                "args": {"quarter": "Q3", "year": 2025, "regions": ["NA", "EU", "APAC"]},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Sales data for Q3 2025: " + ("{revenue: 1234567, units: 8910} " * 6),
    },
    {"role": "assistant", "content": "Based on the Q3 data, the key trends are: " * 6},
]

# BaseMessage form of the same conversation, used as the initial
# ``messages`` channel for the graph benchmarks. Production langgraph
# apps almost never invoke a graph from a single-message cold start; the
# thread already has accumulated history. Pre-loading this list makes
# every checkpoint write the langgraph runtime issues during ``invoke``
# include ~1 KB of message state, putting the graph benchmarks in the
# same size regime as the raw-IO ones.
#
# Built as BaseMessage instances once at module scope (not dicts) so the
# ``add_messages`` reducer doesn't pay per-call dict-to-BaseMessage
# conversion cost. Sharing instances across threads is safe: each call
# uses a fresh ``thread_id`` so there's no cross-thread id collision in
# the reducer's dedup logic.
_MEDIUM_INITIAL_MESSAGES: list[BaseMessage] = [
    SystemMessage(content="You are a helpful assistant. " * 4),
    HumanMessage(content="I need help analyzing my Q3 sales data. " * 2),
    AIMessage(
        content="Looking up the figures now. " * 3,
        tool_calls=[
            {
                "id": "call_1",
                "name": "fetch_sales",
                "args": {"quarter": "Q3", "year": 2025, "regions": ["NA", "EU", "APAC"]},
            }
        ],
    ),
    ToolMessage(
        content="Sales data for Q3 2025: " + ("{revenue: 1234567, units: 8910} " * 6),
        tool_call_id="call_1",
    ),
    AIMessage(content="Based on the Q3 data, the key trends are: " * 6),
]

_MEDIUM_RETRIEVED_DOCS: list[dict[str, Any]] = [
    {
        "id": f"doc-{i}",
        "content": "Document content paragraph for the retrieval result. " * 6,
        "metadata": {"source": f"sales_report_{i}.pdf", "page": i, "section": "summary"},
        "score": 0.95 - i * 0.1,
    }
    for i in range(2)
]

# Full ``Checkpoint`` skeleton. Tracks more than the historic toy payload:
# multiple channels, nested ``versions_seen`` mirroring multi-node graphs,
# and a non-empty ``pending_sends`` so any saver code that special-cases
# the empty list still gets exercised. The shape is what matters for
# coverage; the field-level content is sized so the whole thing serialises
# to ~3 KB.
_MEDIUM_CHECKPOINT_TEMPLATE: dict[str, Any] = {
    "v": 4,
    "id": "",  # populated per call
    "ts": "",  # populated per call
    "channel_values": {
        "messages": _MEDIUM_MESSAGES,
        "retrieved_docs": _MEDIUM_RETRIEVED_DOCS,
        "scratchpad": {
            "current_step": "analyzing_regional_data",
            "intermediate_findings": ["NA strongest", "APAC growing", "EU flat"],
            "next_actions": ["fetch_forecast_model", "compute_projections"],
        },
        "metadata": {
            "trace_id": "trace-bench-medium",
            "model": "gpt-4o-mini",
            "tokens_in": 1245,
            "tokens_out": 382,
            "latency_ms": 521,
        },
    },
    "channel_versions": {
        "__start__": 1,
        "messages": 5,
        "retrieved_docs": 2,
        "scratchpad": 3,
        "metadata": 4,
    },
    "versions_seen": {
        "__start__": {"__start__": 1},
        "agent": {"messages": 4, "scratchpad": 2},
        "tools": {"messages": 3, "retrieved_docs": 1},
    },
    "pending_sends": [{"node": "agent", "args": {"step": "consolidate"}}],
    "updated_channels": None,
}

# Representative ``put_writes`` payload (~600 B). A node typically writes
# the new assistant message (with tool calls) plus a small scratchpad
# update -- meaningfully larger than the historic ``[("messages", "ack")]``
# few bytes, but proportional to the medium checkpoint above. Stored once
# so the per-call cost in ``_do_checkpoint_put_writes`` is the saver IO,
# not the literal construction.
_REALISTIC_WRITE: list[tuple[str, Any]] = [
    (
        "messages",
        {
            "role": "assistant",
            "content": "Computed result for the requested breakdown: " * 8,
            "tool_calls": [
                {
                    "id": "call_x",
                    "name": "next_step",
                    "args": {"params": list(range(8)), "note": "fan-out"},
                }
            ],
        },
    ),
    (
        "scratchpad",
        {
            "current_step": "post_tool",
            "notes": "consolidating partial results " * 4,
        },
    ),
]


def _make_checkpoint() -> dict[str, Any]:
    """Build a medium-sized LangGraph ``Checkpoint`` payload (~3 KB).

    Deep-copies the module-scope ``_MEDIUM_CHECKPOINT_TEMPLATE`` and
    stamps a fresh UUID + ISO timestamp so each ``put`` writes a distinct
    row rather than overwriting in place. The deepcopy is intentional:
    every saver mutates the dict it's handed during serialization, so a
    shared reference would corrupt the template across workers and
    produce nondeterministic payload sizes.
    """
    cp = copy.deepcopy(_MEDIUM_CHECKPOINT_TEMPLATE)
    cp["id"] = str(uuid.uuid4())
    cp["ts"] = datetime.now(tz=timezone.utc).isoformat()
    return cp


def _cfg(thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
    configurable: dict[str, Any] = {
        "thread_id": thread_id,
        "checkpoint_ns": _CHECKPOINT_NS,
    }
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _fanout_dispatch(state: MessagesState) -> list[Send]:
    """Dispatch ``_FANOUT_BRANCHES`` parallel Sends to the ``branch`` node.

    Each Send forwards the entry node's accumulated message history plus
    a per-branch marker as starting state, so every branch processes a
    realistic-sized conversation (not a single ``HumanMessage``). The
    ``add_messages`` reducer on ``MessagesState.messages`` merges the
    branch outputs back together when the graph converges at ``join``.
    """
    history: list[BaseMessage] = list(state["messages"])
    payload: dict[str, Any] = {"messages": [*history, HumanMessage(content="branch")]}
    return [Send("branch", payload) for _ in range(_FANOUT_BRANCHES)]


def _cyclic_router(state: MessagesState) -> str:
    """Terminate the cyclic loop after a fixed number of iterations.

    Each iteration appends two messages (one from ``agent``, one from
    ``tool``), so we can count iterations by counting messages added on
    top of the seeded history. Count-based termination keeps per-invoke
    cost deterministic -- a random/LLM-driven router would let
    variance leak into latency measurements.
    """
    target = len(_MEDIUM_INITIAL_MESSAGES) + _CYCLIC_ITERATIONS * 2
    return END if len(state["messages"]) >= target else "tool"


def _human_review_node(state: MessagesState) -> dict[str, list[BaseMessage]]:
    """Suspend the graph at a human-in-the-loop checkpoint.

    On the first invocation ``interrupt()`` raises ``GraphInterrupt``; the
    langgraph runtime catches it, persists the current state plus an
    ``__interrupt__`` marker through the checkpointer, and ``graph.invoke``
    returns to the caller without finishing. On the second invocation
    with ``Command(resume=value)``, the runtime re-enters this node, and
    ``interrupt()`` returns ``value`` instead of raising. The node has no
    side effects before the ``interrupt()`` call, so the re-execution
    semantics are safe.
    """
    decision = interrupt({"question": "approve?"})
    return {"messages": [HumanMessage(content=f"decision:{decision}")]}


class LangGraphIoWorkload(BaseBenchmarkWorkload):
    """Benchmark workload exercising the IO LangGraph itself issues plus full graph runs.

    The framework discovers methods by prefix, so each backend gets eight
    methods:

    * Four ``<backend>_checkpoint_<op>`` raw-IO benchmarks that call the
      saver directly with a medium-sized ~3 KB payload (``put``,
      ``put_writes``, ``get_tuple``, ``list``).
    * Four ``<backend>_graph_<shape>`` end-to-end benchmarks that invoke a
      compiled :class:`StateGraph` whose checkpointer is the saver
      (``sequential``, ``fanout``, ``resume``, ``cyclic``). Each invoke is
      seeded with ``_MEDIUM_INITIAL_MESSAGES`` (~1 KB of chat history as
      BaseMessage instances) so the langgraph runtime's per-step
      checkpoint writes reflect a realistic conversation-mid-thread
      state, not a single short HumanMessage.

    Both layers share a single saver instance per backend so resource use
    matches a real LangGraph application's. A couple of comparability
    caveats worth keeping in mind when reading results:

    * ``graph_resume``'s per-call cost spans **two** ``graph.invoke``
      round-trips (pause + resume), so its numbers shouldn't be compared
      1:1 against the single-invoke graph methods.
    * ``graph_cyclic``'s per-call cost spans ``2 * _CYCLIC_ITERATIONS``
      saver round-trips (vs. 5 for ``sequential``, 5 for ``fanout``,
      4 for ``resume``); higher per-call latency is expected and
      reflects the longer trajectory, not a per-step regression.
    """

    def __init__(
        self,
        aerospike_connection_string: str | None = None,
        postgres_connection_string: str | None = None,
        redis_connection_string: str | None = None,
        aerospike_max_conns: int = _DEFAULT_AEROSPIKE_MAX_CONNS,
        postgres_pool_size: int = _DEFAULT_POSTGRES_POOL_SIZE,
        redis_pool_size: int = _DEFAULT_REDIS_POOL_SIZE,
    ) -> None:
        super().__init__(
            aerospike_connection_string=aerospike_connection_string,
            postgres_connection_string=postgres_connection_string,
            redis_connection_string=redis_connection_string,
        )
        self._aerospike_max_conns = aerospike_max_conns
        self._postgres_pool_size = postgres_pool_size
        self._redis_pool_size = redis_pool_size

        # ``itertools.count`` is atomic in CPython so worker threads can
        # bump these without an extra lock.
        self._round_robin = count()
        self._task_seq = count()

        self._aerospike_client: Any | None = None
        self._aerospike_saver: Any | None = None
        self._aerospike_graph_sequential: CompiledStateGraph | None = None
        self._aerospike_graph_fanout: CompiledStateGraph | None = None
        self._aerospike_graph_resume: CompiledStateGraph | None = None
        self._aerospike_graph_cyclic: CompiledStateGraph | None = None

        # Postgres uses a real connection pool so concurrent worker threads
        # don't serialise through a single connection. The pool itself is
        # the resource we need to close in teardown.
        self._postgres_pool: Any | None = None
        self._postgres_saver: Any | None = None
        self._postgres_graph_sequential: CompiledStateGraph | None = None
        self._postgres_graph_fanout: CompiledStateGraph | None = None
        self._postgres_graph_resume: CompiledStateGraph | None = None
        self._postgres_graph_cyclic: CompiledStateGraph | None = None

        # RedisSaver can build its own unbounded client, but benchmark runs
        # need an explicit pool so the client cannot exhaust file handles.
        self._redis_client: Any | None = None
        self._redis_saver: Any | None = None
        self._redis_graph_sequential: CompiledStateGraph | None = None
        self._redis_graph_fanout: CompiledStateGraph | None = None
        self._redis_graph_resume: CompiledStateGraph | None = None
        self._redis_graph_cyclic: CompiledStateGraph | None = None

        # ``(backend, thread_idx) -> latest checkpoint_id``. Refreshed on
        # every successful ``put`` so ``put_writes`` always targets a
        # checkpoint that actually exists.
        self._latest_checkpoint_id: dict[tuple[str, int], str] = {}

        # Shared fake LLM. ``FakeListChatModel`` cycles deterministically
        # through ``responses`` so each node returns the same canned reply
        # in negligible time -- node compute drops out of the measurement.
        self._llm = FakeListChatModel(responses=[_FAKE_LLM_RESPONSE])

    # ---------- lifecycle ----------

    def setup(self) -> None:
        if self.is_aerospike_enabled():
            self._setup_aerospike()
        if self.is_postgres_enabled():
            self._setup_postgres()
        if self.is_redis_enabled():
            self._setup_redis()

    def between_benchmarks(self) -> None:
        return None

    def teardown(self) -> None:
        if self._aerospike_client is not None:
            with suppress(Exception):
                self._aerospike_client.close()
        if self._postgres_pool is not None:
            self._postgres_pool.close()
        if self._redis_client is not None:
            with suppress(Exception):
                self._redis_client.close()
            connection_pool = getattr(self._redis_client, "connection_pool", None)
            if connection_pool is not None:
                with suppress(Exception):
                    connection_pool.disconnect()

    # ---------- per-backend setup ----------

    def _setup_aerospike(self) -> None:
        import aerospike
        from langgraph.checkpoint.aerospike import AerospikeSaver

        assert self.aerospike_connection_string is not None
        # Tolerate both ``aerospike://host:port/<namespace>`` and the bare
        # ``host:port/<namespace>`` form people typically paste in -
        # ``urlparse`` mis-classifies the latter (treats ``host`` as the
        # scheme), so re-parse with an explicit scheme prepended.
        raw_uri = self.aerospike_connection_string
        if "://" not in raw_uri:
            raw_uri = f"aerospike://{raw_uri}"
        url = urllib.parse.urlparse(raw_uri)
        host = url.hostname or "localhost"
        port = url.port or 3000
        # Allow ``aerospike://host:port/<namespace>`` to override the
        # default ``test`` namespace.
        namespace = url.path.lstrip("/") or "test"

        # The default ``max_conns_per_node`` is 100, which is far too small
        # for a benchmark with high ``worker_thread_count``: each in-flight
        # call holds a connection for its whole duration, and ops like
        # ``list`` issue 20+ round-trips per call. Bump the ceiling so the
        # pool can grow to whatever the worker pool actually needs.
        self._aerospike_client = aerospike.client(
            {
                "hosts": [(host, port)],
                "max_conns_per_node": self._aerospike_max_conns,
            }
        ).connect()
        self._aerospike_saver = AerospikeSaver(client=self._aerospike_client, namespace=namespace)
        self._truncate_aerospike(self._aerospike_client, namespace, self._aerospike_saver)
        self._aerospike_graph_sequential = self._build_sequential_graph(self._aerospike_saver)
        self._aerospike_graph_fanout = self._build_fanout_graph(self._aerospike_saver)
        self._aerospike_graph_resume = self._build_resume_graph(self._aerospike_saver)
        self._aerospike_graph_cyclic = self._build_cyclic_graph(self._aerospike_saver)
        self._seed("aerospike", self._aerospike_saver)

    def _setup_postgres(self) -> None:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        assert self.postgres_connection_string is not None
        # ``PostgresSaver.from_conn_string`` returns a saver wrapping a
        # *single* psycopg connection, which serialises every concurrent
        # worker through one socket and dominates the measured latency at
        # nontrivial qps. Use a real connection pool instead so the
        # benchmark reflects Postgres' actual throughput, not queueing
        # delay. ``autocommit=True`` and ``row_factory=dict_row`` are
        # required by ``PostgresSaver`` (the saver accesses rows by name,
        # and migrations in ``.setup()`` won't commit without autocommit).
        # Pool sized to absorb ``queries_per_second * per_call_latency``
        # in-flight calls without queueing. The Terraform-provisioned
        # Postgres has ``max_connections = 200``, so leave headroom.
        # Crucially, ``min_size == max_size`` so ``wait()`` pre-warms the
        # full pool before the benchmark starts; otherwise on-demand
        # connection growth (TCP + TLS + Postgres handshake, serialised
        # by the pool's grow worker) leaks into the measured latency.
        self._postgres_pool = ConnectionPool(
            self.postgres_connection_string,
            min_size=self._postgres_pool_size,
            max_size=self._postgres_pool_size,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=True,
        )
        self._postgres_pool.wait()
        self._postgres_saver = PostgresSaver(self._postgres_pool)
        # Idempotent migration step; required on first use of a fresh DB.
        self._postgres_saver.setup()
        self._truncate_postgres(self._postgres_pool)
        self._postgres_graph_sequential = self._build_sequential_graph(self._postgres_saver)
        self._postgres_graph_fanout = self._build_fanout_graph(self._postgres_saver)
        self._postgres_graph_resume = self._build_resume_graph(self._postgres_saver)
        self._postgres_graph_cyclic = self._build_cyclic_graph(self._postgres_saver)
        self._seed("postgres", self._postgres_saver)

    def _setup_redis(self) -> None:
        from langgraph.checkpoint.redis import RedisSaver
        from redis import BlockingConnectionPool, Redis

        assert self.redis_connection_string is not None
        # redis-py's default pool can grow very large under a huge worker
        # pool. Bound it explicitly so benchmark pressure shows up as Redis
        # latency instead of OS-level "too many open files" failures.
        redis_pool = BlockingConnectionPool.from_url(
            self.redis_connection_string,
            max_connections=self._redis_pool_size,
            timeout=30,
        )
        self._redis_client = Redis(connection_pool=redis_pool)
        # Drop all prior-run keys (and RediSearch indices) before the saver
        # rebuilds its indices, so every run starts from an empty DB.
        self._truncate_redis(self._redis_client)
        self._redis_saver = RedisSaver(redis_client=self._redis_client)
        # Creates RediSearch indices used for ``list`` / filtered queries.
        self._redis_saver.setup()
        self._redis_graph_sequential = self._build_sequential_graph(self._redis_saver)
        self._redis_graph_fanout = self._build_fanout_graph(self._redis_saver)
        self._redis_graph_resume = self._build_resume_graph(self._redis_saver)
        self._redis_graph_cyclic = self._build_cyclic_graph(self._redis_saver)
        self._seed("redis", self._redis_saver)

    # ---------- per-backend truncation ----------

    def _truncate_aerospike(self, client: Any, namespace: str, saver: Any) -> None:
        """Server-side truncate of every set the saver writes to.

        ``client.truncate`` is the idiomatic Aerospike wipe: one op per set,
        no per-record round-trips. Secondary indexes survive truncate, so the
        ``thread_id`` index created in ``AerospikeSaver.__init__`` stays valid.
        """
        import aerospike.exception

        for set_name in (saver.set_cp, saver.set_writes, saver.set_meta):
            with suppress(aerospike.exception.AerospikeError):
                client.truncate(namespace, set_name, 0)

    def _truncate_postgres(self, pool: Any) -> None:
        """Wipe the LangGraph checkpoint tables, preserving the schema.

        Truncates data rows only; the ``checkpoint_migrations`` table (schema
        versioning) is left intact so ``PostgresSaver.setup()`` stays a no-op
        on the next run. ``RESTART IDENTITY CASCADE`` clears any sequences and
        dependent rows in one statement.
        """
        with pool.connection() as conn:
            conn.execute(
                "TRUNCATE checkpoints, checkpoint_blobs, checkpoint_writes RESTART IDENTITY CASCADE"
            )

    def _truncate_redis(self, client: Any) -> None:
        """Flush the selected Redis logical DB (keys + RediSearch indices).

        ``FLUSHDB`` drops every key and index in the connection's DB, so the
        subsequent ``RedisSaver.setup()`` rebuilds its RediSearch indices from
        a clean slate.
        """
        client.flushdb()

    def _seed(self, backend: str, saver: Any) -> None:
        """Pre-populate one checkpoint per thread on a freshly truncated DB.

        The per-backend truncation in ``_setup_*`` already wipes all prior-run
        data (raw ``bench-*`` threads plus the per-invocation graph threads),
        so this only has to lay down the fixed ``bench-*`` corpus the read /
        list / put_writes methods round-robin over.
        """
        for idx in range(_THREAD_POOL_SIZE):
            thread_id = f"bench-{idx}"
            checkpoint = _make_checkpoint()
            saver.put(_cfg(thread_id), checkpoint, {}, {})
            self._latest_checkpoint_id[(backend, idx)] = checkpoint["id"]

    # ---------- graph construction ----------

    def _build_sequential_graph(self, saver: Any) -> CompiledStateGraph:
        graph: StateGraph = StateGraph(MessagesState)
        node_names = [f"node_{i}" for i in range(1, _SEQUENTIAL_NODE_COUNT + 1)]
        for name in node_names:
            graph.add_node(name, self._llm_node)
        graph.add_edge(START, node_names[0])
        for prev_name, next_name in zip(node_names, node_names[1:], strict=False):
            graph.add_edge(prev_name, next_name)
        graph.add_edge(node_names[-1], END)
        return graph.compile(checkpointer=saver)

    def _build_fanout_graph(self, saver: Any) -> CompiledStateGraph:
        graph: StateGraph = StateGraph(MessagesState)
        graph.add_node("entry", self._llm_node)
        graph.add_node("branch", self._llm_node)
        graph.add_node("join", self._llm_node)
        graph.add_edge(START, "entry")
        graph.add_conditional_edges("entry", _fanout_dispatch, ["branch"])
        graph.add_edge("branch", "join")
        graph.add_edge("join", END)
        return graph.compile(checkpointer=saver)

    def _build_resume_graph(self, saver: Any) -> CompiledStateGraph:
        """``pre -> review -> post`` chain where ``review`` calls ``interrupt()``.

        Smallest shape that produces (1) a ``put`` persisting an
        ``__interrupt__``, (2) a ``get_tuple`` on resume that returns a
        checkpoint with non-empty pending writes, and (3) a second
        ``invoke`` against an existing ``thread_id`` -- none of which the
        ``sequential`` or ``fanout`` graphs exercise.
        """
        graph: StateGraph = StateGraph(MessagesState)
        graph.add_node("pre", self._llm_node)
        graph.add_node("review", _human_review_node)
        graph.add_node("post", self._llm_node)
        graph.add_edge(START, "pre")
        graph.add_edge("pre", "review")
        graph.add_edge("review", "post")
        graph.add_edge("post", END)
        return graph.compile(checkpointer=saver)

    def _build_cyclic_graph(self, saver: Any) -> CompiledStateGraph:
        """Two-node cyclic loop: ``agent <-> tool`` for ``_CYCLIC_ITERATIONS``.

        Models the IO shape of a canonical ReAct agent (reason -> act ->
        observe -> reason), though both nodes are bound to the same
        fake-LLM stub -- the benchmark cares about saver round-trips,
        not agent semantics. Unlike the other graphs whose per-invoke
        saver round-trip count is fixed by topology, this one produces
        ``2 * _CYCLIC_ITERATIONS`` writes per invoke, and the state
        grows by 2 messages per iteration -- so the last write is
        meaningfully larger than the first. That per-iteration growth
        is the property this graph probes: backends whose write cost
        scales with thread-history depth (e.g. Aerospike's map bin
        grows on each ``put``) show it here and not in the fixed-depth
        graphs.
        """
        graph: StateGraph = StateGraph(MessagesState)
        graph.add_node("agent", self._llm_node)
        # ``agent`` / ``tool`` names mirror the ReAct topology even
        # though both nodes are the same fake-LLM stub. The benchmark
        # cares about saver round-trips, not tool semantics.
        graph.add_node("tool", self._llm_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", _cyclic_router, ["tool", END])
        graph.add_edge("tool", "agent")
        return graph.compile(checkpointer=saver)

    def _llm_node(self, state: MessagesState) -> dict[str, list[BaseMessage]]:
        reply = self._llm.invoke(state["messages"])
        return {"messages": [reply]}

    # ---------- shared helpers used by the per-backend benchmark methods ----

    def _next_thread_index(self) -> int:
        return next(self._round_robin) % _THREAD_POOL_SIZE

    def _do_checkpoint_put(self, backend: str, saver: Any) -> None:
        idx = self._next_thread_index()
        thread_id = f"bench-{idx}"
        checkpoint = _make_checkpoint()
        saver.put(_cfg(thread_id), checkpoint, {}, {})
        # Keep the seed map current so later ``put_writes`` calls hit a
        # checkpoint that actually exists for this thread.
        self._latest_checkpoint_id[(backend, idx)] = checkpoint["id"]

    def _do_checkpoint_put_writes(self, backend: str, saver: Any) -> None:
        idx = self._next_thread_index()
        thread_id = f"bench-{idx}"
        checkpoint_id = self._latest_checkpoint_id[(backend, idx)]
        cfg = _cfg(thread_id, checkpoint_id=checkpoint_id)
        # Unique ``task_id`` per call avoids artificial map-overwrite
        # contention on Aerospike (where writes are stored in a Map bin).
        task_id = f"task-{next(self._task_seq)}"
        # ``_REALISTIC_WRITE`` is read-only on the saver side, so passing
        # the shared module-scope reference is safe and avoids per-call
        # allocation cost showing up in the measurement.
        saver.put_writes(cfg, _REALISTIC_WRITE, task_id)

    def _do_checkpoint_get_tuple(self, saver: Any) -> None:
        thread_id = f"bench-{self._next_thread_index()}"
        saver.get_tuple(_cfg(thread_id))

    def _do_checkpoint_list(self, saver: Any) -> None:
        thread_id = f"bench-{self._next_thread_index()}"
        # Materialise the iterator inside the timed call so the IO is
        # actually performed, not just deferred.
        list(saver.list(_cfg(thread_id), limit=10))

    def _do_graph_sequential(self, backend: str, graph: CompiledStateGraph) -> None:
        # Per-invocation thread_id keeps each call isolated. UUID rather
        # than counter so concurrent workers can't collide on the same
        # graph state row. Seed with ``_MEDIUM_INITIAL_MESSAGES`` so each
        # node's checkpoint write carries realistic message-state size
        # rather than a single short HumanMessage.
        thread_id = f"bench-{backend}-seq-{uuid.uuid4().hex}"
        graph.invoke(
            {"messages": _MEDIUM_INITIAL_MESSAGES},
            config={"configurable": {"thread_id": thread_id}},
        )

    def _do_graph_fanout(self, backend: str, graph: CompiledStateGraph) -> None:
        thread_id = f"bench-{backend}-fanout-{uuid.uuid4().hex}"
        graph.invoke(
            {"messages": _MEDIUM_INITIAL_MESSAGES},
            config={"configurable": {"thread_id": thread_id}},
        )

    def _do_graph_cyclic(self, backend: str, graph: CompiledStateGraph) -> None:
        # Per-invocation thread_id so the loop's accumulated state never
        # carries over to subsequent calls. Each invoke runs
        # ``_CYCLIC_ITERATIONS`` agent<->tool cycles, producing
        # ``2 * _CYCLIC_ITERATIONS`` per-step checkpoint writes; the final
        # write is roughly 50% larger than the first because state grows
        # monotonically inside the loop.
        thread_id = f"bench-{backend}-cyclic-{uuid.uuid4().hex}"
        graph.invoke(
            {"messages": _MEDIUM_INITIAL_MESSAGES},
            config={"configurable": {"thread_id": thread_id}},
        )

    def _do_graph_resume(self, backend: str, graph: CompiledStateGraph) -> None:
        # Two invokes on the same thread_id: first pauses at the
        # ``interrupt()`` in ``review`` (the runtime persists state +
        # ``__interrupt__``); second carries ``Command(resume=...)`` so
        # the runtime loads the suspended state, hands the resume value
        # back to ``interrupt()``, and runs ``post`` to completion.
        # Measured latency therefore covers pause-and-persist plus
        # load-and-resume -- don't compare 1:1 against the other graph
        # methods which do a single invoke.
        thread_id = f"bench-{backend}-resume-{uuid.uuid4().hex}"
        cfg: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        graph.invoke({"messages": _MEDIUM_INITIAL_MESSAGES}, config=cfg)
        graph.invoke(Command(resume="ok"), config=cfg)

    # ---------- aerospike methods ----------

    def aerospike_checkpoint_put(self) -> None:
        self._do_checkpoint_put("aerospike", self._aerospike_saver)

    def aerospike_checkpoint_put_writes(self) -> None:
        self._do_checkpoint_put_writes("aerospike", self._aerospike_saver)

    def aerospike_checkpoint_get_tuple(self) -> None:
        self._do_checkpoint_get_tuple(self._aerospike_saver)

    def aerospike_checkpoint_list(self) -> None:
        self._do_checkpoint_list(self._aerospike_saver)

    def aerospike_graph_sequential(self) -> None:
        assert self._aerospike_graph_sequential is not None
        self._do_graph_sequential("aerospike", self._aerospike_graph_sequential)

    def aerospike_graph_fanout(self) -> None:
        assert self._aerospike_graph_fanout is not None
        self._do_graph_fanout("aerospike", self._aerospike_graph_fanout)

    def aerospike_graph_resume(self) -> None:
        assert self._aerospike_graph_resume is not None
        self._do_graph_resume("aerospike", self._aerospike_graph_resume)

    def aerospike_graph_cyclic(self) -> None:
        assert self._aerospike_graph_cyclic is not None
        self._do_graph_cyclic("aerospike", self._aerospike_graph_cyclic)

    # ---------- postgres methods ----------

    def postgres_checkpoint_put(self) -> None:
        self._do_checkpoint_put("postgres", self._postgres_saver)

    def postgres_checkpoint_put_writes(self) -> None:
        self._do_checkpoint_put_writes("postgres", self._postgres_saver)

    def postgres_checkpoint_get_tuple(self) -> None:
        self._do_checkpoint_get_tuple(self._postgres_saver)

    def postgres_checkpoint_list(self) -> None:
        self._do_checkpoint_list(self._postgres_saver)

    def postgres_graph_sequential(self) -> None:
        assert self._postgres_graph_sequential is not None
        self._do_graph_sequential("postgres", self._postgres_graph_sequential)

    def postgres_graph_fanout(self) -> None:
        assert self._postgres_graph_fanout is not None
        self._do_graph_fanout("postgres", self._postgres_graph_fanout)

    def postgres_graph_resume(self) -> None:
        assert self._postgres_graph_resume is not None
        self._do_graph_resume("postgres", self._postgres_graph_resume)

    def postgres_graph_cyclic(self) -> None:
        assert self._postgres_graph_cyclic is not None
        self._do_graph_cyclic("postgres", self._postgres_graph_cyclic)

    # ---------- redis methods ----------

    def redis_checkpoint_put(self) -> None:
        self._do_checkpoint_put("redis", self._redis_saver)

    def redis_checkpoint_put_writes(self) -> None:
        self._do_checkpoint_put_writes("redis", self._redis_saver)

    def redis_checkpoint_get_tuple(self) -> None:
        self._do_checkpoint_get_tuple(self._redis_saver)

    def redis_checkpoint_list(self) -> None:
        self._do_checkpoint_list(self._redis_saver)

    def redis_graph_sequential(self) -> None:
        assert self._redis_graph_sequential is not None
        self._do_graph_sequential("redis", self._redis_graph_sequential)

    def redis_graph_fanout(self) -> None:
        assert self._redis_graph_fanout is not None
        self._do_graph_fanout("redis", self._redis_graph_fanout)

    def redis_graph_resume(self) -> None:
        assert self._redis_graph_resume is not None
        self._do_graph_resume("redis", self._redis_graph_resume)

    def redis_graph_cyclic(self) -> None:
        assert self._redis_graph_cyclic is not None
        self._do_graph_cyclic("redis", self._redis_graph_cyclic)


if __name__ == "__main__":
    args = _parse_args()

    workload = LangGraphIoWorkload(
        aerospike_connection_string=args.aerospike_uri,
        postgres_connection_string=args.postgres_uri,
        redis_connection_string=args.redis_uri,
        aerospike_max_conns=args.aerospike_max_conns,
        postgres_pool_size=args.postgres_pool_size,
        redis_pool_size=args.redis_pool_size,
    )

    runner = BenchmarkRunner(
        queries_per_second=args.qps,
        scheduler_thread_count=args.scheduler_thread_count,
        worker_thread_count=args.worker_thread_count,
        runtime_per_function=args.runtime_per_function,
        workload=workload,
    )
    runner.run()
    runner.print_metrics()
