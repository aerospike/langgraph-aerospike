"""Benchmark the LangGraph checkpointer IO patterns against every backend.

Defines a :class:`LangGraphIoWorkload` that issues the four calls a real
LangGraph application makes against a checkpointer (``put``, ``put_writes``,
``get_tuple``, ``list``) using the production saver implementation for each
supported backend:

- ``aerospike`` -> this repo's :class:`AerospikeSaver`
- ``postgres``  -> ``langgraph.checkpoint.postgres.PostgresSaver``
- ``redis``     -> ``langgraph.checkpoint.redis.RedisSaver``

Run with the default backend URIs and benchmark settings::

    uv run python benchmarks/langgraph_workload.py

Override connection URIs, QPS, or worker count with CLI flags. Pass ``none``
for any URI to disable that backend; the framework will simply skip it.
"""

from __future__ import annotations

import argparse
import urllib.parse
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from itertools import count
from typing import Any

from ai_ecosystem_benchmark import BaseBenchmarkWorkload, BenchmarkRunner

# Number of distinct ``thread_id`` values the workload pre-populates per
# backend. Read/list calls round-robin over this pool so concurrent workers
# touch different rows/keys (avoids artificial single-row contention) while
# still hitting data that was seeded in ``setup``.
_THREAD_POOL_SIZE = 1024
_CHECKPOINT_NS = "bench"
_DEFAULT_AEROSPIKE_URI = "aerospike://10.100.0.4:3000/test"
_DEFAULT_POSTGRES_URI = "postgresql://bench:benchpassword@10.100.0.2:5432/bench"
_DEFAULT_REDIS_URI = "redis://10.100.0.5:6379/0"
_DEFAULT_QPS = 1000
_DEFAULT_WORKER_THREAD_COUNT = 8192
_DEFAULT_SCHEDULER_THREAD_COUNT = 8
_DEFAULT_RUNTIME_PER_FUNCTION = 30
_DEFAULT_AEROSPIKE_MAX_CONNS = 1024
_DEFAULT_POSTGRES_POOL_SIZE = 64
_DEFAULT_REDIS_POOL_SIZE = 256


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
        description="Benchmark LangGraph checkpointer IO patterns against configured backends."
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


def _make_checkpoint() -> dict[str, Any]:
    """Build a minimal but valid LangGraph ``Checkpoint`` payload.

    Uses ``v=4`` (current format) and a fresh UUID per call so each ``put``
    creates a new row rather than overwriting in place; that's the
    interesting load to benchmark.
    """
    return {
        "v": 4,
        "id": str(uuid.uuid4()),
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "channel_values": {"messages": [{"role": "user", "content": "hello"}]},
        "channel_versions": {"__start__": 1, "messages": 2},
        "versions_seen": {"__start__": {"__start__": 1}},
        "pending_sends": [],
        "updated_channels": None,
    }


def _cfg(thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
    configurable: dict[str, Any] = {
        "thread_id": thread_id,
        "checkpoint_ns": _CHECKPOINT_NS,
    }
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


class LangGraphIoWorkload(BaseBenchmarkWorkload):
    """Benchmark workload exercising the IO LangGraph itself issues.

    The methods are the four ``BaseCheckpointSaver`` operations a graph
    invocation triggers per node: ``put`` (write the new checkpoint),
    ``put_writes`` (record pending channel writes), ``get_tuple`` (load
    the latest state on resume), and ``list`` (timeline / debug UIs).

    The framework discovers methods by prefix, so each backend gets its
    own four ``<backend>_checkpoint_*`` methods that delegate to a shared
    implementation parameterised on the saver and backend tag.
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

        # Postgres uses a real connection pool so concurrent worker threads
        # don't serialise through a single connection. The pool itself is
        # the resource we need to close in teardown.
        self._postgres_pool: Any | None = None
        self._postgres_saver: Any | None = None

        # RedisSaver can build its own unbounded client, but benchmark runs
        # need an explicit pool so the client cannot exhaust file handles.
        self._redis_client: Any | None = None
        self._redis_saver: Any | None = None

        # ``(backend, thread_idx) -> latest checkpoint_id``. Refreshed on
        # every successful ``put`` so ``put_writes`` always targets a
        # checkpoint that actually exists.
        self._latest_checkpoint_id: dict[tuple[str, int], str] = {}

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
        self._redis_saver = RedisSaver(redis_client=self._redis_client)
        # Creates RediSearch indices used for ``list`` / filtered queries.
        self._redis_saver.setup()
        self._seed("redis", self._redis_saver)

    def _seed(self, backend: str, saver: Any) -> None:
        """Reset prior-run data, then pre-populate one checkpoint per thread.

        The benchmark only ever touches the fixed ``bench-*`` thread pool,
        so deleting those threads up front gives every run an identical
        clean slate. Without this, re-runs accumulate checkpoints/writes
        (and grow Aerospike's per-thread timeline), inflating ``list``
        latency and making results incomparable across runs.
        """
        for idx in range(_THREAD_POOL_SIZE):
            thread_id = f"bench-{idx}"
            saver.delete_thread(thread_id)
            checkpoint = _make_checkpoint()
            saver.put(_cfg(thread_id), checkpoint, {}, {})
            self._latest_checkpoint_id[(backend, idx)] = checkpoint["id"]

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
        saver.put_writes(cfg, [("messages", "ack")], task_id)

    def _do_checkpoint_get_tuple(self, saver: Any) -> None:
        thread_id = f"bench-{self._next_thread_index()}"
        saver.get_tuple(_cfg(thread_id))

    def _do_checkpoint_list(self, saver: Any) -> None:
        thread_id = f"bench-{self._next_thread_index()}"
        # Materialise the iterator inside the timed call so the IO is
        # actually performed, not just deferred.
        list(saver.list(_cfg(thread_id), limit=10))

    # ---------- aerospike methods ----------

    def aerospike_checkpoint_put(self) -> None:
        self._do_checkpoint_put("aerospike", self._aerospike_saver)

    def aerospike_checkpoint_put_writes(self) -> None:
        self._do_checkpoint_put_writes("aerospike", self._aerospike_saver)

    def aerospike_checkpoint_get_tuple(self) -> None:
        self._do_checkpoint_get_tuple(self._aerospike_saver)

    def aerospike_checkpoint_list(self) -> None:
        self._do_checkpoint_list(self._aerospike_saver)

    # ---------- postgres methods ----------

    def postgres_checkpoint_put(self) -> None:
        self._do_checkpoint_put("postgres", self._postgres_saver)

    def postgres_checkpoint_put_writes(self) -> None:
        self._do_checkpoint_put_writes("postgres", self._postgres_saver)

    def postgres_checkpoint_get_tuple(self) -> None:
        self._do_checkpoint_get_tuple(self._postgres_saver)

    def postgres_checkpoint_list(self) -> None:
        self._do_checkpoint_list(self._postgres_saver)

    # ---------- redis methods ----------

    def redis_checkpoint_put(self) -> None:
        self._do_checkpoint_put("redis", self._redis_saver)

    def redis_checkpoint_put_writes(self) -> None:
        self._do_checkpoint_put_writes("redis", self._redis_saver)

    def redis_checkpoint_get_tuple(self) -> None:
        self._do_checkpoint_get_tuple(self._redis_saver)

    def redis_checkpoint_list(self) -> None:
        self._do_checkpoint_list(self._redis_saver)


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
