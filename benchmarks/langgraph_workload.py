"""Benchmark the LangGraph checkpointer IO patterns against every backend.

Defines a :class:`LangGraphIoWorkload` that issues the four calls a real
LangGraph application makes against a checkpointer (``put``, ``put_writes``,
``get_tuple``, ``list``) using the production saver implementation for each
supported backend:

- ``aerospike`` -> this repo's :class:`AerospikeSaver`
- ``postgres``  -> ``langgraph.checkpoint.postgres.PostgresSaver``
- ``redis``     -> ``langgraph.checkpoint.redis.RedisSaver``

Edit the ``AEROSPIKE_URI`` / ``POSTGRES_URI`` / ``REDIS_URI`` constants at
the bottom of the file and run::

    uv run python benchmarks/langgraph_workload.py

Set any URI to ``None`` to disable that backend; the framework will simply
skip it.
"""

from __future__ import annotations

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
_THREAD_POOL_SIZE = 16
_CHECKPOINT_NS = "bench"


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
    ) -> None:
        super().__init__(
            aerospike_connection_string=aerospike_connection_string,
            postgres_connection_string=postgres_connection_string,
            redis_connection_string=redis_connection_string,
        )
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

        # ``from_conn_string`` is a context manager; we keep the live
        # context around so we can ``__exit__`` it in ``teardown``.
        self._redis_ctx: Any | None = None
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
        if self._redis_ctx is not None:
            self._redis_ctx.__exit__(None, None, None)

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
                "max_conns_per_node": 4096,
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
        pool_size = 64
        self._postgres_pool = ConnectionPool(
            self.postgres_connection_string,
            min_size=pool_size,
            max_size=pool_size,
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

        assert self.redis_connection_string is not None
        self._redis_ctx = RedisSaver.from_conn_string(self.redis_connection_string)
        self._redis_saver = self._redis_ctx.__enter__()
        # Creates RediSearch indices used for ``list`` / filtered queries.
        self._redis_saver.setup()
        self._seed("redis", self._redis_saver)

    def _seed(self, backend: str, saver: Any) -> None:
        """Pre-populate one checkpoint per thread so reads have data."""
        for idx in range(_THREAD_POOL_SIZE):
            thread_id = f"bench-{idx}"
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
    # Edit these to point at the backends you want to benchmark. Set any
    # of them to ``None`` to disable that backend (the framework simply
    # skips backends with no connection string).
    #
    # Aerospike: ``host:port[/<namespace>]`` or ``aerospike://host:port[/<namespace>]``.
    #            ``<namespace>`` defaults to ``test`` if omitted.
    # Postgres : standard libpq URI.
    # Redis    : standard ``redis://`` URI. Server must have RediSearch +
    #            RedisJSON loaded (e.g. Redis Stack); plain Redis OSS
    #            won't work because ``langgraph-checkpoint-redis`` builds
    #            JSON-backed search indices in ``setup()``.
    AEROSPIKE_URI: str | None = "aerospike://10.100.0.2:3000/test"
    POSTGRES_URI: str | None = "postgresql://bench:benchpassword@10.100.0.4:5432/bench"
    REDIS_URI: str | None = "redis://10.100.0.3:6379/0"

    workload = LangGraphIoWorkload(
        aerospike_connection_string=AEROSPIKE_URI,
        postgres_connection_string=POSTGRES_URI,
        redis_connection_string=REDIS_URI,
    )

    runner = BenchmarkRunner(
        queries_per_second=2000,
        scheduler_thread_count=8,
        worker_thread_count=10000,
        runtime_per_function=30,
        workload=workload,
    )
    runner.run()
    runner.print_metrics()
