# LangGraph IO Benchmark

A runnable example showing how to drive the
[`ai-ecosystem-benchmark`](https://github.com/aerospike/ai-ecosystem-benchmark)
framework against the LangGraph checkpointer IO patterns
(`put` / `put_writes` / `get_tuple` / `list`) for all three supported
backends:

| Backend   | Saver used                                       | Source                              |
| --------- | ------------------------------------------------ | ----------------------------------- |
| Aerospike | `langgraph.checkpoint.aerospike.AerospikeSaver`  | this repo's workspace package       |
| Postgres  | `langgraph.checkpoint.postgres.PostgresSaver`    | `langgraph-checkpoint-postgres`     |
| Redis     | `langgraph.checkpoint.redis.RedisSaver`          | `langgraph-checkpoint-redis`        |

The script lives in this folder and stays decoupled from the rest of the
repo on purpose: its non-Aerospike backend dependencies are **not**
declared in `pyproject.toml`, so cloning and `uv sync`ing this repo
won't pull `psycopg`, `redisvl`, etc. You install those only if/when you
want to run the benchmark.

## Prerequisites

1. A workspace clone of the repo plus the standard project setup:

   ```bash
   git clone <this-repo>
   cd aerospike-langgraph
   uv sync
   ```

2. The benchmark framework itself, installed as an editable workspace
   sibling so changes to its source are picked up without a reinstall:

   ```bash
   git clone https://github.com/aerospike/ai-ecosystem-benchmark.git ../ai-ecosystem-benchmark
   uv add --editable ../ai-ecosystem-benchmark
   ```

   This is the only `pyproject.toml` change the benchmark requires; it
   adds `ai-ecosystem-benchmark` as a runtime dependency so the script
   can `from ai_ecosystem_benchmark import BaseBenchmarkWorkload, BenchmarkRunner`.

3. The optional per-backend LangGraph savers, installed **into the venv
   only** so they don't pollute `pyproject.toml` / `uv.lock`:

   ```bash
   uv pip install langgraph-checkpoint-postgres langgraph-checkpoint-redis
   ```

   `uv pip install` writes to `.venv/` directly without touching the
   project manifest. If you ever `uv sync`, these will get pruned and
   you'll need to reinstall them — that's intentional, it keeps the
   committed dependency surface clean.

   Skip this step entirely if you only plan to benchmark Aerospike (set
   `POSTGRES_URI = None` and `REDIS_URI = None` in the script and the
   imports for the missing backends are never hit — they're guarded
   inside `setup()`).

## Backend setup

The script's `setup()` lifecycle hook calls each saver's own `.setup()`,
so it'll create whatever tables / secondary indexes / RediSearch indexes
each backend needs on first run. You only have to provision the
*server* and a database/namespace.

### Aerospike

- A reachable Aerospike server (Community Edition is fine).
- A namespace configured on that server. The URI's path component
  selects which namespace to use: `aerospike://host:port/<namespace>`.
  If the path is omitted, the script defaults to `test`. Whatever you
  put here must match the `namespace` block in the server's
  `aerospike.conf`.

A quick local sanity check:

```bash
docker run -d --name aerospike -p 3000-3002:3000-3002 \
  container.aerospike.com/aerospike/aerospike-server
```

The bundled image ships with a `test` namespace, so
`aerospike://localhost:3000/test` works out of the box.

### Postgres

- Postgres 12+ reachable at the URI you configure.
- A pre-created database and a role with `CREATE TABLE` privileges on
  it. `PostgresSaver.setup()` will run idempotent migrations to create
  the checkpoint / blob / writes tables and their indexes.
- `psycopg`'s native driver is required (installed as a transitive
  dep of `langgraph-checkpoint-postgres` above).

A quick local sanity check:

```bash
docker run -d --name pg -e POSTGRES_PASSWORD=password -p 5432:5432 postgres:17
```

Then set `POSTGRES_URI = "postgresql://postgres:password@localhost:5432/postgres"`.

### Redis

- **Redis Stack** (or any Redis with both **RediSearch** and
  **RedisJSON** modules loaded). Plain Redis OSS will *not* work:
  `langgraph-checkpoint-redis` stores checkpoints as JSON documents and
  builds RediSearch indexes over them in `setup()`. You'll see
  `unknown command 'FT._LIST'` (RediSearch missing) or
  `Invalid rule type: JSON` (RedisJSON missing) if either module isn't
  present.

A quick local sanity check:

```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server:latest
```

Then set `REDIS_URI = "redis://localhost:6379/0"`.

## Running the benchmark

1. Open `benchmarks/langgraph_checkpoint_workload.py` and edit the three
   connection-string constants near the bottom of the file. Set any of
   them to `None` to disable that backend.

   ```python
   AEROSPIKE_URI: str | None = "aerospike://localhost:3000/test"
   POSTGRES_URI: str | None = "postgresql://postgres:password@localhost:5432/postgres"
   REDIS_URI: str | None = "redis://localhost:6379/0"
   ```

2. (Optional) Tweak the `BenchmarkRunner` parameters in the same block:

   - `queries_per_second` — total load across all scheduler threads.
   - `runtime_per_function` — seconds each `<backend>_<op>` method runs.
   - `worker_thread_count` — concurrency cap for in-flight calls.

3. Run it:

   ```bash
   uv run python benchmarks/langgraph_checkpoint_workload.py
   ```

The output is one block per backend with `p50` / `p90` / `p99` latency
in milliseconds for each of the four checkpointer operations:

```
=== Benchmark Metrics (ms) ===

[aerospike]
  aerospike_checkpoint_get_tuple : calls=200 failures=0 p50=147ms p90=193ms p99=210ms
  aerospike_checkpoint_list      : calls=200 failures=0 p50=254ms p90=262ms p99=306ms
  aerospike_checkpoint_put       : calls=200 failures=0 p50=149ms p90=155ms p99=164ms
  aerospike_checkpoint_put_writes: calls=200 failures=0 p50= 48ms p90= 50ms p99= 55ms
...
```

## Notes

- The framework discovers benchmark methods by **prefix**: any method
  on the workload whose name starts with `aerospike_`, `postgres_`, or
  `redis_` is treated as a benchmark for that backend, and is skipped
  if the corresponding URI is `None`. To add a new operation, just
  define `<backend>_<op>` for each backend you care about.

- `setup()` pre-seeds 16 checkpoints per enabled backend so that
  `get_tuple` and `list` always have data to read. Read-side methods
  round-robin over those 16 thread IDs to avoid artificial single-row
  contention; `put` methods cycle through them too so they don't all
  hammer the same row.

- Postgres uses a real `psycopg_pool.ConnectionPool` instead of
  `PostgresSaver.from_conn_string(...)` so concurrent worker threads
  aren't serialised through one connection. The pool is pre-warmed
  with `min_size == max_size` so the connection-establishment cost
  doesn't leak into the measured latency.

- Aerospike's client is constructed with `max_conns_per_node = 4096`;
  the default of 100 is way too small for high `worker_thread_count` +
  the chatty `list` op (which issues 20+ round-trips per call).
