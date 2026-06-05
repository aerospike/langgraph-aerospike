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

1. Run it with the default backend URIs:

   ```bash
   uv run python benchmarks/langgraph_workload.py
   ```

2. Override backend URIs, load, duration, or connection pool sizes from the CLI:

   ```bash
   uv run python benchmarks/langgraph_workload.py \
     --aerospike-uri aerospike://localhost:3000/test \
     --postgres-uri postgresql://postgres:password@localhost:5432/postgres \
     --redis-uri redis://localhost:6379/0 \
     --qps 500 \
     --worker-thread-count 512 \
     --postgres-pool-size 64 \
     --redis-pool-size 256
   ```

   Pass `none` for a URI to disable that backend.

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

- `setup()` pre-seeds 1024 checkpoints per enabled backend so that
  `get_tuple` and `list` always have data to read. Read-side methods
  round-robin over those thread IDs to avoid artificial single-row
  contention; `put` methods cycle through them too so they don't all
  hammer the same row.

- Postgres uses a real `psycopg_pool.ConnectionPool` instead of
  `PostgresSaver.from_conn_string(...)` so concurrent worker threads
  aren't serialised through one connection. The pool is pre-warmed
  with `min_size == max_size` so the connection-establishment cost
  doesn't leak into the measured latency. Keep `--postgres-pool-size`
  below the server's `max_connections` with room for admin sessions.

- Redis uses an explicit `BlockingConnectionPool`. Without a bounded
  pool, very large worker counts can create enough sockets to hit the
  process `ulimit -n`, producing `Too many open files` instead of useful
  benchmark data.

- Treat `worker_thread_count` as the maximum in-flight operation count,
  not as "more is better". A value like 30000 can overwhelm the Python
  scheduler, Redis file descriptors, and Postgres connection limits. For
  a fair comparison, pick a worker count just above
  `qps * expected_p99_latency_seconds`, then sweep `--qps` upward until
  a backend saturates. Saturation is the comparison point; client-side
  warnings or failures are not.

- The built-in pool defaults are conservative starting points, not
  universal tuning advice. For serious comparison runs, set them
  explicitly: keep Postgres below `max_connections`, Redis below the
  client and server file-descriptor budget, and Aerospike high enough to
  cover the desired in-flight operations.

- Aerospike's client is constructed with a configurable
  `max_conns_per_node`; the driver default of 100 is often too small for
  high `worker_thread_count` + the chatty `list` op.
