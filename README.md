# Aerospike LangGraph

Aerospike-backed persistence for [LangGraph](https://github.com/langchain-ai/langgraph). This monorepo provides drop-in checkpoint and store implementations so your LangGraph agents can durably save state to an [Aerospike](https://aerospike.com/) cluster.

![Aerospike + LangGraph flow](./assets/Langgraph-Aerospike-Flow.png)

## Why this exists

[LangGraph](https://github.com/langchain-ai/langgraph) is an orchestration framework for stateful, multi-step AI agents. Every super-step produces a _checkpoint_, a snapshot of graph state. [Aerospike](https://aerospike.com/) is a real-time NoSQL database with sub-millisecond latency, native time to live (TTL) per record, and atomic map operations. Together they give LangGraph agents durable, expiry-aware, forkable state without a relational database.

| Agent requirement | Aerospike primitive |
|---|---|
| Checkpoint durability | Record write with `COMMIT_LEVEL_ALL` |
| Session expiry without a sweeper | Per-record TTL set at write time |
| Resume from any past step | Checkpoint records keyed by `thread_id` + `checkpoint_id` |
| Fork from a saved state | Direct key-value read, no replay or scan |
| Long-term user memory across sessions | `AerospikeStore` with batch get/put |

## Cookbooks

Two production-shaped examples in [`cookbooks/`](./cookbooks):

- [Expiring chat sessions](./cookbooks/expiring-chat-sessions): configure a TTL on the checkpointer so Aerospike reclaims abandoned sessions automatically. No cron job, no DELETE query. Demonstrates the full lifecycle: create, resume, expire, verify.
- [Agent path correction](./cookbooks/agent-path-correction): rewind a thread to a past checkpoint and resume down a different path, reusing context the agent already derived. Demonstrates listing history, rehydrating state, and forking as direct low-latency key reads.

Both demos use deterministic stubs instead of a live LLM. No API key is required. Swap in a real model by replacing the stub in `agent.py`. The graph shape and Aerospike behavior are unchanged.

To run a working demo in under five minutes:

```bash
docker run -d --name aerospike -p 3000-3002:3000-3002 container.aerospike.com/aerospike/aerospike-server
uv sync
uv run python cookbooks/expiring-chat-sessions/demo.py --skip-wait
```

Aerospike Community Edition (the default Docker image above) is sufficient for all cookbooks.

## Packages

| Package                                                                     | Description                                          | Install                                         |
| --------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------- |
| [langgraph-checkpoint-aerospike](./packages/langgraph-checkpoint-aerospike) | Checkpoint saver for LangGraph graph execution state | `pip install -U langgraph-checkpoint-aerospike` |
| [langgraph-store-aerospike](./packages/langgraph-store-aerospike)           | key-value store with batch ops, search, and TTL      | `pip install -U langgraph-store-aerospike`      |

## Requirements

- Python >= 3.10
- Aerospike Server (or the [Docker image](https://hub.docker.com/_/aerospike))
- `aerospike` Python client >= 15
- `langgraph` >= 1.0

## Quickstart

### 1. Start Aerospike

```bash
docker run -d --name aerospike -p 3000-3002:3000-3002 container.aerospike.com/aerospike/aerospike-server
```

### 2. Install

```bash
pip install -U langgraph-checkpoint-aerospike langgraph-store-aerospike
```

### 3. Use the Checkpoint Saver

```python
import aerospike
from langgraph.checkpoint.aerospike import AerospikeSaver

client = aerospike.client({"hosts": [("127.0.0.1", 3000)]}).connect()
saver = AerospikeSaver(client=client, namespace="test")

compiled = graph.compile(checkpointer=saver)
compiled.invoke({"input": "hello"}, config={"configurable": {"thread_id": "demo"}})
```

### 4. Use the Store

```python
import aerospike
from langgraph.store.aerospike import AerospikeStore

client = aerospike.client({"hosts": [("127.0.0.1", 3000)]}).connect()
store = AerospikeStore(client=client, namespace="test", set="langgraph_store")

store.put(namespace=("users", "profiles"), key="user_123", value={"name": "Alice", "age": 30})
item = store.get(namespace=("users", "profiles"), key="user_123")
```

For complete, runnable examples that demonstrate these APIs end to end, see the [cookbooks](#cookbooks).

## Configuration

Both packages read connection details from environment variables by default:

| Variable              | Default              | Description                 |
| --------------------- | -------------------- | --------------------------- |
| `AEROSPIKE_HOST`      | `127.0.0.1`          | Aerospike cluster seed host |
| `AEROSPIKE_PORT`      | `3000`               | Aerospike cluster seed port |
| `AEROSPIKE_NAMESPACE` | `test`               | Aerospike namespace to use  |
| `AEROSPIKE_SET`       | _(package-specific)_ | Aerospike set name          |

## Development

Each package is independently installable and testable. Install whichever you're modifying:

```bash
pip install -e "packages/langgraph-checkpoint-aerospike[dev]"
pip install -e "packages/langgraph-store-aerospike[dev]"
```

Run the tests:

```bash
pytest packages/langgraph-checkpoint-aerospike/tests
pytest packages/langgraph-store-aerospike/tests
```

Tests connect to a real Aerospike cluster. Defaults are `127.0.0.1:3000`, namespace `test`. Override with environment variables if needed:

| Variable              | Default     |
| --------------------- | ----------- |
| `AEROSPIKE_HOST`      | `127.0.0.1` |
| `AEROSPIKE_PORT`      | `3000`      |
| `AEROSPIKE_NAMESPACE` | `test`      |

If you have [uv](https://docs.astral.sh/uv/) installed, `uv sync` from the repo root replaces all of the above. The repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), so it installs both packages in editable mode and pulls in dev dependencies.

```bash
uv sync
uv run pytest packages/langgraph-checkpoint-aerospike/tests
uv run pytest packages/langgraph-store-aerospike/tests
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for linting, commit conventions, and CI details.

## License

[Apache 2.0](./LICENSE)
