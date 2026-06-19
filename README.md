# Aerospike LangGraph

Aerospike-backed persistence for [LangGraph](https://github.com/langchain-ai/langgraph). This monorepo provides drop-in checkpoint and store implementations so your LangGraph agents can durably save state to an [Aerospike](https://aerospike.com/) cluster.

![Aerospike + LangGraph flow](./assets/Langgraph-Aerospike-Flow.png)

## Packages

| Package                                                                     | Description                                          | Install                                         |
| --------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------- |
| [langgraph-checkpoint-aerospike](./packages/langgraph-checkpoint-aerospike) | Checkpoint saver for LangGraph graph execution state | `pip install -U langgraph-checkpoint-aerospike` |
| [langgraph-store-aerospike](./packages/langgraph-store-aerospike)           | Key/value store with batch ops, search, and TTL      | `pip install -U langgraph-store-aerospike`      |

## Cookbooks

Runnable, production-shaped recipes live in [`cookbooks/`](./cookbooks):

| Cookbook                                                              | What it shows                                                                       |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [auto-expiring-chat](./cookbooks/auto-expiring-chat)                  | Expire LangGraph chat checkpoints automatically with native Aerospike TTL           |
| [fork-from-checkpoint](./cookbooks/fork-from-checkpoint)                | Fork from a past checkpoint and resume down a new path, reusing context the agent already derived |

## Requirements

- Python >= 3.10
- Aerospike Server (or the [Docker image](https://hub.docker.com/_/aerospike))
- `aerospike` Python client >= 15
- `langgraph` >= 1.0

## Quick Start

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

## Configuration

Both packages read connection details from environment variables by default:

| Variable              | Default              | Description                 |
| --------------------- | -------------------- | --------------------------- |
| `AEROSPIKE_HOST`      | `127.0.0.1`          | Aerospike cluster seed host |
| `AEROSPIKE_PORT`      | `3000`               | Aerospike cluster seed port |
| `AEROSPIKE_NAMESPACE` | `test`               | Aerospike namespace to use  |
| `AEROSPIKE_SET`       | _(package-specific)_ | Aerospike set name          |

## Development

Each package in this monorepo is independently installable and testable. The workflow below is per-package — install both if you want to develop them together.

### Prerequisites

- Python >= 3.10
- A running Aerospike server (see [Quick Start](#1-start-aerospike))

### 1. Clone and create a virtualenv

```bash
git clone https://github.com/aerospike/aerospike-langgraph.git
cd aerospike-langgraph

python -m venv venv
source venv/bin/activate           # Linux / macOS
# .\venv\Scripts\Activate.ps1      # Windows PowerShell

python -m pip install --upgrade pip
```

### 2. Install the package(s) you want to work on

Each package declares a `[dev]` extra that pulls in everything needed to run its tests. Install whichever you're modifying:

```bash
# Checkpoint saver
pip install -e "packages/langgraph-checkpoint-aerospike[dev]"

# Store
pip install -e "packages/langgraph-store-aerospike[dev]"
```

The `-e` (editable) flag means your source changes are picked up immediately without reinstalling. You can install both at once if you want:

```bash
pip install -e "packages/langgraph-checkpoint-aerospike[dev]" \
            -e "packages/langgraph-store-aerospike[dev]"
```

### 3. Run the tests

Tests are integration tests that connect to a real Aerospike cluster. Defaults are `127.0.0.1:3000`, namespace `test` — override via environment variables if needed:

| Variable              | Default     |
| --------------------- | ----------- |
| `AEROSPIKE_HOST`      | `127.0.0.1` |
| `AEROSPIKE_PORT`      | `3000`      |
| `AEROSPIKE_NAMESPACE` | `test`      |

```bash
pytest packages/langgraph-checkpoint-aerospike/tests
pytest packages/langgraph-store-aerospike/tests
```

### Alternative: using `uv`

If you have [uv](https://docs.astral.sh/uv/) installed, the whole setup collapses to a single command. The repo is configured as a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), so `uv sync` creates a `.venv`, installs both packages in editable mode, and pulls in all dev dependencies.

```bash
# One-time: install uv (see https://docs.astral.sh/uv/getting-started/installation/)

# Install everything
uv sync

# Run tests
uv run pytest packages/langgraph-checkpoint-aerospike/tests
uv run pytest packages/langgraph-store-aerospike/tests
```

uv is optional — the pip-based workflow above remains fully supported. Adopt whichever you prefer.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for linting, commit conventions, and CI details.

## License

[Apache 2.0](./LICENSE)
