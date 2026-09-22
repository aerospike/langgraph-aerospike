# LangGraph Store Aerospike

Store LangGraph state and data in Aerospike using the provided `AerospikeStore`.

## Installation

```bash
pip install -U langgraph-store-aerospike
```

## Usage

1. Bring up Aerospike locally using prebuilt [Aerospike Docker Image](https://hub.docker.com/_/aerospike):

```bash
docker run -d --name aerospike -p 3000-3002:3000-3002 container.aerospike.com/aerospike/aerospike-server
```

2. Point the store at your cluster (Default):
   - `AEROSPIKE_HOST=127.0.0.1`
   - `AEROSPIKE_PORT=3000`
   - `AEROSPIKE_NAMESPACE=langgraph` (default namespace for the store)
   - `AEROSPIKE_SET=store` (default set name)

3. Compile a LangGraph graph with the store. Nodes reach it through
   `get_store()`, so the same long-term memory is shared across every run and
   every thread (unlike a checkpointer, which is scoped to a single thread):

```python
from typing import TypedDict

from aerospike_sdk import Behavior
from aerospike_sdk.sync import ClusterDefinition
from langgraph.config import get_store
from langgraph.graph import START, END, StateGraph

from langgraph.store.aerospike import AerospikeStore

# 1. Connect to Aerospike and build the store.
cluster = ClusterDefinition("127.0.0.1", 3000).connect()
session = cluster.create_session(Behavior.DEFAULT)
store = AerospikeStore(session=session, namespace="test", set="langgraph_store")


# 2. Define a graph whose node reads and writes long-term memory.
class State(TypedDict):
    user_id: str
    food: str


def remember_preference(state: State) -> State:
    store = get_store()
    namespace = ("users", state["user_id"])

    # Persist something we learned about this user.
    store.put(namespace, key="profile", value={"favorite_food": state["food"]})

    # Read it back (would also be visible in any future run / thread).
    profile = store.get(namespace, key="profile")
    print(profile.value)  # {"favorite_food": "pizza"}
    return state


builder = StateGraph(State)
builder.add_node("remember_preference", remember_preference)
builder.add_edge(START, "remember_preference")
builder.add_edge("remember_preference", END)

# 3. Compile with the Aerospike store and run.
graph = builder.compile(store=store)
graph.invoke({"user_id": "user_123", "food": "pizza"})
```

The store is also a standalone `BaseStore`, so you can use it directly outside a
graph for the same cross-thread memory:

```python
# Search within a namespace prefix, filtering on stored fields.
results = store.search(("users",), filter={"favorite_food": "pizza"}, limit=10)

# Delete an item.
store.delete(("users", "user_123"), key="profile")
```

### Vector Similarity Search (Pre-Release)

Store item embeddings and search by semantic similarity:

```python
from aerospike_sdk import Behavior
from aerospike_sdk.sync import ClusterDefinition
from langgraph.store.aerospike import AerospikeStore

cluster = ClusterDefinition("127.0.0.1", 3000).connect()
session = cluster.create_session(Behavior.DEFAULT)
store = AerospikeStore(session=session, namespace="test", set="langgraph_store")

# Put an item with an embedding
embedding = [0.1, 0.2, 0.3, 0.4]  # 4-dimensional embedding
store.put(
    namespace=("documents",),
    key="doc_123",
    value={
        "title": "Aerospike Vector Search",
        "text": "Learn to search by similarity...",
        "__embedding__": embedding,  # Special key for embeddings
    }
)

# Search by semantic similarity
query_embedding = [0.15, 0.25, 0.35, 0.45]  # Query embedding
results = store.search(
    namespace_prefix=("documents",),
    query=query_embedding,  # Pass embedding as query
    limit=5,  # Top-5 results
)

# Results are ranked by distance (highest score = most similar)
for item in results:
    print(f"Key: {item.key}, Similarity: {item.score:.4f}")
```

**Distance Metrics:**

By default, embeddings are ranked by cosine similarity. You can also use:

```python
# Euclidean distance (lower is more similar)
results = store.search(
    namespace_prefix=("documents",),
    query={"embedding": query_embedding, "metric": "euclidean"},
    limit=5,
)

# Dot product (for pre-normalized vectors)
results = store.search(
    namespace_prefix=("documents",),
    query={"embedding": query_embedding, "metric": "dotproduct"},
    limit=5,
)
```

**Requirements:**

- Aerospike Server 8.1.3+ with vector support — a dev-only build (e.g. `aerospike-server:latest-dev-vector`); not yet a GA image
- The **`aie/vector` branch** of the Aerospike Python SDK — vector support (`Vector`, `Exp.vector_bin`, Top-K) is not in the published `aerospike-sdk` PyPI package yet; see [VECTOR_SEARCH.md](VECTOR_SEARCH.md#prerequisites) for install steps

**Limitations (Phase 1):**

- Brute-force Top-K ranking: every query still scans and ranks the queried set — there is no ANN/HNSW index — but ranking runs server-side via `order_by(...).top_k(...)` pushdown rather than being computed after fetching every candidate record (acceptable for < 1M items)
- No ANN/HNSW indexes yet
- FLOAT32 embeddings only
- Single embedding per record (stored in `embedding` bin)

**When Phase 2 Arrives:**

HNSW indexes may be added for larger datasets (> 1M items). Existing code will continue to work without changes.
