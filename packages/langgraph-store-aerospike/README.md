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

import aerospike
from langgraph.config import get_store
from langgraph.graph import START, END, StateGraph

from langgraph.store.aerospike import AerospikeStore

# 1. Connect to Aerospike and build the store.
client = aerospike.client({"hosts": [("127.0.0.1", 3000)]}).connect()
store = AerospikeStore(client=client, namespace="test", set="langgraph_store")

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
