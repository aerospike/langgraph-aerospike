# LangGraph Checkpoint Aerospike

Store LangGraph checkpoints in Aerospike using the provided `AerospikeSaver`.

## Installation

```bash
pip install -U langgraph-checkpoint-aerospike
```

## Usage

1. Bring up Aerospike locally using prebuilt [Aerospike Docker Image](https://hub.docker.com/_/aerospike):

```bash
docker run -d --name aerospike -p 3000-3002:3000-3002 container.aerospike.com/aerospike/aerospike-server
```

2. Point the saver at your cluster (Default):
   - `AEROSPIKE_HOST=127.0.0.1`
   - `AEROSPIKE_PORT=3000`
   - `AEROSPIKE_NAMESPACE=test`

3. Build a LangGraph graph and compile it with the saver as the checkpointer.
   Passing the same `thread_id` on a later run resumes from the persisted state:

   ```python
   from typing import Annotated, TypedDict

   import aerospike
   from langgraph.graph import START, END, StateGraph
   from langgraph.graph.message import add_messages
   from langchain_core.messages import HumanMessage

   from langgraph.checkpoint.aerospike import AerospikeSaver

   # 1. Connect to Aerospike and build the checkpointer.
   client = aerospike.client({"hosts": [("127.0.0.1", 3000)]}).connect()
   checkpointer = AerospikeSaver(client=client, namespace="test")

   # 2. Define a minimal LangGraph graph.
   class State(TypedDict):
       messages: Annotated[list, add_messages]

   def chatbot(state: State) -> State:
       last = state["messages"][-1].content
       return {"messages": [("assistant", f"You said: {last}")]}

   builder = StateGraph(State)
   builder.add_node("chatbot", chatbot)
   builder.add_edge(START, "chatbot")
   builder.add_edge("chatbot", END)

   # 3. Compile with the Aerospike checkpointer.
   graph = builder.compile(checkpointer=checkpointer)

   # 4. Each thread_id is a separate, persisted conversation.
   config = {"configurable": {"thread_id": "demo"}}
   graph.invoke({"messages": [HumanMessage("hello")]}, config)

   # 5. A later call with the same thread_id resumes from the saved state,
   #    even in a brand-new process — the history lives in Aerospike.
   result = graph.invoke({"messages": [HumanMessage("are you still there?")]}, config)
   for message in result["messages"]:
       message.pretty_print()
   ```

You can also inspect persisted checkpoints directly through the saver, e.g.
`checkpointer.get_tuple(config)` for the latest checkpoint or
`checkpointer.list(config)` to walk a thread's history.
