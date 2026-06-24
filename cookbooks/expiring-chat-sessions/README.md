# Auto-Expiring Chat Sessions Using Aerospike TTL

LangGraph writes a checkpoint after every agent step. Without expiration, those checkpoints accumulate indefinitely. Relational databases require a scheduled job to clean them up:

```sql
DELETE FROM checkpoints WHERE updated_at < NOW() - INTERVAL '7 days';
```

That job needs to run reliably, not delete active sessions, and scale with your session volume. Aerospike solves this at the storage layer. Each checkpoint record carries a time to live (TTL), and Aerospike reclaims it automatically. No scheduled job, no cleanup query.

## When to use this pattern

- Users leave sessions open and never return. After the TTL elapses, Aerospike removes the records. Abandoned conversations do not accumulate.
- Your product has a data retention requirement (compliance, internal policy) that mandates session data expires after a fixed period.
- Active sessions should stay alive as long as the user is engaged. The `refresh_on_read` option extends a session's TTL each time a checkpoint is read, so active users keep their context while idle sessions expire.

## Prerequisites

Before you start, confirm you have:

| Component | Requirement |
|-----------|-------------|
| Python | 3.10 or newer |
| [uv](https://docs.astral.sh/uv/) | Installed (repo uses `uv run` for cookbook scripts) |
| Aerospike Database | Running and reachable on port 3000 (local or remote) |
| This repo | Cloned; dependencies installed with `uv sync` from the repo root |

No LLM API keys are required. The agent uses LangChain's `FakeListChatModel` for
deterministic output.

## How to use this cookbook

Each step explains the code, shows it, and points to the matching location in the finished files. Open `agent.py` and `demo.py` side by side to follow along.

Files:

- `agent.py`: the LangGraph chat agent (Steps 1–3)
- `demo.py`: connecting, configuring TTL, and proving expiry (Steps 4–9)

File map for `agent.py` (helps when following Steps 2–3):

```text
agent.py
├── Step 1:  ChatState                          (module level)
├── Step 2:  _make_llm() -> BaseChatModel       (module level; returns FakeListChatModel)
└── build_chat_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph
    ├── Step 2:  chatbot(state: ChatState) -> ChatState   (inner function — the node)
    └── Step 3:  StateGraph wiring + compile              (tail of same function)
```

File map for `demo.py` (helps when following Steps 4–9):

```text
demo.py
├── Constants:  AEROSPIKE_HOST, CHAT_TTL_MINUTES, THREAD_ID, ...
├── Step 4:  _connect() -> Iterator[aerospike.Client]
├── Step 5:  _build_checkpointer(client) -> AerospikeSaver
├── Step 7:  _checkpoint_ttl_seconds(saver, client, config) -> int | None
├── Step 6:  _say(graph, config, text) -> int
└── main() -> int
    ├── Step 6:  wire graph + checkpointer, first _say() call
    ├── Step 8:  second _say() call (resume)
    └── Step 9:  wait, verify expiry, third _say() call (fresh start)
```

---

## Step 1 — Define the conversation state

**What this step does:** Every LangGraph graph operates on a shared state
object. For a chat agent, that state is the running list of messages. Use the
`add_messages` reducer so each turn *appends* to the history instead of
replacing it — this is what lets a checkpoint accumulate a real conversation.

```python
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

**In the code:** `agent.py`, marked `# === Step 1 ===` immediately before `class ChatState`.
The `Annotated[list[BaseMessage], add_messages]` annotation is the entire reason
resuming a thread grows the message count instead of overwriting it.

---

## Step 2 — Create the model and the chat node

**What this step does:** A LangGraph node is a function that takes the current
state and returns an update. The `chatbot` node sends the conversation to a model
and appends the reply. Split this into two pieces in `agent.py`:

1. **`_make_llm()`** — creates the model (the demo uses `FakeListChatModel` with
   scripted replies so you need no API key).
2. **`chatbot`** — the node function that calls the model. It lives *inside*
   `build_chat_graph()` because it closes over `llm`.

**Part A — create the model** (`agent.py`, `# === Step 2 ===`):

```python
def _make_llm() -> BaseChatModel:        # generic interface; FakeListChatModel is one impl
    return FakeListChatModel(
        responses=[
            "Hi! I'm your support assistant. How can I help?",
            "Got it -- I've noted that down for this session.",
            "Hello again! This looks like a brand-new session to me.",
        ]
    )
```

**Part B — define the node** (`agent.py`, inside `build_chat_graph()`, look for
`# Step 2 (cont.)`):

```python
def build_chat_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    llm = _make_llm()   # <-- Part A is called here

    def chatbot(state: ChatState) -> ChatState:
        system = SystemMessage(content="You are a concise, helpful support assistant.")
        response: AIMessage = llm.invoke([system, *state["messages"]])
        return {"messages": [response]}
    # ... Step 3 continues in the next section
```

**In the code:** open `agent.py` and search for `Step 2`. You will find
`_make_llm() -> BaseChatModel` at module level (implemented with `FakeListChatModel`),
then `chatbot` nested inside
`build_chat_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph`. The
node returns only the *new* message; `add_messages` from Step 1 merges it into the
history. Returning the generic `BaseChatModel` is what makes the model a clean swap
point — only the body of `_make_llm()` changes when you switch providers.

> **Tip:** To use a real model, replace the body of `_make_llm()` with, for
> example, `ChatOpenAI(model="gpt-4o-mini")`. The graph shape and the Aerospike
> behavior stay the same.

---

## Step 3 — Build and compile the graph

**What this step does:** Still inside `build_chat_graph()`, wire the `chatbot`
node from Step 2 into a graph (`START → chatbot → END`) and compile it. The key
argument is `checkpointer=`: this is where persistence plugs in. The graph does
not know or care that the checkpointer is Aerospike-backed.

```python
    # still inside build_chat_graph(), after the chatbot function from Step 2:

    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    return builder.compile(checkpointer=checkpointer)
```

**In the code:** `agent.py`, marked `# === Step 3 ===`. This is the tail of
`build_chat_graph()` — the same function where `chatbot` was defined in Step 2.
`builder.compile(checkpointer=checkpointer)` returns a `CompiledStateGraph`, which
is the runnable object you call `.invoke()` on. The checkpointer is passed in as
a parameter so `demo.py` can supply an Aerospike-backed one in Step 5.

---

## Step 4 — Connect to Aerospike

**What this step does:** Before you can persist anything, connect a client to the
Aerospike server. Connection settings live as typed constants at the top of
`demo.py`. Wrap the client in a context manager so the connection is always
closed, and turn a connection failure into a clear, actionable message.

```python
AEROSPIKE_HOST: str = "127.0.0.1"
AEROSPIKE_PORT: int = 3000
AEROSPIKE_NAMESPACE: str = "test"

@contextmanager
def _connect() -> Iterator[aerospike.Client]:
    client = aerospike.client({"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}).connect()
    try:
        yield client
    finally:
        client.close()
```

**In the code:** `demo.py`, marked `# === Step 4 ===` (`_connect()`), with the
`AEROSPIKE_HOST` / `AEROSPIKE_PORT` / `AEROSPIKE_NAMESPACE` constants in the
same section. Edit those constants to match your environment.

---

## Step 5 — Configure the checkpointer with a TTL

**What this step does:** Configure checkpoint expiration. Create an
`AerospikeSaver` and pass it a `ttl` dict. From this point on, *every* checkpoint
the graph writes is stamped with an expiration, and Aerospike deletes it
automatically when the time elapses.

```python
CHAT_TTL_MINUTES: int = 1
CHAT_REFRESH_ON_READ: bool = False

def _build_checkpointer(client: aerospike.Client) -> AerospikeSaver:
    return AerospikeSaver(
        client=client,
        namespace=AEROSPIKE_NAMESPACE,
        ttl={
            "default_ttl": CHAT_TTL_MINUTES,
            "refresh_on_read": CHAT_REFRESH_ON_READ,
        },
    )
```

**In the code:** `demo.py`, marked `# === Step 5 ===` (`_build_checkpointer()`),
driven by the `CHAT_TTL_MINUTES` and `CHAT_REFRESH_ON_READ` constants in the
same section. Two fields control everything:

- `default_ttl` — retention in whole minutes. The cookbook uses `1` so you
  can watch expiry happen; production is typically hours or days.
- `refresh_on_read` — when `True`, reading a checkpoint resets its TTL (sliding
  expiration, so active chats stay alive). Keep it `False` here so the TTL
  counts down while you wait for Step 9.

Internally the saver translates `default_ttl` into an Aerospike write policy, so
the TTL lands on every checkpoint, pending-write, and metadata record it creates.

---

## Step 6 — Run one chat turn and persist it

**What this step does:** Combine the graph (Steps 1–3) with the TTL
checkpointer (Step 5), then invoke the graph with a `thread_id`. The `thread_id`
is the identity of the conversation — it is the key under which Aerospike stores
this thread's checkpoints.

```python
THREAD_ID: str = "session-demo-001"

config: RunnableConfig = {"configurable": {"thread_id": THREAD_ID}}

def _say(graph: CompiledStateGraph, config: RunnableConfig, text: str) -> int:
    result = graph.invoke({"messages": [HumanMessage(text)]}, config)
    return len(result["messages"])
```

Inside `main()`, the graph and checkpointer are wired together:

```python
with _connect() as client:
    saver = _build_checkpointer(client)
    graph = build_chat_graph(saver)
    count = _say(graph, config, "Hello, I need help with my order.")
```

**In the code:** `demo.py`, marked `# === Step 6 ===` (`_say()`), with the
wiring in `main()` immediately after the `with _connect()` block. After this first turn
the thread has 2 messages (your input + the reply).

---

## Step 7 — Prove the checkpoint actually carries a TTL

**What this step does:** It is not enough to *assume* the TTL was applied — we
read it back. Aerospike stores the remaining TTL in record **metadata**, so we
locate the checkpoint record and read its `ttl` field directly.

```python
def _checkpoint_ttl_seconds(
    saver: AerospikeSaver,
    client: aerospike.Client,
    config: RunnableConfig,
) -> int | None:
    tpl = saver.get_tuple(config)
    if tpl is None:
        return None
    conf = tpl.config["configurable"]
    key = saver._key_cp(conf["thread_id"], conf["checkpoint_ns"], conf["checkpoint_id"])
    try:
        _, meta, _ = client.get(key)
    except aerospike.exception.RecordNotFound:
        return None
    ttl = meta.get("ttl")
    return ttl if isinstance(ttl, int) else None
```

**In the code:** `demo.py`, marked `# === Step 7 ===` (`_checkpoint_ttl_seconds()`).
It returns the seconds remaining, or `None` once the record no longer exists —
which is how you detect expiry in Step 9.

---

## Step 8 — Resume the same thread (history is preserved)

**What this step does:** While the TTL is still active, invoking the graph again
with the **same** `thread_id` resumes from the stored checkpoint. The new turn
appends to the existing history rather than starting over.

```python
# same config / thread_id as before
count = _say(graph, config, "Are you still tracking my conversation?")
```

**In the code:** `demo.py`, marked `# === Step 8 ===` (the "Phase 3" block in
`main()`). The message count goes from 2 to 4, which is the proof that the prior
state was loaded from Aerospike.

---

## Step 9 — Wait for expiry and prove the state is gone

**What this step does:** Wait past the TTL window, then check the same
thread. `get_tuple()` returns `None` and the raw record is gone — Aerospike
reclaimed it on its own. Invoking the thread again now starts a *fresh* session.

```python
wait_seconds = CHAT_TTL_MINUTES * 60 + 10
time.sleep(wait_seconds)

tpl = saver.get_tuple(config)
if tpl is not None:
    return 1   # checkpoint still present — TTL has not elapsed yet

count = _say(graph, config, "Hello again?")  # back to 2 messages
```

**In the code:** `demo.py`, marked `# === Step 9 ===` (the "Phase 4" wait and
"Phase 5" expiry checks in `main()`). The drop from a 4-message resumed thread
back to a 2-message fresh thread is the whole point: no cron job deleted it, the
TTL did.

---

## Run it

From the repo root, install dependencies once:

```bash
uv sync
```

Then run the demo:

```bash
# Quick validation: Steps 1–8 (create and resume a checkpoint), skip the wait.
uv run python cookbooks/expiring-chat-sessions/demo.py --skip-wait

# Full lifecycle: includes Step 9's ~70 second wait for a 1-minute TTL to expire.
uv run python cookbooks/expiring-chat-sessions/demo.py
```

## What to expect

```text
================================================================
Phase 1 - Configure TTL
================================================================
  namespace        : test
  thread_id        : session-demo-001
  default_ttl      : 1 minute(s)
  refresh_on_read  : False
  -> Every checkpoint write is stamped with this TTL by Aerospike.

================================================================
Phase 2 - Start a chat session
================================================================
  user      > Hello, I need help with my order.
  assistant > Hi! I'm your support assistant. How can I help?
  messages stored  : 2
  checkpoint TTL   : 60 seconds (set natively by Aerospike)

================================================================
Phase 3 - Resume the same thread
================================================================
  user      > Are you still tracking my conversation?
  assistant > Got it -- I've noted that down for this session.
  messages stored  : 4  (history preserved across the resume)

================================================================
Phase 4 - Wait 70s for the TTL to elapse
================================================================
  expiry window elapsed.

================================================================
Phase 5 - Prove the state expired
================================================================
  get_tuple()      : None
  raw record       : gone
  user      > Hello again?
  assistant > Hello again! This looks like a brand-new session to me.
  messages stored  : 2  (fresh session -- no prior history)

================================================================
Result
================================================================
  Aerospike expired the abandoned session automatically.
  No cron job, no sweeper, no DELETE query.
```

The exact TTL seconds can vary slightly by server timing. The message count is
the easiest proof: the resumed session has `4` messages (Step 8), while the
post-expiry session has only `2` (Step 9) because the previous checkpoints are
gone.

## Files

| File | Key functions | Steps |
| --- | --- | --- |
| `agent.py` | `ChatState`, `_make_llm()`, `build_chat_graph()` | 1–3 |
| `demo.py` | `_connect()`, `_build_checkpointer()`, `_say()`, `_checkpoint_ttl_seconds()`, `main()` | 4–9 |
