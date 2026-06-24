# Fork from a Checkpoint with Aerospike

When an agent derives context along the way (an order lookup, a document parse, an API call), that work is expensive to repeat. If the conversation needs to take a different path, you want to resume from a saved moment with the corrected input, not start over from scratch.

LangGraph writes a checkpoint after every step. Aerospike stores each one as a key-value record keyed by `thread_id` and `checkpoint_id`, so listing history, rehydrating a past state, and resuming from it are direct, low-latency reads with no replay or scan.

## The scenario

A customer reports broken headphones and asks for a refund. The agent identifies the order, classifies the intent, and selects a refund. The customer then says: "actually, send a replacement instead."

The corrected message no longer mentions the product. A fresh run would not know which order to use. Instead, you rewind to the checkpoint where the order was already identified and resume from there with the correction applied. The fork reuses the saved `order_id`. The original refund checkpoint stays in Aerospike and provides the before-state for a handoff note.

## When to use this pattern

- A user corrects themselves after the agent has already done expensive setup work. Forking reuses the derived context without repeating it.
- A human reviewer overrides an agent decision. Fork from before the decision point and resume with the correction applied.
- You want to explore two resolution paths from the same intermediate state without re-running the full pipeline.

## How to use this cookbook

Each step explains the code, shows it, and points to the matching location in the finished files. Open `agent.py` and `demo.py` side by side to follow along.

Files:

- `agent.py`: the LangGraph support agent (Steps 1–5)
- `demo.py`: connecting, running, listing history, rehydrating, and forking (Steps 6–12)

Most of `agent.py` lives inside one factory function, `build_support_graph(checkpointer)`: it defines the graph's nodes as inner functions, wires them together, and returns a compiled graph. The module-level helpers (`_lookup_order`, `_classify_intent`) sit outside it.

File map for `agent.py` (Steps 1–5):

```text
agent.py
├── Step 1:  SupportState                                  (module level)
├── Step 2:  _lookup_order()  + identify_order node        (lookup at module level)
├── Step 3:  _classify_intent() + classify node            (classifier at module level)
└── build_support_graph(checkpointer) -> CompiledStateGraph
    ├── Step 4:  refund / replacement / escalate(state)     (inner nodes, use order_id)
    └── Step 5:  route(state) + StateGraph wiring + compile (tail)
```

File map for `demo.py` (Steps 6–12):

```text
demo.py
├── Constants:  AEROSPIKE_HOST, THREAD_ID, ORIGINAL_REQUEST, CORRECTED_REQUEST, ...
├── Step 6:  _connect() -> Iterator[aerospike.Client]
├── Step 7:  _build_checkpointer(client) -> AerospikeSaver
├── Step 8:  _resolve(graph, config, text) -> SupportOutcome
└── main() -> int
    ├── Step 9:   list checkpoint history (Phase 3)
    ├── Step 10:  rehydrate the post-lookup checkpoint (Phase 3)
    ├── Step 11:  fork from it, reusing the saved order id (Phase 4)
    └── Step 12:  write a handoff note from saved state (Phase 5)
```

This cookbook assumes you have an Aerospike server running on port 3000. See the [repo README](../../README.md) for a one-line Docker command.

## Prerequisites

Before you start, confirm you have:

| Component | Requirement |
|-----------|-------------|
| Python | 3.10 or newer |
| [uv](https://docs.astral.sh/uv/) | Installed (repo uses `uv run` for cookbook scripts) |
| Aerospike Database | Running and reachable on port 3000 (local or remote) |
| This repo | Cloned; dependencies installed with `uv sync` from the repo root |

No LLM API keys are required. The agent uses deterministic helpers instead of a
live model.

---

## Step 1 — Define the support state

**What this step does:** The graph operates on a shared state holding the
conversation plus the context the agent derives. `order_id` is the field this
whole cookbook revolves around: it is derived early and must survive a rewind.
Use the `add_messages` reducer so resuming with a new user turn *appends* to
history.

```python
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Intent = Literal["refund", "replacement", "escalate"]

class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    item: str | None
    order_id: str | None
    intent: str | None
    resolution: str | None
```

**In the code:** `agent.py`, marked `# === Step 1 ===`.

---

## Step 2 — Look up the order (the context-deriving step)

**What this step does:** The first node reads what the customer described and
turns it into an `order_id` with a catalog lookup (a stand-in for an orders API).
It also treats `order_id` as durable state: if the id is already present, the
node leaves it alone instead of doing the lookup again.

**Part A — the lookup** (`agent.py`, module level, `# === Step 2 ===`):

```python
_CATALOG = {
    "headphones": ("AeroPro Wireless Headphones", "ORD-10482"),
    "keyboard": ("AeroPro Mechanical Keyboard", "ORD-20913"),
    "monitor": ("AeroPro 4K Monitor", "ORD-33170"),
}

def _lookup_order(text: str) -> tuple[str | None, str | None]:
    lowered = text.lower()
    for keyword, (item, order_id) in _CATALOG.items():
        if keyword in lowered:
            return item, order_id
    return None, None
```

**Part B — the node, with the reuse guard** (an inner function of
`build_support_graph`):

```python
def identify_order(state: SupportState) -> SupportState:
    if state.get("order_id"):
        return {}  # already known -- reuse it, don't look it up again
    item, order_id = _lookup_order(_latest_user_text(state["messages"]))
    return {"item": item, "order_id": order_id}
```

**In the code:** `agent.py`, `# === Step 2 ===`. That `if state.get("order_id")`
guard makes the node idempotent: if this node ever runs on state that already
has an order id, it does not repeat the lookup. In Step 11 you fork from *after*
this node, so the saved `ORD-10482` is already in the checkpoint and carries
forward into the new path.

> **Tip:** To use real systems, replace `_lookup_order` with your orders API and
> the body of `_classify_intent` (Step 3) with an LLM. The graph shape and
> Aerospike behavior stay the same.

---

## Step 3 — Detect what the customer wants

**What this step does:** A node reads the latest user message and sets `intent`.
Because it reads the *latest* message, forking from before this node and feeding
a corrected request re-runs it and produces a new intent — which is what reroutes
the graph.

```python
def _classify_intent(text: str) -> Intent:
    lowered = text.lower()
    if "replace" in lowered:
        return "replacement"
    if "refund" in lowered or "money back" in lowered:
        return "refund"
    return "escalate"

def classify(state: SupportState) -> SupportState:
    intent = _classify_intent(_latest_user_text(state["messages"]))
    return {"intent": intent}
```

**In the code:** `agent.py`, `# === Step 3 ===`.

---

## Step 4 — Resolution nodes that reference the order id

**What this step does:** Three terminal nodes, one per intent. Each reads
`order_id` out of state and bakes it into the resolution — so the resolution is
only correct if the order id survived to this point. They only write strings to
state; there are no payment or fulfillment calls.

```python
def refund(state: SupportState) -> SupportState:
    order_id = state["order_id"]
    return {
        "resolution": f"Refund selected for order {order_id}",
        "messages": [AIMessage(content=f"I can handle that as a refund for order {order_id}.")],
    }

def replacement(state: SupportState) -> SupportState:
    order_id = state["order_id"]
    return {
        "resolution": f"Replacement selected for order {order_id}",
        "messages": [
            AIMessage(content=f"I can handle that as a replacement for order {order_id}.")
        ],
    }

def escalate(state: SupportState) -> SupportState:
    order_id = state["order_id"]
    return {
        "resolution": f"Escalated order {order_id} to a human agent",
        "messages": [AIMessage(content=f"I've escalated order {order_id} to a human agent.")],
    }
```

**In the code:** `agent.py`, `# === Step 4 ===`, inside `build_support_graph()`.

---

## Step 5 — Wire routing and compile

**What this step does:** Run `identify_order` first, then `classify`, then a
conditional edge that routes to one of the resolution nodes on `intent`. The
`identify_order → classify` order is what creates the checkpoint you want to
rewind to: order id known, decision not yet made.

```python
def route(state: SupportState) -> Intent:
    return state["intent"] or "escalate"

builder = StateGraph(SupportState)
builder.add_node("identify_order", identify_order)
builder.add_node("classify", classify)
builder.add_node("refund", refund)
builder.add_node("replacement", replacement)
builder.add_node("escalate", escalate)

builder.add_edge(START, "identify_order")
builder.add_edge("identify_order", "classify")
builder.add_conditional_edges(
    "classify", route,
    {"refund": "refund", "replacement": "replacement", "escalate": "escalate"},
)
builder.add_edge("refund", END)
builder.add_edge("replacement", END)
builder.add_edge("escalate", END)

return builder.compile(checkpointer=checkpointer)
```

**In the code:** `agent.py`, `# === Step 5 ===` (tail of `build_support_graph()`).
The checkpointer is passed in so `demo.py` can supply an Aerospike-backed one.

---

## Step 6 — Connect to Aerospike

**What this step does:** Open a client, wrapped in a context manager so it is
always closed, and turn a connection failure into a clear message. Connection
settings are typed constants at the top of `demo.py`.

```python
@contextmanager
def _connect() -> Iterator[aerospike.Client]:
    client = aerospike.client({"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}).connect()
    try:
        yield client
    finally:
        client.close()
```

**In the code:** `demo.py`, `# === Step 6 ===`. Edit the constants to match your
environment.

---

## Step 7 — Build the checkpointer

**What this step does:** Create an `AerospikeSaver`. The checkpointer uses no
TTL — the checkpoint history must persist so you can travel back through it.

```python
def _build_checkpointer(client: aerospike.Client) -> AerospikeSaver:
    return AerospikeSaver(client=client, namespace=AEROSPIKE_NAMESPACE)
```

**In the code:** `demo.py`, `# === Step 7 ===`. A stable `THREAD_ID` and
`CHECKPOINT_NS` are constants too — the fork resumes from a checkpoint in this
*same* thread.

---

## Step 8 — Run the thread to a resolution

**What this step does:** Send one user turn through the graph. The agent
identifies the order, classifies, resolves, and Aerospike persists a checkpoint
at each super-step.

```python
def _resolve(graph, config, text) -> SupportOutcome:
    result = graph.invoke({"messages": [HumanMessage(text)]}, config)
    return SupportOutcome(result["order_id"], result["intent"], result["resolution"])
```

In `main()`, this is Phase 1 — the original request, *"I bought a pair of AeroPro
headphones that arrived broken. I'd like a refund."* The demo first prints the
blank starting state (`order_id=None`, `intent=None`, `resolution=None`) so it's
clear the agent derives everything from scratch, then resolves to a refund for
`ORD-10482`. The demo stashes the final checkpoint's config so Step 12 can compare the
original run with the corrected run.

Phase 2 has no new code — it captures the customer's correction (*"send a
replacement instead"*) before you touch the history. Because that correction no
longer mentions "headphones," Phase 3 has a concrete job: find a saved state that
already knows the order id.

**In the code:** `demo.py`, `# === Step 8 ===` (`_resolve`) and Phase 1 in `main()`.

---

## Step 9 — List the checkpoint history from Aerospike

**What this step does:** Ask the saver for every checkpoint belonging to the
thread. `list()` reads the thread's timeline straight out of Aerospike and yields
`CheckpointTuple`s newest-first; the demo prints them oldest-first so the story
is easier to follow. It also maps LangGraph's numeric step metadata through
`_stage()` so the table says `thread start`, `order identified`, and so on
instead of exposing raw step numbers. Watch the `order_id` column: it's empty
until `identify_order` runs, then carried by every later checkpoint.

```python
history = list(saver.list(prod_config))
for i, tpl in enumerate(reversed(history), start=1):
    values = tpl.checkpoint.get("channel_values", {})
    print(i, tpl.config["configurable"]["checkpoint_id"], _stage(tpl.metadata.get("step")),
          values.get("order_id"), values.get("intent"), values.get("resolution"))
```

**In the code:** `demo.py`, `# === Step 9 ===` (Phase 3, after the customer's
correction is introduced in Phase 2).

> Heads-up on the printed ids: LangGraph checkpoint ids are *time-ordered*, so
> within one run they share a long **leading** prefix and differ only in the
> tail. The demo prints the last 12 characters (`_short()`) so the rows are
> distinguishable — truncating the front would make them look identical.
>
> The first row is the initial checkpoint LangGraph writes before any node runs.
> LangGraph records that internal metadata as step `-1`; the demo labels it
> `thread start` so it reads like part of the timeline instead of a stray
> negative step.

---

## Step 10 — Rehydrate the checkpoint where the order id is known

**What this step does:** Pick the moment to rewind to — the earliest checkpoint
where `order_id` is set but nothing has been resolved yet (the point *after* the
lookup, *before* the decision). Scanning oldest-first finds it; one `get_tuple()`
rehydrates it.

```python
fork_point = next(
    (tpl for tpl in reversed(history)
     if _values(tpl).get("order_id") and not _values(tpl).get("resolution")),
    history[-1],
)
rehydrated = saver.get_tuple(fork_point.config)   # one checkpoint read from Aerospike
# rehydrated state: order_id == "ORD-10482", intent == None, resolution == None
```

**In the code:** `demo.py`, `# === Step 10 ===` (also Phase 3 — listing and
rehydrating are both "look at what Aerospike saved"). The order id is sitting
right there in the rehydrated state — the checkpoint state is restored, and the
order lookup node does not run again.

---

## Step 11 — Fork from it, reusing the saved order id

**What this step does:** Resume from the fork point. `fork_point.config` carries the
historical `checkpoint_id`, so invoking the graph with it resumes from that saved
state. Because the fork point is *after* `identify_order`, the saved `order_id`
is already present and the resumed path continues at `classify`. The corrected
message changes the intent to `replacement`, and the resolution node uses the
same `ORD-10482`. LangGraph writes new checkpoints from here; the refund run from
Phase 1 stays in Aerospike under its own checkpoint ids.

```python
fork_config = fork_point.config                       # thread_id + checkpoint_ns + checkpoint_id
forked = _resolve(graph, fork_config, CORRECTED_REQUEST)   # "...send a replacement instead."
# forked.order_id   == "ORD-10482"   (reused from the checkpoint)
# forked.resolution == "Replacement selected for order ORD-10482"
```

**In the code:** `demo.py`, `# === Step 11 ===` (Phase 4). The demo asserts
`forked.order_id == original.order_id` — proof the rewind preserved the context.

---

## Step 12 — Use history to write the handoff note

**What this step does:** Forking resumes from an earlier checkpoint, so the refund
checkpoints were never touched. Read the original decision back out of Aerospike
by its checkpoint id and use it with the corrected run to write a short internal
handoff note. That gives the kept history a concrete purpose: the note says what
the customer first asked for, what changed, which order id was reused, and which
outcome to use now.

```python
original_after = saver.get_tuple(original_final_config).checkpoint["channel_values"]
messages = original_after.get("messages") or []
based_on = messages[0].content if messages else "(unknown)"

print("Internal note from saved state:")
print("  Customer first asked for a refund, then corrected the request to a replacement.")
print(f"  Order id stayed {forked.order_id}; no second order lookup was needed.")
print(f"  Use current outcome: {forked.resolution}")

print("Source fields read from Aerospike:")
print(f'  original request : "{based_on}"')
print(f"  original decision: {original_after.get('resolution')}")
print(f'  corrected request: "{CORRECTED_REQUEST}"')
```

The demo prints a compact handoff note from saved state:

- original request — broken headphones, refund requested
- original decision — `Refund selected for order ORD-10482`
- corrected request — replacement requested
- current outcome — `Replacement selected for order ORD-10482`
- reused context — the same `ORD-10482`

Both `resolution` strings are graph state — this demo never calls payment or
fulfillment systems.

**In the code:** `demo.py`, `# === Step 12 ===` (Phase 5). The demo returns a
non-zero exit code if the historical checkpoint changed, the fork failed to reroute, or
the order id wasn't reused.

---

## Run it

From the repo root, install dependencies once:

```bash
uv sync
```

Then run the demo:

```bash
uv run python cookbooks/agent-path-correction/demo.py
```

The walkthrough **pauses between phases** — press Enter to advance through each
one so you can follow along.

## What to expect

You'll be prompted (`... press Enter for the next step ...`) between each phase.
Shown here without the prompts:

```text
================================================================
Phase 1 - The original ticket
================================================================
  starting state : order_id=None  intent=None  resolution=None  (nothing derived yet)
  A ticket arrives. The agent works the order out from scratch:
  user        > I bought a pair of AeroPro headphones that arrived broken. I'd like a refund.
  assistant   > I can handle that as a refund for order ORD-10482.
  order_id    : ORD-10482  (derived from 'headphones')
  intent      : refund
  resolution  : Refund selected for order ORD-10482

================================================================
Phase 2 - The customer changes their mind
================================================================
  customer    > On second thought, please send a replacement instead.
  This request does not mention headphones, so a fresh run would not know
  which order to use. You need a saved checkpoint that already has order_id.

================================================================
Phase 3 - Find the reuse point in Aerospike
================================================================
  pending request: "On second thought, please send a replacement instead."
  Need: order_id set, but no decision selected yet.

  seq checkpoint    saved after       order_id   intent       resolution
  1   20ec38b21e75  thread start      None       None         None
  2   84284ccc7cdd  request received  None       None         None
  3   17d2ccd0475f  order identified  ORD-10482  None         None
  4   168a498759db  intent classified ORD-10482  refund       None
  5   7ebb1769c0a7  decision selected ORD-10482  refund       Refund selected for order ORD-10482
  -> 5 checkpoints saved for this thread.

  reuse point : 17d2ccd0475f (order identified, not yet resolved)
  order_id    : ORD-10482  <- already derived, sitting in the checkpoint
  -> Restored the checkpoint state; the order lookup node did not run again.

================================================================
Phase 4 - Fork from that checkpoint
================================================================
  user        > On second thought, please send a replacement instead.
  assistant   > I can handle that as a replacement for order ORD-10482.
  order_id    : ORD-10482  (reused from the checkpoint)
  intent      : replacement
  resolution  : Replacement selected for order ORD-10482
  -> The correction never mentioned the product, yet the order id carried over.

================================================================
Phase 5 - Use history to write the handoff note
================================================================
  Internal note from saved state:
    Customer first asked for a refund, then corrected the request to a replacement.
    Order id stayed ORD-10482; no second order lookup was needed.
    Use current outcome: Replacement selected for order ORD-10482

  Source fields read from Aerospike:
    original request : "I bought a pair of AeroPro headphones that arrived broken. I'd like a refund."
    original decision: Refund selected for order ORD-10482
    corrected request: "On second thought, please send a replacement instead."
  -> The kept checkpoint supplies the before-state for this handoff note.

================================================================
Result
================================================================
  - Rewound to a saved checkpoint that already knew the order id and
    resumed down a new path -- reusing that context, not recomputing it.
  - The old final checkpoint is useful context for a handoff note:
    what the customer first asked for, what changed, and what to do now.
```

The exact `checkpoint_id`s will differ on your run. The proof is in three facts:
the rehydrated checkpoint already holds `ORD-10482`, the fork reuses that same id
while rerouting to `replacement`, and the original `refund` decision supplies
the before-state for the handoff note.

## Files

| File | Key functions | Steps |
| --- | --- | --- |
| `agent.py` | `SupportState`, `_lookup_order()`, `_classify_intent()`, `build_support_graph()` | 1–5 |
| `demo.py` | `_connect()`, `_build_checkpointer()`, `_resolve()`, `main()` | 6–12 |
