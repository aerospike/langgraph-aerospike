"""A small support agent that builds up context before resolving a ticket.

The interesting part of this graph is that resolving a ticket depends on
context the agent *derives earlier in the conversation*: it reads what the
customer says they bought, looks up the matching order id, and only then
issues a resolution that references that order. That derived order id is the
"expensive" piece of state -- the thing we don't want to recompute -- which is
exactly what a checkpoint is good at holding onto.

There is intentionally no LLM here. Order lookup and intent detection are pure
functions of the conversation, so the demo's fork is reproducible: the only
thing that changes on the new path is the customer's corrected request.
Swapping either helper for a real model (entity extraction, an orders API,
intent classification) is a localized change; the graph shape and the Aerospike
checkpoint behavior are identical.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

# The three paths the graph can resolve down. Routing on this value is what
# makes a fork visible: change the request, get a different path.
Intent = Literal["refund", "replacement", "escalate"]

# Stand-in for an orders database. In production this lookup would be an API
# call keyed off an extracted product name; here a keyword is enough.
_CATALOG: dict[str, tuple[str, str]] = {
    "headphones": ("AeroPro Wireless Headphones", "ORD-10482"),
    "keyboard": ("AeroPro Mechanical Keyboard", "ORD-20913"),
    "monitor": ("AeroPro 4K Monitor", "ORD-33170"),
}


# === Step 1: Define the support state ===
class SupportState(TypedDict):
    # `add_messages` appends, so resuming an older checkpoint with a new user
    # turn grows the history instead of replacing it.
    messages: Annotated[list[BaseMessage], add_messages]
    # Context the agent derives from the conversation. `order_id` is the value
    # we care about preserving across a rewind.
    item: str | None
    order_id: str | None
    intent: str | None
    resolution: str | None


def _latest_user_text(messages: list[BaseMessage]) -> str:
    """Return the content of the most recent human message, or ``""``."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


# === Step 2: Look up the order from what the customer described ===
def _lookup_order(text: str) -> tuple[str | None, str | None]:
    """Map a product mentioned in free text to its (item_name, order_id)."""
    lowered = text.lower()
    for keyword, (item, order_id) in _CATALOG.items():
        if keyword in lowered:
            return item, order_id
    return None, None


# === Step 3: Detect what the customer is asking for ===
def _classify_intent(text: str) -> Intent:
    """Map free text to an intent. Deterministic on purpose (see module docs)."""
    lowered = text.lower()
    if "replace" in lowered:
        return "replacement"
    if "refund" in lowered or "money back" in lowered:
        return "refund"
    return "escalate"


def build_support_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Compile the support graph backed by the given checkpointer.

    Flow: identify_order -> classify -> (refund | replacement | escalate).
    `identify_order` runs first and seeds `order_id`; everything downstream
    references it. That ordering is what lets a rewind to a post-lookup
    checkpoint reuse the order id instead of deriving it again.
    """

    def identify_order(state: SupportState) -> SupportState:
        # Reuse the order id if this node ever runs on state that already has
        # one. The demo forks from after this node, so the saved id simply
        # carries forward; this guard keeps the lookup idempotent in variants
        # that may re-enter the node with an existing order id.
        if state.get("order_id"):
            return {}  # type: ignore[return-value]
        item, order_id = _lookup_order(_latest_user_text(state["messages"]))
        return {"item": item, "order_id": order_id}  # type: ignore[return-value]

    def classify(state: SupportState) -> SupportState:
        intent = _classify_intent(_latest_user_text(state["messages"]))
        return {"intent": intent}  # type: ignore[return-value]

    # === Step 4: Resolution nodes -- each references the derived order id ===
    def refund(state: SupportState) -> SupportState:
        order_id = state["order_id"]
        return {  # type: ignore[return-value]
            "resolution": f"Refund selected for order {order_id}",
            "messages": [AIMessage(content=f"I can handle that as a refund for order {order_id}.")],
        }

    def replacement(state: SupportState) -> SupportState:
        order_id = state["order_id"]
        return {  # type: ignore[return-value]
            "resolution": f"Replacement selected for order {order_id}",
            "messages": [
                AIMessage(content=f"I can handle that as a replacement for order {order_id}.")
            ],
        }

    def escalate(state: SupportState) -> SupportState:
        order_id = state["order_id"]
        return {  # type: ignore[return-value]
            "resolution": f"Escalated order {order_id} to a human agent",
            "messages": [AIMessage(content=f"I've escalated order {order_id} to a human agent.")],
        }

    # === Step 5: Wire routing and compile ===
    def route(state: SupportState) -> Intent:
        # The conditional edge reads the intent that `classify` just set.
        return state["intent"] or "escalate"  # type: ignore[return-value]

    builder = StateGraph(SupportState)
    builder.add_node("identify_order", identify_order)
    builder.add_node("classify", classify)
    builder.add_node("refund", refund)
    builder.add_node("replacement", replacement)
    builder.add_node("escalate", escalate)

    builder.add_edge(START, "identify_order")
    builder.add_edge("identify_order", "classify")
    builder.add_conditional_edges(
        "classify",
        route,
        {"refund": "refund", "replacement": "replacement", "escalate": "escalate"},
    )
    builder.add_edge("refund", END)
    builder.add_edge("replacement", END)
    builder.add_edge("escalate", END)

    return builder.compile(checkpointer=checkpointer)
