"""A minimal, production-shaped chat agent.

The agent itself is intentionally boring -- the point of this cookbook is the
checkpoint lifecycle, not model quality. We use ``FakeListChatModel`` so the
demo is fully deterministic and needs no API keys or local model server
(no Ollama, no OpenAI). Swapping in a real model is a one-line change:

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")

The graph shape (StateGraph + add_messages + a single model node) is identical
either way, and so is the Aerospike checkpoint behavior.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph


# === Step 1: Define the conversation state ===
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# === Step 2: Create the model node (the model here, the node in build_chat_graph) ===
def _make_llm() -> BaseChatModel:
    # Returns the generic BaseChatModel interface so swapping FakeListChatModel
    # for ChatOpenAI (etc.) is a one-line change with no signature churn.
    # Scripted replies are consumed in order across calls. Deterministic by
    # design; the ordering matches the demo's three turns (start, resume,
    # post-expiry fresh start).
    return FakeListChatModel(
        responses=[
            "Hi! I'm your support assistant. How can I help?",
            "Got it -- I've noted that down for this session.",
            "Hello again! This looks like a brand-new session to me.",
        ]
    )


def build_chat_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Compile a one-node chat graph backed by the given checkpointer."""
    llm = _make_llm()

    # Step 2 (cont.): the node is just a function over the state.
    def chatbot(state: ChatState) -> ChatState:
        system = SystemMessage(content="You are a concise, helpful support assistant.")
        response: AIMessage = llm.invoke([system, *state["messages"]])
        return {"messages": [response]}

    # === Step 3: Build and compile the graph ===
    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    return builder.compile(checkpointer=checkpointer)
