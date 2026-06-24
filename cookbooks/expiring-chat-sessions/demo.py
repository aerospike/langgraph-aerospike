"""Auto-expiring chat sessions, end to end.

Run this against a local Aerospike to watch a LangGraph chat session save its
state, resume from that state, and then *vanish on its own* once the Aerospike
TTL elapses -- no cron job, no cleanup query.

    uv run python cookbooks/expiring-chat-sessions/demo.py
    uv run python cookbooks/expiring-chat-sessions/demo.py --skip-wait   # phases 1-3 only

Phases:
    1. Connect and configure TTL on the checkpointer.
    2. Start a chat session and prove the checkpoint carries a TTL.
    3. Resume the same thread -- history is intact.
    4. Wait for the TTL to elapse.
    5. Prove the state is gone and a new turn starts fresh.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from contextlib import contextmanager

import aerospike
from agent import build_chat_graph
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.aerospike import AerospikeSaver
from langgraph.graph.state import CompiledStateGraph

# Edit these values for your environment.
AEROSPIKE_HOST: str = "127.0.0.1"
AEROSPIKE_PORT: int = 3000
AEROSPIKE_NAMESPACE: str = "test"

# AerospikeSaver expresses TTL in whole minutes. One minute keeps the cookbook
# quick enough to run while still proving real server-side expiry.
CHAT_TTL_MINUTES: int = 1

# Keep this false for the expiry demo so the TTL actually counts down while we
# wait. In production, set it true for sliding TTL: active chats stay alive.
CHAT_REFRESH_ON_READ: bool = False

THREAD_ID: str = "session-demo-001"


def _hr(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


# === Step 4: Connect to Aerospike ===
@contextmanager
def _connect() -> Iterator[aerospike.Client]:
    """Open an Aerospike client, guaranteeing it is closed afterwards."""
    try:
        client = aerospike.client({"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}).connect()
    except aerospike.exception.AerospikeError as exc:
        raise SystemExit(
            f"Could not connect to Aerospike at {AEROSPIKE_HOST}:{AEROSPIKE_PORT}. "
            "Confirm your Aerospike server is running and update the constants "
            "at the top of demo.py if needed."
        ) from exc
    try:
        yield client
    finally:
        client.close()


# === Step 5: Configure the checkpointer with a TTL (the heart of the cookbook) ===
def _build_checkpointer(client: aerospike.Client) -> AerospikeSaver:
    """Create an AerospikeSaver that applies a native TTL to every checkpoint."""
    return AerospikeSaver(
        client=client,
        namespace=AEROSPIKE_NAMESPACE,
        ttl={
            "default_ttl": CHAT_TTL_MINUTES,
            "refresh_on_read": CHAT_REFRESH_ON_READ,
        },
    )


# === Step 7: Read the TTL back off the stored checkpoint ===
def _checkpoint_ttl_seconds(
    saver: AerospikeSaver,
    client: aerospike.Client,
    config: RunnableConfig,
) -> int | None:
    """Read the raw TTL (seconds) Aerospike holds for the latest checkpoint.

    Aerospike stores TTL in record *metadata*, not in a bin -- the same fact
    the package's own TTL tests assert on.
    """
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


# === Step 6: Run one chat turn and persist it ===
def _say(graph: CompiledStateGraph, config: RunnableConfig, text: str) -> int:
    """Send one user turn; return the resulting message count for the thread."""
    result = graph.invoke({"messages": [HumanMessage(text)]}, config)
    messages = result["messages"]
    print(f"  user      > {text}")
    print(f"  assistant > {messages[-1].content}")
    return len(messages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Run phases 1-3 only (skip the TTL wait and expiry proof).",
    )
    args = parser.parse_args()

    config: RunnableConfig = {"configurable": {"thread_id": THREAD_ID}}

    with _connect() as client:
        # Step 6: combine the graph (Steps 1-3) with the TTL checkpointer (Step 5).
        saver = _build_checkpointer(client)
        graph = build_chat_graph(saver)

        _hr("Phase 1 - Configure TTL")
        print(f"  namespace        : {AEROSPIKE_NAMESPACE}")
        print(f"  thread_id        : {THREAD_ID}")
        print(f"  default_ttl      : {CHAT_TTL_MINUTES} minute(s)")
        print(f"  refresh_on_read  : {CHAT_REFRESH_ON_READ}")
        print("  -> Every checkpoint write is stamped with this TTL by Aerospike.")

        # Start clean so re-runs are reproducible regardless of prior state.
        saver.delete_thread(THREAD_ID)

        _hr("Phase 2 - Start a chat session")
        count = _say(graph, config, "Hello, I need help with my order.")
        print(f"  messages stored  : {count}")
        ttl = _checkpoint_ttl_seconds(saver, client, config)
        print(f"  checkpoint TTL   : {ttl} seconds (set natively by Aerospike)")

        # === Step 8: Resume the same thread (history is preserved) ===
        _hr("Phase 3 - Resume the same thread")
        count = _say(graph, config, "Are you still tracking my conversation?")
        print(f"  messages stored  : {count}  (history preserved across the resume)")

        if args.skip_wait:
            _hr("Done (--skip-wait)")
            print("  Skipped the TTL wait. Re-run without --skip-wait to watch it expire.")
            return 0

        # === Step 9: Wait for expiry, then prove the state is gone ===
        wait_seconds = CHAT_TTL_MINUTES * 60 + 10
        _hr(f"Phase 4 - Wait {wait_seconds}s for the TTL to elapse")
        for remaining in range(wait_seconds, 0, -5):
            print(f"  ...{remaining:>3}s until expiry", end="\r", flush=True)
            time.sleep(min(5, remaining))
        print("  expiry window elapsed.            ")

        _hr("Phase 5 - Prove the state expired")
        tpl = saver.get_tuple(config)
        print(f"  get_tuple()      : {tpl!r}")
        ttl = _checkpoint_ttl_seconds(saver, client, config)
        print(f"  raw record       : {'gone' if ttl is None else f'still here (ttl={ttl})'}")

        if tpl is not None:
            print("  ! Checkpoint still present -- the TTL has not elapsed yet.")
            return 1

        count = _say(graph, config, "Hello again?")
        print(f"  messages stored  : {count}  (fresh session -- no prior history)")

        _hr("Result")
        print("  Aerospike expired the abandoned session automatically.")
        print("  No cron job, no sweeper, no DELETE query.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
