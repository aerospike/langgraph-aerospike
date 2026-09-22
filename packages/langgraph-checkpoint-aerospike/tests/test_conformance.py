"""Run the upstream LangGraph checkpoint-conformance suite against AerospikeSaver.

This is the canonical way to validate a third-party `BaseCheckpointSaver`
implementation: `langgraph-checkpoint-conformance` ships a battery of
async tests covering put / put_writes / get_tuple / list / delete_thread
plus the optional `delete_for_runs` / `copy_thread` / `prune` extensions
(auto-detected and skipped if not overridden).

If this test fails, AerospikeSaver has drifted from the public
BaseCheckpointSaver contract and any consumer relying on the standard
LangGraph API may break.
"""

from __future__ import annotations

import asyncio
import contextlib
import os

import pytest
from aerospike_sdk import Behavior, DataSet
from aerospike_sdk.exceptions import AerospikeError
from aerospike_sdk.sync import ClusterDefinition
from langgraph.checkpoint.aerospike import AerospikeSaver
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.report import ProgressCallbacks

# These match the env-var conventions used by the rest of the test suite
# (see conftest.py). The conformance suite spins up many threads and
# namespaces, so we wipe all three sets before each run to avoid bleed-in
# from previous tests / previous CI runs.
_HOST = os.getenv("AEROSPIKE_HOST", "localhost")
_PORT = int(os.getenv("AEROSPIKE_PORT", "3000"))
_NAMESPACE = os.getenv("AEROSPIKE_NAMESPACE", "test")
_SETS = ("lg_cp", "lg_cp_w", "lg_cp_meta")


def _connect():
    return ClusterDefinition(_HOST, _PORT).connect()


def _truncate_all(session) -> None:
    """Wipe every checkpoint-related set so the suite starts from empty."""
    for s in _SETS:
        ds = DataSet.of(_NAMESPACE, s)
        with contextlib.suppress(AerospikeError):
            session.truncate(ds, before_nanos=0)


@checkpointer_test(name="AerospikeSaver")
async def aerospike_checkpointer():
    """Async-generator factory the conformance suite calls per test set.

    The synchronous Aerospike SDK cannot be used directly inside a running
    asyncio event loop, so all blocking setup and teardown is offloaded to
    worker threads via ``asyncio.to_thread``.
    """
    cluster = await asyncio.to_thread(_connect)
    session = await asyncio.to_thread(cluster.create_session, Behavior.DEFAULT)
    await asyncio.to_thread(_truncate_all, session)
    saver = await asyncio.to_thread(AerospikeSaver, session, _NAMESPACE)
    try:
        yield saver
    finally:
        await asyncio.to_thread(_truncate_all, session)
        with contextlib.suppress(Exception):
            await asyncio.to_thread(cluster.close)


def test_aerospike_saver_passes_base_conformance():
    """Hard requirement: every base capability must pass.

    Extended capabilities (delete_for_runs, copy_thread, prune) are
    auto-skipped because AerospikeSaver doesn't override them yet;
    when those are added, the suite picks them up automatically.
    """
    try:
        cluster = _connect()
        cluster.close()
    except AerospikeError as e:
        pytest.skip(f"Could not connect to Aerospike: {e}")

    report = asyncio.run(
        validate(
            aerospike_checkpointer,
            progress=ProgressCallbacks.verbose(),
        )
    )
    report.print_report()
    assert report.passed_all_base(), (
        "AerospikeSaver failed the LangGraph base checkpoint conformance suite. "
        "See printed report above for the failing capability."
    )
