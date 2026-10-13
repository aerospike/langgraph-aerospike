import contextlib
import os

import pytest
from aerospike_sdk import Behavior, DataSet
from aerospike_sdk.exceptions import AerospikeError
from aerospike_sdk.sync import ClusterDefinition

# Single shared set name. Tests that need isolation get it from the
# `truncate_sets` fixture clearing this set before/after every test.
_STORE_SET = "store_test"


@pytest.fixture(scope="session")
def session():
    """Single Aerospike SDK session shared across the whole test session."""
    host = os.getenv("AEROSPIKE_HOST", "127.0.0.1")
    port = int(os.getenv("AEROSPIKE_PORT", "3000"))
    try:
        cluster = ClusterDefinition(host, port).connect()
    except AerospikeError as e:
        pytest.skip(f"Could not connect to Aerospike at {host}:{port}: {e}")
    sess = cluster.create_session(Behavior.DEFAULT)
    yield sess
    with contextlib.suppress(Exception):
        cluster.close()


@pytest.fixture(scope="session")
def namespace():
    """Aerospike namespace used for tests (matches Docker default)."""
    return os.getenv("AEROSPIKE_NAMESPACE", "test")


@pytest.fixture()
def truncate_sets(session, namespace):
    """Return a callable that truncates the given Aerospike sets.

    Truncate is the idiomatic Aerospike way to wipe state: a single
    server-side op per set, LUT-filtered so any record written *after* the
    call is preserved. Cheap to call before/after every test.
    """

    def _do(sets):
        for s in sets:
            ds = DataSet.of(namespace, s)
            with contextlib.suppress(AerospikeError):
                session.truncate(ds, before_nanos=0)

    return _do


@pytest.fixture(scope="session")
def vector_session():
    """Sync `aerospike_sdk` session retained for backward compatibility.

    The main ``session`` fixture is now an SDK session and can read/write
    VECTOR bins directly, so this is no longer required. It is kept so any
    existing tests that request it still get a working SDK session.
    """
    host = os.getenv("AEROSPIKE_HOST", "127.0.0.1")
    port = int(os.getenv("AEROSPIKE_PORT", "3000"))
    try:
        cluster = ClusterDefinition(host, port).connect()
    except Exception:
        yield None
        return
    try:
        yield cluster.create_session(Behavior.DEFAULT)
    finally:
        with contextlib.suppress(Exception):
            cluster.close()


@pytest.fixture()
def store(session, namespace, truncate_sets, vector_session):
    """Yield a freshly-truncated `AerospikeStore` for each test."""
    from langgraph.store.aerospike.store import AerospikeStore

    truncate_sets((_STORE_SET,))
    try:
        yield AerospikeStore(
            session=session,
            namespace=namespace,
            set=_STORE_SET,
            vector_session=vector_session,
        )
    finally:
        truncate_sets((_STORE_SET,))
