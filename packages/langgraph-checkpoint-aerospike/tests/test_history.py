from __future__ import annotations

from unittest.mock import MagicMock

import pytest

_DEFAULT_CHECKPOINT_SETS = ("lg_cp", "lg_cp_w", "lg_cp_meta")


def test_list_enumerates_via_secondary_index_mocked():
    """``list()`` queries the ``thread_id`` index and sorts newest-first."""
    from langgraph.checkpoint.aerospike import AerospikeSaver

    mock_client = MagicMock()
    mock_query = MagicMock()
    mock_client.query.return_value = mock_query
    rows = [
        (None, None, {"checkpoint_ns": "ns1", "checkpoint_id": "c1", "ts": "2026-01-01T00:00:00"}),
        (None, None, {"checkpoint_ns": "ns1", "checkpoint_id": "c3", "ts": "2026-01-03T00:00:00"}),
        (None, None, {"checkpoint_ns": "ns1", "checkpoint_id": "c2", "ts": "2026-01-02T00:00:00"}),
        # Different namespace for the same thread -> must be excluded.
        (None, None, {"checkpoint_ns": "other", "checkpoint_id": "x", "ts": "2026-01-09T00:00:00"}),
    ]
    mock_query.results.return_value = rows

    saver = AerospikeSaver(client=mock_client, namespace="test", ttl={})
    pairs = saver._list_checkpoint_ids("t1", "ns1")

    assert [cid for _, cid in pairs] == ["c3", "c2", "c1"]
    mock_client.query.assert_called_once_with("test", "lg_cp")
    mock_query.where.assert_called_once()
    mock_query.select.assert_called_once_with("checkpoint_ns", "checkpoint_id", "ts")


@pytest.fixture
def history_saver(aerospike_saver_cls, client, aerospike_namespace, truncate_sets):
    """A saver with a plain TTL, truncated before/after each test."""
    truncate_sets(_DEFAULT_CHECKPOINT_SETS)
    saver = aerospike_saver_cls(
        client=client,
        namespace=aerospike_namespace,
        ttl={"default_ttl": 60},
    )
    try:
        yield saver
    finally:
        truncate_sets(_DEFAULT_CHECKPOINT_SETS)


def _put(saver, thread_id, ns, cid, ts):
    cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns}}
    saver.put(cfg, {"id": cid, "ts": ts}, {}, {})


def _ids(tuples):
    return [t.config["configurable"]["checkpoint_id"] for t in tuples]


def test_list_orders_limits_and_filters_namespace(history_saver):
    """Integration: list() is newest-first, honors limit/before, and is ns-scoped."""
    s = history_saver
    _put(s, "t", "nsA", "a1", "2026-01-01T00:00:00+00:00")
    _put(s, "t", "nsA", "a2", "2026-01-02T00:00:00+00:00")
    _put(s, "t", "nsA", "a3", "2026-01-03T00:00:00+00:00")
    # Same thread, different namespace: must not leak into the nsA listing.
    _put(s, "t", "nsB", "b1", "2026-01-09T00:00:00+00:00")

    cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": "nsA"}}

    assert _ids(s.list(cfg)) == ["a3", "a2", "a1"]
    assert _ids(s.list(cfg, limit=2)) == ["a3", "a2"]

    before = {"configurable": {"thread_id": "t", "checkpoint_ns": "nsA", "checkpoint_id": "a3"}}
    assert _ids(s.list(cfg, before=before)) == ["a2", "a1"]

    cfg_b = {"configurable": {"thread_id": "t", "checkpoint_ns": "nsB"}}
    assert _ids(s.list(cfg_b)) == ["b1"]


def test_list_filters_by_metadata(history_saver):
    """Integration: list(filter=...) returns only checkpoints whose metadata matches."""
    s = history_saver
    cfg = {"configurable": {"thread_id": "tf", "checkpoint_ns": "ns"}}
    s.put(cfg, {"id": "m1", "ts": "2026-03-01T00:00:00+00:00"}, {"step": 1, "kind": "a"}, {})
    s.put(cfg, {"id": "m2", "ts": "2026-03-02T00:00:00+00:00"}, {"step": 2, "kind": "b"}, {})
    s.put(cfg, {"id": "m3", "ts": "2026-03-03T00:00:00+00:00"}, {"step": 3, "kind": "a"}, {})

    # Newest-first, and only "kind == a" survives the filter.
    assert _ids(s.list(cfg, filter={"kind": "a"})) == ["m3", "m1"]
    assert _ids(s.list(cfg, filter={"kind": "b"})) == ["m2"]


def test_list_handles_large_history_without_single_record(history_saver):
    """Many checkpoints in one thread/ns must all be listable.

    The old single ``__timeline__`` map record would eventually hit
    Aerospike's max record size; the secondary-index enumeration stores one
    record per checkpoint instead, so this scales without a per-record cap.
    """
    s = history_saver
    n = 50
    for i in range(n):
        _put(s, "big", "ns", f"ck-{i:04d}", f"2026-02-01T00:{i // 60:02d}:{i % 60:02d}+00:00")

    cfg = {"configurable": {"thread_id": "big", "checkpoint_ns": "ns"}}
    listed = _ids(s.list(cfg))
    assert len(listed) == n
    # Newest-first.
    assert listed[0] == "ck-0049"
    assert listed[-1] == "ck-0000"
