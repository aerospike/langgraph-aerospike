"""Mocked tests for secondary-index creation and query dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock

from aerospike_sdk import DataSet
from langgraph.store.aerospike.store import AerospikeStore
from langgraph.store.base import ListNamespacesOp, MatchCondition, SearchOp


def _mock_store() -> tuple[AerospikeStore, MagicMock, MagicMock]:
    session = MagicMock()
    query = MagicMock()
    query.execute.return_value = []
    session.query.return_value = query
    store = AerospikeStore(session=session, namespace="test", set="store_test")
    return store, session, query


# --------------------------------------------------------------------------- #
# Pure helpers (denormalized anchors)
# --------------------------------------------------------------------------- #


def test_ns_prefixes_are_contiguous_joined_prefixes():
    assert AerospikeStore._ns_prefixes(("a", "b", "c")) == ["a", "a|b", "a|b|c"]


def test_ns_suffixes_are_contiguous_joined_suffixes():
    assert AerospikeStore._ns_suffixes(("a", "b", "c")) == ["c", "b|c", "a|b|c"]


def test_leading_anchor_stops_at_first_wildcard():
    assert AerospikeStore._leading_anchor(("a", "b", "*")) == "a|b"
    assert AerospikeStore._leading_anchor(("*", "b")) is None


def test_trailing_anchor_stops_at_last_wildcard():
    assert AerospikeStore._trailing_anchor(("*", "b", "c")) == "b|c"
    assert AerospikeStore._trailing_anchor(("b", "*")) is None


# --------------------------------------------------------------------------- #
# Index creation
# --------------------------------------------------------------------------- #


def test_construction_creates_prefix_and_suffix_indexes():
    _store, session, _query = _mock_store()
    index_builder = session.index.return_value
    created_bins = {
        call.kwargs.get("bin_name") or call.args[0]
        for call in index_builder.on_bin.call_args_list
    }
    assert created_bins == {"ns_prefixes", "ns_suffixes"}


# --------------------------------------------------------------------------- #
# Query dispatch: index-backed vs. full scan
# --------------------------------------------------------------------------- #


def test_search_with_anchored_prefix_uses_index():
    store, session, query = _mock_store()
    store._handle_search(SearchOp(namespace_prefix=("users", "123")))
    session.query.assert_called_once_with(DataSet.of("test", "store_test"))
    query.where.assert_called_once()


def test_list_namespaces_prefix_uses_index():
    store, _session, query = _mock_store()
    store._handle_list_namespaces(
        ListNamespacesOp(match_conditions=[MatchCondition(match_type="prefix", path=("a", "b"))])
    )
    query.where.assert_called_once()


def test_list_namespaces_suffix_uses_index():
    store, _session, query = _mock_store()
    store._handle_list_namespaces(
        ListNamespacesOp(match_conditions=[MatchCondition(match_type="suffix", path=("f",))])
    )
    query.where.assert_called_once()


def test_list_namespaces_unconditioned_does_full_scan():
    _store, _session, query = _mock_store()
    store = AerospikeStore(session=_session, namespace="test", set="store_test")
    store._handle_list_namespaces(ListNamespacesOp(match_conditions=None))
    query.where.assert_not_called()


def test_list_namespaces_leading_wildcard_prefix_uses_expression_filter():
    """A leading wildcard cannot use an index, but an expression filter is still applied."""
    _store, _session, query = _mock_store()
    store = AerospikeStore(session=_session, namespace="test", set="store_test")
    store._handle_list_namespaces(
        ListNamespacesOp(
            match_conditions=[MatchCondition(match_type="prefix", path=("*", "users"))]
        )
    )
    query.where.assert_called_once()
