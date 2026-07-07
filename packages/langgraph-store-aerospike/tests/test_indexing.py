"""Mocked tests for secondary-index creation and query dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock

from langgraph.store.aerospike.store import AerospikeStore
from langgraph.store.base import ListNamespacesOp, MatchCondition, SearchOp


def _mock_store() -> tuple[AerospikeStore, MagicMock, MagicMock]:
    client = MagicMock()
    query = MagicMock()
    query.results.return_value = []
    client.query.return_value = query
    store = AerospikeStore(client=client, namespace="test", set="store_test")
    return store, client, query


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
    _store, client, _query = _mock_store()
    created_bins = {call.args[2] for call in client.index_list_create.call_args_list}
    assert created_bins == {"ns_prefixes", "ns_suffixes"}


# --------------------------------------------------------------------------- #
# Query dispatch: index-backed vs. full scan
# --------------------------------------------------------------------------- #


def test_search_with_anchored_prefix_uses_prefix_index():
    store, client, query = _mock_store()
    store._handle_search(SearchOp(namespace_prefix=("users", "123")))
    client.query.assert_called_once_with("test", "store_test")
    query.where.assert_called_once()
    assert "ns_prefixes" in query.where.call_args.args[0]


def test_list_namespaces_prefix_uses_prefix_index():
    store, _client, query = _mock_store()
    store._handle_list_namespaces(
        ListNamespacesOp(match_conditions=[MatchCondition(match_type="prefix", path=("a", "b"))])
    )
    query.where.assert_called_once()
    assert "ns_prefixes" in query.where.call_args.args[0]


def test_list_namespaces_suffix_uses_suffix_index():
    store, _client, query = _mock_store()
    store._handle_list_namespaces(
        ListNamespacesOp(match_conditions=[MatchCondition(match_type="suffix", path=("f",))])
    )
    query.where.assert_called_once()
    assert "ns_suffixes" in query.where.call_args.args[0]


def test_list_namespaces_unconditioned_does_full_scan():
    store, _client, query = _mock_store()
    store._handle_list_namespaces(ListNamespacesOp(match_conditions=None))
    query.where.assert_not_called()


def test_list_namespaces_leading_wildcard_prefix_does_full_scan():
    store, _client, query = _mock_store()
    store._handle_list_namespaces(
        ListNamespacesOp(
            match_conditions=[MatchCondition(match_type="prefix", path=("*", "users"))]
        )
    )
    query.where.assert_not_called()
