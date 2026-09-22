"""Spec-style contract tests for ``BaseStore`` implementations.

These tests are ported from upstream LangGraph's
``libs/checkpoint/tests/test_store.py`` and parameterized over both
``InMemoryStore`` (the reference implementation) and ``AerospikeStore``.

The intent is to act as a hand-rolled "store conformance suite": any test
that passes against ``InMemoryStore`` but fails against ``AerospikeStore`` is
a real semantic divergence in our adapter. This is the closest substitute we
have until upstream ships an official ``langgraph-store-conformance``
package (today only ``langgraph-checkpoint-conformance`` exists, which only
covers ``BaseCheckpointSaver``).

Things deliberately NOT covered here:

* Vector / embedding behavior. ``AerospikeStore`` explicitly raises
  ``NotImplementedError`` when ``query=...`` is passed; that contract is
  asserted in ``test_aerospike_rejects_semantic_search``.
* ``AsyncBatchedBaseStore`` machinery (cancellation, deduplication,
  background-task resilience). Those upstream tests exercise the async
  batching layer itself, not the BaseStore contract that adapters must
  satisfy.

Async paths are bridged through ``asyncio.run`` rather than depending on
``pytest-asyncio`` so we don't need to add a new test dependency.
"""

from __future__ import annotations

import asyncio

import pytest
from langgraph.store.aerospike.store import AerospikeStore
from langgraph.store.base import (
    InvalidNamespaceError,
    PutOp,
    SearchOp,
)
from langgraph.store.memory import InMemoryStore

# Dedicated set so this suite stays isolated from the other store tests
# that target the conftest-default `store_test` set.
_AEROSPIKE_SET = "store_spec"


@pytest.fixture(params=["inmemory", "aerospike"])
def spec_store(request, session, namespace, truncate_sets):
    """Yield a freshly-emptied store for each backend under test.

    Parameterized so every test runs once against ``InMemoryStore`` (the
    reference) and once against ``AerospikeStore``. Tests can read
    ``request.param`` if they need to branch on backend.
    """
    label = request.param
    if label == "inmemory":
        yield InMemoryStore(), label
        return

    truncate_sets((_AEROSPIKE_SET,))
    try:
        yield AerospikeStore(session=session, namespace=namespace, set=_AEROSPIKE_SET), label
    finally:
        truncate_sets((_AEROSPIKE_SET,))


# --------------------------------------------------------------------------- #
# Basic CRUD
# --------------------------------------------------------------------------- #


def test_put_and_get_roundtrip(spec_store):
    store, _ = spec_store
    ns = ("users", "alice")
    doc = {"email": "alice@example.com", "age": 30}

    store.put(ns, "profile", doc)

    item = store.get(ns, "profile")
    assert item is not None
    assert item.value == doc
    assert item.namespace == ns
    assert item.key == "profile"


def test_get_missing_returns_none(spec_store):
    store, _ = spec_store
    assert store.get(("nope",), "missing") is None


def test_put_overwrites_existing_value(spec_store):
    store, _ = spec_store
    ns = ("users", "bob")
    store.put(ns, "profile", {"v": 1})
    store.put(ns, "profile", {"v": 2})

    item = store.get(ns, "profile")
    assert item is not None
    assert item.value == {"v": 2}


def test_delete_removes_item(spec_store):
    store, _ = spec_store
    ns = ("docs",)
    store.put(ns, "to_remove", {"v": 1})
    assert store.get(ns, "to_remove") is not None

    store.delete(ns, "to_remove")
    assert store.get(ns, "to_remove") is None


def test_put_value_none_deletes_via_batch(spec_store):
    store, _ = spec_store
    ns = ("docs",)
    store.put(ns, "k", {"v": 1})
    store.batch([PutOp(ns, "k", None)])
    assert store.get(ns, "k") is None


# --------------------------------------------------------------------------- #
# Async CRUD (sync test bodies, async work bridged via asyncio.run)
# --------------------------------------------------------------------------- #


def _arun(coro):
    return asyncio.run(coro)


def test_aput_and_aget_roundtrip(spec_store):
    store, _ = spec_store
    ns = ("users", "carol")
    doc = {"v": 42}

    async def _go():
        await store.aput(ns, "k", doc)
        return await store.aget(ns, "k")

    item = _arun(_go())
    assert item is not None
    assert item.value == doc


def test_adelete_removes_item(spec_store):
    store, _ = spec_store
    ns = ("docs",)

    async def _go():
        await store.aput(ns, "k", {"v": 1})
        await store.adelete(ns, "k")
        return await store.aget(ns, "k")

    assert _arun(_go()) is None


# --------------------------------------------------------------------------- #
# Namespace validation (BaseStore.put / aput must enforce these)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_namespace",
    [
        (),
        ("the", "thing.about"),
        ("some", "fun", ""),
        ("langgraph", "foo"),
    ],
)
def test_put_rejects_invalid_namespace_sync(spec_store, bad_namespace):
    store, _ = spec_store
    with pytest.raises(InvalidNamespaceError):
        store.put(bad_namespace, "k", {"v": 1})


@pytest.mark.parametrize(
    "bad_namespace",
    [
        (),
        ("the", "thing.about"),
        ("some", "fun", ""),
        ("langgraph", "foo"),
    ],
)
def test_aput_rejects_invalid_namespace(spec_store, bad_namespace):
    store, _ = spec_store

    async def _go():
        await store.aput(bad_namespace, "k", {"v": 1})

    with pytest.raises(InvalidNamespaceError):
        _arun(_go())


def test_raw_batch_bypasses_namespace_validation(spec_store):
    """Per upstream contract, validation lives in `put`/`aput`, not `batch`.

    A caller who reaches into the lower-level `batch([PutOp(...)])` API is
    trusted to pass valid namespaces. Both backends must accept a
    "langgraph"-prefixed namespace through this path.
    """
    store, _ = spec_store
    ns = ("langgraph", "ok_via_batch")
    store.batch([PutOp(ns, "k", {"v": 1})])
    item = store.get(ns, "k")
    assert item is not None and item.value == {"v": 1}


# --------------------------------------------------------------------------- #
# search (no-query, filter-based) -- this is the only search mode Aerospike
# supports, so it's the only mode we exercise in the shared spec.
# --------------------------------------------------------------------------- #


def _populate_search_corpus(store) -> None:
    docs = [
        ("d1", {"color": "red", "score": 4.5}),
        ("d2", {"color": "red", "score": 3.0}),
        ("d3", {"color": "green", "score": 4.0}),
        ("d4", {"color": "blue", "score": 3.5}),
    ]
    for k, v in docs:
        store.put(("test",), k, v)


def test_search_no_filter_returns_all_in_prefix(spec_store):
    store, _ = spec_store
    _populate_search_corpus(store)
    results = store.batch([SearchOp(namespace_prefix=("test",))])[0]
    assert len(results) == 4
    assert {r.key for r in results} == {"d1", "d2", "d3", "d4"}


def test_search_with_exact_filter(spec_store):
    store, _ = spec_store
    _populate_search_corpus(store)
    results = store.batch([SearchOp(namespace_prefix=("test",), filter={"color": "red"})])[0]
    assert {r.key for r in results} == {"d1", "d2"}


@pytest.mark.parametrize(
    ("op", "value", "expected_keys"),
    [
        ("$gt", 3.5, {"d1", "d3"}),
        ("$gte", 4.0, {"d1", "d3"}),
        ("$lt", 4.0, {"d2", "d4"}),
        ("$lte", 3.5, {"d2", "d4"}),
        ("$ne", "red", {"d3", "d4"}),
    ],
)
def test_search_with_comparison_filter(spec_store, op, value, expected_keys):
    store, _ = spec_store
    _populate_search_corpus(store)
    field = "color" if op == "$ne" else "score"
    results = store.batch([SearchOp(namespace_prefix=("test",), filter={field: {op: value}})])[0]
    assert {r.key for r in results} == expected_keys


def test_search_with_multiple_filters(spec_store):
    store, _ = spec_store
    _populate_search_corpus(store)
    results = store.batch(
        [
            SearchOp(
                namespace_prefix=("test",),
                filter={"color": "red", "score": {"$gte": 4.0}},
            )
        ]
    )[0]
    assert {r.key for r in results} == {"d1"}


# --------------------------------------------------------------------------- #
# list_namespaces -- ported directly from upstream's
# `test_list_namespaces_*` suite, with sort()-tolerant assertions where
# upstream relies on InMemoryStore's stable insertion order.
# --------------------------------------------------------------------------- #


def _populate_namespaces(store, namespaces) -> None:
    for i, ns in enumerate(namespaces):
        store.put(namespace=ns, key=f"id_{i}", value={"data": f"v_{i:02d}"})


def test_list_namespaces_basic(spec_store):
    store, _ = spec_store
    namespaces = [
        ("a", "b", "c"),
        ("a", "b", "d", "e"),
        ("a", "b", "d", "i"),
        ("a", "b", "f"),
        ("a", "c", "f"),
        ("b", "a", "f"),
        ("users", "123"),
        ("users", "456", "settings"),
        ("admin", "users", "789"),
    ]
    _populate_namespaces(store, namespaces)

    assert sorted(store.list_namespaces(prefix=("a", "b"))) == sorted(
        [("a", "b", "c"), ("a", "b", "d", "e"), ("a", "b", "d", "i"), ("a", "b", "f")]
    )

    assert sorted(store.list_namespaces(suffix=("f",))) == sorted(
        [("a", "b", "f"), ("a", "c", "f"), ("b", "a", "f")]
    )

    assert sorted(store.list_namespaces(prefix=("a",), suffix=("f",))) == sorted(
        [("a", "b", "f"), ("a", "c", "f")]
    )

    assert sorted(store.list_namespaces(prefix=("a", "b"), max_depth=3)) == sorted(
        [("a", "b", "c"), ("a", "b", "d"), ("a", "b", "f")]
    )

    assert sorted(store.list_namespaces(prefix=("a", "*", "f"))) == sorted(
        [("a", "b", "f"), ("a", "c", "f")]
    )

    assert sorted(store.list_namespaces(suffix=("*", "f"))) == sorted(
        [("a", "b", "f"), ("a", "c", "f"), ("b", "a", "f")]
    )

    assert store.list_namespaces(prefix=("nonexistent",)) == []

    assert store.list_namespaces(prefix=("users", "123")) == [("users", "123")]


def test_list_namespaces_with_wildcards(spec_store):
    store, _ = spec_store
    namespaces = [
        ("users", "123"),
        ("users", "456"),
        ("users", "789", "settings"),
        ("admin", "users", "789"),
        ("guests", "123"),
        ("guests", "456", "preferences"),
    ]
    _populate_namespaces(store, namespaces)

    assert sorted(store.list_namespaces(prefix=("users", "*"))) == sorted(
        [("users", "123"), ("users", "456"), ("users", "789", "settings")]
    )

    assert store.list_namespaces(suffix=("*", "preferences")) == [
        ("guests", "456", "preferences"),
    ]

    assert store.list_namespaces(prefix=("*", "users"), suffix=("*", "settings")) == []


def test_list_namespaces_max_depth(spec_store):
    store, _ = spec_store
    namespaces = [
        ("a", "b", "c", "d"),
        ("a", "b", "c", "e"),
        ("a", "b", "f"),
        ("a", "g"),
        ("h", "i", "j", "k"),
    ]
    _populate_namespaces(store, namespaces)

    assert sorted(store.list_namespaces(max_depth=2)) == sorted(
        [("a", "b"), ("a", "g"), ("h", "i")]
    )


def test_list_namespaces_no_conditions(spec_store):
    store, _ = spec_store
    namespaces = [("a", "b"), ("c", "d"), ("e", "f", "g")]
    _populate_namespaces(store, namespaces)
    assert sorted(store.list_namespaces()) == sorted(namespaces)


def test_list_namespaces_empty_store(spec_store):
    store, _ = spec_store
    assert store.list_namespaces() == []


def test_list_namespaces_pagination(spec_store):
    """Pagination must be deterministic and cover the full set with no
    overlap or gaps. We don't pin the ordering scheme (alphabetical vs
    insertion) — only that the union of pages equals the full listing and
    pages don't duplicate entries."""
    store, _ = spec_store
    for i in range(20):
        store.put(namespace=("ns", f"sub_{i:02d}"), key=f"id_{i:02d}", value={"i": i})

    full = store.list_namespaces(prefix=("ns",), limit=100)
    assert len(full) == 20

    pages = [
        store.list_namespaces(prefix=("ns",), limit=5, offset=offset) for offset in (0, 5, 10, 15)
    ]
    for page in pages:
        assert len(page) == 5

    flattened = [ns for page in pages for ns in page]
    assert len(flattened) == 20
    assert set(flattened) == set(full), "pagination dropped or duplicated entries"


# --------------------------------------------------------------------------- #
# Aerospike-only contract: ``query=`` must be a vector search payload.
# --------------------------------------------------------------------------- #


def test_aerospike_rejects_text_semantic_search(session, namespace, truncate_sets):
    """``AerospikeStore`` vector search takes a list of floats (or an
    ``{"embedding": [...], "metric": ...}`` dict). A raw text query is not a
    supported form and must be rejected explicitly rather than silently
    ignored."""
    truncate_sets((_AEROSPIKE_SET,))
    store = AerospikeStore(session=session, namespace=namespace, set=_AEROSPIKE_SET)
    with pytest.raises(ValueError, match="Unsupported query type"):
        store.search(("anything",), query="hello world")
