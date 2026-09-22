import contextlib

import pytest

# Skip all tests if dev SDK not available
try:
    from aerospike_sdk import DataSet, Vector  # noqa: F401

    HAS_VECTOR_SDK = True
except ImportError:
    HAS_VECTOR_SDK = False


@pytest.fixture(scope="module")
def vector_supported(session):
    """Skip the vector suite if the connected server does not support VECTOR bins."""
    pytest.importorskip("aerospike_sdk", reason="Vector SDK not available")
    probe_set = "store_test_vec_probe"
    ds = DataSet.of("test", probe_set)
    try:
        session.upsert(ds.id("__probe__")).bin("v").set_to_vector(
            Vector([1.0])
        ).execute()
    except Exception as e:
        if "VECTOR" in str(e) or "vector" in str(e).lower():
            pytest.skip(f"Aerospike server does not support VECTOR bins: {e}")
        raise
    finally:
        with contextlib.suppress(Exception):
            session.delete(ds.id("__probe__")).execute()
        with contextlib.suppress(Exception):
            session.truncate(ds, before_nanos=0)


pytestmark = pytest.mark.usefixtures("vector_supported")


class TestVectorStorage:
    """Test vector embedding storage and retrieval."""

    def test_store_and_retrieve_embedding(self, store):
        """Store item with embedding and retrieve it."""
        # Create item with embedding
        embedding = [0.1, 0.2, 0.3, 0.4]
        value = {
            "text": "hello world",
            "__embedding__": embedding,
        }

        store.put(namespace=("docs",), key="doc1", value=value)

        # Retrieve and verify
        item = store.get(namespace=("docs",), key="doc1")
        assert item is not None
        assert item.value["text"] == "hello world"
        # __embedding__ should be removed from stored value
        assert "__embedding__" not in item.value

    def test_store_multiple_embeddings(self, store):
        """Store multiple items with different embeddings."""
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.9, 0.8, 0.7],
            [0.5, 0.5, 0.5],
        ]

        for i, emb in enumerate(embeddings):
            store.put(
                namespace=("docs",),
                key=f"doc{i}",
                value={"text": f"doc {i}", "__embedding__": emb},
            )

        # Verify all stored
        for i in range(3):
            item = store.get(namespace=("docs",), key=f"doc{i}")
            assert item is not None


class TestVectorSearch:
    """Test vector similarity search (cosine, euclidean, dotproduct)."""

    def test_cosine_similarity_search(self, store):
        """Search by cosine similarity and verify Top-K ranking."""
        # Store vectors: [1, 0], [0, 1], [0.7, 0.7] (normalized)
        vectors = [
            ([1.0, 0.0], "doc_right"),
            ([0.0, 1.0], "doc_up"),
            ([0.7071, 0.7071], "doc_diagonal"),  # ~45 degrees
        ]

        for emb, key in vectors:
            store.put(
                namespace=("test",),
                key=key,
                value={"text": key, "__embedding__": emb},
            )

        # Query with [1, 0] — should rank doc_right first
        query_embedding = [1.0, 0.0]
        results = store.search(
           ("test",),
            query=query_embedding,
            limit=10,
        )

        assert len(results) == 3
        assert results[0].key == "doc_right"  # Cosine sim = 1.0
        assert abs(results[0].score - 1.0) < 0.01
        assert results[1].key == "doc_diagonal"  # Cosine sim ≈ 0.707
        assert results[2].key == "doc_up"  # Cosine sim = 0.0

    def test_euclidean_distance_search(self, store):
        """Search by euclidean distance and verify Top-K ranking."""
        # Store points: (0, 0), (1, 1), (3, 0)
        vectors = [
            ([0.0, 0.0], "origin"),
            ([1.0, 1.0], "diagonal"),
            ([3.0, 0.0], "far"),
        ]

        for emb, key in vectors:
            store.put(
                namespace=("search",),
                key=key,
                value={"text": key, "__embedding__": emb},
            )

        # Query from (1, 0) — euclidean squared distance
        # To origin: 1^2 + 0^2 = 1
        # To diagonal: (1-1)^2 + (0-1)^2 = 1
        # To far: (1-3)^2 + 0^2 = 4
        query_embedding = [1.0, 0.0]
        results = store.search(
           ("search",),
            query={
                "embedding": query_embedding,
                "metric": "euclidean",
            },
            limit=10,
        )

        assert len(results) == 3
        assert results[0].key in ("origin", "diagonal")  # Distance 1
        assert results[2].key == "far"  # Distance 4

    def test_dotproduct_search(self, store):
        """Search by dot product."""
        vectors = [
            ([1.0, 0.0], "x_axis"),
            ([0.0, 1.0], "y_axis"),
            ([2.0, 2.0], "both"),
        ]

        for emb, key in vectors:
            store.put(
                namespace=("dp",),
                key=key,
                value={"text": key, "__embedding__": emb},
            )

        # Dot product with [2, 1]
        # [1, 0] · [2, 1] = 2
        # [0, 1] · [2, 1] = 1
        # [2, 2] · [2, 1] = 6
        query_embedding = [2.0, 1.0]
        results = store.search(
           ("dp",),
            query={"embedding": query_embedding, "metric": "dotproduct"},
            limit=10,
        )

        assert len(results) == 3
        assert results[0].key == "both"  # Dot product = 6
        assert results[1].key == "x_axis"  # Dot product = 2
        assert results[2].key == "y_axis"  # Dot product = 1


class TestVectorSearchFilters:
    """Test vector search combined with scalar filters and namespace prefix."""

    def test_vector_search_with_namespace_filter(self, store):
        """Vector search respects namespace prefix."""
        # Store in different namespaces
        store.put(
            namespace=("docs", "ai"),
            key="ai1",
            value={"text": "AI paper", "__embedding__": [1.0, 0.0]},
        )
        store.put(
            namespace=("docs", "biology"),
            key="bio1",
            value={"text": "Bio paper", "__embedding__": [0.9, 0.1]},
        )

        # Search only in docs/ai
        results = store.search(
           ("docs", "ai"),
            query=[1.0, 0.0],
            limit=10,
        )

        assert len(results) == 1
        assert results[0].key == "ai1"

    def test_vector_search_with_scalar_filter(self, store):
        """Vector search respects scalar filters."""
        # Store items with status field
        store.put(
            namespace=("items",),
            key="active1",
            value={"status": "active", "text": "a1", "__embedding__": [1.0, 0.0]},
        )
        store.put(
            namespace=("items",),
            key="inactive1",
            value={"status": "inactive", "text": "i1", "__embedding__": [0.9, 0.1]},
        )

        # Search only active items
        results = store.search(
           ("items",),
            query=[1.0, 0.0],
            filter={"status": "active"},
            limit=10,
        )

        assert len(results) == 1
        assert results[0].key == "active1"


class TestVectorSearchPagination:
    """Test offset and limit with vector search."""

    def test_offset_and_limit(self, store):
        """Offset and limit apply after ranking."""
        # Store 5 items
        for i in range(5):
            store.put(
                namespace=("pages",),
                key=f"item{i}",
                value={"text": f"item {i}", "__embedding__": [float(i), 0.0]},
            )

        # All embeddings are collinear, so cosine scores would tie at 1.0.
        # Use euclidean so the ranking is deterministic:
        # distances (4-i)^2 -> item4, item3, item2, item1, item0.
        # Get page 2 (items 2-3)
        results = store.search(
           ("pages",),
            query={"embedding": [4.0, 0.0], "metric": "euclidean"},
            limit=2,
            offset=2,
        )

        assert len(results) == 2
        # Should be items 2 and 1 (after skipping 4 and 3)
        keys = {r.key for r in results}
        assert keys == {"item2", "item1"}


class TestVectorSearchErrors:
    """Test error handling for invalid vector operations."""

    def test_dimension_mismatch(self, store):
        """Dimension mismatch should skip record or raise error."""
        # Store 3D embedding
        store.put(
            namespace=("dims",),
            key="vec3d",
            value={"text": "3d", "__embedding__": [1.0, 0.0, 0.0]},
        )

        # Query with 2D embedding — should skip the 3d record
        results = store.search(
           ("dims",),
            query=[1.0, 0.0],  # 2D
            limit=10,
        )

        assert len(results) == 0  # No compatible records

    def test_missing_embedding_bin(self, store):
        """Record with no embedding should be skipped."""
        # Store without embedding
        store.put(
            namespace=("mixed",),
            key="no_embed",
            value={"text": "no embedding"},
        )
        # Store with embedding
        store.put(
            namespace=("mixed",),
            key="with_embed",
            value={"text": "has embedding", "__embedding__": [1.0, 0.0]},
        )

        # Search should only return the one with embedding
        results = store.search(
           ("mixed",),
            query=[1.0, 0.0],
            limit=10,
        )

        assert len(results) == 1
        assert results[0].key == "with_embed"

    def test_invalid_metric(self, store):
        """Invalid metric name should raise ValueError."""
        store.put(
            namespace=("test",),
            key="test",
            value={"text": "test", "__embedding__": [1.0, 0.0]},
        )

        with pytest.raises(ValueError, match="Unsupported.*metric"):
            store.search(
               ("test",),
                query={"embedding": [1.0, 0.0], "metric": "invalid"},
            )


class TestVectorSearchBackwardCompatibility:
    """Ensure scalar searches still work."""

    def test_scalar_search_unaffected(self, store):
        """Scalar filter searches should work as before."""
        store.put(
            namespace=("items",),
            key="a",
            value={"status": "active", "text": "item a"},
        )
        store.put(
            namespace=("items",),
            key="b",
            value={"status": "inactive", "text": "item b"},
        )

        # Scalar search (no query parameter)
        results = store.search(
           ("items",),
            filter={"status": "active"},
            limit=10,
        )

        assert len(results) == 1
        assert results[0].key == "a"
