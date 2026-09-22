# Vector Similarity Search in AerospikeStore

This guide explains how to use vector embeddings with `langgraph-store-aerospike` for semantic search.

## Prerequisites

> **Pre-release feature.** Vector support (the `Vector` bin type,
> `Exp.vector_bin`/distance expressions, and Top-K query pushdown) is real
> but, as of this writing, only exists on the **unmerged `aie/vector`
> branch** of the Aerospike Python SDK — it is not in a published PyPI
> release and not in a GA Aerospike Server image. Vector search will not
> work against a normal `pip install aerospike-sdk` + a released server;
> follow the dev-build steps below, and expect both to move once vector
> support ships in a real release.

1. **Aerospike Server 8.1.3+** with vector support — a locally built dev
   image; there is no public/GA server image with this feature yet:
   ```bash
   docker run -d --name aerospike \
     -p 3000:3000 \
     aerospike/aerospike-server:latest-dev-vector
   ```

2. **Aerospike Python SDK, `aie/vector` branch** (not the published
   `aerospike-sdk` package — that release does not include `Vector`,
   `Exp.vector_bin`, or Top-K support):
   ```bash
   git clone https://github.com/aerospike/aerospike-client-python-sdk.git
   cd aerospike-client-python-sdk
   git checkout aie/vector
   pip install -e .
   ```
   Verify you actually have the branch build, not the released package:
   ```python
   from aerospike_sdk import Vector  # raises ImportError on the released package
   ```

## Basic Usage

### Store Embeddings

Embeddings are stored via the special `__embedding__` key in the value dict:

```python
from aerospike_sdk import Behavior
from aerospike_sdk.sync import ClusterDefinition
from langgraph.store.aerospike import AerospikeStore

cluster = ClusterDefinition("127.0.0.1", 3000).connect()
session = cluster.create_session(Behavior.DEFAULT)
store = AerospikeStore(session=session, namespace="test", set="store")

# Create embedding (e.g., from a model)
embedding = [0.1, 0.2, 0.3, 0.4]

# Store with embedding
store.put(
    namespace=("docs", "ai"),
    key="paper_001",
    value={
        "title": "Vectors in Aerospike",
        "author": "Jane Doe",
        "__embedding__": embedding,
    }
)

# The __embedding__ key is removed from stored value (not duplicated)
item = store.get(namespace=("docs", "ai"), key="paper_001")
print(item.value)  # {"title": "...", "author": "..."}
```

### Search by Similarity

Pass the query embedding as the `query` parameter:

```python
# Simple cosine similarity search (default)
query_embedding = [0.1, 0.2, 0.3, 0.4]
results = store.search(
    namespace_prefix=("docs", "ai"),
    query=query_embedding,
    limit=5,
)

# Results are ranked by distance (highest score first for cosine)
for item in results:
    print(f"{item.key}: similarity={item.score:.4f}")
```

## Distance Metrics

Three metrics are supported:

| Metric | Use Case | Range | Sort Order |
|---|---|---|---|
| **cosine** (default) | General embeddings, semantic search | [-1, 1] | Descending (higher = more similar) |
| **euclidean** | Geometric distance, magnitude matters | [0, ∞) | Ascending (lower = more similar) |
| **dotproduct** | Pre-normalized vectors, efficiency | Varies | Descending |

```python
# Euclidean distance
results = store.search(
    namespace_prefix=("docs",),
    query={"embedding": query_embedding, "metric": "euclidean"},
    limit=5,
)

# Dot product
results = store.search(
    namespace_prefix=("docs",),
    query={"embedding": query_embedding, "metric": "dotproduct"},
    limit=5,
)
```

## Advanced Usage

### Combine with Scalar Filters

Vector search works with namespace prefixes and scalar filters:

```python
# Find similar documents that are marked as "published"
results = store.search(
    namespace_prefix=("docs", "published"),
    query=query_embedding,
    filter={"status": "approved"},  # Also filter by scalar field
    limit=10,
)
```

### Pagination

Offset and limit are applied after ranking (not before):

```python
# Get results 10-20 by similarity
results = store.search(
    namespace_prefix=("docs",),
    query=query_embedding,
    limit=10,
    offset=10,
)
```

### Async Support

Async search is inherited from `BaseStore`:

```python
# Async search
results = await store.asearch(
    namespace_prefix=("docs",),
    query=query_embedding,
    limit=5,
)
```

## Performance Considerations

### Brute-Force Top-K

Vector search ranks by scanning every record the query matches (there is no ANN/HNSW index) — but ranking itself runs server-side via the SDK's `order_by(...).top_k(...)` Top-K pushdown (`ORDER BY <distance> LIMIT k`), not by fetching every candidate record and sorting client-side. Budget query cost like a full scan over the queried set, not an index lookup. This is acceptable for:

- **Dataset size:** < 1 million items
- **Query latency:** < 1 second typical
- **Accuracy:** Exact Top-K (no approximation)

For larger datasets, ANN indexes (HNSW) may be added in Phase 2.

### Optimization Tips

1. **Use namespace prefixes** to reduce search scope
2. **Combine with scalar filters** to further narrow results
3. **Batch inserts** when storing many embeddings
4. **Monitor query time** with large datasets (watch Aerospike metrics)

## Error Handling

### Dimension Mismatch

Query and stored embeddings must have the same dimension:

```python
store.put(namespace=("docs",), key="doc1", value={"__embedding__": [1.0, 2.0]})

# Query with different dimension — record will be skipped
results = store.search(
    namespace_prefix=("docs",),
    query=[1.0, 2.0, 3.0],  # 3D vs 2D stored
    limit=5,
)
# results will not include doc1
```

### Missing Embedding

Records without an `embedding` bin are skipped:

```python
store.put(namespace=("docs",), key="no_embed", value={"text": "no embedding"})
store.put(namespace=("docs",), key="with_embed", value={"text": "has one", "__embedding__": [1.0]})

results = store.search(
    namespace_prefix=("docs",),
    query=[1.0],
    limit=10,
)
# Only with_embed is returned
```

### Invalid Metric

```python
# Raises ValueError
store.search(namespace_prefix=("docs",), query={"embedding": [1.0], "metric": "invalid"})
```

## Limitations & Roadmap

### Phase 1 (Current)

✓ Vector storage and retrieval  
✓ Cosine, euclidean, dot product similarity  
✓ Brute-force Top-K ranking  
✓ Namespace + filter + vector search  
✓ Async support  

### Phase 2 (Future)

- [ ] HNSW/ANN indexes for large datasets
- [ ] Additional vector element types (INT32, FLOAT64)
- [ ] Vectors in nested structures (inside value map)
- [ ] Client library migration (optional)

## Troubleshooting

### "Vector SDK not available" error

This means the *published* `aerospike-sdk` package is installed instead of the `aie/vector` branch build. Install the branch (see [Prerequisites](#prerequisites)):

```bash
git clone https://github.com/aerospike/aerospike-client-python-sdk.git
cd aerospike-client-python-sdk && git checkout aie/vector
pip install -e .
```

Verify:
```python
from aerospike_sdk import Vector
print("Vector SDK ready")
```

### Server version mismatch

Ensure Aerospike Server 8.1.3+ is running:

```bash
# Check server version
docker inspect <aerospike-container> | grep -i image
```

### No results from vector search

1. Check that items were stored with `__embedding__` key
2. Verify query embedding has same dimension as stored embeddings
3. Try scalar search to ensure basic store works
4. Check Aerospike logs for errors

## Examples

### RAG Retrieval

```python
# Embed documents at index time
documents = load_documents()
for doc in documents:
    embedding = embed_model.embed(doc["text"])
    store.put(
        namespace=("rag", "documents"),
        key=doc["id"],
        value={
            "source": doc["source"],
            "text": doc["text"],
            "__embedding__": embedding,
        }
    )

# At query time, find similar documents
query_text = "How does Aerospike handle vectors?"
query_embedding = embed_model.embed(query_text)
results = store.search(
    namespace_prefix=("rag", "documents"),
    query=query_embedding,
    limit=5,
)

for item in results:
    print(f"Match: {item.value['source']} (score: {item.score:.4f})")
    print(f"Text: {item.value['text'][:100]}...")
```

### Semantic Caching

```python
# Store query-response pairs with embeddings
query_embedding = embed_model.embed(user_query)

# Check cache for similar queries
cached = store.search(
    namespace_prefix=("cache", "queries"),
    query=query_embedding,
    limit=1,
)

if cached and cached[0].score > 0.95:  # High similarity threshold
    # Use cached response
    response = cached[0].value["response"]
else:
    # Generate new response
    response = generate_response(user_query)
    
    # Cache it
    store.put(
        namespace=("cache", "queries"),
        key=generate_id(),
        value={
            "query": user_query,
            "response": response,
            "__embedding__": query_embedding,
        },
    )
```

## References

- [Aerospike Vector Documentation](https://docs.aerospike.com/server/guide/data-types/vectors)
- [langgraph Store API](https://langchain-ai.github.io/langgraph/concepts/store/)
- [Aerospike Python SDK](https://github.com/aerospike/aerospike-client-python-sdk)
