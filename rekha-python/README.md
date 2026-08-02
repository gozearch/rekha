# Rekha Python SDK

Pure Python gRPC client for Rekha, the distributed vector database.

The API follows [ChromaDB](https://docs.trychroma.com/) conventions: client → collection → add/query/get/delete.

## Installation

```bash
pip install grpcio protobuf
pip install .
```

## Usage

```python
from rekha import RekhaClient

# Connect to cluster
client = RekhaClient(seeds=["localhost:50051"])

# Create a collection
coll = client.create_collection("images", dim=256, nlist=4096, nprobe=32)

# Or get an existing one
coll = client.get_collection("images")

# Add vectors with metadata
coll.add(
    ids=[1, 2, 3],
    embeddings=[
        [0.12, 0.34, 0.56, 0.78, 0.91, 0.23, 0.45, 0.67],
        [0.98, 0.76, 0.54, 0.32, 0.10, 0.87, 0.65, 0.43],
        [0.21, 0.43, 0.65, 0.87, 0.09, 0.81, 0.63, 0.45],
    ],
    metadatas=[
        {"tag": "cat", "source": "flickr"},
        {"tag": "dog", "source": "instagram"},
        {"tag": "bird", "source": "flickr"},
    ],
)

# Add with explicit consistency level
from rekha.types import ConsistencyLevel
coll.add(ids=[4], embeddings=[[0.33] * 256], consistency=ConsistencyLevel.ALL)

# Search
results = coll.query(query_embeddings=[[0.15, 0.35, 0.55, 0.75, 0.90, 0.20, 0.40, 0.60]], n_results=5)
for id_, dist, md in zip(results["ids"][0], results["distances"][0], results["metadatas"][0]):
    print(f"id={id_}, distance={dist:.4f}, metadata={md}")

# Retrieve by ID
result = coll.get(ids=[1, 2], include=["embeddings", "metadatas"])
for id_, md in zip(result["ids"], result["metadatas"]):
    print(f"id={id_}, metadata={md}")

# Peek at first N records
result = coll.peek(limit=5)

# Count vectors
count = coll.count()

# Delete by IDs
coll.delete(ids=[3])

# Rekha-specific: dimension range search
results = coll.search_dim_range(
    query_embeddings=[[0.1] * 8],
    n_results=10,
    dim_start=0,
    dim_end=4,
)

# Manage collections
client.list_collections()
client.count_collections()
client.delete_collection("images")

# Health check
client.heartbeat()

client.close()
```

### API Reference

| Method | Description |
|--------|-------------|
| `RekhaClient(seeds)` | Connect to a cluster |
| `client.create_collection(name, **config)` | Create a collection |
| `client.get_collection(name)` | Get an existing collection |
| `client.get_or_create_collection(name, **config)` | Get or create |
| `client.delete_collection(name)` | Drop a collection |
| `client.list_collections(limit, offset)` | List all collections |
| `client.count_collections()` | Count collections |
| `client.heartbeat()` | Ping the cluster |

| `Collection` Method | Description | Backend RPC |
|---------------------|-------------|-------------|
| `coll.add(ids, embeddings, metadatas, documents, consistency)` | Add vectors with metadata | `InsertBatch` |
| `coll.get(ids, limit, offset, include, consistency)` | Retrieve vectors | `Fetch` (by ids) / `Search` (scan) |
| `coll.query(query_embeddings, n_results, include, consistency)` | ANN search | `Search` |
| `coll.peek(limit)` | First N records | `Search` (scan) |
| `coll.count()` | Vector count | `ListCollections` |
| `coll.delete(ids, consistency)` | Delete by IDs | `Delete` |
| `coll.search_dim_range(query_embeddings, n_results, dim_start, dim_end)` | Dim-range search | `SearchDimRange` |
