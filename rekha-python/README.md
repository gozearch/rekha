# Rekha Python SDK

Pure Python gRPC client for Rekha, the distributed vector database.

## Installation

```bash
pip install grpcio protobuf
pip install .
```

## Usage

```python
from rekha import RekhaClient

# Connect to cluster
client = RekhaClient.connect(["localhost:50051"])

# Insert a vector (id=0 means auto-generate)
client.insert([0.1, 0.2, 0.3], "default", payload=b'{"hello": "world"}')

# Insert with explicit ID
client.insert([0.1, 0.2, 0.3], "default", id=42, payload=b'{"hello": "world"}')

# Search
results = client.search([0.1, 0.2, 0.3], top_k=10, collection_name="default")
for r in results:
    print(f"id={r.id}, score={r.score}")

# Delete vectors
deleted = client.delete([42, 43], "default")
print(f"Deleted {deleted} vectors")

# Fetch vectors
points = client.fetch([42], "default", include_payloads=True)

# Cluster info
topology = client.cluster_info()
print(f"Connected to {topology.cluster_id}")
for peer in topology.peers:
    print(f"  {peer.node_id} at {peer.address}")

client.close()
```

### API

| Method | Description |
|---|---|
| `connect(seeds)` | Connect to any seed node |
| `insert(vector, collection_name, id=0, payload=None)` | Insert a vector (id=0 = auto-generate) |
| `search(query, top_k, collection_name)` | Search top-k NN |
| `search_with_params(query, top_k, collection_name, params, local_only=False)` | Search with ef_search, beam_width, etc. |
| `search_stream(query, top_k, collection_name, params=None, local_only=False)` | Streaming search (generator) |
| `delete(ids, collection_name)` | Delete by ID list |
| `fetch(ids, collection_name, include_payloads=False)` | Fetch vectors by ID |
| `cluster_info()` | Get cluster topology |
| `close()` | Close gRPC channel |
