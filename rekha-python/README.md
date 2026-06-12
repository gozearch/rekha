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

# Insert a vector
client.insert(42, [0.1, 0.2, 0.3], payload=b'{"hello": "world"}')

# Search
results = client.search([0.1, 0.2, 0.3], top_k=10)
for r in results:
    print(f"id={r.id}, score={r.score}")

# Delete vectors
deleted = client.delete([42, 43])
print(f"Deleted {deleted} vectors")

# Fetch vectors
points = client.fetch([42], include_payloads=True)

# Cluster info
topology = client.cluster_info()
print(f"Connected to {topology.cluster_id}")
for peer in topology.peers:
    print(f"  {peer.node_id} at {peer.address}")

client.close()
```
