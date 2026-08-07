<div align="center">

![RekhaDB Logo](assets/logo.jpg)

# RekhaDB

**A scalable, resource-efficient distributed vector database in Rust**

[![CI](https://github.com/gozearch/rekha-db/actions/workflows/ci.yml/badge.svg)](https://github.com/gozearch/rekha-db/actions/workflows/ci.yml)
[![Docker](https://github.com/gozearch/rekha-db/actions/workflows/docker.yml/badge.svg)](https://github.com/gozearch/rekha-db/actions/workflows/docker.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

</div>

---

## What is RekhaDB?

RekhaDB is a **Chroma-compatible vector database** built in Rust for performance, efficiency, and distributed deployments. It stores, indexes, and queries vector embeddings with metadata — perfect for AI/ML applications, semantic search, recommendation systems, and RAG pipelines.

### Key Features

| Feature | Description |
|---------|-------------|
| **ChromaDB Compatible** | Works with the official [Python client SDK](https://pypi.org/project/chromadb/) — drop-in replacement for ChromaDB |
| **HNSW Index** | Approximate nearest neighbor search via [usearch](https://github.com/unum-cloud/usearch) with Chroma-exact distance semantics |
| **Distributed Cluster** | Multi-node replication via [openraft](https://github.com/datafuselabs/openraft) consensus — automatic leader election and failover |
| **WAL + Checkpointing** | Write-ahead log for durability, periodic snapshots for fast recovery |
| **mmap Segments** | Vector data stored in memory-mapped segment files for zero-copy reads |
| **RAM Efficient** | Embeddings stripped from memory after checkpoint, loaded on-demand from mmap |
| **REST API** | Chroma-compatible HTTP API at `/tenants/{t}/databases/{d}/collections/...` |
| **Docker Ready** | Multi-stage Dockerfile, docker-compose for single node and 3-node clusters |

## Quick Start

### Docker (Recommended)

```bash
# Single node
docker compose up -d

# 3-node cluster
docker compose -f docker-compose.cluster.yml up -d
```

### Python Client

```python
import chromadb

# Connect to RekhaDB
client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    headers={"x-chroma-token": "your-api-key"}
)

# Create collection (dimension inferred from first add)
collection = client.get_or_create_collection("my-vectors")

# Add documents
collection.add(
    ids=["doc1", "doc2", "doc3"],
    embeddings=[
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ],
    metadatas=[
        {"source": "web", "category": "tech"},
        {"source": "pdf", "category": "science"},
        {"source": "db", "category": "tech"}
    ],
    documents=[
        "Rust is a systems programming language",
        "Quantum computing uses qubits",
        "Vector databases enable semantic search"
    ]
)

# Query
results = collection.query(
    query_embeddings=[[0.1, 0.2, 0.3, 0.4]],
    n_results=2,
    include=["documents", "metadatas", "distances"]
)
print(results)
# {'ids': [['doc1', 'doc3']], 'documents': [['Rust is...', 'Vector databases...']], ...}

# Count
print(collection.count())  # 3

# Delete
collection.delete(ids=["doc1"])
print(collection.count())  # 2
```

### CLI

```bash
# Start server
rekha serve --port 8000

# Collection operations
rekha collection create my-vectors --dimension 128
rekha collection list

# Data operations (from JSON files)
rekha add <collection-id> add_payload.json
rekha query <collection-id> query_payload.json
rekha count <collection-id>
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client (Python/CLI)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────┐
│                    RekhaDB API Server                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Chroma API  │  │  Raft RPC   │  │  WAL Delta Shipping │ │
│  │  (axum)     │  │  (openraft) │  │  (HTTP polling)     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────────┼─────────────┘
          │                │                    │
┌─────────▼────────────────▼────────────────────▼─────────────┐
│                      Engine Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Collection  │  │  HNSW Index │  │  WAL + Checkpoints  │ │
│  │  (Records)   │  │  (usearch)  │  │  (redb + mmap)      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

### Chroma-Compatible (Python Client)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/heartbeat` | Health check |
| GET | `/version` | Server version |
| GET | `/pre-flight-checks` | Pre-flight validation |
| POST | `/tenants/{t}/databases/{d}/collections` | Create/get collection |
| GET | `/tenants/{t}/databases/{d}/collections` | List collections |
| GET | `/tenants/{t}/databases/{d}/collections/{name}` | Get collection |
| DELETE | `/tenants/{t}/databases/{d}/collections/{name}` | Delete collection |
| POST | `.../collections/{name}/add` | Add records |
| POST | `.../collections/{name}/upsert` | Upsert records |
| POST | `.../collections/{name}/update` | Update metadata |
| POST | `.../collections/{name}/delete` | Delete records |
| POST | `.../collections/{name}/query` | kNN search |
| POST | `.../collections/{name}/get` | Get by ID |
| GET | `.../collections/{name}/count` | Count records |

### Internal (Cluster)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/internal/raft/membership` | Cluster membership |
| POST | `/internal/raft/add_learner` | Add node as learner |
| POST | `/internal/raft/remove_member` | Remove node |
| GET | `/internal/wal/{id}/delta` | WAL delta for replication |
| GET | `/internal/wal/{id}/status` | WAL status |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (always 200) |
| GET | `/ready` | Readiness check |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REKHA_DATA_DIR` | `rekha-data` | Data directory path |
| `REKHA_HOST` | `0.0.0.0` | Listen host |
| `REKHA_PORT` | `8000` | Listen port |
| `REKHA_API_KEY` | — | API key for authentication |
| `REKHA_TLS_CERT` | — | TLS certificate path |
| `REKHA_TLS_KEY` | — | TLS private key path |
| `REKHA_METRICS_PORT` | — | Prometheus metrics port |

### CLI Flags

```bash
rekha serve \
  --port 8000 \
  --data-dir /data \
  --node-id 1 \
  --peers "node2:8000,node3:8000" \
  --tls-cert cert.pem \
  --tls-key key.pem
```

## Docker

### Single Node

```bash
docker compose up -d
```

### 3-Node Cluster

```bash
docker compose -f docker-compose.cluster.yml up -d
```

### Build Image

```bash
docker build -t rekha .
```

## Development

```bash
# Run tests
cargo test --workspace

# Build release
cargo build --release

# Format
cargo fmt --all

# Lint
cargo clippy --workspace --all-targets
```

## Project Structure

```
rekha-db/
├── crates/
│   ├── rekha-core/       # Core types, traits, config
│   ├── rekha-distance/   # Vector distance functions (SIMD)
│   ├── rekha-storage/    # Catalog (redb) + Storage (local/S3)
│   ├── rekha-wal/        # Write-ahead log (RKW1 format)
│   ├── rekha-index/      # HNSW index (usearch)
│   ├── rekha-engine/     # Engine: WAL→buffer→HNSW pipeline
│   ├── rekha-cluster/    # Cluster types, Raft log, state machine
│   ├── rekha-api/        # REST API (axum) + ChromaDB compat
│   ├── rekha-cli/        # CLI binary (rekha)
│   ├── rekha-bench/      # Benchmarks
│   ├── rekha-repl/       # REPL (placeholder)
│   └── rekha-objectstore/# Object store (placeholder)
├── Dockerfile
├── docker-compose.yml
├── docker-compose.cluster.yml
└── README.md
```

## Distance Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| L2 | Squared Euclidean distance | `sum((a-b)^2)` |
| IP | Inner product (similarity) | `1 - dot(a, b)` |
| Cosine | Cosine similarity | `L2-normalize(a,b) + dot(a,b)` |

All metrics: **lower = more similar**.

## License

Apache-2.0

## Acknowledgments

- [ChromaDB](https://www.trychroma.com/) — API compatibility target
- [usearch](https://github.com/unum-cloud/usearch) — HNSW index engine
- [openraft](https://github.com/datafuselabs/openraft) — Raft consensus library
- [redb](https://github.com/cberner/redb) — Embedded database for catalog
- [axum](https://github.com/tokio-rs/axum) — HTTP framework
