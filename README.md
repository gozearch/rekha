<div align="center">

![RekhaDB Logo](assets/logo.jpg)

# RekhaDB

**A scalable, resource-efficient distributed vector database in Rust**

[![CI](https://github.com/gozearch/rekha/actions/workflows/ci.yml/badge.svg)](https://github.com/gozearch/rekha-db/actions/workflows/ci.yml)
[![Docker](https://github.com/gozearch/rekha/actions/workflows/docker.yml/badge.svg)](https://github.com/gozearch/rekha-db/actions/workflows/docker.yml)
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
| **Sharded Cluster** | Each collection is assigned to a shard by the metadata leader for node load, and each shard's R replicas are chosen by rendezvous hashing, with one [openraft](https://github.com/datafuselabs/openraft) group per shard — data is stored once per replica, not once per node |
| **WAL + Checkpointing** | Write-ahead log for durability, periodic snapshots for fast recovery |
| **mmap Segments** | Vector data stored in memory-mapped segment files for zero-copy reads |
| **RAM Efficient** | Embeddings stripped from memory after checkpoint, loaded on-demand from mmap |
| **REST API** | Chroma-compatible HTTP API at `/tenants/{t}/databases/{d}/collections/...` |
| **Docker Ready** | Multi-stage `Containerfile`, compose files for single node and a working 3-node cluster |

## Quick Start

### Docker (Recommended)

```bash
# Single node
export REKHA_API_KEY="$(openssl rand -hex 16)"
docker compose up -d --build

# 3-node cluster (public API on ports 8001–8003)
export REKHA_API_KEY="$(openssl rand -hex 16)"
export REKHA_CLUSTER_SECRET="$(openssl rand -hex 32)"
docker compose -f docker-compose.cluster.yml up -d --build
```

### Python Client

```python
import chromadb

# Connect to RekhaDB
# For the 3-node compose file, use port 8001, 8002, or 8003.
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

# Operations: check, back up, restore
rekha verify --data-dir ./rekha-data            # read-only; exits non-zero on damage
rekha verify --data-dir ./rekha-data --json     # machine-readable, same exit codes
rekha backup --data-dir ./rekha-data --out ./backup-2026-11-01
rekha restore --data-dir ./restored --from ./backup-2026-11-01
```

`verify` reports every problem it finds, not the first, and distinguishes a
recoverable torn WAL tail (warning, exit 0) from a hole or a checksum mismatch
(error, exit 1). `restore` stages the copy, checks it against the backup's
manifest, and only then moves it into place — a refused restore leaves the
target untouched.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client (Python/CLI)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────┐
│                    RekhaDB API Server                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Chroma API  │  │  Raft RPC   │  │ Request Forwarding  │ │
│  │  (axum)     │  │  (openraft) │  │  (to the leader)    │ │
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
| GET | `/internal/cluster/membership` | Cluster membership and per-shard leaders |
| GET | `/internal/cluster/placement` | Shard placement and owner sets |
| POST | `/internal/cluster/add_learner` | Add a node as a learner (catch-up; promotion is not automated yet) |
| POST | `/internal/cluster/remove_member` | Remove a node (refused while shards are placed) |
| POST | `/internal/raft/{group}/append_entries` | Raft replication; group `0` is metadata, `1..` are shards |
| POST | `/internal/raft/{group}/vote` | Raft vote |
| POST | `/internal/raft/{group}/install_snapshot` | Raft snapshot install |
| POST | `/internal/forward/{*rest}` | A write forwarded from a non-leader node |
| POST | `/internal/stats/counts` | Record counts held by this node |

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
| `REKHA_HOST` | `0.0.0.0` | Public listen host |
| `REKHA_PORT` | `8000` | Public listen port |
| `REKHA_INTERNAL_HOST` | `REKHA_HOST` | Internal (Raft/forwarding) listen host |
| `REKHA_INTERNAL_PORT` | — | Internal listen port; a separate listener when set |
| `REKHA_NODE_ID` | `1` | Node id; persisted in `<data-dir>/node.id` and refused on mismatch |
| `REKHA_PEERS` | — | Topology as `id@internal-addr`, including this node |
| `REKHA_BOOTSTRAP` | `false` | Initialize a new cluster from this node (exactly one node) |
| `REKHA_CLUSTER_SECRET` | — | Shared secret for internal routes; ≥32 bytes in cluster mode |
| `REKHA_KEY_PEPPER` | — | Optional HMAC pepper for API-key hashes; set it to keep a secret out of the data directory |
| `REKHA_SHARDS` | `4 × nodes` | Placement shards; creation-only and persisted |
| `REKHA_REPLICATION_FACTOR` | `3` | Replicas per shard; creation-only and persisted |
| `REKHA_WAL_DURABILITY` | `durable` | `durable` fsyncs each batch; `relaxed` does not |
| `REKHA_SNAPSHOT_EVERY_N_LOGS` | `0` | Compact the Raft log every N entries, keeping N. `0` disables snapshotting, so the log grows but is never purged. Raise `raft_log.redb` size alerts before enabling |
| `REKHA_API_KEY` | — | API key for authentication; bootstraps the superadmin |
| `REKHA_TLS_CERT` | — | TLS certificate path |
| `REKHA_TLS_KEY` | — | TLS private key path |
| `REKHA_METRICS_PORT` | — | Prometheus metrics port |
| `REKHA_METRICS_HOST` | `127.0.0.1` | Metrics listener host; must be loopback |

### CLI Flags

```bash
# Exactly one node in a new cluster sets --bootstrap. Every node lists every
# node, including itself, and uses the same string. The addresses are the
# internal listener (9000), not the public API port.
export REKHA_API_KEY="$(openssl rand -hex 16)"
export REKHA_CLUSTER_SECRET="$(openssl rand -hex 32)"

rekha serve \
  --port 8000 \
  --internal-port 9000 \
  --data-dir /data \
  --node-id 1 \
  --peers "1@node1:9000,2@node2:9000,3@node3:9000" \
  --bootstrap \
  --tls-cert cert.pem \
  --tls-key key.pem
```

## Docker

### Single Node

```bash
export REKHA_API_KEY="$(openssl rand -hex 16)"
docker compose up -d
```

### 3-Node Cluster

The cluster compose file requires both secrets and bootstraps node 1. The
public API is on ports 8001–8003; Raft and forwarded writes use the internal
listener on 9000.

```bash
export REKHA_API_KEY="$(openssl rand -hex 16)"
export REKHA_CLUSTER_SECRET="$(openssl rand -hex 32)"
docker compose -f docker-compose.cluster.yml up -d --build
```

### Build Image

```bash
docker build -f Containerfile -t rekha .
```

## Production checklist

Before exposing a node to real traffic:

- Set `REKHA_API_KEY` and (in a cluster) `REKHA_CLUSTER_SECRET` from a secret
  manager. The binary refuses an API key shorter than 32 characters and a
  cluster secret shorter than 32 bytes. `REKHA_KEY_PEPPER` is optional
  defence-in-depth: set it if you want a secret the data directory does not
  contain.
- Terminate TLS on the public listener, and put the internal listener on a
  private interface. The internal routes are authenticated by the shared
  cluster secret, not by mTLS.
- Use `--wal-durability durable` (the default) in single-node mode. In cluster
  mode durability comes from the Raft log.
- Size the cluster once: `--shards` and `--replication-factor` are persisted and
  cannot be changed by a restart. A learner can be added, but elastic scale-out
  (promoting it to a voter and moving shards) is not supported yet; plan for the
  capacity you expect.
- By default the Raft log is never purged, so it grows and a restart replays it.
  Set `REKHA_SNAPSHOT_EVERY_N_LOGS=<n>` to compact it every `n` entries: a
  restarted node resumes from the purge watermark, and a node that was offline
  long enough to fall behind is repaired by a snapshot streamed from a replica
  that still has one. Keep the peer list and `--shards` fixed for the life of the
  data directory, because membership is reconstructed from the placement
  document after a purge.
- Size disk for snapshots. Each group keeps the last two archives under
  `<data-dir>/wal/snapshots/<group>/`, and an archive is a copy of its
  collection data — so a compacting node needs up to **three times** its data
  size (live data plus two archives). Retention is deliberately two while the
  transport still reads an archive into memory as one buffer; it drops to one
  when the transfer becomes a stream.
- Run `rekha verify` against a stopped data directory, take backups with
  `rekha backup`, and rehearse `rekha restore` before you need it.
- Reads are linearizable by default and are answered by the shard's leader, so
  size read throughput per shard rather than per node. Use `?consistency=local`
  for analytics, where a stale answer is acceptable.
- Scrape `REKHA_METRICS_PORT` from loopback only, and alert on `/ready`
  returning non-200, `rekha_raft_quorum_age_ms`, apply lag, and disk.
- Pin the image tag and keep the previous data directory until a rollback has
  been tested. On-disk formats may change before 1.0.

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
│   ├── rekha-wal/        # Write-ahead log (RKW2 batch format)
│   ├── rekha-index/      # HNSW index (usearch)
│   ├── rekha-engine/     # Engine: WAL→buffer→HNSW pipeline
│   ├── rekha-cluster/    # Cluster types, Raft log, state machine
│   ├── rekha-api/        # REST API (axum) + ChromaDB compat
│   ├── rekha-cli/        # CLI binary (rekha)
│   └── rekha-bench/      # Benchmarks
│
│  Shelved and excluded from the workspace: rekha-repl (a TUI SQL console)
│  and rekha-objectstore (a placeholder). Neither is referenced by the binary.
├── Containerfile
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
