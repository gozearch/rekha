# Rekha — Distributed Vector Database

Rekha is a distributed vector database built in Rust, designed for billion-scale Approximate Nearest Neighbor Search (ANNS). It uses **IVF (inverted file) indexing with Product Quantization (PQ)**, **LWW timestamp conflict resolution with hinted handoff replication**, and **Dynamo-style consistent hashing** for shard distribution.

> **Architecture details**: See [AGENTS.md](../AGENTS.md) for the full architecture document.

---

## Table of Contents

1. [Quick Start (Local Dev)](#1-quick-start-local-dev)
2. [Docker Cluster](#2-docker-cluster)
3. [Production Deployment](#3-production-deployment)
4. [Configuration Reference](#4-configuration-reference)
5. [CLI Reference](#5-cli-reference)
6. [Client SDKs](#6-client-sdks)
7. [Development Guide](#7-development-guide)

---

## 1. Quick Start (Local Dev)

### Prerequisites

- Rust 1.80+ (`rustup default stable`)
- `cargo install just` (optional — for the `justfile`)

### Build

```bash
git clone https://github.com/gozearch/rekha.git && cd rekha
cargo build --release
```

> For the Python SDK, see `rekha-python/`. Pure gRPC client — no PyO3 required.

### Run a Single-Node Server

```bash
# Start a single-node server with default dev config
./target/release/rekha server --config config.yaml
```

The included `config.yaml` is pre-configured for single-node dev:

```yaml
cluster:
  node_id: "node-1"
  seed_nodes: ["127.0.0.1:50051"]
  default_write_consistency: "quorum"
  hinted_handoff_enabled: true
  max_hint_window_secs: 3600
  heartbeat_interval_ms: 1000
  heartbeat_timeout_ms: 5000
  default_rf: 3
  seed_nodes: []

storage:
  max_payload_size: 4194304
  max_inline_size: 1024
  gc_grace_seconds: 86400
  gc_interval_secs: 3600

tls:
  enabled: false

observability:
  enable_tracing: false
  enable_metrics: false
  metrics_port: 9090
```

Start the server:

```bash
./target/release/rekha server --config config.yaml
```

### Insert & Search via CLI

```bash
# Insert — reads vector from stdin (space-separated floats)
echo "0.1 0.2 0.3 0.4" | rekha insert -c default

# With explicit ID
echo "0.1 0.2 0.3 0.4" | rekha insert -c default -i 42

# Search — reads query from stdin
echo "0.1 0.2 0.3 0.4" | rekha search -c default -k 10 -n 16

# Delete
rekha delete 42 43 44 -c default

# Health check
rekha health
```

### Import / Export

```bash
# Import from JSONL file (one vector per line)
rekha import -c default -i data.jsonl

# Export to JSONL file
rekha export -c default -o out.jsonl --offset 0 --limit 1000
```

### Generate Self-Signed Certificates (for TLS testing)

```bash
# CA key + cert
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days 365 -out ca.crt \
  -subj "/CN=Rekha Test CA"

# Server key + CSR + cert (signed by CA)
openssl genrsa -out server.key 4096
openssl req -new -key server.key -out server.csr \
  -subj "/CN=localhost"
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out server.crt -days 365 -sha256

# Client key + CSR + cert (for mTLS)
openssl genrsa -out client.key 4096
openssl req -new -key client.key -out client.csr \
  -subj "/CN=client"
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out client.crt -days 365 -sha256
```

Then set in `config.yaml`:

```yaml
tls:
  enabled: true
  cert_path: "/path/to/server.crt"
  key_path: "/path/to/server.key"
  client_ca_cert_path: "/path/to/ca.crt"   # Optional — for mTLS
```

### Run Tests

```bash
just test                # All unit tests (workspace)
just test-name transfer  # Tests matching "transfer" (regex)
cargo test --workspace   # Same as above, without `just`
```

> Integration tests that need a real gRPC server run by default. Tests that are slow are in `rekha-bench` (excluded from `just test`).

---

## 2. Docker Cluster

A 3-node Rekha cluster can be started locally with Docker Compose. Each node runs in its own container with a dedicated data volume.

### Prerequisites

- Docker Engine 24+
- Docker Compose v2

### Build & Start

```bash
# Build the image and start the 3-node cluster
docker compose up -d

# Wait for cluster to converge (~30s)
docker compose logs -f
```

The cluster topology:

| Container | Host Port | Node ID | Data Volume |
|---|---|---|---|
| `rekha-node-1` | `:50051` | `node-1` | `rekha-data-1` |
| `rekha-node-2` | `:50052` | `node-2` | `rekha-data-2` |
| `rekha-node-3` | `:50053` | `node-3` | `rekha-data-3` |

### Insert & Search

```bash
# Insert a vector via node-1
echo "0.1 0.2 0.3 0.4" | docker compose exec -T node-1 rekha insert -c default

# Insert via node-2
echo "0.4 0.3 0.2 0.1" | docker compose exec -T node-2 rekha insert -c default

# Search via node-3
echo "0.1 0.2 0.3 0.4" | docker compose exec -T node-3 rekha search -c default -k 10 -n 16

# Check cluster health
docker compose exec node-1 rekha health
```

### Configuration

Config files: `docker/node-{1,2,3}.yaml`. All three share:

- **Index**: IVF with `nlist=4096`, `nprobe=32`, PQ `m=64`, `k=256`
- **Replication**: factor 3, hinted handoff enabled
- **Consistency**: QUORUM default write
- **TLS**: disabled (plaintext inter-node)

They differ only in `cluster.node_id` and `listen` port.

### Tear Down

```bash
# Stop containers (preserves volumes)
docker compose down

# Stop containers and delete all data
docker compose down -v
```

### Justfile Commands

```bash
just docker-build       # Build Docker image
just docker-up          # Start cluster
just docker-down        # Stop cluster
just docker-down-clean  # Stop cluster + delete volumes
just docker-logs        # Follow logs
just docker-exec n cmd  # Run command on node
```

---

## 3. Production Deployment

### Topology Design

A Rekha cluster uses **consistent hashing (128 virtual nodes, SipHash)** to distribute shards across nodes. Each collection is an independent IVF index sharded by vector ID hash.

| Parameter | Example | Meaning |
|---|---|---|
| `default_rf` | 3 | Replication factor (copies per vector) |
| `default_write_consistency` | "quorum" | ONE / QUORUM / ALL |
| `hinted_handoff_enabled` | true | Store writes for unreachable replicas |
| `max_hint_window_secs` | 3600 | Max time to retain hints |

**Minimum nodes**: `default_rf` (e.g., 3). For balanced load, node count should divide the virtual node space evenly.

### 3-Node Cluster Example

Node 1 config (`/etc/rekha/node-1.yaml`):

```yaml
cluster:
  node_id: "node-1"
  seed_nodes:
    - "node-1:50051"
    - "node-2:50051"
    - "node-3:50051"
  default_write_consistency: "quorum"
  hinted_handoff_enabled: true
  max_hint_window_secs: 3600
  heartbeat_interval_ms: 1000
  heartbeat_timeout_ms: 5000
  default_rf: 3

storage:
  data_dir: "/data/rekha"
  max_payload_size: 4194304
  max_inline_size: 1024
  gc_grace_seconds: 86400
  gc_interval_secs: 3600

tls:
  enabled: true
  cert_path: "/etc/rekha/certs/node-1.crt"
  key_path: "/etc/rekha/certs/node-1.key"
  client_ca_cert_path: "/etc/rekha/certs/ca.crt"

observability:
  enable_tracing: true
  enable_metrics: true
  metrics_port: 9090
```

Node 2 — same config, but `node_id: "node-2"`, `listen: "0.0.0.0:50052"`, and its own cert/key.

Node 3 — same, `node_id: "node-3"`, `listen: "0.0.0.0:50053"`.

All nodes share the same `seed_nodes` list for cluster discovery.

### TLS Certificates for Production

Use a real CA or internal PKI:

```bash
# 1. Generate CA (offline, secure)
openssl req -x509 -newkey rsa:4096 -keyout ca.key -out ca.crt \
  -days 3650 -nodes -subj "/CN=Rekha Cluster CA"

# 2. Per-node certificates
for node in node-1 node-2 node-3; do
  openssl genrsa -out ${node}.key 4096
  openssl req -new -key ${node}.key -out ${node}.csr \
    -subj "/CN=${node}"
  openssl x509 -req -in ${node}.csr -CA ca.crt -CAkey ca.key \
    -CAcreateserial -out ${node}.crt -days 365 -sha256
done
```

Distribute:
- Each node: its own `{node}.crt`, `{node}.key`, plus `ca.crt`
- Clients: `ca.crt` (and optionally their own cert for mTLS)

### Cluster Startup

1. Start all nodes in any order (they discover each other via seed nodes)
2. Chord ring converges as nodes exchange membership via heartbeats
3. Cluster is ready when `rekha health` returns OK for all nodes
4. Each collection's IVF index is local; replicas are updated via async RPC

### Rolling Upgrade

```bash
# 1. Stop node (SIGTERM — graceful shutdown flushes RocksDB)
kill -TERM <pid>

# 2. Upgrade binary
cp rekha /usr/local/bin/rekha

# 3. Restart
/usr/local/bin/rekha server --config /etc/rekha/node-1.yaml

# 4. Wait for membership to reconverge
# 5. Repeat for remaining nodes
```

During upgrade of an RF=3 collection:
- 1 node down: quorum intact (2/3), writes succeed at QUORUM
- 2 nodes down: quorum lost → writes pause at QUORUM until majority restored

### Backup & Restore

```bash
# Backup (stop node first for consistent snapshot)
cp -r /data/rekha /backup/rekha-$(date +%Y%m%d)

# Restore
rsync -a /backup/rekha-20260603/ /data/rekha/
```

For live backup, use RocksDB's `Checkpoint` API (see `RocksVectorStore::create_checkpoint` in `rekha-storage`).

---

## 4. Configuration Reference

> **Full reference**: See [`config.md`](config.md) for the complete configuration documentation, including every field, its default, type constraints, sizing guidance, TLS examples, and config loading.

A minimal single-node config:

```yaml
cluster:
  node_id: "node-1"
  seed_nodes: ["127.0.0.1:50051"]
  default_write_consistency: "quorum"
  hinted_handoff_enabled: true
  max_hint_window_secs: 3600
  heartbeat_interval_ms: 1000
  heartbeat_timeout_ms: 5000
  default_rf: 3

storage:
  data_dir: "/tmp/rekha-data"
  max_payload_size: 4194304
  max_inline_size: 1024
  gc_grace_seconds: 86400
  gc_interval_secs: 3600

tls:
  enabled: false

observability:
  enable_tracing: false
  enable_metrics: false
  metrics_port: 9090
```

### Cluster Section

| Field | Type | Default | Description |
|---|---|---|---|
| `node_id` | string | random UUID | Unique node identifier |
| `seed_nodes` | []string | [] | Known peer addresses for bootstrapping |
| `default_write_consistency` | "one"\|"quorum"\|"all" | "quorum" | Write consistency level |
| `hinted_handoff_enabled` | bool | true | Enable hinted handoff for unavailable replicas |
| `max_hint_window_secs` | i64 | 3600 | Max seconds to retain hints |
| `heartbeat_interval_ms` | u64 | 1000 | Interval between heartbeats |
| `heartbeat_timeout_ms` | u64 | 5000 | Peer timeout before marking dead |
| `default_rf` | u32 | 3 | Default replication factor |

### Storage Section

| Field | Type | Default | Description |
|---|---|---|---|
| `data_dir` | string | "/tmp/rekha-data" | RocksDB data directory |
| `max_payload_size` | usize | 4MB | Max payload bytes per vector |
| `max_inline_size` | usize | 1KB | Inline payload threshold (else stored in CF) |
| `gc_grace_seconds` | i64 | 86400 | Tombstone retention before compaction |
| `gc_interval_secs` | u64 | 3600 | Interval between GC runs |

### TLS Section

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | false | Enable TLS |
| `cert_path` | string | null | Server certificate (PEM) |
| `key_path` | string | null | Server private key (PEM) |
| `client_ca_cert_path` | string | null | CA cert for mTLS client verification |

### Observability Section

| Field | Type | Default | Description |
|---|---|---|---|
| `enable_tracing` | bool | false | Enable tokio-console / tracing |
| `enable_metrics` | bool | false | Enable Prometheus metrics |
| `metrics_port` | u16 | 9090 | Prometheus HTTP listen port |

---

## 5. CLI Reference

The `rekha` binary provides admin commands for interacting with the cluster.

### Global Flags

| Flag | Env | Default | Description |
|---|---|---|---|
| `-c`, `--collection` | – | `default` | Collection name (most subcommands) |

### Subcommands

#### `rekha server`

Start a Rekha server node.

```bash
rekha server --config /etc/rekha/node-1.yaml
```

| Flag | Default | Description |
|---|---|---|
| `--config` | `config.yaml` | Path to YAML config file |

#### `rekha create-collection`

Create a new collection with IVF+PQ config.

```bash
# Minimal (uses defaults)
rekha create-collection -c images --rf 3

# Full config
rekha create-collection -c images --rf 3 --config '{"dim":256,"nlist":4096,"nprobe":32,"pq_m":64,"pq_k":256,"distance_metric":"l2"}'
```

| Flag | Default | Description |
|---|---|---|
| `-c`, `--collection` | — | Collection name |
| `--rf` | `1` | Replication factor |
| `--config` | see below | JSON IVF config |

**Default IVF config** (`--config`):
```json
{"dim":256,"nlist":4096,"nprobe":32,"pq_m":64,"pq_k":256,"distance_metric":"l2"}
```

#### `rekha list-collections`

List all collections.

```bash
rekha list-collections
```

#### `rekha collection-exists`

Check if a collection exists.

```bash
rekha collection-exists -c images
```

#### `rekha insert`

Insert a vector with optional payload. The vector is read from stdin as space-separated floats on a single line. ID is auto-generated if not provided.

```bash
# Insert (vector from stdin, id auto-generated)
echo "0.1 0.2 0.3 0.4" | rekha insert -c images

# With explicit ID
echo "0.1 0.2 0.3 0.4" | rekha insert -c images -i 42
```

| Flag | Default | Description |
|---|---|---|
| `-c`, `--collection` | `default` | Collection name |
| `-i`, `--id` | auto | Vector ID (u64) |

#### `rekha search`

Search for nearest neighbors. The query vector is read from stdin.

```bash
rekha search -c images -k 10 -n 16
echo "0.1 0.2 0.3 0.4" | rekha search -c images -k 10 -n 16
```

| Flag | Default | Description |
|---|---|---|
| `-c`, `--collection` | `default` | Collection name |
| `-k` | `10` | Number of results (top-k) |
| `-n`, `--nprobe` | `16` | IVF probes (higher = more accurate) |

#### `rekha delete`

Delete vectors by ID.

```bash
rekha delete 42 -c images
rekha delete 42 43 44 -c images
```

| Arg | Description |
|---|---|
| `ids` | One or more vector IDs (u64) |

#### `rekha import`

Bulk import vectors from a JSONL file. Each line: `{"id": 1, "vector": [0.1, ...], "payload": "optional", "timestamp": 123}`.

```bash
rekha import -c images -i data.jsonl
```

| Flag | Default | Description |
|---|---|---|
| `-c`, `--collection` | — | Collection name |
| `-i`, `--input` | — | Input JSONL file path |

#### `rekha export`

Export vectors to a JSONL file.

```bash
rekha export -c images -o out.jsonl --offset 0 --limit 1000
```

| Flag | Default | Description |
|---|---|---|
| `-c`, `--collection` | — | Collection name |
| `-o`, `--output` | — | Output JSONL file path |
| `--offset` | `0` | Starting offset |
| `--limit` | `1000` | Max vectors to export |

#### `rekha health`

Cluster health check.

```bash
rekha health
```

---

## 6. Client SDKs

### Rust SDK

Add to `Cargo.toml`:

```toml
[dependencies]
rekha-client = { git = "https://github.com/gozearch/rekha", package = "rekha-client" }
rekha-core = { git = "https://github.com/gozearch/rekha", package = "rekha-core" }
```

#### Connect

```rust
use rekha_client::Client;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Connect to a seed node (plaintext)
    let mut client = Client::connect("http://localhost:50051").await?;

    // Or with TLS
    // let mut client = Client::connect("https://node-1:50051").await?;
}
```

#### CRUD

```rust
use rekha_core::{ConsistencyLevel, IvfConfig, SearchParams};

// Create collection
let config = IvfConfig {
    dim: 256,
    nlist: 4096,
    nprobe: 32,
    pq_m: 64,
    pq_k: 256,
    replication_factor: 3,
    distance_metric: rekha_core::DistanceMetric::L2,
};
client.create_collection("images", config).await?;

// Insert (id=0 means auto-generate)
let id = 0;
client.insert(
    "images",
    id,
    vec![0.1, 0.2, 0.3, 0.4],
    None,                        // optional payload: Option<Vec<u8>>
    1000,                        // timestamp
    ConsistencyLevel::One,
).await?;

// Search
let params = SearchParams {
    nprobe: 32,
    k: 10,
    include_payloads: false,
    pre_filter: None,
    local_only: false,
};
let results = client.search("images", vec![0.1, 0.2, 0.3, 0.4], 10, params).await?;
for r in &results {
    println!("id={} score={}", r.id, r.score);
}

// Delete
let deleted = client.delete("images", &[id], 1001, ConsistencyLevel::One).await?;
```

#### SearchParams

| Field | Type | Default | Description |
|---|---|---|---|
| `nprobe` | u32 | 16 | Number of IVF clusters to probe |
| `k` | u32 | 10 | Top-k results |
| `include_payloads` | bool | false | Include payload in results |
| `pre_filter` | Option<...> | None | Pre-filter (reserved) |
| `local_only` | bool | false | Skip remote replicas |

### Python SDK

Package in `rekha-python/`. Pure gRPC client (no PyO3).

```python
from rekha import RekhaClient

# Connect
with RekhaClient.connect(["localhost:50051"]) as client:
    # Create collection
    client.create_collection("images", dim=256, nlist=4096, nprobe=32, pq_m=64, pq_k=256, rf=3)

    # Insert (id=0 means auto-generate)
    client.insert([0.1, 0.2, 0.3, 0.4], "images", payload=b'{"k":"v"}')

    # Search
    results = client.search([0.1, 0.2, 0.3, 0.4], top_k=10, collection_name="images", nprobe=32)
    for r in results:
        print(f"id={r.id}, score={r.score}")

    # Streaming search
    for r in client.search_stream([0.1, 0.2, 0.3, 0.4], 10, "images", nprobe=32):
        print(f"streamed: id={r.id}")

    # Delete
    count = client.delete([42, 43], "images")
```

---

## 7. Development Guide

### Crate Map

| Crate | Responsibility | Dependencies |
|---|---|---|
| `rekha-core` | Types, traits, errors, distance math | (zero) |
| `rekha-storage` | RocksDB wrapper (4 column families) | `rekha-core` |
| `rekha-index` | IVF index + Product Quantization | `rekha-core`, `rekha-storage` |
| `rekha-quant` | Product Quantization & K-Means | `rekha-core` |
| `rekha-replication` | LWW timestamps, quorum gate, hinted handoff | `rekha-core` |
| `rekha-cluster` | Membership, consistent hash ring, peer state | `rekha-core` |
| `rekha-coordinator` | Write/read path orchestration, collection DDL, peer pool | All above |
| `rekha-server` | gRPC server, service handlers, config loading | `rekha-coordinator` + tonic |
| `rekha-client` | Rust SDK | `rekha-core` + tonic |
| `rekha-proto` | Shared protobuf types & converters | prost/tonic |
| `rekha-cli` | Admin CLI binary | `rekha-client` + clap |
| `rekha-bench` | Benchmarks (excluded from default test) | `rekha-core` |

### Justfile Commands

| Command | What it does |
|---|---|
| `just test` | Run all unit tests (workspace, excluding `rekha-bench`) |
| `just lint` | Format check (`cargo fmt`) + clippy (deny warnings) |
| `just fix` | Auto-format with `cargo fmt --all` |
| `just build` | Build workspace (debug) |
| `just release-build` | Build release binaries |
| `just check` | Fast compilation check (`cargo check`) |
| `just coverage` | Generate LCOV coverage report (needs nightly + cargo-llvm-cov) |
| `just coverage-html` | Generate HTML coverage report and open in browser |
| `just test-name <name>` | Run tests matching `<name>` (regex, e.g., `just test-name transfer`) |

### Testing Conventions

- **Unit tests**: Standard `#[test]` functions alongside the code they test. Run by `just test`.
- **Integration tests**: Grouped in `tests/` directories (`rekha-server/tests/integration.rs`, `rekha-server/tests/multi_node.rs`, `rekha-server/tests/service_handlers.rs`). Run by `just test`.
- **Parallel safety**: Tests that use RocksDB use unique temp directories (via `AtomicU64` counter) to avoid collisions.
- **Coverage**: `just coverage` runs all workspace tests with `cargo-llvm-cov` (requires `nightly` Rust and `cargo install cargo-llvm-cov`).

### Adding a New Crate

1. Create `rekha-<name>/` with `Cargo.toml` and `src/lib.rs`
2. Add to `members` and `default-members` in workspace `Cargo.toml`
3. Add dependencies following existing patterns
4. Add tests following error-propagation conventions

### Error Propagation Pattern

```rust
// Each layer defines its own error enum, which converts to RekhaError:
impl From<MyLayerError> for RekhaError {
    fn from(e: MyLayerError) -> Self {
        RekhaError::MyVariant { source: Box::new(e), detail: e.to_string() }
    }
}
```