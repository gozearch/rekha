# Rekha Configuration Reference

All configuration is YAML. The schema is defined by `ServerConfig` in
`rekha-server/src/config.rs`.

**Loading order** (first found wins):
1. `--config <PATH>` CLI flag
2. `REKHA_CONFIG` environment variable (path to YAML file)
3. `ServerConfig::dev_default("node-1", "/tmp/rekha-data")` (embedded default)

---

## `cluster`

Required. Identifies this node and tells it how to find the cluster.

```yaml
cluster:
  node_id: "node-1"
  seed_nodes: ["127.0.0.1:50051"]
  bind_addr: "0.0.0.0:50051"
  data_dir: "/data/rekha"
```

| Field | Type | Default | Description |
|---|---|---|---|
| `node_id` | `string` | — | Unique identifier for this node (e.g. `"node-1"`, `"us-east-1a-rekha-3"`). Used in cluster topology, Raft leader IDs, and logs. Must be unique within the cluster. |
| `seed_nodes` | `string[]` | — | Initial contact points for cluster discovery. Every node in the cluster should list all nodes here (or at least a quorum of them). On startup the node picks one seed, performs a `Handshake`, and receives the full peer list. Order does not matter. |
| `bind_addr` | `string` | — | The gRPC listen address, e.g. `"0.0.0.0:50051"` (all interfaces) or `"10.0.1.5:50051"` (specific IP). The port must match what other nodes use in their `seed_nodes`. |
| `data_dir` | `string` | — | Persistent data directory. Contains: RocksDB database, per-shard Raft log, index metadata, and configuration state. Should be on a fast filesystem (SSD/NVMe recommended). |

---

## `tls`

Optional. Uses `rustls` (pure Rust, no OpenSSL dependency). Controls both
server-side encryption and optional mutual TLS (mTLS).

```yaml
tls:
  enabled: false
  cert_path: null
  key_path: null
  ca_cert_path: null
```

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` | When `true`, the gRPC server loads the TLS certificate and key, and all inter-node and client connections must use TLS. When `false`, all communication is plaintext HTTP/2. |
| `cert_path` | `string` | `null` | Path to the server TLS certificate in PEM format. Required when `enabled: true`. The certificate's CN/SAN must match the hostname clients use to connect. |
| `key_path` | `string` | `null` | Path to the server TLS private key in PEM format. Required when `enabled: true`. The key must not have a passphrase (rustls does not support encrypted keys). |
| `ca_cert_path` | `string` | `null` | Optional. Path to a CA certificate in PEM format. When set, the server performs mTLS — it requests and verifies client certificates against this CA. Recommended for production inter-node communication to prevent unauthorized nodes from joining. |

**Examples:**

```yaml
# Plaintext (dev)
tls:
  enabled: false

# Server-side TLS only (clients verify server, server does not verify clients)
tls:
  enabled: true
  cert_path: "/etc/rekha/certs/server.crt"
  key_path: "/etc/rekha/certs/server.key"

# Mutual TLS (both sides verify)
tls:
  enabled: true
  cert_path: "/etc/rekha/certs/server.crt"
  key_path: "/etc/rekha/certs/server.key"
  ca_cert_path: "/etc/rekha/certs/ca.crt"
```

**Generating test certificates:** See `docs/README.md` § Quick Start.

---

## `partition`

Controls how vectors are split across the cluster — both horizontally (by
vector shard) and vertically (by dimension group).

```yaml
partition:
  num_vector_shards: 16
  replication_factor: 3
  num_dim_groups: 4
  dim_group_size: 192
```

| Field | Type | Default | Description |
|---|---|---|---|
| `num_vector_shards` | `u64` | `1` | Number of horizontal shards (Raft replication groups). Vectors are assigned to shards by `id % num_vector_shards`. Increase this to scale write throughput and reduce per-shard memory pressure. Each shard maintains its own Vamana graph. |
| `replication_factor` | `usize` | `1` | Number of replicas per shard. `1` = no replication. Raft majority is `floor(replication_factor / 2) + 1`. Must be ≤ the number of nodes in the cluster (otherwise some shards cannot form quorum). |
| `num_dim_groups` | `u32` | `4` | Number of vertical dimension partitions. Vectors are split into `num_dim_groups` groups of `dim_group_size` consecutive dimensions. During search, the coordinator queries a subset of groups first (early-stop). |
| `dim_group_size` | `usize` | `64` | Dimensions per group. **Constraint:** `dim_group_size * num_dim_groups` must equal the vector dimension. For 768-dim vectors: `192 * 4 = 768` or `64 * 12 = 768`. |

**Sizing guidance:**

- **`num_vector_shards`**: Start with `2× node_count` for balanced load. Each shard adds ~R× graph memory (where R = `graph_degree`). For 1M vectors with `graph_degree=64`: ~64 edges × 8 bytes × 2 (forward+reverse) ≈ 1 KB per vector per shard × replica count.
- **`num_dim_groups`**: 4 is a sensible default for most workloads. Higher values enable finer-grained early-stop (faster search) but increase metadata overhead. If your vector dimension is small (<128), use fewer groups.
- **`replication_factor`**: 3 is standard for production (tolerates 1 node failure per shard). 5 for higher availability (tolerates 2 failures). 1 for dev only.

---

## `index`

Vamana + PQ index parameters. These affect accuracy, memory usage, and search
latency.

```yaml
index:
  type: "vamana"
  graph_degree: 64
  search_list_size: 128
  pq_num_sub_vectors: 64
  pq_num_centroids: 256
  re_rank_k: 256
```

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | `"vamana"` | Index algorithm. Currently only `"vamana"` is supported. |
| `graph_degree` | `usize` | `64` | Vamana graph out-degree (R). Controls the maximum number of outgoing edges per node in the graph. Higher values → more accurate search, more memory (8 bytes per edge per direction), slower index build. Typical range: 32–128. |
| `search_list_size` | `usize` | `128` | Beam width during search (`ef_search` in HNSW/DiskANN terms). Controls the size of the candidate set maintained during greedy search. Higher → more accurate, slower. Typically 1.5–2× `graph_degree`. |
| `pq_num_sub_vectors` | `usize` | `64` | Number of sub-vectors for Product Quantization (PQ) compression (M). The vector dimension must be divisible by this value. Each sub-vector is encoded as `pq_num_centroids` clusters. Lower M → better recall (less compression), higher M → smaller memory footprint. **Constraint:** `dim_group_size % pq_num_sub_vectors == 0` (for in-group search). |
| `pq_num_centroids` | `usize` | `256` | Number of centroids per PQ sub-vector (K). `256` = 1 byte per sub-vector (8-bit encoding). Valid values: 16, 32, 64, 128, 256. Larger values → better accuracy, more memory for codebooks. |
| `re_rank_k` | `usize` | `256` | Number of candidates to re-rank with full-precision vectors after the initial PQ-distance search. Set to 0 to skip re-ranking (PQ distance only). Higher → better final accuracy, more I/O (full vectors read from RocksDB). Typically 2–4× `top_k` of the query. |

**Memory formula:**

For 1M 768-dim vectors with `graph_degree=64`, `pq_num_sub_vectors=64`:

| Component | Memory |
|---|---|
| Vamana graph (edges) | 1M × 64 × 8 B × 2 ≈ 1 GB |
| PQ codes | 1M × 64 bytes ≈ 64 MB |
| PQ codebooks | 256 × 64 × 4 B × `num_dim_groups` ≈ 256 KB |
| Total | ≈ 1.1 GB per shard |

---

## `raft`

Raft consensus parameters. Controls leader election speed, heartbeat frequency,
and log compaction.

```yaml
raft:
  heartbeat_interval_ms: 100
  election_timeout_min_ms: 300
  election_timeout_max_ms: 500
  snapshot_interval: 10000
```

| Field | Type | Default | Description |
|---|---|---|---|
| `heartbeat_interval_ms` | `u64` | `100` | Leader heartbeat interval in milliseconds. The leader sends `AppendEntries` (heartbeats) to followers at this rate. Lower → faster failure detection, more network traffic. Higher → less network overhead, slower convergence. |
| `election_timeout_min_ms` | `u64` | `300` | Minimum election timeout in milliseconds. When a follower receives no heartbeat within its timeout window (randomized between min and max), it starts a new election. Should be ≥ `2× heartbeat_interval_ms`. |
| `election_timeout_max_ms` | `u64` | `500` | Maximum election timeout in milliseconds. The actual timeout is randomized `[min, max)` per node to prevent split votes. |
| `snapshot_interval` | `u64` | `10000` | Number of log entries between Raft snapshots. When this many entries have been committed since the last snapshot, the state machine takes a snapshot and truncates the log up to that point. Lower → faster recovery from crashes, more I/O. Higher → less snapshot overhead, longer replay on restart. |

**Tuning guidance:**

- **Local cluster (low latency, low jitter):** `heartbeat=50ms`, `election=[150, 300]`
- **Production LAN (1–5ms RTT):** `heartbeat=100ms`, `election=[300, 500]`
- **Geo-distributed (50–100ms RTT):** `heartbeat=500ms`, `election=[1500, 3000]`
- **Snapshot interval:** For 1M entries at ~200 bytes each = 200 MB log. Set `snapshot_interval` so that log size stays manageable (10K–100K is typical).

---

## `storage`

RocksDB storage parameters.

```yaml
storage:
  max_payload_size: 1048576
  max_inline_size: 1048576
```

| Field | Type | Default | Description |
|---|---|---|---|
| `max_payload_size` | `usize` | `1048576` | Maximum allowed payload size in bytes (default 1 MB). Insert requests with payloads larger than this are rejected with `InvalidArgument`. This prevents memory exhaustion from oversized payloads. |
| `max_inline_size` | `usize` | `1048576` | Maximum inline value size in bytes for RocksDB. Values smaller than this are stored inline in the SST file; larger values cause RocksDB to spill to a separate blob file. Tuning this affects read amplification vs. write amplification. The default (1 MB) keeps most payloads inline. |

**Payload limits:** If your application stores large binary blobs (images,
documents), increase `max_payload_size` accordingly. Recommended maximum:
10 MB. Beyond that, store references (e.g. S3 URIs) in the payload and the
blobs externally.

---

## `observability`

```yaml
observability:
  metrics: "prometheus"
  tracing: "none"
  logging: "structured"
```

| Field | Type | Default | Description |
|---|---|---|---|
| `metrics` | `string` | `"prometheus"` | Metrics backend. `"prometheus"` exposes a Prometheus `/metrics` HTTP endpoint. `"none"` disables metrics. |
| `tracing` | `string` | `"none"` | Distributed tracing backend. `"jaeger"` exports OpenTelemetry traces to Jaeger. `"none"` disables tracing. |
| `logging` | `string` | `"structured"` | Log format. `"structured"` outputs JSON-structured logs (machine-parseable). `"plain"` outputs human-readable text. |

---

## Complete Examples

### Single-Node Development

```yaml
cluster:
  node_id: "node-1"
  seed_nodes: ["127.0.0.1:50051"]
  bind_addr: "0.0.0.0:50051"
  data_dir: "/tmp/rekha-data"

partition:
  num_vector_shards: 1
  replication_factor: 1
  num_dim_groups: 4
  dim_group_size: 64

index:
  type: "vamana"
  graph_degree: 64
  search_list_size: 128
  pq_num_sub_vectors: 64
  pq_num_centroids: 256
  re_rank_k: 256

raft:
  heartbeat_interval_ms: 100
  election_timeout_min_ms: 300
  election_timeout_max_ms: 500
  snapshot_interval: 10000

storage:
  max_payload_size: 1048576
  max_inline_size: 1048576

tls:
  enabled: false

observability:
  metrics: "prometheus"
  tracing: "none"
  logging: "structured"
```

### 3-Node Production Cluster

Node 1 (`/etc/rekha/node-1.yaml`):

```yaml
cluster:
  node_id: "node-1"
  seed_nodes:
    - "node-1:50051"
    - "node-2:50051"
    - "node-3:50051"
  bind_addr: "0.0.0.0:50051"
  data_dir: "/data/rekha"

partition:
  num_vector_shards: 12
  replication_factor: 3
  num_dim_groups: 4
  dim_group_size: 192

index:
  type: "vamana"
  graph_degree: 64
  search_list_size: 128
  pq_num_sub_vectors: 64
  pq_num_centroids: 256
  re_rank_k: 256

raft:
  heartbeat_interval_ms: 100
  election_timeout_min_ms: 300
  election_timeout_max_ms: 500
  snapshot_interval: 10000

storage:
  max_payload_size: 1048576
  max_inline_size: 1048576

tls:
  enabled: true
  cert_path: "/etc/rekha/certs/node-1.crt"
  key_path: "/etc/rekha/certs/node-1.key"
  ca_cert_path: "/etc/rekha/certs/ca.crt"

observability:
  metrics: "prometheus"
  tracing: "jaeger"
  logging: "structured"
```

Nodes 2 and 3 are identical except for `node_id`, `bind_addr` port, and certificate paths.

### Geo-Distributed (50ms RTT)

```yaml
raft:
  heartbeat_interval_ms: 500
  election_timeout_min_ms: 1500
  election_timeout_max_ms: 3000
  # Keep defaults for everything else
```

---

## Loading Configuration

### From CLI

```bash
rekha server --config /etc/rekha/node-1.yaml
```

### From Environment Variable

```bash
export REKHA_CONFIG=/etc/rekha/node-1.yaml
rekha server
```

### Programmatic (Rust)

```rust
use rekha_server::config::ServerConfig;

let config = ServerConfig::from_file("/etc/rekha/node-1.yaml")?;

// Or use dev defaults
let config = ServerConfig::dev_default("node-1", "/tmp/rekha-data");
```

### YAML Validation

The config file is deserialized with `serde_yaml`. Common errors:

| Error | Likely Cause |
|---|---|
| `missing field 'cluster'` | Config file is empty or wrong structure |
| `invalid type: string ""`, expected u64 | Field has wrong type (e.g., string instead of integer) |
| `unknown field 'foo'` | Typo in field name |
| `EOF while parsing a value` | YAML syntax error (check indentation) |
| `No such file or directory` | Path does not exist or is relative when absolute expected |

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `REKHA_CONFIG` | Path to server configuration YAML file |
| `RUST_LOG` | Log level for `tracing`/`env_logger` (e.g. `info`, `debug`, `rekha_server=debug`) |
| `ROCKSDB_LOG` | RocksDB internal logging (off by default) |
