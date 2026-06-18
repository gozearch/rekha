# Rekha — Distributed Vector Database Design Document

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core Data Model](#2-core-data-model)
3. [Multi-Granularity Partitioning](#3-multi-granularity-partitioning)
4. [Index Layer: Vamana + PQ](#4-index-layer-vamana--pq)
5. [Storage Layer: RocksDB](#5-storage-layer-rocksdb)
6. [Raft Consensus](#6-raft-consensus)
7. [Coordinator & Query Execution](#7-coordinator--query-execution)
8. [Client SDKs](#8-client-sdks)
9. [Error Handling](#9-error-handling)
10. [Observability](#10-observability)
11. [Configuration Reference](#11-configuration-reference)

---

## 1. Architecture Overview

Rekha is a distributed vector database designed for billion-scale Approximate
Nearest Neighbor Search (ANNS) with support for arbitrary payload storage.
It is built from the ground up in Rust.

**Key design principles:**
- **Disk-first**: Vectors live on disk (HDD/SSD), with only PQ codes and the
  Vamana graph in memory. This enables billion-scale on modest hardware.
- **Graceful degradation**: Every failure path is handled with clear error
  messages, automatic retries, and circuit breakers. No silent data loss.
- **Multi-granularity partitioning**: Combines vector-based (horizontal) and
  dimension-based (vertical) partitioning to balance load and enable early-stop
  pruning during search.
- **Strong consistency**: Raft consensus per partition ensures linearizable
  writes. Clients can read from followers for lower latency.

```
┌─────────────────────────────────────────────────────────┐
│                     Client (Rust/Python)                  │
└─────────────────────┬───────────────────────────────────┘
                      │ gRPC (Tonic)
┌─────────────────────▼───────────────────────────────────┐
│                    Coordinator                            │
│  ┌──────────┐  ┌──────────┐  ┌──────┐  ┌────────────┐  │
│  │Query     │  │Result    │  │Peer  │  │Raft Node   │  │
│  │Router    │  │Merger    │  │Pool  │  │Registry    │  │
│  └──────┬───┘  └────┬─────┘  └──┬───┘  └─────┬──────┘  │
└─────────┼────────────┼──────────┼─────────────┼─────────┘
          │            │          │             │
          ▼            ▼          ▼             ▼
┌─────────────────────────────────────────────────────────┐
│                   Rekha Node                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ Vamana + │  │ RocksDB  │  │ Raft     │  │Partition│  │
│  │ PQ Index │  │ (4 CFs)  │  │ Consensus│  │ Manager │  │
│  │          │  │ ┌──────┐ │  │ (Timers) │  │         │  │
│  │          │  │ │Raft  │ │  │          │  │         │  │
│  │          │  │ │Log   │ │  │          │  │         │  │
│  │          │  │ │Store │ │  │          │  │         │  │
│  └──────────┘  │ └──────┘ │  └──────────┘  └────────┘  │
│                └──────────┘                              │
└─────────────────────────────────────────────────────────┘
```

### Crate Architecture (Cargo Workspace)

| Crate | Responsibility | Implemented Features |
|---|---|---|
| `rekha-core` | Types, traits, errors, distance math (zero deps) | 36 error variants, 3 distance metrics |
| `rekha-storage` | RocksDB wrapper with column families | 4 CFs (vectors, payloads, metadata, raft_log), WriteBatch, WAL flush on drop |
| `rekha-index` | Vamana graph + Product Quantization | Build, search, dim-range search, early-stop pruning |
| `rekha-partition` | Multi-granularity partition strategy | 2-level grid, topology management, node health |
| `rekha-raft` | Raft consensus per partition | Leader election, log replication, persistence via RaftLogStore, timer loops |
| `rekha-server` | gRPC server, coordinator, service handlers | Query coordinator (local + peer fan-out + re-rank), PeerPool, heartbeat loop, raft timers, handshake/heartbeat RPCs |
| `rekha-client` | Rust SDK (connection pool, retry, streaming) | Multi-seed connect, exponential backoff + jitter, TLS support |
| `rekha-python` | Python SDK (pure gRPC, no PyO3) | Insert, search, delete, fetch, streaming, cluster info |
| `rekha-cli` | Admin CLI tool | server start, insert, search, delete |

---

## 2. Core Data Model

### Vector

```rust
pub struct Vector {
    pub id: u64,
    pub data: Vec<f32>,  // Full-precision vector
}
```

### Payload

```rust
pub struct Payload {
    pub content_type: PayloadType,  // Text, Json, or Raw
    pub data: Vec<u8>,             // Arbitrary bytes
}
```

Payloads are stored alongside vectors in RocksDB but are fetched **lazily**
during search — they are loaded only after the ANN result set is finalized
(Phase 2), not during distance computation (Phase 1).

### ScoredPoint

```rust
pub struct ScoredPoint {
    pub id: u64,
    pub score: f32,         // Distance metric
    pub payload: Option<Payload>,
}
```

### Distance Metrics

- **L2** (Euclidean): `sum((a_i - b_i)^2)` — supports exact early-stop
- **Cosine**: `1 - cos_sim` — supports approximate early-stop
- **Inner Product**: `-dot(a, b)` — supports approximate early-stop

### Error Types

A single `RekhaError` enum flows through the entire system, with specific
sub-error enums for each layer:

```
RekhaError
├── NotFound
├── InvalidArgument
├── IndexFull
├── InvalidDimension
├── Storage(StorageError)
├── Index(IndexError)
├── Partition(PartitionError)
├── Consensus(RaftError)
├── Timeout
├── ClusterChanged
├── Unavailable
└── Internal
```

Each error:
- Implements `Display` with a human-readable message
- Implements `Error` with `source()` chain
- Is `Clone` (not just `Debug`) — important for async error propagation
- Converts automatically from layer-specific errors via `From` impls

---

## 3. Multi-Granularity Partitioning

Rekha implements the **two-level partition strategy** described in HARMONY:

### Level 1: Vector-based (horizontal sharding)

Vectors are assigned to shards via `id % num_vector_shards`. Each shard
is a Raft replication group with `R` replicas. This provides:
- Horizontal scaling: add nodes → redistribute shards
- Fault tolerance: R replicas per shard
- Write throughput: Raft leader handles writes; followers serve reads

### Level 2: Dimension-based (vertical partitioning)

Within each vector shard, vectors are split across dimension groups.
Each dimension group handles a contiguous range of dimensions:
`[g * dims_per_group, (g+1) * dims_per_group)`.

This enables **early-stop pruning** during search:
1. Each node computes L2 distance only for its assigned dimension range
2. After each dimension group, the partial distance is compared to the
   current k-th best distance
3. For L2: partial ≤ total (monotonic) → if partial > threshold, STOP
4. This eliminates unnecessary computation and network transfer

### Partition Grid

```
                    Dim Group 0    Dim Group 1    Dim Group 2    Dim Group 3
Vector Shard 0      [Node A]       [Node B]       [Node A]       [Node C]
Vector Shard 1      [Node B]       [Node C]       [Node B]       [Node A]
Vector Shard 2      [Node C]       [Node A]       [Node C]       [Node B]
Vector Shard 3      [Node A]       [Node B]       [Node A]       [Node C]
```

A search query is:
1. Fanned out to all relevant (shard, dim_group) pairs
2. Each node computes partial distances with early-stop
3. The coordinator merges partial results using a min-heap → top-k

---

## 4. Index Layer: Vamana + PQ

Based on the DiskANN design, optimized for HDD-friendly access:

### In-Memory (fast)
- **PQ codes**: 64 bytes per vector (for M=64, K=256, 768-dim vector).
  This is 48× smaller than full-precision storage (3072 bytes).
- **Distance tables**: 64 × 256 f32 = 64 KB per query — computed once,
  used for all candidates during search.
- **Vamana graph adjacency**: Each node stores R neighbors (e.g., 64),
  so 512 bytes per vector. For 1B vectors: ~512 GB — too much for RAM.
  **Solution**: Store graph on disk with memory-mapped pages.

### On Disk (HDD-friendly)
- **Full-precision vectors**: Flat array organized for sequential reads.
  PQ-filtered candidates are re-ranked by reading their full vectors.
- **Vamana graph edges**: Fixed-size adjacency pages designed for
  sequential I/O (HDD-friendly). Graph construction minimizes random seeks.
- **Payloads**: In a separate RocksDB column family.

### Search Algorithm

```
1. Build PQ distance table for query vector (once, fast)
2. Start from medoid node in Vamana graph
3. Beam search with list size L (ef_search):
   a. Pop closest unvisited node from candidate set
   b. Use PQ-ADC distance (from precomputed table) for speed
   c. Add its neighbors to candidate set
   d. If closest candidate is worse than k-th result, STOP
4. Take top candidates from beam
5. Re-rank top candidates using full-precision vectors (from disk)
6. Return top-k with scores
```

### Early-Stop Pruning (Dimension-Aware)

For dimension-based partitioning:
```
fn search_dim_range(query, k, dim_start, dim_end, params):
    candidates = []
    for each vector in this node's shard:
        partial = l2_partial(query, vector, dim_start, dim_end)
        if partial > current_kth_best AND metric == L2:
            skip  // L2 is monotonic: adding more dims only increases distance
        candidates.push((partial, id))
    sort candidates by partial
    top = candidates[:ef_search * 2]  // extra for safety
    re-rank top with full distance (read from disk)
    return top-k
```

---

## 5. Storage Layer: RocksDB

### Column Families

| Column Family | Content | Key | Value |
|---|---|---|---|
| `vectors` | Full-precision vector data | `u64 BE` | `[f32; D] LE` |
| `payloads` | Arbitrary user payloads | `u64 BE` | `[u8]` |
| `metadata` | Index/PQ/partition config | string | JSON |
| `raft_log` | Raft WAL entries | `[partition_id (8B BE)] + [log_index (8B BE)]` | `bincode(RaftLogEntry)` |

The `raft_log` column family uses a composite key (`partition_id` + `log_index`, both
big-endian) so that entries for different partitions are stored contiguously and can
be iterated by prefix scan.

### RaftLogStore

A `RaftLogStore` wraps `RocksVectorStore` and provides typed access to the `raft_log`
column family:

- `store_entry()` / `store_entries()` — persist single or batched `RaftLogEntry` values
  serialized with `bincode`
- `load_entries(partition_id, from_index)` — prefix-scan entries starting from a given index
- `last_log_index()` / `last_log_term()` — reverse-iterate to find the latest entry
- `truncate_entries()` — delete entries from a given index onward (used on log conflict)
- `store_state()` / `load_state()` — persist/load `(term, voted_for)` tuples for durable
  election state
- `entry_count()` — count entries for a partition

### Generic CF Access

The `db()` accessor function in `RaftLogStore` returns a handle to the underlying RocksDB
instance, enabling any caller to operate on any column family by name. This pattern is used
by the `RaftLogStore` itself for direct CF operations and is available to other components
that need raw DB access.

### Write Path (Atomicity)

Vectors and payloads are written atomically via RocksDB's `WriteBatch`:

```rust
store.write_batch()
    .put_vector(id, &vector)
    .put_payload(id, &payload)
    .commit()?;  // Atomic: all or nothing
```

### Compaction

- Raft log compaction: periodic snapshots (every 10K entries)
- Old log entries are garbage-collected after snapshot
- Compaction filter removes deleted vectors from SST files

---

## 6. Raft Consensus

Each vector shard has its own Raft group with `R` replicas.

### Roles
- **Leader**: Handles all writes, replicates log to followers
- **Followers**: Serve reads (possibly stale), replicate log from leader
- **Candidates**: Election phase only

### Key Parameters
- Heartbeat interval: 100ms
- Election timeout: 300-500ms (randomized)
- Snapshot interval: 10,000 entries

### Write Path
1. Client sends `Insert` to any node
2. Node routes to Raft leader for the target partition
3. Leader persists entry to RocksDB (via `RaftLogStore.store_entry`) — **WAL-first**
4. Entry appended to in-memory log and applied to state machine immediately
5. Leader responds to client (single-node; multi-node would replicate to followers first)

### Persistence (RaftLogStore)

The `RaftLogStore` provides durable storage for the Raft log and election state:

- **Log entries**: serialized with `bincode` into the `raft_log` CF keyed by
  `[partition_id][log_index]`. This enables range iteration for loading entries
  on node restart.
- **Election state**: `(term, voted_for)` stored at a sentinel key per partition
  (`[partition_id][0xFF * 8]`).
- **Startup recovery**: `RaftNode::with_store()` calls `load_state()` and
  `load_entries()` on construction to restore in-memory state from RocksDB.
  This means a restarted node picks up exactly where it left off — no data loss.
- **Log compaction**: entries can be truncated from a given index onward
  (used on log conflicts during AppendEntries). Periodic snapshot-based
  compaction is planned.

### Read Path
1. Client sends `Search` to any node
2. Node routes to any replica (leader or follower) for the partition
3. Replica runs local search on its copy of the index
4. Results are returned directly (no Raft round-trip)

### Timer Loops

Two timer loops run in the server's background task:

1. **Heartbeat loop** (`spawn_heartbeat_loop`): Periodically sends `HeartbeatRequest`
   gRPC messages to all seed nodes at `heartbeat_interval_ms` (default 100ms). Each
   heartbeat carries the node's current raft term and commit index. Also checks peer
   health every 10th tick by calling `Coordinator::check_peer_health()`.

2. **Election timer** (`spawn_raft_timers`): Periodically checks election timeout
   for every Raft node on this server. The check frequency is the minimum of
   `election_timeout_min_ms` and 100ms. When a timeout elapses (with randomized
   jitter of up to 50% of the base timeout), the node transitions to candidate
   and starts an election.

### gRPC Wiring

The Raft protocol messages flow through gRPC handlers in `RekhaService`:

- **`raft_append_entries`**: Converts proto `RaftLogEntry` → internal `RaftLogEntry`,
   delegates to `RaftNode::handle_append_entries()`. Returns success status, current
   term, and commit index.
- **`raft_request_vote`**: Converts proto vote request → `RaftNode::handle_request_vote()`.
   Returns vote grant status and current term.
- **`raft_install_snapshot`**: Streaming snapshot transfer (stub — returns success
   without data).
- **Proto conversion**: `proto_raft_command_to_internal()` maps proto `Insert`/`Delete`/`Custom`
   commands to the internal `RaftCommand` enum used by the state machine.

---

## 7. Coordinator & Query Execution

### Search Flow (Three-Phase Execution)

The coordinator splits a search query into three phases:

**Phase 1 — Local Dimension Group Fan-Out:**
The query is split into `num_dim_groups` contiguous dimension ranges. Each
range is searched against the local index via `search_dim_range()`, which
computes partial L2 distances over that range with early-stop pruning.
Results from all local groups are merged into a single candidate list.

**Phase 2 — Peer Fan-Out:**
If the `PeerPool` has connected peers, the query is fanned out to all of
them via `PeerPool::search_fan_out()`. Each peer runs its own local search
and returns partial results. Peers with 3 consecutive errors are removed
from the pool. Results from all responding peers are merged into the
candidate list.

**Phase 3 — Re-Rank:**
The merged candidate list is sorted by partial distance and truncated to
`k * 2`. Each candidate is re-ranked using the exact L2 distance against
the full-precision vector from local storage. Results are then sorted by
exact distance and truncated to the final top-k.

```
Client                  Coordinator              Peer Pool           Local Index
  │                         │                         │                   │
  │──── Search(query,k)────→│                         │                   │
  │                         │──── Phase 1: ───────────│──────────────────→│
  │                         │    local dim groups      │                   │
  │                         │←── partial results ─────│───────────────────│
  │                         │                         │                   │
  │                         │──── Phase 2: ──────────→│                   │
  │                         │    peer fan-out          │                   │
  │                         │←── peer results ────────│                   │
  │                         │                         │                   │
  │                         │──── Phase 3: ───────────│──────────────────→│
  │                         │    re-rank (full vecs)    │                   │
  │                         │                         │                   │
  │←── top-k results ───────│                         │                   │
```

### Peer Management

**PeerPool** — maintains a `HashMap<String, PeerClient>` of gRPC connections
to known peer nodes. The pool is refreshed periodically:

- `refresh()` reconciles the pool with the current healthy peer list: drops
  removed peers, connects to new peers.
- `search_fan_out()` sends the query to all connected peers, aggregates
  results, and tracks error counts per peer.
- A peer is automatically removed after 3 consecutive errors (circuit-breaker
  lite).

**PeerClient** — wraps a `rekha_client::RekhaClient` with:
- Lazy connection: the client is created when the peer is first used.
- Error tracking: each failed gRPC call increments the error counter.
- Auto-reconnect: on the next `refresh()`, dropped peers can be reconnected.

**Health Monitoring:**
- `Coordinator::check_peer_health()` marks peers that haven't sent a heartbeat
  in `PEER_TIMEOUT` (10s) as `Unreachable`.
- When an Unreachable peer sends a heartbeat again, it's restored to `Healthy`.
- The `heartbeat` gRPC handler on each node registers/updates the sender's
  `NodeInfo` with a current timestamp.

---

## 8. Client SDKs

### Rust Client (`rekha-client`)

The canonical Rust SDK with full retry logic, TLS, and topology discovery.

```rust
use rekha_client::RekhaClient;

let client = RekhaClient::connect(&["localhost:50051".into()]).await?;
client.insert(42, vec![0.1, 0.2, 0.3], Some(b"payload".to_vec())).await?;
let results = client.search(vec![0.1, 0.2, 0.3], 10).await?;
```

### Python Client (`rekha-python/src/rekha/`)

Pure Python gRPC client (no PyO3). Uses `grpcio` and generated proto stubs.

```python
from rekha import RekhaClient

with RekhaClient.connect(["localhost:50051"]) as client:
    # Insert (id=0 means auto-generate)
    client.insert([0.1, 0.2, 0.3], "default", payload=b'{"k":"v"}')

    # Insert with explicit ID
    client.insert([0.1, 0.2, 0.3], "default", id=42, payload=b'{"k":"v"}')

    results = client.search([0.1, 0.2, 0.3], top_k=10, collection_name="default")
    for r in results:
        print(f"id={r.id}, score={r.score}")

    # Streaming search
    for r in client.search_stream([0.1, 0.2, 0.3], 10, "default"):
        print(f"streamed: id={r.id}, score={r.score}")

    # Cluster info
    topology = client.cluster_info()
    print(f"Cluster: {topology.cluster_id}")
```

**API surface:**

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

**Retry behavior:** Exponential backoff + jitter (100ms base, doubles per attempt, up to 3 retries by default). Only retries on `UNAVAILABLE`-style errors; fatal errors (e.g. `INVALID_ARGUMENT`) propagate immediately.

### Rust SDK

```rust
use rekha_client::RekhaClient;

let client = RekhaClient::connect(&["localhost:50051"]).await?;

// Insert
client.insert(42, vec![0.1, 0.2, 0.3], Some(b"...")).await?;

// Search
let results = client.search(vec![0.1, 0.2, 0.3], 10).await?;

// Advanced search
let (results, stats) = client.search_with_params(
    query,
    10,
    SearchParams { ef_search: 200, include_payloads: true, ..default() },
).await?;
```

### Python SDK

```python
import rekha

client = rekha.connect("localhost:50051")

client.insert(42, [0.1, 0.2, 0.3], payload={"title": "..."})

results = client.search([0.1, 0.2, 0.3], top_k=10)
for r in results:
    print(r.id, r.score)
```

### Automatic Retry
All mutating and query gRPC calls use `with_retry()` internally:

- **Exponential backoff**: base delay = `100ms × 2^attempt` (100ms, 200ms, 400ms)
- **Jitter**: randomized `±25%` of the base delay per attempt to avoid
  thundering herd (derived from system time nanos).
- **Max retries**: 3 (configurable via `ClientConfig::max_retries`).
- **Fatal errors**: non-retryable gRPC status codes (e.g., `INVALID_ARGUMENT`)
  propagate immediately without retry.
- **Circuit breaker (planned)**: 5 failures in 30s → open for 60s.

### Connection Options

The client supports `connect_with_config(seeds, config)` for advanced setup:

```rust
let config = ClientConfig {
    connect_timeout: Duration::from_secs(5),
    request_timeout: Duration::from_secs(30),
    max_retries: 5,
    max_connections: 50,
    use_tls: true,
    ca_cert: Some(pem_bytes),
};
let client = RekhaClient::connect_with_config(&seeds, config).await?;
```

### TLS Support

- **`use_tls`**: when true, the client uses `https` scheme and configures
  `ClientTlsConfig` via tonic/rustls.
- **`ca_cert`**: optional PEM-encoded CA certificate for custom TLS roots
  (used instead of system roots).

---

## 9. Error Handling

### Philosophy

Every operation returns `Result<T, RekhaError>`. There are no panics in the
Rust API. The error type is:
- **Descriptive**: Human-readable error messages with context
- **Chained**: `source()` links to underlying errors
- **Cloneable**: Can be shared across async boundaries
- **Mappable**: Automatically converts to gRPC status codes

### Error Mapping to gRPC

| `RekhaError` | gRPC Status |
|---|---|
| `NotFound` | `NOT_FOUND` |
| `InvalidArgument` / `InvalidDimension` | `INVALID_ARGUMENT` |
| `IndexFull` | `RESOURCE_EXHAUSTED` |
| `Timeout` | `DEADLINE_EXCEEDED` |
| `Unavailable` / `Consensus(NotLeader)` | `UNAVAILABLE` / `FAILED_PRECONDITION` |
| All others | `INTERNAL` |

### Node-Level Fault Tolerance

| Scenario | Behavior |
|---|---|
| Node unreachable | Mark unhealthy, retry ×3, then circuit-break |
| Raft leader failure | Election in 300-500ms, client redirected |
| Disk full | Pre-allocate storage, fail writes with clear error |
| Index corruption | RocksDB WAL + snapshot recovery |
| Partial search failure | Return results with `stats.warnings` |
| Network partition | Raft maintains quorum; minority partition stalls |

---

## 10. Observability

### Metrics (Prometheus)
- `rekha_vectors_total` — indexed vector count
- `rekha_search_latency_ms` — search latency histogram
- `rekha_insert_latency_ms` — insert latency histogram
- `rekha_raft_commits_total` — committed log entries
- `rekha_node_health` — 1=healthy, 0=unhealthy

### Tracing (Jaeger / OpenTelemetry)
- Span per search request with per-node timing breakdown
- Span per Raft log replication
- Trace IDs propagated via gRPC metadata

### Health Check Endpoint
```json
{
  "status": "ok" | "degraded" | "unhealthy",
  "node_id": "node-1",
  "partitions_owned": 12,
  "partitions_healthy": 11,
  "storage_usage_gb": 42.5,
  "raft_leader_count": 4,
  "uptime_seconds": 86400
}
```

---

## 11. Configuration Reference

```yaml
cluster:
  node_id: "node-1"
  seed_nodes: ["node-1:50051", "node-2:50051", "node-3:50051"]
  bind_addr: "0.0.0.0:50051"
  data_dir: "/data/rekha"

tls:
  enabled: false
  cert_path: ~
  key_path: ~
  ca_cert_path: ~

partition:
  num_vector_shards: 16
  replication_factor: 3
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
  max_payload_size: 1048576  # 1MB
  max_inline_size: 1048576

observability:
  metrics: "prometheus"
  tracing: "jaeger"
  logging: "structured"
```

---

## Implementation Status

| Module | Status | Notes |
|---|---|---|
| Core types, errors, math | ✅ | 36 error variants, 3 distance metrics, serde on all types |
| RocksDB storage | ✅ | 4 CFs, WriteBatch, WAL flush on drop, configurable max payload |
| Vamana + PQ index | ✅ | Build, search, dim-range search, early-stop, memory usage tracking |
| Multi-granularity partition | ✅ | Strategy + manager, topology rebuild, healthy-node filtering |
| Raft consensus | ✅ | Leader election, log replication, persistence via RaftLogStore, timer loops |
| gRPC service handlers | ✅ | Insert, delete, fetch, search (streaming + unary), handshake, heartbeat, Raft AppendEntries/Vote |
| Coordinator / Search | ✅ | Three-phase: local fan-out + peer fan-out + re-rank with exact vectors |
| Peer management | ✅ | PeerPool with auto-connect/remove, health checks via heartbeat timeout |
| TLS support | ✅ | rustls, optional, mTLS with CA cert verification |
| Client SDK (Rust) | ✅ | Multi-seed connect, retry + jitter, TLS, configurable timeouts |
| CLI tool | ✅ | Server start, insert, search, delete |
| Graceful shutdown | ✅ | ctrl+c handler via serve_with_shutdown |
| Python SDK | ✅ | Pure gRPC client (no PyO3). Insert, search, delete, fetch, cluster_info, streaming |
| Circuit breaker | 🔜 | Phase 5 |
| Metrics / tracing | 🔜 | Planned
