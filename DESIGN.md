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
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Query Router │  │ Result Merger │  │ Topology Mgmt  │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘  │
└─────────┼─────────────────┼──────────────────┼────────────┘
          │       ┌─────────┘                  │
          ▼       ▼                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Rekha Node                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ Vamana + │  │ RocksDB  │  │ Raft     │  │Partition│  │
│  │ PQ Index │  │ Storage  │  │ Consensus│  │ Manager │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Crate Architecture (Cargo Workspace)

| Crate | Responsibility |
|---|---|
| `rekha-core` | Types, traits, errors, distance math (zero deps) |
| `rekha-storage` | RocksDB wrapper with column families |
| `rekha-index` | Vamana graph + Product Quantization |
| `rekha-partition` | Multi-granularity partition strategy |
| `rekha-raft` | Raft consensus per partition |
| `rekha-server` | gRPC server, coordinator, service handlers |
| `rekha-client` | Rust SDK (connection pool, retry, streaming) |
| `rekha-py` | Python SDK (PyO3 + maturin) |
| `rekha-cli` | Admin CLI tool |

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
| `raft_log` | Raft WAL entries | `u64 (index)` | protobuf |

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
3. Leader appends to Raft log, replicates to followers
4. Once committed (majority ack), leader applies to state machine
5. Leader responds to client

### Read Path
1. Client sends `Search` to any node
2. Node routes to any replica (leader or follower) for the partition
3. Replica runs local search on its copy of the index
4. Results are returned directly (no Raft round-trip)

---

## 7. Coordinator & Query Execution

### Search Flow (Detailed)

```
Client                  Coordinator              Node A (dim 0-191)   Node B (dim 192-383)
  │                         │                         │                      │
  │──── Search(query,k)────→│                         │                      │
  │                         │──── dim_range(0,192)───→│                      │
  │                         │──── dim_range(192,384)────────────────────────→│
  │                         │──── dim_range(384,576)──→│                      │
  │                         │──── dim_range(576,768)────────────────────────→│
  │                         │                         │                      │
  │                         │←── partial_results ─────│                      │
  │                         │←── partial_results ────────────────────────────│
  │                         │←── partial_results ─────│                      │
  │                         │←── partial_results ────────────────────────────│
  │                         │                         │                      │
  │                         │ [Merge & early-stop]     │                      │
  │                         │ [Re-rank with full vecs] │                      │
  │                         │ [Fetch payloads]         │                      │
  │                         │                         │                      │
  │←── top-k results ───────│                         │                      │
```

### Early-Stop During Merge
- Coordinator maintains a min-heap of the current top-k results
- As each dimension group reports partial distances, candidates whose
  partial distance already exceeds the k-th best are discarded
- This reduces network transfer and re-ranking cost

---

## 8. Client SDKs

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
- Exponential backoff: 100ms, 200ms, 400ms, 800ms (max 3 retries)
- Jitter added to avoid thundering herd
- Circuit breaker: 5 failures in 30s → open for 60s

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

| Phase | Module | Status |
|---|---|---|
| 1 | Core types, errors, math | ✅ Built |
| 2 | RocksDB storage | ✅ Built |
| 3 | Vamana graph + PQ index | ✅ Built |
| 4 | Multi-granularity partition | ✅ Built |
| 5 | Raft consensus | ✅ Built |
| 6 | Protobuf definitions | ✅ Built |
| 7 | gRPC server + coordinator | ✅ Built |
| 8 | Rust client SDK | ✅ Built |
| 9 | Python SDK (PyO3) | ✅ Built |
| 10 | CLI tool | ✅ Built |
| 11 | Integration + hardening | 🔜 Next |
