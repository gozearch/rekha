# Rekha — Agent Instructions

Distributed vector database (IVF + PQ, consistent hashing, gRPC) in Rust.

## Workspace

8 crates. `rekha-bench` excluded from default-members — always add `--exclude rekha-bench`.

**Dependency flow**: `rekha-core` → `rekha-storage` → `rekha-index` → `rekha-server` → `rekha-client` / `rekha-cli`. `rekha-partition` depends on `rekha-core` only.

**Only binary crate**: `rekha-cli`. The server is a library crate — wired via the `server` subcommand.

**Entrypoints**:
- `rekha-cli/src/main.rs` — CLI binary
- `rekha-server/src/server.rs` — `ServerInstance` (opens RocksDB, starts gRPC)
- `rekha-server/src/coordinator.rs` — query routing, replication, collection management
- `rekha-server/src/service.rs` — gRPC handlers
- `rekha-server/src/config.rs` — `ServerConfig` (infrastructure-only: cluster, tls, storage, observability)
- `proto/rekha.proto` — all protobuf; codegen via `tonic-build` in `rekha-server/build.rs` and `rekha-client/build.rs`
- `rekha-index/src/index.rs` — `RekhaIndex`: collection registry holding `HashMap<String, CollectionState>`, each with its own `IvfIndex`
- `rekha-partition/src/ring.rs` — `ConsistentHashRing`: 128 vnodes, SipHash, `BTreeMap`-backed `replicas_for(shard, rf)`

## Developer Commands

All Cargo commands must exclude `rekha-bench`. Use the exact forms below:

```bash
cargo test --workspace --exclude rekha-bench
cargo clippy --workspace --exclude rekha-bench -- -D warnings
cargo fmt --all --check
cargo check --workspace --exclude rekha-bench
cargo build --workspace --exclude rekha-bench
```

If `just` is installed, `just test`, `just lint`, `just check`, `just build` run the equivalents.

Run a specific test:
```bash
cargo test -p rekha-server -- coordinator::tests::test_coordinator_insert
```

## CI

- `.github/workflows/test.yml` — runs `cargo test` on every push to any branch.
- `.github/workflows/ci.yml` — lint + test + coverage on master pushes only.

## Architecture Notes

**No Raft or Vamana.** Raft was removed entirely (`rekha-raft/` deleted). Vamana graphs replaced by IVF inverted file index (`IvfIndex`). The dimension pipeline and planner modules were deleted. Search is always vector-based IVF.

**Storage**: RocksDB with 3 column families: `vectors`, `payloads`, `metadata`. Vector keys are collection-namespaced: `{collection}\0{u64 BE id}`. The `metadata` CF stores collection configs at key `collection:{name}` as serialized `CollectionConfig` JSON. The open path auto-discovers existing CFs for backward compatibility with old databases.

**Collection management**: Collections are created at runtime via `rekha create-collection -c NAME --rf N --config '{"dim":256,...}'`. Each collection gets its own `IvfIndex` (centroids, inverted lists, PQ). Multiple collections with different dimensions coexist in the same node.

**Replication**: `coordinator.insert()` writes locally, then calls `partition_manager.replicas_for(shard, rf)` to forward to RF peers via gRPC. Receiver uses `is_replication` flag on proto to prevent re-replication loops. Collection creation broadcasts to all peers. Default `replication_factor` is 1 (no replication) unless `--rf` specified.

**Consistent hashing**: `PartitionManager` wraps `ConsistentHashRing`. `replicas_for(shard, rf)` skips unhealthy nodes. Adding/removing nodes reassigns ~1/N of shards (not all).

**Error handling**: Layered — `StorageError`, `IndexError`, `PartitionError` funnel through `RekhaError` via `From` impls. All errors are `Clone`. No `RaftError` (removed).

**Config**: YAML, loaded by `ServerConfig::from_file()`. Infrastructure-only — no `partition`, `index`, `planner`, or `raft` sections. Run `rekha server --config config.yaml`.

**Proto**: Codegen in two places — `rekha-server/build.rs` (server + client stubs) and `rekha-client/build.rs` (client stubs only). They produce **different Rust types** with identical field layouts. When mapping server's `crate::proto` to client's `rekha_client::proto`, convert field-by-field.

## Testing Conventions

- Unit tests inline next to code (`#[cfg(test)] mod tests`).
- RocksDB tests use unique temp dirs via `AtomicU64` counters to avoid collisions.
- Tests that create coordinators without calling `initialize()` will fail on insert/search because no default collection exists — use `coord.initialize(index).await` first.
- Test utilities: `tempfile`, `proptest`, `approx` (0.5), `rand`.
- Benchmark: `rekha-bench` uses Criterion.

## Docker + E2E

```bash
# Build fresh image (skip cache due to Docker layer caching quirks)
docker compose build --no-cache

# Start 3-node cluster
docker compose up -d --wait

# Run production E2E test (collection create, insert, search, failover, restart)
./scripts/e2e_prod.sh
```

## CLI Usage

```
rekha server --config config.yaml
rekha create-collection -c images --rf 3 --config '{"dim":256,"nlist":4096,"nprobe":32}'
rekha list-collections
rekha collection-exists -c images
echo "0.1 0.2 ..." | rekha insert -c images
echo "0.5 0.5 ..." | rekha search -c images -k 10
rekha delete 42 43 44
rekha health
```

Vector input is space-separated floats from stdin.

## Code Style

- No comments in production code.
- `Serialize`/`Deserialize` derive on all data types.
- Manual `Display` + `Error` impls (not `thiserror` derive).
- `#[allow(clippy::too_many_arguments)]` on functions with 7+ args.
