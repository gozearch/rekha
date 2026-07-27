# Rekha — Agent Instructions

Distributed vector database (IVF + PQ, Dynamo-style replication, consistent hashing, gRPC) in Rust.

## Build prerequisite

`protoc` **27.5** must be on PATH (CI pins `PROTOC_VERSION: "27.5"`). Without it, `rekha-proto`'s `tonic-build` codegen fails.

## Workspace

12 workspace crates. `rekha-python/` is **not** a workspace member — it has its own `pyproject.toml` and generated stubs.

`rekha-bench` is in `default-members` but excluded from every test/lint/check command (Criterion suite is slow). Always include `--exclude rekha-bench`.

**Crates**:
- `rekha-core` — types, traits, distance metrics, errors, `ConsistencyLevel`, `CollectionConfig`.
- `rekha-proto` — **single** proto source. `build.rs` runs `tonic-build`; `src/proto` re-exports generated types; `src/conversions.rs` holds shared proto↔core converters. Server, client, coordinator reuse `rekha_proto::proto` — do **not** field-by-field convert across "different proto types" (that pattern was removed).
- `rekha-storage` — RocksDB. 4 CFs: `vectors`, `payloads`, `metadata`, `hints`. Hint storage lives in `hint_store.rs` (`HintStore`); reach via `store.hint_store()`.
- `rekha-index` — `RekhaIndex` (collection registry → `IvfIndex` each), `ivf.rs`.
- `rekha-quant` — Product quantization + K-Means clustering.
- `rekha-replication` — LWW timestamps (`lww.rs`), consistency gate ONE/QUORUM/ALL (`consistency_gate.rs`), hinted handoff (`hinted_handoff.rs`).
- `rekha-cluster` — membership, peer-state, consistent hash ring (`ring.rs`: 128 vnodes, SipHash, `BTreeMap`-backed `replicas_for`).
- `rekha-coordinator` — write/read path orchestration, collection DDL, peer pool. Split into `coordinator.rs`, `write_path.rs`, `read_path.rs`, `collection.rs`, `peer_pool.rs`, `membership.rs`.
- `rekha-server` — gRPC server + config. Re-exports `rekha_proto::proto` and `rekha_coordinator::Coordinator`.
- `rekha-client` — Rust SDK (gRPC client, retries, cluster mgmt).
- `rekha-cli` — **only binary crate**; server runs via `server` subcommand.

**Dependency flow**: `rekha-core` → `rekha-proto` / `rekha-storage` / `rekha-index` / `rekha-quant` / `rekha-replication` / `rekha-cluster` → `rekha-coordinator` → `rekha-server` → `rekha-client` → `rekha-cli`.

**Entrypoints**:
- `rekha-cli/src/main.rs` — CLI binary
- `rekha-server/src/server.rs` — `ServerInstance` (opens RocksDB, starts gRPC)
- `rekha-coordinator/src/coordinator.rs` — `Coordinator` (write/read/query router; *not* `rekha-server/src/coordinator.rs` — that file no longer exists)
- `rekha-server/src/service.rs` — gRPC handlers
- `rekha-server/src/config.rs` — `ServerConfig`
- `proto/rekha.proto` — proto source of truth
- `rekha-index/src/index.rs` — `RekhaIndex` collection registry

## Developer Commands

All cargo commands exclude `rekha-bench`:

```bash
cargo test --workspace --exclude rekha-bench
cargo clippy --workspace --exclude rekha-bench -- -D warnings
cargo fmt --all --check
cargo fmt --all
cargo check --workspace --exclude rekha-bench
cargo build --workspace --exclude rekha-bench
```

`just` aliases: `just test`, `just lint`, `just check`, `just build`, `just fix`, `just coverage`, `just test-name <name>`, `just coverage-html` (needs nightly + `cargo-llvm-cov`).

Run one test:
```bash
cargo test -p rekha-coordinator -- coordinator::tests::test_coordinator_insert
```

## CI

- `test.yml` — `cargo test` on every push.
- `ci.yml` — lint + test + coverage, on push **and** PR to `master`. Pins protoc 27.5.
- `docker.yml` / `opencode.yml` — image build and agent infra.

## Architecture Notes


IVF inverted-file index (`IvfIndex`). Do not reintroduce `RaftError`, planner, or dimension-pipeline concepts.

**Storage**: RocksDB, 4 CFs: `vectors`, `payloads`, `metadata`, `hints`. Vector keys: `{collection}\0{u64 BE id}`. `metadata` CF stores collection configs at `collection:{name}` as `CollectionMeta` JSON. Open path auto-discovers existing CFs.

**Collection management**: runtime via `rekha create-collection -c NAME --rf N --config '{"dim":256,...}'`. Each collection gets its own `IvfIndex`. `Coordinator::initialize()` auto-creates a default `dim=8` collection if none exists.

**Replication (Dynamo-style)**: local write → consistency gate (ONE/QUORUM/ALL) → forward to RF peers via `replicas_for(shard, rf)` → LWW timestamp reconcile. Receiver sets `is_replication = true` on the forwarded proto request to prevent re-replication loops. Hinted handoff stores writes for unreachable peers (`hinted_handoff_enabled`, `max_hint_window_secs`). Default write consistency is **QUORUM**; `--rf` defaults to 1.

**Consistent hashing**: `rekha-cluster::Membership` wraps the ring. `replicas_for(shard, rf)` skips unhealthy nodes. Covers ~1/N shards per new/departed node.

**Error handling**: `StorageError` / `IndexError` / `PartitionError` → `RekhaError` via `From`. All `Clone`. No `RaftError`.

**Config**: YAML, `ServerConfig::from_file()`. Sections: `cluster` (`default_write_consistency`, `hinted_handoff_enabled`, `max_hint_window_secs`), `tls`, `observability`, `storage` (`max_payload_size`, `max_inline_size`, `gc_grace_seconds`). Run `rekha server --config config.yaml`.

## Testing

- Inline `#[cfg(test)] mod tests`. Integration tests at `rekha-server/tests/integration.rs`.
- RocksDB tests use unique temp dirs via `AtomicU64` counter to avoid collisions.
- `Coordinator` will fail on insert/search unless `coord.initialize(index).await` is called first.
- Test utilities: `tempfile`, `proptest`, `approx` (0.5), `rand`.
- `rekha-bench` uses Criterion — reason for `--exclude rekha-bench`.

## Docker + E2E

```bash
docker compose build --no-cache
docker compose up -d
./scripts/e2e_prod.sh
./scripts/e2e_test.sh
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
- Manual `Display` + `Error` impls (not `thiserror`).
- `#[allow(clippy::too_many_arguments)]` on functions with 7+ args.
