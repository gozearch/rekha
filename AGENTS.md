# Rekha — Agent Instructions

High-performance distributed vector database (Vamana + PQ, Raft, gRPC) in Rust.

## Workspace

9 crates in a Cargo workspace. `rekha-bench` (Criterion) is excluded from default-members — most workspace commands must explicitly exclude it.

**Dependency flow**: `rekha-core` (zero deps) → `rekha-storage`, `rekha-partition` → `rekha-index`, `rekha-raft` → `rekha-server`, `rekha-client`, `rekha-cli`

**Only binary crate**: `rekha-cli` (CLI tool). The server is a library crate — it's wired to the CLI binary via the `server` subcommand.

**Entrypoints**:
- `rekha-cli/src/main.rs` — CLI binary entry
- `rekha-server/src/server.rs` — `ServerInstance` (loads config, opens RocksDB, starts gRPC)
- `rekha-server/src/service.rs` — gRPC service handlers
- `rekha-server/src/coordinator.rs` — query coordinator (search fan-out + merge)
- `rekha-server/src/config.rs` — `ServerConfig` (loads YAML)
- `proto/rekha.proto` — all protobuf definitions; codegen via `tonic-build` in build.rs

## Developer Commands

All via `justfile` (requires `cargo install just`):

| Command | What it runs |
|---|---|
| `just test` | `cargo test --workspace --exclude rekha-bench` |
| `just lint` | `cargo fmt --all --check && cargo clippy -- -- -D warnings` (same exclude) |
| `just fix` | `cargo fmt --all` |
| `just check` | `cargo check --workspace --exclude rekha-bench` |
| `just build` | `cargo build ...` (same exclude) |
| `just release-build` | `cargo build --release ...` (same exclude) |
| `just test-name <name>` | `cargo test <name> --workspace --exclude rekha-bench` |
| `just coverage` | `cargo llvm-cov ...` (requires nightly + cargo-llvm-cov) |
| `just coverage-html` | generates HTML report, opens in browser |

**CI order** (`.github/workflows/ci.yml`): `lint → test (ubuntu + macos) → coverage (nightly, ubuntu-only)`

## Testing Conventions

- Unit tests are inline next to code (`#[cfg(test)] mod tests`).
- Tests requiring real gRPC server are marked `#[ignore]` — run with `cargo test -- --ignored`.
- RocksDB tests use unique temp directories via `AtomicU64` counters to avoid collisions.
- Test utilities: `tempfile`, `proptest`, `approx` (0.5), `rand`.
- Benchmark: `rekha-bench` uses Criterion with async Tokio support.

## Architecture Notes

- **Error handling**: Layered error types (`StorageError`, `IndexError`, `PartitionError`, `RaftError`) all funnel through `RekhaError` via `From` impls. All errors are `Clone` (not just `Debug`) for async propagation.
- **Storage**: RocksDB with 4 column families: `vectors` (f32 LE bytes), `payloads` (arbitrary bytes), `metadata` (JSON), `raft_log` (protobuf). Vector IDs encoded as u64 big-endian keys.
- **Search flow**: coordinator fans out query to dimension groups → partial distance with early-stop → merge → re-rank with full-precision vectors.
- **Raft**: custom implementation (not `openraft`/`raft-rs`). Per-shard Raft groups.
- **Config**: YAML, loaded by `ServerConfig::from_file()`, env var `REKHA_CONFIG`. Dev defaults via `ServerConfig::dev_default()`.
- **TLS**: optional, via `rustls` (pure Rust). `TlsConfig` is part of `ServerConfig`.

## CLI Usage

```
rekha server --config config.yaml    # start server
echo "0.1 0.2 0.3" | rekha insert 42  # insert, vector from stdin
rekha insert 42 --payload '{"k":"v"}'  # with payload
echo "0.1 0.2 0.3" | rekha search -k 10
rekha delete 42 43 44
```

## Code Style

- No comments in production code.
- `Serialize`/`Deserialize` derive on all data types.
- `async-trait` for async trait methods.
- Manual `Display` + `Error` impls (not `thiserror` derive).
