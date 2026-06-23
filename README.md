<p align="center">
  <img src="assets/logo.jpg" width="300" alt="Rekha Logo">
</p>

<h1 align="center">Rekha</h1>

<p align="center">
  <strong>High-performance, distributed vector database built for billion-scale ANNS.</strong>
</p>

<p align="center">
  <a href="https://github.com/gozearch/rekha/actions/workflows/ci.yml">
    <img src="https://github.com/gozearch/rekha/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  </a>
  <a href="https://codecov.io/gh/gozearch/rekha">
    <img src="https://codecov.io/gh/gozearch/rekha/branch/master/graph/badge.svg" alt="Code Coverage">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
  <img src="https://img.shields.io/badge/rust-1.80%2B-orange.svg" alt="Rust Version">
</p>

---

Rekha is a distributed vector database designed for billion-scale Approximate Nearest Neighbor Search (ANNS). Built from the ground up in Rust, it utilizes an IVF (inverted file) index combined with Product Quantization (PQ) to achieve high performance with a low memory footprint, making it suitable for both SSD and HDD storage.

## ✨ Key Features

- **IVF + PQ Indexing**: Inverted-file index with Product Quantization compression for efficient approximate search and low memory footprint.
- **Consistent Hash Routing**: Dynamo-style consistent hashing distributes shards across nodes with minimal redistribution on topology changes.
- **Tunable Consistency**: ONE/QUORUM/ALL consistency levels with LWW timestamp conflict resolution and hinted handoff replication.
- **Payload Support**: Store and fetch arbitrary metadata alongside vectors with lazy loading during search.
- **gRPC Native**: Robust communication layer built with Tonic, featuring built-in TLS and auto-discovery.

## 🗺 Roadmap

These features are planned but not yet implemented:

- **Shard transfer**: Full implementation with centroid migration and receiver-side index rebuild.
- **Collection repair**: Automatic repair of inconsistent replicas.
- **Live checkpointing**: RocksDB snapshot-based backup without node shutdown.
- **Connection pooling**: Multi-channel client with load balancing across seeds.

## 📦 Architecture

Rekha is organized into a modular workspace:

| Crate | Responsibility |
|---|---|---|
| `rekha-core` | Core types, traits, distance metrics, and error handling. |
| `rekha-storage` | RocksDB-backed persistent storage (4 column families). |
| `rekha-index` | IVF index and Product Quantization implementation. |
| `rekha-quant` | Product quantization and K-Means clustering. |
| `rekha-replication` | LWW timestamps, quorum consistency gate, hinted handoff. |
| `rekha-cluster` | Cluster membership, consistent hash ring, peer state tracking. |
| `rekha-coordinator` | Write/read path orchestration, collection DDL, peer pool. |
| `rekha-server` | gRPC server, service handlers, and configuration loading. |
| `rekha-client` | Rust SDK with gRPC client, retries, and cluster management. |
| `rekha-proto` | Shared protobuf types and protocol↔core converters. |
| `rekha-cli` | Command-line admin interface. |

## 🚀 Quick Start

### Prerequisites

- Rust 1.80+
- [Just](https://github.com/casey/just) (optional, for development tasks)

### Installation

```bash
git clone https://github.com/gozearch/rekha.git
cd rekha
cargo build --release
```

### Run a Node

```bash
# Start a single-node server with default dev config
./target/release/rekha server --config config.yaml
```

## 📖 Documentation

For detailed guides and deep dives, see the following:

- [**Operations & Deployment**](docs/README.md) — How to run Rekha in production, TLS setup, and configuration reference.
- [**Client SDK Guide**](docs/README.md#5-client-sdks) — Using Rekha from Rust or Python.

## 🛠 Development

Common tasks are managed via `justfile`:

```bash
just test          # Run all tests
just lint          # Run clippy and fmt
just coverage      # Generate coverage report
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
