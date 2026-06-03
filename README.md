<p align="center">
  <img src="assets/logo.jpg" width="300" alt="Rekha Logo">
</p>

<h1 align="center">Rekha</h1>

<p align="center">
  <strong>High-performance, distributed vector database built for billion-scale ANNS.</strong>
</p>

<p align="center">
  <a href="https://github.com/user/rekha/actions/workflows/ci.yml">
    <img src="https://github.com/user/rekha/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
  <img src="https://img.shields.io/badge/rust-1.75%2B-orange.svg" alt="Rust Version">
</p>

---

Rekha is a distributed vector database designed for billion-scale Approximate Nearest Neighbor Search (ANNS). Built from the ground up in Rust, it utilizes a DiskANN-style Vamana index combined with Product Quantization (PQ) to achieve high performance with a low memory footprint, making it suitable for both SSD and HDD storage.

## ✨ Key Features

- **Disk-First Indexing**: Vamana graph and PQ codes optimized for sequential disk access and minimal random I/O.
- **Multi-Granularity Partitioning**: Combines horizontal vector sharding with vertical dimension-based partitioning for efficient query pruning.
- **Strong Consistency**: Per-partition Raft consensus ensuring linearizable writes and high availability.
- **Payload Support**: Store and fetch arbitrary metadata alongside vectors with lazy loading during search.
- **gRPC Native**: Robust communication layer built with Tonic, featuring built-in TLS and auto-discovery.

## 📦 Architecture

Rekha is organized into a modular workspace:

| Crate | Responsibility |
|---|---|
| `rekha-core` | Core types, distance metrics, and error handling. |
| `rekha-server` | gRPC server, query coordinator, and service logic. |
| `rekha-client` | Rust SDK with connection pooling and retries. |
| `rekha-index` | Vamana graph and Product Quantization implementation. |
| `rekha-storage` | RocksDB-backed persistent storage layer. |
| `rekha-raft` | Custom Raft implementation for partition replication. |
| `rekha-partition` | Partition management and routing strategies. |

## 🚀 Quick Start

### Prerequisites

- Rust 1.75+
- [Just](https://github.com/casey/just) (optional, for development tasks)

### Installation

```bash
git clone https://github.com/user/rekha.git
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
- [**Design Specification**](DESIGN.md) — Deep dive into the architecture, partitioning strategy, and index internals.
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
