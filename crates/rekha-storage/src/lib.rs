//! RekhaDB `rekha-storage` crate — durable metadata and blob storage.
//!
//! This crate implements the two persistence seams of RekhaDB's
//! **compute/storage separation** design:
//!
//! - [`catalog`] / [`Catalog`]: the transactional metadata store (tenants,
//!   databases, collections, and — in Phase 4 — the shard map). Backed by
//!   [`RedbCatalog`] today; the same trait targets Postgres in cluster mode.
//!   Think of it as Chroma's sysdb.
//! - [`storage`] / [`Storage`]: the object store where vector blocks and
//!   segments live. Backed by [`LocalStorage`] (local filesystem) today;
//!   S3/GCS in the cluster phase.
//!
//! Two design notes worth carrying into later phases:
//!
//! - **The catalog doubles as the future shard map.** Collection records are
//!   keyed by id; the reserved `segments` / `segment_metadata` tables map
//!   `(collection_id, segment_id)` to a node and log range, so "which replica
//!   owns this collection's segments" is a catalog query, not out-of-band
//!   state.
//! - **The WAL is the source of truth; indexes are derived.** `max_seq_id` per
//!   collection records how much of the log has been materialized, so WAL
//!   compaction/pruning knows it can drop records up to that watermark —
//!   avoiding Chroma's unbounded-`embeddings_queue` footgun.
//!
//! Both traits are deliberately small, **sync** (redb is sync; the async engine
//! wraps calls in `spawn_blocking`), and `Send + Sync` so they can be shared
//! behind `Arc<dyn Catalog>` / `Arc<dyn Storage>` across request handlers.

pub mod catalog;
pub mod local_storage;
pub mod redb_catalog;
pub mod storage;

pub use catalog::{Catalog, CatalogError, CatalogResult, CollectionRecord};
pub use local_storage::LocalStorage;
pub use redb_catalog::RedbCatalog;
pub use storage::{Storage, StorageError};
