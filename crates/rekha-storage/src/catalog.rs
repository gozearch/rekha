//! Catalog: the transactional metadata store behind RekhaDB.
//!
//! The catalog owns everything that must be queryable by name and consistent
//! across crashes: tenants, databases, collections, and (in a later phase) the
//! shard map. It is the same "sysdb" role Chroma gives its Postgres-backed
//! metadata store.
//!
//! Two design notes:
//!
//! - **The catalog doubles as the future shard map.** A `CollectionRecord` is
//!   keyed by collection id; Phase 4 will add `segments` / `segment_metadata`
//!   tables that map `(collection_id, segment_id)` to a node and log range.
//!   The `Catalog` trait is intentionally the seam: redb today, Postgres in
//!   cluster mode later.
//! - **`max_seq_id` is the compaction watermark.** The WAL is the source of
//!   truth and indexes are derived, so compaction must know how much of a
//!   collection's log has already been materialized. [`CollectionRecord`]
//!   tracks that watermark (Chroma's `index_metadata.max_seq_id`) so RekhaDB
//!   does not inherit Chroma's unbounded-`embeddings_queue` footgun.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use rekha_core::config::CollectionConfig;
use rekha_core::types::Metadata;

/// A fully materialized collection record as stored in the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionRecord {
    /// The collection definition: tenancy, naming, dimension, space, metadata,
    /// and HNSW tuning.
    pub config: CollectionConfig,
    /// Highest WAL seq that has been materialized into the collection's
    /// indexes (Chroma `index_metadata.max_seq_id` analog). WAL compaction may
    /// prune records at or below this watermark.
    pub max_seq_id: u64,
    /// Cumulative number of elements ever added (Chroma
    /// `total_elements_added` analog).
    pub total_elements: u64,
}

/// Errors returned by the catalog.
#[derive(Debug, Error)]
pub enum CatalogError {
    /// The requested tenant, database, or collection does not exist.
    #[error("not found")]
    NotFound,
    /// A tenant, database, or collection with the same identity already exists.
    #[error("already exists")]
    AlreadyExists,
    /// The caller supplied a malformed identity (empty name, ...).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// The underlying transactional store failed.
    #[error("storage error: {0}")]
    Storage(#[from] redb::Error),
    /// A record could not be (de)serialized.
    #[error("serialization error: {0}")]
    Serialization(String),
}

impl From<redb::DatabaseError> for CatalogError {
    fn from(e: redb::DatabaseError) -> Self {
        CatalogError::Storage(redb::Error::from(e))
    }
}

impl From<redb::TransactionError> for CatalogError {
    fn from(e: redb::TransactionError) -> Self {
        CatalogError::Storage(redb::Error::from(e))
    }
}

impl From<redb::TableError> for CatalogError {
    fn from(e: redb::TableError) -> Self {
        CatalogError::Storage(redb::Error::from(e))
    }
}

impl From<redb::CommitError> for CatalogError {
    fn from(e: redb::CommitError) -> Self {
        CatalogError::Storage(redb::Error::from(e))
    }
}

impl From<redb::StorageError> for CatalogError {
    fn from(e: redb::StorageError) -> Self {
        CatalogError::Storage(redb::Error::from(e))
    }
}

/// Convenience alias for catalog operations.
pub type CatalogResult<T> = Result<T, CatalogError>;

/// Transactional metadata store for tenants, databases, collections, and (in a
/// later phase) the shard map.
///
/// Implementations must be cheaply shareable (`&self` for every operation) so
/// the engine can hand out `Arc<dyn Catalog>` to request handlers. The trait is
/// deliberately small and sync: redb is a sync, single-writer store, and the
/// engine will wrap calls in `spawn_blocking` for the async API.
pub trait Catalog: Send + Sync {
    /// Create a tenant namespace. Errors with [`CatalogError::AlreadyExists`]
    /// if the tenant already exists.
    fn create_tenant(&self, name: &str) -> Result<(), CatalogError>;

    /// Create a database inside a tenant. Errors with
    /// [`CatalogError::NotFound`] if the tenant is missing and
    /// [`CatalogError::AlreadyExists`] if the database already exists.
    fn create_database(&self, tenant: &str, name: &str) -> Result<(), CatalogError>;

    /// Create a collection. The name must be unique within `(tenant, database)`;
    /// errors with [`CatalogError::AlreadyExists`] otherwise. The returned
    /// record starts with `max_seq_id = 0` and `total_elements = 0`.
    fn create_collection(
        &self,
        config: &CollectionConfig,
    ) -> Result<CollectionRecord, CatalogError>;

    /// Look up a collection by `(tenant, database, name)`.
    fn get_collection(
        &self,
        tenant: &str,
        database: &str,
        name: &str,
    ) -> Result<Option<CollectionRecord>, CatalogError>;

    /// Look up a collection by id.
    fn get_collection_by_id(&self, id: &Uuid) -> Result<Option<CollectionRecord>, CatalogError>;

    /// List all collections in a `(tenant, database)` scope.
    fn list_collections(
        &self,
        tenant: &str,
        database: &str,
    ) -> Result<Vec<CollectionRecord>, CatalogError>;

    /// List every collection across all tenants and databases. The engine uses
    /// this to discover what to reopen after a restart (Phase 4a replaces its
    /// WAL-directory scan with this catalog query).
    fn list_collections_all(&self) -> Result<Vec<CollectionRecord>, CatalogError>;

    /// Replace a collection's metadata with the given map.
    fn update_collection_metadata(
        &self,
        id: &Uuid,
        metadata: &Metadata,
    ) -> Result<(), CatalogError>;

    /// Remove a collection and its name entry.
    fn delete_collection(&self, id: &Uuid) -> Result<(), CatalogError>;

    /// Advance the materialized log offset (Chroma's compaction register step).
    /// The stored value is the max of the current value and `seq`, so a
    /// duplicate compaction register can never move the watermark backwards.
    fn advance_log_offset(&self, id: &Uuid, seq: u64) -> Result<(), CatalogError>;
}
