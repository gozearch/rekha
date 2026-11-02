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

use rekha_core::cluster::ShardId;
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
    /// The shard whose replicas hold this collection's data.
    ///
    /// `None` means "not sharded": a single-node deployment, where the one
    /// engine holds everything. A cluster always writes `Some`, and a cluster
    /// that reads `None` refuses rather than guessing which node owns it.
    ///
    /// This is the *only* record of the placement decision. It is on the record
    /// rather than derived from the collection id because it is a decision, not
    /// a hash: the metadata leader chooses the shard for load, and routing
    /// reads it back. Deriving it would let routing and storage disagree the
    /// moment the voter set changed, and the data would not move with it.
    pub shard: Option<ShardId>,
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
    ///
    /// `shard` is the placement decision, recorded durably so routing never has
    /// to recompute it.
    fn create_collection(
        &self,
        config: &CollectionConfig,
        shard: Option<ShardId>,
    ) -> Result<CollectionRecord, CatalogError>;

    /// Record a collection in the catalog without materializing it.
    ///
    /// Idempotent: a re-applied `AddCollection` entry, or a node that already
    /// has the row, is not an error. This is the metadata half of creation;
    /// the data half is `Engine::materialize_collection`.
    fn record_collection(
        &self,
        config: &CollectionConfig,
        shard: Option<ShardId>,
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

    /// List tenant names, ascending.
    ///
    /// Needed to capture the registry for a snapshot: a node joining after log
    /// compaction has no other way to learn which namespaces exist, and an
    /// empty tenant list would make every request answer "not found".
    fn list_tenants(&self) -> CatalogResult<Vec<String>>;

    /// List `(tenant, database)` pairs, ordered by tenant then name.
    fn list_databases(&self) -> CatalogResult<Vec<(String, String)>>;

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

    /// Persist an inferred dimension (used when a collection was created with
    /// `dimension = 0` for Chroma compatibility).
    fn update_collection_dimension(&self, id: &Uuid, dimension: usize) -> Result<(), CatalogError>;

    // ------------------------------------------------------------------
    // RBAC
    // ------------------------------------------------------------------

    /// Create a user. Errors with `AlreadyExists` if the id or name already exists.
    fn create_user(&self, user: &rekha_core::auth::UserRecord) -> CatalogResult<()>;

    /// Look up a user by id.
    fn get_user(&self, id: &str) -> CatalogResult<Option<rekha_core::auth::UserRecord>>;

    /// Look up a user by name.
    fn get_user_by_name(&self, name: &str) -> CatalogResult<Option<rekha_core::auth::UserRecord>>;

    /// List all users.
    fn list_users(&self) -> CatalogResult<Vec<rekha_core::auth::UserRecord>>;

    /// Enable or disable a user.
    fn set_user_enabled(&self, id: &str, enabled: bool) -> CatalogResult<()>;

    /// Delete a user and all of their keys/assignments.
    fn delete_user(&self, id: &str) -> CatalogResult<()>;

    /// Store an API key hash for a user under a human label.
    fn create_api_key(&self, user_id: &str, key_hash: &[u8; 32], label: &str) -> CatalogResult<()>;

    /// Look up which user owns `key_hash`. Returns `(user_id, KeyMeta)` if found.
    fn lookup_api_key(
        &self,
        key_hash: &[u8; 32],
    ) -> CatalogResult<Option<(String, rekha_core::auth::KeyMeta)>>;

    /// Revoke a single API key.
    fn delete_api_key(&self, user_id: &str, key_hash: &[u8; 32]) -> CatalogResult<()>;

    /// List all keys for a user.
    fn list_user_keys(&self, user_id: &str) -> CatalogResult<Vec<rekha_core::auth::KeyMeta>>;

    /// Create a role. Errors with `AlreadyExists` if the id or name already exists.
    fn create_role(&self, role: &rekha_core::auth::RoleRecord) -> CatalogResult<()>;

    /// Look up a role by id.
    fn get_role(&self, id: &str) -> CatalogResult<Option<rekha_core::auth::RoleRecord>>;

    /// Look up a role by name.
    fn get_role_by_name(&self, name: &str) -> CatalogResult<Option<rekha_core::auth::RoleRecord>>;

    /// List all roles.
    fn list_roles(&self) -> CatalogResult<Vec<rekha_core::auth::RoleRecord>>;

    /// Delete a role. Rejects built-in roles.
    fn delete_role(&self, id: &str) -> CatalogResult<()>;

    /// Add a permission to a role.
    fn add_permission(
        &self,
        role_id: &str,
        perm: &rekha_core::auth::Permission,
    ) -> CatalogResult<()>;

    /// Remove a permission from a role.
    fn remove_permission(&self, role_id: &str, perm_id: &str) -> CatalogResult<()>;

    /// List all permissions for a role.
    fn get_role_permissions(
        &self,
        role_id: &str,
    ) -> CatalogResult<Vec<rekha_core::auth::Permission>>;

    /// Assign a role to a user. Idempotent — re-assigning is a no-op.
    fn assign_role(&self, user_id: &str, role_id: &str, assigned_by: &str) -> CatalogResult<()>;

    /// Remove a role assignment.
    fn unassign_role(&self, user_id: &str, role_id: &str) -> CatalogResult<()>;

    /// List every role assignment, in id order.
    ///
    /// The assignment carries `assigned_by` and its timestamp, which
    /// [`Catalog::get_user_roles`] does not return. A snapshot that restored
    /// assignments through `get_user_roles` would rewrite audit fields to
    /// whatever the restoring node chose.
    fn list_assignments(&self) -> CatalogResult<Vec<rekha_core::auth::RoleAssignment>>;

    /// List all roles (with their permissions) for a user. Returns
    /// `Vec<(RoleRecord, Vec<Permission>)>`.
    fn get_user_roles(
        &self,
        user_id: &str,
    ) -> CatalogResult<
        Vec<(
            rekha_core::auth::RoleRecord,
            Vec<rekha_core::auth::Permission>,
        )>,
    >;

    /// Collect every permission a user holds via all of their role assignments.
    fn get_user_permissions(
        &self,
        user_id: &str,
    ) -> CatalogResult<Vec<rekha_core::auth::Permission>>;

    /// Ensure all built-in roles exist.
    fn ensure_builtin_roles(&self) -> CatalogResult<()>;

    /// Bootstrap a superadmin user if no users exist. Returns the generated API key if created.
    fn bootstrap_superadmin_if_empty(
        &self,
        env_key: Option<String>,
    ) -> CatalogResult<Option<(String, String)>>;
}
