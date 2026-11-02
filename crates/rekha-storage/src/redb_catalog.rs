//! redb-backed catalog implementation.
//!
//! Tables (all values are bincode-serialized to `&[u8]` for forward
//! compatibility; we do **not** rely on redb's serde integration):
//!
//! - `tenants` — `Table<&str, &[u8]>`, value unused (presence = existence)
//! - `databases` — `Table<(&str, &str), &[u8]>`, keyed `(tenant, database)`
//! - `collections` — `Table<&str, &[u8]>`, keyed by collection UUID string →
//!   bincode([`CollectionRecord`])
//! - `collection_by_name` — `Table<(&str, &str, &str), &[u8]>`, keyed
//!   `(tenant, database, name)` → bincode(collection UUID string). This is the
//!   unique-name index within a `(tenant, database)` scope.
//! - `segments` — `Table<&str, &[u8]>`, reserved for Phase 4 (shard map).
//!   Created but not populated.
//! - `segment_metadata` — `Table<(&str, &str), &[u8]>`, reserved for Phase 4,
//!   keyed `(collection_id, segment_id)`. Created but not populated.
//!
//! Every catalog mutation is a single redb write transaction, which matches
//! Chroma's documented single-writer SQLite sysdb limitation: only one writer
//! is admitted at a time, and each method's reads and writes commit atomically.

use std::path::Path;

use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use rekha_core::cluster::ShardId;
use serde::Serialize;
use serde::de::DeserializeOwned;
use uuid::Uuid;

use rekha_core::auth::{KeyMeta, Permission, RoleAssignment, RoleRecord, UserRecord};
use rekha_core::config::CollectionConfig;
use rekha_core::types::Metadata;

use crate::catalog::{Catalog, CatalogError, CatalogResult, CollectionRecord};

const TENANTS: TableDefinition<&str, &[u8]> = TableDefinition::new("tenants");
const DATABASES: TableDefinition<(&str, &str), &[u8]> = TableDefinition::new("databases");
const COLLECTIONS: TableDefinition<&str, &[u8]> = TableDefinition::new("collections");
const COLLECTION_BY_NAME: TableDefinition<(&str, &str, &str), &[u8]> =
    TableDefinition::new("collection_by_name");
const SEGMENTS: TableDefinition<&str, &[u8]> = TableDefinition::new("segments");
const SEGMENT_METADATA: TableDefinition<(&str, &str), &[u8]> =
    TableDefinition::new("segment_metadata");

// RBAC tables
const AUTH_USERS: TableDefinition<&str, &[u8]> = TableDefinition::new("auth_users");
const AUTH_USER_BY_NAME: TableDefinition<&str, &[u8]> = TableDefinition::new("auth_user_by_name");
const AUTH_USER_KEYS: TableDefinition<(&str, &[u8]), &[u8]> =
    TableDefinition::new("auth_user_keys");
const AUTH_KEY_INDEX: TableDefinition<&[u8], &[u8]> = TableDefinition::new("auth_key_index");
const AUTH_ROLES: TableDefinition<&str, &[u8]> = TableDefinition::new("auth_roles");
const AUTH_ROLE_BY_NAME: TableDefinition<&str, &[u8]> = TableDefinition::new("auth_role_by_name");
const AUTH_PERMS: TableDefinition<(&str, &str), &[u8]> = TableDefinition::new("auth_perms");
const AUTH_ASSIGNMENTS: TableDefinition<(&str, &str), &[u8]> =
    TableDefinition::new("auth_assignments");

/// The records a bootstrap superadmin would consist of, before anything is
/// written.
///
/// A single node writes them to its own catalog. A clustered deployment
/// proposes them through the metadata log from the bootstrap node, so the
/// cluster ends up with one `admin` holding one key instead of one per node.
pub struct SuperadminBootstrap {
    pub user: rekha_core::auth::UserRecord,
    pub key_hash: [u8; 32],
    /// The plaintext key. Printed once, by the node that creates the user.
    pub key: String,
}

/// A [`Catalog`] backed by an embedded redb database file.
///
/// [`RedbCatalog::open`] creates the database if the file does not exist (or is
/// empty) and opens it otherwise, so re-opening the same path after a drop
/// sees every committed mutation.
pub struct RedbCatalog {
    db: Database,
    /// HMAC pepper used when this catalog hashes API keys it creates itself
    /// (currently only the bootstrap superadmin). Every other key is hashed by
    /// the caller (API/CLI) with the same pepper, so authentication stays
    /// consistent when `REKHA_KEY_PEPPER` is set.
    pepper: Vec<u8>,
}

impl std::fmt::Debug for RedbCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedbCatalog")
            .field(
                "pepper",
                &if self.pepper.is_empty() {
                    "<unset>"
                } else {
                    "<set>"
                },
            )
            .finish_non_exhaustive()
    }
}

impl RedbCatalog {
    /// Open (or create) the catalog database at `path` with no pepper.
    ///
    /// Use [`RedbCatalog::open_with_pepper`] in production so bootstrap key
    /// hashes match the pepper the auth middleware verifies with.
    pub fn open(path: impl AsRef<Path>) -> CatalogResult<Self> {
        Self::open_with_pepper(path, Vec::new())
    }

    /// Open (or create) the catalog database at `path`, hashing any key this
    /// catalog creates with `pepper`.
    pub fn open_with_pepper(
        path: impl AsRef<Path>,
        pepper: impl Into<Vec<u8>>,
    ) -> CatalogResult<Self> {
        let db = Database::create(path)?;
        Self::ensure_tables(&db)?;
        Ok(Self {
            db,
            pepper: pepper.into(),
        })
    }

    /// Ensure all built-in roles exist, creating them with default permissions if missing.
    pub fn ensure_builtin_roles(&self) -> CatalogResult<()> {
        use rekha_core::auth::{BUILTIN_ROLES, RoleRecord, builtin_permissions};

        for role_name in BUILTIN_ROLES {
            if self.get_role_by_name(role_name)?.is_some() {
                continue;
            }
            let role = RoleRecord {
                id: rekha_core::auth::builtin_role_id(role_name),
                name: role_name.to_string(),
                description: format!("Built-in {role_name} role"),
                built_in: true,
            };
            // Create role first
            self.create_role(&role)?;
            // Add default permissions (global scope)
            for perm in builtin_permissions(role_name, None, None) {
                let _ = self.add_permission(&role.id, &perm);
            }
        }
        Ok(())
    }

    /// Bootstrap a superadmin user if no users exist. Returns the generated API key if created.
    /// If `env_key` is `Some`, that key is used instead of generating a new one (for REKHA_API_KEY migration).
    ///
    /// Single-node only. A clustered deployment proposes the records this
    /// returns through the metadata log instead, because a user minted locally
    /// on every node is a different user on every node.
    pub fn bootstrap_superadmin_if_empty(
        &self,
        env_key: Option<String>,
    ) -> CatalogResult<Option<(String, String)>> {
        let Some(pending) = self.pending_superadmin(env_key)? else {
            return Ok(None);
        };
        self.create_user(&pending.user)?;
        self.create_api_key(&pending.user.id, &pending.key_hash, "bootstrap")?;

        // Ensure superadmin role exists and assign it
        self.ensure_builtin_roles()?;
        if let Some(role) = self.get_role_by_name(rekha_core::auth::ROLE_SUPERADMIN)? {
            let _ = self.assign_role(&pending.user.id, &role.id, &pending.user.id);
        }

        Ok(Some((pending.user.id, pending.key)))
    }

    /// The records a bootstrap superadmin would consist of, without writing
    /// them anywhere.
    ///
    /// Returns `None` when a user already exists, so a caller can use this as
    /// the "is this store fresh?" test before proposing anything.
    pub fn pending_superadmin(
        &self,
        env_key: Option<String>,
    ) -> CatalogResult<Option<SuperadminBootstrap>> {
        if !self.list_users()?.is_empty() {
            return Ok(None);
        }
        // No users — the records for a superadmin.
        let api_key = env_key.unwrap_or_else(rekha_core::auth::generate_api_key);
        // Hash with the configured pepper, matching the auth middleware's
        // verification path. Hashing unpeppered here silently produced a key
        // that could never authenticate once REKHA_KEY_PEPPER was set.
        let key_hash = rekha_core::auth::hash_api_key_with_pepper(&api_key, &self.pepper);
        let user = rekha_core::auth::UserRecord {
            id: uuid::Uuid::new_v4().to_string(),
            name: "admin".to_string(),
            enabled: true,
            created_at_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };
        Ok(Some(SuperadminBootstrap {
            user,
            key_hash,
            key: api_key,
        }))
    }

    /// Create all tables (idempotent). Reserved tables are created here too so
    /// Phase 4 can rely on them existing without a migration.
    fn ensure_tables(db: &Database) -> CatalogResult<()> {
        let write_txn = db.begin_write()?;
        {
            write_txn.open_table(TENANTS)?;
            write_txn.open_table(DATABASES)?;
            write_txn.open_table(COLLECTIONS)?;
            write_txn.open_table(COLLECTION_BY_NAME)?;
            write_txn.open_table(SEGMENTS)?;
            write_txn.open_table(SEGMENT_METADATA)?;
            write_txn.open_table(AUTH_USERS)?;
            write_txn.open_table(AUTH_USER_BY_NAME)?;
            write_txn.open_table(AUTH_USER_KEYS)?;
            write_txn.open_table(AUTH_KEY_INDEX)?;
            write_txn.open_table(AUTH_ROLES)?;
            write_txn.open_table(AUTH_ROLE_BY_NAME)?;
            write_txn.open_table(AUTH_PERMS)?;
            write_txn.open_table(AUTH_ASSIGNMENTS)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Bincode-encode a value. Uses our own `&[u8]` encoding rather than redb's
    /// serde integration so the on-disk format is stable across redb upgrades.
    fn encode<T: Serialize>(value: &T) -> CatalogResult<Vec<u8>> {
        bincode::serialize(value).map_err(|e| CatalogError::Serialization(e.to_string()))
    }

    fn decode<T: DeserializeOwned>(bytes: &[u8]) -> CatalogResult<T> {
        bincode::deserialize(bytes).map_err(|e| CatalogError::Serialization(e.to_string()))
    }

    fn get_collection_record(&self, id: &Uuid) -> CatalogResult<Option<CollectionRecord>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(COLLECTIONS)?;
        let key = id.to_string();
        match table.get(key.as_str())? {
            Some(guard) => Ok(Some(Self::decode(guard.value())?)),
            None => Ok(None),
        }
    }
}

impl Catalog for RedbCatalog {
    fn create_tenant(&self, name: &str) -> Result<(), CatalogError> {
        if name.is_empty() {
            return Err(CatalogError::InvalidArgument(
                "tenant name must not be empty".to_owned(),
            ));
        }
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(TENANTS)?;
            if table.get(name)?.is_some() {
                return Err(CatalogError::AlreadyExists);
            }
            let empty: &[u8] = &[];
            table.insert(name, empty)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn create_database(&self, tenant: &str, name: &str) -> Result<(), CatalogError> {
        if name.is_empty() {
            return Err(CatalogError::InvalidArgument(
                "database name must not be empty".to_owned(),
            ));
        }
        let write_txn = self.db.begin_write()?;
        {
            let tenants = write_txn.open_table(TENANTS)?;
            if tenants.get(tenant)?.is_none() {
                return Err(CatalogError::NotFound);
            }
            let mut databases = write_txn.open_table(DATABASES)?;
            let key = (tenant, name);
            if databases.get(key)?.is_some() {
                return Err(CatalogError::AlreadyExists);
            }
            let empty: &[u8] = &[];
            databases.insert(key, empty)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn create_collection(
        &self,
        config: &CollectionConfig,
        shard: Option<ShardId>,
    ) -> Result<CollectionRecord, CatalogError> {
        if config.name.is_empty() {
            return Err(CatalogError::InvalidArgument(
                "collection name must not be empty".to_owned(),
            ));
        }
        let record = CollectionRecord {
            config: config.clone(),
            max_seq_id: 0,
            total_elements: 0,
            shard,
        };
        let id_key = config.id.to_string();
        let record_bytes = Self::encode(&record)?;
        let id_bytes = Self::encode(&id_key)?;
        let name_key = (
            config.tenant.as_str(),
            config.database.as_str(),
            config.name.as_str(),
        );

        let write_txn = self.db.begin_write()?;
        {
            let mut by_name = write_txn.open_table(COLLECTION_BY_NAME)?;
            if by_name.get(name_key)?.is_some() {
                return Err(CatalogError::AlreadyExists);
            }
            let mut collections = write_txn.open_table(COLLECTIONS)?;
            collections.insert(id_key.as_str(), record_bytes.as_slice())?;
            by_name.insert(name_key, id_bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(record)
    }

    fn record_collection(
        &self,
        config: &CollectionConfig,
        shard: Option<ShardId>,
    ) -> Result<CollectionRecord, CatalogError> {
        // Idempotent by design: replicated entries are re-applied on restart and
        // on every node, and "already recorded" is the expected case.
        if let Some(existing) = self.get_collection_by_id(&config.id)? {
            return Ok(existing);
        }
        self.create_collection(config, shard)
    }

    fn list_tenants(&self) -> CatalogResult<Vec<String>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(TENANTS)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            out.push(entry?.0.value().to_owned());
        }
        out.sort_unstable();
        Ok(out)
    }

    fn list_databases(&self) -> CatalogResult<Vec<(String, String)>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(DATABASES)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let key = entry?.0;
            let (tenant, name) = key.value();
            out.push((tenant.to_owned(), name.to_owned()));
        }
        out.sort_unstable();
        Ok(out)
    }

    fn list_assignments(&self) -> CatalogResult<Vec<rekha_core::auth::RoleAssignment>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(AUTH_ASSIGNMENTS)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let bytes = entry?.1.value().to_vec();
            out.push(Self::decode(&bytes)?);
        }
        out.sort_unstable_by(|a: &rekha_core::auth::RoleAssignment, b| {
            (a.user_id.as_str(), a.role_id.as_str()).cmp(&(b.user_id.as_str(), b.role_id.as_str()))
        });
        Ok(out)
    }

    fn get_collection(
        &self,
        tenant: &str,
        database: &str,
        name: &str,
    ) -> Result<Option<CollectionRecord>, CatalogError> {
        let read_txn = self.db.begin_read()?;
        let by_name = read_txn.open_table(COLLECTION_BY_NAME)?;
        let name_key = (tenant, database, name);
        let id_bytes = match by_name.get(name_key)? {
            Some(guard) => guard.value().to_vec(),
            None => return Ok(None),
        };
        let id_str: String = Self::decode(&id_bytes)?;
        let id =
            Uuid::parse_str(&id_str).map_err(|e| CatalogError::Serialization(e.to_string()))?;
        self.get_collection_record(&id)
    }

    fn get_collection_by_id(&self, id: &Uuid) -> Result<Option<CollectionRecord>, CatalogError> {
        self.get_collection_record(id)
    }

    fn list_collections(
        &self,
        tenant: &str,
        database: &str,
    ) -> Result<Vec<CollectionRecord>, CatalogError> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(COLLECTIONS)?;
        let mut records = Vec::new();
        for entry in table.iter()? {
            let (_k, v) = entry.map_err(CatalogError::from)?;
            let record: CollectionRecord = Self::decode(v.value())?;
            if record.config.tenant == tenant && record.config.database == database {
                records.push(record);
            }
        }
        Ok(records)
    }

    fn list_collections_all(&self) -> Result<Vec<CollectionRecord>, CatalogError> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(COLLECTIONS)?;
        let mut records = Vec::new();
        for entry in table.iter()? {
            let (_k, v) = entry.map_err(CatalogError::from)?;
            records.push(Self::decode(v.value())?);
        }
        Ok(records)
    }

    fn update_collection_metadata(
        &self,
        id: &Uuid,
        metadata: &Metadata,
    ) -> Result<(), CatalogError> {
        let id_key = id.to_string();
        let write_txn = self.db.begin_write()?;
        {
            let mut collections = write_txn.open_table(COLLECTIONS)?;
            let record_bytes = match collections.get(id_key.as_str())? {
                Some(guard) => guard.value().to_vec(),
                None => return Err(CatalogError::NotFound),
            };
            let mut record: CollectionRecord = Self::decode(&record_bytes)?;
            record.config.metadata = Some(metadata.clone());
            let bytes = Self::encode(&record)?;
            collections.insert(id_key.as_str(), bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn delete_collection(&self, id: &Uuid) -> Result<(), CatalogError> {
        let id_key = id.to_string();
        let write_txn = self.db.begin_write()?;
        {
            let mut collections = write_txn.open_table(COLLECTIONS)?;
            let record_bytes = match collections.get(id_key.as_str())? {
                Some(guard) => guard.value().to_vec(),
                None => return Err(CatalogError::NotFound),
            };
            let record: CollectionRecord = Self::decode(&record_bytes)?;
            collections.remove(id_key.as_str())?;
            let mut by_name = write_txn.open_table(COLLECTION_BY_NAME)?;
            let name_key = (
                record.config.tenant.as_str(),
                record.config.database.as_str(),
                record.config.name.as_str(),
            );
            by_name.remove(name_key)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn advance_log_offset(&self, id: &Uuid, seq: u64) -> Result<(), CatalogError> {
        let id_key = id.to_string();
        let write_txn = self.db.begin_write()?;
        {
            let mut collections = write_txn.open_table(COLLECTIONS)?;
            let record_bytes = match collections.get(id_key.as_str())? {
                Some(guard) => guard.value().to_vec(),
                None => return Err(CatalogError::NotFound),
            };
            let mut record: CollectionRecord = Self::decode(&record_bytes)?;
            record.max_seq_id = record.max_seq_id.max(seq);
            let bytes = Self::encode(&record)?;
            collections.insert(id_key.as_str(), bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn update_collection_dimension(&self, id: &Uuid, dimension: usize) -> Result<(), CatalogError> {
        let id_key = id.to_string();
        let write_txn = self.db.begin_write()?;
        {
            let mut collections = write_txn.open_table(COLLECTIONS)?;
            let record_bytes = match collections.get(id_key.as_str())? {
                Some(guard) => guard.value().to_vec(),
                None => return Err(CatalogError::NotFound),
            };
            let mut record: CollectionRecord = Self::decode(&record_bytes)?;
            record.config.dimension = dimension;
            let bytes = Self::encode(&record)?;
            collections.insert(id_key.as_str(), bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // RBAC — Users
    // ------------------------------------------------------------------

    fn create_user(&self, user: &UserRecord) -> CatalogResult<()> {
        if user.id.is_empty() || user.name.is_empty() {
            return Err(CatalogError::InvalidArgument(
                "user id and name must not be empty".into(),
            ));
        }
        let bytes = Self::encode(user)?;
        let write_txn = self.db.begin_write()?;
        {
            let mut users = write_txn.open_table(AUTH_USERS)?;
            if users.get(user.id.as_str())?.is_some() {
                return Err(CatalogError::AlreadyExists);
            }
            let mut by_name = write_txn.open_table(AUTH_USER_BY_NAME)?;
            if by_name.get(user.name.as_str())?.is_some() {
                return Err(CatalogError::AlreadyExists);
            }
            users.insert(user.id.as_str(), bytes.as_slice())?;
            let id_bytes = Self::encode(&user.id)?;
            by_name.insert(user.name.as_str(), id_bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn get_user(&self, id: &str) -> CatalogResult<Option<UserRecord>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(AUTH_USERS)?;
        match table.get(id)? {
            Some(g) => Ok(Some(Self::decode(g.value())?)),
            None => Ok(None),
        }
    }

    fn get_user_by_name(&self, name: &str) -> CatalogResult<Option<UserRecord>> {
        let read_txn = self.db.begin_read()?;
        let by_name = read_txn.open_table(AUTH_USER_BY_NAME)?;
        let id_bytes = match by_name.get(name)? {
            Some(g) => g.value().to_vec(),
            None => return Ok(None),
        };
        let id: String = Self::decode(&id_bytes)?;
        self.get_user(&id)
    }

    fn list_users(&self) -> CatalogResult<Vec<UserRecord>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(AUTH_USERS)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let (_, v) = entry.map_err(CatalogError::from)?;
            out.push(Self::decode(v.value())?);
        }
        Ok(out)
    }

    fn set_user_enabled(&self, id: &str, enabled: bool) -> CatalogResult<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut users = write_txn.open_table(AUTH_USERS)?;
            let raw = match users.get(id)? {
                Some(g) => g.value().to_vec(),
                None => return Err(CatalogError::NotFound),
            };
            let mut user: UserRecord = Self::decode(&raw)?;
            user.enabled = enabled;
            let bytes = Self::encode(&user)?;
            users.insert(id, bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn delete_user(&self, id: &str) -> CatalogResult<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut users = write_txn.open_table(AUTH_USERS)?;
            let raw = match users.get(id)? {
                Some(g) => g.value().to_vec(),
                None => return Err(CatalogError::NotFound),
            };
            let user: UserRecord = Self::decode(&raw)?;
            users.remove(id)?;
            let mut by_name = write_txn.open_table(AUTH_USER_BY_NAME)?;
            by_name.remove(user.name.as_str())?;

            // Remove all keys for this user (both per-user and global index)
            let mut user_keys = write_txn.open_table(AUTH_USER_KEYS)?;
            let mut key_index = write_txn.open_table(AUTH_KEY_INDEX)?;
            let mut to_delete: Vec<(String, Vec<u8>)> = Vec::new();
            let mut global_to_delete: Vec<Vec<u8>> = Vec::new();
            for entry in user_keys.iter()? {
                let (k, _) = entry.map_err(CatalogError::from)?;
                if k.value().0 == id {
                    to_delete.push((k.value().0.to_owned(), k.value().1.to_vec()));
                    global_to_delete.push(k.value().1.to_vec());
                }
            }
            for (uid, hash) in to_delete {
                user_keys.remove((uid.as_str(), hash.as_slice()))?;
            }
            for hash in global_to_delete {
                key_index.remove(hash.as_slice())?;
            }

            // Remove all assignments for this user
            let mut assignments = write_txn.open_table(AUTH_ASSIGNMENTS)?;
            let mut assign_to_delete: Vec<(String, String)> = Vec::new();
            for entry in assignments.iter()? {
                let (k, _) = entry.map_err(CatalogError::from)?;
                if k.value().0 == id {
                    assign_to_delete.push((k.value().0.to_owned(), k.value().1.to_owned()));
                }
            }
            for (uid, rid) in assign_to_delete {
                assignments.remove((uid.as_str(), rid.as_str()))?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    fn create_api_key(&self, user_id: &str, key_hash: &[u8; 32], label: &str) -> CatalogResult<()> {
        if label.is_empty() {
            return Err(CatalogError::InvalidArgument(
                "key label must not be empty".into(),
            ));
        }
        let write_txn = self.db.begin_write()?;
        {
            // Verify user exists
            let users = write_txn.open_table(AUTH_USERS)?;
            if users.get(user_id)?.is_none() {
                return Err(CatalogError::NotFound);
            }
            let mut key_index = write_txn.open_table(AUTH_KEY_INDEX)?;
            if key_index.get(key_hash.as_slice())?.is_some() {
                return Err(CatalogError::AlreadyExists);
            }
            let mut user_keys = write_txn.open_table(AUTH_USER_KEYS)?;
            let meta = KeyMeta {
                name: label.to_owned(),
                key_hash: *key_hash,
                created_at_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
                last_used_ms: None,
            };
            let meta_bytes = Self::encode(&meta)?;
            user_keys.insert((user_id, key_hash.as_slice()), meta_bytes.as_slice())?;
            let user_id_bytes = Self::encode(&user_id.to_owned())?;
            key_index.insert(key_hash.as_slice(), user_id_bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn lookup_api_key(&self, key_hash: &[u8; 32]) -> CatalogResult<Option<(String, KeyMeta)>> {
        let read_txn = self.db.begin_read()?;
        let key_index = read_txn.open_table(AUTH_KEY_INDEX)?;
        let user_id_bytes = match key_index.get(key_hash.as_slice())? {
            Some(g) => g.value().to_vec(),
            None => return Ok(None),
        };
        let user_id: String = Self::decode(&user_id_bytes)?;
        let user_keys = read_txn.open_table(AUTH_USER_KEYS)?;
        match user_keys.get((user_id.as_str(), key_hash.as_slice()))? {
            Some(g) => {
                let meta: KeyMeta = Self::decode(g.value())?;
                Ok(Some((user_id, meta)))
            }
            None => Ok(None),
        }
    }

    fn delete_api_key(&self, user_id: &str, key_hash: &[u8; 32]) -> CatalogResult<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut user_keys = write_txn.open_table(AUTH_USER_KEYS)?;
            if user_keys.get((user_id, key_hash.as_slice()))?.is_none() {
                return Err(CatalogError::NotFound);
            }
            user_keys.remove((user_id, key_hash.as_slice()))?;
            let mut key_index = write_txn.open_table(AUTH_KEY_INDEX)?;
            key_index.remove(key_hash.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn list_user_keys(&self, user_id: &str) -> CatalogResult<Vec<KeyMeta>> {
        let read_txn = self.db.begin_read()?;
        // Verify user exists
        let users = read_txn.open_table(AUTH_USERS)?;
        if users.get(user_id)?.is_none() {
            return Err(CatalogError::NotFound);
        }
        let user_keys = read_txn.open_table(AUTH_USER_KEYS)?;
        let mut out = Vec::new();
        for entry in user_keys.iter()? {
            let (k, v) = entry.map_err(CatalogError::from)?;
            if k.value().0 == user_id {
                out.push(Self::decode(v.value())?);
            }
        }
        Ok(out)
    }

    fn create_role(&self, role: &RoleRecord) -> CatalogResult<()> {
        if role.id.is_empty() || role.name.is_empty() {
            return Err(CatalogError::InvalidArgument(
                "role id and name must not be empty".into(),
            ));
        }
        let bytes = Self::encode(role)?;
        let write_txn = self.db.begin_write()?;
        {
            let mut roles = write_txn.open_table(AUTH_ROLES)?;
            if roles.get(role.id.as_str())?.is_some() {
                return Err(CatalogError::AlreadyExists);
            }
            let mut by_name = write_txn.open_table(AUTH_ROLE_BY_NAME)?;
            if by_name.get(role.name.as_str())?.is_some() {
                return Err(CatalogError::AlreadyExists);
            }
            roles.insert(role.id.as_str(), bytes.as_slice())?;
            let id_bytes = Self::encode(&role.id)?;
            by_name.insert(role.name.as_str(), id_bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn get_role(&self, id: &str) -> CatalogResult<Option<RoleRecord>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(AUTH_ROLES)?;
        match table.get(id)? {
            Some(g) => Ok(Some(Self::decode(g.value())?)),
            None => Ok(None),
        }
    }

    fn get_role_by_name(&self, name: &str) -> CatalogResult<Option<RoleRecord>> {
        let read_txn = self.db.begin_read()?;
        let by_name = read_txn.open_table(AUTH_ROLE_BY_NAME)?;
        let id_bytes = match by_name.get(name)? {
            Some(g) => g.value().to_vec(),
            None => return Ok(None),
        };
        let id: String = Self::decode(&id_bytes)?;
        self.get_role(&id)
    }

    fn list_roles(&self) -> CatalogResult<Vec<RoleRecord>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(AUTH_ROLES)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let (_, v) = entry.map_err(CatalogError::from)?;
            out.push(Self::decode(v.value())?);
        }
        Ok(out)
    }

    fn delete_role(&self, id: &str) -> CatalogResult<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut roles = write_txn.open_table(AUTH_ROLES)?;
            let raw = match roles.get(id)? {
                Some(g) => g.value().to_vec(),
                None => return Err(CatalogError::NotFound),
            };
            let role: RoleRecord = Self::decode(&raw)?;
            if role.built_in {
                return Err(CatalogError::InvalidArgument(
                    "cannot delete built-in role".into(),
                ));
            }
            roles.remove(id)?;
            let mut by_name = write_txn.open_table(AUTH_ROLE_BY_NAME)?;
            by_name.remove(role.name.as_str())?;

            // Remove all permissions for this role
            let mut perms = write_txn.open_table(AUTH_PERMS)?;
            let mut perms_to_delete: Vec<(String, String)> = Vec::new();
            for entry in perms.iter()? {
                let (k, _) = entry.map_err(CatalogError::from)?;
                if k.value().0 == id {
                    perms_to_delete.push((k.value().0.to_owned(), k.value().1.to_owned()));
                }
            }
            for (rid, pid) in perms_to_delete {
                perms.remove((rid.as_str(), pid.as_str()))?;
            }

            // Remove all assignments for this role
            let mut assignments = write_txn.open_table(AUTH_ASSIGNMENTS)?;
            let mut assign_to_delete: Vec<(String, String)> = Vec::new();
            for entry in assignments.iter()? {
                let (k, _) = entry.map_err(CatalogError::from)?;
                if k.value().1 == id {
                    assign_to_delete.push((k.value().0.to_owned(), k.value().1.to_owned()));
                }
            }
            for (uid, rid) in assign_to_delete {
                assignments.remove((uid.as_str(), rid.as_str()))?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    fn add_permission(&self, role_id: &str, perm: &Permission) -> CatalogResult<()> {
        if perm.id.is_empty() || perm.action.is_empty() {
            return Err(CatalogError::InvalidArgument(
                "permission id and action must not be empty".into(),
            ));
        }
        let write_txn = self.db.begin_write()?;
        {
            let roles = write_txn.open_table(AUTH_ROLES)?;
            if roles.get(role_id)?.is_none() {
                return Err(CatalogError::NotFound);
            }
            let mut perms = write_txn.open_table(AUTH_PERMS)?;
            if perms.get((role_id, perm.id.as_str()))?.is_some() {
                return Err(CatalogError::AlreadyExists);
            }
            let bytes = Self::encode(perm)?;
            perms.insert((role_id, perm.id.as_str()), bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn remove_permission(&self, role_id: &str, perm_id: &str) -> CatalogResult<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut perms = write_txn.open_table(AUTH_PERMS)?;
            if perms.get((role_id, perm_id))?.is_none() {
                return Err(CatalogError::NotFound);
            }
            perms.remove((role_id, perm_id))?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn get_role_permissions(&self, role_id: &str) -> CatalogResult<Vec<Permission>> {
        let read_txn = self.db.begin_read()?;
        let roles = read_txn.open_table(AUTH_ROLES)?;
        if roles.get(role_id)?.is_none() {
            return Err(CatalogError::NotFound);
        }
        let perms = read_txn.open_table(AUTH_PERMS)?;
        let mut out = Vec::new();
        for entry in perms.iter()? {
            let (k, v) = entry.map_err(CatalogError::from)?;
            if k.value().0 == role_id {
                out.push(Self::decode(v.value())?);
            }
        }
        Ok(out)
    }

    fn assign_role(&self, user_id: &str, role_id: &str, assigned_by: &str) -> CatalogResult<()> {
        let write_txn = self.db.begin_write()?;
        {
            let users = write_txn.open_table(AUTH_USERS)?;
            if users.get(user_id)?.is_none() {
                return Err(CatalogError::NotFound);
            }
            let roles = write_txn.open_table(AUTH_ROLES)?;
            if roles.get(role_id)?.is_none() {
                return Err(CatalogError::NotFound);
            }
            let mut assignments = write_txn.open_table(AUTH_ASSIGNMENTS)?;
            if assignments.get((user_id, role_id))?.is_some() {
                return Ok(());
            }
            let assignment = RoleAssignment {
                user_id: user_id.to_owned(),
                role_id: role_id.to_owned(),
                assigned_at_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
                assigned_by: assigned_by.to_owned(),
            };
            let bytes = Self::encode(&assignment)?;
            assignments.insert((user_id, role_id), bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn unassign_role(&self, user_id: &str, role_id: &str) -> CatalogResult<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut assignments = write_txn.open_table(AUTH_ASSIGNMENTS)?;
            if assignments.get((user_id, role_id))?.is_none() {
                return Err(CatalogError::NotFound);
            }
            assignments.remove((user_id, role_id))?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn get_user_roles(&self, user_id: &str) -> CatalogResult<Vec<(RoleRecord, Vec<Permission>)>> {
        let read_txn = self.db.begin_read()?;
        let users = read_txn.open_table(AUTH_USERS)?;
        if users.get(user_id)?.is_none() {
            return Err(CatalogError::NotFound);
        }
        let assignments = read_txn.open_table(AUTH_ASSIGNMENTS)?;
        let roles_table = read_txn.open_table(AUTH_ROLES)?;
        let perms_table = read_txn.open_table(AUTH_PERMS)?;

        let mut out = Vec::new();
        for entry in assignments.iter()? {
            let (k, _) = entry.map_err(CatalogError::from)?;
            if k.value().0 != user_id {
                continue;
            }
            let role_id = k.value().1;
            let role_raw = match roles_table.get(role_id)? {
                Some(g) => g.value().to_vec(),
                None => continue,
            };
            let role: RoleRecord = Self::decode(&role_raw)?;
            let mut perms = Vec::new();
            for p_entry in perms_table.iter()? {
                let (pk, pv) = p_entry.map_err(CatalogError::from)?;
                if pk.value().0 == role_id {
                    perms.push(Self::decode(pv.value())?);
                }
            }
            out.push((role, perms));
        }
        Ok(out)
    }

    fn get_user_permissions(&self, user_id: &str) -> CatalogResult<Vec<Permission>> {
        let roles = self.get_user_roles(user_id)?;
        let mut out = Vec::new();
        for (_, perms) in roles {
            out.extend(perms);
        }
        Ok(out)
    }

    fn ensure_builtin_roles(&self) -> CatalogResult<()> {
        RedbCatalog::ensure_builtin_roles(self)
    }

    fn bootstrap_superadmin_if_empty(
        &self,
        env_key: Option<String>,
    ) -> CatalogResult<Option<(String, String)>> {
        RedbCatalog::bootstrap_superadmin_if_empty(self, env_key)
    }
}

impl RedbCatalog {
    /// Ensure the reserved Phase 4 tables exist (validated by tests).
    #[cfg(test)]
    fn reserved_tables_exist(&self) -> CatalogResult<bool> {
        use redb::ReadableTableMetadata;

        let read_txn = self.db.begin_read()?;
        let segments = read_txn.open_table(SEGMENTS)?;
        let segment_metadata = read_txn.open_table(SEGMENT_METADATA)?;
        Ok(segments.len()? == 0 && segment_metadata.len()? == 0)
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use rekha_core::config::CollectionConfig;
    use rekha_core::types::{Distance, Metadata, MetadataValue};

    use super::*;

    fn config_named(name: &str) -> CollectionConfig {
        let mut c = CollectionConfig::new(name.to_owned(), 8, Distance::Cosine);
        c.tenant = "tenant_a".to_owned();
        c.database = "db_a".to_owned();
        c
    }

    fn open_catalog() -> (TempDir, RedbCatalog) {
        let dir = TempDir::new().unwrap();
        let catalog = RedbCatalog::open(dir.path().join("catalog.redb")).unwrap();
        (dir, catalog)
    }

    fn seed(catalog: &RedbCatalog) -> CollectionRecord {
        catalog.create_tenant("tenant_a").unwrap();
        catalog.create_database("tenant_a", "db_a").unwrap();
        let cfg = config_named("coll_1");
        catalog.create_collection(&cfg, None).unwrap()
    }

    #[test]
    fn catalog_roundtrip_lifecycle() {
        let (_dir, catalog) = open_catalog();

        catalog.create_tenant("tenant_a").unwrap();
        catalog.create_database("tenant_a", "db_a").unwrap();
        let created = catalog
            .create_collection(&config_named("coll_1"), None)
            .unwrap();
        assert_eq!(created.config.name, "coll_1");
        assert_eq!(created.max_seq_id, 0);
        assert_eq!(created.total_elements, 0);

        let by_name = catalog
            .get_collection("tenant_a", "db_a", "coll_1")
            .unwrap()
            .expect("collection findable by name");
        assert_eq!(by_name.config.id, created.config.id);

        let by_id = catalog
            .get_collection_by_id(&created.config.id)
            .unwrap()
            .expect("collection findable by id");
        assert_eq!(by_id.config.id, created.config.id);
        assert_eq!(by_id.config.dimension, 8);
        assert_eq!(by_id.config.space, Distance::Cosine);

        let listed = catalog.list_collections("tenant_a", "db_a").unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].config.id, created.config.id);
        assert!(
            catalog
                .list_collections("tenant_a", "other_db")
                .unwrap()
                .is_empty()
        );

        let mut meta = Metadata::new();
        meta.insert("dataset".to_owned(), MetadataValue::Str("wiki".to_owned()));
        catalog
            .update_collection_metadata(&created.config.id, &meta)
            .unwrap();
        let updated = catalog
            .get_collection_by_id(&created.config.id)
            .unwrap()
            .unwrap();
        let stored = updated.config.metadata.expect("metadata stored");
        assert_eq!(
            stored.get("dataset"),
            Some(&MetadataValue::Str("wiki".to_owned()))
        );

        catalog.advance_log_offset(&created.config.id, 42).unwrap();
        catalog.advance_log_offset(&created.config.id, 7).unwrap();
        let advanced = catalog
            .get_collection_by_id(&created.config.id)
            .unwrap()
            .unwrap();
        assert_eq!(advanced.max_seq_id, 42);

        catalog.delete_collection(&created.config.id).unwrap();
        assert!(
            catalog
                .get_collection_by_id(&created.config.id)
                .unwrap()
                .is_none()
        );
        assert!(
            catalog
                .get_collection("tenant_a", "db_a", "coll_1")
                .unwrap()
                .is_none()
        );
        assert!(
            catalog
                .list_collections("tenant_a", "db_a")
                .unwrap()
                .is_empty()
        );

        assert!(matches!(
            catalog.delete_collection(&created.config.id),
            Err(CatalogError::NotFound)
        ));
    }

    #[test]
    fn duplicate_collection_name_is_rejected() {
        let (_dir, catalog) = open_catalog();
        catalog.create_tenant("tenant_a").unwrap();
        catalog.create_database("tenant_a", "db_a").unwrap();

        catalog
            .create_collection(&config_named("dup"), None)
            .unwrap();
        let err = catalog
            .create_collection(&config_named("dup"), None)
            .unwrap_err();
        assert!(matches!(err, CatalogError::AlreadyExists));

        let mut other_cfg = config_named("dup");
        other_cfg.database = "db_b".to_owned();
        catalog.create_database("tenant_a", "db_b").unwrap();
        catalog.create_collection(&other_cfg, None).unwrap();
        assert_eq!(
            catalog.list_collections("tenant_a", "db_a").unwrap().len(),
            1
        );
    }

    #[test]
    fn duplicate_tenant_and_database_rejected() {
        let (_dir, catalog) = open_catalog();
        catalog.create_tenant("tenant_a").unwrap();
        assert!(matches!(
            catalog.create_tenant("tenant_a"),
            Err(CatalogError::AlreadyExists)
        ));
        assert!(matches!(
            catalog.create_database("missing_tenant", "db"),
            Err(CatalogError::NotFound)
        ));
        catalog.create_database("tenant_a", "db").unwrap();
        assert!(matches!(
            catalog.create_database("tenant_a", "db"),
            Err(CatalogError::AlreadyExists)
        ));
    }

    #[test]
    fn list_collections_all_spans_tenants_and_databases() {
        let (_dir, catalog) = open_catalog();

        catalog.create_tenant("tenant_a").unwrap();
        catalog.create_database("tenant_a", "db_a").unwrap();
        catalog.create_database("tenant_a", "db_b").unwrap();
        catalog.create_tenant("tenant_b").unwrap();
        catalog.create_database("tenant_b", "db_a").unwrap();

        let c1 = config_named("coll_1"); // tenant_a/db_a
        let mut c2 = config_named("coll_2"); // tenant_a/db_b
        let mut c3 = config_named("coll_3"); // tenant_b/db_a
        c2.database = "db_b".to_owned();
        c3.tenant = "tenant_b".to_owned();
        c3.database = "db_a".to_owned();

        let r1 = catalog.create_collection(&c1, None).unwrap();
        let r2 = catalog.create_collection(&c2, None).unwrap();
        let r3 = catalog.create_collection(&c3, None).unwrap();

        let all = catalog.list_collections_all().unwrap();
        assert_eq!(all.len(), 3);
        let mut ids: Vec<Uuid> = all.iter().map(|r| r.config.id).collect();
        ids.sort();
        let mut want = vec![r1.config.id, r2.config.id, r3.config.id];
        want.sort();
        assert_eq!(ids, want);

        // Scoped list still sees only its own scope.
        assert_eq!(
            catalog.list_collections("tenant_a", "db_a").unwrap().len(),
            1
        );
        assert_eq!(
            catalog.list_collections("tenant_b", "db_a").unwrap().len(),
            1
        );
    }

    #[test]
    fn reserved_phase4_tables_created() {
        let (_dir, catalog) = open_catalog();
        assert!(catalog.reserved_tables_exist().unwrap());
    }

    #[test]
    fn catalog_persists_across_reopen() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("catalog.redb");
        {
            let catalog = RedbCatalog::open(&path).unwrap();
            let created = seed(&catalog);
            catalog.advance_log_offset(&created.config.id, 99).unwrap();
        }

        let reopened = RedbCatalog::open(&path).unwrap();
        assert!(
            reopened
                .get_collection_by_id(&Uuid::new_v4())
                .unwrap()
                .is_none()
        );
        let record = reopened
            .get_collection("tenant_a", "db_a", "coll_1")
            .unwrap();
        let record = record.expect("collection survives reopen");
        assert_eq!(record.config.name, "coll_1");
        assert_eq!(record.max_seq_id, 99);
    }
}
