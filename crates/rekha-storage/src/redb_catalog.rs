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
use serde::Serialize;
use serde::de::DeserializeOwned;
use uuid::Uuid;

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

/// A [`Catalog`] backed by an embedded redb database file.
///
/// [`RedbCatalog::open`] creates the database if the file does not exist (or is
/// empty) and opens it otherwise, so re-opening the same path after a drop
/// sees every committed mutation.
#[derive(Debug)]
pub struct RedbCatalog {
    db: Database,
}

impl RedbCatalog {
    /// Open (or create) the catalog database at `path`.
    pub fn open(path: impl AsRef<Path>) -> CatalogResult<Self> {
        let db = Database::create(path)?;
        Self::ensure_tables(&db)?;
        Ok(Self { db })
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
        catalog.create_collection(&cfg).unwrap()
    }

    #[test]
    fn catalog_roundtrip_lifecycle() {
        let (_dir, catalog) = open_catalog();

        catalog.create_tenant("tenant_a").unwrap();
        catalog.create_database("tenant_a", "db_a").unwrap();
        let created = catalog.create_collection(&config_named("coll_1")).unwrap();
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

        catalog.create_collection(&config_named("dup")).unwrap();
        let err = catalog.create_collection(&config_named("dup")).unwrap_err();
        assert!(matches!(err, CatalogError::AlreadyExists));

        let mut other_cfg = config_named("dup");
        other_cfg.database = "db_b".to_owned();
        catalog.create_database("tenant_a", "db_b").unwrap();
        catalog.create_collection(&other_cfg).unwrap();
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

        let r1 = catalog.create_collection(&c1).unwrap();
        let r2 = catalog.create_collection(&c2).unwrap();
        let r3 = catalog.create_collection(&c3).unwrap();

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
