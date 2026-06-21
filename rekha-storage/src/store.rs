use rekha_core::{RekhaError, StorageError, VectorStoreBackend};
use rocksdb::{ColumnFamilyDescriptor, DBWithThreadMode, IteratorMode, MultiThreaded, Options};
use std::path::Path;
use std::sync::Arc;

const CF_VECTORS: &str = "vectors";
const CF_PAYLOADS: &str = "payloads";
const CF_METADATA: &str = "metadata";
const CF_RAFT_LOG: &str = "raft_log";

/// RocksDB-backed vector storage.
///
/// Manages multiple column families:
/// - `vectors`: vector binary data indexed by ID
/// - `payloads`: user payloads (JSON/text) indexed by ID
/// - `metadata`: cluster config, PQ centroids, index stats
/// - `raft_log`: Raft WAL entries for replication
///
/// Supports optional key namespacing: when `namespace` is set, all keys are
/// prefixed with `{namespace}\0` to isolate data for different collections
/// within the same RocksDB instance.
#[derive(Clone)]
pub struct RocksVectorStore {
    db: Arc<DBWithThreadMode<MultiThreaded>>,
    namespace: Option<String>,
    max_payload_size: usize,
}

impl RocksVectorStore {
    /// Open or create a RocksDB database at `path` with all required column families.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RekhaError> {
        let path = path.as_ref();
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cf_descriptors: Vec<ColumnFamilyDescriptor> =
            [CF_VECTORS, CF_PAYLOADS, CF_METADATA, CF_RAFT_LOG]
                .iter()
                .map(|name| {
                    let mut cf_opts = Options::default();
                    cf_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
                    ColumnFamilyDescriptor::new(*name, cf_opts)
                })
                .collect();

        let db =
            DBWithThreadMode::<MultiThreaded>::open_cf_descriptors(&opts, path, cf_descriptors)
                .map_err(|e| StorageError::DbOpen {
                    path: path.display().to_string(),
                    source: e.to_string(),
                })?;

        Ok(Self {
            db: Arc::new(db),
            namespace: None,
            max_payload_size: 1024 * 1024,
        })
    }

    /// Create a store from an existing RocksDB handle with an optional namespace.
    pub fn from_db(db: Arc<DBWithThreadMode<MultiThreaded>>, namespace: Option<String>) -> Self {
        Self {
            db,
            namespace,
            max_payload_size: 1024 * 1024,
        }
    }

    /// Set the namespace prefix for key isolation.
    pub fn with_namespace(mut self, namespace: String) -> Self {
        self.namespace = Some(namespace);
        self
    }

    /// Get the namespace, if any.
    pub fn get_namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    /// Configure the maximum allowed payload size.
    pub fn with_max_payload_size(mut self, max: usize) -> Self {
        self.max_payload_size = max;
        self
    }

    /// Access the underlying RocksDB handle.
    pub fn db(&self) -> &Arc<DBWithThreadMode<MultiThreaded>> {
        &self.db
    }

    /// Encode a u64 ID into a big-endian key, optionally prefixed with namespace.
    fn encode_key(&self, id: u64) -> Vec<u8> {
        let ns = self.namespace.as_deref();
        if let Some(ns) = ns {
            let mut key = Vec::with_capacity(ns.len() + 1 + 8);
            key.extend_from_slice(ns.as_bytes());
            key.push(0);
            key.extend_from_slice(&id.to_be_bytes());
            key
        } else {
            id.to_be_bytes().to_vec()
        }
    }

    /// Get the namespace prefix bytes (for iteration seek), if namespaced.
    fn namespace_prefix(&self) -> Option<Vec<u8>> {
        self.namespace.as_ref().map(|ns| {
            let mut prefix = Vec::with_capacity(ns.len() + 2);
            prefix.extend_from_slice(ns.as_bytes());
            prefix.push(0);
            prefix
        })
    }

    /// Decode a key back into a u64 ID, handling optional namespace prefix.
    fn decode_id(key: &[u8]) -> Option<u64> {
        // Last 8 bytes are always the u64 BE id
        if key.len() >= 8 {
            let id_bytes = &key[key.len() - 8..];
            Some(u64::from_be_bytes(id_bytes.try_into().ok()?))
        } else {
            None
        }
    }
}

impl Drop for RocksVectorStore {
    fn drop(&mut self) {
        let _ = self.db.flush_wal(true);
    }
}

impl RocksVectorStore {
    /// Delete all keys within the current namespace across all column families.
    pub fn delete_all_in_namespace(&self) -> Result<u64, RekhaError> {
        let prefix = self
            .namespace_prefix()
            .ok_or_else(|| RekhaError::Internal {
                detail: "delete_all_in_namespace requires a namespace".into(),
            })?;
        let mut count = 0u64;
        for cf_name in &[CF_VECTORS, CF_PAYLOADS] {
            let cf = self
                .db
                .cf_handle(cf_name)
                .ok_or_else(|| StorageError::ColumnFamily {
                    name: cf_name.to_string(),
                    source: "handle not found".into(),
                })?;
            let mut batch = rocksdb::WriteBatch::default();
            let iter = self.db.iterator_cf(
                &cf,
                IteratorMode::From(&prefix, rocksdb::Direction::Forward),
            );
            for result in iter {
                let (key, _) = result.map_err(|e| RekhaError::Internal {
                    detail: format!("db iteration error: {e}"),
                })?;
                if key.len() < prefix.len() || key[..prefix.len()] != prefix[..] {
                    break;
                }
                batch.delete_cf(&cf, &key);
                count += 1;
            }
            self.db.write(batch).map_err(|e| RekhaError::Internal {
                detail: format!("failed to delete namespace keys: {e}"),
            })?;
        }
        Ok(count)
    }

    /// Return an estimate of storage usage (number of vectors).
    pub fn get_storage_estimate(&self) -> Result<u64, RekhaError> {
        self.iter_ids().map(|ids| ids.len() as u64)
    }
}

impl VectorStoreBackend for RocksVectorStore {
    fn put_vector(&self, id: u64, data: &[f32]) -> Result<(), RekhaError> {
        let key = self.encode_key(id);
        let value = vector_to_bytes(data);
        let cf = self
            .db
            .cf_handle(CF_VECTORS)
            .ok_or_else(|| StorageError::ColumnFamily {
                name: CF_VECTORS.into(),
                source: "handle not found".into(),
            })?;
        self.db.put_cf(&cf, key, value).map_err(|e| {
            StorageError::Write {
                source: e.to_string(),
            }
            .into()
        })
    }

    fn get_vector(&self, id: u64) -> Result<Option<Vec<f32>>, RekhaError> {
        let key = self.encode_key(id);
        let cf = self
            .db
            .cf_handle(CF_VECTORS)
            .ok_or_else(|| StorageError::ColumnFamily {
                name: CF_VECTORS.into(),
                source: "handle not found".into(),
            })?;
        match self.db.get_cf(&cf, key) {
            Ok(Some(bytes)) => Ok(Some(bytes_to_vector(&bytes))),
            Ok(None) => Ok(None),
            Err(e) => Err(StorageError::Read {
                key: id.to_be_bytes().to_vec(),
                source: e.to_string(),
            }
            .into()),
        }
    }

    fn put_payload(&self, id: u64, payload: &[u8]) -> Result<(), RekhaError> {
        if payload.len() > self.max_payload_size {
            return Err(StorageError::PayloadTooLarge {
                size: payload.len(),
                max: self.max_payload_size,
            }
            .into());
        }
        let key = self.encode_key(id);
        let cf = self
            .db
            .cf_handle(CF_PAYLOADS)
            .ok_or_else(|| StorageError::ColumnFamily {
                name: CF_PAYLOADS.into(),
                source: "handle not found".into(),
            })?;
        self.db.put_cf(&cf, key, payload).map_err(|e| {
            StorageError::Write {
                source: e.to_string(),
            }
            .into()
        })
    }

    fn get_payload(&self, id: u64) -> Result<Option<Vec<u8>>, RekhaError> {
        let key = self.encode_key(id);
        let cf = self
            .db
            .cf_handle(CF_PAYLOADS)
            .ok_or_else(|| StorageError::ColumnFamily {
                name: CF_PAYLOADS.into(),
                source: "handle not found".into(),
            })?;
        match self.db.get_cf(&cf, key) {
            Ok(Some(bytes)) => Ok(Some(bytes.to_vec())),
            Ok(None) => Ok(None),
            Err(e) => Err(StorageError::Read {
                key: id.to_be_bytes().to_vec(),
                source: e.to_string(),
            }
            .into()),
        }
    }

    fn delete(&self, ids: &[u64]) -> Result<u64, RekhaError> {
        if ids.is_empty() {
            return Ok(0);
        }

        let cf_vec = self
            .db
            .cf_handle(CF_VECTORS)
            .ok_or_else(|| StorageError::ColumnFamily {
                name: CF_VECTORS.into(),
                source: "handle not found".into(),
            })?;
        let cf_pay = self
            .db
            .cf_handle(CF_PAYLOADS)
            .ok_or_else(|| StorageError::ColumnFamily {
                name: CF_PAYLOADS.into(),
                source: "handle not found".into(),
            })?;

        let mut batch = rocksdb::WriteBatch::default();
        for id in ids {
            let key = self.encode_key(*id);
            batch.delete_cf(&cf_vec, &key);
            batch.delete_cf(&cf_pay, &key);
        }
        self.db.write(batch).map_err(|e| StorageError::Write {
            source: e.to_string(),
        })?;

        Ok(ids.len() as u64)
    }

    fn iter_ids(&self) -> Result<Vec<u64>, RekhaError> {
        let cf = self
            .db
            .cf_handle(CF_VECTORS)
            .ok_or_else(|| StorageError::ColumnFamily {
                name: CF_VECTORS.into(),
                source: "handle not found".into(),
            })?;
        let mut ids = Vec::new();

        let prefix = self.namespace_prefix();
        let iter_mode = match &prefix {
            Some(p) => IteratorMode::From(p, rocksdb::Direction::Forward),
            None => IteratorMode::Start,
        };
        let prefix_len = prefix.as_ref().map(|p| p.len());

        let iter = self.db.iterator_cf(&cf, iter_mode);
        for result in iter {
            let (key, _) = result.map_err(|e| RekhaError::Internal {
                detail: format!("db iteration error: {e}"),
            })?;
            // If namespaced, skip keys that don't match the prefix.
            if let Some(plen) = prefix_len {
                if key.len() < plen || &key[..plen] != prefix.as_ref().unwrap() {
                    break;
                }
            }
            if let Some(id) = Self::decode_id(&key) {
                ids.push(id);
            }
        }
        Ok(ids)
    }
}

/// Serialize a `Vec<f32>` to bytes (little-endian f32 array).
fn vector_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Deserialize bytes back into a `Vec<f32>`.
fn bytes_to_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_roundtrip() {
        let dir = std::env::temp_dir().join("rekha_test_store");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        store.put_vector(42, &data).unwrap();
        let retrieved = store.get_vector(42).unwrap().unwrap();
        assert!((retrieved[0] - 1.0).abs() < 1e-6);
        assert_eq!(retrieved.len(), 4);
    }

    #[test]
    fn payload_roundtrip() {
        let dir = std::env::temp_dir().join("rekha_test_store2");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        let payload = b"hello world".to_vec();
        store.put_payload(42, &payload).unwrap();
        let retrieved = store.get_payload(42).unwrap().unwrap();
        assert_eq!(retrieved, payload);
    }

    #[test]
    fn delete_vector() {
        let dir = std::env::temp_dir().join("rekha_test_delete");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        store.put_vector(1, &[1.0, 2.0]).unwrap();
        store.put_vector(2, &[3.0, 4.0]).unwrap();
        let deleted = store.delete(&[1, 2]).unwrap();
        assert_eq!(deleted, 2);
        assert!(store.get_vector(1).unwrap().is_none());
        assert!(store.get_vector(2).unwrap().is_none());
    }

    #[test]
    fn get_nonexistent_vector() {
        let dir = std::env::temp_dir().join("rekha_test_nonexist");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        let result = store.get_vector(999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn iter_ids_empty() {
        let dir = std::env::temp_dir().join("rekha_test_iter_empty");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        let ids = store.iter_ids().unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn iter_ids_after_inserts() {
        let dir = std::env::temp_dir().join("rekha_test_iter");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        store.put_vector(10, &[0.1; 4]).unwrap();
        store.put_vector(20, &[0.2; 4]).unwrap();
        store.put_vector(30, &[0.3; 4]).unwrap();

        let mut ids = store.iter_ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    #[test]
    fn large_vector_roundtrip() {
        let dir = std::env::temp_dir().join("rekha_test_large");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        let data: Vec<f32> = (0..768).map(|i| i as f32).collect();
        store.put_vector(1, &data).unwrap();
        let retrieved = store.get_vector(1).unwrap().unwrap();
        assert_eq!(retrieved.len(), 768);
        assert!((retrieved[0] - 0.0).abs() < 1e-6);
        assert!((retrieved[767] - 767.0).abs() < 1e-6);
    }

    #[test]
    fn payload_too_large() {
        let dir = std::env::temp_dir().join("rekha_test_payload_large");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir)
            .unwrap()
            .with_max_payload_size(10);

        let large = vec![0u8; 100];
        let result = store.put_payload(42, &large);
        assert!(result.is_err());
    }

    #[test]
    fn overwrite_vector() {
        let dir = std::env::temp_dir().join("rekha_test_overwrite");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        store.put_vector(1, &[1.0, 2.0]).unwrap();
        store.put_vector(1, &[3.0, 4.0]).unwrap();
        let retrieved = store.get_vector(1).unwrap().unwrap();
        assert!((retrieved[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn delete_partial() {
        let dir = std::env::temp_dir().join("rekha_test_delete_partial");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        store.put_vector(1, &[1.0]).unwrap();
        store.put_vector(2, &[2.0]).unwrap();
        store.put_vector(3, &[3.0]).unwrap();
        let deleted = store.delete(&[2]).unwrap();
        assert_eq!(deleted, 1);
        assert!(store.get_vector(1).unwrap().is_some());
        assert!(store.get_vector(2).unwrap().is_none());
        assert!(store.get_vector(3).unwrap().is_some());
    }

    #[test]
    fn test_get_payload_nonexistent() {
        let dir = std::env::temp_dir().join("rekha_test_payload_nonexist");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();
        let result = store.get_payload(999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_max_payload_size_default() {
        let dir = std::env::temp_dir().join("rekha_test_default_max");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();
        // Default max payload is 1MB: 1MB payload should be fine
        let large = vec![0u8; 1024 * 1024];
        store.put_payload(1, &large).unwrap();
        // Slightly over 1MB should fail
        let too_large = vec![0u8; 1024 * 1024 + 1];
        let result = store.put_payload(2, &too_large);
        assert!(result.is_err());
    }

    #[test]
    fn test_store_with_custom_max() {
        let dir = std::env::temp_dir().join("rekha_test_custom_max");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir)
            .unwrap()
            .with_max_payload_size(100);
        let ok_sized = vec![0u8; 50];
        store.put_payload(1, &ok_sized).unwrap();
        let too_big = vec![0u8; 150];
        let result = store.put_payload(2, &too_big);
        assert!(result.is_err());
    }

    #[test]
    fn test_store_drop_flush() {
        let dir = std::env::temp_dir().join("rekha_test_drop_flush");
        let _ = std::fs::remove_dir_all(&dir);
        // Insert, drop, then re-open and verify data persists
        {
            let store = RocksVectorStore::open(&dir).unwrap();
            store.put_vector(42, &[1.0, 2.0, 3.0]).unwrap();
            // drop triggers WAL flush
        }
        let store = RocksVectorStore::open(&dir).unwrap();
        let v = store.get_vector(42).unwrap().unwrap();
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_db_and_namespace_helpers() {
        let dir = std::env::temp_dir().join("rekha_test_from_db");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();
        let db = store.db().clone();

        // Create from_db without namespace
        let store2 = RocksVectorStore::from_db(db.clone(), None);
        assert!(store2.get_namespace().is_none());

        // Create from_db with namespace
        let store3 = RocksVectorStore::from_db(db.clone(), Some("ns1".into()));
        assert_eq!(store3.get_namespace(), Some("ns1"));

        // with_namespace builder
        let store4 = store.clone().with_namespace("ns2".into());
        assert_eq!(store4.get_namespace(), Some("ns2"));
    }

    #[test]
    fn test_encode_key_roundtrip() {
        let dir = std::env::temp_dir().join("rekha_test_encode");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        // Non-namespaced
        let key = store.encode_key(42);
        assert_eq!(key.len(), 8);
        assert_eq!(key, 42u64.to_be_bytes());
        assert_eq!(RocksVectorStore::decode_id(&key), Some(42));

        // Namespaced
        let store_ns = store.clone().with_namespace("col".into());
        let key_ns = store_ns.encode_key(42);
        assert_eq!(key_ns.len(), 12); // "col" + null + 8 bytes u64
        assert_eq!(&key_ns[..4], &b"col\0"[..]);
        assert_eq!(&key_ns[4..], &42u64.to_be_bytes());
        assert_eq!(RocksVectorStore::decode_id(&key_ns), Some(42));

        // decode_id with short key
        assert_eq!(RocksVectorStore::decode_id(&[0u8; 4]), None);
    }

    #[test]
    fn test_namespace_prefix() {
        let dir = std::env::temp_dir().join("rekha_test_ns_prefix");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        // Non-namespaced -> None
        let prefix = store.namespace_prefix();
        assert!(prefix.is_none());

        // Namespaced -> Some prefix
        let store_ns = store.clone().with_namespace("col".into());
        let prefix = store_ns.namespace_prefix();
        assert_eq!(prefix, Some(b"col\0".to_vec()));
    }

    #[test]
    fn test_delete_empty_ids() {
        let dir = std::env::temp_dir().join("rekha_test_delete_empty");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();
        let deleted = store.delete(&[]).unwrap();
        assert_eq!(deleted, 0);
    }

    #[test]
    fn test_delete_all_in_namespace() {
        let dir = std::env::temp_dir().join("rekha_test_del_ns");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        // delete_all requires namespace - should error without one
        let result = store.delete_all_in_namespace();
        assert!(result.is_err());

        // With namespace: insert data, then delete all
        let store_ns = store.with_namespace("col".into());
        store_ns.put_vector(1, &[1.0]).unwrap();
        store_ns.put_vector(2, &[2.0]).unwrap();
        store_ns.put_payload(1, b"p1").unwrap();

        let count = store_ns.delete_all_in_namespace().unwrap();
        // vectors CF: 2 entries, payloads CF: 1 entry = 3 total
        assert_eq!(count, 3);

        assert!(store_ns.get_vector(1).unwrap().is_none());
        assert!(store_ns.get_vector(2).unwrap().is_none());
        assert!(store_ns.get_payload(1).unwrap().is_none());
    }

    #[test]
    fn test_db_accessor() {
        let dir = std::env::temp_dir().join("rekha_test_db_accessor");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();
        let db = store.db();
        // db is an Arc - just verify it's valid by using it
        let cf = db.cf_handle("vectors").unwrap();
        let _ = cf;
    }

    #[test]
    fn test_open_invalid_path() {
        let dir = std::env::temp_dir().join("rekha_test_invalid_open");
        // Create a file at that path to prevent DB creation
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::write(&dir, "not a rocksdb").unwrap();
        let result = RocksVectorStore::open(&dir);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_delete_mixed_found_missing() {
        let dir = std::env::temp_dir().join("rekha_test_del_mixed");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();
        store.put_vector(1, &[1.0]).unwrap();
        store.put_vector(3, &[3.0]).unwrap();
        let count = store.delete(&[1, 2, 3, 4]).unwrap();
        // delete returns ids.len (batch count), not actual deleted count
        assert_eq!(count, 4);
        assert!(store.get_vector(1).unwrap().is_none());
        assert!(store.get_vector(3).unwrap().is_none());
    }

    #[test]
    fn test_delete_with_payload_removes_vector() {
        let dir = std::env::temp_dir().join("rekha_test_del_payload");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();
        store.put_vector(42, &[1.0]).unwrap();
        store.put_payload(42, b"data").unwrap();
        let count = store.delete(&[42]).unwrap();
        assert_eq!(count, 1); // ids.len = 1
        assert!(store.get_vector(42).unwrap().is_none());
        assert!(store.get_payload(42).unwrap().is_none());
    }

    #[test]
    fn test_put_get_roundtrip_namespace() {
        let dir = std::env::temp_dir().join("rekha_test_ns_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();
        let store_ns = store.clone().with_namespace("col".into());
        store_ns.put_vector(1, &[9.0, 8.0]).unwrap();
        let v = store_ns.get_vector(1).unwrap().unwrap();
        assert!((v[0] - 9.0).abs() < 1e-6);
        assert!((v[1] - 8.0).abs() < 1e-6);
        // Non-namespaced store should not see it
        assert!(store.get_vector(1).unwrap().is_none());
    }

    #[test]
    fn test_put_payload_namespace_isolation() {
        let dir = std::env::temp_dir().join("rekha_test_payload_ns");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();
        let store_ns = store.clone().with_namespace("col".into());
        store_ns.put_payload(5, b"ns-payload").unwrap();
        assert_eq!(store_ns.get_payload(5).unwrap().unwrap(), b"ns-payload");
        assert!(store.get_payload(5).unwrap().is_none());
    }

    #[test]
    fn test_iter_ids_with_namespace() {
        let dir = std::env::temp_dir().join("rekha_test_iter_ns");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();
        let store_ns = store.clone().with_namespace("col".into());

        // Insert into both namespaced and non-namespaced stores
        store.put_vector(10, &[0.1]).unwrap();
        store_ns.put_vector(20, &[0.2]).unwrap();
        store_ns.put_vector(30, &[0.3]).unwrap();

        // Non-namespaced iter should see all
        let mut all = store.iter_ids().unwrap();
        all.sort();
        assert_eq!(all, vec![10, 20, 30]);

        // Namespaced iter should only see its own
        let mut ns_ids = store_ns.iter_ids().unwrap();
        ns_ids.sort();
        assert_eq!(ns_ids, vec![20, 30]);
    }
}
