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
#[derive(Clone)]
pub struct RocksVectorStore {
    db: Arc<DBWithThreadMode<MultiThreaded>>,
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
            max_payload_size: 1024 * 1024,
        })
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

    /// Encode a u64 ID into a big-endian key for sorted iteration.
    fn encode_key(id: u64) -> Vec<u8> {
        id.to_be_bytes().to_vec()
    }

    /// Decode a big-endian key back into a u64 ID.
    fn decode_key(key: &[u8]) -> Option<u64> {
        if key.len() == 8 {
            Some(u64::from_be_bytes(key.try_into().ok()?))
        } else {
            None
        }
    }
}

impl VectorStoreBackend for RocksVectorStore {
    fn put_vector(&self, id: u64, data: &[f32]) -> Result<(), RekhaError> {
        let key = Self::encode_key(id);
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
        let key = Self::encode_key(id);
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
        let key = Self::encode_key(id);
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
        let key = Self::encode_key(id);
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

        let mut deleted = 0u64;
        for id in ids {
            let key = Self::encode_key(*id);
            self.db
                .delete_cf(&cf_vec, &key)
                .map_err(|e| StorageError::Write {
                    source: e.to_string(),
                })?;
            self.db
                .delete_cf(&cf_pay, &key)
                .map_err(|e| StorageError::Write {
                    source: e.to_string(),
                })?;
            deleted += 1;
        }
        Ok(deleted)
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
        let iter = self.db.iterator_cf(&cf, IteratorMode::Start);
        for result in iter {
            let (key, _) = result.map_err(|e| RekhaError::Internal {
                detail: format!("db iteration error: {e}"),
            })?;
            if let Some(id) = Self::decode_key(&key) {
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
}
