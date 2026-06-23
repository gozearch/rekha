use crate::store::RocksVectorStore;
use rekha_core::{RekhaError, StorageError};
use rocksdb::WriteBatch as RocksWriteBatch;

const CF_VECTORS: &str = "vectors";
const CF_PAYLOADS: &str = "payloads";
const CF_METADATA: &str = "metadata";

/// A batch of writes that are applied atomically to RocksDB.
///
/// This ensures that vector data, payloads, and metadata are always consistent.
pub struct WriteBatch<'a> {
    store: &'a RocksVectorStore,
    batch: RocksWriteBatch,
    pending: usize,
}

impl<'a> WriteBatch<'a> {
    pub fn new(store: &'a RocksVectorStore) -> Self {
        Self {
            store,
            batch: RocksWriteBatch::default(),
            pending: 0,
        }
    }

    pub fn put_vector(mut self, id: u64, timestamp: u64, data: &[f32]) -> Self {
        let key = self.store.encode_key(id);
        let value = RocksVectorStore::encode_vector_value(timestamp, 0x00, data);
        let cf = self.store.db().cf_handle(CF_VECTORS).unwrap();
        self.batch.put_cf(&cf, key, value);
        self.pending += 1;
        self
    }

    pub fn put_payload(mut self, id: u64, payload: &[u8]) -> Self {
        let key = self.store.encode_key(id);
        let cf = self.store.db().cf_handle(CF_PAYLOADS).unwrap();
        self.batch.put_cf(&cf, key, payload);
        self.pending += 1;
        self
    }

    pub fn put_tombstone(mut self, id: u64, timestamp: u64) -> Self {
        let key = self.store.encode_key(id);
        let value = RocksVectorStore::encode_vector_value(timestamp, 0x01, &[]);
        let cf_v = self.store.db().cf_handle(CF_VECTORS).unwrap();
        let cf_p = self.store.db().cf_handle(CF_PAYLOADS).unwrap();
        self.batch.put_cf(&cf_v, key.clone(), value);
        self.batch.delete_cf(&cf_p, key);
        self.pending += 2;
        self
    }

    pub fn put_metadata(mut self, key: &[u8], value: &[u8]) -> Self {
        let cf = self.store.db().cf_handle(CF_METADATA).unwrap();
        self.batch.put_cf(&cf, key, value);
        self.pending += 1;
        self
    }

    /// Commit the batch atomically.
    pub fn commit(self) -> Result<(), RekhaError> {
        if self.pending == 0 {
            return Ok(());
        }
        self.store.db().write(self.batch).map_err(|e| {
            StorageError::BatchWrite {
                committed: 0,
                failed: self.pending,
                msg: e.to_string(),
            }
            .into()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::RocksVectorStore;
    use rekha_core::VectorStoreBackend;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn setup_store(name: &str) -> RocksVectorStore {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("{}_{}", name, id));
        let _ = std::fs::remove_dir_all(&dir);
        RocksVectorStore::open(&dir).unwrap()
    }

    #[test]
    fn test_write_batch_vector_roundtrip() {
        let store = setup_store("rekha_test_batch_vec");
        let batch = WriteBatch::new(&store).put_vector(42, 100, &[1.0, 2.0, 3.0, 4.0]);
        batch.commit().unwrap();

        let rec = store.get_vector_record(42).unwrap().unwrap();
        assert!(!rec.is_tombstone);
        assert_eq!(rec.timestamp, 100);
        assert_eq!(rec.data, Some(vec![1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_write_batch_tombstone_roundtrip() {
        let store = setup_store("rekha_test_batch_ts");
        let batch = WriteBatch::new(&store)
            .put_vector(1, 100, &[10.0])
            .put_tombstone(1, 200);
        batch.commit().unwrap();

        let rec = store.get_vector_record(1).unwrap().unwrap();
        assert!(rec.is_tombstone);
        assert_eq!(rec.timestamp, 200);
        assert!(store.get_vector(1).unwrap().is_none());
    }

    #[test]
    fn test_write_batch_commit() {
        let store = setup_store("rekha_test_batch");
        let batch = WriteBatch::new(&store).put_payload(1, b"payload1");
        batch.commit().unwrap();

        assert_eq!(store.get_payload(1).unwrap().unwrap(), b"payload1");
    }

    #[test]
    fn test_write_batch_empty() {
        let store = setup_store("rekha_test_batch_empty");
        let batch = WriteBatch::new(&store);
        batch.commit().unwrap();
    }

    #[test]
    fn test_write_batch_metadata() {
        let store = setup_store("rekha_test_batch_meta");
        let batch = WriteBatch::new(&store).put_metadata(b"cluster_config", br#"{"key": "value"}"#);
        batch.commit().unwrap();

        let cf = store.db().cf_handle("metadata").unwrap();
        let val = store.db().get_cf(&cf, b"cluster_config").unwrap().unwrap();
        assert_eq!(val, br#"{"key": "value"}"#);
    }
}
