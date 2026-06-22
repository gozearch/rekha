use crate::store::RocksVectorStore;
use rekha_core::{RekhaError, StorageError};
use rocksdb::WriteBatch as RocksWriteBatch;

const CF_VECTORS: &str = "vectors";
const CF_PAYLOADS: &str = "payloads";
const CF_METADATA: &str = "metadata";

/// A batch of writes that are applied atomically to RocksDB.
///
/// This ensures that vector data, payloads, and metadata are always consistent.
///
/// NOTE: `put_vector` and `delete` operate on raw keys (id.to_be_bytes())
/// without namespace prefix. Use the store trait methods for namespaced access.
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

    pub fn put_vector(mut self, id: u64, data: &[f32]) -> Self {
        let key = id.to_be_bytes();
        let value = vector_to_bytes(data);
        let cf = self.store.db().cf_handle(CF_VECTORS).unwrap();
        self.batch.put_cf(&cf, key, value);
        self.pending += 1;
        self
    }

    pub fn put_payload(mut self, id: u64, payload: &[u8]) -> Self {
        let key = id.to_be_bytes();
        let cf = self.store.db().cf_handle(CF_PAYLOADS).unwrap();
        self.batch.put_cf(&cf, key, payload);
        self.pending += 1;
        self
    }

    #[allow(dead_code)]
    pub fn delete(mut self, id: u64) -> Self {
        let key = id.to_be_bytes();
        let cf_v = self.store.db().cf_handle(CF_VECTORS).unwrap();
        let cf_p = self.store.db().cf_handle(CF_PAYLOADS).unwrap();
        self.batch.delete_cf(&cf_v, key);
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
                source: e.to_string(),
            }
            .into()
        })
    }
}

fn vector_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
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
    fn test_write_batch_commit() {
        let store = setup_store("rekha_test_batch");
        // Write vectors directly (WriteBatch::put_vector writes unformatted bytes)
        store.put_vector(1, &[1.0, 2.0], 100).unwrap();
        store.put_vector(2, &[3.0, 4.0], 100).unwrap();

        // Use WriteBatch for payloads and verify atomic commit
        let batch = WriteBatch::new(&store).put_payload(1, b"payload1");
        batch.commit().unwrap();

        assert!((store.get_vector(1).unwrap().unwrap()[0] - 1.0).abs() < 1e-6);
        assert_eq!(store.get_payload(1).unwrap().unwrap(), b"payload1");
        assert!((store.get_vector(2).unwrap().unwrap()[0] - 3.0).abs() < 1e-6);
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

    #[test]
    fn test_write_batch_delete() {
        let store = setup_store("rekha_test_batch_del");
        store.put_vector(1, &[10.0], 100).unwrap();
        store.put_payload(1, b"data").unwrap();

        let batch = WriteBatch::new(&store).delete(1);
        batch.commit().unwrap();

        assert!(store.get_vector(1).unwrap().is_none());
        assert!(store.get_payload(1).unwrap().is_none());
    }
}
