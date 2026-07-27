use std::sync::Arc;

use rekha_core::RekhaError;
use rocksdb::{IteratorMode, DB};

pub struct HintStore {
    db: Arc<DB>,
    cf: String,
}

impl HintStore {
    pub fn new(db: Arc<DB>) -> Self {
        HintStore {
            db,
            cf: "hints".to_string(),
        }
    }

    fn cf_handle(&self) -> Result<&rocksdb::ColumnFamily, RekhaError> {
        self.db
            .cf_handle(&self.cf)
            .ok_or_else(|| RekhaError::Internal("hints cf not found".into()))
    }

    pub fn put_hint(&self, key: &[u8], value: &[u8]) -> Result<(), RekhaError> {
        let cf = self.cf_handle()?;
        self.db
            .put_cf(cf, key, value)
            .map_err(|e| RekhaError::Storage(e.to_string()))
    }

    pub fn iter_hints_for_node(
        &self,
        node_prefix: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>, RekhaError> {
        let cf = self.cf_handle()?;
        let iter = self.db.iterator_cf(
            cf,
            IteratorMode::From(node_prefix, rocksdb::Direction::Forward),
        );
        let mut results = Vec::new();
        for item in iter {
            let (key, value) = item.map_err(|e| RekhaError::Storage(e.to_string()))?;
            if !key.starts_with(node_prefix) {
                break;
            }
            results.push((key.to_vec(), value.to_vec()));
        }
        Ok(results)
    }

    pub fn delete_hint(&self, key: &[u8]) -> Result<(), RekhaError> {
        let cf = self.cf_handle()?;
        self.db
            .delete_cf(cf, key)
            .map_err(|e| RekhaError::Storage(e.to_string()))
    }

    pub fn hint_count(&self) -> Result<u64, RekhaError> {
        let cf = self.cf_handle()?;
        let iter = self.db.iterator_cf(cf, IteratorMode::Start);
        let mut count = 0u64;
        for item in iter {
            let _ = item.map_err(|e| RekhaError::Storage(e.to_string()))?;
            count += 1;
        }
        Ok(count)
    }

    pub fn iter_all(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>, RekhaError> {
        let cf = self.cf_handle()?;
        let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);
        let mut results = Vec::new();
        for item in iter {
            let (key, value) = item.map_err(|e| RekhaError::Storage(e.to_string()))?;
            results.push((key.to_vec(), value.to_vec()));
        }
        Ok(results)
    }

    pub fn delete_expired_hints(&self, cutoff_prefix: &[u8]) -> Result<u64, RekhaError> {
        let cf = self.cf_handle()?;
        let iter = self.db.iterator_cf(cf, IteratorMode::Start);
        let mut deleted = 0u64;
        for item in iter {
            let (key, _) = item.map_err(|e| RekhaError::Storage(e.to_string()))?;
            if key.as_ref() <= cutoff_prefix {
                let _ = self.db.delete_cf(cf, key.as_ref());
                deleted += 1;
            }
        }
        self.db
            .flush()
            .map_err(|e| RekhaError::Storage(e.to_string()))?;
        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup() -> (TempDir, HintStore) {
        let dir = TempDir::new().unwrap();
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        let db = DB::open_cf_descriptors(
            &opts,
            dir.path(),
            vec![rocksdb::ColumnFamilyDescriptor::new(
                "hints",
                rocksdb::Options::default(),
            )],
        )
        .unwrap();
        let store = HintStore::new(Arc::new(db));
        (dir, store)
    }

    #[test]
    fn test_hint_put_get_delete() {
        let (_dir, store) = setup();
        store.put_hint(b"node1:hint1", b"value1").unwrap();
        let results = store.iter_hints_for_node(b"node1:").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, b"value1");

        store.delete_hint(b"node1:hint1").unwrap();
        let results = store.iter_hints_for_node(b"node1:").unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_iter_hints_multiple_nodes() {
        let (_dir, store) = setup();
        store.put_hint(b"node1:a", b"val1").unwrap();
        store.put_hint(b"node1:b", b"val2").unwrap();
        store.put_hint(b"node2:a", b"val3").unwrap();

        let r1 = store.iter_hints_for_node(b"node1:").unwrap();
        assert_eq!(r1.len(), 2);

        let r2 = store.iter_hints_for_node(b"node2:").unwrap();
        assert_eq!(r2.len(), 1);
    }
}
