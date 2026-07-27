use rekha_core::RekhaError;
use rekha_storage::HintStore;

pub struct HintedHandoff {
    store: HintStore,
    enabled: bool,
    max_window_secs: i64,
}

impl HintedHandoff {
    pub fn new(store: HintStore, enabled: bool, max_window_secs: i64) -> Self {
        HintedHandoff {
            store,
            enabled,
            max_window_secs,
        }
    }

    pub fn store_hint(
        &self,
        target_node: &str,
        collection: &str,
        id: u64,
        data: &[u8],
        timestamp: i64,
    ) -> Result<(), RekhaError> {
        if !self.enabled {
            return Ok(());
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        if timestamp < now - self.max_window_secs {
            return Ok(());
        }
        let key = format!("{}:{}:{}", target_node, collection, id);
        self.store.put_hint(key.as_bytes(), data)
    }

    pub fn drain_hints(&self, target_node: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>, RekhaError> {
        let prefix = format!("{}:", target_node);
        self.store.iter_hints_for_node(prefix.as_bytes())
    }

    pub fn delete_hint(
        &self,
        target_node: &str,
        collection: &str,
        id: u64,
    ) -> Result<(), RekhaError> {
        let key = format!("{}:{}:{}", target_node, collection, id);
        self.store.delete_hint(key.as_bytes())
    }

    pub fn delete_hint_by_key(&self, key: &[u8]) -> Result<(), RekhaError> {
        self.store.delete_hint(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocksdb::DB;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn setup() -> (TempDir, HintedHandoff) {
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
        let hint_store = HintStore::new(Arc::new(db));
        let hh = HintedHandoff::new(hint_store, true, 3600);
        (dir, hh)
    }

    #[test]
    fn test_store_and_drain() {
        let (_dir, hh) = setup();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        hh.store_hint("node1", "col1", 1, b"data1", now).unwrap();
        hh.store_hint("node1", "col1", 2, b"data2", now).unwrap();
        hh.store_hint("node2", "col1", 3, b"data3", now).unwrap();

        let hints = hh.drain_hints("node1").unwrap();
        assert_eq!(hints.len(), 2);

        hh.delete_hint("node1", "col1", 1).unwrap();
        let hints = hh.drain_hints("node1").unwrap();
        assert_eq!(hints.len(), 1);
    }

    #[test]
    fn test_disabled() {
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
        let hint_store = HintStore::new(Arc::new(db));
        let hh = HintedHandoff::new(hint_store, false, 3600);
        hh.store_hint("node1", "col1", 1, b"data", 1000).unwrap();
        let hints = hh.drain_hints("node1").unwrap();
        assert_eq!(hints.len(), 0);
    }
}
