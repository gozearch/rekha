use rekha_core::{RekhaError, now_epoch_secs};
use rekha_storage::{HintEntry, HintStore};

pub struct HintedHandoff {
    enabled: bool,
    max_window_secs: u64,
}

impl HintedHandoff {
    pub fn new(enabled: bool, max_window_secs: u64) -> Self {
        Self { enabled, max_window_secs }
    }

    pub fn is_enabled(&self) -> bool { self.enabled }

    pub fn store_hint(
        &self, hint_store: &HintStore, target_node_id: &str, collection: &str,
        id: u64, vector: &[f32], payload: Option<&[u8]>, timestamp: u64,
    ) {
        if !self.enabled { return; }
        let _ = hint_store.put_hint(target_node_id, collection, id, vector, payload, timestamp);
    }

    pub fn store_collection_hint(
        &self, hint_store: &HintStore, target_node_id: &str, collection: &str,
        config_bytes: &[u8], timestamp: u64, op: u8,
    ) {
        if !self.enabled { return; }
        let _ = hint_store.put_collection_hint(target_node_id, collection, config_bytes, timestamp, op);
    }

    pub fn replay_hints(&self, hint_store: &HintStore, peer_id: &str) -> Result<Vec<HintEntry>, RekhaError> {
        let hints = hint_store.iter_hints_for_node(peer_id)?;
        let cutoff = now_epoch_secs().saturating_sub(self.max_window_secs);
        Ok(hints.into_iter()
            .filter(|h| h.timestamp / 1_000_000 >= cutoff)
            .collect())
    }

    pub fn delete_hint(&self, hint_store: &HintStore, target: &str, collection: &str, id: u64) {
        let _ = hint_store.delete_hint(target, collection, id);
    }

    pub fn delete_collection_hint(&self, hint_store: &HintStore, target: &str, collection: &str) {
        let _ = hint_store.delete_collection_hint(target, collection);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rekha_storage::RocksVectorStore;

    fn setup() -> (HintedHandoff, HintStore) {
        let hh = HintedHandoff::new(true, 10800);
        let dir = tempfile::TempDir::new().unwrap();
        let store = RocksVectorStore::open(dir.path()).unwrap();
        let hint_store = HintStore::new(store.db().clone());
        (hh, hint_store)
    }

    #[test]
    fn test_store_and_replay() {
        let (hh, hints) = setup();
        let now = rekha_core::now_micros();
        hh.store_hint(&hints, "node2", "col1", 1, &[1.0, 2.0], None, now);
        let replayed = hh.replay_hints(&hints, "node2").unwrap();
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].id, 1);
    }

    #[test]
    fn test_expired_hint_dropped() {
        let (hh, hints) = setup();
        // Timestamp of 1 microsecond since epoch = unix epoch, definitely expired
        hh.store_hint(&hints, "node2", "col1", 1, &[1.0], None, 1);
        let replayed = hh.replay_hints(&hints, "node2").unwrap();
        assert!(replayed.is_empty());
    }

    #[test]
    fn test_disabled_noop() {
        let hh = HintedHandoff::new(false, 10800);
        let dir = tempfile::TempDir::new().unwrap();
        let store = RocksVectorStore::open(dir.path()).unwrap();
        let hints = HintStore::new(store.db().clone());
        hh.store_hint(&hints, "node2", "col1", 1, &[1.0], None, 1_000_000);
        assert!(hh.replay_hints(&hints, "node2").unwrap().is_empty());
    }
}
