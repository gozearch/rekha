use crate::node::RaftLogEntry;
use rekha_core::RekhaError;
use rekha_storage::RocksVectorStore;
use rocksdb::{BoundColumnFamily, DBWithThreadMode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub struct RaftLogStore {
    store: Arc<RocksVectorStore>,
    namespace: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct PersistedState {
    term: u64,
    voted_for: Option<String>,
}

fn db(store: &RocksVectorStore) -> &DBWithThreadMode<rocksdb::MultiThreaded> {
    store.db().as_ref()
}

fn namespace_prefix(namespace: &Option<String>) -> Vec<u8> {
    match namespace {
        Some(ns) => {
            let mut buf = Vec::with_capacity(ns.len() + 1);
            buf.extend_from_slice(ns.as_bytes());
            buf.push(0);
            buf
        }
        None => Vec::new(),
    }
}

fn state_key(namespace: &Option<String>, partition_id: u64) -> Vec<u8> {
    let mut key = namespace_prefix(namespace);
    key.extend_from_slice(&partition_id.to_be_bytes());
    key.extend_from_slice(&u64::MAX.to_be_bytes());
    key
}

fn entry_key(namespace: &Option<String>, partition_id: u64, index: u64) -> Vec<u8> {
    let mut key = namespace_prefix(namespace);
    key.extend_from_slice(&partition_id.to_be_bytes());
    key.extend_from_slice(&index.to_be_bytes());
    key
}

fn entry_prefix(namespace: &Option<String>, partition_id: u64) -> Vec<u8> {
    let mut key = namespace_prefix(namespace);
    key.extend_from_slice(&partition_id.to_be_bytes());
    key
}

impl RaftLogStore {
    /// Access the raft_log column family handle.
    fn raft_log_cf(&self) -> Result<Arc<BoundColumnFamily<'_>>, RekhaError> {
        db(&self.store)
            .cf_handle("raft_log")
            .ok_or_else(|| RekhaError::Internal {
                detail: "raft_log column family not found".into(),
            })
    }

    /// Serialize a RaftLogEntry to bytes.
    fn serialize_entry(entry: &RaftLogEntry) -> Result<Vec<u8>, RekhaError> {
        bincode::serialize(entry).map_err(|e| RekhaError::Internal {
            detail: format!("failed to serialize Raft entry: {e}"),
        })
    }

    /// Find the last entry for a partition by reverse-iterating.
    fn last_entry(&self, partition_id: u64) -> Result<Option<RaftLogEntry>, RekhaError> {
        let cf = self.raft_log_cf()?;
        let prefix = entry_prefix(&self.namespace, partition_id);
        let mut start = prefix.clone();
        start.extend_from_slice(&u64::MAX.to_be_bytes());
        let iter = db(&self.store).iterator_cf(
            &cf,
            rocksdb::IteratorMode::From(&start, rocksdb::Direction::Reverse),
        );
        for result in iter {
            let (key, value) = result.map_err(|e| RekhaError::Internal {
                detail: format!("db iteration error: {e}"),
            })?;
            if key.starts_with(&prefix) && key.len() == prefix.len() + 8 {
                return Ok(Some(Self::deserialize_entry(&value)?));
            }
        }
        Ok(None)
    }

    fn deserialize_entry(value: &[u8]) -> Result<RaftLogEntry, RekhaError> {
        bincode::deserialize(value).map_err(|e| RekhaError::Internal {
            detail: format!("failed to deserialize Raft entry: {e}"),
        })
    }
}

impl Clone for RaftLogStore {
    fn clone(&self) -> Self {
        Self {
            store: self.store.clone(),
            namespace: self.namespace.clone(),
        }
    }
}

impl RaftLogStore {
    pub fn new(store: Arc<RocksVectorStore>) -> Self {
        Self {
            store,
            namespace: None,
        }
    }

    pub fn with_namespace(store: Arc<RocksVectorStore>, namespace: String) -> Self {
        Self {
            store,
            namespace: Some(namespace),
        }
    }

    pub fn store_entry(&self, partition_id: u64, entry: &RaftLogEntry) -> Result<(), RekhaError> {
        let key = entry_key(&self.namespace, partition_id, entry.index);
        let value = Self::serialize_entry(entry)?;
        let cf = self.raft_log_cf()?;
        db(&self.store)
            .put_cf(&cf, key, value)
            .map_err(|e| RekhaError::Internal {
                detail: format!("failed to write Raft entry: {e}"),
            })
    }

    pub fn store_entries(
        &self,
        partition_id: u64,
        entries: &[RaftLogEntry],
    ) -> Result<(), RekhaError> {
        let cf = self.raft_log_cf()?;
        let mut batch = rocksdb::WriteBatch::default();
        for entry in entries {
            let key = entry_key(&self.namespace, partition_id, entry.index);
            let value = Self::serialize_entry(entry)?;
            batch.put_cf(&cf, key, value);
        }
        db(&self.store)
            .write(batch)
            .map_err(|e| RekhaError::Internal {
                detail: format!("failed to write Raft entries batch: {e}"),
            })
    }

    pub fn load_entries(
        &self,
        partition_id: u64,
        from_index: u64,
    ) -> Result<Vec<RaftLogEntry>, RekhaError> {
        let cf = self.raft_log_cf()?;
        let prefix = entry_prefix(&self.namespace, partition_id);
        let from_key = entry_key(&self.namespace, partition_id, from_index);
        let mut entries = Vec::new();
        let iter = db(&self.store).iterator_cf(
            &cf,
            rocksdb::IteratorMode::From(&from_key, rocksdb::Direction::Forward),
        );
        for result in iter {
            let (key, value) = result.map_err(|e| RekhaError::Internal {
                detail: format!("db iteration error: {e}"),
            })?;
            if !key.starts_with(&prefix) || key.len() != prefix.len() + 8 {
                break;
            }
            entries.push(Self::deserialize_entry(&value)?);
        }
        Ok(entries)
    }

    pub fn last_log_index(&self, partition_id: u64) -> Result<u64, RekhaError> {
        Ok(self.last_entry(partition_id)?.map(|e| e.index).unwrap_or(0))
    }

    pub fn last_log_term(&self, partition_id: u64) -> Result<u64, RekhaError> {
        Ok(self.last_entry(partition_id)?.map(|e| e.term).unwrap_or(0))
    }

    pub fn truncate_entries(&self, partition_id: u64, from_index: u64) -> Result<(), RekhaError> {
        let cf = self.raft_log_cf()?;
        let prefix = entry_prefix(&self.namespace, partition_id);
        let from_key = entry_key(&self.namespace, partition_id, from_index);
        let mut batch = rocksdb::WriteBatch::default();
        let iter = db(&self.store).iterator_cf(
            &cf,
            rocksdb::IteratorMode::From(&from_key, rocksdb::Direction::Forward),
        );
        for result in iter {
            let (key, _value) = result.map_err(|e| RekhaError::Internal {
                detail: format!("db iteration error: {e}"),
            })?;
            if !key.starts_with(&prefix) || key.len() != prefix.len() + 8 {
                break;
            }
            batch.delete_cf(&cf, &key);
        }
        db(&self.store)
            .write(batch)
            .map_err(|e| RekhaError::Internal {
                detail: format!("failed to truncate Raft entries: {e}"),
            })
    }

    pub fn store_state(
        &self,
        partition_id: u64,
        term: u64,
        voted_for: Option<&str>,
    ) -> Result<(), RekhaError> {
        let key = state_key(&self.namespace, partition_id);
        let state = PersistedState {
            term,
            voted_for: voted_for.map(|s| s.to_string()),
        };
        let value = bincode::serialize(&state).map_err(|e| RekhaError::Internal {
            detail: format!("failed to serialize Raft state: {e}"),
        })?;
        let cf = self.raft_log_cf()?;
        db(&self.store)
            .put_cf(&cf, key, value)
            .map_err(|e| RekhaError::Internal {
                detail: format!("failed to write Raft state: {e}"),
            })
    }

    pub fn load_state(&self, partition_id: u64) -> Result<(u64, Option<String>), RekhaError> {
        let key = state_key(&self.namespace, partition_id);
        let cf = self.raft_log_cf()?;
        match db(&self.store).get_cf(&cf, key) {
            Ok(Some(value)) => {
                let state: PersistedState =
                    bincode::deserialize(&value).map_err(|e| RekhaError::Internal {
                        detail: format!("failed to deserialize Raft state: {e}"),
                    })?;
                Ok((state.term, state.voted_for))
            }
            Ok(None) => Ok((0, None)),
            Err(e) => Err(RekhaError::Internal {
                detail: format!("failed to read Raft state: {e}"),
            }),
        }
    }

    #[cfg(test)]
    fn test_store() -> (Self, std::sync::Arc<rekha_storage::RocksVectorStore>) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("rekha_raft_log_test_{}", n));
        let _ = std::fs::remove_dir_all(&dir);
        let store = std::sync::Arc::new(rekha_storage::RocksVectorStore::open(&dir).unwrap());
        (Self::new(store.clone()), store)
    }

    pub fn entry_count(&self, partition_id: u64) -> Result<u64, RekhaError> {
        let cf = self.raft_log_cf()?;
        let prefix = entry_prefix(&self.namespace, partition_id);
        let mut count = 0u64;
        let iter = db(&self.store).iterator_cf(
            &cf,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        );
        for result in iter {
            let (key, _) = result.map_err(|e| RekhaError::Internal {
                detail: format!("db iteration error: {e}"),
            })?;
            if !key.starts_with(&prefix) || key.len() != prefix.len() + 8 {
                break;
            }
            count += 1;
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::RaftCommand;

    #[test]
    fn test_store_entry_and_load() {
        let (log_store, _store) = RaftLogStore::test_store();
        let entry = RaftLogEntry {
            term: 1,
            index: 1,
            command: RaftCommand::NoOp,
        };
        log_store.store_entry(0, &entry).unwrap();
        let entries = log_store.load_entries(0, 1).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].term, 1);
        assert_eq!(entries[0].index, 1);
    }

    #[test]
    fn test_store_entries_batch() {
        let (log_store, _store) = RaftLogStore::test_store();
        let entries: Vec<RaftLogEntry> = (1..=5)
            .map(|i| RaftLogEntry {
                term: 1,
                index: i,
                command: RaftCommand::NoOp,
            })
            .collect();
        log_store.store_entries(0, &entries).unwrap();
        let loaded = log_store.load_entries(0, 1).unwrap();
        assert_eq!(loaded.len(), 5);
    }

    #[test]
    fn test_load_entries_from_index() {
        let (log_store, _store) = RaftLogStore::test_store();
        let entries: Vec<RaftLogEntry> = (1..=10)
            .map(|i| RaftLogEntry {
                term: 1,
                index: i,
                command: RaftCommand::NoOp,
            })
            .collect();
        log_store.store_entries(0, &entries).unwrap();
        let loaded = log_store.load_entries(0, 5).unwrap();
        assert_eq!(loaded.len(), 6);
        assert_eq!(loaded[0].index, 5);
    }

    #[test]
    fn test_last_log_index_empty() {
        let (log_store, _store) = RaftLogStore::test_store();
        assert_eq!(log_store.last_log_index(0).unwrap(), 0);
    }

    #[test]
    fn test_last_log_index_after_insert() {
        let (log_store, _store) = RaftLogStore::test_store();
        let entries: Vec<RaftLogEntry> = (1..=3)
            .map(|i| RaftLogEntry {
                term: 1,
                index: i,
                command: RaftCommand::NoOp,
            })
            .collect();
        log_store.store_entries(0, &entries).unwrap();
        assert_eq!(log_store.last_log_index(0).unwrap(), 3);
    }

    #[test]
    fn test_last_log_term_empty() {
        let (log_store, _store) = RaftLogStore::test_store();
        assert_eq!(log_store.last_log_term(0).unwrap(), 0);
    }

    #[test]
    fn test_last_log_term_after_insert() {
        let (log_store, _store) = RaftLogStore::test_store();
        let entry = RaftLogEntry {
            term: 5,
            index: 1,
            command: RaftCommand::NoOp,
        };
        log_store.store_entry(0, &entry).unwrap();
        assert_eq!(log_store.last_log_term(0).unwrap(), 5);
    }

    #[test]
    fn test_truncate_entries() {
        let (log_store, _store) = RaftLogStore::test_store();
        let entries: Vec<RaftLogEntry> = (1..=5)
            .map(|i| RaftLogEntry {
                term: 1,
                index: i,
                command: RaftCommand::NoOp,
            })
            .collect();
        log_store.store_entries(0, &entries).unwrap();
        log_store.truncate_entries(0, 3).unwrap();
        let loaded = log_store.load_entries(0, 1).unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_store_state_and_load() {
        let (log_store, _store) = RaftLogStore::test_store();
        log_store.store_state(0, 7, Some("node-1")).unwrap();
        let (term, voted_for) = log_store.load_state(0).unwrap();
        assert_eq!(term, 7);
        assert_eq!(voted_for, Some("node-1".to_string()));
    }

    #[test]
    fn test_store_state_none_voted() {
        let (log_store, _store) = RaftLogStore::test_store();
        log_store.store_state(0, 3, None).unwrap();
        let (term, voted_for) = log_store.load_state(0).unwrap();
        assert_eq!(term, 3);
        assert_eq!(voted_for, None);
    }

    #[test]
    fn test_entry_count() {
        let (log_store, _store) = RaftLogStore::test_store();
        assert_eq!(log_store.entry_count(0).unwrap(), 0);
        let entries: Vec<RaftLogEntry> = (1..=7)
            .map(|i| RaftLogEntry {
                term: 1,
                index: i,
                command: RaftCommand::NoOp,
            })
            .collect();
        log_store.store_entries(0, &entries).unwrap();
        assert_eq!(log_store.entry_count(0).unwrap(), 7);
    }

    #[test]
    fn test_multiple_partitions() {
        let (log_store, _store) = RaftLogStore::test_store();
        let e1 = RaftLogEntry {
            term: 1,
            index: 1,
            command: RaftCommand::NoOp,
        };
        let e2 = RaftLogEntry {
            term: 1,
            index: 1,
            command: RaftCommand::NoOp,
        };
        log_store.store_entry(0, &e1).unwrap();
        log_store.store_entry(1, &e2).unwrap();
        assert_eq!(log_store.last_log_index(0).unwrap(), 1);
        assert_eq!(log_store.last_log_index(1).unwrap(), 1);
    }

    #[test]
    fn test_namespace_prefix_with_ns() {
        let ns = Some("test_collection".to_string());
        let prefix = super::namespace_prefix(&ns);
        assert_eq!(prefix, b"test_collection\0");
    }

    #[test]
    fn test_namespace_prefix_none() {
        let ns: Option<String> = None;
        let prefix = super::namespace_prefix(&ns);
        assert!(prefix.is_empty());
    }

    #[test]
    fn test_with_namespace_isolation() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CNT: AtomicU64 = AtomicU64::new(0);
        let n = CNT.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("rekha_raft_ns_test_{}", n));
        let _ = std::fs::remove_dir_all(&dir);
        let store = std::sync::Arc::new(rekha_storage::RocksVectorStore::open(&dir).unwrap());
        let ns_store = RaftLogStore::with_namespace(store.clone(), "col1".into());
        let ns_store2 = RaftLogStore::with_namespace(store.clone(), "col2".into());
        let entry = RaftLogEntry {
            term: 1,
            index: 1,
            command: RaftCommand::NoOp,
        };
        ns_store.store_entry(0, &entry).unwrap();
        ns_store2.store_entry(0, &entry).unwrap();
        assert_eq!(ns_store.entry_count(0).unwrap(), 1);
        assert_eq!(ns_store2.entry_count(0).unwrap(), 1);
    }

    #[test]
    fn test_truncate_entries_empty() {
        let (log_store, _store) = RaftLogStore::test_store();
        log_store.truncate_entries(0, 1).unwrap();
        assert_eq!(log_store.entry_count(0).unwrap(), 0);
    }

    #[test]
    fn test_truncate_entries_start_beyond_len() {
        let (log_store, _store) = RaftLogStore::test_store();
        let entries: Vec<RaftLogEntry> = (1..=3)
            .map(|i| RaftLogEntry {
                term: 1,
                index: i,
                command: RaftCommand::NoOp,
            })
            .collect();
        log_store.store_entries(0, &entries).unwrap();
        log_store.truncate_entries(0, 10).unwrap();
        assert_eq!(log_store.entry_count(0).unwrap(), 3);
    }

    #[test]
    fn test_load_entries_empty_partition() {
        let (log_store, _store) = RaftLogStore::test_store();
        let entries = log_store.load_entries(0, 1).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_entry_count_multiple_partitions() {
        let (log_store, _store) = RaftLogStore::test_store();
        for i in 1..=3 {
            let entry = RaftLogEntry {
                term: 1,
                index: i,
                command: RaftCommand::NoOp,
            };
            log_store.store_entry(0, &entry).unwrap();
        }
        assert_eq!(log_store.entry_count(0).unwrap(), 3);
        assert_eq!(log_store.entry_count(1).unwrap(), 0);
    }
}
