//! Disk-backed Raft log storage using redb.

#![allow(clippy::result_large_err)]

use std::fmt::Debug;
use std::ops::RangeBounds;
use std::path::Path;
use std::sync::Arc;

use openraft::storage::{LogFlushed, RaftLogReader, RaftLogStorage, RaftSnapshotBuilder};
use openraft::{
    CommittedLeaderId, Entry, LogId, LogState, Snapshot, SnapshotMeta, StorageError,
    StoredMembership, Vote,
};
use redb::{Database, ReadableDatabase, TableDefinition};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::raft_types::RaftTypeConfig;

const LOG_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("raft_log");
const VOTE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("raft_vote");
const STATE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("raft_state");

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializedState {
    last_purged_index: Option<u64>,
    last_purged_term: Option<u64>,
}

impl SerializedState {
    fn last_purged_log_id(&self) -> Option<LogId<u64>> {
        match (self.last_purged_index, self.last_purged_term) {
            (Some(idx), Some(term)) => Some(LogId::new(CommittedLeaderId::new(term, 0), idx)),
            _ => None,
        }
    }
}

fn io_error(msg: impl ToString) -> StorageError<u64> {
    StorageError::IO {
        source: openraft::StorageIOError::read(&std::io::Error::other(msg.to_string())),
    }
}

/// Persistent Raft log store backed by redb.
pub struct RedbLogStore {
    db: Arc<Database>,
    inner: Arc<Mutex<RedbLogStoreInner>>,
}

struct RedbLogStoreInner {
    last_purged: Option<LogId<u64>>,
}

#[derive(Clone)]
pub struct RedbLogReader {
    db: Arc<Database>,
}

fn read_log_entries(
    db: &Database,
    start: u64,
    end: u64,
) -> Result<Vec<Entry<RaftTypeConfig>>, StorageError<u64>> {
    let read_txn = db.begin_read().map_err(io_error)?;
    let table = match read_txn.open_table(LOG_TABLE) {
        Ok(t) => t,
        Err(_) => return Ok(Vec::new()),
    };

    let mut entries = Vec::new();
    for idx in start..end {
        let key = idx.to_string();
        if let Ok(Some(val)) = table.get(key.as_str())
            && let Ok(entry) = bincode::deserialize::<Entry<RaftTypeConfig>>(val.value())
        {
            entries.push(entry);
        }
    }
    Ok(entries)
}

fn find_last_log_id(db: &Database) -> Result<Option<LogId<u64>>, StorageError<u64>> {
    let read_txn = db.begin_read().map_err(io_error)?;
    let table = match read_txn.open_table(LOG_TABLE) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    for idx in (0..1_000_000u64).rev() {
        let key = idx.to_string();
        if let Ok(Some(val)) = table.get(key.as_str())
            && let Ok(entry) = bincode::deserialize::<Entry<RaftTypeConfig>>(val.value())
        {
            return Ok(Some(entry.log_id));
        }
    }
    Ok(None)
}

impl RedbLogStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError<u64>> {
        let db = Database::create(path).map_err(io_error)?;

        // Ensure all tables exist (open_table in write txn creates them).
        {
            let write_txn = db.begin_write().map_err(io_error)?;
            {
                let _ = write_txn.open_table(LOG_TABLE);
                let _ = write_txn.open_table(VOTE_TABLE);
                let _ = write_txn.open_table(STATE_TABLE);
            }
            write_txn.commit().map_err(io_error)?;
        }

        // Read persisted state
        let last_purged = {
            let read_txn = db.begin_read().map_err(io_error)?;
            if let Ok(table) = read_txn.open_table(STATE_TABLE) {
                if let Ok(Some(val)) = table.get("state") {
                    bincode::deserialize::<SerializedState>(val.value())
                        .ok()
                        .and_then(|s| s.last_purged_log_id())
                } else {
                    None
                }
            } else {
                None
            }
        };

        Ok(Self {
            db: Arc::new(db),
            inner: Arc::new(Mutex::new(RedbLogStoreInner { last_purged })),
        })
    }

    fn save_state(&self, inner: &RedbLogStoreInner) -> Result<(), StorageError<u64>> {
        let state = SerializedState {
            last_purged_index: inner.last_purged.map(|l| l.index),
            last_purged_term: inner.last_purged.map(|l| l.leader_id.term),
        };
        let bytes = bincode::serialize(&state).map_err(io_error)?;
        let write_txn = self.db.begin_write().map_err(io_error)?;
        {
            let mut table = write_txn.open_table(STATE_TABLE).map_err(io_error)?;
            table.insert("state", bytes.as_slice()).map_err(io_error)?;
        }
        write_txn.commit().map_err(io_error)?;
        Ok(())
    }
}

impl RaftLogReader<RaftTypeConfig> for RedbLogStore {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug + Send>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<RaftTypeConfig>>, StorageError<u64>> {
        let start = match range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => 1_000_000,
        };
        read_log_entries(&self.db, start, end)
    }
}

impl RaftLogReader<RaftTypeConfig> for RedbLogReader {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug + Send>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<RaftTypeConfig>>, StorageError<u64>> {
        let start = match range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => 1_000_000,
        };
        read_log_entries(&self.db, start, end)
    }
}

impl RaftLogStorage<RaftTypeConfig> for RedbLogStore {
    type LogReader = RedbLogReader;

    async fn get_log_state(&mut self) -> Result<LogState<RaftTypeConfig>, StorageError<u64>> {
        let inner = self.inner.lock().await;
        let last_log_id = find_last_log_id(&self.db)?.or(inner.last_purged);
        Ok(LogState {
            last_purged_log_id: inner.last_purged,
            last_log_id,
        })
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        RedbLogReader {
            db: self.db.clone(),
        }
    }

    async fn save_vote(&mut self, vote: &Vote<u64>) -> Result<(), StorageError<u64>> {
        let bytes = bincode::serialize(vote).map_err(io_error)?;
        let write_txn = self.db.begin_write().map_err(io_error)?;
        {
            let mut table = write_txn.open_table(VOTE_TABLE).map_err(io_error)?;
            table.insert("vote", bytes.as_slice()).map_err(io_error)?;
        }
        write_txn.commit().map_err(io_error)?;
        Ok(())
    }

    async fn read_vote(&mut self) -> Result<Option<Vote<u64>>, StorageError<u64>> {
        let read_txn = self.db.begin_read().map_err(io_error)?;
        let table = match read_txn.open_table(VOTE_TABLE) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        if let Ok(Some(val)) = table.get("vote") {
            let vote: Vote<u64> = bincode::deserialize(val.value()).map_err(io_error)?;
            return Ok(Some(vote));
        }
        Ok(None)
    }

    async fn append<I>(
        &mut self,
        entries: I,
        callback: LogFlushed<RaftTypeConfig>,
    ) -> Result<(), StorageError<u64>>
    where
        I: IntoIterator<Item = Entry<RaftTypeConfig>> + Send,
        I::IntoIter: Send,
    {
        let entries: Vec<_> = entries.into_iter().collect();
        let write_txn = self.db.begin_write().map_err(io_error)?;
        {
            let mut table = write_txn.open_table(LOG_TABLE).map_err(io_error)?;
            for entry in &entries {
                let key = entry.log_id.index.to_string();
                let bytes = bincode::serialize(entry).map_err(io_error)?;
                table
                    .insert(key.as_str(), bytes.as_slice())
                    .map_err(io_error)?;
            }
        }
        write_txn.commit().map_err(io_error)?;
        callback.log_io_completed(Ok(()));
        Ok(())
    }

    async fn truncate(&mut self, log_id: LogId<u64>) -> Result<(), StorageError<u64>> {
        let write_txn = self.db.begin_write().map_err(io_error)?;
        {
            let mut table = write_txn.open_table(LOG_TABLE).map_err(io_error)?;
            // Remove entries at and after log_id.index; stop when we hit a gap
            for idx in log_id.index..=log_id.index + 10_000 {
                let key = idx.to_string();
                if table.remove(key.as_str()).is_err() {
                    break;
                }
            }
        }
        write_txn.commit().map_err(io_error)?;
        Ok(())
    }

    async fn purge(&mut self, log_id: LogId<u64>) -> Result<(), StorageError<u64>> {
        let write_txn = self.db.begin_write().map_err(io_error)?;
        {
            let mut table = write_txn.open_table(LOG_TABLE).map_err(io_error)?;
            for idx in 0..=log_id.index {
                let key = idx.to_string();
                let _ = table.remove(key.as_str());
            }
        }
        write_txn.commit().map_err(io_error)?;

        let mut inner = self.inner.lock().await;
        inner.last_purged = Some(log_id);
        self.save_state(&inner)?;
        Ok(())
    }
}

pub struct RedbSnapshotBuilder {
    db: Arc<Database>,
    inner: Arc<Mutex<RedbLogStoreInner>>,
}

impl RaftSnapshotBuilder<RaftTypeConfig> for RedbSnapshotBuilder {
    async fn build_snapshot(&mut self) -> Result<Snapshot<RaftTypeConfig>, StorageError<u64>> {
        let inner = self.inner.lock().await;
        let last_purged = inner.last_purged;
        drop(inner);

        let last_log_id = find_last_log_id(&self.db)?.or(last_purged);

        let snapshot_id = uuid::Uuid::new_v4().to_string();
        let data = vec![];
        let cursor = std::io::Cursor::new(data);

        let meta = SnapshotMeta {
            last_log_id,
            last_membership: StoredMembership::default(),
            snapshot_id,
        };

        Ok(Snapshot {
            meta,
            snapshot: Box::new(cursor),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raft_types::ClusterOperation;
    use openraft::EntryPayload;
    use openraft::storage::RaftLogStorageExt;

    fn make_log_id(index: u64) -> LogId<u64> {
        LogId::new(CommittedLeaderId::new(1, 0), index)
    }

    #[test]
    fn redb_log_store_open_and_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raft_log.redb");
        let mut store = RedbLogStore::open(&path).unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();
        let state = rt.block_on(store.get_log_state()).unwrap();
        assert_eq!(state.last_purged_log_id, None);
        assert_eq!(state.last_log_id, None);
    }

    #[test]
    fn redb_log_store_append_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raft_log.redb");
        let mut store = RedbLogStore::open(&path).unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();

        let entries = vec![Entry {
            log_id: make_log_id(0),
            payload: EntryPayload::Normal(ClusterOperation::AddNode {
                node_id: 1,
                addr: "127.0.0.1:8001".into(),
            }),
        }];
        rt.block_on(store.blocking_append(entries)).unwrap();

        let mut reader = rt.block_on(store.get_log_reader());
        let got = rt.block_on(reader.try_get_log_entries(0..1)).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].log_id.index, 0);
    }

    #[test]
    fn redb_log_store_truncate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raft_log.redb");
        let mut store = RedbLogStore::open(&path).unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();

        let entries = vec![
            Entry {
                log_id: make_log_id(0),
                payload: EntryPayload::Blank,
            },
            Entry {
                log_id: make_log_id(1),
                payload: EntryPayload::Blank,
            },
        ];
        rt.block_on(store.blocking_append(entries)).unwrap();
        rt.block_on(store.truncate(make_log_id(1))).unwrap();

        let state = rt.block_on(store.get_log_state()).unwrap();
        assert_eq!(state.last_log_id, Some(make_log_id(0)));
    }

    #[test]
    fn redb_log_store_purge() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raft_log.redb");
        let mut store = RedbLogStore::open(&path).unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();

        let entries = vec![
            Entry {
                log_id: make_log_id(0),
                payload: EntryPayload::Blank,
            },
            Entry {
                log_id: make_log_id(1),
                payload: EntryPayload::Blank,
            },
        ];
        rt.block_on(store.blocking_append(entries)).unwrap();
        rt.block_on(store.purge(make_log_id(0))).unwrap();

        let state = rt.block_on(store.get_log_state()).unwrap();
        assert_eq!(state.last_purged_log_id, Some(make_log_id(0)));
        assert_eq!(state.last_log_id, Some(make_log_id(1)));
    }

    #[test]
    fn redb_log_store_vote_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raft_log.redb");
        let mut store = RedbLogStore::open(&path).unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();

        let vote_none = rt.block_on(store.read_vote()).unwrap();
        assert_eq!(vote_none, None);

        let vote = Vote::new(3, 42);
        rt.block_on(store.save_vote(&vote)).unwrap();
        let vote_read = rt.block_on(store.read_vote()).unwrap();
        assert_eq!(vote_read, Some(vote));
    }

    #[test]
    fn redb_log_store_persistence_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raft_log.redb");

        {
            let mut store = RedbLogStore::open(&path).unwrap();
            let rt = tokio::runtime::Runtime::new().unwrap();

            let entries = vec![
                Entry {
                    log_id: make_log_id(0),
                    payload: EntryPayload::Normal(ClusterOperation::AddNode {
                        node_id: 1,
                        addr: "127.0.0.1:8001".into(),
                    }),
                },
                Entry {
                    log_id: make_log_id(1),
                    payload: EntryPayload::Normal(ClusterOperation::AddNode {
                        node_id: 2,
                        addr: "127.0.0.1:8002".into(),
                    }),
                },
            ];
            rt.block_on(store.blocking_append(entries)).unwrap();

            let vote = Vote::new(5, 99);
            rt.block_on(store.save_vote(&vote)).unwrap();

            rt.block_on(store.purge(make_log_id(0))).unwrap();
        }

        {
            let mut store = RedbLogStore::open(&path).unwrap();
            let rt = tokio::runtime::Runtime::new().unwrap();

            let state = rt.block_on(store.get_log_state()).unwrap();
            assert_eq!(state.last_purged_log_id, Some(make_log_id(0)));
            assert_eq!(state.last_log_id, Some(make_log_id(1)));

            let vote = rt.block_on(store.read_vote()).unwrap();
            assert_eq!(vote, Some(Vote::new(5, 99)));
        }
    }
}
