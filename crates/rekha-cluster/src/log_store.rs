//! In-memory Raft log storage for openraft.

use std::fmt::Debug;
use std::ops::RangeBounds;
use std::sync::Arc;

use openraft::Entry;
use openraft::LogId;
use openraft::LogState;
use openraft::Snapshot;
use openraft::SnapshotMeta;
use openraft::StorageError;
use openraft::StoredMembership;
use openraft::Vote;
use openraft::storage::LogFlushed;
use openraft::storage::RaftLogReader;
use openraft::storage::RaftLogStorage;
use openraft::storage::RaftSnapshotBuilder;
use tokio::sync::Mutex;

use crate::raft_types::RaftTypeConfig;

pub type LogStore = Arc<Mutex<MemoryLogStoreInner>>;

#[derive(Debug)]
pub struct MemoryLogStoreInner {
    entries: Vec<Entry<RaftTypeConfig>>,
    last_purged_log_id: Option<LogId<u64>>,
    committed: Option<LogId<u64>>,
    vote: Option<Vote<u64>>,
}

impl MemoryLogStoreInner {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            last_purged_log_id: None,
            committed: None,
            vote: None,
        }
    }

    fn last_log_id(&self) -> Option<LogId<u64>> {
        self.entries.last().map(|e| e.log_id)
    }

    fn log_state(&self) -> LogState<RaftTypeConfig> {
        LogState {
            last_purged_log_id: self.last_purged_log_id,
            last_log_id: self.last_log_id().or(self.last_purged_log_id),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryLogReader {
    log: LogStore,
}

impl RaftLogReader<RaftTypeConfig> for MemoryLogReader {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug + Send>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<RaftTypeConfig>>, StorageError<u64>> {
        let store = self.log.lock().await;
        let start = match range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => u64::MAX,
        };

        let entries: Vec<_> = store
            .entries
            .iter()
            .filter(|e| e.log_id.index >= start && e.log_id.index < end)
            .cloned()
            .collect();
        Ok(entries)
    }
}

#[derive(Debug, Clone)]
pub struct MemoryLogStore {
    inner: LogStore,
}

impl MemoryLogStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MemoryLogStoreInner::new())),
        }
    }
}

impl Default for MemoryLogStore {
    fn default() -> Self {
        Self::new()
    }
}

impl RaftLogReader<RaftTypeConfig> for MemoryLogStore {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug + Send>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<RaftTypeConfig>>, StorageError<u64>> {
        let mut reader = MemoryLogReader {
            log: self.inner.clone(),
        };
        reader.try_get_log_entries(range).await
    }
}

impl RaftLogStorage<RaftTypeConfig> for MemoryLogStore {
    type LogReader = MemoryLogReader;

    async fn get_log_state(&mut self) -> Result<LogState<RaftTypeConfig>, StorageError<u64>> {
        let store = self.inner.lock().await;
        Ok(store.log_state())
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        MemoryLogReader {
            log: self.inner.clone(),
        }
    }

    async fn save_vote(&mut self, vote: &Vote<u64>) -> Result<(), StorageError<u64>> {
        let mut store = self.inner.lock().await;
        store.vote = Some(*vote);
        Ok(())
    }

    async fn read_vote(&mut self) -> Result<Option<Vote<u64>>, StorageError<u64>> {
        let store = self.inner.lock().await;
        Ok(store.vote)
    }

    async fn save_committed(
        &mut self,
        committed: Option<LogId<u64>>,
    ) -> Result<(), StorageError<u64>> {
        let mut store = self.inner.lock().await;
        store.committed = committed;
        Ok(())
    }

    async fn read_committed(&mut self) -> Result<Option<LogId<u64>>, StorageError<u64>> {
        let store = self.inner.lock().await;
        Ok(store.committed)
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
        let mut store = self.inner.lock().await;
        for entry in entries {
            store.entries.push(entry);
        }
        callback.log_io_completed(Ok(()));
        Ok(())
    }

    async fn truncate(&mut self, log_id: LogId<u64>) -> Result<(), StorageError<u64>> {
        let mut store = self.inner.lock().await;
        store.entries.retain(|e| e.log_id.index < log_id.index);
        Ok(())
    }

    async fn purge(&mut self, log_id: LogId<u64>) -> Result<(), StorageError<u64>> {
        let mut store = self.inner.lock().await;
        store.last_purged_log_id = Some(log_id);
        store.entries.retain(|e| e.log_id.index > log_id.index);
        Ok(())
    }
}

pub struct MemorySnapshotBuilder {
    log: LogStore,
}

impl RaftSnapshotBuilder<RaftTypeConfig> for MemorySnapshotBuilder {
    async fn build_snapshot(&mut self) -> Result<Snapshot<RaftTypeConfig>, StorageError<u64>> {
        let store = self.log.lock().await;
        let last_log_id = store.last_log_id().or(store.last_purged_log_id);

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
    use openraft::CommittedLeaderId;
    use openraft::EntryPayload;
    use openraft::storage::RaftLogStorageExt;

    fn make_log_id(index: u64) -> LogId<u64> {
        LogId::new(CommittedLeaderId::new(1, 0), index)
    }

    #[tokio::test]
    async fn test_new_log_store_is_empty() {
        let mut store = MemoryLogStore::new();
        let state = store.get_log_state().await.unwrap();
        assert_eq!(state.last_purged_log_id, None);
        assert_eq!(state.last_log_id, None);
    }

    #[tokio::test]
    async fn test_append_and_read() {
        let mut store = MemoryLogStore::new();

        let entries = vec![Entry {
            log_id: make_log_id(0),
            payload: EntryPayload::Normal(ClusterOperation::AddNode {
                node_id: 1,
                addr: "127.0.0.1:8001".into(),
            }),
        }];

        store.blocking_append(entries).await.unwrap();

        let mut reader = store.get_log_reader().await;
        let got = reader.try_get_log_entries(0..1).await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].log_id.index, 0);
    }

    #[tokio::test]
    async fn test_truncate() {
        let mut store = MemoryLogStore::new();

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

        store.blocking_append(entries).await.unwrap();
        store.truncate(make_log_id(1)).await.unwrap();

        let state = store.get_log_state().await.unwrap();
        assert_eq!(state.last_log_id, Some(make_log_id(0)));
    }

    #[tokio::test]
    async fn test_purge() {
        let mut store = MemoryLogStore::new();

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

        store.blocking_append(entries).await.unwrap();
        store.purge(make_log_id(0)).await.unwrap();

        let state = store.get_log_state().await.unwrap();
        assert_eq!(state.last_purged_log_id, Some(make_log_id(0)));
        assert_eq!(state.last_log_id, Some(make_log_id(1)));
    }

    #[tokio::test]
    async fn test_vote_roundtrip() {
        let mut store = MemoryLogStore::new();
        assert_eq!(store.read_vote().await.unwrap(), None);

        let vote = Vote::new(3, 42);
        store.save_vote(&vote).await.unwrap();
        assert_eq!(store.read_vote().await.unwrap(), Some(vote));
    }

    #[tokio::test]
    async fn test_committed_roundtrip() {
        let mut store = MemoryLogStore::new();
        assert_eq!(store.read_committed().await.unwrap(), None);

        store.save_committed(Some(make_log_id(5))).await.unwrap();
        assert_eq!(store.read_committed().await.unwrap(), Some(make_log_id(5)));
    }
}
