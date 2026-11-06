//! Disk-backed Raft log storage using redb.

#![allow(clippy::result_large_err)]

use std::fmt::Debug;
use std::ops::RangeBounds;
use std::path::Path;
use std::sync::Arc;

use openraft::storage::{LogFlushed, RaftLogReader, RaftLogStorage};
use openraft::{CommittedLeaderId, Entry, LogId, LogState, StorageError, Vote};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::raft_types::RaftTypeConfig;

const LOG_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("raft_log");
const VOTE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("raft_vote");
const STATE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("raft_state");

/// Current persisted-state key. The legacy `"state"` key is still read so a
/// data directory written by an earlier build keeps its purge watermark;
/// bincode is not self-describing, so a field cannot be added in place.
const STATE_KEY: &str = "state_v2";
const LEGACY_STATE_KEY: &str = "state";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializedState {
    last_purged_index: Option<u64>,
    last_purged_term: Option<u64>,
    /// The leader that purged. Reconstructing this as `node 0` made two log
    /// ids with the same term and index compare unequal after a restart, which
    /// trips openraft's ordering invariants.
    #[serde(default)]
    last_purged_node: Option<u64>,
    /// Last log id known to be committed.
    ///
    /// openraft restores this on restart and asserts `last_purged <= committed`.
    /// Leaving it unset while the log had been purged made every restart panic
    /// inside `RaftState::new`.
    committed_index: Option<u64>,
    committed_term: Option<u64>,
    #[serde(default)]
    committed_node: Option<u64>,
}

/// The previous state blob: no committed log id, no purging node.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacySerializedState {
    last_purged_index: Option<u64>,
    last_purged_term: Option<u64>,
}

fn decode_log_id(index: Option<u64>, term: Option<u64>, node: Option<u64>) -> Option<LogId<u64>> {
    match (index, term) {
        (Some(idx), Some(term)) => Some(LogId::new(
            CommittedLeaderId::new(term, node.unwrap_or(0)),
            idx,
        )),
        _ => None,
    }
}

impl SerializedState {
    fn last_purged_log_id(&self) -> Option<LogId<u64>> {
        decode_log_id(
            self.last_purged_index,
            self.last_purged_term,
            self.last_purged_node,
        )
    }

    fn committed_log_id(&self) -> Option<LogId<u64>> {
        decode_log_id(
            self.committed_index,
            self.committed_term,
            self.committed_node,
        )
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
    committed: Option<LogId<u64>>,
}

#[derive(Clone)]
pub struct RedbLogReader {
    db: Arc<Database>,
    /// Indexes at or below this were purged and are legitimately absent.
    last_purged: Option<u64>,
}

/// Read log entries, refusing to paper over a hole.
///
/// The original skipped any key that was missing or failed to deserialize and
/// returned whatever else it found, so a damaged middle of the log was
/// indistinguishable from a purged prefix and callers proceeded on a shortened
/// history without knowing.
///
/// Two absences are legitimate: an index at or below the purge watermark, and
/// an index past the last written entry (openraft asks for ranges that extend
/// beyond the log, e.g. when replaying `[last_applied, committed]`). The latter
/// is decided lazily — only a missing key costs a scan — so the hot path stays
/// proportional to the range.
fn read_log_entries(
    db: &Database,
    last_purged: Option<u64>,
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
        match table.get(idx) {
            Ok(Some(val)) => {
                let entry =
                    bincode::deserialize::<Entry<RaftTypeConfig>>(val.value()).map_err(|e| {
                        io_error(format!(
                            "raft log entry {idx} is undecodable ({e}); the log is corrupt"
                        ))
                    })?;
                entries.push(entry);
            }
            Ok(None) if last_purged.is_some_and(|purged| idx <= purged) => {
                // Purged by a snapshot: absent by design.
            }
            Ok(None) => {
                // Missing above the purge watermark. Either this index is past
                // the end of the log (legitimate) or the log has a hole.
                let last = find_last_log_id(db)?;
                if last.is_some_and(|l| idx > l.index) {
                    break;
                }
                return Err(io_error(format!(
                    "raft log entry {idx} is missing: the log has a hole between \
                     last_purged={} and last_log_id={}. Refusing to serve a shortened history.",
                    last_purged
                        .map(|p| p.to_string())
                        .unwrap_or_else(|| "none".into()),
                    last.map(|l| l.to_string()).unwrap_or_else(|| "none".into()),
                )));
            }
            Err(e) => {
                return Err(io_error(format!("raft log entry {idx} is unreadable: {e}")));
            }
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

    if let Some(entry) = table.iter().map_err(io_error)?.next_back() {
        let (_k, v) = entry.map_err(io_error)?;
        let entry = bincode::deserialize::<Entry<RaftTypeConfig>>(v.value())
            .map_err(|e| io_error(format!("failed to deserialize last raft entry: {e}")))?;
        return Ok(Some(entry.log_id));
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

        // Read persisted state. A `state_v2` blob wins; a directory written by
        // an earlier build has only `state`, which carries the purge watermark
        // but no committed log id.
        let (last_purged, committed) = {
            let read_txn = db.begin_read().map_err(io_error)?;
            let mut last_purged = None;
            let mut committed = None;
            if let Ok(table) = read_txn.open_table(STATE_TABLE) {
                if let Ok(Some(val)) = table.get(STATE_KEY) {
                    if let Ok(state) = bincode::deserialize::<SerializedState>(val.value()) {
                        last_purged = state.last_purged_log_id();
                        committed = state.committed_log_id();
                    }
                } else if let Ok(Some(val)) = table.get(LEGACY_STATE_KEY)
                    && let Ok(state) = bincode::deserialize::<LegacySerializedState>(val.value())
                {
                    // The old format recorded no node id; 0 is what it
                    // reconstructed one as.
                    last_purged =
                        decode_log_id(state.last_purged_index, state.last_purged_term, None);
                }
            }
            (last_purged, committed)
        };

        Ok(Self {
            db: Arc::new(db),
            inner: Arc::new(Mutex::new(RedbLogStoreInner {
                last_purged,
                committed,
            })),
        })
    }

    /// Serialize the mutable part of the store's state.
    fn state_bytes(inner: &RedbLogStoreInner) -> Result<Vec<u8>, StorageError<u64>> {
        let state = SerializedState {
            last_purged_index: inner.last_purged.map(|l| l.index),
            last_purged_term: inner.last_purged.map(|l| l.leader_id.term),
            last_purged_node: inner.last_purged.map(|l| l.leader_id.node_id),
            committed_index: inner.committed.map(|l| l.index),
            committed_term: inner.committed.map(|l| l.leader_id.term),
            committed_node: inner.committed.map(|l| l.leader_id.node_id),
        };
        bincode::serialize(&state).map_err(io_error)
    }

    fn save_state(&self, inner: &RedbLogStoreInner) -> Result<(), StorageError<u64>> {
        let bytes = Self::state_bytes(inner)?;
        let write_txn = self.db.begin_write().map_err(io_error)?;
        {
            let mut table = write_txn.open_table(STATE_TABLE).map_err(io_error)?;
            table
                .insert(STATE_KEY, bytes.as_slice())
                .map_err(io_error)?;
        }
        write_txn.commit().map_err(io_error)?;
        Ok(())
    }

    /// The persisted purge and commit watermarks, for diagnostics and metrics.
    ///
    /// `last_purged > committed` is the state that made startup panic, so being
    /// able to observe both without parsing the log is worth the accessor.
    pub async fn watermarks(&mut self) -> (Option<u64>, Option<u64>) {
        let inner = self.inner.lock().await;
        (
            inner.last_purged.map(|l| l.index),
            inner.committed.map(|l| l.index),
        )
    }

    /// The log id the log was purged up to, if any.
    ///
    /// A purged prefix is by construction covered by a snapshot, so this is the
    /// boundary a restarted state machine must resume from. Without it the
    /// state machine reports "applied nothing" and openraft tries to replay
    /// entries that no longer exist.
    pub async fn purge_watermark(&mut self) -> Option<LogId<u64>> {
        self.inner.lock().await.last_purged
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
            std::ops::Bound::Included(&n) => n.saturating_add(1),
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => find_last_log_id(&self.db)?
                .map(|l| l.index.saturating_add(1))
                .unwrap_or(start),
        };
        if end <= start {
            return Ok(Vec::new());
        }
        let last_purged = self.inner.lock().await.last_purged.map(|l| l.index);
        read_log_entries(&self.db, last_purged, start, end)
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
            std::ops::Bound::Included(&n) => n.saturating_add(1),
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => find_last_log_id(&self.db)?
                .map(|l| l.index.saturating_add(1))
                .unwrap_or(start),
        };
        if end <= start {
            return Ok(Vec::new());
        }
        read_log_entries(&self.db, self.last_purged, start, end)
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
        let last_purged = self.inner.lock().await.last_purged.map(|l| l.index);
        RedbLogReader {
            db: self.db.clone(),
            last_purged,
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
        let write_txn = self.db.begin_write().map_err(io_error)?;
        {
            let mut table = write_txn.open_table(LOG_TABLE).map_err(io_error)?;
            for entry in entries {
                let key = entry.log_id.index;
                let bytes = bincode::serialize(&entry).map_err(io_error)?;
                table.insert(key, bytes.as_slice()).map_err(io_error)?;
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
            let keys_to_remove: Vec<u64> = table
                .range(log_id.index..)
                .map_err(io_error)?
                .filter_map(|e| e.ok().map(|(k, _)| k.value()))
                .collect();
            for key in keys_to_remove {
                let _ = table.remove(key);
            }
        }
        write_txn.commit().map_err(io_error)?;
        Ok(())
    }

    async fn purge(&mut self, log_id: LogId<u64>) -> Result<(), StorageError<u64>> {
        let mut inner = self.inner.lock().await;
        inner.last_purged = Some(log_id);
        // A purge never moves the commit point backwards, and openraft's
        // startup invariant requires `last_purged <= committed`.
        if inner
            .committed
            .is_none_or(|committed| committed.index < log_id.index)
        {
            inner.committed = Some(log_id);
        }
        let state = Self::state_bytes(&inner)?;

        // Entries, the purge watermark, and the commit point are written in
        // **one** transaction. Previously the entries committed first and the
        // watermark second, so a crash between them left a log whose prefix was
        // gone while its state still claimed it was there.
        let write_txn = self.db.begin_write().map_err(io_error)?;
        {
            let mut table = write_txn.open_table(LOG_TABLE).map_err(io_error)?;
            let keys_to_remove: Vec<u64> = table
                .range(..=log_id.index)
                .map_err(io_error)?
                .filter_map(|e| e.ok().map(|(k, _)| k.value()))
                .collect();
            for key in keys_to_remove {
                let _ = table.remove(key);
            }
        }
        {
            let mut table = write_txn.open_table(STATE_TABLE).map_err(io_error)?;
            table
                .insert(STATE_KEY, state.as_slice())
                .map_err(io_error)?;
        }
        write_txn.commit().map_err(io_error)?;
        Ok(())
    }

    /// Persist the committed log id.
    ///
    /// openraft validates `last_purged <= committed` when it starts. Without a
    /// persisted commit point, a node that had purged its log came back with
    /// `committed = None` and panicked inside `RaftState::new`.
    async fn save_committed(
        &mut self,
        committed: Option<LogId<u64>>,
    ) -> Result<(), StorageError<u64>> {
        let mut inner = self.inner.lock().await;
        inner.committed = committed;
        self.save_state(&inner)
    }

    async fn read_committed(&mut self) -> Result<Option<LogId<u64>>, StorageError<u64>> {
        let inner = self.inner.lock().await;
        Ok(inner.committed)
    }
}

// A snapshot builder belonged here, and it was removed rather than fixed.
//
// It returned `vec![]` as the snapshot body with `StoredMembership::default()`,
// which is a *valid* snapshot as far as openraft is concerned: a follower would
// accept it, install an empty state machine, and report success. The log store
// has no access to the engine, so no implementation of this trait here could
// ever produce a real one. The state machine's `ClusterSnapshotBuilder` is the
// only correct producer, and `RaftStateMachine::SnapshotBuilder` is the only
// hook openraft uses. Leaving a second, wrong builder in the public API was one
// wiring mistake away from silent total data loss on a replica.

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

    /// openraft restores `committed` on startup and asserts
    /// `last_purged <= committed`. A store that returned `None` here panicked
    /// on every restart once its log had been purged.
    #[test]
    fn committed_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raft_log.redb");

        {
            let mut store = RedbLogStore::open(&path).unwrap();
            let rt = tokio::runtime::Runtime::new().unwrap();
            assert_eq!(rt.block_on(store.read_committed()).unwrap(), None);
            rt.block_on(store.save_committed(Some(make_log_id(7))))
                .unwrap();
        }

        let mut store = RedbLogStore::open(&path).unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert_eq!(
            rt.block_on(store.read_committed()).unwrap(),
            Some(make_log_id(7))
        );
    }

    /// Purging must leave a commit point at or past the purge watermark, or the
    /// next startup trips openraft's invariant.
    #[test]
    fn purge_persists_a_commit_point_at_or_past_the_purge_point() {
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
        rt.block_on(store.purge(make_log_id(1))).unwrap();
        drop(store);

        // The invariant openraft checks on startup, asserted directly.
        let mut reopened = RedbLogStore::open(&path).unwrap();
        let state = rt.block_on(reopened.get_log_state()).unwrap();
        let committed = rt.block_on(reopened.read_committed()).unwrap();
        assert!(
            state.last_purged_log_id <= committed,
            "purge watermark {:?} exceeds committed {committed:?}",
            state.last_purged_log_id
        );
    }

    /// A hole between the purge watermark and the head is corruption, not a
    /// shortened log to be served silently.
    #[test]
    fn a_missing_entry_in_the_live_range_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raft_log.redb");
        let mut store = RedbLogStore::open(&path).unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let entries = (0..5)
            .map(|i| Entry {
                log_id: make_log_id(i),
                payload: EntryPayload::Blank,
            })
            .collect::<Vec<_>>();
        rt.block_on(store.blocking_append(entries)).unwrap();

        // Delete the middle entry directly: the kind of damage a partial write
        // or a bad sector leaves behind.
        {
            let write_txn = store.db.begin_write().unwrap();
            {
                let mut table = write_txn.open_table(LOG_TABLE).unwrap();
                table.remove(2u64).unwrap();
            }
            write_txn.commit().unwrap();
        }

        let mut reader = rt.block_on(store.get_log_reader());
        let err = rt
            .block_on(reader.try_get_log_entries(0..5))
            .expect_err("a hole in the live range must be reported, not skipped");
        let text = err.to_string();
        assert!(
            text.contains("missing") && text.contains("hole"),
            "unexpected error: {text}"
        );
    }

    /// An index below the purge watermark is absent by design, and a range that
    /// extends past the head is a request for entries that were never written.
    /// Neither is a hole.
    #[test]
    fn a_purged_prefix_and_an_unwritten_tail_are_not_holes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raft_log.redb");
        let mut store = RedbLogStore::open(&path).unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let entries = (0..4)
            .map(|i| Entry {
                log_id: make_log_id(i),
                payload: EntryPayload::Blank,
            })
            .collect::<Vec<_>>();
        rt.block_on(store.blocking_append(entries)).unwrap();
        rt.block_on(store.purge(make_log_id(2))).unwrap();

        let mut reader = rt.block_on(store.get_log_reader());
        // 0..2 are purged, 3 survives, 10..12 were never written.
        let got = rt.block_on(reader.try_get_log_entries(0..12)).unwrap();
        assert_eq!(got.len(), 1, "only the un-purged tail is returned");
        assert_eq!(got[0].log_id.index, 3);
    }

    /// A data directory written by the previous build stored its watermark
    /// under `state` with no committed log id. Opening it must keep the
    /// watermark rather than treating the log as unpurged.
    #[test]
    fn legacy_state_blob_is_still_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raft_log.redb");

        {
            let store = RedbLogStore::open(&path).unwrap();
            let legacy = LegacySerializedState {
                last_purged_index: Some(9),
                last_purged_term: Some(4),
            };
            let bytes = bincode::serialize(&legacy).unwrap();
            let write_txn = store.db.begin_write().unwrap();
            {
                let mut table = write_txn.open_table(STATE_TABLE).unwrap();
                table.insert(LEGACY_STATE_KEY, bytes.as_slice()).unwrap();
            }
            write_txn.commit().unwrap();
        }

        let mut store = RedbLogStore::open(&path).unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let state = rt.block_on(store.get_log_state()).unwrap();
        assert_eq!(state.last_purged_log_id.map(|l| l.index), Some(9));
    }
}
