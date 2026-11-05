//! Persisted Raft snapshots.
//!
//! openraft asks the state machine for a snapshot only in one situation: a
//! follower's `next_index` has fallen below the leader's purge point, so the
//! entries it needs no longer exist and only a complete state can repair it.
//! There is no fallback — a missing snapshot stops replication for that
//! follower — so the snapshot has to outlive the call that produced it.
//!
//! Before this module, `build_snapshot` assembled an archive, returned it to
//! openraft, and the bytes were dropped when the call returned. `last_applied`
//! was advanced and the log was purged, so a node that was merely *offline* for
//! long enough could never rejoin: replication asked for a snapshot, found none,
//! and stopped. A *restarting* node was fine because its applied state lives in
//! durable storage, which is why this survived so long without being noticed.
//!
//! Layout, per group:
//!
//! ```text
//! <dir>/
//!   <snapshot_id>.snap        the archive
//!   <snapshot_id>.meta.json   its meta, written last
//! ```
//!
//! Publication is a rename, and the meta file is the commit marker: `current`
//! only considers snapshots that have both files and a matching length. A
//! process killed between the two writes leaves an orphan `.snap`, which the
//! next publish prunes. A meta file without a data file is never produced, and
//! is treated as corruption rather than as "no snapshot", because the two mean
//! very different things to an operator.

use std::io;
use std::path::{Path, PathBuf};

use openraft::SnapshotMeta;
use serde::{Deserialize, Serialize};

/// On-disk format for the meta file.
///
/// Separate from the archive's own version (`snapshot::VERSION`): this one
/// describes where a snapshot lives, that one describes what is inside it. A
/// receiver checks both.
pub const STORE_VERSION: u32 = 1;

/// How many snapshots to keep.
///
/// Two, not one: openraft may still be shipping the previous snapshot when a
/// new one is published, and deleting it mid-transfer would fail a follower
/// that was about to succeed.
pub const KEEP: usize = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MetaFile {
    store_version: u32,
    /// Which consensus group this snapshot belongs to: `metadata` or `shard:N`.
    ///
    /// A metadata snapshot and a shard snapshot are both `SnapshotPayload`s to
    /// bincode, so without this field a metadata archive posted to a shard's
    /// endpoint would deserialize and replace that shard's state with the
    /// registry. The identity has to be checked, not assumed from the address.
    group: String,
    data_len: u64,
    meta: SnapshotMeta<u64, u64>,
}

/// A snapshot on disk, with its meta already parsed.
#[derive(Debug, Clone)]
pub struct StoredSnapshot {
    pub meta: SnapshotMeta<u64, u64>,
    pub path: PathBuf,
    pub len: u64,
}

/// A snapshot's meta and its bytes, as read back for transport.
pub type CurrentSnapshot = (SnapshotMeta<u64, u64>, Vec<u8>);

#[derive(Debug, Clone)]
pub struct SnapshotStore {
    dir: PathBuf,
    group: String,
}

#[derive(Debug)]
pub enum SnapshotStoreError {
    Io(io::Error),
    /// The meta file is unreadable, is a version this build does not know, or
    /// describes a file that is not there.
    Corrupt(String),
    /// The snapshot belongs to a different group than the one it was offered to.
    WrongGroup {
        expected: String,
        found: String,
    },
}

impl std::fmt::Display for SnapshotStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Corrupt(detail) => write!(f, "{detail}"),
            Self::WrongGroup { expected, found } => write!(
                f,
                "snapshot is for group `{found}`, but this node asked for `{expected}`"
            ),
        }
    }
}

impl std::error::Error for SnapshotStoreError {}

impl From<io::Error> for SnapshotStoreError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl SnapshotStore {
    /// Open (and create) the snapshot directory for one group.
    pub fn open(dir: impl Into<PathBuf>, group: impl Into<String>) -> io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            group: group.into(),
        })
    }

    pub fn group(&self) -> &str {
        &self.group
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn snap_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.snap"))
    }

    fn meta_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.meta.json"))
    }

    /// Persist a snapshot and return where it landed.
    ///
    /// The data is written and fsynced before the meta file exists, so a reader
    /// that can see the meta can also read the whole archive. Publishing a
    /// snapshot does not delete older ones; see [`Self::prune`].
    pub fn publish(
        &self,
        meta: &SnapshotMeta<u64, u64>,
        data: &[u8],
    ) -> Result<StoredSnapshot, SnapshotStoreError> {
        let id = &meta.snapshot_id;
        if id.is_empty() || id.contains('/') || id.contains('\\') || id.contains("..") {
            return Err(SnapshotStoreError::Corrupt(format!(
                "snapshot id `{id}` is not usable as a file name"
            )));
        }

        let data_tmp = self.dir.join(format!(".{id}.snap.tmp"));
        let meta_tmp = self.dir.join(format!(".{id}.meta.tmp"));
        let snap = self.snap_path(id);
        let meta_path = self.meta_path(id);

        write_atomically(&data_tmp, &snap, data)?;

        let record = MetaFile {
            store_version: STORE_VERSION,
            group: self.group.clone(),
            data_len: data.len() as u64,
            meta: meta.clone(),
        };
        let bytes = serde_json::to_vec(&record).map_err(|e| {
            SnapshotStoreError::Corrupt(format!("snapshot meta could not be encoded: {e}"))
        })?;
        // The meta file is the commit marker, so it is published last.
        write_atomically(&meta_tmp, &meta_path, &bytes)?;
        sync_dir(&self.dir)?;

        Ok(StoredSnapshot {
            meta: meta.clone(),
            path: snap,
            len: data.len() as u64,
        })
    }

    /// The newest snapshot for this group, if any.
    ///
    /// Newest is by `(index, term)` from the snapshot meta, not by file mtime:
    /// a restored data directory can carry mtimes that describe the copy rather
    /// than the snapshot, and serving an older snapshot would roll a follower
    /// backwards — a successful install of stale data, which is worse than no
    /// snapshot at all.
    pub fn current(&self) -> Result<Option<StoredSnapshot>, SnapshotStoreError> {
        let mut best: Option<StoredSnapshot> = None;

        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(id) = name.strip_suffix(".meta.json") else {
                continue;
            };
            // Our own temporary files are skipped rather than parsed.
            if id.starts_with('.') {
                continue;
            }

            let record = self.read_meta(&path)?;
            if record.group != self.group {
                return Err(SnapshotStoreError::WrongGroup {
                    expected: self.group.clone(),
                    found: record.group,
                });
            }

            let snap = self.snap_path(id);
            let on_disk = std::fs::metadata(&snap).map(|m| m.len()).map_err(|e| {
                SnapshotStoreError::Corrupt(format!(
                    "snapshot `{id}` has a meta file but no data file ({}): {e}",
                    snap.display()
                ))
            })?;
            if on_disk != record.data_len {
                return Err(SnapshotStoreError::Corrupt(format!(
                    "snapshot `{id}` records {} bytes but its data file holds {on_disk}",
                    record.data_len
                )));
            }

            let candidate = StoredSnapshot {
                meta: record.meta,
                path: snap,
                len: on_disk,
            };
            if best
                .as_ref()
                .is_none_or(|b| newer(&candidate.meta, &b.meta))
            {
                best = Some(candidate);
            }
        }

        Ok(best)
    }

    /// Read a snapshot's meta, refusing a format this build does not implement.
    fn read_meta(&self, path: &Path) -> Result<MetaFile, SnapshotStoreError> {
        let bytes = std::fs::read(path)?;
        let record: MetaFile = serde_json::from_slice(&bytes).map_err(|e| {
            SnapshotStoreError::Corrupt(format!("{} is unreadable: {e}", path.display()))
        })?;
        if record.store_version != STORE_VERSION {
            return Err(SnapshotStoreError::Corrupt(format!(
                "{} is store version {}, but this build implements {STORE_VERSION}",
                path.display(),
                record.store_version
            )));
        }
        Ok(record)
    }

    /// Read the newest snapshot's bytes and meta.
    pub fn read_current(&self) -> Result<Option<CurrentSnapshot>, SnapshotStoreError> {
        let Some(stored) = self.current()? else {
            return Ok(None);
        };
        let bytes = std::fs::read(&stored.path)?;
        if bytes.len() as u64 != stored.len {
            return Err(SnapshotStoreError::Corrupt(format!(
                "snapshot {} changed size while being read: {} bytes, expected {}",
                stored.path.display(),
                bytes.len(),
                stored.len
            )));
        }
        Ok(Some((stored.meta, bytes)))
    }

    /// Keep only the newest `keep` snapshots.
    ///
    /// Best-effort by design: failing to delete an old snapshot must not fail
    /// the publish that just succeeded, because the new snapshot is what the
    /// cluster needs. The returned count is what was removed, for the caller to
    /// log rather than to act on.
    pub fn prune(&self, keep: usize) -> Result<usize, SnapshotStoreError> {
        let mut found: Vec<StoredSnapshot> = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(id) = name.strip_suffix(".meta.json") else {
                // Orphan data files are left for the next publish to clean.
                continue;
            };
            if id.starts_with('.') {
                continue;
            }
            let Ok(record) = self.read_meta(&path) else {
                continue;
            };
            let snap = self.snap_path(id);
            if !snap.is_file() {
                continue;
            }
            found.push(StoredSnapshot {
                meta: record.meta,
                path: snap,
                len: record.data_len,
            });
        }

        // Newest first, so everything past `keep` is removable. Sorting on a key
        // rather than a `newer` predicate keeps the comparator total: equal
        // elements must compare Equal or the sort is free to reorder anything.
        found.sort_by(|a, b| {
            (b.meta.last_log_id, b.meta.snapshot_id.as_str())
                .cmp(&(a.meta.last_log_id, a.meta.snapshot_id.as_str()))
        });

        let mut removed = 0;
        for stale in found.into_iter().skip(keep) {
            let Some(id) = stale
                .path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".snap"))
            else {
                continue;
            };
            let _ = std::fs::remove_file(self.meta_path(id));
            let _ = std::fs::remove_file(&stale.path);
            removed += 1;
        }
        Ok(removed)
    }
}

/// Is `a` a newer snapshot than `b`?
fn newer(a: &SnapshotMeta<u64, u64>, b: &SnapshotMeta<u64, u64>) -> bool {
    // Ties fall back to the id so the answer is stable rather than dependent on
    // directory order. `None` sorts below any log id, which is correct: a
    // snapshot of nothing is the oldest thing there is.
    (a.last_log_id, a.snapshot_id.as_str()) > (b.last_log_id, b.snapshot_id.as_str())
}

fn write_atomically(tmp: &Path, dest: &Path, bytes: &[u8]) -> io::Result<()> {
    use std::io::Write;

    {
        let mut file = std::fs::File::create(tmp)?;
        file.write_all(bytes)?;
        // Durable before it is visible: a rename over a power loss otherwise
        // publishes a file whose contents are not on disk.
        file.sync_all()?;
    }
    std::fs::rename(tmp, dest)?;
    sync_dir(
        dest.parent()
            .ok_or_else(|| io::Error::other("snapshot path has no parent"))?,
    )
}

/// fsync a directory so a rename survives a crash.
///
/// `InvalidInput` is ignored because some platforms (macOS among them) refuse to
/// fsync a directory; propagating that would turn a working publish into a
/// failure on a platform that cannot express the guarantee at all.
fn sync_dir(dir: &Path) -> io::Result<()> {
    match std::fs::File::open(dir).and_then(|f| f.sync_all()) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::InvalidInput => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openraft::{CommittedLeaderId, LogId, StoredMembership};

    fn meta(id: &str, index: u64) -> SnapshotMeta<u64, u64> {
        SnapshotMeta {
            last_log_id: Some(LogId::new(CommittedLeaderId::new(1, 1), index)),
            last_membership: StoredMembership::default(),
            snapshot_id: id.to_string(),
        }
    }

    #[test]
    fn a_published_snapshot_is_what_current_returns() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SnapshotStore::open(tmp.path(), "metadata").unwrap();

        assert!(store.current().unwrap().is_none());
        store.publish(&meta("s1", 7), b"archive-one").unwrap();

        let (found, bytes) = store.read_current().unwrap().expect("a snapshot");
        assert_eq!(found.snapshot_id, "s1");
        assert_eq!(bytes, b"archive-one");
    }

    #[test]
    fn current_prefers_the_higher_index_regardless_of_directory_order() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SnapshotStore::open(tmp.path(), "metadata").unwrap();

        // Newest published first, so a naive "last file written" answer fails.
        store.publish(&meta("new", 9), b"nine").unwrap();
        store.publish(&meta("old", 3), b"three").unwrap();

        let (found, bytes) = store.read_current().unwrap().expect("a snapshot");
        assert_eq!(found.snapshot_id, "new");
        assert_eq!(bytes, b"nine");
    }

    #[test]
    fn a_meta_file_without_data_is_corruption_not_absence() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SnapshotStore::open(tmp.path(), "metadata").unwrap();
        store.publish(&meta("s1", 1), b"bytes").unwrap();

        std::fs::remove_file(tmp.path().join("s1.snap")).unwrap();
        let err = store.current().unwrap_err().to_string();
        assert!(err.contains("no data file"), "{err}");
    }

    #[test]
    fn a_short_data_file_is_refused_rather_than_served() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SnapshotStore::open(tmp.path(), "metadata").unwrap();
        store.publish(&meta("s1", 1), b"0123456789").unwrap();

        std::fs::write(tmp.path().join("s1.snap"), b"012").unwrap();
        let err = store.current().unwrap_err().to_string();
        assert!(err.contains("records 10 bytes"), "{err}");
    }

    #[test]
    fn a_snapshot_for_another_group_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = SnapshotStore::open(tmp.path(), "shard:2").unwrap();
        writer.publish(&meta("s1", 1), b"bytes").unwrap();

        let reader = SnapshotStore::open(tmp.path(), "metadata").unwrap();
        let err = reader.current().unwrap_err().to_string();
        assert!(err.contains("group `shard:2`"), "{err}");
    }

    #[test]
    fn prune_keeps_the_newest_and_deletes_the_rest() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SnapshotStore::open(tmp.path(), "metadata").unwrap();
        for index in 1..=4 {
            store
                .publish(&meta(&format!("s{index}"), index), b"bytes")
                .unwrap();
        }

        assert_eq!(store.prune(2).unwrap(), 2);
        let remaining: Vec<String> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().to_str().map(str::to_owned))
            .filter(|n| n.ends_with(".meta.json"))
            .collect();
        assert_eq!(remaining.len(), 2, "{remaining:?}");
        assert!(remaining.iter().any(|n| n.starts_with("s4")));
        assert!(remaining.iter().any(|n| n.starts_with("s3")));
        // And the survivor still reads.
        assert_eq!(store.read_current().unwrap().unwrap().0.snapshot_id, "s4");
    }

    #[test]
    fn a_snapshot_id_that_escapes_the_directory_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SnapshotStore::open(tmp.path(), "metadata").unwrap();
        let err = store.publish(&meta("../escape", 1), b"x").unwrap_err();
        assert!(
            err.to_string().contains("not usable as a file name"),
            "{err}"
        );
    }
}
