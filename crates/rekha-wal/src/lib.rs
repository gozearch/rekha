//! RekhaDB `rekha-wal` crate — the durable per-collection write-ahead log.
//!
//! # Design contract
//!
//! **The WAL is the source of truth; indexes are derived.** Every write
//! (`add` / `update` / `upsert` / `delete`, see [`rekha_core::op::Operation`])
//! is appended to the log and acknowledged before any index is touched. A
//! background compaction/indexing service consumes the log tail to build
//! derived indexes, and replicas rebuild their state by replaying the log.
//!
//! This shapes the on-disk format from day one:
//!
//! - **Self-contained, shippable records.** Each record carries its full
//!   payload plus a `seq` and `epoch`, so any fragment of the log can be
//!   transferred to a peer and replayed there. [`Wal::read_range`] is the
//!   delta-transfer primitive replicas use to pull `[from, to]` from the
//!   leader.
//! - **Monotonic seq ids.** `seq` strictly increases per collection file, so
//!   a replica knows exactly where it is in the log and what it is missing.
//! - **Record-level CRC32.** Every record carries a CRC32 over its payload;
//!   torn tails (a crash mid-append) and bit rot are detected on open and the
//!   file is truncated back to the last valid byte.
//! - **Epochs are fencing tokens.** Every record is stamped with the file's
//!   creation epoch. A record with `epoch < file_epoch` is the trace of a
//!   stale (fenced) writer and is rejected — at append time by the API's
//!   invariant, and on replay by treating it as a corrupt tail.
//! - **Pruning is compaction's primitive.** [`Wal::prune`] rewrites the file
//!   atomically (temp file + fsync + rename), keeping only records above a
//!   watermark — the primitive a log compactor needs to bound WAL growth.
//!
//! # Concurrency
//!
//! [`Wal::append`] and [`Wal::prune`] take `&mut self`: the caller (the
//! engine) must hold a per-collection lock, giving exactly one writer per
//! collection file. Read operations (`read_range`, `last_seq`, `len`,
//! `epoch`, `size_bytes`) take `&self`. `Wal` is `Send` (its `File` is
//! `Send`).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use rekha_core::cluster::Epoch;
use rekha_core::op::Operation;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[inline]
fn le_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
#[inline]
fn le_u64(b: &[u8]) -> u64 {
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}
#[inline]
fn le_u16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}


/// Magic bytes identifying a RekhaDB WAL file: `"RKW1"`.
const MAGIC: [u8; 4] = *b"RKW1";
/// On-disk format version.
const VERSION: u16 = 1;
/// Fixed size of the file header: magic(4) + version(2) + epoch(8) + crc(4).
const HEADER_LEN: usize = 18;
/// Fixed size of a record prefix: length(4) + crc(4) + seq(8) + epoch(8).
const RECORD_HEADER_LEN: usize = 24;

/// Open options for a [`Wal`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalOptions {
    /// `true` = fsync on every append (durable); `false` = rely on the OS to
    /// flush (faster, weaker crash guarantees).
    pub fsync: bool,
    /// Seq id to resume appending at **when the file has no records on open**
    /// (either brand-new or header-only after a prune).
    ///
    /// A checkpointed collection prunes its WAL down to the un-checkpointed
    /// tail (possibly empty); the checkpoint carries the next seq it needs, so
    /// the engine passes `start_seq = Some(next_seq)` on reopen. When the file
    /// already contains records the file's own `last_seq + 1` wins and this
    /// value is ignored.
    pub start_seq: Option<u64>,
}

impl WalOptions {
    /// Options with the given `fsync` behavior and no `start_seq` override
    /// (fresh files resume at seq 1).
    pub fn new(fsync: bool) -> Self {
        Self {
            fsync,
            start_seq: None,
        }
    }
}

/// A single log record: a fully self-contained, replayable operation.
#[derive(Debug, Clone)]
pub struct WalRecord {
    /// Monotonic per-collection sequence id.
    pub seq: u64,
    /// Epoch this record was stamped with (the file's creation epoch).
    pub epoch: u64,
    /// The operation payload.
    pub op: Operation,
}

/// Errors produced by [`Wal`].
#[derive(Debug, thiserror::Error)]
pub enum WalError {
    /// Underlying I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The WAL file is malformed: bad magic/version/header CRC, truncated
    /// header, torn tail, out-of-order seq, or a fenced record.
    #[error("invalid WAL file: {0}")]
    Corrupt(String),
    /// Payload serialization / deserialization failure.
    #[error("bincode: {0}")]
    Encode(#[from] bincode::Error),
    /// A record stamped with an older epoch was rejected (fencing).
    #[error("fencing: record epoch {record} < file epoch {file}")]
    Fenced { record: u64, file: u64 },
}

/// Convenience alias.
pub type WalResult<T> = Result<T, WalError>;

/// An append-only, per-collection write-ahead log.
///
/// On disk: one file per collection at `{prefix}/{collection_id}.wal`
///
/// ```text
/// [FileHeader] { magic "RKW1" | version u16 | epoch u64 | header_crc u32 }
/// [Record]×N   { length u32 | crc u32 | seq u64 | epoch u64 | payload }
/// ```
///
/// `length` spans the whole record (`24 + len(payload)`); the record CRC
/// covers the payload only. `seq` is strictly increasing per file (1, 2, 3, …;
/// a resumed file continues at `last_seq + 1`).
///
/// On open the header is verified and any torn tail (partial final record,
/// bad CRC, out-of-order seq, or fenced record) is truncated away.
///
/// Single-writer: [`append`](Wal::append) and [`prune`](Wal::prune) take
/// `&mut self`; the engine must serialize writers per collection.
pub struct Wal {
    file: File,
    path: PathBuf,
    header_epoch: u64,
    last_seq: u64,
    len: u64,
    file_len: u64,
    options: WalOptions,
}

impl Wal {
    /// Open (or create) the WAL for `collection_id` at `path` under `epoch`.
    ///
    /// `path` is the directory prefix; the file lives at
    /// `path/{collection_id}.wal`. A brand-new file is stamped with `epoch`;
    /// an existing file keeps its header epoch (which is also what
    /// [`Wal::epoch`] reports).
    ///
    /// On open: verify the header, scan all records and recover any torn tail
    /// (truncate back to the last valid byte), then position this handle at
    /// the recovered `last_seq`. A bad header (magic/version/CRC) is a
    /// [`WalError::Corrupt`]; the header is never truncated away.
    ///
    /// When the file is brand-new and [`WalOptions::start_seq`] is `Some(n)`,
    /// the next append resumes at `n` instead of `1` (a checkpointed collection
    /// reopening an empty pruned log). An existing file always resumes at its
    /// own `last_seq + 1`; `start_seq` is ignored.
    pub fn open(
        path: impl AsRef<Path>,
        collection_id: &Uuid,
        epoch: Epoch,
        options: WalOptions,
    ) -> WalResult<Self> {
        let dir = path.as_ref();
        std::fs::create_dir_all(dir)?;
        let file_path = dir.join(format!("{collection_id}.wal"));

        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&file_path)?;

        let file_len = file.metadata()?.len();

        let (header_epoch, valid_len, len, last_seq) = if file_len == 0 {
            // Brand-new file: write the header.
            let header = build_header(epoch.0);
            file.write_all(&header)?;
            if options.fsync {
                file.sync_all()?;
            }
            // A `start_seq` override moves the next append up to that seq, so
            // a checkpointed-and-fully-pruned collection resumes where its
            // checkpoint said it would.
            let last_seq = options.start_seq.map(|n| n.saturating_sub(1)).unwrap_or(0);
            (epoch.0, HEADER_LEN as u64, 0, last_seq)
        } else {
            // File position wins: `start_seq` is ignored when records exist.
            let header_epoch = read_and_verify_header(&mut file)?;
            let (valid_len, len, last_seq) =
                recover(&mut file, file_len, header_epoch, options.fsync)?;
            // A header-only file (0 records) after prune: honor start_seq so
            // a checkpointed collection resumes appending at the correct seq.
            let last_seq = if len == 0 {
                options
                    .start_seq
                    .map(|n| n.saturating_sub(1))
                    .unwrap_or(last_seq)
            } else {
                last_seq
            };
            (header_epoch, valid_len, len, last_seq)
        };

        Ok(Wal {
            file,
            path: file_path,
            header_epoch,
            last_seq,
            len,
            file_len: valid_len,
            options,
        })
    }

    /// Append `op`, assigning it the next `seq`. Fsyncs if
    /// [`WalOptions::fsync`]. Returns the record as written.
    pub fn append(&mut self, op: Operation) -> WalResult<WalRecord> {
        let seq = self.last_seq + 1;
        let epoch = self.header_epoch;

        // Fencing invariant: records are always stamped with the file's epoch,
        // so a record below the file epoch is impossible through this API. The
        // check is kept as a belt-and-suspenders guard; the same rule is what
        // makes a fenced record replayable-safe (it is treated as a corrupt
        // tail on open).
        if epoch < self.header_epoch {
            return Err(WalError::Fenced {
                record: epoch,
                file: self.header_epoch,
            });
        }

        let payload = bincode::serialize(&op)?;
        let total_len = RECORD_HEADER_LEN as u64 + payload.len() as u64;
        if total_len > u32::MAX as u64 {
            return Err(WalError::Corrupt(format!(
                "record too large for WAL framing: {total_len} bytes"
            )));
        }
        let record_len = total_len as u32;

        let mut buf = Vec::with_capacity(total_len as usize);
        buf.extend_from_slice(&record_len.to_le_bytes());
        buf.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
        buf.extend_from_slice(&seq.to_le_bytes());
        buf.extend_from_slice(&epoch.to_le_bytes());
        buf.extend_from_slice(&payload);

        self.file.seek(SeekFrom::Start(self.file_len))?;
        self.file.write_all(&buf)?;
        if self.options.fsync {
            self.file.sync_all()?;
        }

        self.file_len += buf.len() as u64;
        self.last_seq = seq;
        self.len += 1;

        Ok(WalRecord { seq, epoch, op })
    }

    /// Highest seq present in this file (0 if empty and no
    /// [`WalOptions::start_seq`] override was given; otherwise the resume
    /// position `start_seq - 1`, so an empty file opened with a start_seq
    /// reports where the next append will land).
    pub fn last_seq(&self) -> u64 {
        self.last_seq
    }

    /// Number of records (after torn-tail truncation).
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Returns `true` if the WAL contains no records.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read all records with `seq` in `[from, to]` inclusive, in seq order —
    /// the delta-transfer primitive for replication. Empty if `from > to`.
    pub fn read_range(&self, from: u64, to: u64) -> WalResult<Vec<WalRecord>> {
        if from > to {
            return Ok(Vec::new());
        }
        let mut file = self.file.try_clone()?;
        let mut out = Vec::new();
        let mut offset = HEADER_LEN as u64;
        let mut buf: Vec<u8> = Vec::new();
        let mut prev_seq: Option<u64> = None;

        while offset + RECORD_HEADER_LEN as u64 <= self.file_len {
            file.seek(SeekFrom::Start(offset))?;
            let mut hdr = [0u8; RECORD_HEADER_LEN];
            file.read_exact(&mut hdr)?;

            let record_len = le_u32(&hdr[0..4]) as u64;
            let stored_crc = le_u32(&hdr[4..8]);
            let seq = le_u64(&hdr[8..16]);
            let rec_epoch = le_u64(&hdr[16..24]);

            if record_len < RECORD_HEADER_LEN as u64 || offset + record_len > self.file_len {
                return Err(WalError::Corrupt(format!(
                    "record at offset {offset} is out of bounds"
                )));
            }
            if rec_epoch < self.header_epoch {
                return Err(WalError::Corrupt(format!(
                    "fenced record (epoch {rec_epoch} < file epoch {})",
                    self.header_epoch
                )));
            }
            if let Some(prev) = prev_seq
                && seq <= prev
            {
                return Err(WalError::Corrupt("non-monotonic seq during read".into()));
            }
            prev_seq = Some(seq);

            let payload_len = (record_len - RECORD_HEADER_LEN as u64) as usize;
            if buf.len() < payload_len {
                buf.resize(payload_len, 0);
            }
            file.read_exact(&mut buf[..payload_len])?;

            if crc32fast::hash(&buf[..payload_len]) != stored_crc {
                return Err(WalError::Corrupt(format!("record {seq} CRC mismatch")));
            }

            if seq >= from {
                if seq > to {
                    break;
                }
                let op: Operation = bincode::deserialize(&buf[..payload_len])?;
                out.push(WalRecord {
                    seq,
                    epoch: rec_epoch,
                    op,
                });
            }
            offset += record_len;
        }
        Ok(out)
    }

    /// Rewrite the file keeping only records with `seq > keep_from` (WAL
    /// pruning / compaction primitive). Safe: writes a temp file, fsyncs it,
    /// atomically renames it over the original, fsyncs the directory, then
    /// repositions this handle. On failure the original file is untouched.
    pub fn prune(&mut self, keep_from: u64) -> WalResult<()> {
        let tmp_path = tmp_path_for(&self.path);

        let result = (|| -> WalResult<()> {
            let mut tmp = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)?;
            tmp.write_all(&build_header(self.header_epoch))?;

            let mut src = self.file.try_clone()?;
            let mut offset = HEADER_LEN as u64;
            let mut buf: Vec<u8> = Vec::new();
            while offset + RECORD_HEADER_LEN as u64 <= self.file_len {
                src.seek(SeekFrom::Start(offset))?;
                let mut hdr = [0u8; RECORD_HEADER_LEN];
                src.read_exact(&mut hdr)?;

                let record_len = le_u32(&hdr[0..4]) as u64;
                let seq = le_u64(&hdr[8..16]);
                if record_len < RECORD_HEADER_LEN as u64 || offset + record_len > self.file_len {
                    return Err(WalError::Corrupt(format!(
                        "record at offset {offset} is out of bounds during prune"
                    )));
                }
                if seq > keep_from {
                    let rec_len = record_len as usize;
                    if buf.len() < rec_len {
                        buf.resize(rec_len, 0);
                    }
                    src.seek(SeekFrom::Start(offset))?;
                    src.read_exact(&mut buf[..rec_len])?;
                    tmp.write_all(&buf[..rec_len])?;
                }
                offset += record_len;
            }
            tmp.sync_all()?;
            drop(tmp);
            std::fs::rename(&tmp_path, &self.path)?;
            sync_dir(&self.path)?;
            Ok(())
        })();
        if let Err(e) = result {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e);
        }

        let mut new_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)?;
        let file_len = new_file.metadata()?.len();
        let (valid_len, len, last_seq) = recover(
            &mut new_file,
            file_len,
            self.header_epoch,
            self.options.fsync,
        )?;
        self.file = new_file;
        self.file_len = valid_len;
        self.len = len;
        self.last_seq = last_seq;
        Ok(())
    }

    /// The epoch this file was created with.
    pub fn epoch(&self) -> u64 {
        self.header_epoch
    }

    /// Total on-disk size in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.file_len
    }
}

/// Build the 18-byte file header for `epoch`.
fn build_header(epoch: u64) -> [u8; HEADER_LEN] {
    let mut hdr = [0u8; HEADER_LEN];
    hdr[0..4].copy_from_slice(&MAGIC);
    hdr[4..6].copy_from_slice(&VERSION.to_le_bytes());
    hdr[6..14].copy_from_slice(&epoch.to_le_bytes());
    let crc = crc32fast::hash(&hdr[0..14]);
    hdr[14..18].copy_from_slice(&crc.to_le_bytes());
    hdr
}

/// Read and validate the file header; returns the file's creation epoch.
fn read_and_verify_header(file: &mut File) -> WalResult<u64> {
    file.seek(SeekFrom::Start(0))?;
    let mut hdr = [0u8; HEADER_LEN];
    match file.read_exact(&mut hdr) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(WalError::Corrupt("truncated WAL header".into()));
        }
        Err(e) => return Err(WalError::Io(e)),
    }

    if hdr[0..4] != MAGIC {
        return Err(WalError::Corrupt(format!(
            "bad magic: expected RKW1, got {:?}",
            &hdr[0..4]
        )));
    }
    let version = le_u16(&hdr[4..6]);
    if version != VERSION {
        return Err(WalError::Corrupt(format!(
            "unsupported WAL version {version} (expected {VERSION})"
        )));
    }
    let stored_crc = le_u32(&hdr[14..18]);
    let actual_crc = crc32fast::hash(&hdr[0..14]);
    if stored_crc != actual_crc {
        return Err(WalError::Corrupt("header CRC mismatch".into()));
    }
    Ok(le_u64(&hdr[6..14]))
}

/// Scan `file` from the header onwards, stopping at the first invalid record
/// (bad length, truncated, bad CRC, non-monotonic seq, fenced epoch, or
/// payload that fails to deserialize). Returns `(valid_end, record_count,
/// last_seq)`.
fn scan(file: &mut File, header_epoch: u64, file_len: u64) -> WalResult<(u64, u64, u64)> {
    let mut offset = HEADER_LEN as u64;
    let mut count: u64 = 0;
    let mut last_seq: u64 = 0;
    let mut prev_seq: Option<u64> = None;
    let mut buf: Vec<u8> = Vec::new();

    while offset + RECORD_HEADER_LEN as u64 <= file_len {
        file.seek(SeekFrom::Start(offset))?;
        let mut hdr = [0u8; RECORD_HEADER_LEN];
        file.read_exact(&mut hdr)?;

        let record_len = le_u32(&hdr[0..4]) as u64;
        let stored_crc = le_u32(&hdr[4..8]);
        let seq = le_u64(&hdr[8..16]);
        let rec_epoch = le_u64(&hdr[16..24]);

        if record_len < RECORD_HEADER_LEN as u64 || offset + record_len > file_len {
            break;
        }
        // A record stamped with an epoch older than the file's creation epoch
        // is the trace of a stale (fenced) writer; discard it as corrupt tail.
        if rec_epoch < header_epoch {
            break;
        }
        if let Some(prev) = prev_seq
            && seq <= prev
        {
            break;
        }

        let payload_len = (record_len - RECORD_HEADER_LEN as u64) as usize;
        if buf.len() < payload_len {
            buf.resize(payload_len, 0);
        }
        file.read_exact(&mut buf[..payload_len])?;

        if crc32fast::hash(&buf[..payload_len]) != stored_crc {
            break;
        }
        if bincode::deserialize::<Operation>(&buf[..payload_len]).is_err() {
            break;
        }

        offset += record_len;
        count += 1;
        last_seq = seq;
        prev_seq = Some(seq);
    }

    Ok((offset, count, last_seq))
}

/// Scan and truncate any torn tail. Returns `(valid_len, count, last_seq)`.
fn recover(
    file: &mut File,
    file_len: u64,
    header_epoch: u64,
    fsync: bool,
) -> WalResult<(u64, u64, u64)> {
    let (valid_end, count, last_seq) = scan(file, header_epoch, file_len)?;
    if valid_end != file_len {
        file.set_len(valid_end)?;
        if fsync {
            file.sync_all()?;
        }
    }
    Ok((valid_end, count, last_seq))
}

/// The temp-file path used for an atomic prune rewrite.
fn tmp_path_for(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".tmp");
    PathBuf::from(os)
}

/// Fsync the directory containing `path` so a rename is durable.
fn sync_dir(path: &Path) -> WalResult<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let dir = std::fs::File::open(parent)?;
    dir.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek, SeekFrom, Write};

    use tempfile::TempDir;
    use uuid::Uuid;

    use super::*;

    fn setup() -> (TempDir, Uuid) {
        let dir = tempfile::tempdir().unwrap();
        let id = Uuid::new_v4();
        (dir, id)
    }

    fn wal_path(dir: &TempDir, id: &Uuid) -> PathBuf {
        dir.path().join(format!("{id}.wal"))
    }

    fn add(id: &str) -> Operation {
        Operation::Add {
            id: id.into(),
            embedding: vec![1.0, 2.0, 3.0].into(),
            metadata: None,
            document: None,
        }
    }

    fn update(id: &str) -> Operation {
        Operation::Update {
            id: id.into(),
            metadata: None,
            document: Some("doc".into()),
        }
    }

    fn delete(id: &str) -> Operation {
        Operation::Delete { id: id.into() }
    }

    fn upsert(id: &str) -> Operation {
        Operation::Upsert {
            id: id.into(),
            embedding: vec![4.0, 5.0].into(),
            metadata: None,
            document: None,
        }
    }

    fn open_fresh(path: &TempDir, id: &Uuid, epoch: u64, fsync: bool) -> Wal {
        Wal::open(path.path(), id, Epoch(epoch), WalOptions::new(fsync)).unwrap()
    }

    #[test]
    fn append_and_replay() {
        let (dir, id) = setup();
        let opts = WalOptions::new(false);
        let mut wal = Wal::open(dir.path(), &id, Epoch(1), opts.clone()).unwrap();
        let ops = vec![add("a"), update("a"), delete("b"), upsert("c")];
        for op in &ops {
            wal.append(op.clone()).unwrap();
        }
        assert_eq!(wal.last_seq(), 4);
        assert_eq!(wal.len(), 4);
        drop(wal);

        let wal = Wal::open(dir.path(), &id, Epoch(1), opts).unwrap();
        assert_eq!(wal.last_seq(), 4);
        assert_eq!(wal.len(), 4);
        assert_eq!(wal.epoch(), 1);

        let recs = wal.read_range(1, 4).unwrap();
        assert_eq!(recs.len(), 4);
        for (i, (rec, op)) in recs.iter().zip(&ops).enumerate() {
            assert_eq!(rec.op, *op);
            assert_eq!(rec.epoch, 1);
            assert_eq!(rec.seq, i as u64 + 1);
        }
        // Reading past the end is bounded by the log contents.
        assert_eq!(wal.read_range(1, 100).unwrap().len(), 4);
    }

    #[test]
    fn fsync_and_no_fsync_paths() {
        for fsync in [true, false] {
            let (dir, id) = setup();
            let opts = WalOptions::new(fsync);
            let mut wal = Wal::open(dir.path(), &id, Epoch(1), opts.clone()).unwrap();
            wal.append(add("x")).unwrap();
            wal.append(update("x")).unwrap();
            drop(wal);

            let wal = Wal::open(dir.path(), &id, Epoch(1), opts).unwrap();
            assert_eq!(wal.len(), 2);
            assert_eq!(wal.last_seq(), 2);
            assert_eq!(wal.read_range(1, 2).unwrap().len(), 2);
        }
    }

    #[test]
    fn torn_tail_recovery() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        // Track boundaries so we know where record 4 lives.
        wal.append(add("1")).unwrap();
        wal.append(add("2")).unwrap();
        wal.append(add("3")).unwrap();
        let off3 = wal.size_bytes(); // end of record 3 == start of record 4
        wal.append(add("4")).unwrap();
        let off4 = wal.size_bytes(); // end of record 4 == start of record 5
        wal.append(add("5")).unwrap();
        drop(wal);

        // Truncate mid-record-4 (record 4 spans [off3, off4)).
        let mid4 = off3 + (off4 - off3) / 2;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(wal_path(&dir, &id))
            .unwrap();
        file.set_len(mid4).unwrap();
        drop(file);

        let mut wal = Wal::open(dir.path(), &id, Epoch(1), WalOptions::new(true)).unwrap();
        assert_eq!(wal.len(), 3);
        assert_eq!(wal.last_seq(), 3);
        assert_eq!(wal.epoch(), 1);
        // File was re-truncated cleanly to the end of record 3.
        assert_eq!(wal.size_bytes(), off3);

        let recs = wal.read_range(1, 3).unwrap();
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[0].seq, 1);
        assert_eq!(recs[1].seq, 2);
        assert_eq!(recs[2].seq, 3);

        // Appends continue at seq 4.
        let r = wal.append(delete("after")).unwrap();
        assert_eq!(r.seq, 4);
        assert_eq!(wal.len(), 4);
        assert_eq!(wal.last_seq(), 4);
    }

    #[test]
    fn corrupt_middle_discards_tail() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        wal.append(add("1")).unwrap();
        let off1 = wal.size_bytes();
        wal.append(add("2")).unwrap();
        let off2 = wal.size_bytes();
        wal.append(add("3")).unwrap();
        let off3 = wal.size_bytes();
        wal.append(add("4")).unwrap();
        let off4 = wal.size_bytes();
        wal.append(add("5")).unwrap();
        drop(wal);
        assert!(off2 > off1 + 24 && off3 > off2 && off4 > off3);

        // Flip one payload byte of record 2 (payload starts at off1+24).
        let path = wal_path(&dir, &id);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        let mut byte = [0u8; 1];
        file.seek(SeekFrom::Start(off1 + 24)).unwrap();
        file.read_exact(&mut byte).unwrap();
        byte[0] ^= 0xFF;
        file.seek(SeekFrom::Start(off1 + 24)).unwrap();
        file.write_all(&byte).unwrap();
        drop(file);

        let wal = Wal::open(dir.path(), &id, Epoch(1), WalOptions::new(true)).unwrap();
        assert_eq!(wal.len(), 1);
        assert_eq!(wal.last_seq(), 1);
        assert_eq!(wal.size_bytes(), off1);
        let recs = wal.read_range(1, 100).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].seq, 1);
    }

    #[test]
    fn header_corruption_errors() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        wal.append(add("x")).unwrap();
        drop(wal);

        // Corrupt the magic bytes.
        let path = wal_path(&dir, &id);
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(b"XXXX").unwrap();
        drop(file);

        let result = Wal::open(dir.path(), &id, Epoch(1), WalOptions::new(true));
        match result {
            Err(WalError::Corrupt(_)) => {}
            other => panic!("expected Corrupt, got {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn fenced_record_discarded_on_replay() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 5, true);
        wal.append(add("1")).unwrap();
        wal.append(add("2")).unwrap();
        drop(wal);

        // Craft a raw record with a VALID CRC but a fenced epoch (0 < file 5).
        let path = wal_path(&dir, &id);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::End(0)).unwrap();
        let payload = bincode::serialize(&delete("fenced")).unwrap();
        let record_len = (RECORD_HEADER_LEN as u32) + payload.len() as u32;
        let mut rec = Vec::with_capacity(RECORD_HEADER_LEN + payload.len());
        rec.extend_from_slice(&record_len.to_le_bytes());
        rec.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
        rec.extend_from_slice(&3u64.to_le_bytes());
        rec.extend_from_slice(&0u64.to_le_bytes());
        rec.extend_from_slice(&payload);
        file.write_all(&rec).unwrap();
        file.sync_all().unwrap();
        drop(file);

        // Reopen at the file's epoch: the fenced record is a corrupt tail.
        let mut wal = Wal::open(dir.path(), &id, Epoch(5), WalOptions::new(true)).unwrap();
        assert_eq!(wal.epoch(), 5);
        assert_eq!(wal.len(), 2);
        assert_eq!(wal.last_seq(), 2);

        // Appends continue cleanly at seq 3, stamped with the file epoch.
        let r = wal.append(upsert("after")).unwrap();
        assert_eq!(r.seq, 3);
        assert_eq!(r.epoch, 5);
        assert_eq!(wal.len(), 3);
    }

    #[test]
    fn prune_keeps_only_records_above_watermark() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        for i in 0..5 {
            wal.append(add(&format!("r{i}"))).unwrap();
        }
        assert_eq!(wal.len(), 5);

        wal.prune(3).unwrap();
        assert_eq!(wal.len(), 2);
        assert_eq!(wal.last_seq(), 5);

        let recs = wal.read_range(1, 5).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].seq, 4);
        assert_eq!(recs[1].seq, 5);
    }

    #[test]
    fn prune_persists_across_reopen() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        for i in 0..5 {
            wal.append(add(&format!("r{i}"))).unwrap();
        }
        wal.prune(3).unwrap();
        drop(wal);

        let wal = Wal::open(dir.path(), &id, Epoch(1), WalOptions::new(true)).unwrap();
        assert_eq!(wal.len(), 2);
        assert_eq!(wal.last_seq(), 5);
        let recs = wal.read_range(1, 100).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].seq, 4);
        assert_eq!(recs[1].seq, 5);

        // Appends continue at seq 6.
        let mut wal = wal;
        let r = wal.append(delete("x")).unwrap();
        assert_eq!(r.seq, 6);
        assert_eq!(wal.len(), 3);
        assert_eq!(wal.last_seq(), 6);
    }

    #[test]
    fn prune_to_empty_restarts_seq() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        wal.append(add("1")).unwrap();
        wal.append(add("2")).unwrap();
        wal.append(add("3")).unwrap();
        wal.prune(3).unwrap(); // keep seq > 3 → nothing

        assert_eq!(wal.len(), 0);
        assert_eq!(wal.last_seq(), 0);
        assert_eq!(wal.size_bytes(), HEADER_LEN as u64);

        let r = wal.append(add("fresh")).unwrap();
        assert_eq!(r.seq, 1);
        assert_eq!(wal.len(), 1);
    }

    #[test]
    fn empty_file_opens_and_appends() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        assert_eq!(wal.len(), 0);
        assert_eq!(wal.last_seq(), 0);
        assert_eq!(wal.epoch(), 1);
        assert_eq!(wal.read_range(1, 1).unwrap().len(), 0);

        let r = wal.append(add("first")).unwrap();
        assert_eq!(r.seq, 1);
        assert_eq!(wal.len(), 1);
        assert_eq!(wal.last_seq(), 1);
    }

    #[test]
    fn read_range_out_of_order_is_empty() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        wal.append(add("1")).unwrap();
        assert!(wal.read_range(2, 1).unwrap().is_empty());
    }

    #[test]
    fn large_payload_roundtrips() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        let big: Vec<f32> = (0..100_000).map(|i| i as f32 / 1000.0).collect();
        let op = Operation::Add {
            id: "big".into(),
            embedding: big.into(),
            metadata: None,
            document: None,
        };
        let rec = wal.append(op.clone()).unwrap();
        assert_eq!(rec.seq, 1);
        assert!(wal.size_bytes() > 400_000);
        drop(wal);

        let wal = Wal::open(dir.path(), &id, Epoch(1), WalOptions::new(true)).unwrap();
        let recs = wal.read_range(1, 1).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].op, op);
        assert_eq!(recs[0].seq, 1);
    }

    #[test]
    fn start_seq_on_fresh_file_seeds_next_append() {
        let (dir, id) = setup();
        let opts = WalOptions {
            fsync: true,
            start_seq: Some(501),
        };
        let mut wal = Wal::open(dir.path(), &id, Epoch(1), opts.clone()).unwrap();
        assert_eq!(wal.len(), 0);
        // The resume position is start_seq - 1: the next append lands on 501.
        assert_eq!(wal.last_seq(), 500);

        let r1 = wal.append(add("a")).unwrap();
        assert_eq!(r1.seq, 501);
        let r2 = wal.append(add("b")).unwrap();
        assert_eq!(r2.seq, 502);
        assert_eq!(wal.last_seq(), 502);

        // A reopen of a non-empty file ignores start_seq (file position wins).
        drop(wal);
        let mut wal = Wal::open(dir.path(), &id, Epoch(1), opts).unwrap();
        assert_eq!(wal.len(), 2);
        assert_eq!(wal.last_seq(), 502);
        let r3 = wal.append(add("c")).unwrap();
        assert_eq!(r3.seq, 503);
    }

    #[test]
    fn start_seq_ignored_when_file_has_records() {
        let (dir, id) = setup();
        {
            let mut wal = open_fresh(&dir, &id, 1, true);
            for i in 0..3 {
                wal.append(add(&format!("r{i}"))).unwrap();
            }
            assert_eq!(wal.last_seq(), 3);
        }

        // start_seq = Some(1000) must be ignored: records already exist.
        let opts = WalOptions {
            fsync: true,
            start_seq: Some(1000),
        };
        let mut wal = Wal::open(dir.path(), &id, Epoch(1), opts).unwrap();
        assert_eq!(wal.last_seq(), 3);
        let r = wal.append(add("after")).unwrap();
        assert_eq!(r.seq, 4);
    }
}
