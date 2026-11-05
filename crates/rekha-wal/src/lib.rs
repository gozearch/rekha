#![deny(clippy::unwrap_used, clippy::expect_used)]

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
//! - **Batch framing with a commit marker.** Operations are written one batch
//!   per request, each with a CRC32 over its header and entries plus a trailing
//!   commit sentinel. A batch is complete only if both are intact, so an
//!   interrupted write is *provably* incomplete instead of merely suspicious.
//!   Recovery discards a suspect final batch and refuses to truncate damage
//!   that has committed data behind it
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
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Process-wide count of WAL `fsync` calls for appended batches.
///
/// A per-handle counter is useless here: a checkpoint prunes the log and
/// reopens it, which would reset the count to zero mid-run and hide the very
/// cost being measured.
static SYNC_COUNT: AtomicU64 = AtomicU64::new(0);

/// Total `fsync` calls this process has issued for appended WAL batches.
pub fn sync_count() -> u64 {
    SYNC_COUNT.load(Ordering::Relaxed)
}

/// Magic bytes identifying a RekhaDB WAL file: `"RKW2"`.
///
/// Operations are framed into **batches** with a trailing commit marker, so a
/// partially written batch is distinguishable from a complete one.
const MAGIC: [u8; 4] = *b"RKW2";
/// Only on-disk format version this build reads and writes.
const VERSION: u16 = 2;
/// Sentinel written after a batch's entries. Its presence (plus a matching
/// checksum) is what makes "this batch is complete" provable.
const COMMIT_MAGIC: u32 = 0xB47C_4005;
/// `batch_len(4) + crc(4) + first_seq(8) + count(4)`.
const BATCH_HEADER_LEN: usize = 20;
/// `seq(8) + epoch(8) + payload_len(4)`.
const ENTRY_HEADER_LEN: usize = 20;
/// Smallest possible batch: header, one entry header, and the commit marker.
const MIN_BATCH_LEN: u64 = 4 + 8 + 4 + ENTRY_HEADER_LEN as u64 + 4;
/// Fixed size of the file header: magic(4) + version(2) + epoch(8) + crc(4).
const HEADER_LEN: usize = 18;

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

/// Why the tail of a log could not be verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailReason {
    /// Fewer bytes remain than the frame claims.
    TruncatedFrame,
    /// A complete frame whose payload does not match its checksum.
    ChecksumMismatch,
    /// A frame whose sequence does not advance.
    NonMonotonicSeq,
    /// A frame whose payload does not decode.
    UndecodablePayload,
    /// A frame stamped with an epoch older than the file's.
    FencedEpoch,
}

impl TailReason {
    /// Human-readable description, for diagnostics.
    pub fn describe(self) -> &'static str {
        match self {
            TailReason::TruncatedFrame => "truncated frame",
            TailReason::ChecksumMismatch => "checksum mismatch",
            TailReason::NonMonotonicSeq => "non-monotonic seq",
            TailReason::UndecodablePayload => "undecodable payload",
            TailReason::FencedEpoch => "fenced epoch",
        }
    }
}

/// What the end of the log looks like.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalTail {
    /// Every byte is a complete, verified record.
    Clean,
    /// The invalid region runs to end-of-file.
    ///
    /// A missing commit marker is *proof* that the batch never completed. A
    /// batch whose marker is present but whose checksum fails is ambiguous —
    /// a torn write that persisted the marker, or bit rot — but because nothing
    /// follows it, discarding it is still the only safe repair.
    Suspect { reason: TailReason, at_seq: u64 },
    /// Damage followed by more data: this is not a tail, it is a hole.
    Corrupt { reason: TailReason, at_offset: u64 },
}

/// A read-only view of a log file, produced without modifying it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalInspection {
    /// Whether the file exists at all.
    pub exists: bool,
    pub epoch: u64,
    pub record_count: u64,
    /// Sequence of the first verified record; `0` when there are none.
    pub first_seq: u64,
    pub last_verified_seq: u64,
    /// Byte length of the verified prefix.
    pub valid_len: u64,
    pub file_len: u64,
    pub tail: WalTail,
}

/// How [`Wal::open`] may treat an unverifiable tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailRepair {
    /// Never modify the file. Any non-clean tail is an error.
    Refuse,
    /// Discard a `Suspect` tail. A `Corrupt` tail is always an error.
    DiscardSuspect,
}

/// An append-only, per-collection write-ahead log.
///
/// On disk: one file per collection at `{prefix}/{collection_id}.wal`
///
/// ```text
/// [FileHeader] { magic "RKW2" | version u16 | epoch u64 | header_crc u32 }
/// [Record]×N   { length u32 | crc u32 | seq u64 | epoch u64 | payload }
/// ```
///
/// `length` spans the whole record (`24 + len(payload)`); the record CRC
/// covers the payload only. `seq` is strictly increasing per file (1, 2, 3, …;
/// a resumed file continues at `last_seq + 1`).
///
/// On open the header is verified and the batch chain is scanned. An
/// incomplete final batch may be discarded (see [`TailRepair`]); anything else
/// is an error.
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
    /// On open: verify the header, scan the batches, and apply `repair` to an
    /// incomplete final batch, then position this handle at the recovered
    /// `last_seq`. A bad header (magic/version/CRC) is a [`WalError::Corrupt`];
    /// the header is never truncated away.
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
        repair: TailRepair,
    ) -> WalResult<Self> {
        Self::open_named(path, &collection_id.to_string(), epoch, options, repair)
    }

    /// Open (or create) the WAL for a named stream (e.g. partition) at `path` under `epoch`.
    pub fn open_named(
        path: impl AsRef<Path>,
        name: &str,
        epoch: Epoch,
        options: WalOptions,
        repair: TailRepair,
    ) -> WalResult<Self> {
        let dir = path.as_ref();
        std::fs::create_dir_all(dir)?;
        let file_path = dir.join(format!("{name}.wal"));

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
                // The file's *name* is not durable until its directory is. A
                // crash between creating the file and syncing the directory can
                // leave the header blocks on disk with no directory entry
                // pointing at them, and recovery then reports the collection as
                // missing even though every acknowledged write to it was synced.
                sync_dir(&file_path)?;
            }
            // A `start_seq` override moves the next append up to that seq, so
            // a checkpointed-and-fully-pruned collection resumes where its
            // checkpoint said it would.
            let last_seq = options.start_seq.map(|n| n.saturating_sub(1)).unwrap_or(0);
            (epoch.0, HEADER_LEN as u64, 0, last_seq)
        } else {
            // File position wins: `start_seq` is ignored when records exist.
            let header_epoch = read_and_verify_header(&mut file)?;
            let scanned = recover(&mut file, file_len, header_epoch, repair, options.fsync)?;
            let (valid_len, len, last_seq) = (scanned.valid_end, scanned.count, scanned.last_seq);
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

    /// Inspect a log **without modifying it**.
    ///
    /// Opening a collection used to repair the log before the checkpoint was
    /// read, so a truncation decision was taken before the code knew whether the
    /// checkpoint already covered those records. Inspection separates the two:
    /// the caller decides, then opens with an explicit [`TailRepair`].
    ///
    /// A missing file is reported as `exists: false` rather than created.
    pub fn inspect(path: impl AsRef<Path>, collection_id: &Uuid) -> WalResult<WalInspection> {
        Self::inspect_named(path, &collection_id.to_string())
    }

    /// Inspect a named log **without modifying it**.
    pub fn inspect_named(path: impl AsRef<Path>, name: &str) -> WalResult<WalInspection> {
        let file_path = path.as_ref().join(format!("{name}.wal"));
        let mut file = match std::fs::File::open(&file_path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(WalInspection {
                    exists: false,
                    epoch: 0,
                    record_count: 0,
                    first_seq: 0,
                    last_verified_seq: 0,
                    valid_len: 0,
                    file_len: 0,
                    tail: WalTail::Clean,
                });
            }
            Err(e) => return Err(e.into()),
        };

        let file_len = file.metadata()?.len();
        if file_len == 0 {
            return Ok(WalInspection {
                exists: true,
                epoch: 0,
                record_count: 0,
                first_seq: 0,
                last_verified_seq: 0,
                valid_len: 0,
                file_len,
                tail: WalTail::Clean,
            });
        }

        let header_epoch = read_and_verify_header(&mut file)?;
        let scanned = scan(&mut file, header_epoch, file_len)?;
        Ok(WalInspection {
            exists: true,
            epoch: header_epoch,
            record_count: scanned.count,
            first_seq: scanned.first_seq,
            last_verified_seq: scanned.last_seq,
            valid_len: scanned.valid_end,
            file_len,
            tail: scanned.tail,
        })
    }

    /// Append one operation. Equivalent to a single-operation batch.
    pub fn append(&mut self, op: Operation) -> WalResult<WalRecord> {
        let mut records = self.append_batch(&[op])?;
        Ok(records.remove(0))
    }

    /// Append a batch of operations as **one atomic unit**.
    ///
    /// The whole batch is written as a single framed unit with a trailing
    /// commit marker and fsynced once, so a crash either preserves the entire
    /// batch or none of it.
    pub fn append_batch(&mut self, ops: &[Operation]) -> WalResult<Vec<WalRecord>> {
        if ops.is_empty() {
            return Ok(Vec::new());
        }

        let mut records = Vec::with_capacity(ops.len());
        let mut payloads = Vec::with_capacity(ops.len());
        let mut seq = self.last_seq;
        for op in ops {
            seq += 1;
            payloads.push(bincode::serialize(op)?);
            records.push(WalRecord {
                seq,
                epoch: self.header_epoch,
                op: op.clone(),
            });
        }

        let encoded: Vec<(u64, u64, Vec<u8>)> = records
            .iter()
            .zip(payloads.iter())
            .map(|(r, p)| (r.seq, r.epoch, p.clone()))
            .collect();
        let buf = encode_batch(&encoded, self.last_seq + 1)?;

        self.file.seek(SeekFrom::Start(self.file_len))?;
        self.file.write_all(&buf)?;
        if self.options.fsync {
            self.file.sync_all()?;
            SYNC_COUNT.fetch_add(1, Ordering::Relaxed);
        }

        self.file_len += buf.len() as u64;
        self.last_seq = seq;
        self.len += ops.len() as u64;
        Ok(records)
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
        scan_entries(&mut file, self.file_len, from, to)?
            .into_iter()
            .map(|(seq, epoch, payload)| {
                let op: Operation = bincode::deserialize(&payload)?;
                Ok(WalRecord { seq, epoch, op })
            })
            .collect()
    }

    /// Rewrite the file keeping only records with `seq > keep_from` (WAL
    /// pruning / compaction primitive). Safe: writes a temp file, fsyncs it,
    /// atomically renames it over the original, fsyncs the directory, then
    /// repositions this handle. On failure the original file is untouched.
    pub fn prune(&mut self, keep_from: u64) -> WalResult<()> {
        let tmp_path = tmp_path_for(&self.path);

        let result = (|| -> WalResult<()> {
            // Collect the retained entries and re-encode them as one batch:
            // copying raw frame bytes would carry the old batch boundaries, so
            // the surviving records have to be re-framed.
            let mut src = self.file.try_clone()?;
            let retained: Vec<(u64, u64, Vec<u8>)> =
                scan_entries(&mut src, self.file_len, keep_from, u64::MAX)?
                    .into_iter()
                    .filter(|(seq, _, _)| *seq > keep_from)
                    .collect();

            let mut tmp = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)?;
            tmp.write_all(&build_header(self.header_epoch))?;

            if !retained.is_empty() {
                let first_seq = retained.first().map(|(seq, _, _)| *seq).unwrap_or(1);
                tmp.write_all(&encode_batch(&retained, first_seq)?)?;
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
        let scanned = recover(
            &mut new_file,
            file_len,
            self.header_epoch,
            // The file was just written by this process; anything unverifiable
            // in it is a bug here, not a crash artifact.
            TailRepair::Refuse,
            self.options.fsync,
        )?;
        self.file = new_file;
        self.file_len = scanned.valid_end;
        self.len = scanned.count;
        self.last_seq = scanned.last_seq;
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
            "bad magic: expected RKW2, got {:?}",
            &hdr[0..4]
        )));
    }
    let version = le_u16(&hdr[4..6]);
    if version != VERSION {
        return Err(WalError::Corrupt(format!(
            "unsupported WAL version {version} (this build reads {VERSION})"
        )));
    }
    let stored_crc = le_u32(&hdr[14..18]);
    let actual_crc = crc32fast::hash(&hdr[0..14]);
    if stored_crc != actual_crc {
        return Err(WalError::Corrupt("header CRC mismatch".into()));
    }
    Ok(le_u64(&hdr[6..14]))
}

/// Result of scanning a log file.
struct ScanResult {
    /// Byte offset just past the last verified record.
    valid_end: u64,
    count: u64,
    first_seq: u64,
    last_seq: u64,
    tail: WalTail,
}

/// Scan and, only when the caller allows it, discard a `Suspect` tail.
fn recover(
    file: &mut File,
    file_len: u64,
    header_epoch: u64,
    repair: TailRepair,
    fsync: bool,
) -> WalResult<ScanResult> {
    let scan = scan(file, header_epoch, file_len)?;
    match scan.tail {
        WalTail::Clean => Ok(scan),
        WalTail::Suspect { reason, at_seq } => match repair {
            TailRepair::DiscardSuspect => {
                tracing::warn!(
                    reason = reason.describe(),
                    at_seq,
                    discarded_bytes = file_len - scan.valid_end,
                    "discarding an unverifiable WAL tail"
                );
                file.set_len(scan.valid_end)?;
                if fsync {
                    file.sync_all()?;
                }
                Ok(ScanResult {
                    valid_end: scan.valid_end,
                    count: scan.count,
                    first_seq: scan.first_seq,
                    last_seq: scan.last_seq,
                    tail: WalTail::Clean,
                })
            }
            TailRepair::Refuse => Err(WalError::Corrupt(format!(
                "unverifiable tail ({}) at seq {at_seq}; refusing to discard it",
                reason.describe()
            ))),
        },
        WalTail::Corrupt { reason, at_offset } => Err(WalError::Corrupt(format!(
            "corrupted history ({}) at byte offset {at_offset}; this is not a tail and \
             will not be truncated",
            reason.describe()
        ))),
    }
}

/// Read records in `[from, to]` from a batch-framed file.
fn scan_entries(
    file: &mut File,
    file_len: u64,
    from: u64,
    to: u64,
) -> WalResult<Vec<(u64, u64, Vec<u8>)>> {
    let mut out = Vec::new();
    let mut offset = HEADER_LEN as u64;
    let mut buf: Vec<u8> = Vec::new();

    while offset < file_len {
        if offset + BATCH_HEADER_LEN as u64 + 4 > file_len {
            break;
        }
        file.seek(SeekFrom::Start(offset))?;
        let mut hdr = [0u8; BATCH_HEADER_LEN];
        file.read_exact(&mut hdr)?;
        let batch_len = le_u32(&hdr[0..4]) as u64;
        if batch_len < MIN_BATCH_LEN || offset + 4 + batch_len > file_len {
            return Err(WalError::Corrupt(format!(
                "batch at offset {offset} is out of bounds"
            )));
        }
        let body_len = batch_len as usize;
        if buf.len() < body_len {
            buf.resize(body_len, 0);
        }
        file.seek(SeekFrom::Start(offset + 4))?;
        file.read_exact(&mut buf[..body_len])?;

        let commit_at = body_len - 4;
        if le_u32(&buf[commit_at..commit_at + 4]) != COMMIT_MAGIC {
            return Err(WalError::Corrupt(format!(
                "batch at offset {offset} has no commit marker"
            )));
        }
        let stored_crc = le_u32(&hdr[4..8]);
        if crc32fast::hash(&buf[4..commit_at]) != stored_crc {
            return Err(WalError::Corrupt(format!(
                "batch at offset {offset} failed its checksum"
            )));
        }

        let entry_count = le_u32(&buf[12..16]) as u64;
        let mut pos = 16;
        let mut past_range = false;
        for _ in 0..entry_count {
            if pos + ENTRY_HEADER_LEN > commit_at {
                return Err(WalError::Corrupt("entry header out of bounds".into()));
            }
            let seq = le_u64(&buf[pos..pos + 8]);
            let epoch = le_u64(&buf[pos + 8..pos + 16]);
            let plen = le_u32(&buf[pos + 16..pos + 20]) as usize;
            pos += ENTRY_HEADER_LEN;
            if pos + plen > commit_at {
                return Err(WalError::Corrupt("entry payload out of bounds".into()));
            }
            if seq > to {
                past_range = true;
                break;
            }
            if seq >= from {
                out.push((seq, epoch, buf[pos..pos + plen].to_vec()));
            }
            pos += plen;
        }
        if past_range {
            break;
        }
        offset += 4 + batch_len;
    }
    Ok(out)
}

/// Encode one batch frame from `ops`, starting at `first_seq`.
fn encode_batch(ops: &[(u64, u64, Vec<u8>)], first_seq: u64) -> WalResult<Vec<u8>> {
    let count = ops.len() as u32;
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(&first_seq.to_le_bytes());
    body.extend_from_slice(&count.to_le_bytes());
    for (seq, epoch, payload) in ops {
        let plen = payload.len() as u32;
        body.extend_from_slice(&seq.to_le_bytes());
        body.extend_from_slice(&epoch.to_le_bytes());
        body.extend_from_slice(&plen.to_le_bytes());
        body.extend_from_slice(payload);
    }

    let crc = crc32fast::hash(&body);
    // `batch_len` covers everything after the length field itself.
    let batch_len = (4 + body.len() + 4) as u32;

    let mut frame = Vec::with_capacity(4 + batch_len as usize);
    frame.extend_from_slice(&batch_len.to_le_bytes());
    frame.extend_from_slice(&crc.to_le_bytes());
    frame.extend_from_slice(&body);
    frame.extend_from_slice(&COMMIT_MAGIC.to_le_bytes());
    Ok(frame)
}

/// Whether a complete, checksum-valid batch that continues the sequence exists
/// after `from`.
///
/// This is what separates a torn tail from a hole. A torn tail is the last thing
/// in the file, so nothing valid can follow it. If a valid continuation does
/// exist, the damage is in the middle of committed history: the bytes behind it
/// were acknowledged, and discarding them because the *length field* of the
/// preceding frame is unreadable is silent data loss.
///
/// The search is byte-by-byte, because a corrupted length field is exactly the
/// case where the next frame's position cannot be computed. It runs only on the
/// failure path, over an in-memory copy of the remainder, so a normal open pays
/// nothing for it. `RESYNC_LIMIT` bounds the copy: a continuation further away
/// than that is not treated as evidence, and the classification stays
/// conservative in the direction that preserves data (it reports a tail rather
/// than refusing to start).
fn has_valid_continuation(file: &mut File, file_len: u64, from: u64, after_seq: u64) -> bool {
    const RESYNC_LIMIT: u64 = 256 * 1024 * 1024;

    if from >= file_len {
        return false;
    }
    let end = file_len.min(from.saturating_add(RESYNC_LIMIT));
    let len = (end - from) as usize;
    let mut bytes = vec![0u8; len];
    if file.seek(SeekFrom::Start(from)).is_err() || file.read_exact(&mut bytes).is_err() {
        return false;
    }

    let header = BATCH_HEADER_LEN;
    let mut at = 0usize;
    while at + header + 4 <= bytes.len() {
        let batch_len = le_u32(&bytes[at..at + 4]) as usize;
        let stored_crc = le_u32(&bytes[at + 4..at + 8]);
        let batch_first = le_u64(&bytes[at + 8..at + 16]);
        // Any frame carrying a sequence beyond the last one successfully read is
        // a continuation. Comparing against the damaged frame's own `batch_first`
        // would be circular, and batches are not contiguous in index the way a
        // single-entry frame suggests: the next batch simply starts after it.
        if batch_first > after_seq
            && batch_len as u64 >= MIN_BATCH_LEN
            && at + 4 + batch_len <= bytes.len()
        {
            let commit_at = batch_len - 4;
            let body = &bytes[at + 4..at + 4 + batch_len];
            if le_u32(&body[commit_at..commit_at + 4]) == COMMIT_MAGIC
                && crc32fast::hash(&body[4..commit_at]) == stored_crc
            {
                return true;
            }
        }
        at += 1;
    }
    false
}

/// Scan the log, classifying anything that stops it.
///
/// A batch is complete only if its commit marker is present and its checksum
/// matches. That is the whole point of the format: an incomplete final batch is
/// provably incomplete, rather than "a record whose payload looks wrong".
///
/// The distinction that matters is **tail versus hole**: a problem in the final
/// batch may be discarded, while the same problem with more data behind it is a
/// hole in committed history and must never be truncated.
fn scan(file: &mut File, header_epoch: u64, file_len: u64) -> WalResult<ScanResult> {
    let mut offset = HEADER_LEN as u64;
    let mut count: u64 = 0;
    let mut first_seq: u64 = 0;
    let mut last_seq: u64 = 0;
    let mut buf: Vec<u8> = Vec::new();

    // A batch can only be *discarded* when nothing survives after it. Damage
    // with more data behind it is a hole in committed history.
    let outcome = |reason: TailReason,
                   at_seq: u64,
                   at: u64,
                   ends_log: bool,
                   count: u64,
                   first: u64,
                   last: u64|
     -> ScanResult {
        ScanResult {
            valid_end: at,
            count,
            first_seq: first,
            last_seq: last,
            tail: if ends_log {
                WalTail::Suspect { reason, at_seq }
            } else {
                WalTail::Corrupt {
                    reason,
                    at_offset: at,
                }
            },
        }
    };

    while offset < file_len {
        if offset + BATCH_HEADER_LEN as u64 + 4 > file_len {
            return Ok(outcome(
                TailReason::TruncatedFrame,
                last_seq + 1,
                offset,
                true,
                count,
                first_seq,
                last_seq,
            ));
        }

        file.seek(SeekFrom::Start(offset))?;
        let mut hdr = [0u8; BATCH_HEADER_LEN];
        file.read_exact(&mut hdr)?;

        let batch_len = le_u32(&hdr[0..4]) as u64;
        let stored_crc = le_u32(&hdr[4..8]);
        let batch_first = le_u64(&hdr[8..16]);

        if batch_len < MIN_BATCH_LEN || offset + 4 + batch_len > file_len {
            // A length field that cannot describe a frame is not proof of a torn
            // tail: a truncated write leaves a *correct* length with a short
            // body, while a corrupted one leaves a length that fits nothing. Ask
            // whether the sequence continues behind this offset before deciding.
            let next = batch_first.max(last_seq + 1);
            let ends_log = !has_valid_continuation(file, file_len, offset + 1, last_seq);
            return Ok(outcome(
                TailReason::TruncatedFrame,
                next,
                offset,
                ends_log,
                count,
                first_seq,
                last_seq,
            ));
        }

        // `ends_log` cannot be judged from position alone: a corrupted length
        // can make a mid-file frame appear to run to the end of the file. Every
        // failure below therefore also asks whether the sequence continues after
        // this offset, and a continuation means the damage is a hole.
        let last_by_position = offset + 4 + batch_len >= file_len;
        let body_len = batch_len as usize;
        if buf.len() < body_len {
            buf.resize(body_len, 0);
        }
        file.seek(SeekFrom::Start(offset + 4))?;
        file.read_exact(&mut buf[..body_len])?;

        let commit_at = body_len - 4;
        if le_u32(&buf[commit_at..commit_at + 4]) != COMMIT_MAGIC {
            // No marker: the write did not finish.
            return Ok(outcome(
                TailReason::TruncatedFrame,
                batch_first.max(last_seq + 1),
                offset,
                last_by_position && !has_valid_continuation(file, file_len, offset + 1, last_seq),
                count,
                first_seq,
                last_seq,
            ));
        }
        if crc32fast::hash(&buf[4..commit_at]) != stored_crc {
            return Ok(outcome(
                TailReason::ChecksumMismatch,
                batch_first.max(last_seq + 1),
                offset,
                last_by_position && !has_valid_continuation(file, file_len, offset + 1, last_seq),
                count,
                first_seq,
                last_seq,
            ));
        }

        let mut pos = 16;
        let entry_count = le_u32(&buf[12..16]) as u64;
        let mut batch_last = last_seq;
        let mut parsed: u64 = 0;
        for _ in 0..entry_count {
            if pos + ENTRY_HEADER_LEN > commit_at {
                return Ok(outcome(
                    TailReason::UndecodablePayload,
                    batch_first.max(last_seq + 1),
                    offset,
                    last_by_position
                        && !has_valid_continuation(file, file_len, offset + 1, last_seq),
                    count,
                    first_seq,
                    last_seq,
                ));
            }
            let seq = le_u64(&buf[pos..pos + 8]);
            let epoch = le_u64(&buf[pos + 8..pos + 16]);
            let plen = le_u32(&buf[pos + 16..pos + 20]) as usize;
            pos += ENTRY_HEADER_LEN;
            if pos + plen > commit_at {
                return Ok(outcome(
                    TailReason::UndecodablePayload,
                    batch_first.max(last_seq + 1),
                    offset,
                    last_by_position
                        && !has_valid_continuation(file, file_len, offset + 1, last_seq),
                    count,
                    first_seq,
                    last_seq,
                ));
            }
            if epoch < header_epoch {
                return Ok(outcome(
                    TailReason::FencedEpoch,
                    seq,
                    offset,
                    last_by_position
                        && !has_valid_continuation(file, file_len, offset + 1, last_seq),
                    count,
                    first_seq,
                    last_seq,
                ));
            }
            if parsed == 0 && seq != batch_first {
                return Ok(outcome(
                    TailReason::NonMonotonicSeq,
                    seq,
                    offset,
                    last_by_position
                        && !has_valid_continuation(file, file_len, offset + 1, last_seq),
                    count,
                    first_seq,
                    last_seq,
                ));
            }
            if parsed > 0 && seq <= batch_last {
                return Ok(outcome(
                    TailReason::NonMonotonicSeq,
                    seq,
                    offset,
                    last_by_position
                        && !has_valid_continuation(file, file_len, offset + 1, last_seq),
                    count,
                    first_seq,
                    last_seq,
                ));
            }
            batch_last = seq;
            parsed += 1;
            pos += plen;
        }

        if count == 0 && parsed > 0 {
            first_seq = batch_first;
        }
        count += parsed;
        last_seq = batch_last;
        offset += 4 + batch_len;
    }

    Ok(ScanResult {
        valid_end: offset,
        count,
        first_seq,
        last_seq,
        tail: WalTail::Clean,
    })
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
        Wal::open(
            path.path(),
            id,
            Epoch(epoch),
            WalOptions::new(fsync),
            TailRepair::DiscardSuspect,
        )
        .unwrap()
    }

    #[test]
    fn append_and_replay() {
        let (dir, id) = setup();
        let opts = WalOptions::new(false);
        let mut wal = Wal::open(
            dir.path(),
            &id,
            Epoch(1),
            opts.clone(),
            TailRepair::DiscardSuspect,
        )
        .unwrap();
        let ops = vec![add("a"), update("a"), delete("b"), upsert("c")];
        for op in &ops {
            wal.append(op.clone()).unwrap();
        }
        assert_eq!(wal.last_seq(), 4);
        assert_eq!(wal.len(), 4);
        drop(wal);

        let wal = Wal::open(dir.path(), &id, Epoch(1), opts, TailRepair::DiscardSuspect).unwrap();
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
            let mut wal = Wal::open(
                dir.path(),
                &id,
                Epoch(1),
                opts.clone(),
                TailRepair::DiscardSuspect,
            )
            .unwrap();
            wal.append(add("x")).unwrap();
            wal.append(update("x")).unwrap();
            drop(wal);

            let wal =
                Wal::open(dir.path(), &id, Epoch(1), opts, TailRepair::DiscardSuspect).unwrap();
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

        let mut wal = Wal::open(
            dir.path(),
            &id,
            Epoch(1),
            WalOptions::new(true),
            TailRepair::DiscardSuspect,
        )
        .unwrap();
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

    /// A batch is atomic: cutting into it leaves none of its operations.
    #[test]
    fn append_batch_is_atomic_on_disk() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        wal.append_batch(&[add("1"), add("2")]).unwrap();
        let after_first = wal.size_bytes();
        wal.append_batch(&[add("3"), add("4")]).unwrap();
        assert_eq!(wal.last_seq(), 4);
        drop(wal);

        // Cut into the second batch.
        let path = wal_path(&dir, &id);
        let len = std::fs::metadata(&path).unwrap().len();
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(len - 5).unwrap();
        drop(file);

        let wal = Wal::open(
            dir.path(),
            &id,
            Epoch(1),
            WalOptions::new(true),
            TailRepair::DiscardSuspect,
        )
        .unwrap();
        assert_eq!(wal.len(), 2, "only the complete batch survives");
        assert_eq!(wal.last_seq(), 2);
        assert_eq!(wal.size_bytes(), after_first);

        let recs = wal.read_range(1, 10).unwrap();
        assert_eq!(recs.len(), 2, "no partial batch may become visible");
        assert_eq!(recs[0].seq, 1);
        assert_eq!(recs[1].seq, 2);
    }

    /// A batch with no commit marker is provably incomplete.
    #[test]
    fn missing_commit_marker_is_suspect() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        wal.append_batch(&[add("1")]).unwrap();
        drop(wal);

        let path = wal_path(&dir, &id);
        let mut bytes = std::fs::read(&path).unwrap();
        let at = bytes.len() - 1;
        bytes[at] ^= 0xFF; // clobber the commit sentinel
        std::fs::write(&path, &bytes).unwrap();

        let inspection = Wal::inspect(dir.path(), &id).unwrap();
        assert!(
            matches!(
                inspection.tail,
                WalTail::Suspect {
                    reason: TailReason::TruncatedFrame,
                    ..
                }
            ),
            "expected a missing-marker tail, got {:?}",
            inspection.tail
        );
        assert_eq!(inspection.record_count, 0);
    }

    /// Pruning a batch-framed file re-encodes batches and keeps sequences.
    #[test]
    fn prune_preserves_batches_and_seqs() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        for i in 1..=5 {
            wal.append_batch(&[add(&i.to_string())]).unwrap();
        }
        wal.prune(2).unwrap();
        assert_eq!(wal.last_seq(), 5);
        assert_eq!(wal.len(), 3);

        let recs = wal.read_range(1, 10).unwrap();
        let seqs: Vec<u64> = recs.iter().map(|r| r.seq).collect();
        assert_eq!(seqs, vec![3, 4, 5], "retained records keep their seqs");

        // Still readable after a reopen.
        drop(wal);
        let inspection = Wal::inspect(dir.path(), &id).unwrap();
        assert_eq!(inspection.tail, WalTail::Clean);
        assert_eq!(inspection.first_seq, 3);
    }

    /// Inspecting must never create or modify the file.
    #[test]
    fn inspect_is_read_only() {
        let (dir, id) = setup();
        let path = wal_path(&dir, &id);

        let inspection = Wal::inspect(dir.path(), &id).unwrap();
        assert!(!inspection.exists);
        assert_eq!(inspection.record_count, 0);
        assert!(!path.exists(), "inspect must not create the log file");

        let mut wal = open_fresh(&dir, &id, 1, true);
        wal.append(add("1")).unwrap();
        wal.append(add("2")).unwrap();
        drop(wal);
        let before = std::fs::read(&path).unwrap();

        let inspection = Wal::inspect(dir.path(), &id).unwrap();
        assert!(inspection.exists);
        assert_eq!(inspection.record_count, 2);
        assert_eq!(inspection.first_seq, 1);
        assert_eq!(inspection.last_verified_seq, 2);
        assert_eq!(inspection.tail, WalTail::Clean);
        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "inspect must not modify the log"
        );
    }

    /// A frame shorter than it claims is an incomplete final write.
    #[test]
    fn truncated_tail_is_suspect() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        wal.append(add("1")).unwrap();
        wal.append(add("2")).unwrap();
        drop(wal);
        let path = wal_path(&dir, &id);

        let len = std::fs::metadata(&path).unwrap().len();
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(len - 3).unwrap();
        drop(file);

        let inspection = Wal::inspect(dir.path(), &id).unwrap();
        assert!(
            matches!(
                inspection.tail,
                WalTail::Suspect {
                    reason: TailReason::TruncatedFrame,
                    ..
                }
            ),
            "expected a truncated suspect tail, got {:?}",
            inspection.tail
        );
        assert_eq!(inspection.record_count, 1);
        assert_eq!(inspection.last_verified_seq, 1);

        // Discarding recovers the verified prefix.
        let wal = Wal::open(
            dir.path(),
            &id,
            Epoch(1),
            WalOptions::new(true),
            TailRepair::DiscardSuspect,
        )
        .unwrap();
        assert_eq!(wal.len(), 1);
        assert_eq!(wal.last_seq(), 1);
    }

    /// `Refuse` must leave a suspect tail in place.
    #[test]
    fn refuse_leaves_tail_untouched() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        wal.append(add("1")).unwrap();
        wal.append(add("2")).unwrap();
        drop(wal);
        let path = wal_path(&dir, &id);

        let len = std::fs::metadata(&path).unwrap().len();
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(len - 3).unwrap();
        drop(file);
        let shrunk = std::fs::metadata(&path).unwrap().len();

        assert!(
            Wal::open(
                dir.path(),
                &id,
                Epoch(1),
                WalOptions::new(true),
                TailRepair::Refuse
            )
            .is_err(),
            "Refuse must reject a suspect tail"
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            shrunk,
            "Refuse must not truncate"
        );
    }

    /// A checksum failure in the *final* frame is an ambiguous tail: it may be a
    /// partially durable write, so it is offered for discard but not proven
    /// incomplete.
    #[test]
    fn final_frame_checksum_failure_is_suspect() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        wal.append(add("1")).unwrap();
        let off1 = wal.size_bytes();
        wal.append(add("2")).unwrap();
        drop(wal);
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

        let inspection = Wal::inspect(dir.path(), &id).unwrap();
        assert!(
            matches!(
                inspection.tail,
                WalTail::Suspect {
                    reason: TailReason::ChecksumMismatch,
                    ..
                }
            ),
            "expected a checksum suspect tail, got {:?}",
            inspection.tail
        );
    }

    /// A corrupted `batch_len` in the middle of the log is a hole, not a tail.
    ///
    /// The length field is what tells the scanner where the next frame starts,
    /// so damaging it used to look exactly like a torn final batch: the frame
    /// "ended" at or past the end of the file and the rest of the log was
    /// discarded. An acknowledged middle frame is not a tail, however it is
    /// damaged, so the classifier now asks whether the sequence continues behind
    /// the damage.
    #[test]
    fn corrupted_batch_len_in_the_middle_is_a_hole() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        wal.append(add("1")).unwrap();
        let off1 = wal.size_bytes();
        wal.append(add("2")).unwrap();
        wal.append(add("3")).unwrap();
        wal.append(add("4")).unwrap();
        drop(wal);
        let path = wal_path(&dir, &id);
        let len_before = std::fs::metadata(&path).unwrap().len();

        // Make frame 2's length run past the end of the file.
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(off1)).unwrap();
        file.write_all(&0xFFFF_FFF0u32.to_le_bytes()).unwrap();
        drop(file);

        let inspection = Wal::inspect(dir.path(), &id).unwrap();
        assert!(
            matches!(inspection.tail, WalTail::Corrupt { .. }),
            "expected Corrupt, got {:?}",
            inspection.tail
        );
        assert_eq!(
            inspection.record_count, 1,
            "only the frame before the damage is provably intact"
        );

        // And a discard policy must not act on it.
        assert!(
            Wal::open(
                dir.path(),
                &id,
                Epoch(1),
                WalOptions::new(true),
                TailRepair::DiscardSuspect
            )
            .is_err(),
            "DiscardSuspect must refuse a hole"
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            len_before,
            "a hole must never be truncated"
        );
    }

    /// The same damage in the *final* frame is still a tolerable tail: nothing
    /// follows it, so no acknowledged record is being thrown away.
    #[test]
    fn corrupted_batch_len_at_the_end_is_suspect() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        wal.append(add("1")).unwrap();
        let off1 = wal.size_bytes();
        wal.append(add("2")).unwrap();
        drop(wal);
        let path = wal_path(&dir, &id);

        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(off1)).unwrap();
        file.write_all(&0xFFFF_FFF0u32.to_le_bytes()).unwrap();
        drop(file);

        let inspection = Wal::inspect(dir.path(), &id).unwrap();
        assert!(
            matches!(inspection.tail, WalTail::Suspect { .. }),
            "expected Suspect, got {:?}",
            inspection.tail
        );
    }

    /// Damage in the middle of the log is a hole, not a tail.
    ///
    /// This test previously asserted that a corrupt middle record caused the
    /// rest of the file to be discarded. That silently shortened the database:
    /// the acknowledged records after the damage were thrown away and the
    /// truncation was reported as a successful open.
    #[test]
    fn corrupt_middle_is_refused_and_leaves_the_file_intact() {
        let (dir, id) = setup();
        let mut wal = open_fresh(&dir, &id, 1, true);
        wal.append(add("1")).unwrap();
        let off1 = wal.size_bytes();
        wal.append(add("2")).unwrap();
        let off2 = wal.size_bytes();
        wal.append(add("3")).unwrap();
        wal.append(add("4")).unwrap();
        wal.append(add("5")).unwrap();
        drop(wal);
        assert!(off2 > off1 + 24);

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
        let len_before = std::fs::metadata(&path).unwrap().len();

        // Inspection classifies it as a hole even though a discard policy is
        // offered.
        let inspection = Wal::inspect(dir.path(), &id).unwrap();
        assert!(
            matches!(inspection.tail, WalTail::Corrupt { .. }),
            "expected Corrupt, got {:?}",
            inspection.tail
        );

        let err = match Wal::open(
            dir.path(),
            &id,
            Epoch(1),
            WalOptions::new(true),
            TailRepair::DiscardSuspect,
        ) {
            Ok(_) => panic!("corrupted history must not open"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("corrupted history"),
            "unexpected error: {err}"
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            len_before,
            "a corrupted file must not be truncated"
        );
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

        let result = Wal::open(
            dir.path(),
            &id,
            Epoch(1),
            WalOptions::new(true),
            TailRepair::DiscardSuspect,
        );
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

        // Append a batch stamped with an older epoch than the file's, which is
        // what a fenced writer would leave behind.
        let path = wal_path(&dir, &id);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::End(0)).unwrap();
        let payload = bincode::serialize(&delete("fenced")).unwrap();
        let frame = encode_batch(&[(3u64, 0u64, payload)], 3).unwrap();
        file.write_all(&frame).unwrap();
        file.sync_all().unwrap();
        drop(file);

        // Reopen at the file's epoch: the fenced record is a corrupt tail.
        let mut wal = Wal::open(
            dir.path(),
            &id,
            Epoch(5),
            WalOptions::new(true),
            TailRepair::DiscardSuspect,
        )
        .unwrap();
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

        let wal = Wal::open(
            dir.path(),
            &id,
            Epoch(1),
            WalOptions::new(true),
            TailRepair::DiscardSuspect,
        )
        .unwrap();
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

        let wal = Wal::open(
            dir.path(),
            &id,
            Epoch(1),
            WalOptions::new(true),
            TailRepair::DiscardSuspect,
        )
        .unwrap();
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
        let mut wal = Wal::open(
            dir.path(),
            &id,
            Epoch(1),
            opts.clone(),
            TailRepair::DiscardSuspect,
        )
        .unwrap();
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
        let mut wal =
            Wal::open(dir.path(), &id, Epoch(1), opts, TailRepair::DiscardSuspect).unwrap();
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
        let mut wal =
            Wal::open(dir.path(), &id, Epoch(1), opts, TailRepair::DiscardSuspect).unwrap();
        assert_eq!(wal.last_seq(), 3);
        let r = wal.append(add("after")).unwrap();
        assert_eq!(r.seq, 4);
    }
}
