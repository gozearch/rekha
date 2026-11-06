//! Per-collection state and the two-tier write pipeline.
//!
//! A [`Collection`] is the unit of concurrency: the engine holds one
//! `Arc<Mutex<Collection>>` per collection, and a single mutex serializes WAL
//! appends (the WAL is single-writer) and index access (usearch's search
//! mutates the shared C++ expansion state).

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rekha_core::cluster::Epoch;
use rekha_core::config::CollectionConfig;
use rekha_core::filter::{WhereDocument, WhereFilter};
use rekha_core::op::Operation;
use rekha_core::types::{Distance, Document, Embedding, Id, Metadata};
use rekha_distance::distance;
use rekha_index::{Index, UsearchIndex};
use rekha_storage::Catalog;
use rekha_wal::{TailRepair, Wal, WalInspection, WalOptions, WalTail};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::fault;
use crate::postings::Postings;
use crate::{
    CheckpointOutcome, EngineError, EngineResult, OptimizeStats, QueryOptions, Record, ScoredPoint,
};

/// In-memory state of one collection: the live records map, the brute-force
/// buffer, the committed HNSW index, the metadata postings, and the WAL they
/// are all derived from.
pub(crate) struct Collection {
    pub(crate) config: CollectionConfig,
    pub(crate) shard: Option<rekha_core::cluster::ShardId>,
    pub(crate) wal: Wal,
    /// All live records (metadata + document + current embedding).
    pub(crate) records: HashMap<Id, Record>,
    /// Uncommitted embedding-bearing records waiting to enter the HNSW index.
    /// Kept unique per id (the newest version of each unflushed id).
    pub(crate) buffer: Vec<Record>,
    /// Committed vectors (the HNSW index).
    pub(crate) index: Box<dyn Index>,
    /// Highest WAL seq materialized into the index (Chroma's `max_seq_id`
    /// analog; checkpoints push it to `catalog.advance_log_offset`).
    pub(crate) flushed_seq: u64,
    /// Highest sequence applied to this collection, from either write path.
    ///
    /// Single-node: the WAL sequence. Cluster: the Raft log index, which is
    /// what a record's `seq` is there — and the local WAL is never written, so
    /// it cannot be used as the checkpoint boundary the way single-node does.
    /// Without this, a clustered checkpoint claimed to cover sequence 0 while
    /// holding records with seq 10, and validation rejected the generation with
    /// "record has seq past the snapshot boundary".
    pub(crate) applied_seq: u64,
    /// Internal offset per id, assigned monotonically by `next_offset` and
    /// never reused after delete (Chroma's offset scheme). The metadata
    /// postings reference these offsets, so they are stable across a record's
    /// lifetime.
    pub(crate) offsets: HashMap<Id, u32>,
    /// Reverse of `offsets`, for mapping a postings bitmap back to ids.
    pub(crate) id_by_offset: HashMap<u32, Id>,
    /// Next internal offset to assign; never rolled back on delete.
    next_offset: u32,
    /// Inverted index over record metadata (Chroma's metadata segment).
    pub(crate) postings: Postings,
    /// Directory holding this collection's checkpoint files
    /// (`{wal_dir}/{id}.checkpoint/`).
    pub(crate) checkpoint_dir: PathBuf,
    /// The WAL directory this collection lives in.
    ///
    /// Stored rather than derived from `checkpoint_dir.parent()`. Deriving it
    /// made a layout assumption (`checkpoint_dir` is always exactly one level
    /// below the WAL directory) load-bearing, and the failure mode was a panic
    /// in the middle of a checkpoint.
    pub(crate) wal_dir: PathBuf,
    /// WAL seq covered by the last checkpoint (used by the auto-checkpoint
    /// trigger to bound WAL growth: checkpoint when the WAL has grown
    /// `sync_threshold` records past this watermark).
    last_checkpoint_seq: u64,
    /// Entries applied since the last checkpoint. Cluster mode has no local WAL
    /// to measure growth against, so this is the checkpoint trigger there.
    pub(crate) applied_since_checkpoint: u64,
    /// Whether to fsync WAL writes (passed through to WAL reopen after checkpoint prune).
    fsync: bool,
    /// Set when a checkpoint was renamed into place but the publication could
    /// not be confirmed (e.g. the directory sync failed). The active generation
    /// is then unknown for this process, so mutations are refused until a
    /// reopen re-establishes it. Clearing it optimistically would risk writing
    /// on top of a generation nobody can name.
    publication_uncertain: bool,
    /// Where the most recent checkpoint spent its time. Kept so `optimize` can
    /// report a breakdown without threading an out-parameter through recovery.
    pub(crate) last_checkpoint_timings: crate::CheckpointTimings,
    /// Mmap'd vector segment handles, in offset order.
    ///
    /// The vector space is append-only across checkpoints: a new generation
    /// hard-links the previous segments and appends one segment holding only
    /// the vectors written since. `segment_starts[i]` is segment `i`'s base
    /// offset in that logical space.
    pub(crate) segments: Vec<std::sync::Arc<crate::segment::Segment>>,
    /// Base offset of each segment: `segment_starts[i]` + local index = global
    /// offset stored in `segment_index`.
    pub(crate) segment_starts: Vec<usize>,
    /// Maps record id → offset within the segment (for fast vector lookup).
    pub(crate) segment_index: HashMap<Id, usize>,
}

/// The only checkpoint manifest format this build reads and writes.
///
/// It carries the file inventory, the owning collection, the vector
/// configuration, and the explicit `snapshot_seq` boundary. There is no older
/// format to fall back to: a manifest that does not match is rejected rather
/// than guessed at, because a future format may assign new meaning to an
/// existing field.
pub(crate) const CHECKPOINT_FORMAT_VERSION: u32 = 3;

/// The JSON manifest (`checkpoint.json`) inside a generation directory.
///
/// It is written **last within its generation**, so its presence means every
/// file beside it belongs to the same snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CheckpointMeta {
    /// Manifest format. Must equal [`CHECKPOINT_FORMAT_VERSION`].
    pub(crate) format_version: u32,
    /// Highest WAL seq materialized into the **index** when the checkpoint ran.
    ///
    /// This is an indexing watermark, not the snapshot boundary: `records.bin`
    /// contains every live record regardless of whether its vector had reached
    /// the ANN graph. Recovery uses it only to know which records need a buffer
    /// rebuild or an index rebuild.
    pub(crate) flushed_seq: u64,
    /// WAL seq this snapshot **covers** (the checkpoint contains the effect of
    /// every operation up to and including it).
    ///
    /// This is the replay boundary: recovery resumes after it. It is distinct
    /// from `flushed_seq`, which only says how far the index had caught up.
    pub(crate) snapshot_seq: u64,
    /// Seq the WAL will resume at: `wal.last_seq() + 1` at checkpoint time.
    /// Reopen passes this as `WalOptions::start_seq`, so a fully-pruned
    /// (empty) WAL continues appending exactly where the checkpoint left off.
    next_seq: u64,
    /// `records.len()` at checkpoint time, validated on load.
    pub(crate) record_count: u64,
    /// Collection this generation belongs to. A generation directory copied or
    /// left over from another collection must not be activated for this one.
    pub(crate) collection_id: String,
    /// Vector width the vectors in this generation were written for.
    pub(crate) dimension: usize,
    /// Distance space the index was built for, as [`Distance::name`].
    pub(crate) space: String,
    /// Inventory of the authoritative files in this generation.
    pub(crate) files: Vec<CheckpointFile>,
}

/// One file listed in a generation's manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CheckpointFile {
    /// Path relative to the generation directory.
    pub(crate) path: String,
    pub(crate) len: u64,
    /// CRC-32 of the contents. `None` for files that carry their own integrity
    /// check (vector segments verify their data block on open) or that are
    /// derived and rebuildable (the ANN index).
    #[serde(default)]
    pub(crate) crc32: Option<u32>,
}

/// A fully validated generation, read from disk but not yet applied.
///
/// Recovery validates the *whole* generation into this value before touching
/// live collection state. The earlier implementation assigned `coll.records`
/// and then validated segments, so a failure part-way through left a
/// half-restored collection that a fallback replay would then build on top of.
pub(crate) struct LoadedGeneration {
    meta: CheckpointMeta,
    records: HashMap<Id, Record>,
    index: Option<UsearchIndex>,
    segments: Vec<std::sync::Arc<crate::segment::Segment>>,
    segment_index: HashMap<Id, usize>,
    /// Persisted insertion order. Rebuilding this from `HashMap` iteration
    /// changed `get`/pagination order across a restart.
    offsets: HashMap<Id, u32>,
    next_offset: u32,
}

/// Ranking and projection parameters shared by the query paths.
///
/// Grouped into a struct so the planner functions stay readable and within the
/// argument-count lint.
struct SearchParams {
    k: usize,
    ef: usize,
    oversampling: usize,
    with_embedding: bool,
}

/// The persisted id → offset mapping and its counter.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedOrder {
    offsets: HashMap<Id, u32>,
    next_offset: u32,
}

/// Outcome of looking for a committed checkpoint on disk.
pub(crate) enum CheckpointLoad {
    /// A generation was read and validated; the caller applies it. Boxed to
    /// keep the error variants small (clippy `large_enum_variant`).
    Loaded(Box<LoadedGeneration>),
    /// No checkpoint has ever been published; the WAL holds the full history.
    Absent,
    /// A checkpoint was published but cannot be trusted. Recoverable only if
    /// the WAL prefix is still intact.
    Corrupt(String),
}

/// Link `src` into a new generation directory.
///
/// A hard link is O(1) and keeps each generation self-contained. Copying is a
/// correctness-preserving fallback for filesystems that reject links, but it
/// reintroduces the whole-corpus cost this design removes, so it is warned
/// about rather than silently accepted.
fn link_or_copy(src: &Path, dst: &Path) -> std::io::Result<()> {
    match std::fs::hard_link(src, dst) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => {
            tracing::warn!(
                src = %src.display(),
                "hard link failed ({e}); copying the segment instead, which restores \
                 whole-corpus checkpoint cost"
            );
            std::fs::copy(src, dst).map(|_| ())
        }
    }
}

/// Segment fan-out at which a routine checkpoint compacts.
///
/// Each incremental checkpoint appends one segment, and every checkpoint
/// re-opens and checksum-verifies the whole set. Left unbounded, a long ingest
/// turns that into whole-corpus work again. Compacting at a small bound keeps
/// the per-checkpoint verification proportional to a fixed number of segments
/// and amortises the full rewrite across several checkpoints.
const MAX_SEGMENTS: usize = 8;

/// Temp-file path for an atomic write-then-rename (same `{path}.tmp` scheme
/// the WAL prune uses).
fn tmp_path_for(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".tmp");
    PathBuf::from(os)
}

/// Atomically replace `path` with `bytes`: write a sibling temp file, fsync it,
/// then rename over the target. A crash mid-write leaves the previous (complete)
/// file in place.
fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = tmp_path_for(path);
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp, path)?;
    if let Some(parent) = path.parent() {
        sync_dir(parent)?;
    }
    Ok(())
}

/// Fsync a directory so a rename/creation inside it is durable.
fn sync_dir(dir: &Path) -> std::io::Result<()> {
    let handle = std::fs::File::open(dir)?;
    handle.sync_all()
}

/// Name of the pointer file naming the active checkpoint generation.
pub(crate) const CURRENT_POINTER: &str = "CURRENT";
/// Subdirectory holding immutable checkpoint generations.
pub(crate) const GENERATIONS_DIR: &str = "generations";

/// Check that a persisted order is a usable, unambiguous offset assignment.
fn validate_order(order: &PersistedOrder, records: &HashMap<Id, Record>) -> Result<(), String> {
    if order.offsets.len() != records.len() {
        return Err(format!(
            "{} offsets for {} live records",
            order.offsets.len(),
            records.len()
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(order.offsets.len());
    let mut max_offset = 0u32;
    for (id, offset) in &order.offsets {
        if !records.contains_key(id) {
            return Err(format!("offset recorded for unknown record `{id}`"));
        }
        if !seen.insert(*offset) {
            return Err(format!(
                "offset {offset} is assigned to more than one record"
            ));
        }
        max_offset = max_offset.max(*offset);
    }
    // `next_offset` must be strictly greater than every issued offset, or a
    // later insert would collide with a live record.
    if !order.offsets.is_empty() && order.next_offset <= max_offset {
        return Err(format!(
            "next_offset {} does not exceed the highest issued offset {max_offset}",
            order.next_offset
        ));
    }
    Ok(())
}

/// Verify every file a manifest lists, before any of them is trusted.
///
/// Lengths are always checked. A CRC is checked only where the manifest records
/// one: segments carry their own data checksum (verified by `Segment::open`) and
/// the ANN index is derived, so re-reading them here would double the startup
/// I/O without adding a guarantee.
fn verify_inventory(dir: &Path, meta: &CheckpointMeta) -> Result<(), String> {
    let mut seen = std::collections::HashSet::with_capacity(meta.files.len());
    for entry in &meta.files {
        if !seen.insert(entry.path.as_str()) {
            return Err(format!("manifest lists `{}` more than once", entry.path));
        }
        if entry.path.is_empty()
            || entry.path.starts_with('/')
            || entry.path.contains("..")
            || entry.path.contains('\\')
        {
            return Err(format!("manifest lists an unsafe path `{}`", entry.path));
        }
        let path = dir.join(&entry.path);
        match entry.crc32 {
            Some(expected) => {
                let bytes = std::fs::read(&path)
                    .map_err(|e| format!("`{}` is unreadable: {e}", entry.path))?;
                if bytes.len() as u64 != entry.len {
                    return Err(format!(
                        "`{}` is {} bytes but the manifest declares {}",
                        entry.path,
                        bytes.len(),
                        entry.len
                    ));
                }
                let actual = crc32fast::hash(&bytes);
                if actual != expected {
                    return Err(format!(
                        "`{}` checksum mismatch: expected {expected:#010x}, got {actual:#010x}",
                        entry.path
                    ));
                }
            }
            None => {
                let metadata = std::fs::metadata(&path)
                    .map_err(|e| format!("`{}` is unreadable: {e}", entry.path))?;
                if metadata.len() != entry.len {
                    return Err(format!(
                        "`{}` is {} bytes but the manifest declares {}",
                        entry.path,
                        metadata.len(),
                        entry.len
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Resolve the active checkpoint generation directory.
///
/// Returns `None` when there is no valid `CURRENT` pointer. The caller decides
/// what that means: a base with no checkpoint files at all has never been
/// checkpointed, while checkpoint files without a pointer are the
/// pre-generation layout and are rejected. A `CURRENT` value is rejected if it
/// is empty or tries to escape the generations directory.
pub(crate) fn active_generation(base: &Path) -> Option<PathBuf> {
    let raw = std::fs::read_to_string(base.join(CURRENT_POINTER)).ok()?;
    let name = raw.trim();
    if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains("..") {
        return None;
    }
    let dir = base.join(GENERATIONS_DIR).join(name);
    dir.is_dir().then_some(dir)
}

impl Collection {
    /// Open a collection, preferring its published checkpoint generation and
    /// falling back to a full WAL replay.
    ///
    /// `durable_seq` is the catalog's durable watermark (`max_seq_id`) for this
    /// collection. It is the only record that survives deletion of the
    /// checkpoint directory, so it is what lets recovery tell "this collection
    /// has never been checkpointed" apart from "this collection *was*
    /// checkpointed and its generation is now missing".
    pub(crate) fn open(
        id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        config: CollectionConfig,
        wal_dir: &Path,
        epoch: Epoch,
        fsync: bool,
        durable_seq: u64,
    ) -> EngineResult<Self> {
        let stem = match shard {
            Some(s) => format!("{id}.shard-{}", s.0),
            None => id.to_string(),
        };
        let checkpoint_dir = wal_dir.join(format!("{stem}.checkpoint"));

        // Inspect before anything is opened for writing. Opening the log used to
        // repair (truncate) it before the checkpoint was even read, so the
        // decision to discard records was taken without knowing whether the
        // checkpoint already covered them.
        let inspection = Wal::inspect_named(wal_dir, &stem)?;
        let load = Self::read_generation(&checkpoint_dir, &config);

        // Decide, from all the evidence together, how the log may be treated and
        // where replay must resume.
        let (repair, replay_after) = Self::recovery_plan(id, durable_seq, &inspection, &load)?;

        let wal = Wal::open_named(
            wal_dir,
            &stem,
            epoch,
            WalOptions {
                fsync,
                start_seq: None,
            },
            repair,
        )?;
        let mut coll = Self::with_wal(config, shard, wal_dir, fsync, wal)?;

        match load {
            CheckpointLoad::Loaded(generation) => {
                coll.apply_generation(*generation)?;
                coll.replay_after(replay_after)?;
            }
            CheckpointLoad::Absent => coll.replay_after(0)?,
            CheckpointLoad::Corrupt(reason) => {
                tracing::warn!(
                    collection = %id,
                    "checkpoint unreadable ({reason}); recovering from the full WAL"
                );
                coll.replay_after(0)?;
            }
        }
        // Re-apply the flush policy so a reopened collection behaves like the
        // pre-crash one (its buffer is drained up to batch_size).
        coll.flush_if_needed()?;
        Ok(coll)
    }

    /// Fresh in-memory state around an already-opened WAL and an empty HNSW
    /// index.
    ///
    /// The WAL is opened by the caller so exactly one `Wal::open` happens per
    /// recovery, with a repair policy chosen after inspecting both the log and
    /// the checkpoint. A second open would risk re-deciding on a file the first
    /// one had already modified.
    fn with_wal(
        config: CollectionConfig,
        shard: Option<rekha_core::cluster::ShardId>,
        wal_dir: &Path,
        fsync: bool,
        wal: Wal,
    ) -> EngineResult<Self> {
        let stem = match shard {
            Some(s) => format!("{}.shard-{}", config.id, s.0),
            None => config.id.to_string(),
        };
        let checkpoint_dir = wal_dir.join(format!("{stem}.checkpoint"));
        let index = UsearchIndex::with_hnsw(config.space, config.dimension, &config.hnsw)?;
        Ok(Self {
            config,
            shard,
            wal,
            records: HashMap::new(),
            buffer: Vec::new(),
            index: Box::new(index),
            flushed_seq: 0,
            applied_seq: 0,
            offsets: HashMap::new(),
            id_by_offset: HashMap::new(),
            next_offset: 0,
            postings: Postings::new(),
            checkpoint_dir,
            wal_dir: wal_dir.to_path_buf(),
            last_checkpoint_seq: 0,
            applied_since_checkpoint: 0,
            fsync,
            publication_uncertain: false,
            last_checkpoint_timings: crate::CheckpointTimings::default(),
            segments: Vec::new(),
            segment_starts: Vec::new(),
            segment_index: HashMap::new(),
        })
    }

    /// Choose the WAL repair policy and the replay start from every available
    /// signal: the catalog's durable watermark, the log inspection, and the
    /// checkpoint.
    ///
    /// The previous logic opened (and truncated) the log first and used
    /// `first_seq == 1` as its only completeness test. Both are unsafe: a first
    /// sequence of 1 does not prove the history is complete, and truncating an
    /// unparsed record can shorten an acknowledged collection.
    fn recovery_plan(
        id: &Uuid,
        durable_seq: u64,
        inspection: &WalInspection,
        load: &CheckpointLoad,
    ) -> EngineResult<(TailRepair, u64)> {
        let has_log = inspection.record_count > 0 || !matches!(inspection.tail, WalTail::Clean);

        // A corrupt interior is always fatal: the file has a hole in it, so no
        // amount of checkpoint coverage makes the remaining bytes trustworthy.
        if let WalTail::Corrupt { reason, at_offset } = inspection.tail {
            return Err(EngineError::Recovery(format!(
                "collection {id}: WAL is corrupted ({}) at byte offset {at_offset}; this is \
                 not an end-of-file tail and will not be truncated. Restore from backup.",
                reason.describe()
            )));
        }

        // A suspect tail may be discarded: it is the last thing in the file.
        let repair = if matches!(inspection.tail, WalTail::Clean) {
            TailRepair::Refuse
        } else {
            TailRepair::DiscardSuspect
        };

        match load {
            CheckpointLoad::Loaded(generation) => {
                let boundary = generation.meta.snapshot_seq;
                if inspection.record_count > 0 {
                    // Replay must start where the snapshot ends. A log that
                    // begins after the boundary has lost the operations in
                    // between.
                    if inspection.first_seq > boundary + 1 {
                        return Err(EngineError::Recovery(format!(
                            "collection {id}: the checkpoint covers sequence {boundary} but the \
                             WAL resumes at {}; sequences {}..={} are missing. Restore from \
                             backup.",
                            inspection.first_seq,
                            boundary + 1,
                            inspection.first_seq - 1
                        )));
                    }
                }
                Ok((repair, boundary))
            }
            CheckpointLoad::Absent => {
                // Treating "no checkpoint" as "fresh" is only safe when nothing
                // was ever checkpointed and the log still starts at 1.
                if durable_seq > 0 {
                    return Err(EngineError::Recovery(format!(
                        "collection {id}: the catalog records durable progress (max_seq_id \
                         {durable_seq}) but no checkpoint generation is readable. Refusing to \
                         start with a partial collection; restore from backup."
                    )));
                }
                // Only meaningful when something verified survived. A log whose
                // first batch is unreadable leaves zero records, which is
                // indistinguishable from a brand-new collection — and with
                // `durable_seq == 0` there is no other evidence of history.
                if inspection.record_count > 0 && inspection.first_seq != 1 {
                    return Err(EngineError::Recovery(format!(
                        "collection {id}: no checkpoint exists and the WAL starts at sequence \
                         {}, so the earlier history is missing. Restore from backup.",
                        inspection.first_seq
                    )));
                }
                Ok((repair, 0))
            }
            CheckpointLoad::Corrupt(reason) => {
                // Only an intact full history can stand in for a lost
                // checkpoint; once the prefix was pruned it cannot.
                if has_log && inspection.first_seq == 1 {
                    Ok((repair, 0))
                } else {
                    Err(EngineError::Recovery(format!(
                        "collection {id}: checkpoint is unreadable ({reason}) and the WAL does \
                         not hold the full history (starts at sequence {}); refusing to recover \
                         a partial collection. Restore from backup.",
                        inspection.first_seq
                    )))
                }
            }
        }
    }

    /// Read and fully validate the published checkpoint generation.
    ///
    /// Nothing in `self` is modified: the caller applies the result only after
    /// validation succeeds, so a failed or partial read cannot leave live state
    /// half-restored.
    pub(crate) fn read_generation(base: &Path, config: &CollectionConfig) -> CheckpointLoad {
        let expected_id = &config.id;
        let dir = match active_generation(base) {
            Some(generation) => generation,
            None if base.join(CURRENT_POINTER).exists() => {
                return CheckpointLoad::Corrupt(format!(
                    "`{CURRENT_POINTER}` does not name a readable generation"
                ));
            }
            // Checkpoint files at the base, with no `CURRENT` naming a
            // generation, are the pre-generation layout. Treating them as
            // absent would silently discard the checkpoint and recover from
            // whatever the WAL still holds, which is a smaller collection than
            // the one that was acknowledged.
            None if base.join("checkpoint.json").exists() => {
                return CheckpointLoad::Corrupt(
                    "checkpoint files are in the pre-generation layout, which this build no \
                     longer reads; restore from backup or start on an empty data directory"
                        .into(),
                );
            }
            None => return CheckpointLoad::Absent,
        };
        let dir = dir.as_path();

        let meta: CheckpointMeta = match std::fs::read(dir.join("checkpoint.json")) {
            Ok(bytes) => match serde_json::from_slice(&bytes) {
                Ok(m) => m,
                Err(e) => return CheckpointLoad::Corrupt(format!("checkpoint.json: {e}")),
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return CheckpointLoad::Corrupt("checkpoint.json is missing".into());
            }
            Err(e) => return CheckpointLoad::Corrupt(format!("checkpoint.json: {e}")),
        };

        if meta.format_version != CHECKPOINT_FORMAT_VERSION {
            return CheckpointLoad::Corrupt(format!(
                "unsupported checkpoint format version {} (this build writes \
                 {CHECKPOINT_FORMAT_VERSION})",
                meta.format_version
            ));
        }

        if meta.collection_id != expected_id.to_string() {
            return CheckpointLoad::Corrupt(format!(
                "generation belongs to collection {}, not {expected_id}",
                meta.collection_id
            ));
        }

        // Bind the generation to the collection configuration it was written
        // for. A metric change under a checkpoint makes the persisted vectors
        // and graph meaningless, so it must not be silently reinterpreted.
        if config.dimension != 0 && meta.dimension != config.dimension {
            return CheckpointLoad::Corrupt(format!(
                "generation holds {}-dimensional vectors but the collection is \
                 {}-dimensional",
                meta.dimension, config.dimension
            ));
        }
        if meta.space != config.space.name() {
            let space = &meta.space;
            return CheckpointLoad::Corrupt(format!(
                "generation was written for space `{space}` but the collection uses `{}`",
                config.space.name()
            ));
        }

        // The manifest must inventory its authoritative files. An empty list is
        // not a pass: it would silently skip every check below.
        if meta.files.is_empty() {
            return CheckpointLoad::Corrupt("manifest declares no file inventory".into());
        }
        if !meta.files.iter().any(|f| f.path == "records.bin") {
            return CheckpointLoad::Corrupt("manifest does not inventory records.bin".into());
        }

        // Verify the inventory before trusting any file.
        if let Err(reason) = verify_inventory(dir, &meta) {
            return CheckpointLoad::Corrupt(reason);
        }

        let records_bytes = match std::fs::read(dir.join("records.bin")) {
            Ok(bytes) => bytes,
            Err(e) => return CheckpointLoad::Corrupt(format!("records.bin: {e}")),
        };
        let records: HashMap<Id, Record> = match bincode::deserialize(&records_bytes) {
            Ok(r) => r,
            Err(e) => return CheckpointLoad::Corrupt(format!("records.bin: {e}")),
        };
        if records.len() as u64 != meta.record_count {
            return CheckpointLoad::Corrupt(format!(
                "records.bin holds {} records but the manifest declares {}",
                records.len(),
                meta.record_count
            ));
        }

        // Derived: a missing or corrupt index is degraded, not fatal. It is
        // rebuilt from the vectors on the next flush/optimize, and queries fall
        // back to the exact path meanwhile.
        let index = Self::load_index(dir, config);

        let mut segments = Vec::new();
        let seg_dir = dir.join("segments");
        if seg_dir.is_dir() {
            let mut seg_idx = 0u32;
            loop {
                let seg_path = seg_dir.join(format!("seg-{seg_idx}.bin"));
                if !seg_path.exists() {
                    break;
                }
                match crate::segment::Segment::open(&seg_path) {
                    Ok(seg) if seg.is_empty() => {
                        return CheckpointLoad::Corrupt(format!(
                            "segment `{}` holds no vectors",
                            seg_path.display()
                        ));
                    }
                    Ok(seg) => segments.push(std::sync::Arc::new(seg)),
                    Err(e) => {
                        return CheckpointLoad::Corrupt(format!(
                            "segment `{}`: {e}",
                            seg_path.display()
                        ));
                    }
                }
                seg_idx += 1;
            }
        }

        let seg_index_path = dir.join("segment_index.bin");
        let mut segment_index: HashMap<Id, usize> = HashMap::new();
        if segments.is_empty() {
            if seg_index_path.exists() {
                return CheckpointLoad::Corrupt(
                    "segment_index.bin exists but no segment files were found".into(),
                );
            }
        } else {
            match std::fs::read(&seg_index_path)
                .ok()
                .and_then(|b| bincode::deserialize::<HashMap<Id, usize>>(&b).ok())
            {
                Some(idx) => segment_index = idx,
                None => {
                    return CheckpointLoad::Corrupt(
                        "segment_index.bin is missing or unreadable".into(),
                    );
                }
            }
        }

        // Insertion order, persisted so `get`/pagination order is stable across
        // a restart. It is authoritative: a generation is always written with
        // it, so its absence means the generation is incomplete, not that an
        // older format should be reconstructed into a different order.
        let order_bytes = match std::fs::read(dir.join("order.bin")) {
            Ok(bytes) => bytes,
            Err(e) => return CheckpointLoad::Corrupt(format!("order.bin: {e}")),
        };
        let order: PersistedOrder = match bincode::deserialize(&order_bytes) {
            Ok(order) => order,
            Err(e) => return CheckpointLoad::Corrupt(format!("order.bin: {e}")),
        };
        if let Err(reason) = validate_order(&order, &records) {
            return CheckpointLoad::Corrupt(format!("order.bin: {reason}"));
        }
        let (offsets, next_offset) = (order.offsets, order.next_offset);

        // Record sequence bounds. A record carrying a seq beyond the snapshot
        // boundary means the manifest and the record set disagree about what
        // this generation contains.
        let boundary = meta.snapshot_seq;
        for (id, record) in &records {
            if record.id != *id {
                return CheckpointLoad::Corrupt(format!(
                    "records.bin key `{id}` holds a record with id `{}`",
                    record.id
                ));
            }
            if record.seq > boundary {
                return CheckpointLoad::Corrupt(format!(
                    "record `{id}` has seq {} beyond the snapshot boundary {boundary}",
                    record.seq
                ));
            }
        }

        // Vector resolution. Every record the checkpoint stripped must resolve
        // to an in-bounds vector of the collection's width. Checking only that
        // the id appears in the mapping accepts an offset that points past the
        // segment, which surfaces later as a record that exists but can never
        // be matched.
        let expected_dim = meta.dimension;
        let starts = Self::segment_starts(&segments);
        for (id, record) in &records {
            let stripped = record.embedding.is_none() && record.seq <= meta.flushed_seq;
            match segment_index.get(id) {
                Some(&offset) => {
                    let resolved = Self::resolve_vector(&segments, &starts, offset);
                    match resolved {
                        Some(vector) => {
                            if expected_dim != 0 && vector.len() != expected_dim {
                                return CheckpointLoad::Corrupt(format!(
                                    "record `{id}` resolves to a {}-dimensional vector but the \
                                     collection is {expected_dim}-dimensional",
                                    vector.len()
                                ));
                            }
                        }
                        None => {
                            return CheckpointLoad::Corrupt(format!(
                                "record `{id}` maps to vector offset {offset}, which is out of \
                                 bounds for {} segment(s)",
                                segments.len()
                            ));
                        }
                    }
                }
                None if stripped => {
                    return CheckpointLoad::Corrupt(format!(
                        "record `{id}` is covered by the checkpoint but has no vector in the \
                         generation's segment"
                    ));
                }
                None => {}
            }
        }

        // The mapping must not reference ids that are no longer live: a stale
        // entry would resolve a query hit to a deleted record.
        for id in segment_index.keys() {
            if !records.contains_key(id) {
                return CheckpointLoad::Corrupt(format!(
                    "segment_index.bin references unknown record `{id}`"
                ));
            }
        }

        CheckpointLoad::Loaded(Box::new(LoadedGeneration {
            meta,
            records,
            index,
            segments,
            segment_index,
            offsets,
            next_offset,
        }))
    }

    /// Load the persisted ANN index for **this collection's** configuration.
    ///
    /// Validating the index against its own persisted metadata (the previous
    /// `load_any`) only proves it is internally consistent. If the collection's
    /// space or dimension changed, a self-consistent graph would still be
    /// loaded and every distance computed from it would be wrong. A mismatch
    /// returns `None`, which triggers a rebuild from the authoritative vectors.
    fn load_index(dir: &Path, config: &CollectionConfig) -> Option<UsearchIndex> {
        UsearchIndex::load(&dir.join("index.bin"), config.space, config.dimension).ok()
    }

    /// Install a validated generation into this collection and resume its WAL.
    fn apply_generation(&mut self, generation: LoadedGeneration) -> EngineResult<()> {
        let LoadedGeneration {
            meta,
            records,
            index,
            segments,
            segment_index,
            offsets,
            next_offset,
        } = generation;

        self.records = records;
        // The checkpoint is the authority on ordering. Rebuilding offsets by
        // iterating `records` (a HashMap) produced a different order on every
        // start, so pagination was not stable across a restart.
        self.offsets = offsets;
        self.id_by_offset = self
            .offsets
            .iter()
            .map(|(id, offset)| (*offset, id.clone()))
            .collect();
        self.postings = Postings::new();
        for (id, record) in self.records.iter() {
            if let Some(offset) = self.offsets.get(id) {
                self.postings.insert(*offset, record.metadata.as_ref());
            }
        }
        self.next_offset = next_offset;
        self.flushed_seq = meta.flushed_seq;
        self.last_checkpoint_seq = meta.flushed_seq;
        // The generation covers every sequence at or below `snapshot_seq`, so
        // that is where this collection's applied watermark resumes. A cluster
        // node cannot recover it from the WAL, which is why it is taken from
        // the manifest.
        self.applied_seq = meta.snapshot_seq;
        if let Some(index) = index {
            self.index = Box::new(index);
        }
        self.segment_starts = Self::segment_starts(&segments);
        self.segments = segments;
        self.segment_index = segment_index;

        // Vectors now live in the segment; drop the inline copies.
        if !self.segments.is_empty() {
            for record in self.records.values_mut() {
                if record.seq <= self.flushed_seq {
                    record.embedding = None;
                }
            }
        }

        // Reconcile the derived structures with the restored records.
        //
        // A v3 checkpoint does not replay from the index watermark, so records
        // that were never flushed into the ANN graph must be re-queued here
        // instead of being rediscovered by a replay. And if the persisted graph
        // could not be used (missing, or written for a different space or
        // width), it must be rebuilt now: leaving it empty would make the ANN
        // path return nothing for a collection large enough to use it.
        // "Has a vector" must resolve through the segment too: the strip above
        // removed the inline copies for every covered record, so testing
        // `record.embedding.is_some()` would miss exactly the records that need
        // rebuilding.
        let any_vector = self
            .records
            .keys()
            .any(|id| self.embedding_for(id).is_some());
        if self.index.is_empty() && any_vector {
            tracing::warn!(
                collection = %self.config.id,
                "rebuilding the ANN index from records; the persisted graph was unusable"
            );
            self.rebuild_index()?;
            self.buffer.clear();
        } else {
            let indexed: std::collections::HashSet<Id> = self
                .records
                .keys()
                .filter(|id| self.index.contains(id))
                .cloned()
                .collect();
            self.buffer = self
                .records
                .values()
                .filter(|r| r.embedding.is_some() && !indexed.contains(&r.id))
                .cloned()
                .collect();
            self.buffer.sort_by_key(|r| r.seq);
        }

        // Reopen the WAL resuming at the checkpoint's next_seq (a pruned WAL is
        // empty and must continue appending where the checkpoint left off).
        let epoch = Epoch(self.wal.epoch());
        let id = self.config.id;
        let stem = match self.shard {
            Some(s) => format!("{id}.shard-{}", s.0),
            None => id.to_string(),
        };
        self.wal = Wal::open_named(
            &self.wal_dir,
            &stem,
            epoch,
            WalOptions {
                fsync: self.fsync,
                start_seq: Some(meta.next_seq),
            },
            // Already inspected and opened once; this is a reposition, not a
            // second recovery decision.
            TailRepair::Refuse,
        )?;
        Ok(())
    }

    /// Replay WAL records with seq in `(flushed_seq, last_seq]` into in-memory
    /// state. Called with `flushed_seq = 0` for a full replay, or with the
    /// checkpoint's `flushed_seq` to replay just the tail.
    fn replay_after(&mut self, flushed_seq: u64) -> EngineResult<()> {
        let last = self.wal.last_seq();
        if last > flushed_seq {
            for rec in self.wal.read_range(flushed_seq + 1, last)? {
                self.apply_operation(&rec.op, rec.seq)?;
            }
        }
        Ok(())
    }

    /// Publish a checkpoint generation.
    ///
    /// The ordering below is the correctness mechanism, and it is why this is
    /// split into explicit stages rather than a sequence of writes:
    ///
    /// ```text
    /// S0  stage     create + write + fsync a new generation; live state untouched
    /// S1  validate  open the staged segment and check the generation is coherent
    /// S2  publish   atomically replace CURRENT (the only thing readers key off)
    /// S3  activate  swap the in-memory segment handles, strip, prune, reclaim
    /// ```
    ///
    /// Before S2 the previous generation is still authoritative and a failure
    /// leaves live state untouched. The earlier implementation assigned
    /// `self.segments` at S0, so a failure later in the write — or a silently
    /// ignored `Segment::open` failure — left queries resolving vectors out of a
    /// staging directory that `CURRENT` did not name and the next checkpoint's
    /// GC would delete.
    ///
    /// Returns [`CheckpointOutcome`]: `Err` means *nothing was published*; a
    /// checkpoint that is durable but has pending cleanup is a success, because
    /// reporting it as an error would tell the caller the opposite of the truth.
    pub(crate) fn checkpoint(&mut self, catalog: &dyn Catalog) -> EngineResult<CheckpointOutcome> {
        self.checkpoint_with(catalog, false)
    }

    /// Checkpoint and **compact**: rewrite every live vector into one segment,
    /// dropping superseded versions and deleted ids.
    ///
    /// Routine checkpoints are incremental and deliberately leave garbage
    /// behind; `optimize` is the operation that promises to reclaim it.
    fn checkpoint_compact(&mut self, catalog: &dyn Catalog) -> EngineResult<CheckpointOutcome> {
        self.checkpoint_with(catalog, true)
    }

    fn checkpoint_with(
        &mut self,
        catalog: &dyn Catalog,
        compact: bool,
    ) -> EngineResult<CheckpointOutcome> {
        let started = Instant::now();
        let mut timings = crate::CheckpointTimings::default();
        let result = self.checkpoint_inner(catalog, &mut timings, compact);
        timings.total_us = started.elapsed().as_micros() as u64;
        self.last_checkpoint_timings = timings;
        crate::metrics::record_checkpoint(&timings, result.is_ok());
        result
    }

    fn checkpoint_inner(
        &mut self,
        catalog: &dyn Catalog,
        timings: &mut crate::CheckpointTimings,
        compact: bool,
    ) -> EngineResult<CheckpointOutcome> {
        // Bound segment fan-out: an append-only checkpoint set grows with every
        // write batch, and every checkpoint re-verifies every segment.
        let compact = compact || self.segments.len() >= MAX_SEGMENTS;
        if compact && !self.segments.is_empty() {
            crate::metrics::record_compaction();
        }
        // Once a publication is uncertain, further mutations could build on a
        // generation we can no longer identify. Only a reopen clears this.
        if self.publication_uncertain {
            return Err(EngineError::Recovery(format!(
                "collection {}: a previous checkpoint publication is unresolved; \
                 restart the node to re-establish the active generation before writing",
                self.config.id
            )));
        }

        let next_seq = self.wal.last_seq().max(self.applied_seq) + 1;
        let snapshot_seq = next_seq - 1;
        let mut files: Vec<CheckpointFile> = Vec::new();
        let mut phase = Instant::now();

        // ---- S0: stage, with live state untouched -------------------------
        let generations = self.checkpoint_dir.join(GENERATIONS_DIR);
        std::fs::create_dir_all(&generations)?;
        // A new directory entry is only durable once its parent is synced;
        // syncing files inside it is not enough.
        sync_dir(&self.checkpoint_dir)?;

        let gen_name = format!(
            "gen-{snapshot_seq:020}-{}",
            &Uuid::new_v4().simple().to_string()[..8]
        );
        let gen_dir = generations.join(&gen_name);
        std::fs::create_dir_all(&gen_dir)?;
        sync_dir(&generations)?;
        fault::check("checkpoint.staging")?;

        let records_bytes = bincode::serialize(&self.records)
            .map_err(|e| EngineError::Serialization(e.to_string()))?;
        atomic_write(&gen_dir.join("records.bin"), &records_bytes)?;
        files.push(CheckpointFile {
            path: "records.bin".to_owned(),
            len: records_bytes.len() as u64,
            crc32: Some(crc32fast::hash(&records_bytes)),
        });
        timings.records_us = phase.elapsed().as_micros() as u64;
        phase = Instant::now();

        // The index is derived, so it is deliberately not inventoried: a lost
        // graph must not reject a generation whose authoritative data is intact.
        // The graph is derived and not inventoried, so it *could* be skipped.
        // It is still saved on every checkpoint: recovery rebuilds the graph
        // from scratch when it is absent, which is O(corpus) at startup, and
        // trading a bounded 78 ms per checkpoint (at 40k records) for an
        // unbounded restart cost is the wrong direction. Measured in
        // bench/RESULTS.md.
        if !self.index.is_empty() {
            self.index.save(&gen_dir.join("index.bin"))?;
        }
        timings.index_us = phase.elapsed().as_micros() as u64;
        phase = Instant::now();

        // Vectors are append-only across generations. A record whose vector is
        // already in a segment keeps its offset; only records carrying an
        // unsaved `embedding` are written to a new segment. The previous
        // implementation rewrote every vector on every checkpoint, so checkpoint
        // cost grew with the collection instead of with the write that
        // triggered it.
        // Compact: one segment holding every live vector, offsets renumbered.
        // Incremental: keep persisted offsets and append only unsaved vectors.
        let mut entries: Vec<(Id, Embedding)> = if compact {
            self.records
                .keys()
                .filter_map(|id| self.embedding_for(id).map(|e| (id.clone(), e)))
                .collect()
        } else {
            self.records
                .iter()
                .filter_map(|(id, record)| record.embedding.clone().map(|e| (id.clone(), e)))
                .collect()
        };
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut staged_index: HashMap<Id, usize> = HashMap::new();
        if !compact {
            for (id, &offset) in &self.segment_index {
                let dirty_now = self
                    .records
                    .get(id)
                    .is_some_and(|record| record.embedding.is_some());
                if self.records.contains_key(id) && !dirty_now {
                    staged_index.insert(id.clone(), offset);
                }
            }
        }

        let seg_dir = gen_dir.join("segments");
        let mut staged_segments: Vec<std::sync::Arc<crate::segment::Segment>> = Vec::new();
        let base: usize = if compact {
            0
        } else {
            self.segments.iter().map(|s| s.len()).sum()
        };

        if !entries.is_empty() {
            use crate::segment::SegmentWriter;
            std::fs::create_dir_all(&seg_dir)?;
            sync_dir(&gen_dir)?;

            let mut writer = SegmentWriter::new(self.config.dimension);
            for (i, (id, embedding)) in entries.iter().enumerate() {
                writer.push(embedding.as_ref());
                staged_index.insert(id.clone(), base + i);
            }
            let name = if compact {
                "seg-0.bin".to_owned()
            } else {
                format!("seg-{}.bin", self.segments.len())
            };
            let seg_path = seg_dir.join(&name);
            let seg_bytes = writer.write(&seg_path)?;
            // The directory is synced once, after linking, when every entry in
            // it is final; syncing twice only adds latency.
            files.push(CheckpointFile {
                path: format!("segments/{name}"),
                len: seg_bytes,
                // The segment verifies its own data block on open.
                crc32: None,
            });
        }

        if !staged_index.is_empty() {
            // Link the already-persisted segments into this generation: it must
            // stay self-contained for validation, backup, and GC, and a hard
            // link costs no data copy.
            if !compact && !self.segments.is_empty() {
                let previous = self.published_generation().ok_or_else(|| {
                    EngineError::Serialization(
                        "collection holds segments but has no published generation".into(),
                    )
                })?;
                std::fs::create_dir_all(&seg_dir)?;
                sync_dir(&gen_dir)?;
                for i in 0..self.segments.len() {
                    let name = format!("seg-{i}.bin");
                    let src = previous.join("segments").join(&name);
                    let dst = seg_dir.join(&name);
                    link_or_copy(&src, &dst)?;
                    files.push(CheckpointFile {
                        path: format!("segments/{name}"),
                        len: std::fs::metadata(&dst)?.len(),
                        crc32: None,
                    });
                }
            }
            sync_dir(&seg_dir)?;

            // Open the new segment *now*, before publishing. A failure here must
            // abort: publishing a manifest whose segments cannot be opened
            // would leave queries resolving vectors from files the next GC is
            // about to delete.
            //
            // Already-open segments are reused rather than reopened: they are
            // hard links to the same inodes, and `Segment::open` re-hashes the
            // whole data block. Reopening every segment turned each checkpoint
            // back into whole-corpus work.
            if compact {
                let path = seg_dir.join("seg-0.bin");
                let seg = crate::segment::Segment::open(&path).map_err(|e| {
                    EngineError::Serialization(format!(
                        "staged segment `{}` is unreadable: {e}",
                        path.display()
                    ))
                })?;
                staged_segments.push(std::sync::Arc::new(seg));
            } else if !entries.is_empty() {
                staged_segments.extend(self.segments.iter().cloned());
                let path = seg_dir.join(format!("seg-{}.bin", self.segments.len()));
                let seg = crate::segment::Segment::open(&path).map_err(|e| {
                    EngineError::Serialization(format!(
                        "staged segment `{}` is unreadable: {e}",
                        path.display()
                    ))
                })?;
                staged_segments.push(std::sync::Arc::new(seg));
            } else {
                staged_segments.extend(self.segments.iter().cloned());
            }

            timings.vectors_us = phase.elapsed().as_micros() as u64;
            phase = Instant::now();

            let idx_bytes = bincode::serialize(&staged_index)
                .map_err(|e| EngineError::Serialization(e.to_string()))?;
            atomic_write(&gen_dir.join("segment_index.bin"), &idx_bytes)?;
            files.push(CheckpointFile {
                path: "segment_index.bin".to_owned(),
                len: idx_bytes.len() as u64,
                crc32: Some(crc32fast::hash(&idx_bytes)),
            });
        }

        timings.index_map_us = phase.elapsed().as_micros() as u64;
        phase = Instant::now();

        let order = PersistedOrder {
            offsets: self.offsets.clone(),
            next_offset: self.next_offset,
        };
        let order_bytes =
            bincode::serialize(&order).map_err(|e| EngineError::Serialization(e.to_string()))?;
        atomic_write(&gen_dir.join("order.bin"), &order_bytes)?;
        files.push(CheckpointFile {
            path: "order.bin".to_owned(),
            len: order_bytes.len() as u64,
            crc32: Some(crc32fast::hash(&order_bytes)),
        });

        let meta = CheckpointMeta {
            format_version: CHECKPOINT_FORMAT_VERSION,
            flushed_seq: self.flushed_seq,
            snapshot_seq,
            next_seq,
            record_count: self.records.len() as u64,
            collection_id: self.config.id.to_string(),
            dimension: self.config.dimension,
            space: self.config.space.name().to_owned(),
            files,
        };
        atomic_write(
            &gen_dir.join("checkpoint.json"),
            &serde_json::to_vec(&meta).map_err(|e| EngineError::Serialization(e.to_string()))?,
        )?;
        sync_dir(&gen_dir)?;
        timings.manifest_us = phase.elapsed().as_micros() as u64;
        phase = Instant::now();
        fault::check("checkpoint.files")?;

        // ---- S1: validate before anything becomes authoritative -----------
        // On failure the staging directory is removed: it was never published,
        // so leaving it behind only accumulates orphan generations and makes
        // the next checkpoint re-attempt more expensive.
        if let Err(e) = self.validate_staged(&gen_dir, &meta, &staged_segments, &staged_index) {
            let _ = std::fs::remove_dir_all(&gen_dir);
            return Err(e);
        }
        timings.validate_us = phase.elapsed().as_micros() as u64;
        phase = Instant::now();
        fault::check("checkpoint.validated")?;

        // ---- S2: publish ---------------------------------------------------
        // Before this rename a reopen observes the previous generation; after
        // it, this one. Never a mixture.
        atomic_write(
            &self.checkpoint_dir.join(CURRENT_POINTER),
            gen_name.as_bytes(),
        )?;
        // Past the rename the checkpoint is durable even if the directory sync
        // did not complete, so an injected failure here must be reported as an
        // unresolved publication rather than as "nothing happened".
        if let Err(e) = fault::check("checkpoint.published") {
            self.publication_uncertain = true;
            return Err(e);
        }

        // ---- S3: activate --------------------------------------------------
        self.segment_starts = Self::segment_starts(&staged_segments);
        self.segments = staged_segments;
        self.segment_index = staged_index;
        for record in self.records.values_mut() {
            if record.seq <= meta.flushed_seq {
                record.embedding = None;
            }
        }
        self.last_checkpoint_seq = snapshot_seq;
        self.applied_since_checkpoint = 0;
        timings.publish_us = phase.elapsed().as_micros() as u64;
        phase = Instant::now();

        // Cleanup below this line cannot invalidate the published state.
        if fault::check_deferred("checkpoint.catalog", "catalog").is_some() {
            return Ok(CheckpointOutcome::Published {
                deferred: "catalog",
            });
        }
        if let Err(e) = catalog.advance_log_offset(&self.config.id, snapshot_seq) {
            tracing::warn!(
                collection = %self.config.id,
                "checkpoint published but the catalog watermark was not advanced: {e}"
            );
            return Ok(CheckpointOutcome::Published {
                deferred: "catalog",
            });
        }

        if fault::check_deferred("checkpoint.pruned", "prune").is_some() {
            return Ok(CheckpointOutcome::Published { deferred: "prune" });
        }
        if let Err(e) = self.wal.prune(snapshot_seq) {
            tracing::warn!(
                collection = %self.config.id,
                "checkpoint published but the WAL prefix was not pruned: {e}"
            );
            return Ok(CheckpointOutcome::Published { deferred: "prune" });
        }
        // Reopen so appends resume where the pruned log left off.
        let epoch = Epoch(self.wal.epoch());
        let wal_dir = self.wal_dir.clone();
        let id = self.config.id;
        let stem = match self.shard {
            Some(s) => format!("{id}.shard-{}", s.0),
            None => id.to_string(),
        };
        self.wal = Wal::open_named(
            &wal_dir,
            &stem,
            epoch,
            WalOptions {
                fsync: self.fsync,
                start_seq: Some(next_seq),
            },
            TailRepair::Refuse,
        )?;

        if fault::check_deferred("checkpoint.gc", "gc").is_some() {
            return Ok(CheckpointOutcome::Published { deferred: "gc" });
        }
        self.gc_old_generations(&gen_name);
        timings.cleanup_us = phase.elapsed().as_micros() as u64;
        Ok(CheckpointOutcome::Complete)
    }

    /// Check that a just-written generation is internally coherent.
    ///
    /// This runs before publication, so every failure here aborts with live
    /// state untouched. It deliberately avoids re-reading `records.bin` and
    /// `order.bin` — they were serialized from the structures checked below, and
    /// a full re-read would double checkpoint I/O. Recovery performs the
    /// independent, from-disk validation in [`Collection::read_generation`].
    ///
    /// Incremental generations legitimately retain dead vectors: segments are
    /// append-only and hard-linked, so a deleted or superseded vector stays on
    /// disk until `optimize` compacts. The invariant is therefore *not* that
    /// the segment vector count equals the live count, but that every live
    /// vector resolves and every staged mapping points at a live record.
    fn validate_staged(
        &self,
        gen_dir: &Path,
        meta: &CheckpointMeta,
        segments: &[std::sync::Arc<crate::segment::Segment>],
        segment_index: &HashMap<Id, usize>,
    ) -> EngineResult<()> {
        let reject = |reason: String| Err(EngineError::Serialization(reason));

        if meta.collection_id != self.config.id.to_string() {
            return reject("staged manifest names a different collection".into());
        }
        if meta.dimension != self.config.dimension {
            return reject("staged manifest records a different vector width".into());
        }
        if meta.space != self.config.space.name() {
            return reject("staged manifest records a different distance space".into());
        }
        if meta.snapshot_seq != meta.next_seq.saturating_sub(1) {
            return reject("staged manifest has an inconsistent snapshot boundary".into());
        }
        if meta.record_count != self.records.len() as u64 {
            return reject(format!(
                "staged manifest declares {} records but {} are live",
                meta.record_count,
                self.records.len()
            ));
        }
        for (id, record) in &self.records {
            if record.seq > meta.snapshot_seq {
                return reject(format!(
                    "record `{id}` has seq {} past the snapshot boundary",
                    record.seq
                ));
            }
        }

        if segments.is_empty() {
            if !segment_index.is_empty() {
                return reject("staged generation has a segment index but no segment".into());
            }
            // No vectors were persisted. Any live record that needs a segment
            // (stripped of its inline embedding) would be unresolvable, and any
            // record still carrying an embedding should have been written.
            for (id, record) in &self.records {
                if record.embedding.is_some() {
                    return reject(format!(
                        "record `{id}` carries a vector but the staged generation holds no segment"
                    ));
                }
                if record.seq <= meta.flushed_seq {
                    return reject(format!(
                        "record `{id}` is covered by the checkpoint but the staged generation holds no segment"
                    ));
                }
            }
        } else {
            let total: usize = segments.iter().map(|s| s.len()).sum();
            for (i, segment) in segments.iter().enumerate() {
                if segment.is_empty() {
                    return reject(format!("staged segment {i} holds no vectors"));
                }
                if segment.dimension() != self.config.dimension {
                    return reject(format!(
                        "staged segment {i} is {}-dimensional but the collection is {}",
                        segment.dimension(),
                        self.config.dimension
                    ));
                }
            }
            // Dead vectors are expected: incremental checkpoints hard-link
            // previous segments, which still contain deleted and superseded
            // vectors. The segment store may therefore hold *more* vectors
            // than the live set references. Holding fewer is corruption.
            if total < segment_index.len() {
                return reject(format!(
                    "staged segments hold {total} vectors but the staged index references {} records",
                    segment_index.len()
                ));
            }
            let starts = Self::segment_starts(segments);
            let mut seen_offsets = std::collections::HashSet::with_capacity(segment_index.len());
            for (id, &offset) in segment_index {
                if offset >= total {
                    return reject(format!(
                        "staged segment index maps `{id}` to offset {offset}, out of bounds"
                    ));
                }
                if !seen_offsets.insert(offset) {
                    return reject(format!(
                        "staged segment index maps more than one record to offset {offset}"
                    ));
                }
                if !self.records.contains_key(id) {
                    return reject(format!(
                        "staged segment index references unknown record `{id}`"
                    ));
                }
                // The offset must resolve through the staged segments, not
                // just be in range: a gap (e.g. a short write) would otherwise
                // surface later as a record that exists but can never be matched.
                if Self::resolve_vector(segments, &starts, offset).is_none() {
                    return reject(format!(
                        "staged segment index maps `{id}` to offset {offset}, which does not resolve"
                    ));
                }
            }
            // Every live vector must be referenced. Records carrying an inline
            // embedding are the newly written ones; stripped records (no
            // inline embedding, seq within the flushed watermark) live in the
            // carried-over segments. Metadata-only records (no embedding, seq
            // past the watermark) legitimately have no vector.
            for (id, record) in &self.records {
                if record.embedding.is_some() {
                    if !segment_index.contains_key(id) {
                        return reject(format!(
                            "live record `{id}` carries a vector missing from the staged index"
                        ));
                    }
                } else if record.seq <= meta.flushed_seq && !segment_index.contains_key(id) {
                    return reject(format!(
                        "record `{id}` is covered by the checkpoint but has no vector in the staged generation"
                    ));
                }
            }
        }

        if self.offsets.len() != self.records.len() {
            return reject(format!(
                "ordering covers {} records but {} are live",
                self.offsets.len(),
                self.records.len()
            ));
        }
        let max_offset = self.offsets.values().copied().max().unwrap_or(0);
        if !self.offsets.is_empty() && self.next_offset <= max_offset {
            return reject(format!(
                "next_offset {} does not exceed the highest issued offset {max_offset}",
                self.next_offset
            ));
        }

        // Inventory lengths must match what is actually on disk: a short write
        // that still reported success would otherwise be published as a
        // complete generation.
        for entry in &meta.files {
            let path = gen_dir.join(&entry.path);
            match std::fs::metadata(&path) {
                Ok(md) if md.len() == entry.len => {}
                Ok(md) => {
                    return reject(format!(
                        "staged `{}` is {} bytes but the manifest declares {}",
                        entry.path,
                        md.len(),
                        entry.len
                    ));
                }
                Err(e) => {
                    return reject(format!("staged `{}` is unreadable: {e}", entry.path));
                }
            }
        }
        Ok(())
    }

    /// Delete every generation directory except `current`.
    ///
    /// Only call after `CURRENT` names `current` and has been fsynced: any
    /// generation still referenced by the pointer must survive.
    fn gc_old_generations(&self, current: &str) {
        let gens = self.checkpoint_dir.join(GENERATIONS_DIR);
        let Ok(entries) = std::fs::read_dir(&gens) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if name == current {
                continue;
            }
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }

    /// Adopt a dimension inferred from the first write.
    ///
    /// The collection is created with `dimension = 0` when the caller did not
    /// state one, and `Collection::new` builds its HNSW index at that width. The
    /// index was never re-created when the real width arrived, so it stayed at
    /// dimension 0 and the *first* flush or rebuild failed with
    /// "index is 0, got N" — which surfaced only once the buffer reached
    /// `batch_size`, not on the write that set the dimension.
    ///
    /// No vectors can exist in a dimension-0 index, so replacing it loses
    /// nothing.
    pub(crate) fn adopt_dimension(&mut self, dimension: usize) -> EngineResult<()> {
        if self.config.dimension == dimension {
            return Ok(());
        }
        self.config.dimension = dimension;
        self.index = Box::new(UsearchIndex::with_hnsw(
            self.config.space,
            dimension,
            &self.config.hnsw,
        )?);
        Ok(())
    }

    /// Rebuild the HNSW index from the live records (used by `optimize` to
    /// reclaim space after deletes and by reopen to restore a postings set).
    pub(crate) fn rebuild_index(&mut self) -> EngineResult<()> {
        let index =
            UsearchIndex::with_hnsw(self.config.space, self.config.dimension, &self.config.hnsw)?;
        // Resolve every record's *current* vector before swapping the index.
        // Segment-first resolution would silently rebuild the graph from stale
        // vectors for anything upserted after the last checkpoint.
        let mut to_add: Vec<(Id, Embedding)> = self
            .records
            .keys()
            .filter_map(|id| self.embedding_for(id).map(|e| (id.clone(), e)))
            .collect();
        to_add.sort_by(|a, b| a.0.cmp(&b.0));
        self.index = Box::new(index);
        for (id, embedding) in to_add {
            self.index.add(&id, &embedding)?;
        }
        Ok(())
    }

    /// Force the whole buffer into the index, rebuild the HNSW graph from live
    /// records, checkpoint, and report the WAL size change. `optimize` is the
    /// only path that shrinks the graph after deletes.
    pub(crate) fn optimize(&mut self, catalog: &dyn Catalog) -> EngineResult<OptimizeStats> {
        let records = self.records.len() as u64;
        let t = Instant::now();
        self.flush_buffer()?;
        let flush_us = t.elapsed().as_micros() as u64;

        let indexed = self.index.len() as u64;
        let wal_bytes_before = self.wal.size_bytes();
        let t = Instant::now();
        self.rebuild_index()?;
        let rebuild_us = t.elapsed().as_micros() as u64;

        self.checkpoint_compact(catalog)?;
        let wal_bytes_after = self.wal.size_bytes();
        Ok(OptimizeStats {
            records,
            indexed,
            wal_bytes_before,
            wal_bytes_after,
            flush_us,
            rebuild_us,
            checkpoint: self.last_checkpoint_timings,
        })
    }

    /// Auto-checkpoint when the WAL has accumulated `sync_threshold` records
    /// past the last checkpoint watermark. Best-effort: the write already
    /// committed to the WAL, so a checkpoint failure is logged and the WAL
    /// tail is kept — it must never turn an acknowledged write into an error.
    /// Record how many entries have been applied since the last checkpoint.
    ///
    /// Cluster mode has no local WAL to measure growth against, so the trigger
    /// is the number of applied log entries instead.
    pub(crate) fn note_applied(&mut self, count: u64) {
        self.applied_since_checkpoint = self.applied_since_checkpoint.saturating_add(count);
    }

    /// Whether a cluster-mode checkpoint is due.
    pub(crate) fn checkpoint_due(&self) -> bool {
        let threshold = self.config.hnsw.sync_threshold as u64;
        threshold > 0 && self.applied_since_checkpoint >= threshold
    }

    pub(crate) fn auto_checkpoint(&mut self, catalog: &dyn Catalog) -> EngineResult<()> {
        let threshold = self.config.hnsw.sync_threshold as u64;
        let new_records = self.wal.last_seq().saturating_sub(self.last_checkpoint_seq);
        // Deliberately not gated on a non-empty record map. A collection whose
        // records were all deleted has an empty map but a WAL full of delete
        // operations; skipping it meant that history could never be pruned and
        // every restart replayed all of it.
        if threshold > 0
            && new_records >= threshold
            && let Err(e) = self.checkpoint(catalog)
        {
            // The batch already committed to the WAL before this ran, so the
            // data is durable. Failing the write here would tell the client the
            // write failed when it did not (clients retry, duplicating it)
            // while also leaving an orphan staging directory per attempt.
            tracing::warn!(
                collection = %self.config.id,
                "auto-checkpoint failed; WAL tail retained: {e}"
            );
            crate::metrics::record_checkpoint(&crate::CheckpointTimings::default(), false);
        }
        Ok(())
    }

    /// The published checkpoint generation directory, if one exists.
    ///
    /// Anything that ships or inspects checkpoint files (object-store export,
    /// backup) must read from here: this is the only layout the engine writes,
    /// and `CURRENT` is the only pointer to it.
    pub(crate) fn published_generation(&self) -> Option<PathBuf> {
        active_generation(&self.checkpoint_dir)
    }

    /// Commit a whole batch to the WAL first (each op gets its seq), then apply
    /// the ops to in-memory state. A WAL failure aborts the batch: nothing is
    /// applied to memory (the WAL may hold a prefix, which replay re-applies).
    pub(crate) fn write_ops(&mut self, ops: &[Operation]) -> EngineResult<()> {
        if self.publication_uncertain {
            return Err(EngineError::Recovery(format!(
                "collection {}: refusing writes while a checkpoint publication is \
                 unresolved; restart the node to re-establish the active generation",
                self.config.id
            )));
        }
        // One atomic append: on a batch-framed log the whole request either
        // lands or none of it does, so a crash cannot expose half a batch.
        // The previous loop appended per operation, and a failure part-way
        // left a persisted prefix that replay then applied.
        let appended = self.wal.append_batch(ops)?;
        for rec in &appended {
            self.apply_operation(&rec.op, rec.seq)?;
        }
        crate::metrics::record_write(
            ops.len(),
            self.wal.size_bytes(),
            self.records.len(),
            self.fsync,
        );
        Ok(())
    }

    /// Apply one operation to records + buffer (+ index for deletes) + postings.
    /// Used both at write time and on replay (replay rebuilds records, buffer,
    /// index, and postings from the WAL). The buffer holds at most one entry
    /// per id (the newest), so flush and query merging never see duplicate ids.
    ///
    /// Postings maintenance happens here so replay rebuilds them for free: the
    /// metadata index and the offset maps are derived state, exactly like the
    /// index.
    pub(crate) fn apply_operation(&mut self, op: &Operation, seq: u64) -> EngineResult<()> {
        // Both write paths converge here, so this is the one place that knows
        // how far the collection has been advanced regardless of where the
        // sequence came from (WAL seq single-node, Raft index in a cluster).
        self.applied_seq = self.applied_seq.max(seq);
        match op {
            Operation::Add {
                id,
                embedding,
                metadata,
                document,
            }
            | Operation::Upsert {
                id,
                embedding,
                metadata,
                document,
            } => {
                // The offset pre-exists on an upsert of an existing id and is
                // allocated fresh on an add. Old metadata is un-indexed before
                // the new metadata is indexed.
                let offset = self.offset_for(id);
                if let Some(old) = self.records.get(id) {
                    self.postings.remove(offset, old.metadata.as_ref());
                }
                self.postings.insert(offset, metadata.as_ref());
                let record = Record {
                    id: id.clone(),
                    embedding: Some(embedding.clone()),
                    metadata: metadata.clone(),
                    document: document.clone(),
                    seq,
                };
                self.records.insert(id.clone(), record.clone());
                self.buffer.retain(|r| r.id != *id);
                self.buffer.push(record);
            }
            Operation::Update {
                id,
                metadata,
                document,
            } => {
                let record = self.records.entry(id.clone()).or_insert_with(|| Record {
                    id: id.clone(),
                    embedding: None,
                    metadata: None,
                    document: None,
                    seq,
                });
                let changed = record.metadata != *metadata;
                let old_metadata = record.metadata.clone();
                record.metadata = metadata.clone();
                record.document = document.clone();
                record.seq = seq;
                if changed {
                    let offset = self.offset_for(id);
                    self.postings.remove(offset, old_metadata.as_ref());
                    self.postings.insert(offset, metadata.as_ref());
                }
                // Keep the pending buffer entry in step with the record.
                // `filtered_query` evaluates eligibility against the buffer's
                // metadata, so a metadata-only update that left the buffer
                // untouched made a record that no longer matched the filter
                // still appear in (or still be missing from) filtered results
                // until it was flushed.
                if let Some(pending) = self.buffer.iter_mut().find(|r| r.id == *id) {
                    pending.metadata = metadata.clone();
                    pending.document = document.clone();
                    pending.seq = seq;
                }
            }
            Operation::Delete { id } => {
                if let Some(offset) = self.offsets.get(id).copied() {
                    if let Some(record) = self.records.get(id) {
                        self.postings.remove(offset, record.metadata.as_ref());
                    }
                    self.postings.remove_all(offset);
                    self.id_by_offset.remove(&offset);
                    self.offsets.remove(id);
                }
                self.records.remove(id);
                self.buffer.retain(|r| r.id != *id);
                // Eager physical delete: a flushed vector must not survive a
                // record delete in query results.
                self.index.delete(id)?;
            }
        }
        Ok(())
    }

    /// Cumulative base offset of each segment in the logical vector space.
    pub(crate) fn segment_starts(
        segments: &[std::sync::Arc<crate::segment::Segment>],
    ) -> Vec<usize> {
        let mut starts = Vec::with_capacity(segments.len());
        let mut base = 0;
        for segment in segments {
            starts.push(base);
            base += segment.len();
        }
        starts
    }

    /// Resolve a global vector offset to its segment and local index.
    fn resolve_vector<'a>(
        segments: &'a [std::sync::Arc<crate::segment::Segment>],
        starts: &[usize],
        offset: usize,
    ) -> Option<&'a [f32]> {
        // Empty segments are never written, but a duplicated start would make
        // `binary_search` ambiguous, so reject them explicitly.
        let idx = match starts.binary_search(&offset) {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i - 1,
        };
        segments.get(idx)?.get_vector(offset - starts[idx])
    }

    /// Load a vector from segments without borrowing self.
    fn load_vector_from<'a>(
        segments: &'a [std::sync::Arc<crate::segment::Segment>],
        starts: &[usize],
        segment_index: &'a HashMap<Id, usize>,
        id: &Id,
    ) -> Option<&'a [f32]> {
        let &offset = segment_index.get(id)?;
        Self::resolve_vector(segments, starts, offset)
    }

    /// Resolve the **current** vector of a live record.
    ///
    /// The in-memory record is consulted first because it always holds the
    /// newest version: a checkpoint only strips `embedding` after the vector has
    /// been written to the segment, and an `upsert` after that point
    /// re-populates it with a vector the segment does not have yet. The segment
    /// is the fallback for records a checkpoint has already covered.
    ///
    /// Reading `record.embedding` directly silently drops every checkpointed
    /// record (it is `None` for them); reading the segment first scores a
    /// record that was upserted after the last checkpoint with its stale
    /// vector. This is the only sanctioned way to obtain a record's vector.
    pub(crate) fn vector_for(&self, id: &Id) -> Option<&[f32]> {
        if let Some(embedding) = self.records.get(id)?.embedding.as_ref() {
            return Some(embedding.as_ref());
        }
        Self::load_vector_from(
            &self.segments,
            &self.segment_starts,
            &self.segment_index,
            id,
        )
    }

    /// Owned variant of [`Collection::vector_for`]. Reuses the record's `Arc`
    /// when the live record already holds the vector, and only copies when the
    /// vector has to be read out of a mmap segment.
    pub(crate) fn embedding_for(&self, id: &Id) -> Option<Embedding> {
        if let Some(embedding) = self.records.get(id)?.embedding.as_ref() {
            return Some(embedding.clone());
        }
        Self::load_vector_from(
            &self.segments,
            &self.segment_starts,
            &self.segment_index,
            id,
        )
        .map(|v| Embedding::from(v.to_vec()))
    }

    /// Return the internal offset for `id`, allocating a new monotonic offset
    /// (and registering the reverse map entry) if the id has none yet. Offsets
    /// are never reused after delete.
    fn offset_for(&mut self, id: &Id) -> u32 {
        if let Some(&offset) = self.offsets.get(id) {
            return offset;
        }
        let offset = self.next_offset;
        self.next_offset += 1;
        self.offsets.insert(id.clone(), offset);
        self.id_by_offset.insert(offset, id.clone());
        offset
    }

    /// Chroma flush policy: once the buffer reaches `batch_size` after a write
    /// batch, drain it into the HNSW index.
    pub(crate) fn flush_if_needed(&mut self) -> EngineResult<()> {
        if self.buffer.len() >= self.config.hnsw.batch_size {
            self.flush_buffer()?;
        }
        Ok(())
    }

    /// Drain the buffer into the index. An id already committed is deleted then
    /// re-added (covers upsert-over-flushed-vector). Records are processed in
    /// seq order, so the last version of an id wins.
    fn flush_buffer(&mut self) -> EngineResult<()> {
        let batch = std::mem::take(&mut self.buffer);
        let mut max_seq = 0u64;
        for record in &batch {
            let Some(embedding) = &record.embedding else {
                continue;
            };
            if self.index.contains(&record.id) {
                self.index.delete(&record.id)?;
            }
            self.index.add(&record.id, embedding)?;
            max_seq = max_seq.max(record.seq);
        }
        self.flushed_seq = self.flushed_seq.max(max_seq);
        Ok(())
    }

    /// `add` validation: counts and dimensions must match, and no id may
    /// already exist (duplicate within the batch or against live records).
    /// Validates the entire batch before anything is written.
    pub(crate) fn validate_add(
        &self,
        ids: &[Id],
        embeddings: &[Embedding],
        metadatas: Option<&[Option<Metadata>]>,
        documents: Option<&[Option<Document>]>,
    ) -> EngineResult<()> {
        if ids.len() != embeddings.len() {
            return Err(EngineError::Validation(format!(
                "add: {} ids but {} embeddings",
                ids.len(),
                embeddings.len()
            )));
        }
        check_slice_lengths("add", ids, metadatas, "metadatas")?;
        check_slice_lengths("add", ids, documents, "documents")?;
        check_dimensions("add", embeddings, self.config.dimension)?;
        let mut seen = std::collections::HashSet::new();
        for (id, _) in ids.iter().zip(embeddings) {
            if self.records.contains_key(id) {
                return Err(EngineError::Validation(format!(
                    "add: id `{id}` already exists"
                )));
            }
            if !seen.insert(id) {
                return Err(EngineError::Validation(format!(
                    "add: duplicate id `{id}` within the batch"
                )));
            }
        }
        Ok(())
    }

    /// `upsert` validation: counts and dimensions must match. Existing ids are
    /// allowed (that is the point of upsert).
    pub(crate) fn validate_upsert(
        &self,
        ids: &[Id],
        embeddings: &[Embedding],
        metadatas: Option<&[Option<crate::MetadataPatch>]>,
        documents: Option<&[Option<Document>]>,
    ) -> EngineResult<()> {
        if ids.len() != embeddings.len() {
            return Err(EngineError::Validation(format!(
                "upsert: {} ids but {} embeddings",
                ids.len(),
                embeddings.len()
            )));
        }
        check_slice_lengths("upsert", ids, metadatas, "metadatas")?;
        check_slice_lengths("upsert", ids, documents, "documents")?;
        check_dimensions("upsert", embeddings, self.config.dimension)?;
        Ok(())
    }

    /// `update` validation: counts must match and every id must exist.
    /// Validate an update batch.
    ///
    /// Unknown ids are deliberately **not** an error: Chroma answers 200 and
    /// changes nothing for them. Rejecting the whole batch (the previous
    /// behaviour) turned a partially-stale client request into a failure and
    /// made an update that touched one deleted id impossible to apply.
    pub(crate) fn validate_update(
        &self,
        ids: &[Id],
        metadatas: Option<&[Option<crate::MetadataPatch>]>,
        documents: Option<&[Option<Document>]>,
    ) -> EngineResult<()> {
        check_slice_lengths("update", ids, metadatas, "metadatas")?;
        check_slice_lengths("update", ids, documents, "documents")?;
        Ok(())
    }

    /// Run a query. With `options.where_filter == None` this keeps the existing
    /// unfiltered behavior exactly: for small collections
    /// (`index + buffer <= max_scan`) an exact brute-force scan over all live
    /// records, otherwise the index search merged with a buffer scan. With a
    /// filter, see [`Collection::filtered_query`] for the planner and the
    /// recall tradeoff. Results are sorted ascending by distance and truncated
    /// to top-k.
    pub(crate) fn query(
        &self,
        query: &Embedding,
        k: usize,
        options: &QueryOptions,
        with_embedding: bool,
    ) -> EngineResult<Vec<ScoredPoint>> {
        use std::cmp::Ordering;

        if self.config.dimension != 0 && query.len() != self.config.dimension {
            return Err(EngineError::Validation(format!(
                "query: embedding dimension {} != collection dimension {}",
                query.len(),
                self.config.dimension
            )));
        }
        let k = k.max(1);
        let ef = if options.ef == 0 {
            self.config.hnsw.ef_search
        } else {
            options.ef
        };
        let space = self.config.space;

        let params = SearchParams {
            k,
            ef,
            oversampling: options.oversampling,
            with_embedding,
        };
        let mut out = if options.where_document.is_some() {
            // Documents are not indexed, so eligibility cannot come from the
            // graph's candidate set: an ANN over-fetch of `k × oversampling`
            // would return fewer than `k` matches whenever matching documents
            // are sparse. An exact scan is the only plan that honours `k`, and
            // the metadata filter becomes a predicate too so both compose.
            self.scan_matching(query, space, options, with_embedding)
        } else {
            match &options.where_filter {
                None => {
                    if self.index.len() + self.buffer.len() <= self.config.hnsw.max_scan {
                        self.brute_force(query, space, with_embedding)
                    } else {
                        self.hybrid_search(query, space, &params)?
                    }
                }
                Some(filter) => self.filtered_query(query, space, filter, &params)?,
            }
        };

        out.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        out.truncate(k);
        Ok(out)
    }

    /// Exact scan over every live record that has a vector and satisfies both
    /// eligibility filters.
    ///
    /// Used when a document filter is present: documents are not indexed, so
    /// there is no candidate set to over-fetch from. Costs one pass over the
    /// collection's records, which is the price of the filter being honest.
    fn scan_matching(
        &self,
        query: &Embedding,
        space: Distance,
        options: &QueryOptions,
        with_embedding: bool,
    ) -> Vec<ScoredPoint> {
        self.records
            .iter()
            .filter(|(_, r)| {
                options
                    .where_filter
                    .as_ref()
                    .is_none_or(|f| r.metadata.as_ref().is_some_and(|m| f.matches(m)))
            })
            .filter(|(_, r)| {
                options
                    .where_document
                    .as_ref()
                    .is_none_or(|d| d.matches(r.document.as_deref()))
            })
            .filter_map(|(id, r)| {
                let embedding = self.vector_for(id)?;
                Some(ScoredPoint {
                    id: id.clone(),
                    distance: distance(space, query, embedding),
                    metadata: r.metadata.clone(),
                    document: r.document.clone(),
                    embedding: with_embedding.then(|| Embedding::from(embedding.to_vec())),
                })
            })
            .collect()
    }

    /// Exact scan over every live record that has a vector.
    fn brute_force(
        &self,
        query: &Embedding,
        space: Distance,
        with_embedding: bool,
    ) -> Vec<ScoredPoint> {
        self.records
            .iter()
            .filter_map(|(id, r)| {
                let embedding = self.vector_for(id)?;
                Some(ScoredPoint {
                    id: id.clone(),
                    distance: distance(space, query, embedding),
                    metadata: r.metadata.clone(),
                    document: r.document.clone(),
                    // Copied from the same slice the distance was computed
                    // from, so the pair cannot disagree.
                    embedding: with_embedding.then(|| Embedding::from(embedding.to_vec())),
                })
            })
            .collect()
    }

    /// Approximate search: index hits plus a brute-force scan of the buffer,
    /// merged by id (buffer wins). Metadata/document always come from the live
    /// records map, which is the newest state for every id.
    fn hybrid_search(
        &self,
        query: &Embedding,
        space: Distance,
        params: &SearchParams,
    ) -> EngineResult<Vec<ScoredPoint>> {
        let hits = self.index.search(query, params.k, params.ef)?;
        let mut dist: HashMap<Id, f32> = HashMap::with_capacity(hits.len() + self.buffer.len());
        for hit in hits {
            dist.entry(hit.id).or_insert(hit.distance);
        }
        for record in &self.buffer {
            if let Some(embedding) = &record.embedding {
                dist.insert(record.id.clone(), distance(space, query, embedding));
            }
        }
        Ok(self.build_scored(dist, params.with_embedding))
    }

    /// Chroma-style metadata-filtered query.
    ///
    /// Filters decide **eligibility, never ranking**: the eligible set is the
    /// records whose metadata matches `filter`, and candidates are ranked by
    /// distance only.
    ///
    /// # The planner
    ///
    /// 1. **Buffer candidates**: every eligible record still in the brute-force
    ///    buffer is scanned exhaustively (buffer records carry their embedding;
    ///    metadata-only records — `embedding: None` — are skipped).
    /// 2. **Index candidates**: `postings.evaluate(filter)` yields the bitmap
    ///    of eligible internal offsets → ids via `id_by_offset`. If the
    ///    candidate count fits `config.hnsw.max_scan`, every eligible record's
    ///    embedding is scanned directly — **exact**. Otherwise the **ANN path**
    ///    walks the HNSW graph with `ef` and fetches `k * oversampling`
    ///    candidates, keeping only the ones in the eligible set.
    /// 3. Merge buffer + index results, dedupe by id (buffer wins — newer),
    ///    sort ascending by distance, truncate to top-k.
    ///
    /// # Recall tradeoff
    ///
    /// The filtered query is exact when the eligible set is small enough for
    /// the brute-force path (`candidate_count <= max_scan`). Otherwise it is
    /// approximate: a selective filter (few eligible records) combined with a
    /// small oversampling factor can miss true nearest neighbors that the
    /// graph walk does not surface, because the HNSW search returns its global
    /// top candidates before eligibility is applied. Raising `ef` and
    /// `oversampling` (or growing `max_scan`) improves recall at a latency
    /// cost.
    fn filtered_query(
        &self,
        query: &Embedding,
        space: Distance,
        filter: &WhereFilter,
        params: &SearchParams,
    ) -> EngineResult<Vec<ScoredPoint>> {
        // Buffer candidates: eligible buffer records are never in the index, so
        // they are always brute-forced. This also covers the case where an id
        // sits in both the buffer and the index (upsert over a flushed vector):
        // the buffer distance inserted here wins via `or_insert` below.
        let mut dist: HashMap<Id, f32> = HashMap::new();
        for record in &self.buffer {
            if !filter_matches_opt(filter, &record.metadata) {
                continue;
            }
            if let Some(embedding) = &record.embedding {
                dist.insert(record.id.clone(), distance(space, query, embedding));
            }
        }

        // Index candidates: eligible offsets from the postings.
        let candidates = self.postings.evaluate(filter);
        if candidates.len() as usize <= self.config.hnsw.max_scan {
            // Exact path: scan every eligible record's current vector.
            for offset in candidates.iter() {
                let Some(id) = self.id_by_offset.get(&offset) else {
                    continue;
                };
                if !self.records.contains_key(id) {
                    continue;
                }
                // Resolve through `vector_for`: a checkpointed record carries no
                // in-memory embedding and must be read from the mmap segment.
                let Some(embedding) = self.vector_for(id) else {
                    continue;
                };
                dist.entry(id.clone())
                    .or_insert_with(|| distance(space, query, embedding));
            }
        } else {
            // ANN path: widen the beam, fetch k * oversampling candidates, and
            // keep only the ones eligible per the postings.
            let fetch = params.k.saturating_mul(params.oversampling.max(1));
            let hits = self.index.search(query, fetch, params.ef)?;
            for hit in hits {
                let Some(offset) = self.offsets.get(&hit.id) else {
                    continue;
                };
                if candidates.contains(*offset) {
                    dist.entry(hit.id).or_insert(hit.distance);
                }
            }
        }

        Ok(self.build_scored(dist, params.with_embedding))
    }

    /// Live record ids in internal-offset order (insertion order).
    pub(crate) fn all_ids(&self) -> Vec<Id> {
        let mut offs: Vec<u32> = self.id_by_offset.keys().copied().collect();
        offs.sort_unstable();
        offs.into_iter()
            .filter_map(|o| self.id_by_offset.get(&o))
            .filter(|id| self.records.contains_key(*id))
            .cloned()
            .collect()
    }

    /// Whether the live record `id` satisfies `filter`. Uses the same predicate
    /// the postings index is compiled from, so filtered scans and filtered
    /// queries agree even for records with no metadata.
    pub(crate) fn metadata_matches(&self, id: &Id, filter: &WhereFilter) -> bool {
        let metadata = self.records.get(id).and_then(|r| r.metadata.clone());
        filter_matches_opt(filter, &metadata)
    }

    /// Live record ids matching `filter`, in internal-offset order (which is
    /// insertion order, since offsets are assigned monotonically), with
    /// `offset`/`limit` applied.
    ///
    /// Records are re-checked against `records` because a delete removes the
    /// offset from the postings only on the delete path; this keeps the result
    /// honest for any stale bitmap entry.
    /// The ids a read or a filter-based delete would target, in insertion
    /// order.
    ///
    /// Deliberately mirrors [`Engine::get_records`] so that what a client was
    /// shown and what a delete removes are chosen by the same rules. Callers
    /// hold the collection lock, which is what makes the selection and the
    /// mutation that follows it one operation.
    pub(crate) fn select(
        &self,
        ids: Option<&[Id]>,
        filter: Option<&WhereFilter>,
        document: Option<&WhereDocument>,
    ) -> Vec<Id> {
        let mut selected: Vec<Id> = match ids {
            Some(ids) => {
                let mut wanted: Vec<Id> = ids
                    .iter()
                    .filter(|id| self.records.contains_key(*id))
                    .cloned()
                    .collect();
                wanted.sort_by_key(|id| self.offsets.get(id).copied().unwrap_or(u32::MAX));
                wanted
            }
            None => self.all_ids(),
        };
        if let Some(filter) = filter {
            selected.retain(|id| self.metadata_matches(id, filter));
        }
        if let Some(document) = document {
            selected.retain(|id| {
                document.matches(self.records.get(id).and_then(|r| r.document.as_deref()))
            });
        }
        selected
    }

    /// Map a `(id, distance)` map into scored points, pulling live metadata and
    /// document from the records map.
    fn build_scored(&self, dist: HashMap<Id, f32>, with_embedding: bool) -> Vec<ScoredPoint> {
        dist.into_iter()
            .map(|(id, d)| {
                let record = self.records.get(&id);
                let embedding = with_embedding.then(|| self.embedding_for(&id)).flatten();
                ScoredPoint {
                    id,
                    distance: d,
                    metadata: record.and_then(|r| r.metadata.clone()),
                    document: record.and_then(|r| r.document.clone()),
                    embedding,
                }
            })
            .collect()
    }
}

/// Whether a record's optional metadata satisfies the filter. A record with no
/// metadata is evaluated against an empty map, which is what
/// [`WhereFilter::matches`] itself does for `$nin` (absent key matches).
fn filter_matches_opt(filter: &WhereFilter, metadata: &Option<Metadata>) -> bool {
    match metadata {
        Some(m) => filter.matches(m),
        None => filter.matches(&Metadata::new()),
    }
}

/// Reject a metadata/document slice whose length does not match the id count.
fn check_slice_lengths<T>(
    op: &str,
    ids: &[Id],
    opt: Option<&[Option<T>]>,
    what: &str,
) -> EngineResult<()> {
    if let Some(slice) = opt
        && slice.len() != ids.len()
    {
        return Err(EngineError::Validation(format!(
            "{op}: {} ids but {} {what}",
            ids.len(),
            slice.len()
        )));
    }
    Ok(())
}

/// Reject any embedding whose length differs from the collection dimension.
/// If dimension is 0 (unset, ChromaDB auto-infer mode), skip validation.
pub(crate) fn check_dimensions(
    op: &str,
    embeddings: &[Embedding],
    dimension: usize,
) -> EngineResult<()> {
    for embedding in embeddings {
        if dimension == 0 {
            // The collection dimension is not fixed yet. The batch still has to
            // be internally consistent, otherwise inferring from the first
            // embedding would silently index a mixture of widths.
            if let Some(first) = embeddings.first()
                && embedding.len() != first.len()
            {
                return Err(EngineError::Validation(format!(
                    "{op}: mixed embedding dimensions in one batch ({} and {})",
                    first.len(),
                    embedding.len()
                )));
            }
            continue;
        }
        if embedding.len() != dimension {
            return Err(EngineError::Validation(format!(
                "{op}: embedding dimension {} != collection dimension {dimension}",
                embedding.len()
            )));
        }
    }
    Ok(())
}
