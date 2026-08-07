//! Per-collection state and the two-tier write pipeline.
//!
//! A [`Collection`] is the unit of concurrency: the engine holds one
//! `Arc<Mutex<Collection>>` per collection, and a single mutex serializes WAL
//! appends (the WAL is single-writer) and index access (usearch's search
//! mutates the shared C++ expansion state).

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use rekha_core::cluster::Epoch;
use rekha_core::config::CollectionConfig;
use rekha_core::filter::WhereFilter;
use rekha_core::op::Operation;
use rekha_core::types::{Distance, Document, Embedding, Id, Metadata};
use rekha_distance::distance;
use rekha_index::{Index, UsearchIndex};
use rekha_storage::Catalog;
use rekha_wal::{Wal, WalOptions};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::postings::Postings;
use crate::{EngineError, EngineResult, OptimizeStats, QueryOptions, Record, ScoredPoint};

/// In-memory state of one collection: the live records map, the brute-force
/// buffer, the committed HNSW index, the metadata postings, and the WAL they
/// are all derived from.
pub(crate) struct Collection {
    pub(crate) config: CollectionConfig,
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
    /// WAL seq covered by the last checkpoint (used by the auto-checkpoint
    /// trigger to bound WAL growth: checkpoint when the WAL has grown
    /// `sync_threshold` records past this watermark).
    last_checkpoint_seq: u64,
    /// Whether to fsync WAL writes (passed through to WAL reopen after checkpoint prune).
    fsync: bool,
    /// Mmap'd vector segment handles loaded on reopen.
    pub(crate) segments: Vec<crate::segment::Segment>,
    /// Maps record id → offset within the segment (for fast vector lookup).
    pub(crate) segment_index: HashMap<Id, usize>,
}

/// The JSON state file (`checkpoint.json`) at the head of a checkpoint
/// directory. Written **last** in [`Collection::checkpoint`] so its presence
/// means `records.bin` (and any `index.bin`) are the committed snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointMeta {
    /// Highest WAL seq materialized into the index when the checkpoint ran.
    flushed_seq: u64,
    /// Seq the WAL will resume at: `wal.last_seq() + 1` at checkpoint time.
    /// Reopen passes this as `WalOptions::start_seq`, so a fully-pruned
    /// (empty) WAL continues appending exactly where the checkpoint left off.
    next_seq: u64,
    /// `records.len()` at checkpoint time, validated on load.
    record_count: u64,
}

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
    Ok(())
}

impl Collection {
    /// Open a collection, preferring its checkpoint and falling back to a full
    /// WAL replay. The WAL is the source of truth: a valid checkpoint
    /// (`{wal_dir}/{id}.checkpoint/checkpoint.json` + `records.bin`) restores
    /// records + index + postings up to `flushed_seq`, then the WAL tail above
    /// it is replayed; a corrupt, partial, or missing checkpoint falls through
    /// to the full replay, never a panic.
    pub(crate) fn open(
        id: &Uuid,
        config: CollectionConfig,
        wal_dir: &Path,
        epoch: Epoch,
        fsync: bool,
    ) -> EngineResult<Self> {
        let checkpoint_dir = wal_dir.join(format!("{id}.checkpoint"));
        let mut coll = Self::new(id, config, wal_dir, epoch, fsync)?;
        if Self::load_checkpoint(&mut coll, &checkpoint_dir, epoch, fsync) {
            // Restored records, index, and postings; replay only the WAL tail
            // above the checkpoint's flushed_seq.
            coll.replay_after(coll.flushed_seq)?;
        } else {
            // No usable checkpoint: full replay from seq 1.
            coll.replay_after(0)?;
        }
        // Re-apply the flush policy so a reopened collection behaves like the
        // pre-crash one (its buffer is drained up to batch_size).
        coll.flush_if_needed()?;
        Ok(coll)
    }

    /// Fresh in-memory state with an open WAL and an empty HNSW index.
    fn new(
        id: &Uuid,
        config: CollectionConfig,
        wal_dir: &Path,
        epoch: Epoch,
        fsync: bool,
    ) -> EngineResult<Self> {
        let wal = Wal::open(
            wal_dir,
            id,
            epoch,
            WalOptions {
                fsync,
                start_seq: None,
            },
        )?;
        let index = UsearchIndex::with_hnsw(config.space, config.dimension, &config.hnsw)?;
        Ok(Self {
            config,
            wal,
            records: HashMap::new(),
            buffer: Vec::new(),
            index: Box::new(index),
            flushed_seq: 0,
            offsets: HashMap::new(),
            id_by_offset: HashMap::new(),
            next_offset: 0,
            postings: Postings::new(),
            checkpoint_dir: wal_dir.join(format!("{id}.checkpoint")),
            last_checkpoint_seq: 0,
            fsync,
            segments: Vec::new(),
            segment_index: HashMap::new(),
        })
    }

    /// Try to restore `coll` from a checkpoint directory. Returns `true` on a
    /// fully valid restore (and re-opens the WAL at the checkpoint's
    /// `next_seq`); `false` on any missing/corrupt file — the caller falls back
    /// to a full replay. Never panics on bad checkpoint data.
    fn load_checkpoint(coll: &mut Self, dir: &Path, epoch: Epoch, fsync: bool) -> bool {
        let meta: CheckpointMeta = match std::fs::read(dir.join("checkpoint.json")) {
            Ok(bytes) => match serde_json::from_slice(&bytes) {
                Ok(m) => m,
                Err(_) => return false,
            },
            Err(_) => return false,
        };
        let Ok(records_bytes) = std::fs::read(dir.join("records.bin")) else {
            return false;
        };
        let Ok(records): Result<HashMap<Id, Record>, _> = bincode::deserialize(&records_bytes)
        else {
            return false;
        };
        if records.len() as u64 != meta.record_count {
            return false;
        }

        // Rebuild records, offsets, and postings from the records map.
        coll.records = records;
        coll.offsets.clear();
        coll.id_by_offset.clear();
        coll.postings = Postings::new();
        let mut next_offset = 0u32;
        for (id, record) in coll.records.iter() {
            let offset = next_offset;
            coll.offsets.insert(id.clone(), offset);
            coll.id_by_offset.insert(offset, id.clone());
            coll.postings.insert(offset, record.metadata.as_ref());
            next_offset += 1;
        }
        coll.next_offset = next_offset;
        coll.flushed_seq = meta.flushed_seq;
        coll.last_checkpoint_seq = meta.flushed_seq;

        // Restore the HNSW index. A missing or corrupt index.bin does NOT
        // invalidate the checkpoint: once the WAL is pruned, records.bin is the
        // only copy of the records, so we degrade to an empty index rather than
        // abandon them. (The index is re-derivable from the live records on the
        // next flush/optimize; queries meanwhile fall back to the exact
        // brute-force path.)
        if let Ok(index) = UsearchIndex::load(
            &dir.join("index.bin"),
            coll.config.space,
            coll.config.dimension,
        ) {
            coll.index = Box::new(index);
        }

        // Load mmap'd segments if present.
        let seg_dir = dir.join("segments");
        if seg_dir.is_dir() {
            let mut segments = Vec::new();
            let mut seg_idx = 0u32;
            loop {
                let seg_path = seg_dir.join(format!("seg-{seg_idx}.bin"));
                if !seg_path.exists() {
                    break;
                }
                if let Ok(seg) = crate::segment::Segment::open(&seg_path) {
                    segments.push(seg);
                }
                seg_idx += 1;
            }
            coll.segments = segments;
        }
        // Load segment index if present.
        if let Ok(idx_bytes) = std::fs::read(dir.join("segment_index.bin")) {
            if let Ok(idx) = bincode::deserialize::<HashMap<Id, usize>>(&idx_bytes) {
                coll.segment_index = idx;
            }
        }

        // Strip embeddings from loaded records — segments are the
        // source of truth.  Only if segments are present (backward
        // compat with pre-4b checkpoints).
        if !coll.segments.is_empty() {
            for record in coll.records.values_mut() {
                if record.seq <= coll.flushed_seq {
                    record.embedding = None;
                }
            }
        }

        // Reopen the WAL resuming at the checkpoint's next_seq (a pruned WAL
        // is empty, and must continue appending where the checkpoint left off).
        let id = coll.config.id;
        let wal = match Wal::open(
            coll.checkpoint_dir.parent().expect("wal_dir"),
            &id,
            epoch,
            WalOptions {
                fsync,
                start_seq: Some(meta.next_seq),
            },
        ) {
            Ok(w) => w,
            Err(_) => return false,
        };
        coll.wal = wal;
        true
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

    /// Persist the collection's live records, HNSW index, and id maps to
    /// `{wal_dir}/{id}.checkpoint/` (temp-file + rename each), then prune the
    /// WAL to the un-checkpointed tail. The checkpoint is written before the
    /// prune, so a crash between the two only wastes a checkpoint.
    ///
    /// Returns early for a collection with no live records (nothing to save).
    pub(crate) fn checkpoint(&mut self, catalog: &dyn Catalog) -> EngineResult<()> {
        if self.records.is_empty() {
            return Ok(());
        }
        let next_seq = self.wal.last_seq() + 1;
        let meta = CheckpointMeta {
            flushed_seq: self.flushed_seq,
            next_seq,
            record_count: self.records.len() as u64,
        };

        std::fs::create_dir_all(&self.checkpoint_dir)?;
        // 1. Records (the durable, WAL-derivable truth).
        let records_bytes = bincode::serialize(&self.records)
            .map_err(|e| EngineError::Serialization(e.to_string()))?;
        atomic_write(&self.checkpoint_dir.join("records.bin"), &records_bytes)?;
        // 2. HNSW graph + id maps, only if there are vectors to save. `save`
        //    writes both `index.bin` and `index.meta`.
        if self.index.len() > 0 {
            self.index.save(&self.checkpoint_dir.join("index.bin"))?;
        } else {
            // Ensure stale index files from an older checkpoint don't survive a
            // later empty-index checkpoint.
            for stale in ["index.bin", "index.meta"] {
                let p = self.checkpoint_dir.join(stale);
                match std::fs::remove_file(&p) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(_) => {
                        return Err(EngineError::Io(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("removing stale checkpoint file `{}`", p.display()),
                        )));
                    }
                }
            }
        }
        // 3. The checkpoint marker last: its presence means 1+2 are the
        //    committed snapshot.
        atomic_write(
            &self.checkpoint_dir.join("checkpoint.json"),
            &serde_json::to_vec(&meta).map_err(|e| EngineError::Serialization(e.to_string()))?,
        )?;

        // --- Segment writing: persist vectors to mmap'd segment files ---
        {
            use crate::segment::SegmentWriter;
            let seg_dir = self.checkpoint_dir.join("segments");
            std::fs::create_dir_all(&seg_dir)?;

            // Collect all records that have embeddings, sorted by id for determinism.
            let mut entries: Vec<_> = self
                .records
                .iter()
                .filter_map(|(id, r)| r.embedding.as_ref().map(|e| (id, e.as_ref())))
                .collect();
            entries.sort_by_key(|(id, _)| (*id).clone());

            if !entries.is_empty() {
                let dim = self.config.dimension;
                let mut writer = SegmentWriter::new(dim);
                let mut idx: HashMap<Id, usize> = HashMap::new();
                for (i, (id, emb)) in entries.iter().enumerate() {
                    writer.push(emb);
                    idx.insert((*id).clone(), i);
                }
                let seg_path = seg_dir.join("seg-0.bin");
                writer.write(&seg_path)?;

                // Write the segment index for fast lookup on reopen.
                let idx_bytes = bincode::serialize(&idx)
                    .map_err(|e| EngineError::Serialization(e.to_string()))?;
                atomic_write(&self.checkpoint_dir.join("segment_index.bin"), &idx_bytes)?;

                // Load the just-written segment so in-memory reads can find
                // the vectors (self.segments / self.segment_index are the
                // source of truth for the current session).
                if let Ok(seg) = crate::segment::Segment::open(&seg_path) {
                    self.segments = vec![seg];
                    self.segment_index = idx;
                }
            } else {
                // No vectors: clean up any stale segment files.
                let _ = std::fs::remove_dir_all(&seg_dir);
                let _ = std::fs::remove_file(self.checkpoint_dir.join("segment_index.bin"));
                self.segments = Vec::new();
                self.segment_index = HashMap::new();
            }
        }

        // Strip embeddings from in-memory records — vectors are now in
        // segments.  Only strip records whose seq was covered by the
        // checkpoint (newer records are not yet in segments).
        for record in self.records.values_mut() {
            if record.seq <= meta.flushed_seq {
                record.embedding = None;
            }
        }

        // Persist the flushed watermark (forward-only in the catalog), then
        // drop the covered WAL prefix.
        catalog.advance_log_offset(&self.config.id, meta.flushed_seq)?;
        self.wal.prune(meta.flushed_seq)?;
        // A fully-pruned WAL is empty; reopen it resuming at next_seq so live
        // appends continue exactly where the checkpoint left off (same rule as
        // `load_checkpoint` on reopen).
        let epoch = Epoch(self.wal.epoch());
        let wal_dir = self.checkpoint_dir.parent().expect("wal_dir");
        self.wal = Wal::open(
            wal_dir,
            &self.config.id,
            epoch,
            WalOptions {
                fsync: self.fsync,
                start_seq: Some(next_seq),
            },
        )?;
        self.last_checkpoint_seq = meta.flushed_seq;
        Ok(())
    }

    /// Rebuild the HNSW index from the live records (used by `optimize` to
    /// reclaim space after deletes and by reopen to restore a postings set).
    pub(crate) fn rebuild_index(&mut self) -> EngineResult<()> {
        let index =
            UsearchIndex::with_hnsw(self.config.space, self.config.dimension, &self.config.hnsw)?;
        self.index = Box::new(index);
        let mut to_add: Vec<(Id, Embedding)> = self
            .records
            .iter()
            .filter_map(|(id, r)| {
                // Try segment first (stripped records), then record embedding.
                Self::load_vector_from(&self.segments, &self.segment_index, id)
                    .map(|v| (id.clone(), Embedding::from(v.to_vec())))
                    .or_else(|| r.embedding.as_ref().map(|e| (id.clone(), e.clone())))
            })
            .collect();
        to_add.sort_by(|a, b| a.0.cmp(&b.0));
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
        self.flush_buffer()?;
        let indexed = self.index.len() as u64;
        let wal_bytes_before = self.wal.size_bytes();
        self.rebuild_index()?;
        self.checkpoint(catalog)?;
        let wal_bytes_after = self.wal.size_bytes();
        Ok(OptimizeStats {
            records,
            indexed,
            wal_bytes_before,
            wal_bytes_after,
        })
    }

    /// Auto-checkpoint when the WAL has accumulated `sync_threshold` records
    /// past the last checkpoint watermark. Failures propagate as engine errors
    /// (the write already committed to the WAL; a checkpoint failure is
    /// reported but not rolled back).
    pub(crate) fn auto_checkpoint(&mut self, catalog: &dyn Catalog) -> EngineResult<()> {
        if self.records.is_empty() {
            return Ok(());
        }
        let threshold = self.config.hnsw.sync_threshold as u64;
        let new_records = self.wal.last_seq().saturating_sub(self.last_checkpoint_seq);
        if new_records >= threshold {
            self.checkpoint(catalog)?;
        }
        Ok(())
    }

    /// Commit a whole batch to the WAL first (each op gets its seq), then apply
    /// the ops to in-memory state. A WAL failure aborts the batch: nothing is
    /// applied to memory (the WAL may hold a prefix, which replay re-applies).
    pub(crate) fn write_ops(&mut self, ops: &[Operation]) -> EngineResult<()> {
        let mut appended = Vec::with_capacity(ops.len());
        for op in ops {
            appended.push(self.wal.append(op.clone())?);
        }
        for rec in &appended {
            self.apply_operation(&rec.op, rec.seq)?;
        }
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

    /// Load a vector from segments without borrowing self.
    fn load_vector_from<'a>(
        segments: &'a [crate::segment::Segment],
        segment_index: &'a HashMap<Id, usize>,
        id: &Id,
    ) -> Option<&'a [f32]> {
        let &offset = segment_index.get(id)?;
        segments.first()?.get_vector(offset)
    }

    /// Load a vector from mmap'd segments by record id.
    /// Returns a slice of `dimension` f32 values, or `None` if the id
    /// has no segment entry (e.g. added after the last checkpoint).
    pub(crate) fn load_vector(&self, id: &Id) -> Option<&[f32]> {
        let &offset = self.segment_index.get(id)?;
        self.segments.first()?.get_vector(offset)
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
        metadatas: Option<&[Option<Metadata>]>,
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
    pub(crate) fn validate_update(
        &self,
        ids: &[Id],
        metadatas: Option<&[Option<Metadata>]>,
        documents: Option<&[Option<Document>]>,
    ) -> EngineResult<()> {
        check_slice_lengths("update", ids, metadatas, "metadatas")?;
        check_slice_lengths("update", ids, documents, "documents")?;
        for id in ids {
            if !self.records.contains_key(id) {
                return Err(EngineError::Validation(format!(
                    "update: id `{id}` not found"
                )));
            }
        }
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
    ) -> EngineResult<Vec<ScoredPoint>> {
        use std::cmp::Ordering;

        if query.len() != self.config.dimension {
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

        let mut out = match &options.where_filter {
            None => {
                if self.index.len() + self.buffer.len() <= self.config.hnsw.max_scan {
                    self.brute_force(query, space)
                } else {
                    self.hybrid_search(query, k, ef, space)?
                }
            }
            Some(filter) => {
                self.filtered_query(query, k, ef, space, filter, options.oversampling)?
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

    /// Exact scan over every live record that has an embedding.
    fn brute_force(&self, query: &Embedding, space: Distance) -> Vec<ScoredPoint> {
        self.records
            .iter()
            .filter_map(|(id, r)| {
                let embedding = Self::load_vector_from(&self.segments, &self.segment_index, id)
                    .or_else(|| r.embedding.as_ref().map(|e| e.as_ref()))?;
                Some(ScoredPoint {
                    id: id.clone(),
                    distance: distance(space, query, embedding),
                    metadata: r.metadata.clone(),
                    document: r.document.clone(),
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
        k: usize,
        ef: usize,
        space: Distance,
    ) -> EngineResult<Vec<ScoredPoint>> {
        let hits = self.index.search(query, k, ef)?;
        let mut dist: HashMap<Id, f32> = HashMap::with_capacity(hits.len() + self.buffer.len());
        for hit in hits {
            dist.entry(hit.id).or_insert(hit.distance);
        }
        for record in &self.buffer {
            if let Some(embedding) = &record.embedding {
                dist.insert(record.id.clone(), distance(space, query, embedding));
            }
        }
        Ok(self.build_scored(dist))
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
        k: usize,
        ef: usize,
        space: Distance,
        filter: &WhereFilter,
        oversampling: usize,
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
            // Exact path: scan every eligible record's embedding directly.
            for offset in candidates.iter() {
                let Some(id) = self.id_by_offset.get(&offset) else {
                    continue;
                };
                let Some(record) = self.records.get(id) else {
                    continue;
                };
                let Some(embedding) = &record.embedding else {
                    continue;
                };
                dist.entry(record.id.clone())
                    .or_insert_with(|| distance(space, query, embedding));
            }
        } else {
            // ANN path: widen the beam, fetch k * oversampling candidates, and
            // keep only the ones eligible per the postings.
            let fetch = k.saturating_mul(oversampling.max(1));
            let hits = self.index.search(query, fetch, ef)?;
            for hit in hits {
                let Some(offset) = self.offsets.get(&hit.id) else {
                    continue;
                };
                if candidates.contains(*offset) {
                    dist.entry(hit.id).or_insert(hit.distance);
                }
            }
        }

        Ok(self.build_scored(dist))
    }

    /// Map a `(id, distance)` map into scored points, pulling live metadata and
    /// document from the records map.
    fn build_scored(&self, dist: HashMap<Id, f32>) -> Vec<ScoredPoint> {
        dist.into_iter()
            .map(|(id, d)| {
                let record = self.records.get(&id);
                ScoredPoint {
                    id,
                    distance: d,
                    metadata: record.and_then(|r| r.metadata.clone()),
                    document: record.and_then(|r| r.document.clone()),
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
    if let Some(slice) = opt {
        if slice.len() != ids.len() {
            return Err(EngineError::Validation(format!(
                "{op}: {} ids but {} {what}",
                ids.len(),
                slice.len()
            )));
        }
    }
    Ok(())
}

/// Reject any embedding whose length differs from the collection dimension.
/// If dimension is 0 (unset, ChromaDB auto-infer mode), skip validation.
fn check_dimensions(op: &str, embeddings: &[Embedding], dimension: usize) -> EngineResult<()> {
    if dimension == 0 {
        return Ok(());
    }
    for embedding in embeddings {
        if embedding.len() != dimension {
            return Err(EngineError::Validation(format!(
                "{op}: embedding dimension {} != collection dimension {dimension}",
                embedding.len()
            )));
        }
    }
    Ok(())
}
