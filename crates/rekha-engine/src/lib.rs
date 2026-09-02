//! RekhaDB `rekha-engine` crate — the single-node write/read engine.
//!
//! The engine is Chroma's local mode, mirrored. Each collection is a
//! per-collection state machine built on the lower-level `rekha-core`,
//! `rekha-wal`, `rekha-storage`, `rekha-index`, and `rekha-distance` crates.
//! All write/read API is synchronous; the distributed/cluster mode is a later
//! phase.
//!
//! # The two-tier write pipeline
//!
//! Every mutation is committed to the collection's WAL **first** and
//! acknowledged before any in-memory index is touched. The WAL is the **source
//! of truth**; all in-memory state — the live `records` map, the brute-force
//! `buffer`, and the HNSW `index` — is derived and rebuilt by replaying the log
//! on open.
//!
//! Committed vectors flow through two tiers, matching Chroma:
//!
//! 1. **Brute-force buffer.** New `Add`/`Upsert` vectors land in a per-collection
//!    in-memory buffer and are immediately queryable: the query path scans it
//!    exhaustively with `rekha_distance`.
//! 2. **HNSW index.** Once the buffer reaches `config.hnsw.batch_size` records
//!    after a write batch, it is drained into the HNSW index (`index.add` per
//!    record; an id already committed is deleted then re-added, covering
//!    upsert-over-flushed-vector).
//!
//! # Query strategy
//!
//! A query runs the index search **and** a brute-force scan of the buffer, then
//! merges by id — a buffered record is newer than the same id in the index, so
//! the buffer wins — sorts ascending by distance (lower = closer, Chroma
//! semantics), and takes the top-k. For small collections
//! (`index.len() + buffer.len() <= config.hnsw.max_scan`) the index is skipped
//! entirely and the query is a pure brute-force scan over all live records,
//! which is exact.
//!
//! # Metadata filtering (Phase 3)
//!
//! A `where` filter (`QueryOptions::where_filter`) restricts a query to records
//! whose metadata matches, Chroma-style: **filters decide eligibility, never
//! ranking**. Eligibility is resolved by [`Postings`], a per-collection inverted
//! index mapping `(metadata key, value) → bitmap of internal offsets`, rebuilt
//! from the WAL on reopen (it is derived state). The exact/approximate behavior
//! is chosen by a query planner:
//!
//! - **Exact path**: when the eligible set fits `config.hnsw.max_scan`, every
//!   eligible record's embedding is scanned directly (this is also the path for
//!   eligible records still sitting in the brute-force buffer).
//! - **ANN path**: otherwise the HNSW graph is walked with `ef` and
//!   `k * oversampling` candidates are fetched, then post-filtered to the
//!   eligible set. This is approximate: a selective filter with a small
//!   oversampling factor can miss true nearest neighbors that the graph walk
//!   does not surface (see `Collection::filtered_query`).
//!
//! # Durability contract
//!
//! [`Engine::open`] loads every collection from the catalog. Each collection
//! **checkpoints** (Phase 4a): periodically the live records map, the HNSW
//! graph, and the id↔label maps are persisted to `{wal_dir}/{id}.checkpoint/`
//! and the WAL is pruned to the un-checkpointed tail. On reopen the collection
//! loads the checkpoint (if valid) and replays only the WAL tail above the
//! checkpoint's `flushed_seq`; without a checkpoint it replays the full WAL
//! from seq 1. The WAL remains the **source of truth** — a checkpoint is only
//! an acceleration, so a corrupt or missing checkpoint degrades to a full
//! replay, never to a panic or data loss.
//!
//! [`Engine::optimize`] force-flushes the buffer, rebuilds the HNSW graph from
//! live records (reclaiming deleted-vector space, which usearch never reclaims
//! on its own), and checkpoints.
//!
//! # Durability contract (pre-Phase 4a note)
//!
//! Before checkpoints existed, the engine did not persist records or the index
//! beyond the WAL. That is still true in spirit — everything a checkpoint
//! stores is also derivable from the WAL, which is why the checkpoint is safe
//! to lose. Cluster/replication mode is Phase 6.

mod collection;
mod postings;
pub mod segment;

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use rekha_core::cluster::Epoch;
use rekha_core::config::CollectionConfig;
use rekha_core::filter::WhereFilter;
use rekha_core::op::Operation;
use rekha_core::types::{Document, Embedding, Id, Metadata};
use rekha_index::IndexError;
use rekha_storage::{Catalog, CatalogError, CollectionRecord, Storage, StorageError};
use rekha_wal::WalError;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use collection::Collection;

pub use postings::{Postings, ValueKey};

/// Query-time options. `where_filter == None` keeps the exact unfiltered query
/// behavior; `Some(filter)` restricts results to records whose metadata matches
/// (eligibility only — ranking is by distance alone).
#[derive(Debug, Clone)]
pub struct QueryOptions {
    /// `ef_search` beam width; `0` = use `HnswConfig::ef_search`.
    pub ef: usize,
    /// Metadata eligibility filter; `None` = unfiltered (current behavior).
    pub where_filter: Option<WhereFilter>,
    /// ANN overfetch multiplier for filtered search: fetch `k × oversampling`
    /// candidates from the graph, then post-filter to the eligible set.
    pub oversampling: usize,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            ef: 0,
            where_filter: None,
            oversampling: 4,
        }
    }
}

impl QueryOptions {
    /// Options with a metadata `where` filter and otherwise-default tuning.
    pub fn with_where(filter: WhereFilter) -> Self {
        Self {
            where_filter: Some(filter),
            ..Default::default()
        }
    }
}

/// Engine configuration. Single-node defaults: no fsync on every append and
/// epoch `Epoch(0)`.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Passed through to [`rekha_wal::WalOptions::fsync`]: `true` = fsync on
    /// every append (durable), `false` = rely on the OS (faster).
    pub wal_fsync: bool,
    /// Fencing token stamped on WAL files created by this engine. Single-node
    /// engines use `Epoch(0)`; cluster mode fences per leader generation.
    pub epoch: Epoch,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            wal_fsync: false,
            epoch: Epoch(0),
        }
    }
}

/// A live record: all metadata/document state plus the current embedding.
///
/// `embedding: None` marks a metadata-only record (e.g. created by an `update`
/// of an absent id during replay); `Add`/`Upsert` always carry an embedding,
/// `Update` leaves it untouched, and `Delete` removes the whole record.
///
/// Serde derives let a checkpoint serialize the whole records map to
/// `records.bin` (the embedding field uses `rekha_core`'s `embedding_serde`,
/// since `Arc<[f32]>` has no direct serde impl).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// Record id, unique within the collection.
    pub id: Id,
    /// Current embedding (`None` = metadata-only record).
    /// The record's current embedding, if it has one. `update` ops carry no
    /// vector, so this is `None` until an `add`/`upsert` gives it one.
    #[serde(
        serialize_with = "rekha_core::types::embedding_serde::serialize_opt",
        deserialize_with = "rekha_core::types::embedding_serde::deserialize_opt"
    )]
    pub embedding: Option<Embedding>,
    /// Current metadata, if any.
    pub metadata: Option<Metadata>,
    /// Current document, if any.
    pub document: Option<Document>,
    /// WAL seq of the operation that produced this record version.
    pub seq: u64,
}

/// One query result. Distances follow Chroma semantics: **lower = closer**
/// (L2 = squared Euclidean, IP = `1 - dot`, Cosine = `1 - cos_sim`).
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredPoint {
    /// Record id.
    pub id: Id,
    /// Distance from the query, lower is closer.
    pub distance: f32,
    /// Live metadata of the matched record, if any.
    pub metadata: Option<Metadata>,
    /// Live document of the matched record, if any.
    pub document: Option<Document>,
}

/// Result of [`Engine::optimize`]: what was materialized and how much the WAL
/// shrank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptimizeStats {
    /// Live records in the collection at optimize time.
    pub records: u64,
    /// Vectors materialized into the (rebuilt) HNSW index.
    pub indexed: u64,
    /// WAL file size in bytes before the optimize checkpoint.
    pub wal_bytes_before: u64,
    /// WAL file size in bytes after the checkpoint/prune.
    pub wal_bytes_after: u64,
}

/// Errors surfaced by [`Engine`].
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The underlying catalog (sysdb) failed.
    #[error("catalog: {0}")]
    Catalog(#[from] CatalogError),
    /// The underlying WAL failed.
    #[error("wal: {0}")]
    Wal(#[from] WalError),
    /// The underlying ANN index failed.
    #[error("index: {0}")]
    Index(#[from] IndexError),
    /// An I/O error occurred (checkpoint files, catalog storage files, ...).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// An object store operation failed.
    #[error("storage: {0}")]
    Storage(#[from] StorageError),
    /// A serialization/deserialization error occurred (checkpoint files).
    #[error("serialization: {0}")]
    Serialization(String),
    /// The operation referenced a collection the engine does not hold.
    #[error("collection `{0}` not found")]
    CollectionNotFound(String),
    /// The operation failed semantic validation (dimension mismatch, duplicate
    /// id, ...). The batch is rejected atomically and nothing is written.
    #[error("validation: {0}")]
    Validation(String),
}

/// Convenience alias for engine operations.
pub type EngineResult<T> = Result<T, EngineError>;

/// The single-node engine: collection lifecycle plus the write/read API.
///
/// Every public operation that touches a collection locks its
/// `Arc<Mutex<Collection>>`; a single mutex per collection serializes writes
/// (the WAL is single-writer) and index access (usearch's search mutates the
/// shared C++ expansion state).
pub struct Engine {
    catalog: Arc<dyn Catalog>,
    /// Object store; reserved for Phase 4b (vector blocks / mmap). Unused by
    /// the checkpoint-on-filesystem durability model of this phase.
    storage: Arc<dyn Storage>,
    /// Directory holding one `{collection_id}.wal` file plus one
    /// `{collection_id}.checkpoint/` directory per collection.
    wal_dir: PathBuf,
    config: EngineConfig,
    collections: RwLock<HashMap<Uuid, Arc<Mutex<Collection>>>>,
}

impl Engine {
    /// Open an engine over an existing (or empty) catalog, object store, and
    /// WAL directory.
    ///
    /// Every collection present in the catalog is loaded. Each collection first
    /// loads its checkpoint (`{wal_dir}/{id}.checkpoint/`) if one is valid —
    /// records + HNSW index + id maps — then replays only the WAL tail above
    /// the checkpoint's `flushed_seq`; collections without a valid checkpoint
    /// replay their full WAL from seq 1. The WAL remains the source of truth:
    /// a checkpoint is only an acceleration.
    pub fn open(
        catalog: Arc<dyn Catalog>,
        storage: Arc<dyn Storage>,
        wal_dir: impl AsRef<Path>,
        config: EngineConfig,
    ) -> EngineResult<Self> {
        let wal_dir = wal_dir.as_ref().to_path_buf();
        let mut collections = HashMap::new();
        for record in catalog.list_collections_all()? {
            let id = record.config.id;
            let collection =
                Collection::open(&id, record.config, &wal_dir, config.epoch, config.wal_fsync)?;
            collections.insert(id, Arc::new(Mutex::new(collection)));
        }
        Ok(Self {
            catalog,
            storage,
            wal_dir,
            config,
            collections: RwLock::new(collections),
        })
    }

    /// Create a collection: catalog entry, WAL file, and in-memory state.
    ///
    /// The returned record starts with `max_seq_id = 0` and `total_elements = 0`.
    pub fn create_collection(&self, config: &CollectionConfig) -> EngineResult<CollectionRecord> {
        let record = self.catalog.create_collection(config)?;
        let collection = Collection::open(
            &config.id,
            config.clone(),
            &self.wal_dir,
            self.config.epoch,
            self.config.wal_fsync,
        )?;
        let mut map = self.collections.write().unwrap();
        map.insert(config.id, Arc::new(Mutex::new(collection)));
        Ok(record)
    }

    /// Look up a collection by `(tenant, database, name)`.
    pub fn get_collection(
        &self,
        tenant: &str,
        database: &str,
        name: &str,
    ) -> EngineResult<Option<CollectionRecord>> {
        Ok(self.catalog.get_collection(tenant, database, name)?)
    }

    /// Look up a collection by its UUID.
    pub fn get_collection_by_id(&self, id: &Uuid) -> EngineResult<Option<CollectionRecord>> {
        Ok(self.catalog.get_collection_by_id(id)?)
    }

    /// List every collection in a `(tenant, database)` scope.
    pub fn list_collections(
        &self,
        tenant: &str,
        database: &str,
    ) -> EngineResult<Vec<CollectionRecord>> {
        Ok(self.catalog.list_collections(tenant, database)?)
    }

    /// Delete a collection: remove the catalog entry, drop the in-memory state,
    /// and remove its WAL file and checkpoint directory.
    ///
    /// Lock ordering: the map write lock is held only long enough to remove the
    /// entry; the collection mutex is never held while another collection lock
    /// is live.
    pub fn delete_collection(&self, id: &Uuid) -> EngineResult<()> {
        self.catalog.delete_collection(id)?;
        {
            let mut map = self.collections.write().unwrap();
            map.remove(id);
        }
        let wal_path = self.wal_dir.join(format!("{id}.wal"));
        match std::fs::remove_file(&wal_path) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(EngineError::Wal(e.into())),
        }
        let checkpoint_path = self.wal_dir.join(format!("{id}.checkpoint"));
        match std::fs::remove_dir_all(&checkpoint_path) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(EngineError::Io(e)),
        }
        Ok(())
    }

    /// Number of live records in the collection.
    pub fn count(&self, collection_id: &Uuid) -> EngineResult<u64> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock().unwrap();
        Ok(coll.records.len() as u64)
    }

    /// Add brand-new records. The whole batch is validated first (counts match,
    /// dimensions match, no duplicate ids) and rejected atomically on any
    /// failure — nothing is written.
    ///
    /// After the batch, an auto-checkpoint runs when the WAL has accumulated
    /// `config.hnsw.sync_threshold` records since the last checkpoint (best
    /// effort to keep the WAL bounded; a checkpoint failure is propagated but
    /// durability is unaffected — the WAL still holds the un-checkpointed tail).
    pub fn add(
        &self,
        collection_id: &Uuid,
        ids: &[Id],
        embeddings: &[Embedding],
        metadatas: Option<&[Option<Metadata>]>,
        documents: Option<&[Option<Document>]>,
    ) -> EngineResult<()> {
        let coll = self.collection(collection_id)?;
        let mut coll = coll.lock().unwrap();
        // ChromaDB compatibility: infer dimension from first embedding if unset.
        let inferred = if coll.config.dimension == 0 && !embeddings.is_empty() {
            coll.config.dimension = embeddings[0].len();
            Some(embeddings[0].len())
        } else {
            None
        };
        coll.validate_add(ids, embeddings, metadatas, documents)?;
        if let Some(dim) = inferred {
            // Persist inferred dimension so it survives restart.
            // Failure to persist is not fatal — the WAL replay will re-infer.
            if let Err(e) = self.catalog.update_collection_dimension(collection_id, dim) {
                tracing::warn!("Failed to persist inferred dimension {dim}: {e}");
            }
        }
        let ops = add_ops(ids, embeddings, metadatas, documents);
        coll.write_ops(&ops)?;
        coll.flush_if_needed()?;
        coll.auto_checkpoint(self.catalog.as_ref())
    }

    /// Insert-or-replace records by id: existing ids get their embedding,
    /// metadata, and document replaced. An id already flushed to the index is
    /// deleted and re-added when its new vector is flushed.
    pub fn upsert(
        &self,
        collection_id: &Uuid,
        ids: &[Id],
        embeddings: &[Embedding],
        metadatas: Option<&[Option<Metadata>]>,
        documents: Option<&[Option<Document>]>,
    ) -> EngineResult<()> {
        let coll = self.collection(collection_id)?;
        let mut coll = coll.lock().unwrap();
        // ChromaDB compatibility: infer dimension from first embedding if unset.
        let inferred = if coll.config.dimension == 0 && !embeddings.is_empty() {
            coll.config.dimension = embeddings[0].len();
            Some(embeddings[0].len())
        } else {
            None
        };
        coll.validate_upsert(ids, embeddings, metadatas, documents)?;
        if let Some(dim) = inferred
            && let Err(e) = self.catalog.update_collection_dimension(collection_id, dim)
        {
            tracing::warn!("Failed to persist inferred dimension {dim}: {e}");
        }
        let ops = upsert_ops(ids, embeddings, metadatas, documents);
        coll.write_ops(&ops)?;
        coll.flush_if_needed()?;
        coll.auto_checkpoint(self.catalog.as_ref())
    }

    /// Update the metadata/document of existing records. Every id must exist
    /// (else the batch is rejected); embeddings are unchanged.
    pub fn update(
        &self,
        collection_id: &Uuid,
        ids: &[Id],
        metadatas: Option<&[Option<Metadata>]>,
        documents: Option<&[Option<Document>]>,
    ) -> EngineResult<()> {
        let coll = self.collection(collection_id)?;
        let mut coll = coll.lock().unwrap();
        coll.validate_update(ids, metadatas, documents)?;
        let ops = update_ops(ids, metadatas, documents);
        coll.write_ops(&ops)?;
        coll.auto_checkpoint(self.catalog.as_ref())
    }

    /// Delete records by id. Missing ids are `Ok` (Chroma is lenient on
    /// delete): the record is removed from the map, the buffer, and the index.
    pub fn delete(&self, collection_id: &Uuid, ids: &[Id]) -> EngineResult<()> {
        let coll = self.collection(collection_id)?;
        let mut coll = coll.lock().unwrap();
        let ops: Vec<Operation> = ids
            .iter()
            .map(|id| Operation::Delete { id: id.clone() })
            .collect();
        coll.write_ops(&ops)?;
        coll.auto_checkpoint(self.catalog.as_ref())
    }

    /// Checkpoint one collection: persist its live records, HNSW index, and id
    /// maps to `{wal_dir}/{id}.checkpoint/`, then prune the WAL down to the
    /// un-checkpointed tail. A no-op for an empty collection. The WAL remains
    /// the source of truth — this only accelerates reopen.
    pub fn checkpoint(&self, collection_id: &Uuid) -> EngineResult<()> {
        let coll = self.collection(collection_id)?;
        let mut coll = coll.lock().unwrap();
        coll.checkpoint(self.catalog.as_ref())?;

        // Persist segment files to Storage.
        let seg_dir = coll.checkpoint_dir.join("segments");
        if seg_dir.is_dir() {
            for entry in std::fs::read_dir(&seg_dir)? {
                let entry = entry?;
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.ends_with(".bin") {
                    let key = format!("col/{}/segments/{}", collection_id, name_str);
                    let bytes = std::fs::read(entry.path())?;
                    self.storage.put(&key, &bytes)?;
                }
            }
        }

        Ok(())
    }

    /// Optimize a collection: force-flush the whole buffer into the index,
    /// rebuild the HNSW graph from live records (reclaiming tombstoned/deleted
    /// space — usearch never shrinks), then checkpoint. Returns stats about the
    /// materialized state and WAL size before/after.
    pub fn optimize(&self, collection_id: &Uuid) -> EngineResult<OptimizeStats> {
        let coll = self.collection(collection_id)?;
        let mut coll = coll.lock().unwrap();
        let stats = coll.optimize(self.catalog.as_ref())?;

        // Persist segment files to Storage.
        let seg_dir = coll.checkpoint_dir.join("segments");
        if seg_dir.is_dir() {
            for entry in std::fs::read_dir(&seg_dir)? {
                let entry = entry?;
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.ends_with(".bin") {
                    let key = format!("col/{}/segments/{}", collection_id, name_str);
                    let bytes = std::fs::read(entry.path())?;
                    self.storage.put(&key, &bytes)?;
                }
            }
        }

        Ok(stats)
    }

    /// Debug helper: the WAL seq of the record `id`, or a validation error if
    /// the record is unknown. Used by tests to assert seq continuity across
    /// checkpoints and prunes.
    pub fn seq_of(&self, collection_id: &Uuid, id: &Id) -> EngineResult<u64> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock().unwrap();
        coll.records
            .get(id)
            .map(|r| r.seq)
            .ok_or_else(|| EngineError::Validation(format!("id `{id}` not found")))
    }

    /// Fetch records by id; the result vector parallels `ids`, `None` for
    /// unknown ids.
    pub fn get(&self, collection_id: &Uuid, ids: &[Id]) -> EngineResult<Vec<Option<Record>>> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock().unwrap();
        Ok(ids
            .iter()
            .map(|id| {
                coll.records.get(id).map(|r| {
                    let embedding = coll
                        .load_vector(id)
                        .map(|v| Embedding::from(v.to_vec()))
                        .or_else(|| r.embedding.clone());
                    Record {
                        id: r.id.clone(),
                        embedding,
                        metadata: r.metadata.clone(),
                        document: r.document.clone(),
                        seq: r.seq,
                    }
                })
            })
            .collect())
    }

    /// k-NN search. `k` is clamped to `>= 1`; `ef` of `0` selects
    /// `config.hnsw.ef_search`. With `options.where_filter` set, only records
    /// whose metadata matches are candidates (eligibility, not ranking); the
    /// planner picks an exact brute-force scan when the eligible set fits
    /// `config.hnsw.max_scan`, and an approximate post-filtered index walk
    /// otherwise. For small collections
    /// (`index.len() + buffer.len() <= config.hnsw.max_scan`) an unfiltered
    /// query is an exact brute-force scan over all live records.
    pub fn query(
        &self,
        collection_id: &Uuid,
        embedding: &Embedding,
        k: usize,
        options: &QueryOptions,
    ) -> EngineResult<Vec<ScoredPoint>> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock().unwrap();
        coll.query(embedding, k, options)
    }

    /// Debug helper: live vectors tracked by the engine (`index.len() +
    /// buffer.len()`).
    pub fn indexed_count(&self, collection_id: &Uuid) -> EngineResult<usize> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock().unwrap();
        Ok(coll.index.len() + coll.buffer.len())
    }

    /// Debug helper: vectors materialized into the HNSW index (committed).
    pub fn committed_count(&self, collection_id: &Uuid) -> EngineResult<usize> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock().unwrap();
        Ok(coll.index.len())
    }

    /// Debug helper: highest WAL seq materialized into the index.
    pub fn flushed_seq(&self, collection_id: &Uuid) -> EngineResult<u64> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock().unwrap();
        Ok(coll.flushed_seq)
    }

    /// Read WAL records for a collection starting from `from_seq` and
    /// return them as a `WalDelta` for shipping to followers.
    pub fn wal_delta(
        &self,
        collection_id: &Uuid,
        from_seq: u64,
    ) -> EngineResult<rekha_cluster::WalDelta> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock().unwrap();
        let last = coll.wal.last_seq();
        if from_seq > last {
            return Ok(rekha_cluster::WalDelta {
                leader_node: 0,
                records: Vec::new(),
                target_seq: last,
            });
        }
        let records = coll.wal.read_range(from_seq, last)?;
        Ok(rekha_cluster::WalDelta::from_wal_records(0, records, last))
    }

    /// Get the last WAL seq for a collection (for status reporting).
    pub fn wal_last_seq(&self, collection_id: &Uuid) -> EngineResult<u64> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock().unwrap();
        Ok(coll.wal.last_seq())
    }

    /// Apply remote operations from a leader's WAL delta. These operations
    /// are NOT appended to the local WAL (they came from the leader's WAL).
    /// Used by followers to replay the leader's write stream.
    pub fn apply_remote_ops(
        &self,
        collection_id: &Uuid,
        ops: Vec<(u64, rekha_core::op::Operation)>,
    ) -> EngineResult<()> {
        let coll = self.collection(collection_id)?;
        let mut coll = coll.lock().unwrap();
        for (seq, op) in ops {
            coll.apply_operation(&op, seq)?;
        }
        Ok(())
    }

    /// Clone the collection handle, releasing the map read lock before any
    /// collection mutex is taken (lock ordering: never nest collection locks).
    fn collection(&self, id: &Uuid) -> EngineResult<Arc<Mutex<Collection>>> {
        let map = self.collections.read().unwrap();
        map.get(id)
            .cloned()
            .ok_or_else(|| EngineError::CollectionNotFound(id.to_string()))
    }

    /// Create a snapshot of all collection state for replication.
    pub fn snapshot(&self) -> EngineResult<Vec<u8>> {
        let collections = self
            .collections
            .read()
            .map_err(|e| EngineError::Serialization(format!("lock: {e}")))?;

        let mut snapshot_data: HashMap<Uuid, Vec<Record>> = HashMap::new();
        for (id, coll) in collections.iter() {
            let coll = coll
                .lock()
                .map_err(|e| EngineError::Serialization(format!("lock: {e}")))?;
            let records: Vec<Record> = coll.records.values().cloned().collect();
            snapshot_data.insert(*id, records);
        }

        bincode::serialize(&snapshot_data).map_err(|e| EngineError::Serialization(e.to_string()))
    }

    /// Restore from a snapshot (used by followers receiving a snapshot from leader).
    pub fn restore_snapshot(&self, data: &[u8]) -> EngineResult<()> {
        let snapshot_data: HashMap<Uuid, Vec<Record>> =
            bincode::deserialize(data).map_err(|e| EngineError::Serialization(e.to_string()))?;

        for (id, records) in snapshot_data {
            if let Ok(None) = self.get_collection(
                &self.config_or_default_tenant(),
                &self.config_or_default_database(),
                &id.to_string(),
            ) {
                continue;
            }

            if let Ok(coll_arc) = self.collection(&id)
                && let Ok(mut coll) = coll_arc.lock()
            {
                for record in records {
                    coll.records.insert(record.id.clone(), record.clone());
                }
            }
        }

        Ok(())
    }

    fn config_or_default_tenant(&self) -> String {
        "default_tenant".into()
    }

    fn config_or_default_database(&self) -> String {
        "default_database".into()
    }
}

impl rekha_cluster::EngineSnapshotProvider for Engine {
    fn snapshot(&self) -> Result<Vec<u8>, String> {
        Engine::snapshot(self).map_err(|e| e.to_string())
    }

    fn restore_snapshot(&self, data: &[u8]) -> Result<(), String> {
        Engine::restore_snapshot(self, data).map_err(|e| e.to_string())
    }
}

fn add_ops(
    ids: &[Id],
    embeddings: &[Embedding],
    metadatas: Option<&[Option<Metadata>]>,
    documents: Option<&[Option<Document>]>,
) -> Vec<Operation> {
    ids.iter()
        .enumerate()
        .map(|(i, id)| Operation::Add {
            id: id.clone(),
            embedding: embeddings[i].clone(),
            metadata: metadatas.map(|m| m[i].clone()).unwrap_or(None),
            document: documents.map(|d| d[i].clone()).unwrap_or(None),
        })
        .collect()
}

fn upsert_ops(
    ids: &[Id],
    embeddings: &[Embedding],
    metadatas: Option<&[Option<Metadata>]>,
    documents: Option<&[Option<Document>]>,
) -> Vec<Operation> {
    ids.iter()
        .enumerate()
        .map(|(i, id)| Operation::Upsert {
            id: id.clone(),
            embedding: embeddings[i].clone(),
            metadata: metadatas.map(|m| m[i].clone()).unwrap_or(None),
            document: documents.map(|d| d[i].clone()).unwrap_or(None),
        })
        .collect()
}

fn update_ops(
    ids: &[Id],
    metadatas: Option<&[Option<Metadata>]>,
    documents: Option<&[Option<Document>]>,
) -> Vec<Operation> {
    ids.iter()
        .enumerate()
        .map(|(i, id)| Operation::Update {
            id: id.clone(),
            metadata: metadatas.map(|m| m[i].clone()).unwrap_or(None),
            document: documents.map(|d| d[i].clone()).unwrap_or(None),
        })
        .collect()
}
