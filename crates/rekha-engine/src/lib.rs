#![deny(clippy::unwrap_used, clippy::expect_used)]
// Every `unsafe` block must carry a `// SAFETY:` justification. The engine has
// exactly one, in `segment.rs`.
#![deny(clippy::undocumented_unsafe_blocks)]

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
mod snapshot;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use rekha_core::cluster::Epoch;
use rekha_core::config::CollectionConfig;
use rekha_core::filter::{WhereDocument, WhereFilter};
use rekha_core::op::Operation;
use rekha_core::types::{Document, Embedding, Id, Metadata};
use rekha_index::IndexError;
use rekha_storage::{Catalog, CatalogError, CollectionRecord, Storage, StorageError};
use rekha_wal::WalError;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use collection::Collection;

pub mod backup;
pub mod fault;
mod metrics;
pub mod verify;
pub use postings::{Postings, ValueKey};
// Re-exported so API layers can build update batches without depending on
// `rekha-core` directly.
pub use rekha_core::types::{MetadataPatch, apply_metadata_patch};

/// Query-time options. `where_filter == None` keeps the exact unfiltered query
/// behavior; `Some(filter)` restricts results to records whose metadata matches
/// (eligibility only — ranking is by distance alone).
#[derive(Debug, Clone)]
pub struct QueryOptions {
    /// `ef_search` beam width; `0` = use `HnswConfig::ef_search`.
    pub ef: usize,
    /// Metadata eligibility filter; `None` = unfiltered (current behavior).
    pub where_filter: Option<WhereFilter>,
    /// Document-text eligibility filter; `None` = unfiltered.
    ///
    /// Documents are not indexed, so this always costs an exact scan. See
    /// [`Collection::query`](crate::Engine) for why it is a scan and not an
    /// over-fetched ANN post-filter.
    pub where_document: Option<WhereDocument>,
    /// ANN overfetch multiplier for filtered search: fetch `k × oversampling`
    /// candidates from the graph, then post-filter to the eligible set.
    pub oversampling: usize,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            ef: 0,
            where_filter: None,
            where_document: None,
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

/// How much a successful write promises about surviving a crash.
///
/// This is the contract the API reports as success, so it is stated explicitly
/// rather than inferred from a `bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    /// Acknowledge a write only after its WAL batch is fsynced.
    ///
    /// A successful response means the write survives a process crash and an
    /// OS crash. This is the server default: without it, `POST /add -> 200`
    /// can be followed by `kill -9` and the write is simply gone, with nothing
    /// in the response saying so.
    Durable,
    /// Acknowledge once the write reaches the operating system.
    ///
    /// Faster, and still consistent after a process crash, but a power loss or
    /// kernel panic can lose recently acknowledged writes. Selecting this is an
    /// explicit, logged decision.
    Relaxed,
}

impl Durability {
    /// Whether WAL appends are fsynced before being acknowledged.
    pub fn is_durable(self) -> bool {
        matches!(self, Durability::Durable)
    }

    /// Stable name used in configuration and diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Durability::Durable => "durable",
            Durability::Relaxed => "relaxed",
        }
    }
}

impl std::str::FromStr for Durability {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "durable" => Ok(Durability::Durable),
            "relaxed" => Ok(Durability::Relaxed),
            other => Err(format!(
                "unknown durability `{other}` (expected `durable` or `relaxed`)"
            )),
        }
    }
}

impl std::fmt::Display for Durability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which catalog rows this engine opens.
///
/// An enum rather than an id plus an "empty means everything" list. The two
/// states are genuinely different, and "empty" was one typo away from a cluster
/// node treating every row on disk — including rows it holds no data for — as
/// its own, which reads as an empty collection rather than as an error.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum EngineScope {
    /// Single node: every collection row on disk belongs to this engine.
    #[default]
    All,
    /// Cluster: only collections placed on these shards are opened here.
    Shards(Vec<rekha_core::cluster::ShardId>),
}

impl EngineScope {
    /// Whether a row placed on `shard` is this engine's data.
    ///
    /// An unsharded row inside a cluster is refused: it means the row was
    /// written by a single-node build and nothing knows which nodes hold it.
    pub fn owns(&self, shard: Option<rekha_core::cluster::ShardId>) -> bool {
        match (self, shard) {
            (Self::All, _) => true,
            (Self::Shards(_), None) => false,
            (Self::Shards(hosted), Some(shard)) => hosted.contains(&shard),
        }
    }
}

/// Engine configuration. Single-node defaults: durable writes and `Epoch(0)`.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// This node's id, or `0` for a single-node engine. Diagnostics only:
    /// ownership is decided by [`EngineConfig::scope`], not by comparing ids.
    pub node_id: u64,
    /// Which collection rows this engine opens. See [`EngineScope`].
    pub scope: EngineScope,
    /// What a successful write promises about surviving a crash.
    pub durability: Durability,
    /// Fencing token stamped on WAL files created by this engine. Single-node
    /// engines use `Epoch(0)`; cluster mode fences per leader generation.
    pub epoch: Epoch,
    /// Where durability comes from.
    pub apply_mode: ApplyMode,
}

/// Where a write's durability comes from.
///
/// Getting this wrong is silent data loss in one direction and a doubled fsync
/// in the other, so it is a declared property of the engine rather than a
/// convention at each call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApplyMode {
    /// This node is the durable record. Writes append to the WAL and honour
    /// [`Durability`]. Single-node mode.
    #[default]
    Durable,
    /// Something upstream already made the write durable — in cluster mode the
    /// Raft log, which is fsynced before the entry is committed. The engine
    /// applies in memory only: no WAL append and no fsync, because writing it
    /// twice buys nothing and costs a second fsync per commit.
    Committed,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            node_id: 0,
            scope: EngineScope::All,
            // Durable by default: a database that acknowledges writes it may
            // lose is the wrong default, even though it is faster.
            durability: Durability::Durable,
            epoch: Epoch(0),
            apply_mode: ApplyMode::Durable,
        }
    }
}

/// Build a snapshot archive from explicit `(catalog row, generation dir)` pairs.
///
/// Not part of the public API: exposed so integration tests can construct an
/// archive whose framing is valid but whose *contents* are not, which is the
/// only way to exercise the second validation layer (the checkpoint loader)
/// independently of the first (the archive CRC).
#[doc(hidden)]
pub fn encode_snapshot_archive_for_test(
    state: &[u8],
    collections: Vec<(rekha_storage::CollectionRecord, std::path::PathBuf)>,
) -> EngineResult<Vec<u8>> {
    crate::snapshot::encode(state, collections)
}

/// The operations a write produces, ready to be proposed to the log.
///
/// `dimension` is the effective vector width after inference. A follower that
/// has not fixed its width yet adopts this one, so the index is built at the
/// same width everywhere.
#[derive(Debug, Clone)]
pub struct PreparedBatch {
    pub ops: Vec<Operation>,
    pub dimension: usize,
}

/// What a restore did, for logging and for endpoint responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreReport {
    /// Collections installed from the archive.
    pub collections: usize,
    /// Files copied out of the archive.
    pub files: usize,
    /// Archive size in bytes.
    pub bytes: u64,
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredPoint {
    /// Record id.
    pub id: Id,
    /// Distance from the query, lower is closer.
    pub distance: f32,
    /// Live metadata of the matched record, if any.
    pub metadata: Option<Metadata>,
    /// Live document of the matched record, if any.
    pub document: Option<Document>,
    /// The record's vector, captured **in the same read view** as the distance.
    ///
    /// Only populated when the caller asked for embeddings. Fetching it in a
    /// second call after the lock was released allowed a concurrent write to
    /// pair a distance from one vector version with an embedding from another.
    #[serde(
        default,
        serialize_with = "rekha_core::types::embedding_serde::serialize_opt",
        deserialize_with = "rekha_core::types::embedding_serde::deserialize_opt"
    )]
    pub embedding: Option<Embedding>,
}

/// Outcome of a checkpoint publication.
///
/// `Err` from [`Engine::checkpoint`] means **nothing was published** and live
/// state is unchanged. A checkpoint that became durable but could not finish its
/// cleanup is a *success* with a deferred stage, not an error: reporting it as
/// one would tell the caller the opposite of what happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointOutcome {
    /// Published, activated, WAL pruned, superseded generations reclaimed.
    Complete,
    /// Published and activated; the named cleanup stage is still pending.
    Published { deferred: &'static str },
}

impl CheckpointOutcome {
    /// The cleanup stage that is still pending, if any.
    pub fn deferred_stage(self) -> Option<&'static str> {
        match self {
            CheckpointOutcome::Complete => None,
            CheckpointOutcome::Published { deferred } => Some(deferred),
        }
    }
}

/// Where a checkpoint spent its time, in microseconds.
///
/// Checkpoint cost is dominated by whole-corpus work (`records_us` serializes
/// every record, `vectors_us` rewrites every vector, `index_us` saves the whole
/// graph). Exposing the split is what makes "make checkpoints incremental"
/// targetable instead of a guess.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CheckpointTimings {
    /// Serialize and write `records.bin`.
    pub records_us: u64,
    /// Save the (derived) ANN graph.
    pub index_us: u64,
    /// Collect, write, link, and open the vector segments.
    pub vectors_us: u64,
    /// Serialize and fsync the id-to-offset map.
    pub index_map_us: u64,
    /// Write `order.bin`, the manifest, and fsync the generation directory.
    pub manifest_us: u64,
    /// Validate the staged generation before it becomes authoritative.
    pub validate_us: u64,
    /// Swap `CURRENT` and activate the new generation in memory.
    pub publish_us: u64,
    /// Advance the catalog watermark, prune the WAL, reopen it, and GC old
    /// generations.
    pub cleanup_us: u64,
    /// Whole checkpoint wall time, including staging directory setup.
    pub total_us: u64,
}

impl CheckpointTimings {
    /// The phase breakdown as `(name, microseconds)`, in execution order.
    pub fn phases(&self) -> [(&'static str, u64); 8] {
        [
            ("records", self.records_us),
            ("index", self.index_us),
            ("vectors", self.vectors_us),
            ("index_map", self.index_map_us),
            ("manifest", self.manifest_us),
            ("validate", self.validate_us),
            ("publish", self.publish_us),
            ("cleanup", self.cleanup_us),
        ]
    }
}

/// Result of [`Engine::optimize`]: what was materialized, how much the WAL
/// shrank, and where the time went.
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
    /// Force-flushing the pending buffer into the index.
    pub flush_us: u64,
    /// Rebuilding the graph from live records.
    pub rebuild_us: u64,
    /// The checkpoint phase breakdown.
    pub checkpoint: CheckpointTimings,
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
    /// This node is not one of the collection's replicas.
    ///
    /// Kept distinct from [`EngineError::CollectionNotFound`] on purpose: with
    /// sharding, "I am not a replica" is a routing mistake, while "no such
    /// collection" is a client mistake. Collapsing them would let a broken
    /// route answer an empty result set, which is silent and wrong.
    #[error("collection `{collection}` is not replicated on this node (shard: {shard:?})")]
    NotReplica {
        /// Collection that was requested.
        collection: String,
        /// The shard it was placed on. The nodes holding it are that shard's
        /// owners, which are derivable from placement.
        shard: Option<rekha_core::cluster::ShardId>,
    },
    /// The answer exists but this node cannot produce it right now — the
    /// replica that holds it is unreachable, or the group has no leader.
    ///
    /// Distinct from [`EngineError::Recovery`] because retrying against another
    /// node is safe and expected; distinct from a zero because a client that
    /// asked "how many" must not be told "none".
    #[error("unavailable: {0}")]
    Unavailable(String),
    /// The operation failed semantic validation (dimension mismatch, duplicate
    /// id, ...). The batch is rejected atomically and nothing is written.
    #[error("validation: {0}")]
    Validation(String),
    /// The caller lacks permission to perform the operation.
    #[error("forbidden: {0}")]
    Forbidden(String),
    /// Recovery could not reconstruct the committed state. Retrying is not
    /// safe: the operator must restore from backup or run an explicit rebuild.
    #[error("recovery: {0}")]
    Recovery(String),
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
    collections: RwLock<HashMap<PartitionKey, Arc<Mutex<Collection>>>>,
}

/// Key identifying a partition of a collection on a specific shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartitionKey {
    pub id: Uuid,
    pub shard: Option<rekha_core::cluster::ShardId>,
}

impl PartitionKey {
    pub fn new(id: Uuid, shard: Option<rekha_core::cluster::ShardId>) -> Self {
        Self { id, shard }
    }
}

impl Engine {
    /// Access the underlying catalog (for auth lookups).
    pub fn catalog(&self) -> Arc<dyn Catalog> {
        self.catalog.clone()
    }

    /// What a successful write currently promises about surviving a crash.
    pub fn durability(&self) -> Durability {
        self.config.durability
    }

    /// Phase timings of the collection's most recent checkpoint.
    ///
    /// `optimize` reports its own breakdown, but routine checkpoints (the ones
    /// that actually run in production) are invisible otherwise.
    pub fn last_checkpoint_timings(&self, collection_id: &Uuid) -> EngineResult<CheckpointTimings> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock();
        Ok(coll.last_checkpoint_timings)
    }

    /// Total WAL `fsync` calls issued by this process across all collections.
    ///
    /// One per committed batch in [`Durability::Durable`]; zero in
    /// [`Durability::Relaxed`]. Used by benchmarks to prove that a batching
    /// change actually reduced the number of syncs rather than assuming it.
    pub fn sync_count(&self) -> u64 {
        rekha_wal::sync_count()
    }

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
            match &config.scope {
                EngineScope::All => {
                    let key = PartitionKey::new(id, record.shard);
                    let collection = Collection::open(
                        &id,
                        record.shard,
                        record.config,
                        &wal_dir,
                        config.epoch,
                        config.durability.is_durable(),
                        record.max_seq_id,
                    )?;
                    collections.insert(key, Arc::new(Mutex::new(collection)));
                }
                EngineScope::Shards(hosted) => {
                    if let Some(shard) = record.shard {
                        if hosted.contains(&shard) {
                            let key = PartitionKey::new(id, Some(shard));
                            let collection = Collection::open(
                                &id,
                                Some(shard),
                                record.config,
                                &wal_dir,
                                config.epoch,
                                config.durability.is_durable(),
                                record.max_seq_id,
                            )?;
                            collections.insert(key, Arc::new(Mutex::new(collection)));
                        }
                    } else {
                        // Intra-collection sharded across all shards: open partition for each hosted shard
                        for &shard in hosted {
                            let key = PartitionKey::new(id, Some(shard));
                            let collection = Collection::open(
                                &id,
                                Some(shard),
                                record.config.clone(),
                                &wal_dir,
                                config.epoch,
                                config.durability.is_durable(),
                                record.max_seq_id,
                            )?;
                            collections.insert(key, Arc::new(Mutex::new(collection)));
                        }
                    }
                }
            }
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
        let record = self.record_collection(config, None)?;
        self.materialize_collection(&config.id)?;
        Ok(record)
    }

    /// Record a collection's metadata without opening its data.
    ///
    /// Every node needs this row: it is how a node answers "does this collection
    /// exist" for authorization and routing even when it holds none of the
    /// vectors. Idempotent, because replicated entries are applied again after a
    /// restart and on every node.
    pub fn record_collection(
        &self,
        config: &CollectionConfig,
        shard: Option<rekha_core::cluster::ShardId>,
    ) -> EngineResult<CollectionRecord> {
        Ok(self.catalog.record_collection(config, shard)?)
    }

    /// Open a recorded collection's data on this node.
    ///
    /// Idempotent: already-open is a no-op. Reads the catalog row rather than
    /// taking a config so a replica materializes exactly what was recorded.
    pub fn materialize_collection(&self, id: &Uuid) -> EngineResult<()> {
        match &self.config.scope {
            EngineScope::All => self.materialize_collection_partition(id, None),
            EngineScope::Shards(hosted) => {
                let record = self
                    .catalog
                    .get_collection_by_id(id)?
                    .ok_or_else(|| EngineError::CollectionNotFound(id.to_string()))?;
                if let Some(shard) = record.shard {
                    if hosted.contains(&shard) {
                        self.materialize_collection_partition(id, Some(shard))?;
                    }
                } else {
                    for &shard in hosted {
                        self.materialize_collection_partition(id, Some(shard))?;
                    }
                }
                Ok(())
            }
        }
    }

    /// Open a specific partition of a recorded collection on this node.
    pub fn materialize_collection_partition(
        &self,
        id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
    ) -> EngineResult<()> {
        let key = PartitionKey::new(*id, shard);
        {
            let map = self.collections.read();
            if map.contains_key(&key) {
                return Ok(());
            }
        }
        let record = self
            .catalog
            .get_collection_by_id(id)?
            .ok_or_else(|| EngineError::CollectionNotFound(id.to_string()))?;
        let mut map = self.collections.write();
        if map.contains_key(&key) {
            return Ok(());
        }
        let collection = Collection::open(
            id,
            shard,
            record.config,
            &self.wal_dir,
            self.config.epoch,
            self.config.durability.is_durable(),
            record.max_seq_id,
        )?;
        map.insert(key, Arc::new(Mutex::new(collection)));
        Ok(())
    }

    /// Drop a specific partition from memory and remove its files.
    pub fn drop_collection_partition(
        &self,
        id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
    ) -> EngineResult<()> {
        let key = PartitionKey::new(*id, shard);
        {
            let mut map = self.collections.write();
            map.remove(&key);
        }
        let stem = match shard {
            Some(s) => format!("{id}.shard-{}", s.0),
            None => id.to_string(),
        };
        let wal_path = self.wal_dir.join(format!("{stem}.wal"));
        let _ = std::fs::remove_file(&wal_path);
        let checkpoint_path = self.wal_dir.join(format!("{stem}.checkpoint"));
        let _ = std::fs::remove_dir_all(&checkpoint_path);
        Ok(())
    }

    /// Materialize all collection partitions for a newly hosted shard.
    pub fn materialize_shard(&self, shard: rekha_core::cluster::ShardId) -> EngineResult<()> {
        for record in self.catalog.list_collections_all()? {
            self.materialize_collection_partition(&record.config.id, Some(shard))?;
        }
        Ok(())
    }

    /// Drop all collection partitions for a decommissioned or moved shard.
    pub fn drop_shard(&self, shard: rekha_core::cluster::ShardId) -> EngineResult<()> {
        for record in self.catalog.list_collections_all()? {
            self.drop_collection_partition(&record.config.id, Some(shard))?;
        }
        Ok(())
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
        // Idempotent: a replayed removal, or a delete of a collection that was
        // never materialized on this node, still has a catalog row to clear.
        match self.catalog.delete_collection(id) {
            Ok(()) | Err(CatalogError::NotFound) => {}
            Err(e) => return Err(e.into()),
        }
        let to_drop: Vec<PartitionKey> = {
            let map = self.collections.read();
            map.keys().filter(|k| k.id == *id).copied().collect()
        };
        for key in to_drop {
            self.drop_collection_partition(&key.id, key.shard)?;
        }
        let _ = self.drop_collection_partition(id, None);
        Ok(())
    }

    /// Number of live records in the collection across all local partitions.
    pub fn count(&self, collection_id: &Uuid) -> EngineResult<u64> {
        let partitions = self.collection_partitions(collection_id);
        if partitions.is_empty() {
            let coll = self.collection(collection_id)?;
            return Ok(coll.lock().records.len() as u64);
        }
        let mut total = 0u64;
        for (_, p) in partitions {
            total += p.lock().records.len() as u64;
        }
        Ok(total)
    }

    /// Number of live records in a specific partition.
    pub fn count_partition(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
    ) -> EngineResult<u64> {
        let coll = self.collection_partition(collection_id, shard)?;
        Ok(coll.lock().records.len() as u64)
    }

    /// Number of live records, or `None` when this node is not a replica.
    ///
    /// The distinction matters for metadata responses: a node that holds no
    /// copy cannot count, and answering `0` there is a wrong answer that looks
    /// like an empty collection. Callers that need a number must route instead.
    pub fn count_local(&self, collection_id: &Uuid) -> EngineResult<Option<u64>> {
        let partitions = self.collection_partitions(collection_id);
        if partitions.is_empty() {
            let Some(record) = self.catalog.get_collection_by_id(collection_id)? else {
                return Err(EngineError::CollectionNotFound(collection_id.to_string()));
            };
            if !self.config.scope.owns(record.shard) {
                return Ok(None);
            }
            return Ok(Some(0));
        }
        let mut total = 0u64;
        for (_, part) in partitions {
            total += part.lock().records.len() as u64;
        }
        Ok(Some(total))
    }

    /// Add brand-new records. The whole batch is validated first (counts match,
    /// dimensions match, no duplicate ids) and rejected atomically on any
    /// failure — nothing is written.
    ///
    /// After the batch, an auto-checkpoint runs when the WAL has accumulated
    /// `config.hnsw.sync_threshold` records since the last checkpoint (best
    /// effort to keep the WAL bounded; a checkpoint failure is logged and the
    /// WAL tail is retained — durability is unaffected and the write still
    /// reports success).
    pub fn add(
        &self,
        collection_id: &Uuid,
        ids: &[Id],
        embeddings: &[Embedding],
        metadatas: Option<&[Option<Metadata>]>,
        documents: Option<&[Option<Document>]>,
    ) -> EngineResult<()> {
        let coll = self.collection(collection_id)?;
        let mut coll = coll.lock();
        // Validate the whole batch before touching config: a rejected first
        // write must not leave the collection with a dimension it never
        // committed (which would silently reject every later write of the real
        // width).
        coll.validate_add(ids, embeddings, metadatas, documents)?;
        // ChromaDB compatibility: infer dimension from first embedding if unset.
        // This also re-creates the index at the inferred width (see
        // `Collection::adopt_dimension`).
        if coll.config.dimension == 0
            && let Some(first) = embeddings.first()
        {
            let dim = first.len();
            coll.adopt_dimension(dim)?;
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
    /// Metadata is **merged** into an existing record (Chroma semantics), not
    /// replaced: see [`MetadataPatch`]. Fields that are not mentioned keep their
    /// current value.
    pub fn upsert(
        &self,
        collection_id: &Uuid,
        ids: &[Id],
        embeddings: &[Embedding],
        metadatas: Option<&[Option<MetadataPatch>]>,
        documents: Option<&[Option<Document>]>,
    ) -> EngineResult<()> {
        let coll = self.collection(collection_id)?;
        let mut coll = coll.lock();
        // Validate before inferring so a rejected batch cannot pin the
        // collection's dimension (see `Engine::add`).
        coll.validate_upsert(ids, embeddings, metadatas, documents)?;
        // ChromaDB compatibility: infer dimension from first embedding if unset.
        if coll.config.dimension == 0
            && let Some(first) = embeddings.first()
        {
            let dim = first.len();
            coll.adopt_dimension(dim)?;
            if let Err(e) = self.catalog.update_collection_dimension(collection_id, dim) {
                tracing::warn!("Failed to persist inferred dimension {dim}: {e}");
            }
        }
        let ops = resolve_upsert_ops(&coll, ids, embeddings, metadatas, documents);
        coll.write_ops(&ops)?;
        coll.flush_if_needed()?;
        coll.auto_checkpoint(self.catalog.as_ref())
    }

    /// Update the metadata/document of existing records. Every id must exist
    /// (else the batch is rejected); embeddings are unchanged unless supplied.
    ///
    /// When `embeddings` is `Some`, each record is written as an upsert carrying
    /// the new vector plus the metadata/document given here (Chroma allows
    /// updating a vector and its metadata in one call). Without it, a plain
    /// metadata update is appended.
    /// Every field that is not mentioned keeps its current value, and metadata
    /// is merged rather than replaced (Chroma semantics: see
    /// [`MetadataPatch`]). An id that does not exist is a **silent no-op** —
    /// Chroma answers 200 and changes nothing — rather than a rejected batch.
    ///
    /// The resulting complete values are resolved here, under the collection
    /// lock, and only then encoded into the WAL. Replay therefore still sees
    /// fully-materialized fields and existing log records keep their meaning.
    pub fn update(
        &self,
        collection_id: &Uuid,
        ids: &[Id],
        embeddings: Option<&[Embedding]>,
        metadatas: Option<&[Option<MetadataPatch>]>,
        documents: Option<&[Option<Document>]>,
    ) -> EngineResult<()> {
        let coll = self.collection(collection_id)?;
        let mut coll = coll.lock();
        if let Some(embeddings) = embeddings
            && embeddings.len() != ids.len()
        {
            return Err(EngineError::Validation(format!(
                "update: {} ids but {} embeddings",
                ids.len(),
                embeddings.len()
            )));
        }
        coll.validate_update(ids, metadatas, documents)?;
        if let Some(embeddings) = embeddings {
            crate::collection::check_dimensions("update", embeddings, coll.config.dimension)?;
        }
        let ops = resolve_update_ops(&coll, ids, embeddings, metadatas, documents);
        coll.write_ops(&ops)?;
        // A vector update lands in the buffer like any other write; without
        // this it never reached the index.
        coll.flush_if_needed()?;
        coll.auto_checkpoint(self.catalog.as_ref())
    }

    /// Delete records by id. Missing ids are `Ok` (Chroma is lenient on
    /// delete): the record is removed from the map, the buffer, and the index.
    pub fn delete(&self, collection_id: &Uuid, ids: &[Id]) -> EngineResult<()> {
        let coll = self.collection(collection_id)?;
        let mut coll = coll.lock();
        let ops: Vec<Operation> = ids
            .iter()
            .map(|id| Operation::Delete { id: id.clone() })
            .collect();
        coll.write_ops(&ops)?;
        coll.auto_checkpoint(self.catalog.as_ref())
    }

    // -----------------------------------------------------------------------
    // Cluster write path: validate and build here, apply somewhere else.
    // -----------------------------------------------------------------------
    //
    // In cluster mode the leader validates a batch and proposes the *resolved*
    // operations to the log; the state machine applies them on every node. The
    // resolution has to happen on the leader because `upsert`/`update` merge
    // metadata patches against current values: replicating the patch would let
    // a follower merge it against its own state and diverge, so the log carries
    // complete values instead.

    /// Build the operations an `add` would produce on a specific partition.
    pub fn prepare_add_partition(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        ids: &[Id],
        embeddings: &[Embedding],
        metadatas: Option<&[Option<Metadata>]>,
        documents: Option<&[Option<Document>]>,
    ) -> EngineResult<PreparedBatch> {
        let coll = self.collection_partition(collection_id, shard)?;
        let mut coll = coll.lock();
        coll.validate_add(ids, embeddings, metadatas, documents)?;
        let dimension = self.infer_dimension(&mut coll, collection_id, embeddings)?;
        let ops = add_ops(ids, embeddings, metadatas, documents);
        Ok(PreparedBatch { ops, dimension })
    }

    /// Build the operations an `upsert` would produce on a specific partition.
    pub fn prepare_upsert_partition(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        ids: &[Id],
        embeddings: &[Embedding],
        metadatas: Option<&[Option<MetadataPatch>]>,
        documents: Option<&[Option<Document>]>,
    ) -> EngineResult<PreparedBatch> {
        let coll = self.collection_partition(collection_id, shard)?;
        let mut coll = coll.lock();
        coll.validate_upsert(ids, embeddings, metadatas, documents)?;
        let dimension = self.infer_dimension(&mut coll, collection_id, embeddings)?;
        let ops = resolve_upsert_ops(&coll, ids, embeddings, metadatas, documents);
        Ok(PreparedBatch { ops, dimension })
    }

    /// Build the operations an `update` would produce on a specific partition.
    pub fn prepare_update_partition(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        ids: &[Id],
        embeddings: Option<&[Embedding]>,
        metadatas: Option<&[Option<MetadataPatch>]>,
        documents: Option<&[Option<Document>]>,
    ) -> EngineResult<PreparedBatch> {
        let coll = self.collection_partition(collection_id, shard)?;
        let coll = coll.lock();
        if let Some(embeddings) = embeddings
            && embeddings.len() != ids.len()
        {
            return Err(EngineError::Validation(format!(
                "update: {} ids but {} embeddings",
                ids.len(),
                embeddings.len()
            )));
        }
        coll.validate_update(ids, metadatas, documents)?;
        if let Some(embeddings) = embeddings {
            crate::collection::check_dimensions("update", embeddings, coll.config.dimension)?;
        }
        let dimension = coll.config.dimension;
        let ops = resolve_update_ops(&coll, ids, embeddings, metadatas, documents);
        Ok(PreparedBatch { ops, dimension })
    }

    /// Build the operations a `delete` would produce on a specific partition.
    pub fn prepare_delete_partition(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        ids: &[Id],
    ) -> EngineResult<PreparedBatch> {
        let coll = self.collection_partition(collection_id, shard)?;
        let coll = coll.lock();
        let ops: Vec<Operation> = ids
            .iter()
            .map(|id| Operation::Delete { id: id.clone() })
            .collect();
        Ok(PreparedBatch {
            ops,
            dimension: coll.config.dimension,
        })
    }

    /// Prepare a delete whose target set is chosen by filters on a specific partition.
    pub fn prepare_delete_where_partition(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        ids: Option<&[Id]>,
        filter: Option<&WhereFilter>,
        document: Option<&WhereDocument>,
        limit: Option<usize>,
    ) -> EngineResult<PreparedBatch> {
        let coll = self.collection_partition(collection_id, shard)?;
        let coll = coll.lock();
        let mut ids = coll.select(ids, filter, document);
        if let Some(limit) = limit {
            ids.truncate(limit);
        }
        let ops: Vec<Operation> = ids.into_iter().map(|id| Operation::Delete { id }).collect();
        Ok(PreparedBatch {
            ops,
            dimension: coll.config.dimension,
        })
    }

    /// Prepare a delete whose target set is chosen by filters rather than
    /// listed by the caller.
    pub fn prepare_delete_where(
        &self,
        collection_id: &Uuid,
        ids: Option<&[Id]>,
        filter: Option<&WhereFilter>,
        document: Option<&WhereDocument>,
        limit: Option<usize>,
    ) -> EngineResult<PreparedBatch> {
        self.prepare_delete_where_partition(collection_id, None, ids, filter, document, limit)
    }

    /// Checkpoint every collection whose applied-entry count has crossed the
    /// configured threshold. Returns how many ran.
    ///
    /// Cluster mode only: there, a checkpoint is not durability (the Raft log
    /// is) but log truncation and restart cost, so it must not sit on the write
    /// path. Single-node engines checkpoint inline from `append_ops`.
    /// A per-collection failure is logged and skipped so one bad shard does
    /// not stop the others.
    pub fn checkpoint_if_due(&self) -> EngineResult<usize> {
        let handles: Vec<(PartitionKey, Arc<Mutex<Collection>>)> = {
            let collections = self.collections.read();
            collections
                .iter()
                .map(|(key, c)| (*key, c.clone()))
                .collect()
        };

        let mut ran = 0;
        for (key, handle) in handles {
            let mut collection = handle.lock();
            if !collection.checkpoint_due() {
                continue;
            }
            match collection.checkpoint(self.catalog.as_ref()) {
                Ok(_) => {
                    ran += 1;
                    tracing::debug!(?key, "background checkpoint complete");
                }
                Err(e) => {
                    tracing::warn!(?key, "background checkpoint failed; WAL tail retained: {e}");
                }
            }
        }
        Ok(ran)
    }

    /// Reserve record IDs for an in-flight 2PC transaction on a partition.
    pub fn reserve_tx_ids(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        tx_id: Uuid,
        ids: &[Id],
    ) -> EngineResult<()> {
        let coll = self.collection_partition(collection_id, shard)?;
        let mut coll = coll.lock();
        coll.reserve_tx_ids(tx_id, ids)
    }

    /// Release record IDs reserved by a 2PC transaction upon commit or abort.
    pub fn release_tx_ids(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        tx_id: &Uuid,
    ) -> EngineResult<()> {
        let coll = self.collection_partition(collection_id, shard)?;
        let mut coll = coll.lock();
        coll.release_tx_ids(tx_id);
        Ok(())
    }

    /// Append operations through the WAL and apply them to a specific partition.
    pub fn append_ops_partition(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        ops: &[Operation],
    ) -> EngineResult<()> {
        let coll = self.collection_partition(collection_id, shard)?;
        let mut coll = coll.lock();
        coll.write_ops(ops)?;
        coll.flush_if_needed()?;
        coll.auto_checkpoint(self.catalog.as_ref())
    }

    /// Append operations through the WAL and apply them. The single-node path.
    pub fn append_ops(&self, collection_id: &Uuid, ops: &[Operation]) -> EngineResult<()> {
        self.append_ops_partition(collection_id, None, ops)
    }

    /// Infer the vector width from the first embedding when the collection has
    /// not fixed one, and persist it so a restart keeps it.
    fn infer_dimension(
        &self,
        coll: &mut Collection,
        collection_id: &Uuid,
        embeddings: &[Embedding],
    ) -> EngineResult<usize> {
        if coll.config.dimension == 0
            && let Some(first) = embeddings.first()
        {
            let dim = first.len();
            coll.adopt_dimension(dim)?;
            // Persisting is best effort: replay re-infers it on this node, and
            // the replicated batch carries the width for every other node.
            if let Err(e) = self.catalog.update_collection_dimension(collection_id, dim) {
                tracing::warn!("Failed to persist inferred dimension {dim}: {e}");
            }
        }
        Ok(coll.config.dimension)
    }

    /// Checkpoint a specific partition: persist its live records, HNSW index, and id
    /// maps to `{wal_dir}/{stem}.checkpoint/`, then prune the WAL down to the
    /// un-checkpointed tail. A no-op for an empty collection.
    pub fn checkpoint_partition(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
    ) -> EngineResult<CheckpointOutcome> {
        let coll = self.collection_partition(collection_id, shard)?;
        let mut coll = coll.lock();
        let outcome = coll.checkpoint(self.catalog.as_ref())?;
        // Exporting is also deferred cleanup: the checkpoint is already durable,
        // so a failed upload must not be reported as a failed checkpoint.
        if let Err(e) = self.export_published_segments(&coll) {
            tracing::warn!(
                collection = %collection_id,
                ?shard,
                "checkpoint published but the segment export failed: {e}"
            );
            return Ok(CheckpointOutcome::Published { deferred: "export" });
        }
        Ok(outcome)
    }

    /// Checkpoint one collection: persist its live records, HNSW index, and id
    /// maps to `{wal_dir}/{id}.checkpoint/`, then prune the WAL down to the
    /// un-checkpointed tail. A no-op for an empty collection. The WAL remains
    /// the source of truth — this only accelerates reopen.
    pub fn checkpoint(&self, collection_id: &Uuid) -> EngineResult<CheckpointOutcome> {
        self.checkpoint_partition(collection_id, None)
    }

    /// Optimize a collection: force-flush the whole buffer into the index,
    /// rebuild the HNSW graph from live records (reclaiming tombstoned/deleted
    /// space — usearch never shrinks), then checkpoint. Returns stats about the
    /// materialized state and WAL size before/after.
    pub fn optimize(&self, collection_id: &Uuid) -> EngineResult<OptimizeStats> {
        let coll = self.collection(collection_id)?;
        let mut coll = coll.lock();
        let stats = coll.optimize(self.catalog.as_ref())?;
        self.export_published_segments(&coll)?;
        Ok(stats)
    }

    /// Ship the **published** generation's vector files to the object store.
    ///
    /// This used to read `<checkpoint>/segments/`, the pre-generation flat
    /// layout, which checkpointing no longer writes — so the export silently
    /// shipped stale or empty data. Keys are generation-scoped so an upload for
    /// a new generation can never overwrite the files a previous generation
    /// still references.
    fn export_published_segments(&self, coll: &Collection) -> EngineResult<()> {
        let Some(generation) = coll.published_generation() else {
            return Ok(());
        };
        let Some(generation_name) = generation.file_name().and_then(|n| n.to_str()) else {
            return Ok(());
        };
        let seg_dir = generation.join("segments");
        if !seg_dir.is_dir() {
            return Ok(());
        }
        let collection_id = coll.config.id;
        for entry in std::fs::read_dir(&seg_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if !name_str.ends_with(".bin") {
                continue;
            }
            let key =
                format!("col/{collection_id}/generations/{generation_name}/segments/{name_str}");
            let bytes = std::fs::read(entry.path())?;
            self.storage.put(&key, &bytes)?;
        }
        Ok(())
    }

    /// Debug helper: the WAL seq of the record `id`, or a validation error if
    /// the record is unknown. Used by tests to assert seq continuity across
    /// checkpoints and prunes.
    pub fn seq_of(&self, collection_id: &Uuid, id: &Id) -> EngineResult<u64> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock();
        coll.records
            .get(id)
            .map(|r| r.seq)
            .ok_or_else(|| EngineError::Validation(format!("id `{id}` not found")))
    }

    /// Fetch records by id; the result vector parallels `ids`, `None` for
    /// unknown ids.
    pub fn get(&self, collection_id: &Uuid, ids: &[Id]) -> EngineResult<Vec<Option<Record>>> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock();
        Ok(ids
            .iter()
            .map(|id| {
                coll.records.get(id).map(|r| {
                    let embedding = coll.embedding_for(id);
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

    /// Fetch records from a specific partition.
    #[allow(clippy::too_many_arguments)]
    pub fn get_records_partition(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        ids: Option<&[Id]>,
        filter: Option<&WhereFilter>,
        document: Option<&WhereDocument>,
        limit: Option<usize>,
        offset: usize,
    ) -> EngineResult<Vec<Record>> {
        let coll = self.collection_partition(collection_id, shard)?;
        let coll = coll.lock();

        let mut selected: Vec<Id> = match ids {
            Some(ids) => {
                let mut wanted: Vec<Id> = ids
                    .iter()
                    .filter(|id| coll.records.contains_key(*id))
                    .cloned()
                    .collect();
                wanted.sort_by_key(|id| coll.offsets.get(id).copied().unwrap_or(u32::MAX));
                wanted
            }
            None => coll.all_ids(),
        };

        if let Some(filter) = filter {
            selected.retain(|id| coll.metadata_matches(id, filter));
        }
        if let Some(document) = document {
            selected.retain(|id| {
                document.matches(coll.records.get(id).and_then(|r| r.document.as_deref()))
            });
        }

        Ok(selected
            .into_iter()
            .skip(offset)
            .take(limit.unwrap_or(usize::MAX))
            .filter_map(|id| {
                coll.records.get(&id).map(|r| Record {
                    id: r.id.clone(),
                    embedding: coll.embedding_for(&id),
                    metadata: r.metadata.clone(),
                    document: r.document.clone(),
                    seq: r.seq,
                })
            })
            .collect())
    }

    /// Unified record read used by the compatibility layer: optional explicit
    /// ids, optional metadata filter, optional offset/limit.
    ///
    /// * `ids = Some(..)` preserves the caller's order; unknown ids are dropped
    ///   (the positional `get` API keeps `None` for missing ids instead).
    /// * `ids = None` scans every live record in insertion order.
    /// * `filter` is applied last, using the same predicate the postings index
    ///   is compiled from, so results agree with `query`.
    pub fn get_records(
        &self,
        collection_id: &Uuid,
        ids: Option<&[Id]>,
        filter: Option<&WhereFilter>,
        document: Option<&WhereDocument>,
        limit: Option<usize>,
        offset: usize,
    ) -> EngineResult<Vec<Record>> {
        let partitions = self.collection_partitions(collection_id);
        if partitions.is_empty() || partitions.len() == 1 {
            return self.get_records_partition(
                collection_id,
                None,
                ids,
                filter,
                document,
                limit,
                offset,
            );
        }

        // Multi-partition local read: merge records across all partitions
        let mut all_records = Vec::new();
        for (shard, _) in partitions {
            let recs =
                self.get_records_partition(collection_id, shard, ids, filter, document, None, 0)?;
            all_records.extend(recs);
        }
        all_records.sort_by_key(|r| r.seq);
        all_records.dedup_by(|a, b| a.id == b.id);
        let paged: Vec<Record> = all_records
            .into_iter()
            .skip(offset)
            .take(limit.unwrap_or(usize::MAX))
            .collect();
        Ok(paged)
    }

    /// k-NN search on a specific partition.
    pub fn query_partition(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        embedding: &Embedding,
        k: usize,
        options: &QueryOptions,
    ) -> EngineResult<Vec<ScoredPoint>> {
        let coll = self.collection_partition(collection_id, shard)?;
        let coll = coll.lock();
        coll.query(embedding, k, options, false)
    }

    /// Run several query vectors against **one** read view of a partition.
    pub fn query_many_partition(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        embeddings: &[Embedding],
        k: usize,
        options: &QueryOptions,
        with_embedding: bool,
    ) -> EngineResult<Vec<Vec<ScoredPoint>>> {
        let coll = self.collection_partition(collection_id, shard)?;
        let coll = coll.lock();
        let mut out = Vec::with_capacity(embeddings.len());
        for embedding in embeddings {
            out.push(coll.query(embedding, k, options, with_embedding)?);
        }
        Ok(out)
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
        let partitions = self.collection_partitions(collection_id);
        if partitions.is_empty() || partitions.len() == 1 {
            let coll = self.collection(collection_id)?;
            let coll = coll.lock();
            return coll.query(embedding, k, options, false);
        }
        let mut all_hits = Vec::new();
        for (shard, _) in partitions {
            let hits = self.query_partition(collection_id, shard, embedding, k, options)?;
            all_hits.extend(hits);
        }
        all_hits.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        all_hits.truncate(k);
        Ok(all_hits)
    }

    /// Run several query vectors against **one** read view.
    ///
    /// All vectors are searched while the collection lock is held, and each
    /// result row carries its metadata, document and (when `with_embedding`)
    /// vector as of that same view. The single-vector [`Engine::query`] is a
    /// thin wrapper for callers that only need ids/distances.
    pub fn query_many(
        &self,
        collection_id: &Uuid,
        embeddings: &[Embedding],
        k: usize,
        options: &QueryOptions,
        with_embedding: bool,
    ) -> EngineResult<Vec<Vec<ScoredPoint>>> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock();
        let mut out = Vec::with_capacity(embeddings.len());
        for embedding in embeddings {
            out.push(coll.query(embedding, k, options, with_embedding)?);
        }
        Ok(out)
    }

    /// Debug helper: live vectors tracked by the engine (`index.len() +
    /// buffer.len()`).
    pub fn indexed_count(&self, collection_id: &Uuid) -> EngineResult<usize> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock();
        Ok(coll.index.len() + coll.buffer.len())
    }

    /// Debug helper: vectors materialized into the HNSW index (committed).
    pub fn committed_count(&self, collection_id: &Uuid) -> EngineResult<usize> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock();
        Ok(coll.index.len())
    }

    /// Debug helper: highest WAL seq materialized into the index.
    pub fn flushed_seq(&self, collection_id: &Uuid) -> EngineResult<u64> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock();
        Ok(coll.flushed_seq)
    }

    /// Get the last WAL seq for a collection (for status reporting).
    pub fn wal_last_seq(&self, collection_id: &Uuid) -> EngineResult<u64> {
        let coll = self.collection(collection_id)?;
        let coll = coll.lock();
        Ok(coll.wal.last_seq())
    }

    /// Apply a batch of operations that are already durable elsewhere.
    ///
    /// This is the cluster apply path: the Raft log has fsynced the entry before
    /// it is committed, so re-writing it to a local WAL would double the
    /// durability cost for no gain. Requires [`ApplyMode::Committed`]; in
    /// [`ApplyMode::Durable`] nothing else would make these writes durable and
    /// the call is refused rather than silently weakening the guarantee.
    ///
    /// `seq` is the Raft log index. Every record in a batch shares it; the
    /// visibility rule is `seq <= boundary`, so duplicates are harmless.
    ///
    /// Apply must be **total** — it cannot fail for a committed entry, because
    /// openraft treats an error from the state machine as fatal for the store.
    /// Everything that can be rejected (dimension, duplicates, batch size) is
    /// validated by the leader before the entry is proposed.
    /// Apply a batch of operations that are already durable elsewhere into a specific partition.
    pub fn apply_committed_partition(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        ops: &[Operation],
        seq: u64,
        dimension: usize,
    ) -> EngineResult<()> {
        if self.config.apply_mode != ApplyMode::Committed {
            return Err(EngineError::Validation(format!(
                "collection {collection_id}: apply_committed requires ApplyMode::Committed; \
                 this engine is its own durable record, so skipping the WAL would lose \
                 acknowledged writes"
            )));
        }
        let coll = self.collection_partition(collection_id, shard)?;
        let mut coll = coll.lock();
        if coll.config.dimension == 0 && dimension > 0 {
            coll.adopt_dimension(dimension)?;
        }
        for op in ops {
            coll.apply_operation(op, seq)?;
        }
        coll.note_applied(ops.len() as u64);
        coll.flush_if_needed()?;
        Ok(())
    }

    /// Apply a batch of operations that are already durable elsewhere.
    pub fn apply_committed(
        &self,
        collection_id: &Uuid,
        ops: &[Operation],
        seq: u64,
        dimension: usize,
    ) -> EngineResult<()> {
        self.apply_committed_partition(collection_id, None, ops, seq, dimension)
    }

    /// Clone the collection handle for a specific partition.
    pub(crate) fn collection_partition(
        &self,
        id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
    ) -> EngineResult<Arc<Mutex<Collection>>> {
        let key = PartitionKey::new(*id, shard);
        if let Some(collection) = self.collections.read().get(&key).cloned() {
            return Ok(collection);
        }

        let Some(record) = self.catalog.get_collection_by_id(id)? else {
            return Err(EngineError::CollectionNotFound(id.to_string()));
        };
        if !self.config.scope.owns(shard.or(record.shard)) {
            return Err(EngineError::NotReplica {
                collection: id.to_string(),
                shard: shard.or(record.shard),
            });
        }
        self.materialize_collection_partition(id, shard)?;
        self.collections
            .read()
            .get(&key)
            .cloned()
            .ok_or_else(|| EngineError::CollectionNotFound(id.to_string()))
    }

    /// Clone all partition handles for a collection on this node.
    pub(crate) fn collection_partitions(
        &self,
        id: &Uuid,
    ) -> Vec<(Option<rekha_core::cluster::ShardId>, Arc<Mutex<Collection>>)> {
        let map = self.collections.read();
        map.iter()
            .filter(|(k, _)| k.id == *id)
            .map(|(k, v)| (k.shard, v.clone()))
            .collect()
    }

    /// Clone the collection handle, releasing the map read lock before any
    /// collection mutex is taken (lock ordering: never nest collection locks).
    fn collection(&self, id: &Uuid) -> EngineResult<Arc<Mutex<Collection>>> {
        let Some(record) = self.catalog.get_collection_by_id(id)? else {
            return Err(EngineError::CollectionNotFound(id.to_string()));
        };
        if !self.config.scope.owns(record.shard) {
            return Err(EngineError::NotReplica {
                collection: id.to_string(),
                shard: record.shard,
            });
        }
        let key = PartitionKey::new(*id, record.shard);
        if let Some(collection) = self.collections.read().get(&key).cloned() {
            return Ok(collection);
        }
        for (k, c) in self.collections.read().iter() {
            if k.id == *id {
                return Ok(c.clone());
            }
        }
        self.materialize_collection(id)?;
        if let Some(collection) = self.collections.read().get(&key).cloned() {
            return Ok(collection);
        }
        for (k, c) in self.collections.read().iter() {
            if k.id == *id {
                return Ok(c.clone());
            }
        }
        Err(EngineError::CollectionNotFound(id.to_string()))
    }

    /// The directory this engine owns: collection WALs, checkpoint generations,
    /// and the restore staging area.
    ///
    /// The snapshot store and the API's snapshot staging area live here too, so
    /// a multi-gigabyte archive is staged on the same filesystem as the data it
    /// will replace rather than in `/tmp`, which is often a different mount and
    /// usually smaller.
    pub fn storage_dir(&self) -> &Path {
        &self.wal_dir
    }

    /// Create a snapshot of collection partition state for replication.
    pub fn snapshot_bytes_partition(
        &self,
        state: &[u8],
        shard: Option<rekha_core::cluster::ShardId>,
        scope: &[Uuid],
    ) -> EngineResult<Vec<u8>> {
        let handles: Vec<(Uuid, Arc<Mutex<Collection>>)> = {
            let collections = self.collections.read();
            let mut handles: Vec<_> = scope
                .iter()
                .filter_map(|id| {
                    let key = PartitionKey::new(*id, shard);
                    collections.get(&key).map(|c| (*id, c.clone()))
                })
                .collect();
            // Deterministic order: the same state must produce the same archive.
            handles.sort_by_key(|(id, _)| *id);
            handles
        };

        let mut entries = Vec::with_capacity(handles.len());
        for (id, handle) in handles {
            let collection = handle.lock();
            let generation = crate::collection::active_generation(&collection.checkpoint_dir)
                .ok_or_else(|| {
                    EngineError::Recovery(format!(
                        "collection {id} has no published checkpoint generation; \
                         a snapshot cannot be taken from memory alone because the \
                         vectors live in segments"
                    ))
                })?;
            let record = self
                .catalog
                .get_collection_by_id(&id)?
                .ok_or_else(|| EngineError::CollectionNotFound(id.to_string()))?;
            entries.push((record, generation));
        }

        crate::snapshot::encode(state, entries)
    }

    /// Create a snapshot of all collection state for replication.
    pub fn snapshot_bytes(&self, state: &[u8], scope: &[Uuid]) -> EngineResult<Vec<u8>> {
        self.snapshot_bytes_partition(state, None, scope)
    }

    /// The state blob a snapshot archive carries, without restoring it.
    ///
    /// Used by the receiver to decide whether an archive belongs to this group
    /// before any of it is written.
    pub fn snapshot_state(&self, archive: &[u8]) -> EngineResult<Vec<u8>> {
        crate::snapshot::peek_state(archive)
    }

    /// Checkpoint partitions in scope.
    pub fn checkpoint_all_partition(
        &self,
        shard: Option<rekha_core::cluster::ShardId>,
        scope: &[Uuid],
    ) -> EngineResult<usize> {
        let handles: Vec<(Uuid, Arc<Mutex<Collection>>)> = {
            let collections = self.collections.read();
            scope
                .iter()
                .filter_map(|id| {
                    let key = PartitionKey::new(*id, shard);
                    collections.get(&key).map(|c| (*id, c.clone()))
                })
                .collect()
        };

        let mut count = 0;
        for (id, handle) in handles {
            let mut collection = handle.lock();
            collection.checkpoint(self.catalog.as_ref()).map_err(|e| {
                tracing::error!(%id, "snapshot: checkpoint before snapshot failed: {e}");
                e
            })?;
            count += 1;
        }
        Ok(count)
    }

    /// Checkpoint every collection.
    pub fn checkpoint_all(&self, scope: &[Uuid]) -> EngineResult<usize> {
        self.checkpoint_all_partition(None, scope)
    }

    /// Replace partitions with the contents of `archive`.
    pub fn restore_bytes_partition(
        &self,
        shard: Option<rekha_core::cluster::ShardId>,
        archive: &[u8],
    ) -> EngineResult<(Vec<u8>, RestoreReport)> {
        let decoded = crate::snapshot::decode(archive)?;
        let report = RestoreReport {
            collections: decoded.collections.len(),
            files: decoded.collections.iter().map(|c| c.files.len()).sum(),
            bytes: archive.len() as u64,
        };

        let tag = Uuid::new_v4().simple().to_string();
        let staging_root = self.wal_dir.join(format!(".restore-{tag}"));

        // ---- Phase 1: stage and validate, touching nothing live ------------
        let mut staged = Vec::with_capacity(decoded.collections.len());
        for entry in &decoded.collections {
            let id = entry.record.config.id;
            let base = staging_root.join(id.to_string());
            let generation_dir = base
                .join(crate::collection::GENERATIONS_DIR)
                .join(&entry.generation);
            std::fs::create_dir_all(&generation_dir)?;
            for (relative, bytes) in &entry.files {
                let path = generation_dir.join(relative);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&path, bytes)?;
            }
            std::fs::write(
                base.join(crate::collection::CURRENT_POINTER),
                &entry.generation,
            )?;

            // The checkpoint loader is the authority on whether this generation
            // is usable; re-verify with it rather than trusting our own framing.
            match Collection::read_generation(&base, &entry.record.config) {
                crate::collection::CheckpointLoad::Loaded(_) => {}
                crate::collection::CheckpointLoad::Absent => {
                    return Err(EngineError::Recovery(format!(
                        "snapshot for collection {id} contains no readable generation `{}`",
                        entry.generation
                    )));
                }
                crate::collection::CheckpointLoad::Corrupt(detail) => {
                    return Err(EngineError::Recovery(format!(
                        "snapshot for collection {id} failed validation: {detail}"
                    )));
                }
            }
            staged.push((entry, base));
        }

        // ---- Phase 2: commit ----------------------------------------------
        let aside = self.wal_dir.join(format!(".restore-old-{tag}"));
        std::fs::create_dir_all(&aside)?;
        {
            let mut collections = self.collections.write();

            for (entry, staged_base) in staged {
                let id = entry.record.config.id;
                let key = PartitionKey::new(id, shard);

                // Close the live handle before its files move.
                collections.remove(&key);

                let stem = match shard {
                    Some(s) => format!("{id}.shard-{}", s.0),
                    None => id.to_string(),
                };

                let live_base = self.wal_dir.join(format!("{stem}.checkpoint"));
                if live_base.exists() {
                    std::fs::rename(&live_base, aside.join(format!("{stem}.checkpoint")))?;
                }
                let live_wal = self.wal_dir.join(format!("{stem}.wal"));
                if live_wal.exists() {
                    std::fs::rename(&live_wal, aside.join(format!("{stem}.wal")))?;
                }

                std::fs::rename(&staged_base, &live_base)?;

                if self.catalog.get_collection_by_id(&id)?.is_some() {
                    self.catalog.delete_collection(&id)?;
                }
                self.catalog
                    .create_collection(&entry.record.config, entry.record.shard)?;
                if entry.record.max_seq_id > 0 {
                    self.catalog
                        .advance_log_offset(&id, entry.record.max_seq_id)?;
                }

                let collection = Collection::open(
                    &id,
                    shard,
                    entry.record.config.clone(),
                    &self.wal_dir,
                    self.config.epoch,
                    self.config.durability.is_durable(),
                    entry.record.max_seq_id,
                )?;
                collections.insert(key, Arc::new(Mutex::new(collection)));
            }
        }

        // Best-effort cleanup; leaving these directories would waste space but
        // cannot affect correctness.
        if let Err(e) = std::fs::remove_dir_all(&staging_root) {
            tracing::warn!("restore: could not remove staging directory: {e}");
        }
        if let Err(e) = std::fs::remove_dir_all(&aside) {
            tracing::warn!("restore: could not remove replaced state: {e}");
        }

        Ok((decoded.state, report))
    }

    /// Replace this node's collections with the contents of `archive`.
    pub fn restore_bytes(&self, archive: &[u8]) -> EngineResult<(Vec<u8>, RestoreReport)> {
        self.restore_bytes_partition(None, archive)
    }
}

impl rekha_cluster::EngineSnapshotProvider for Engine {
    fn snapshot_bytes(
        &self,
        state: &[u8],
        shard: Option<rekha_core::cluster::ShardId>,
        collections: &[Uuid],
    ) -> Result<Vec<u8>, String> {
        Engine::snapshot_bytes_partition(self, state, shard, collections).map_err(|e| e.to_string())
    }

    fn record_collection(
        &self,
        config: &rekha_core::config::CollectionConfig,
        shard: Option<rekha_core::cluster::ShardId>,
    ) -> Result<(), String> {
        Engine::record_collection(self, config, shard)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn materialize_collection(
        &self,
        collection_id: &Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
    ) -> Result<(), String> {
        Engine::materialize_collection_partition(self, collection_id, shard)
            .map_err(|e| e.to_string())
    }

    fn restore_bytes(
        &self,
        shard: Option<rekha_core::cluster::ShardId>,
        archive: &[u8],
    ) -> Result<Vec<u8>, String> {
        Engine::restore_bytes_partition(self, shard, archive)
            .map(|(state, _report)| state)
            .map_err(|e| e.to_string())
    }

    fn snapshot_state(&self, archive: &[u8]) -> Result<Vec<u8>, String> {
        Engine::snapshot_state(self, archive).map_err(|e| e.to_string())
    }

    fn checkpoint_all(
        &self,
        shard: Option<rekha_core::cluster::ShardId>,
        collections: &[Uuid],
    ) -> Result<usize, String> {
        Engine::checkpoint_all_partition(self, shard, collections).map_err(|e| e.to_string())
    }

    fn apply_batch(
        &self,
        collection_id: &uuid::Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        ops: &[rekha_core::op::Operation],
        index: u64,
        dimension: usize,
    ) -> Result<(), String> {
        Engine::apply_committed_partition(self, collection_id, shard, ops, index, dimension)
            .map_err(|e| e.to_string())
    }

    fn drop_collection(
        &self,
        collection_id: &uuid::Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
    ) -> Result<(), String> {
        let _ = self.catalog.delete_collection(collection_id);
        Engine::drop_collection_partition(self, collection_id, shard).map_err(|e| e.to_string())
    }

    fn reserve_tx_ids(
        &self,
        collection_id: &uuid::Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        tx_id: uuid::Uuid,
        ids: &[rekha_core::types::Id],
    ) -> Result<(), String> {
        Engine::reserve_tx_ids(self, collection_id, shard, tx_id, ids).map_err(|e| e.to_string())
    }

    fn release_tx_ids(
        &self,
        collection_id: &uuid::Uuid,
        shard: Option<rekha_core::cluster::ShardId>,
        tx_id: &uuid::Uuid,
    ) -> Result<(), String> {
        Engine::release_tx_ids(self, collection_id, shard, tx_id).map_err(|e| e.to_string())
    }
}

pub fn add_ops(
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

/// Resolve one record's document from the optional update list.
///
/// * outer `None` — the request did not mention documents, so the current one
///   stands (this is what the Chroma client sends for a metadata-only update);
/// * `Some([.., None, ..])` at this position — an explicit null clears it;
/// * `Some([.., Some(doc), ..])` — replace it.
fn resolve_document(
    current: Option<&Document>,
    documents: Option<&[Option<Document>]>,
    index: usize,
) -> Option<Document> {
    match documents {
        None => current.cloned(),
        Some(docs) => match docs.get(index) {
            None => current.cloned(),
            Some(None) => None,
            Some(Some(doc)) => Some(doc.clone()),
        },
    }
}

/// Turn an update batch into WAL operations carrying **complete** final values.
///
/// Chroma merges metadata and preserves every field the request does not
/// mention, so the final values are resolved here against the live record and
/// only then encoded. Replay still sees fully-materialized operations, which is
/// what lets the WAL format stay unchanged.
///
/// Ids that do not exist are skipped: Chroma treats such an update as a silent
/// no-op rather than an error.
fn resolve_update_ops(
    coll: &Collection,
    ids: &[Id],
    embeddings: Option<&[Embedding]>,
    metadatas: Option<&[Option<MetadataPatch>]>,
    documents: Option<&[Option<Document>]>,
) -> Vec<Operation> {
    let mut ops = Vec::with_capacity(ids.len());
    for (i, id) in ids.iter().enumerate() {
        let Some(current) = coll.records.get(id) else {
            continue;
        };
        let metadata = match metadatas.and_then(|m| m.get(i)).and_then(|p| p.as_ref()) {
            Some(patch) => apply_metadata_patch(current.metadata.as_ref(), patch),
            None => current.metadata.clone(),
        };
        let document = resolve_document(current.document.as_ref(), documents, i);
        match embeddings {
            Some(embeddings) => ops.push(Operation::Upsert {
                id: id.clone(),
                embedding: embeddings[i].clone(),
                metadata,
                document,
            }),
            None => ops.push(Operation::Update {
                id: id.clone(),
                metadata,
                document,
            }),
        }
    }
    ops
}

/// Same resolution for an upsert, where a missing id becomes a new record.
fn resolve_upsert_ops(
    coll: &Collection,
    ids: &[Id],
    embeddings: &[Embedding],
    metadatas: Option<&[Option<MetadataPatch>]>,
    documents: Option<&[Option<Document>]>,
) -> Vec<Operation> {
    ids.iter()
        .enumerate()
        .map(|(i, id)| {
            let current = coll.records.get(id);
            let metadata = match metadatas.and_then(|m| m.get(i)).and_then(|p| p.as_ref()) {
                Some(patch) => {
                    apply_metadata_patch(current.and_then(|r| r.metadata.as_ref()), patch)
                }
                None => current.and_then(|r| r.metadata.clone()),
            };
            let document =
                resolve_document(current.and_then(|r| r.document.as_ref()), documents, i);
            Operation::Upsert {
                id: id.clone(),
                embedding: embeddings[i].clone(),
                metadata,
                document,
            }
        })
        .collect()
}

#[cfg(test)]
mod lock_tests {
    use super::*;
    use rekha_core::types::Distance;
    use rekha_storage::{LocalStorage, RedbCatalog};

    /// A panic inside a writer must not disable the collection.
    ///
    /// The engine used `std::sync::Mutex`, whose guards poison on panic: after
    /// one panicking writer every later `lock()` returned `Err`, and the
    /// `unwrap()` on it turned that into a second panic, so the collection was
    /// dead for the life of the process. `parking_lot` removes poisoning
    /// entirely, so this test guards the swap rather than a behaviour we wrote.
    ///
    /// The trade-off is deliberate: without poisoning, a panic leaves this
    /// process running, so the release profile uses `panic = "abort"` to make a
    /// panic fail-stop instead of letting a half-applied mutation survive.
    #[test]
    fn a_panicking_writer_does_not_disable_the_collection() {
        let dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
        let storage = Arc::new(LocalStorage::new(dir.path().join("objects")));
        let wal_dir = dir.path().join("wal");
        std::fs::create_dir_all(&wal_dir).unwrap();
        let engine =
            Arc::new(Engine::open(catalog, storage, &wal_dir, EngineConfig::default()).unwrap());

        let config = rekha_core::config::CollectionConfig::new("poison".into(), 2, Distance::L2);
        let id = engine.create_collection(&config).unwrap().config.id;

        let handle = engine
            .collections
            .read()
            .get(&PartitionKey::new(id, None))
            .cloned()
            .expect("collection handle");

        let panicked = std::thread::spawn(move || {
            let _guard = handle.lock();
            panic!("simulated writer panic");
        })
        .join();

        assert!(panicked.is_err(), "the writer thread was meant to panic");
        assert_eq!(
            engine.count(&id).unwrap(),
            0,
            "the collection must still answer after a panicking writer"
        );
    }

    /// A removal must clear the catalog row even when this node never opened
    /// the collection: the group that applies the entry need not host it.
    #[test]
    fn drop_removes_a_collection_that_was_never_materialized() {
        use rekha_cluster::EngineSnapshotProvider;

        let dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
        let storage = Arc::new(LocalStorage::new(dir.path().join("objects")));
        let wal_dir = dir.path().join("wal");
        std::fs::create_dir_all(&wal_dir).unwrap();
        let engine = Engine::open(catalog, storage, &wal_dir, EngineConfig::default()).unwrap();

        let config =
            rekha_core::config::CollectionConfig::new("unmaterialized".into(), 2, Distance::L2);
        let id = config.id;
        engine.record_collection(&config, None).expect("record");

        engine.drop_collection(&id, None).expect("drop");

        assert!(
            engine.get_collection_by_id(&id).unwrap().is_none(),
            "the catalog row must be gone"
        );
    }
}
