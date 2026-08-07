//! RekhaDB `rekha-index` crate — the ANN index seam.
//!
//! This crate defines the [`Index`] trait — the only thing the engine (the
//! next phase) touches for vector search — plus a usearch-backed HNSW
//! implementation, [`UsearchIndex`].
//!
//! # Design contract
//!
//! **The WAL is the source of truth; indexes are derived.** An index over a
//! collection is a materialized projection of that collection's WAL records and
//! can always be rebuilt by replaying the log. Callers may treat removal as a
//! soft-delete + periodic rebuild flow, mirroring hnswlib, precisely because a
//! derived index is always reconstructible. [`UsearchIndex::delete`] is a
//! physical delete where the backend supports it (usearch does), so this crate
//! removes eagerly.
//!
//! # Hit semantics
//!
//! [`IndexHit::distance`] follows Chroma semantics: **LOWER is closer**, and the
//! value is exactly what `rekha_distance::distance(space, a, b)` computes for
//! the collection's space on raw vectors:
//!
//! - `L2` → squared Euclidean distance (`Σ(aᵢ − bᵢ)²`, no square root)
//! - `Ip` → `1 − dot(a, b)`
//! - `Cosine` → `1 − cos_sim(a, b)` computed on raw vectors
//!
//! See [`UsearchIndex`] for the exact usearch metric mapping that achieves this
//! and the label-management scheme (monotonic labels, never reused, kept for
//! future persistence).

mod usearch_index;

use std::path::Path;

use rekha_core::types::{Distance, Embedding, Id};

pub use usearch_index::UsearchIndex;

/// One search hit: the external id and the distance (LOWER = closer, Chroma
/// semantics: l2 = squared L2, ip = 1 − dot, cosine = 1 − cos_sim).
#[derive(Debug, Clone, PartialEq)]
pub struct IndexHit {
    pub id: Id,
    pub distance: f32,
}

/// Errors surfaced by an [`Index`].
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    /// The underlying ANN backend (usearch) failed.
    #[error("usearch: {0}")]
    Usearch(String),
    /// A vector was offered whose length does not match the index dimension.
    #[error("dimension mismatch: index is {index}, got {got}")]
    DimensionMismatch { index: usize, got: usize },
    /// The id is already indexed (use delete before re-adding).
    #[error("id `{0}` already exists (use upsert/delete first)")]
    DuplicateId(String),
    /// The id is not indexed.
    #[error("id `{0}` not found")]
    NotFound(String),
    /// An I/O error occurred while persisting or loading the index.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The index file is malformed, or its header disagrees with the caller
    /// (wrong dimension / distance space / label count).
    #[error("corrupt index file: {0}")]
    Corrupt(String),
    /// A metadata/map payload could not be (de)serialized.
    #[error("bincode: {0}")]
    Encode(#[from] bincode::Error),
}

/// Result alias for index operations.
pub type IndexResult<T> = Result<T, IndexError>;

/// A persistent-capable ANN index over a single collection's vectors.
///
/// Design contract: the WAL is the source of truth; this index is derived and
/// can always be rebuilt by replaying the log. Deletes are physical where the
/// backend supports it (usearch does) but callers may treat removal as a
/// soft-delete + periodic-rebuild flow, mirroring hnswlib.
///
/// # Persistence format (Phase 4a)
///
/// [`Index::save`] writes two files:
///
/// - `path` — the raw usearch graph dump (its own portable format).
/// - `path` with extension `"meta"` — bincode of [`UsearchIndex::IndexMeta`]:
///   the id↔label maps, `next_label`, distance-space name, and dimension.
///
/// Both are written via a temp file + rename so a crash never leaves a
/// half-written index on disk. [`UsearchIndex::load`] restores both and
/// verifies the stored space/dimension/label-count against the caller's.
pub trait Index: Send + Sync {
    /// Insert `id` with vector `embedding`. Errors with [`IndexError::DuplicateId`]
    /// if present and [`IndexError::DimensionMismatch`] if the vector length
    /// differs from [`Index::dimension`].
    fn add(&mut self, id: &Id, embedding: &Embedding) -> IndexResult<()>;

    /// Physically remove `id` if present; `Ok(())` if absent.
    fn delete(&mut self, id: &Id) -> IndexResult<()>;

    /// k-NN search. `k` = how many results to return; `ef` = HNSW beam width at
    /// query time (ef_search). Distances follow Chroma semantics (lower=closer).
    /// Returns at most `k` hits, sorted ascending by distance.
    fn search(&self, query: &Embedding, k: usize, ef: usize) -> IndexResult<Vec<IndexHit>>;

    /// Number of live (non-deleted) elements.
    fn len(&self) -> usize;

    /// Whether the index holds no live elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `id` is currently indexed.
    fn contains(&self, id: &Id) -> bool;

    /// Dimensionality of indexed vectors.
    fn dimension(&self) -> usize;

    /// The distance space this index was built for.
    fn space(&self) -> Distance;

    /// Persist the graph plus the id↔label maps to `path` (and `path` with
    /// extension `"meta"`), atomically. See the trait docs for the format.
    fn save(&self, path: &Path) -> IndexResult<()>;
}
