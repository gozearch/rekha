//! usearch-backed HNSW implementation of the [`Index`] trait.
//!
//! # Label management
//!
//! External `Id`s map to monotonic `u64` labels (0, 1, 2, …) through the
//! `labels` (`id_to_label`) and `ids` (`label_to_id`) maps — the Chroma
//! `id_to_label` / `label_to_id` analog, kept exact because the persistence
//! phase will need them to survive a save/load round-trip. Labels are **never
//! reused**: `next_label` only ever increments, so usearch's physical `remove`
//! leaves label gaps in the graph (harmless — the graph only stores vectors).
//! The maps are mutated *before* the usearch mutation and rolled back if it
//! fails, so the maps never disagree with the graph.
//!
//! # Metric → distance conversion
//!
//! usearch 2.26 reports distances directly in the metric's raw space, which
//! already matches Chroma / `rekha_distance` semantics — **no conversion is
//! applied** and no ingest-time normalization is performed:
//!
//! | `Distance` (Chroma) | usearch `MetricKind` | usearch `Matches::distances` reports | matches `rekha_distance::…` |
//! |---------------------|----------------------|--------------------------------------|-----------------------------|
//! | `L2`                | `L2sq`               | `Σ(aᵢ − bᵢ)²`                        | `l2_squared`                |
//! | `Ip`                | `IP`                 | `1 − Σ aᵢ·bᵢ`                        | `distance(Ip, …)`           |
//! | `Cosine`            | `Cos`                | `1 − Σaᵢbᵢ / (‖a‖ · ‖b‖)`            | `distance(Cosine, …)`       |
//!
//! Vectors are stored as raw `f32` (`ScalarKind::F32`). The `Cos` kernel
//! divides by both vector norms itself, so cosine is computed on raw vectors
//! and matches `rekha_distance`'s `1 − dot(normalize(a), normalize(b))` within
//! f32 rounding. Search results are returned sorted ascending by distance
//! (usearch's `top.sort_ascending()` before dump).
//!
//! # Capacity management
//!
//! usearch does not auto-expand: inserting when `size` reaches `capacity`
//! errors with "Reserve capacity ahead of insertions!". [`UsearchIndex`] grows
//! the reserved capacity geometrically (doubling, minimum 64 slots) from within
//! [`Index::add`], so capacity growth is amortized O(1) per insert and callers
//! never need to reserve up front.
//!
//! # Search `ef`
//!
//! usearch 2.26 has no per-call `SearchOptions`; the search beam is the index's
//! `expansion_search`. [`Index::search`] therefore calls
//! `change_expansion_search(ef)` before every query (internally
//! `expansion = max(ef, k)`). That mutates the C++ index through a shared
//! reference, so concurrent `search` calls on one `UsearchIndex` must be
//! externally synchronized (e.g. a `Mutex` held by the engine).

use std::collections::HashMap;
use std::path::Path;

use rekha_core::config::HnswConfig;
use rekha_core::types::{Distance, Embedding, Id};
use serde::{Deserialize, Serialize};

use crate::{Index, IndexError, IndexHit, IndexResult};

/// The persisted id↔label bookkeeping written alongside the usearch graph by
/// [`Index::save`] (as `path.with_extension("meta")`, bincode-serialized).
///
/// The graph dump stores vectors under opaque `u64` labels; without these maps
/// a restored graph would be anonymous. Saving them is what makes reopen fast:
/// the engine can load a checkpoint's index without rebuilding it from records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexMeta {
    /// External id → internal `u64` label.
    pub labels: HashMap<Id, u64>,
    /// Internal `u64` label → external id.
    pub ids: HashMap<u64, Id>,
    /// Next monotonic label to hand out (labels are never reused, so this must
    /// survive a save/load round-trip or new inserts would collide).
    pub next_label: u64,
    /// [`Distance::name`] the index was built for, verified on load.
    pub space: String,
    /// Vector dimensionality, verified on load.
    pub dimension: usize,
}

/// An HNSW index over a single collection's vectors, backed by the usearch
/// C++ library.
pub struct UsearchIndex {
    inner: usearch::Index,
    space: Distance,
    dimension: usize,
    /// External id → internal `u64` label (Chroma's `id_to_label`).
    labels: HashMap<Id, u64>,
    /// Internal `u64` label → external id (Chroma's `label_to_id`).
    ids: HashMap<u64, Id>,
    /// Next monotonic label to hand out. Only ever incremented, so labels are
    /// never reused after a delete.
    next_label: u64,
}

impl UsearchIndex {
    /// Builds an index with default HNSW tuning ([`HnswConfig::default`]:
    /// `m = 16`, `ef_construction = 200`, `ef_search = 100`).
    pub fn new(space: Distance, dimension: usize) -> IndexResult<Self> {
        Self::with_hnsw(space, dimension, &HnswConfig::default())
    }

    /// Builds an index with explicit HNSW hyperparameters. `connectivity` maps
    /// to `HnswConfig::m`, `expansion_add` to `ef_construction`, and
    /// `expansion_search` to `ef_search` (the per-`search` `ef` argument
    /// overrides the latter at query time).
    pub fn with_hnsw(space: Distance, dimension: usize, hnsw: &HnswConfig) -> IndexResult<Self> {
        let metric = match space {
            Distance::L2 => usearch::MetricKind::L2sq,
            Distance::Ip => usearch::MetricKind::IP,
            Distance::Cosine => usearch::MetricKind::Cos,
        };
        let options = usearch::IndexOptions {
            dimensions: dimension,
            metric,
            quantization: usearch::ScalarKind::F32,
            connectivity: hnsw.m,
            expansion_add: hnsw.ef_construction,
            expansion_search: hnsw.ef_search,
            multi: false,
        };
        let inner =
            usearch::Index::new(&options).map_err(|e| IndexError::Usearch(e.to_string()))?;
        Ok(Self {
            inner,
            space,
            dimension,
            labels: HashMap::new(),
            ids: HashMap::new(),
            next_label: 0,
        })
    }

    /// Grows the underlying usearch capacity when it is exhausted. usearch does
    /// not auto-expand: it errors with "Reserve capacity ahead of insertions!"
    /// if `size` reaches `capacity`. We grow geometrically (at least 64 slots)
    /// to keep the per-add cost amortized O(1).
    fn ensure_capacity(&self) -> IndexResult<()> {
        if self.inner.size() >= self.inner.capacity() {
            let target = (self.inner.capacity() * 2).max(self.inner.size() + 64);
            self.inner
                .reserve(target)
                .map_err(|e| IndexError::Usearch(e.to_string()))?;
        }
        Ok(())
    }
}

impl Index for UsearchIndex {
    fn add(&mut self, id: &Id, embedding: &Embedding) -> IndexResult<()> {
        if embedding.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                index: self.dimension,
                got: embedding.len(),
            });
        }
        if self.labels.contains_key(id) {
            return Err(IndexError::DuplicateId(id.clone()));
        }
        self.ensure_capacity()?;
        let label = self.next_label;
        self.next_label += 1;
        self.labels.insert(id.clone(), label);
        self.ids.insert(label, id.clone());
        let result = self
            .inner
            .add(label, embedding.as_ref())
            .map_err(|e| IndexError::Usearch(e.to_string()));
        if result.is_err() {
            self.labels.remove(id);
            self.ids.remove(&label);
        }
        result
    }

    fn delete(&mut self, id: &Id) -> IndexResult<()> {
        let Some(&label) = self.labels.get(id) else {
            return Ok(());
        };
        // Physical removal. The FFI result (count + errors) is intentionally
        // ignored: the maps are our bookkeeping source of truth, and any label
        // left behind in the graph is filtered out of search results below.
        let _ = self.inner.remove(label);
        self.labels.remove(id);
        self.ids.remove(&label);
        Ok(())
    }

    fn search(&self, query: &Embedding, k: usize, ef: usize) -> IndexResult<Vec<IndexHit>> {
        if query.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                index: self.dimension,
                got: query.len(),
            });
        }
        self.inner.change_expansion_search(ef);
        let matches = self
            .inner
            .search(query.as_ref(), k)
            .map_err(|e| IndexError::Usearch(e.to_string()))?;
        Ok(matches
            .keys
            .into_iter()
            .zip(matches.distances)
            .filter_map(|(label, distance)| {
                self.ids.get(&label).map(|id| IndexHit {
                    id: id.clone(),
                    distance,
                })
            })
            .collect())
    }

    fn len(&self) -> usize {
        self.inner.size()
    }

    fn contains(&self, id: &Id) -> bool {
        self.labels.contains_key(id)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn space(&self) -> Distance {
        self.space
    }

    fn save(&self, path: &Path) -> IndexResult<()> {
        // Graph: usearch's own portable format, via temp + rename.
        let graph_tmp = tmp_sibling(path);
        self.inner
            .save(&graph_tmp.to_string_lossy())
            .map_err(|e| IndexError::Usearch(e.to_string()))?;
        std::fs::rename(&graph_tmp, path)?;

        // Maps: bincode, via temp + rename.
        let meta = IndexMeta {
            labels: self.labels.clone(),
            ids: self.ids.clone(),
            next_label: self.next_label,
            space: self.space.name().to_owned(),
            dimension: self.dimension,
        };
        let bytes = bincode::serialize(&meta)?;
        let meta_path = path.with_extension("meta");
        let meta_tmp = tmp_sibling(&meta_path);
        std::fs::write(&meta_tmp, &bytes)?;
        std::fs::rename(&meta_tmp, meta_path)?;
        Ok(())
    }
}

/// Temp-file sibling of `path` (appends `.tmp` to the file name), used for the
/// atomic write-then-rename in [`Index::save`].
fn tmp_sibling(path: &Path) -> std::path::PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".tmp");
    std::path::PathBuf::from(os)
}

impl UsearchIndex {
    /// Restore an index previously written by [`Index::save`].
    ///
    /// Reads `path.with_extension("meta")` (the bincode maps) and `path` (the
    /// usearch graph), verifies the persisted space name and dimension match
    /// the caller's expectations and that the map label count equals the
    /// graph's live size, then restores the label maps. Any mismatch or I/O
    /// failure is an [`IndexError`] — never a panic.
    pub fn load(path: &Path, space: Distance, dimension: usize) -> IndexResult<Self> {
        let meta_path = path.with_extension("meta");
        let bytes = std::fs::read(&meta_path)?;
        let meta: IndexMeta = bincode::deserialize(&bytes)?;

        if meta.space != space.name() {
            return Err(IndexError::Corrupt(format!(
                "space mismatch: meta says `{}`, expected `{}`",
                meta.space,
                space.name()
            )));
        }
        if meta.dimension != dimension {
            return Err(IndexError::Corrupt(format!(
                "dimension mismatch: meta says {}, expected {dimension}",
                meta.dimension
            )));
        }

        // usearch 2.26 `restore` reads the file header, constructs a matching
        // index, and loads — repopulating connectivity/expansion from the file.
        let inner = usearch::Index::restore(&path.to_string_lossy())
            .map_err(|e| IndexError::Usearch(e.to_string()))?;

        if inner.size() != meta.labels.len() {
            return Err(IndexError::Corrupt(format!(
                "label count mismatch: {} labels but the graph holds {} vectors",
                meta.labels.len(),
                inner.size()
            )));
        }

        Ok(Self {
            inner,
            space,
            dimension,
            labels: meta.labels,
            ids: meta.ids,
            next_label: meta.next_label,
        })
    }
}
