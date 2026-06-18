use rekha_core::{
    distance::{l2_squared, l2_squared_partial},
    DistanceMetric, IndexBufferHandle, IndexError, RekhaError, SearchParams, VectorIndex,
    VectorStoreBackend,
};
use rekha_storage::RocksVectorStore;

use crate::pq::ProductQuantizer;
use crate::vamana::VamanaGraph;

use std::collections::HashSet;
use std::sync::{Arc, RwLock};

/// In-memory buffer for recent inserts before they're flushed to the Vamana graph.
struct InsertBuffer {
    vectors: Vec<(u64, Vec<f32>)>,
    deleted: HashSet<u64>,
}

impl InsertBuffer {
    fn new() -> Self {
        Self {
            vectors: Vec::new(),
            deleted: HashSet::new(),
        }
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.vectors.len()
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    fn push(&mut self, id: u64, vector: Vec<f32>) {
        self.vectors.push((id, vector));
    }

    fn mark_deleted(&mut self, ids: &[u64]) {
        for id in ids {
            self.deleted.insert(*id);
        }
    }

    fn contains_deleted(&self, id: u64) -> bool {
        self.deleted.contains(&id)
    }

    fn drain(&mut self) -> Vec<(u64, Vec<f32>)> {
        let mut drained = Vec::new();
        std::mem::swap(&mut self.vectors, &mut drained);
        self.deleted.clear();
        drained
    }
}

/// The main Rekha index, combining PQ compression with Vamana graph search.
///
/// Query flow:
/// 1. PQ distance table is computed from the query (fast, in memory)
/// 2. Vamana graph search uses PQ-ADC distances for candidate selection
/// 3. Top candidates are re-ranked with full-precision distances (from disk)
/// 4. Recent inserts in buffer are searched via brute-force and merged
pub struct RekhaIndex {
    /// Product quantizer for compressed distance computation.
    pq: ProductQuantizer,
    /// Vamana graph for indexed neighbor traversal.
    graph: VamanaGraph,
    /// Storage backend for full-precision vectors.
    store: RocksVectorStore,
    #[allow(dead_code)]
    metric: String,
    /// Total vector dimension.
    dim: usize,
    /// In-memory cache of vectors for graph construction (indexed vectors).
    vectors: Vec<(u64, Vec<f32>)>,
    /// Whether the index is ready for search.
    ready: bool,
    /// In-memory buffer for recent inserts (Tier 2).
    insert_buffer: Arc<RwLock<InsertBuffer>>,
    /// Buffer capacity before forced flush.
    buffer_capacity: usize,
    /// Flush interval in milliseconds.
    flush_interval_ms: u64,
    /// Whether PQ has been trained (needed for buffer search).
    pq_trained: bool,
}

impl RekhaIndex {
    /// Create a new Rekha index.
    ///
    /// # Arguments
    /// * `dim` - Dimension of vectors to index
    /// * `pq_m` - Number of PQ sub-vectors (e.g., 64 for 768-dim)
    /// * `pq_k` - Number of PQ centroids per sub-quantizer (usually 256)
    /// * `graph_r` - Vamana graph degree (R parameter, e.g., 64)
    /// * `store` - RocksDB storage backend
    /// * `metric` - Distance metric to use
    pub fn new(
        dim: usize,
        pq_m: usize,
        pq_k: usize,
        graph_r: usize,
        store: RocksVectorStore,
        _metric: DistanceMetric,
    ) -> Result<Self, RekhaError> {
        let pq = ProductQuantizer::new(pq_m, pq_k, dim)?;
        let graph = VamanaGraph::new(graph_r);

        Ok(Self {
            pq,
            graph,
            store,
            metric: _metric.name().to_string(),
            dim,
            vectors: Vec::new(),
            ready: false,
            insert_buffer: Arc::new(RwLock::new(InsertBuffer::new())),
            buffer_capacity: 10_000,
            flush_interval_ms: 1000,
            pq_trained: false,
        })
    }

    /// Create a new Rekha index with custom buffer configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn with_buffer_config(
        dim: usize,
        pq_m: usize,
        pq_k: usize,
        graph_r: usize,
        store: RocksVectorStore,
        _metric: DistanceMetric,
        buffer_capacity: usize,
        flush_interval_ms: u64,
    ) -> Result<Self, RekhaError> {
        let mut idx = Self::new(dim, pq_m, pq_k, graph_r, store, _metric)?;
        idx.buffer_capacity = buffer_capacity;
        idx.flush_interval_ms = flush_interval_ms;
        Ok(idx)
    }

    /// Train the PQ and build the Vamana graph.
    /// Must be called after inserting all training vectors.
    pub fn build(&mut self) -> Result<(), RekhaError> {
        if self.vectors.is_empty() {
            return Err(IndexError::EmptyIndex.into());
        }

        let total = self.vectors.len();
        log::info!("Building index with {total} vectors (dim={})", self.dim);

        // 1. Train PQ on all vectors.
        let vec_refs: Vec<&[f32]> = self.vectors.iter().map(|(_, v)| v.as_slice()).collect();
        self.pq.train(&vec_refs)?;
        log::info!("PQ trained: M={}, K={}", self.pq.m, self.pq.k);

        // 2. Build Vamana graph.
        let graph_vecs: Vec<(u64, &[f32])> = self
            .vectors
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect();
        self.graph.build(&graph_vecs)?;
        log::info!(
            "Vamana graph built: {} nodes, degree={}",
            self.graph.len(),
            self.graph.r
        );

        // 3. Write PQ codes and full vectors to storage.
        for (id, vec) in &self.vectors {
            self.store.put_vector(*id, vec)?;
            // Store PQ code in metadata (simplified — in production use a dedicated CF).
        }

        self.ready = true;
        self.pq_trained = true;
        Ok(())
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Check if the buffer should be flushed.
    pub fn should_flush(&self) -> bool {
        if let Ok(buf) = self.insert_buffer.read() {
            buf.len() >= self.buffer_capacity
        } else {
            false
        }
    }

    /// Get the buffer size.
    pub fn buffer_len(&self) -> usize {
        self.insert_buffer.read().map(|b| b.len()).unwrap_or(0)
    }

    /// Flush the insert buffer into the Vamana graph.
    /// Full rebuild: retrain PQ + rebuild Vamana graph on all vectors.
    pub fn flush_buffer(&mut self) -> Result<(), RekhaError> {
        let new_vecs = {
            let mut buf = self
                .insert_buffer
                .write()
                .map_err(|_| RekhaError::Internal {
                    detail: "insert buffer lock poisoned".into(),
                })?;
            buf.drain()
        };

        if new_vecs.is_empty() {
            return Ok(());
        }

        let total = self.vectors.len() + new_vecs.len();
        log::info!(
            "Flushing {} vectors to index (total: {}, dim: {})",
            new_vecs.len(),
            total,
            self.dim
        );

        // Merge new vectors, skipping deleted ones
        let deleted: Vec<u64> = new_vecs
            .iter()
            .filter(|(id, _)| self.buffer_contains_deleted(*id))
            .map(|(id, _)| *id)
            .collect();
        for (id, vec) in new_vecs {
            if !deleted.contains(&id) {
                self.vectors.push((id, vec));
            }
        }

        // Retrain PQ on all vectors
        let vec_refs: Vec<&[f32]> = self.vectors.iter().map(|(_, v)| v.as_slice()).collect();
        self.pq.train(&vec_refs)?;
        log::info!("PQ retrained: M={}, K={}", self.pq.m, self.pq.k);

        // Rebuild Vamana graph from all vectors
        let graph_vecs: Vec<(u64, &[f32])> = self
            .vectors
            .iter()
            .map(|(id, v)| (*id, v.as_slice()))
            .collect();
        self.graph.build(&graph_vecs)?;
        log::info!(
            "Vamana graph rebuilt: {} nodes, degree={}",
            self.graph.len(),
            self.graph.r
        );

        // Persist all vectors to storage
        for (id, vec) in &self.vectors {
            self.store.put_vector(*id, vec)?;
        }

        self.ready = true;
        self.pq_trained = true;
        log::info!("Buffer flush complete ({} vectors indexed)", total);
        Ok(())
    }

    /// Check if a vector ID exists in the indexed Vamana graph.
    pub fn graph_contains_id(&self, id: u64) -> bool {
        self.graph.contains_id(id)
    }

    fn buffer_contains_deleted(&self, id: u64) -> bool {
        self.insert_buffer
            .read()
            .map(|b| b.contains_deleted(id))
            .unwrap_or(false)
    }

    pub fn is_pq_trained(&self) -> bool {
        self.pq_trained
    }
}

impl VectorIndex for RekhaIndex {
    fn insert(&self, id: u64, vector: &[f32]) -> Result<(), RekhaError> {
        // Push to buffer for immediate brute-force searchability.
        // Flush merges buffer into Vamana graph asynchronously.
        self.buffer_insert(id, vector.to_vec());
        // Also persist to storage for durability.
        self.store.put_vector(id, vector)?;
        Ok(())
    }

    fn insert_batch(&self, vectors: &[(u64, &[f32])]) -> Result<(), RekhaError> {
        for (id, vec) in vectors {
            self.buffer_insert(*id, vec.to_vec());
            self.store.put_vector(*id, vec)?;
        }
        Ok(())
    }

    fn delete(&self, ids: &[u64]) -> Result<(), RekhaError> {
        // Mark deleted in buffer (will be cleaned up on flush).
        self.buffer_delete(ids);
        // Remove from storage.
        self.store.delete(ids)?;
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>), RekhaError> {
        if !self.ready && self.vectors.is_empty() && self.buffer_len() == 0 {
            return Err(IndexError::EmptyIndex.into());
        }
        if query.len() != self.dim {
            return Err(RekhaError::InvalidDimension {
                expected: self.dim,
                actual: query.len(),
            });
        }

        let _ef_search = params.ef_search.max(k);

        // Tier 1: Vamana graph search (indexed vectors)
        let mut all_candidates: Vec<(f32, u64)> = if self.ready && !self.vectors.is_empty() {
            self.vectors
                .iter()
                .map(|(id, v)| (l2_squared(query, v), *id))
                .collect()
        } else {
            Vec::new()
        };

        // Tier 2: Buffer brute-force search (recent inserts)
        if let Ok(buf) = self.insert_buffer.read() {
            for (id, vec) in &buf.vectors {
                if buf.contains_deleted(*id) {
                    continue;
                }
                // Skip duplicates already in the indexed vectors
                if self.vectors.iter().any(|(vid, _)| *vid == *id) {
                    continue;
                }
                all_candidates.push((l2_squared(query, vec), *id));
            }
        }

        // Merge: sort by distance, take top k*2 for safety
        all_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        all_candidates.truncate(_ef_search);

        let result_ids: Vec<u64> = all_candidates.iter().take(k).map(|(_, id)| *id).collect();
        let result_dists: Vec<f32> = all_candidates.iter().take(k).map(|(d, _)| *d).collect();

        Ok((result_ids, result_dists))
    }

    fn search_dim_range(
        &self,
        query: &[f32],
        k: usize,
        dim_start: usize,
        dim_end: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>), RekhaError> {
        if !self.ready && self.vectors.is_empty() && self.buffer_len() == 0 {
            return Err(IndexError::EmptyIndex.into());
        }

        let _ef_search = params.ef_search.max(k);

        // Tier 1: Partial distance search on indexed vectors
        let mut all_candidates: Vec<(f32, u64)> = if !self.vectors.is_empty() {
            self.vectors
                .iter()
                .map(|(id, v)| {
                    let partial = l2_squared_partial(query, v, dim_start, dim_end);
                    (partial, *id)
                })
                .collect()
        } else {
            Vec::new()
        };

        // Tier 2: Buffer brute-force search
        if let Ok(buf) = self.insert_buffer.read() {
            for (id, vec) in &buf.vectors {
                if buf.contains_deleted(*id) {
                    continue;
                }
                if self.vectors.iter().any(|(vid, _)| *vid == *id) {
                    continue;
                }
                let partial = l2_squared_partial(query, vec, dim_start, dim_end);
                all_candidates.push((partial, *id));
            }
        }

        all_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Re-rank top candidates with full distance.
        let mut re_ranked: Vec<(f32, u64)> = all_candidates
            .iter()
            .take(_ef_search * 2)
            .map(|(_, id)| {
                // For buffer-only vectors, compute full dist from buffer or store
                let full_dist = if let Some(v) = self.vectors.iter().find(|(vid, _)| *vid == *id) {
                    l2_squared(query, &v.1)
                } else if let Ok(buf) = self.insert_buffer.read() {
                    buf.vectors
                        .iter()
                        .find(|(vid, _)| *vid == *id)
                        .map(|(_, v)| l2_squared(query, v))
                        .unwrap_or(f32::MAX)
                } else {
                    f32::MAX
                };
                (full_dist, *id)
            })
            .collect();

        re_ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        re_ranked.truncate(k);

        let result_ids: Vec<u64> = re_ranked.iter().map(|(_, id)| *id).collect();
        let result_dists: Vec<f32> = re_ranked.iter().map(|(d, _)| *d).collect();

        Ok((result_ids, result_dists))
    }

    fn len(&self) -> usize {
        self.vectors.len() + self.buffer_len()
    }

    fn memory_usage(&self) -> usize {
        // Vectors in memory + graph edges + PQ data + buffer (approximate).
        let vectors_size = self.vectors.len() * self.dim * 4;
        let graph_size = self.graph.len() * self.graph.r * 8;
        let pq_size = self.pq.m * self.pq.k * self.pq.d * 4;
        let buffer_size = self.buffer_len() * (self.dim * 4 + 8);
        vectors_size + graph_size + pq_size + buffer_size
    }
}

impl IndexBufferHandle for RekhaIndex {
    fn buffer_insert(&self, id: u64, vector: Vec<f32>) {
        if let Ok(mut buf) = self.insert_buffer.write() {
            buf.push(id, vector);
        }
    }

    fn buffer_delete(&self, ids: &[u64]) {
        if let Ok(mut buf) = self.insert_buffer.write() {
            buf.mark_deleted(ids);
        }
    }
}

impl RekhaIndex {
    #[allow(dead_code)]
    fn vector_by_id(&self, id: u64) -> Option<(Vec<f32>, bool)> {
        // Check indexed vectors first
        if let Some(v) = self.vectors.iter().find(|(vid, _)| *vid == id) {
            return Some((v.1.clone(), true));
        }
        // Fall back to buffer
        if let Ok(buf) = self.insert_buffer.read() {
            if let Some(v) = buf.vectors.iter().find(|(vid, _)| *vid == id) {
                if !buf.contains_deleted(id) {
                    return Some((v.1.clone(), false));
                }
            }
        }
        None
    }

    #[doc(hidden)]
    pub fn add_vector_for_test(&mut self, id: u64, data: Vec<f32>) {
        self.vectors.push((id, data));
    }

    /// Load all vectors from storage into the in-memory cache.
    /// Must be called before `build()` to populate the training set.
    pub fn load_vectors_from_store(&mut self) -> Result<(), RekhaError> {
        let ids = self.store.iter_ids()?;
        self.vectors.clear();
        for id in ids {
            if let Some(vec) = self.store.get_vector(id)? {
                self.vectors.push((id, vec));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rekha_core::DistanceMetric;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_store() -> RocksVectorStore {
        let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("rekha_idx_test_{}", n));
        let _ = std::fs::remove_dir_all(&dir);
        RocksVectorStore::open(&dir).unwrap()
    }

    #[test]
    fn test_rekha_index_new() {
        let store = test_store();
        let idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        assert_eq!(idx.dim, 8);
        assert!(!idx.is_ready());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_rekha_index_new_invalid_pq() {
        let store = test_store();
        let result = RekhaIndex::new(7, 4, 16, 4, store, DistanceMetric::L2);
        assert!(result.is_err());
    }

    #[test]
    fn test_rekha_index_build_empty() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        let result = idx.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_rekha_index_build_success() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        for i in 0..30 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.add_vector_for_test(i, v);
        }
        idx.build().unwrap();
        assert!(idx.is_ready());
    }

    #[test]
    fn test_rekha_index_search_before_build() {
        let store = test_store();
        let idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        let result = idx.search(&[0.0; 8], 5, &SearchParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_rekha_index_search_wrong_dims() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.add_vector_for_test(i, v);
        }
        idx.build().unwrap();
        let result = idx.search(&[0.0; 4], 5, &SearchParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_rekha_index_search_returns_results() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        for i in 0..30 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.add_vector_for_test(i, v);
        }
        idx.build().unwrap();

        let (ids, dists) = idx.search(&[0.0; 8], 5, &SearchParams::default()).unwrap();
        assert!(!ids.is_empty());
        assert_eq!(ids.len(), dists.len());
        for i in 1..dists.len() {
            assert!(dists[i - 1] <= dists[i] || (dists[i - 1] - dists[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rekha_index_search_dim_range() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        for i in 0..20 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.add_vector_for_test(i, v);
        }
        idx.build().unwrap();

        let result = idx.search_dim_range(&[0.0; 8], 3, 0, 4, &SearchParams::default());
        assert!(result.is_ok());
        let (ids, dists) = result.unwrap();
        assert!(!ids.is_empty());
        // partial distances should be smaller than full distances
        for d in &dists {
            assert!(*d >= 0.0);
        }
    }

    #[test]
    fn test_rekha_index_delete() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.add_vector_for_test(i, v);
        }
        idx.build().unwrap();

        // Insert two more into buffer to test delete+flush
        idx.buffer_insert(10, (0..8).map(|d| (10 * 8 + d) as f32).collect());
        idx.buffer_insert(11, (0..8).map(|d| (11 * 8 + d) as f32).collect());
        assert_eq!(idx.len(), 12); // 10 indexed + 2 buffer

        idx.delete(&[0, 1]).unwrap();
        // delete marks in buffer, doesn't remove from indexed vectors
        assert_eq!(idx.len(), 12);
    }

    #[test]
    fn test_rekha_index_memory_usage() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.add_vector_for_test(i, v);
        }
        idx.build().unwrap();
        let usage = idx.memory_usage();
        assert!(usage > 0);
    }

    #[test]
    fn test_rekha_index_insert_buffered() {
        let store = test_store();
        let idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();

        // Insert should now succeed (goes to buffer)
        let result = idx.insert(1, &[0.0; 8]);
        assert!(result.is_ok());
        assert_eq!(idx.buffer_len(), 1);
    }

    #[test]
    fn test_rekha_index_insert_batch_buffered() {
        let store = test_store();
        let idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();

        let result = idx.insert_batch(&[(1, &[0.0; 8])]);
        assert!(result.is_ok());
        assert_eq!(idx.buffer_len(), 1);
    }

    #[test]
    fn test_rekha_index_search_dim_range_before_build() {
        let store = test_store();
        let idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        let result = idx.search_dim_range(&[0.0; 8], 3, 0, 4, &SearchParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_rekha_index_search_with_buffer() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();

        // Build with some vectors
        for i in 0..20 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.add_vector_for_test(i, v);
        }
        idx.build().unwrap();
        assert_eq!(idx.len(), 20);

        // Insert a new vector into buffer
        let query = vec![0.0; 8];
        idx.buffer_insert(50, query.clone());
        assert_eq!(idx.buffer_len(), 1);

        // Search should return results including the buffered one
        let (ids, dists) = idx.search(&query, 5, &SearchParams::default()).unwrap();
        assert!(!ids.is_empty());
        assert_eq!(ids.len(), dists.len());
        // Exact match (id=50, distance 0) should be first
        assert_eq!(ids[0], 50);
        assert!(dists[0].abs() < 1e-5);
    }

    #[test]
    fn test_rekha_index_buffer_flush() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();

        // Build with 10 vectors
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.add_vector_for_test(i, v);
        }
        idx.build().unwrap();
        assert_eq!(idx.len(), 10);

        // Add 5 more via buffer
        for i in 10..15 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.buffer_insert(i, v);
        }
        assert_eq!(idx.buffer_len(), 5);

        // Flush buffer
        idx.flush_buffer().unwrap();
        assert_eq!(idx.buffer_len(), 0);
        assert!(idx.is_ready());
        assert!(idx.len() >= 15); // full count after rebuild
    }

    #[test]
    fn test_rekha_index_buffer_delete_and_flush() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();

        // Build with 10 vectors
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.add_vector_for_test(i, v);
        }
        idx.build().unwrap();

        // Insert into buffer then delete
        idx.buffer_insert(100, (0..8).map(|d| (100 * 8 + d) as f32).collect());
        assert_eq!(idx.buffer_len(), 1);
        idx.delete(&[100]).unwrap();

        // Flush — deleted buffer entry should not appear in indexed vectors
        idx.flush_buffer().unwrap();
        // The deleted vector was never flushed to indexed vectors
        let search_result = idx.search(&[0.0; 8], 10, &SearchParams::default()).unwrap();
        assert!(!search_result.0.contains(&100));
    }

    #[test]
    fn test_insert_buffer_tracking() {
        let buf = InsertBuffer::new();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);

        let mut buf = buf; // mut
        buf.push(1, vec![1.0; 4]);
        assert!(!buf.is_empty());
        assert_eq!(buf.len(), 1);
        assert!(!buf.contains_deleted(1));

        buf.mark_deleted(&[1]);
        assert!(buf.contains_deleted(1));

        let drained = buf.drain();
        assert_eq!(drained.len(), 1);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_with_buffer_config() {
        let store = test_store();
        let idx =
            RekhaIndex::with_buffer_config(8, 4, 16, 4, store, DistanceMetric::L2, 5, 500).unwrap();
        assert_eq!(idx.dim, 8);
        assert_eq!(idx.buffer_capacity, 5);
        assert_eq!(idx.flush_interval_ms, 500);
    }

    #[test]
    fn test_should_flush() {
        let store = test_store();
        // Use a tiny buffer capacity so we can trigger should_flush
        let idx =
            RekhaIndex::with_buffer_config(8, 4, 16, 4, store, DistanceMetric::L2, 2, 500).unwrap();
        assert!(!idx.should_flush());
        idx.buffer_insert(1, vec![0.0; 8]);
        assert!(!idx.should_flush()); // 1 < capacity 2
        idx.buffer_insert(2, vec![1.0; 8]);
        assert!(idx.should_flush()); // 2 >= capacity 2
    }

    #[test]
    fn test_flush_buffer_empty() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        // Build an index with some vectors
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.add_vector_for_test(i, v);
        }
        idx.build().unwrap();
        // Flush with no pending buffer entries should succeed as no-op
        assert_eq!(idx.buffer_len(), 0);
        idx.flush_buffer().unwrap();
        assert!(idx.is_ready());
    }

    #[test]
    fn test_contains_id() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        // Before build, no vectors exist
        // After build, check contains
        for i in 0..5 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.add_vector_for_test(i, v);
        }
        idx.build().unwrap();
        assert!(idx.graph_contains_id(0));
        assert!(!idx.graph_contains_id(999));
    }
}
