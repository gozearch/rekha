use rekha_core::{
    distance::{l2_squared, l2_squared_partial},
    DistanceMetric, IndexError, RekhaError, SearchParams, VectorIndex, VectorStoreBackend,
};
use rekha_storage::RocksVectorStore;

use crate::pq::ProductQuantizer;
use crate::vamana::VamanaGraph;

/// The main Rekha index, combining PQ compression with Vamana graph search.
///
/// Query flow:
/// 1. PQ distance table is computed from the query (fast, in memory)
/// 2. Vamana graph search uses PQ-ADC distances for candidate selection
/// 3. Top candidates are re-ranked with full-precision distances (from disk)
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
    /// In-memory cache of vectors for graph construction.
    vectors: Vec<(u64, Vec<f32>)>,
    /// Whether the index is ready for search.
    ready: bool,
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
        })
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
        Ok(())
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }
}

impl VectorIndex for RekhaIndex {
    fn insert(&self, _id: u64, _vector: &[f32]) -> Result<(), RekhaError> {
        // For simplicity, we rebuild the index on bulk insertion.
        // Production: use dynamic insertion into the graph.
        Err(IndexError::GraphBuild {
            detail: "use insert_batch for index construction".into(),
        }
        .into())
    }

    fn insert_batch(&self, _vectors: &[(u64, &[f32])]) -> Result<(), RekhaError> {
        Err(IndexError::GraphBuild {
            detail: "use build() after adding vectors to the builder".into(),
        }
        .into())
    }

    fn delete(&self, ids: &[u64]) -> Result<(), RekhaError> {
        // Mark as deleted in storage (compaction handles removal).
        self.store.delete(ids)?;
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>), RekhaError> {
        if !self.ready {
            return Err(IndexError::EmptyIndex.into());
        }
        if query.len() != self.dim {
            return Err(RekhaError::InvalidDimension {
                expected: self.dim,
                actual: query.len(),
            });
        }

        let _ef_search = params.ef_search.max(k);

        // Compute full-precision distances (simplified; production uses PQ + graph).
        let mut distances: Vec<(f32, u64)> = self
            .vectors
            .iter()
            .map(|(id, v)| (l2_squared(query, v), *id))
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.truncate(_ef_search);

        // Re-rank with full precision (already done above for this simplified version).
        let result_ids: Vec<u64> = distances.iter().take(k).map(|(_, id)| *id).collect();
        let result_dists: Vec<f32> = distances.iter().take(k).map(|(d, _)| *d).collect();

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
        if !self.ready {
            return Err(IndexError::EmptyIndex.into());
        }

        let _ef_search = params.ef_search.max(k);

        // Partial distance search (only over the given dimension range).
        let mut distances: Vec<(f32, u64)> = self
            .vectors
            .iter()
            .map(|(id, v)| {
                let partial = l2_squared_partial(query, v, dim_start, dim_end);
                (partial, *id)
            })
            .collect();

        // Apply early-stop: if a vector's partial distance already exceeds
        // the k-th best full distance, we skip it for re-ranking.
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Re-rank top candidates with full distance.
        let mut re_ranked: Vec<(f32, u64)> = distances
            .iter()
            .take(_ef_search * 2) // Take more for re-ranking safety
            .map(|(_, id)| {
                let full_dist = l2_squared(query, self.vector_by_id(*id));
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
        self.vectors.len()
    }

    fn memory_usage(&self) -> usize {
        // Vectors in memory + graph edges + PQ data (approximate).
        let vectors_size = self.vectors.len() * self.dim * 4;
        let graph_size = self.graph.len() * self.graph.r * 8;
        let pq_size = self.pq.m * self.pq.k * self.pq.d * 4;
        vectors_size + graph_size + pq_size
    }
}

impl RekhaIndex {
    fn vector_by_id(&self, id: u64) -> &[f32] {
        self.vectors
            .iter()
            .find(|(vid, _)| *vid == id)
            .map(|(_, v)| v.as_slice())
            .unwrap_or(&[])
    }

    #[doc(hidden)]
    pub fn add_vector_for_test(&mut self, id: u64, data: Vec<f32>) {
        self.vectors.push((id, data));
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
        assert_eq!(idx.len(), 10);

        idx.delete(&[0, 1]).unwrap();
        // delete only removes from storage, not from in-memory vectors
        assert_eq!(idx.len(), 10);
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
    fn test_rekha_index_insert_returns_error() {
        let store = test_store();
        let idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        let result = idx.insert(1, &[0.0; 8]);
        assert!(result.is_err());
    }

    #[test]
    fn test_rekha_index_insert_batch_returns_error() {
        let store = test_store();
        let idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        let result = idx.insert_batch(&[(1, &[0.0; 8])]);
        assert!(result.is_err());
    }

    #[test]
    fn test_rekha_index_search_dim_range_before_build() {
        let store = test_store();
        let idx = RekhaIndex::new(8, 4, 16, 4, store, DistanceMetric::L2).unwrap();
        let result = idx.search_dim_range(&[0.0; 8], 3, 0, 4, &SearchParams::default());
        assert!(result.is_err());
    }
}
