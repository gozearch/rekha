use rekha_core::{
    distance::l2_squared, DistanceMetric, IndexError, RekhaError, SearchParams,
    VectorIndex, VectorStoreBackend,
};
use rekha_storage::RocksVectorStore;

use crate::ivf::IvfIndex;
use crate::pq::ProductQuantizer;

use std::collections::HashSet;
use std::sync::{Arc, RwLock};

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

    fn len(&self) -> usize {
        self.vectors.len()
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

pub struct RekhaIndex {
    ivf: Option<IvfIndex>,
    pq: ProductQuantizer,
    store: RocksVectorStore,
    _metric: DistanceMetric,
    pub dim: usize,
    pub nlist: usize,
    pub nprobe: usize,
    all_vectors: Vec<(u64, Vec<f32>)>,
    ready: bool,
    insert_buffer: Arc<RwLock<InsertBuffer>>,
    buffer_capacity: usize,
    flush_interval_ms: u64,
}

impl RekhaIndex {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        nlist: usize,
        nprobe: usize,
        pq_m: usize,
        pq_k: usize,
        store: RocksVectorStore,
        metric: DistanceMetric,
    ) -> Result<Self, RekhaError> {
        let pq = ProductQuantizer::new(pq_m, pq_k, dim)?;
        Ok(Self {
            ivf: None,
            pq,
            store,
            _metric: metric,
            dim,
            nlist,
            nprobe,
            all_vectors: Vec::new(),
            ready: false,
            insert_buffer: Arc::new(RwLock::new(InsertBuffer::new())),
            buffer_capacity: 10_000,
            flush_interval_ms: 1000,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_buffer_config(
        dim: usize,
        nlist: usize,
        nprobe: usize,
        pq_m: usize,
        pq_k: usize,
        store: RocksVectorStore,
        metric: DistanceMetric,
        buffer_capacity: usize,
        flush_interval_ms: u64,
    ) -> Result<Self, RekhaError> {
        let mut idx = Self::new(dim, nlist, nprobe, pq_m, pq_k, store, metric)?;
        idx.buffer_capacity = buffer_capacity;
        idx.flush_interval_ms = flush_interval_ms;
        Ok(idx)
    }

    pub fn build(&mut self) -> Result<(), RekhaError> {
        let (buf_vectors, deleted_ids) = {
            let mut buf = self.insert_buffer.write().map_err(|_| RekhaError::Internal {
                detail: "insert buffer lock poisoned".into(),
            })?;
            let deleted = buf.deleted.clone();
            let drained = buf.drain();
            (drained, deleted)
        };
        for (id, vec) in buf_vectors {
            if !deleted_ids.contains(&id) {
                self.all_vectors.push((id, vec));
            }
        }

        if self.all_vectors.is_empty() {
            return Err(IndexError::EmptyIndex.into());
        }

        let total = self.all_vectors.len();
        let actual_nlist = self.nlist.min(total / 2).max(1);

        let mut ivf = IvfIndex::build(
            &self.all_vectors,
            actual_nlist,
            self.nprobe,
            self.pq.m,
            self.pq.k,
            self.dim,
        )?;

        let vec_refs: Vec<&[f32]> = self.all_vectors.iter().map(|(_, v)| v.as_slice()).collect();
        self.pq.train(&vec_refs)?;
        ivf.pq = Some(self.pq.clone_for_ref());

        for (id, vec) in &self.all_vectors {
            self.store.put_vector(*id, vec)?;
        }

        self.ivf = Some(ivf);
        self.ready = true;
        Ok(())
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }

    pub fn should_flush(&self) -> bool {
        self.insert_buffer
            .read()
            .map(|b| b.len() >= self.buffer_capacity)
            .unwrap_or(false)
    }

    pub fn buffer_len(&self) -> usize {
        self.insert_buffer.read().map(|b| b.len()).unwrap_or(0)
    }

    pub fn flush_buffer(&mut self) -> Result<(), RekhaError> {
        let new_vecs = {
            let mut buf = self.insert_buffer.write().map_err(|_| RekhaError::Internal {
                detail: "insert buffer lock poisoned".into(),
            })?;
            buf.drain()
        };

        if new_vecs.is_empty() {
            return Ok(());
        }

        let deleted: Vec<u64> = new_vecs
            .iter()
            .filter(|(id, _)| self.buffer_contains_deleted(*id))
            .map(|(id, _)| *id)
            .collect();

        for (id, vec) in new_vecs {
            if !deleted.contains(&id) {
                self.all_vectors.push((id, vec));
            }
        }

        self.build()?;
        Ok(())
    }

    fn buffer_contains_deleted(&self, id: u64) -> bool {
        self.insert_buffer
            .read()
            .map(|b| b.contains_deleted(id))
            .unwrap_or(false)
    }
}

impl VectorIndex for RekhaIndex {
    fn insert(&self, id: u64, vector: &[f32]) -> Result<(), RekhaError> {
        self.buffer_insert_internal(id, vector.to_vec());
        self.store.put_vector(id, vector)?;
        Ok(())
    }

    fn insert_batch(&self, vectors: &[(u64, &[f32])]) -> Result<(), RekhaError> {
        for (id, vec) in vectors {
            self.buffer_insert_internal(*id, vec.to_vec());
            self.store.put_vector(*id, vec)?;
        }
        Ok(())
    }

    fn delete(&self, ids: &[u64]) -> Result<(), RekhaError> {
        self.buffer_delete_internal(ids);
        self.store.delete(ids)?;
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>), RekhaError> {
        if !self.ready && self.buffer_len() == 0 {
            return Err(IndexError::EmptyIndex.into());
        }
        if query.len() != self.dim {
            return Err(RekhaError::InvalidDimension {
                expected: self.dim,
                actual: query.len(),
            });
        }

        let mut all_candidates: Vec<(f32, u64)> = Vec::new();

        if let Some(ref ivf) = self.ivf {
            let nprobe = params.nprobe.max(self.nprobe);
            if params.plan == rekha_core::PlanType::DimensionBased {
                let dim_per_group = self.dim / 4;
                for g in 0..4 {
                    let ds = g * dim_per_group;
                    let de = (ds + dim_per_group).min(self.dim);
                    if let Ok((ids, partials)) =
                        ivf.search_dim_range(query, k * 2, ds, de, Some(nprobe))
                    {
                        for (i, id) in ids.iter().enumerate() {
                            all_candidates.push((partials[i], *id));
                        }
                    }
                }
            } else {
                if let Ok((ids, dists)) = ivf.search(query, k * 2, Some(nprobe)) {
                    for (i, id) in ids.iter().enumerate() {
                        all_candidates.push((dists[i], *id));
                    }
                }
            }
        }

        if let Ok(buf) = self.insert_buffer.read() {
            for (id, vec) in &buf.vectors {
                if buf.contains_deleted(*id) {
                    continue;
                }
                if self.ready && self.ivf.as_ref().is_some_and(|ivf| {
                    ivf.inverted_lists.iter().any(|l| l.iter().any(|(vid, _)| *vid == *id))
                }) {
                    continue;
                }
                all_candidates.push((l2_squared(query, vec), *id));
            }
        }

        all_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let ef = params.ef_search.max(k);
        all_candidates.truncate(ef);

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
        if !self.ready && self.buffer_len() == 0 {
            return Err(IndexError::EmptyIndex.into());
        }

        let mut all_candidates: Vec<(f32, u64)> = Vec::new();

        if let Some(ref ivf) = self.ivf {
            let nprobe = params.nprobe.max(self.nprobe);
            if let Ok((ids, partials)) =
                ivf.search_dim_range(query, k * 2, dim_start, dim_end, Some(nprobe))
            {
                for (i, id) in ids.iter().enumerate() {
                    all_candidates.push((partials[i], *id));
                }
            }
        }

        if let Ok(buf) = self.insert_buffer.read() {
            for (id, vec) in &buf.vectors {
                if buf.contains_deleted(*id) {
                    continue;
                }
                if self.ready && self.ivf.as_ref().is_some_and(|ivf| {
                    ivf.inverted_lists.iter().any(|l| l.iter().any(|(vid, _)| *vid == *id))
                }) {
                    continue;
                }
                let partial = rekha_core::distance::l2_squared_partial(
                    query, vec, dim_start, dim_end,
                );
                all_candidates.push((partial, *id));
            }
        }

        all_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let ef = params.ef_search.max(k);
        all_candidates.truncate(ef);

        let mut re_ranked: Vec<(f32, u64)> = all_candidates
            .iter()
            .take(ef * 2)
            .map(|(_, id)| {
                let full_dist = if let Some(ref ivf) = self.ivf {
                    ivf.inverted_lists
                        .iter()
                        .flatten()
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
        self.all_vectors.len() + self.buffer_len()
    }

    fn memory_usage(&self) -> usize {
        let ivf_size = self
            .ivf
            .as_ref()
            .map(|ivf| ivf.memory_usage())
            .unwrap_or(0);
        let pq_size = self.pq.m * self.pq.k * self.pq.d * 4;
        let buffer_size = self.buffer_len() * (self.dim * 4 + 8);
        ivf_size + pq_size + buffer_size
    }

    fn centroids(&self) -> Vec<Vec<f32>> {
        self.ivf
            .as_ref()
            .map(|ivf| ivf.centroids.clone())
            .unwrap_or_default()
    }

    fn num_clusters(&self) -> usize {
        self.ivf.as_ref().map(|ivf| ivf.nlist).unwrap_or(0)
    }
}

impl RekhaIndex {
    pub fn ivf(&self) -> Option<&IvfIndex> {
        self.ivf.as_ref()
    }

    pub fn buffer_insert_internal(&self, id: u64, vector: Vec<f32>) {
        if let Ok(mut buf) = self.insert_buffer.write() {
            buf.push(id, vector);
        }
    }

    pub fn buffer_delete_internal(&self, ids: &[u64]) {
        if let Ok(mut buf) = self.insert_buffer.write() {
            buf.mark_deleted(ids);
        }
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
        let idx = RekhaIndex::new(8, 4, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        assert_eq!(idx.dim, 8);
        assert!(!idx.is_ready());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_rekha_index_new_invalid_pq() {
        let store = test_store();
        let result = RekhaIndex::new(7, 4, 2, 3, 16, store, DistanceMetric::L2);
        assert!(result.is_err());
    }

    #[test]
    fn test_rekha_index_build_empty() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        let result = idx.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_rekha_index_build_success() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        for i in 0..30 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.insert(i, &v).unwrap();
        }
        idx.build().unwrap();
        assert!(idx.is_ready());
    }

    #[test]
    fn test_rekha_index_search_before_build() {
        let store = test_store();
        let idx = RekhaIndex::new(8, 4, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        let result = idx.search(&[0.0; 8], 5, &SearchParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_rekha_index_search_wrong_dims() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.insert(i, &v).unwrap();
        }
        idx.build().unwrap();
        let result = idx.search(&[0.0; 4], 5, &SearchParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_rekha_index_search_returns_results() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 2, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        for i in 0..30 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.insert(i, &v).unwrap();
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
        let mut idx = RekhaIndex::new(8, 2, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        for i in 0..20 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.insert(i, &v).unwrap();
        }
        idx.build().unwrap();
        let result = idx.search_dim_range(&[0.0; 8], 3, 0, 4, &SearchParams::default());
        assert!(result.is_ok());
        let (ids, dists) = result.unwrap();
        assert!(!ids.is_empty());
        for d in &dists {
            assert!(*d >= 0.0);
        }
    }

    #[test]
    fn test_rekha_index_delete() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 2, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.insert(i, &v).unwrap();
        }
        idx.build().unwrap();
        idx.buffer_insert_internal(10, (0..8).map(|d| (10 * 8 + d) as f32).collect());
        assert_eq!(idx.len(), 11);
        idx.delete(&[0, 1]).unwrap();
        assert_eq!(idx.len(), 11);
    }

    #[test]
    fn test_rekha_index_insert_buffered() {
        let store = test_store();
        let idx = RekhaIndex::new(8, 4, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        let result = idx.insert(1, &[0.0; 8]);
        assert!(result.is_ok());
        assert_eq!(idx.buffer_len(), 1);
    }

    #[test]
    fn test_rekha_index_insert_batch_buffered() {
        let store = test_store();
        let idx = RekhaIndex::new(8, 4, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        let result = idx.insert_batch(&[(1, &[0.0; 8])]);
        assert!(result.is_ok());
        assert_eq!(idx.buffer_len(), 1);
    }

    #[test]
    fn test_rekha_index_search_with_buffer() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 2, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        for i in 0..20 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.insert(i, &v).unwrap();
        }
        idx.build().unwrap();
        let query = vec![0.0; 8];
        idx.buffer_insert_internal(50, query.clone());
        let (ids, dists) = idx.search(&query, 5, &SearchParams::default()).unwrap();
        assert!(!ids.is_empty());
        assert_eq!(ids.len(), dists.len());
    }

    #[test]
    fn test_rekha_index_buffer_flush() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 2, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.insert(i, &v).unwrap();
        }
        idx.build().unwrap();
        for i in 10..15 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.buffer_insert_internal(i, v);
        }
        assert_eq!(idx.buffer_len(), 5);
        idx.flush_buffer().unwrap();
        assert_eq!(idx.buffer_len(), 0);
        assert!(idx.is_ready());
    }

    #[test]
    fn test_rekha_index_memory_usage() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 2, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.insert(i, &v).unwrap();
        }
        idx.build().unwrap();
        let usage = idx.memory_usage();
        assert!(usage > 0);
    }

    #[test]
    fn test_with_buffer_config() {
        let store = test_store();
        let idx = RekhaIndex::with_buffer_config(8, 4, 2, 4, 16, store, DistanceMetric::L2, 5, 500)
            .unwrap();
        assert_eq!(idx.dim, 8);
        assert_eq!(idx.buffer_capacity, 5);
        assert_eq!(idx.flush_interval_ms, 500);
    }

    #[test]
    fn test_should_flush() {
        let store = test_store();
        let idx = RekhaIndex::with_buffer_config(8, 4, 2, 4, 16, store, DistanceMetric::L2, 2, 500)
            .unwrap();
        assert!(!idx.should_flush());
        idx.buffer_insert_internal(1, vec![0.0; 8]);
        assert!(!idx.should_flush());
        idx.buffer_insert_internal(2, vec![1.0; 8]);
        assert!(idx.should_flush());
    }

    #[test]
    fn test_flush_buffer_empty() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 2, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.insert(i, &v).unwrap();
        }
        idx.build().unwrap();
        assert_eq!(idx.buffer_len(), 0);
        idx.flush_buffer().unwrap();
        assert!(idx.is_ready());
    }

    #[test]
    fn test_search_buffer_only_no_indexed_vectors() {
        let store = test_store();
        let idx = RekhaIndex::new(4, 2, 2, 2, 8, store, DistanceMetric::L2).unwrap();
        idx.buffer_insert_internal(1, vec![0.0; 4]);
        idx.buffer_insert_internal(2, vec![1.0, 1.0, 1.0, 1.0]);
        let (ids, _dists) = idx.search(&[0.0; 4], 5, &SearchParams::default()).unwrap();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_centroids_after_build() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 2, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        for i in 0..20 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.insert(i, &v).unwrap();
        }
        idx.build().unwrap();
        let centroids = idx.centroids();
        assert_eq!(centroids.len(), 2);
    }

    #[test]
    fn test_num_clusters_after_build() {
        let store = test_store();
        let mut idx = RekhaIndex::new(8, 4, 2, 4, 16, store, DistanceMetric::L2).unwrap();
        for i in 0..30 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.insert(i, &v).unwrap();
        }
        idx.build().unwrap();
        assert!(idx.num_clusters() > 0);
    }
}
