use rekha_core::{
    distance::{l2_squared, l2_squared_partial}, IndexError, RekhaError, SearchParams,
};

use crate::ivf::IvfIndex;
use crate::pq::ProductQuantizer;

use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

struct InsertBuffer {
    vectors: Vec<(u64, u64, Vec<f32>)>,
    deleted: HashSet<u64>,
}

#[allow(dead_code)]
impl InsertBuffer {
    fn new() -> Self {
        Self { vectors: Vec::new(), deleted: HashSet::new() }
    }

    fn len(&self) -> usize { self.vectors.len() }

    fn push(&mut self, id: u64, timestamp: u64, vector: Vec<f32>) {
        self.vectors.push((id, timestamp, vector));
    }

    fn mark_deleted(&mut self, ids: &[u64]) {
        for id in ids { self.deleted.insert(*id); }
    }

    fn contains_deleted(&self, id: u64) -> bool {
        self.deleted.contains(&id)
    }

    fn drain(&mut self) -> Vec<(u64, u64, Vec<f32>)> {
        let mut drained = Vec::new();
        std::mem::swap(&mut self.vectors, &mut drained);
        self.deleted.clear();
        drained
    }
}

pub struct CollectionState {
    pub ivf: Option<IvfIndex>,
    buffer: InsertBuffer,
    pub pq: ProductQuantizer,
    pub dim: usize,
    pub nlist: usize,
    pub nprobe: usize,
    pub all_vectors: Vec<(u64, u64, Vec<f32>)>,
    pub ready: bool,
}

pub struct RekhaIndex {
    collections: RwLock<HashMap<String, CollectionState>>,
    buffer_capacity: usize,
}

impl RekhaIndex {
    pub fn new() -> Result<Self, RekhaError> {
        Ok(Self {
            collections: RwLock::new(HashMap::new()),
            buffer_capacity: 10_000,
        })
    }

    pub fn create_collection(
        &self, name: &str, dim: usize, nlist: usize, nprobe: usize,
    ) -> Result<(), RekhaError> {
        let pq = ProductQuantizer::new(4.min(dim), 256, dim)?;
        let state = CollectionState {
            ivf: None,
            buffer: InsertBuffer::new(),
            pq,
            dim,
            nlist,
            nprobe,
            all_vectors: Vec::new(),
            ready: false,
        };
        let mut cols = self.collections.write().map_err(|_| RekhaError::Internal {
            detail: "collection lock poisoned".into(),
        })?;
        cols.insert(name.to_string(), state);
        Ok(())
    }

    pub fn drop_collection(&self, name: &str) -> Result<(), RekhaError> {
        let mut cols = self.collections.write().map_err(|_| RekhaError::Internal {
            detail: "collection lock poisoned".into(),
        })?;
        cols.remove(name);
        Ok(())
    }

    pub fn has_collection(&self, name: &str) -> bool {
        self.collections.read().ok().map(|c| c.contains_key(name)).unwrap_or(false)
    }

    pub fn collection_names(&self) -> Vec<String> {
        self.collections.read().ok().map(|c| c.keys().cloned().collect()).unwrap_or_default()
    }

    pub fn collection_dim(&self, name: &str) -> Result<usize, RekhaError> {
        let cols = self.collections.read().map_err(|_| RekhaError::Internal {
            detail: "collection lock poisoned".into(),
        })?;
        let state = cols.get(name).ok_or_else(|| RekhaError::NotFound(name.into()))?;
        Ok(state.dim)
    }

    pub fn insert(
        &self, collection: &str, id: u64, timestamp: u64, vector: &[f32],
    ) -> Result<(), RekhaError> {
        let mut cols = self.collections.write().map_err(|_| RekhaError::Internal {
            detail: "collection lock poisoned".into(),
        })?;
        let state = cols.get_mut(collection).ok_or_else(|| {
            RekhaError::NotFound(collection.into())
        })?;

        if vector.len() != state.dim {
            return Err(RekhaError::InvalidDimension {
                expected: state.dim, actual: vector.len(),
            });
        }

        state.buffer.push(id, timestamp, vector.to_vec());
        Ok(())
    }

    pub fn search(
        &self, collection: &str, query: &[f32], k: usize, params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>), RekhaError> {
        let cols = self.collections.read().map_err(|_| RekhaError::Internal {
            detail: "collection lock poisoned".into(),
        })?;
        let state = cols.get(collection).ok_or_else(|| {
            RekhaError::NotFound(collection.into())
        })?;

        if query.len() != state.dim {
            return Err(RekhaError::InvalidDimension {
                expected: state.dim, actual: query.len(),
            });
        }

        let mut all_candidates: Vec<(f32, u64)> = Vec::new();

        if let Some(ref ivf) = state.ivf {
            let nprobe = params.nprobe.max(state.nprobe);
            if let Ok((ids, dists)) = ivf.search(query, k * 2, Some(nprobe)) {
                for (i, id) in ids.iter().enumerate() {
                    all_candidates.push((dists[i], *id));
                }
            }
        }

        for (id, _ts, vec) in &state.buffer.vectors {
            if state.buffer.contains_deleted(*id) { continue; }
            if state.ready && state.ivf.as_ref().is_some_and(|ivf| {
                ivf.inverted_lists.iter().any(|l| l.iter().any(|(vid, _)| *vid == *id))
            }) { continue; }
            all_candidates.push((l2_squared(query, vec), *id));
        }

        all_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let ef = params.ef_search.max(k);
        all_candidates.truncate(ef);

        let result_ids: Vec<u64> = all_candidates.iter().take(k).map(|(_, id)| *id).collect();
        let result_dists: Vec<f32> = all_candidates.iter().take(k).map(|(d, _)| *d).collect();
        Ok((result_ids, result_dists))
    }

    pub fn search_dim_range(
        &self, collection: &str, query: &[f32], k: usize,
        dim_start: usize, dim_end: usize, params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>), RekhaError> {
        let cols = self.collections.read().map_err(|_| RekhaError::Internal {
            detail: "collection lock poisoned".into(),
        })?;
        let state = cols.get(collection).ok_or_else(|| {
            RekhaError::NotFound(collection.into())
        })?;

        let mut all_candidates: Vec<(f32, u64)> = Vec::new();

        if let Some(ref ivf) = state.ivf {
            let nprobe = params.nprobe.max(state.nprobe);
            if let Ok((ids, partials)) =
                ivf.search_dim_range(query, k * 2, dim_start, dim_end, Some(nprobe))
            {
                for (i, id) in ids.iter().enumerate() {
                    all_candidates.push((partials[i], *id));
                }
            }
        }

        for (id, _ts, vec) in &state.buffer.vectors {
            if state.buffer.contains_deleted(*id) { continue; }
            if state.ready && state.ivf.as_ref().is_some_and(|ivf| {
                ivf.inverted_lists.iter().any(|l| l.iter().any(|(vid, _)| *vid == *id))
            }) { continue; }
            let partial = l2_squared_partial(query, vec, dim_start, dim_end);
            all_candidates.push((partial, *id));
        }

        all_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let ef = params.ef_search.max(k);
        all_candidates.truncate(ef);

        let mut re_ranked: Vec<(f32, u64)> = all_candidates
            .iter()
            .take(ef * 2)
            .map(|(_, id)| {
                let full_dist = state.ivf.as_ref().map_or(f32::MAX, |ivf| {
                    ivf.inverted_lists.iter().flatten()
                        .find(|(vid, _)| *vid == *id)
                        .map(|(_, v)| l2_squared(query, v))
                        .unwrap_or(f32::MAX)
                });
                (full_dist, *id)
            })
            .collect();

        re_ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        re_ranked.truncate(k);

        let result_ids: Vec<u64> = re_ranked.iter().map(|(_, id)| *id).collect();
        let result_dists: Vec<f32> = re_ranked.iter().map(|(d, _)| *d).collect();
        Ok((result_ids, result_dists))
    }

    pub fn len(&self, collection: &str) -> Result<usize, RekhaError> {
        let cols = self.collections.read().map_err(|_| RekhaError::Internal {
            detail: "collection lock poisoned".into(),
        })?;
        let state = cols.get(collection).ok_or_else(|| RekhaError::NotFound(collection.into()))?;
        Ok(state.all_vectors.len() + state.buffer.len())
    }

    pub fn memory_usage(&self) -> usize {
        0 // approximate, not critical
    }

    pub fn should_flush(&self, collection: &str) -> Result<bool, RekhaError> {
        let cols = self.collections.read().map_err(|_| RekhaError::Internal {
            detail: "collection lock poisoned".into(),
        })?;
        if let Some(state) = cols.get(collection) {
            Ok(state.buffer.len() >= self.buffer_capacity)
        } else {
            Ok(false)
        }
    }

    pub fn buffer_len(&self, collection: &str) -> Result<usize, RekhaError> {
        let cols = self.collections.read().map_err(|_| RekhaError::Internal {
            detail: "collection lock poisoned".into(),
        })?;
        if let Some(state) = cols.get(collection) {
            Ok(state.buffer.len())
        } else {
            Ok(0)
        }
    }

    pub fn flush_buffer(&self, collection: &str) -> Result<(), RekhaError> {
        let mut cols = self.collections.write().map_err(|_| RekhaError::Internal {
            detail: "collection lock poisoned".into(),
        })?;
        let state = cols.get_mut(collection).ok_or_else(|| {
            RekhaError::NotFound(collection.into())
        })?;

        let new_vecs = state.buffer.drain();
        if new_vecs.is_empty() { return Ok(()); }

        for (id, ts, vec) in new_vecs {
            if !state.buffer.contains_deleted(id) {
                state.all_vectors.push((id, ts, vec));
            }
        }

        if state.all_vectors.is_empty() {
            return Err(IndexError::EmptyIndex.into());
        }

        let actual_nlist = state.nlist.min(state.all_vectors.len() / 2).max(1);
        let vecs_for_ivf: Vec<(u64, Vec<f32>)> = state.all_vectors.iter()
            .map(|(id, _, vec)| (*id, vec.clone()))
            .collect();
        let mut ivf = IvfIndex::build(
            &vecs_for_ivf, actual_nlist, state.nprobe,
            state.pq.m, state.pq.k, state.dim,
        )?;

        let vec_refs: Vec<&[f32]> = state.all_vectors.iter().map(|(_, _, v)| v.as_slice()).collect();
        state.pq.train(&vec_refs)?;
        ivf.pq = Some(state.pq.clone_for_ref());
        state.ivf = Some(ivf);
        state.ready = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rekha_core::SearchParams;
    use rekha_storage::RocksVectorStore;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_store() -> RocksVectorStore {
        let n = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("rekha_idx_test_{}", n));
        let _ = std::fs::remove_dir_all(&dir);
        RocksVectorStore::open(&dir).unwrap()
    }

    #[test]
    fn test_create_and_drop_collection() {
        let store = test_store();
        let idx = RekhaIndex::new().unwrap();
        idx.create_collection("test", 8, 4, 2).unwrap();
        assert!(idx.has_collection("test"));
        assert_eq!(idx.collection_names(), vec!["test"]);
        idx.drop_collection("test").unwrap();
        assert!(!idx.has_collection("test"));
    }

    #[test]
    fn test_insert_and_search() {
        let store = test_store();
        let idx = RekhaIndex::new().unwrap();
        idx.create_collection("c1", 8, 4, 2).unwrap();
        for i in 0..30 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.insert("c1", i, 100, &v).unwrap();
        }
        idx.flush_buffer("c1").unwrap();
        let (ids, dists) = idx.search("c1", &[0.0; 8], 5, &SearchParams::default()).unwrap();
        assert!(!ids.is_empty());
        assert_eq!(ids.len(), dists.len());
    }

    #[test]
    fn test_wrong_dim_rejected() {
        let store = test_store();
        let idx = RekhaIndex::new().unwrap();
        idx.create_collection("c1", 8, 4, 2).unwrap();
        let result = idx.insert("c1", 1, 0, &[0.0; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_nonexistent() {
        let store = test_store();
        let idx = RekhaIndex::new().unwrap();
        let result = idx.search("nonexistent", &[0.0; 8], 5, &SearchParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_collections() {
        let store = test_store();
        let idx = RekhaIndex::new().unwrap();
        idx.create_collection("a", 4, 2, 1).unwrap();
        idx.create_collection("b", 8, 4, 2).unwrap();
        assert!(idx.has_collection("a"));
        assert!(idx.has_collection("b"));
        let mut names = idx.collection_names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_dim_validation_on_search() {
        let store = test_store();
        let idx = RekhaIndex::new().unwrap();
        idx.create_collection("c1", 8, 4, 2).unwrap();
        let result = idx.search("c1", &[0.0; 4], 5, &SearchParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_search_dim_range() {
        let store = test_store();
        let idx = RekhaIndex::new().unwrap();
        idx.create_collection("c1", 8, 2, 2).unwrap();
        for i in 0..20 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            idx.insert("c1", i, 0, &v).unwrap();
        }
        idx.flush_buffer("c1").unwrap();
        let result = idx.search_dim_range("c1", &[0.0; 8], 3, 0, 4, &SearchParams::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_collection_dim() {
        let store = test_store();
        let idx = RekhaIndex::new().unwrap();
        idx.create_collection("c1", 128, 16, 4).unwrap();
        assert_eq!(idx.collection_dim("c1").unwrap(), 128);
    }

    #[test]
    fn test_flush_empty_buffer() {
        let store = test_store();
        let idx = RekhaIndex::new().unwrap();
        idx.create_collection("c1", 8, 2, 2).unwrap();
        idx.flush_buffer("c1").unwrap(); // no-op, should not error
    }

    #[test]
    fn test_len_empty() {
        let store = test_store();
        let idx = RekhaIndex::new().unwrap();
        idx.create_collection("c1", 8, 2, 2).unwrap();
        assert_eq!(idx.len("c1").unwrap(), 0);
    }
}
