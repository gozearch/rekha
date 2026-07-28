use std::sync::Arc;

use rekha_core::{IvfConfig, RekhaError, ScoredPoint, SearchParams};
use rekha_quant::{KMeans, ProductQuantizer};
use rekha_storage::RekhaStore;

pub struct DiskIvfIndex {
    collection: String,
    store: Arc<RekhaStore>,
    centroids: Vec<Vec<f32>>,
    pq: Option<ProductQuantizer>,
    config: IvfConfig,
}

impl DiskIvfIndex {
    pub fn new(collection: &str, store: Arc<RekhaStore>, config: IvfConfig) -> Self {
        DiskIvfIndex {
            collection: collection.to_string(),
            store,
            centroids: Vec::new(),
            pq: None,
            config,
        }
    }

    pub fn config(&self) -> &IvfConfig {
        &self.config
    }

    pub fn is_trained(&self) -> bool {
        !self.centroids.is_empty() && self.pq.is_some()
    }

    pub fn collection_name(&self) -> &str {
        &self.collection
    }

    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    pub fn pq(&self) -> Option<&ProductQuantizer> {
        self.pq.as_ref()
    }

    pub fn build(&mut self, sample_vectors: &[Vec<f32>]) -> Result<(), RekhaError> {
        if sample_vectors.is_empty() {
            return Err(RekhaError::InvalidArgument(
                "no sample vectors for training".into(),
            ));
        }

        let dim = self.config.dim as usize;
        for v in sample_vectors {
            if v.len() != dim {
                return Err(RekhaError::InvalidArgument(format!(
                    "expected dimension {}, got {}",
                    dim,
                    v.len()
                )));
            }
        }

        let nlist = self.config.nlist as usize;

        let mut kmeans = KMeans::new(nlist, dim, 20, 1e-4);
        kmeans.fit(sample_vectors)?;
        self.centroids = kmeans.centroids;

        self.store
            .store_centroids(&self.collection, &self.centroids)?;

        let mut pq = ProductQuantizer::new(self.config.pq_m as usize, self.config.pq_k as usize);
        pq.train(sample_vectors)?;

        self.store
            .store_pq_codebook(&self.collection, pq.m, pq.k, pq.sub_dim, &pq.codebooks)?;

        self.pq = Some(pq);

        Ok(())
    }

    pub fn add(&self, id: u64, vector: &[f32]) -> Result<(), RekhaError> {
        if !self.is_trained() {
            return Err(RekhaError::Index("index not trained".into()));
        }

        let centroid_id = self.assign_to_centroid(vector) as u32;

        let pq = self.pq.as_ref().unwrap();
        let pq_code = pq.encode(vector);

        self.store
            .inverted_list_append(&self.collection, centroid_id, id, &pq_code)?;
        self.store
            .store_assignment(&self.collection, id, centroid_id)?;

        Ok(())
    }

    pub fn remove(&self, id: u64) -> Result<(), RekhaError> {
        if let Some(centroid_id) = self.store.load_assignment(&self.collection, id)? {
            self.store
                .inverted_list_remove(&self.collection, centroid_id, id)?;
            self.store.delete_assignment(&self.collection, id)?;
        }
        Ok(())
    }

    pub fn search(
        &self,
        query: &[f32],
        params: &SearchParams,
    ) -> Result<Vec<ScoredPoint>, RekhaError> {
        if !self.is_trained() {
            return Err(RekhaError::Index("index not trained".into()));
        }

        let pq = self.pq.as_ref().unwrap();
        let distance_table = pq.distance_table(query);
        let nprobe = params.nprobe as usize;

        let centroid_dists = self.nearest_centroids(query, nprobe);

        let k = params.k as usize;
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        #[derive(Debug, PartialEq)]
        struct HeapEntry(f32, u64);

        impl Eq for HeapEntry {}

        impl Ord for HeapEntry {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.0
                    .partial_cmp(&other.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        impl PartialOrd for HeapEntry {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::with_capacity(k + 1);

        for (centroid_id, _dist) in &centroid_dists {
            let entries = self
                .store
                .inverted_list_scan(&self.collection, *centroid_id as u32)?;
            for (entry_id, pq_code) in &entries {
                let dist = pq.adc_distance(&distance_table, pq_code);

                heap.push(Reverse(HeapEntry(dist, *entry_id)));
                if heap.len() > k {
                    heap.pop();
                }
            }
        }

        let mut results: Vec<ScoredPoint> = Vec::with_capacity(k);
        for Reverse(HeapEntry(score, id)) in heap.into_sorted_vec() {
            if let Ok(Some(record)) = self.store.get_vector(&self.collection, id) {
                if record.is_tombstone {
                    continue;
                }
            }
            results.push(ScoredPoint {
                id,
                score,
                payload: None,
                timestamp: 0,
            });
            if results.len() >= k {
                break;
            }
        }

        if params.include_payloads {
            for sp in &mut results {
                if let Ok(Some(payload)) = self.store.get_payload(&self.collection, sp.id) {
                    sp.payload = Some(payload);
                }
            }
        }

        Ok(results)
    }

    pub fn rebuild(&mut self) -> Result<(), RekhaError> {
        let all_vids = self.collect_all_ids()?;
        if all_vids.len() < self.config.nlist as usize {
            return Err(RekhaError::Index(format!(
                "need at least {} vectors to rebuild, have {}",
                self.config.nlist,
                all_vids.len()
            )));
        }

        let mut sample = Vec::new();
        for &id in &all_vids {
            if let Ok(Some(record)) = self.store.get_vector(&self.collection, id) {
                if !record.is_tombstone {
                    sample.push(record.data);
                }
            }
        }

        if sample.len() < self.config.nlist as usize {
            return Err(RekhaError::Index(format!(
                "not enough non-tombstone vectors to rebuild: need {}, have {}",
                self.config.nlist,
                sample.len()
            )));
        }

        let nlist = self.config.nlist as usize;
        let dim = self.config.dim as usize;
        let mut kmeans = KMeans::new(nlist, dim, 20, 1e-4);
        kmeans.fit(&sample)?;
        self.centroids = kmeans.centroids;
        self.store
            .store_centroids(&self.collection, &self.centroids)?;

        let mut pq_obj =
            ProductQuantizer::new(self.config.pq_m as usize, self.config.pq_k as usize);
        pq_obj.train(&sample)?;
        self.store.store_pq_codebook(
            &self.collection,
            pq_obj.m,
            pq_obj.k,
            pq_obj.sub_dim,
            &pq_obj.codebooks,
        )?;
        self.pq = Some(pq_obj);

        for cid in 0..nlist as u32 {
            let entries = self.store.inverted_list_scan(&self.collection, cid)?;
            for (vid, _) in &entries {
                self.store
                    .inverted_list_remove(&self.collection, cid, *vid)?;
            }
        }

        for &vid in &all_vids {
            if let Ok(Some(record)) = self.store.get_vector(&self.collection, vid) {
                if !record.is_tombstone {
                    let centroid_id = self.assign_to_centroid(&record.data) as u32;
                    let pq_ref = self.pq.as_ref().unwrap();
                    let pq_code = pq_ref.encode(&record.data);
                    self.store.inverted_list_append(
                        &self.collection,
                        centroid_id,
                        vid,
                        &pq_code,
                    )?;
                    self.store
                        .store_assignment(&self.collection, vid, centroid_id)?;
                }
            }
        }

        Ok(())
    }

    pub fn load_from_store(&mut self) -> Result<(), RekhaError> {
        self.centroids = self.store.load_centroids(&self.collection)?;

        let (m, k, sub_dim, codebooks) = self.store.load_pq_codebook(&self.collection)?;
        let mut pq = ProductQuantizer::new(m, k);
        pq.sub_dim = sub_dim;
        pq.codebooks = codebooks;
        pq.trained = true;
        self.pq = Some(pq);

        Ok(())
    }

    fn assign_to_centroid(&self, vector: &[f32]) -> usize {
        let mut best = 0;
        let mut best_dist = f32::MAX;
        for (j, centroid) in self.centroids.iter().enumerate() {
            let d: f32 = vector
                .iter()
                .zip(centroid.iter())
                .map(|(x, y)| {
                    let diff = x - y;
                    diff * diff
                })
                .sum();
            if d < best_dist {
                best_dist = d;
                best = j;
            }
        }
        best
    }

    fn nearest_centroids(&self, query: &[f32], nprobe: usize) -> Vec<(usize, f32)> {
        let mut distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(j, centroid)| {
                let d: f32 = query
                    .iter()
                    .zip(centroid.iter())
                    .map(|(x, y)| {
                        let diff = x - y;
                        diff * diff
                    })
                    .sum();
                (j, d)
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(nprobe);
        distances
    }

    fn collect_all_ids(&self) -> Result<Vec<u64>, RekhaError> {
        self.store.iterate_vector_ids(&self.collection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rekha_core::DistanceMetric;
    use tempfile::TempDir;

    fn setup_store() -> (TempDir, Arc<RekhaStore>) {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
        (dir, store)
    }

    fn make_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|i| (0..dim).map(|d| ((i * 1000 + d) as f32) / 1000.0).collect())
            .collect()
    }

    #[test]
    fn test_build_and_search() {
        let (_dir, store) = setup_store();
        let config = IvfConfig {
            dim: 4,
            nlist: 4,
            nprobe: 4,
            pq_m: 2,
            pq_k: 32,
            replication_factor: 3,
            distance_metric: DistanceMetric::L2,
        };
        let mut index = DiskIvfIndex::new("test", store.clone(), config.clone());

        let data = make_vectors(100, 4);
        index.build(&data).unwrap();
        assert!(index.is_trained());

        for (i, v) in data.iter().enumerate() {
            index.add(i as u64, v).unwrap();
        }

        let params = SearchParams {
            nprobe: 4,
            k: 3,
            include_payloads: false,
            pre_filter: None,
            local_only: false,
        };
        let results = index.search(&data[0], &params).unwrap();
        assert!(!results.is_empty(), "search should return results");
        assert!(results.len() <= 3, "results should not exceed k");
    }

    #[test]
    fn test_load_from_store() {
        let (_dir, store) = setup_store();
        let config = IvfConfig {
            dim: 4,
            nlist: 2,
            nprobe: 2,
            pq_m: 2,
            pq_k: 4,
            replication_factor: 3,
            distance_metric: DistanceMetric::L2,
        };

        let mut index = DiskIvfIndex::new("test", store.clone(), config.clone());
        let data = make_vectors(10, 4);
        index.build(&data).unwrap();

        let mut loaded = DiskIvfIndex::new("test", store.clone(), config.clone());
        loaded.load_from_store().unwrap();
        assert!(loaded.is_trained());
    }

    #[test]
    fn test_search_not_trained() {
        let (_dir, store) = setup_store();
        let config = IvfConfig::default();
        let index = DiskIvfIndex::new("test", store, config);
        let params = SearchParams::default();
        let result = index.search(&[0.0; 8], &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_empty_vectors() {
        let (_dir, store) = setup_store();
        let config = IvfConfig::default();
        let mut index = DiskIvfIndex::new("test", store, config);
        let result = index.build(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_rebuild() {
        let (_dir, store) = setup_store();
        let config = IvfConfig {
            dim: 4,
            nlist: 2,
            nprobe: 2,
            pq_m: 2,
            pq_k: 16,
            replication_factor: 3,
            distance_metric: DistanceMetric::L2,
        };

        {
            let mut index = DiskIvfIndex::new("test", store.clone(), config.clone());
            let data = make_vectors(20, 4);
            index.build(&data).unwrap();
            for (i, v) in data.iter().enumerate() {
                store.put_vector("test", i as u64, v, 1000, false).unwrap();
                index.add(i as u64, v).unwrap();
            }
        }

        let mut index = DiskIvfIndex::new("test", store.clone(), config.clone());
        index.load_from_store().unwrap();
        assert!(index.is_trained(), "index should be trained after load");
        index.rebuild().unwrap();
        assert!(index.is_trained(), "index should be trained after rebuild");
    }
}
