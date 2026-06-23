use rekha_core::distance::{l2_squared, l2_squared_partial};
use rekha_core::{IndexError, RekhaError};

use rekha_quant::KMeans;
use rekha_quant::ProductQuantizer;

pub struct IvfIndex {
    pub centroids: Vec<Vec<f32>>,
    pub inverted_lists: Vec<Vec<(u64, Vec<f32>)>>,
    pub pq: Option<ProductQuantizer>,
    pub dim: usize,
    pub nlist: usize,
    pub nprobe: usize,
    pub trained: bool,
}

impl IvfIndex {
    pub fn new(nlist: usize, nprobe: usize, dim: usize) -> Self {
        Self {
            centroids: Vec::new(),
            inverted_lists: Vec::new(),
            pq: None,
            dim,
            nlist,
            nprobe,
            trained: false,
        }
    }

    pub fn build(
        vectors: &[(u64, Vec<f32>)],
        nlist: usize,
        nprobe: usize,
        pq_m: usize,
        pq_k: usize,
        dim: usize,
    ) -> Result<Self, RekhaError> {
        if vectors.is_empty() {
            return Err(IndexError::EmptyIndex.into());
        }

        let km = KMeans::new(nlist);
        let vec_refs: Vec<&[f32]> = vectors.iter().map(|(_, v)| v.as_slice()).collect();
        let centroids = km.train(&vec_refs, dim)?;

        let mut inverted_lists: Vec<Vec<(u64, Vec<f32>)>> = vec![Vec::new(); nlist];
        for (id, vec) in vectors {
            let cluster = km.assign(vec, &centroids);
            inverted_lists[cluster].push((*id, vec.clone()));
        }

        let mut pq = ProductQuantizer::new(pq_m, pq_k, dim)?;
        if vectors.len() >= nlist * 2 {
            pq.train(&vec_refs)?;
        }

        Ok(Self {
            centroids,
            inverted_lists,
            pq: Some(pq),
            dim,
            nlist,
            nprobe,
            trained: true,
        })
    }

    pub fn add(&mut self, id: u64, vector: Vec<f32>) {
        if !self.trained || self.centroids.is_empty() {
            return;
        }
        let km = KMeans::new(self.nlist);
        let cluster = km.assign(&vector, &self.centroids);
        self.inverted_lists[cluster].push((id, vector));
    }

    pub fn reassign(&mut self) {
        let all_vectors: Vec<(u64, Vec<f32>)> = self
            .inverted_lists
            .iter()
            .flat_map(|list| list.iter().cloned())
            .collect();

        let km = KMeans::new(self.nlist);
        let vec_refs: Vec<&[f32]> = all_vectors.iter().map(|(_, v)| v.as_slice()).collect();
        if let Ok(new_centroids) = km.train(&vec_refs, self.dim) {
            self.centroids = new_centroids;
            let mut new_lists: Vec<Vec<(u64, Vec<f32>)>> = vec![Vec::new(); self.nlist];
            let km2 = KMeans::new(self.nlist);
            for (id, vec) in &all_vectors {
                let cluster = km2.assign(vec, &self.centroids);
                new_lists[cluster].push((*id, vec.clone()));
            }
            self.inverted_lists = new_lists;
        }
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: Option<usize>,
    ) -> Result<(Vec<u64>, Vec<f32>), RekhaError> {
        if !self.trained {
            return Err(IndexError::NotTrained { component: "IVF" }.into());
        }

        let probe = nprobe.unwrap_or(self.nprobe).min(self.nlist);
        let centroid_dists = self.nearest_centroids(query, probe);

        let mut candidates: Vec<(f32, u64)> = Vec::new();
        for &(cid, _) in &centroid_dists {
            for (id, vec) in &self.inverted_lists[cid] {
                let dist = l2_squared(query, vec);
                candidates.push((dist, *id));
            }
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut seen = std::collections::HashSet::new();
        let mut result_ids = Vec::with_capacity(k);
        let mut result_dists = Vec::with_capacity(k);
        for (dist, id) in candidates {
            if seen.insert(id) && result_ids.len() < k {
                result_ids.push(id);
                result_dists.push(dist);
            }
        }

        Ok((result_ids, result_dists))
    }

    pub fn search_with_pq(
        &self,
        query: &[f32],
        k: usize,
        nprobe: Option<usize>,
    ) -> Result<(Vec<u64>, Vec<f32>), RekhaError> {
        if !self.trained {
            return Err(IndexError::NotTrained { component: "IVF" }.into());
        }

        let probe = nprobe.unwrap_or(self.nprobe).min(self.nlist);
        let centroid_dists = self.nearest_centroids(query, probe);

        let pq = match self.pq {
            Some(ref pq) if pq.trained => pq,
            _ => return self.search(query, k, nprobe),
        };

        let table = pq.distance_table(query);

        let mut candidates: Vec<(f32, u64)> = Vec::new();
        let _km = KMeans::new(self.nlist);
        for &(cid, cdist) in &centroid_dists {
            if let Some(pq_codes) = self.pq_codes_for_cluster(cid) {
                for (idx, (id, _)) in self.inverted_lists[cid].iter().enumerate() {
                    let pq_dist = ProductQuantizer::adc_distance(&pq_codes[idx], &table);
                    let total = cdist + pq_dist;
                    candidates.push((total, *id));
                }
            } else {
                for (id, vec) in &self.inverted_lists[cid] {
                    let dist = l2_squared(query, vec);
                    candidates.push((dist, *id));
                }
            }
        }

        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut seen = std::collections::HashSet::new();
        let mut result_ids = Vec::with_capacity(k);
        let mut result_dists = Vec::with_capacity(k);
        for (dist, id) in candidates {
            if seen.insert(id) && result_ids.len() < k {
                result_ids.push(id);
                result_dists.push(dist);
            }
        }

        Ok((result_ids, result_dists))
    }

    pub fn search_dim_range(
        &self,
        query: &[f32],
        k: usize,
        dim_start: usize,
        dim_end: usize,
        nprobe: Option<usize>,
    ) -> Result<(Vec<u64>, Vec<f32>), RekhaError> {
        if !self.trained {
            return Err(IndexError::NotTrained { component: "IVF" }.into());
        }

        let probe = nprobe.unwrap_or(self.nprobe).min(self.nlist);
        let centroid_dists = self.nearest_centroids(query, probe);

        let mut candidates: Vec<(f32, u64)> = Vec::new();
        for &(cid, _) in &centroid_dists {
            for (id, vec) in &self.inverted_lists[cid] {
                let partial = l2_squared_partial(query, vec, dim_start, dim_end);
                candidates.push((partial, *id));
            }
        }

        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut seen = std::collections::HashSet::new();
        let mut result_ids = Vec::with_capacity(k);
        let mut result_dists = Vec::with_capacity(k);
        for (partial, id) in candidates {
            if seen.insert(id) && result_ids.len() < k {
                result_ids.push(id);
                result_dists.push(partial);
            }
        }

        Ok((result_ids, result_dists))
    }

    pub fn nearest_centroids(&self, query: &[f32], nprobe: usize) -> Vec<(usize, f32)> {
        let mut dists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_squared(query, c)))
            .collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        dists.truncate(nprobe);
        dists
    }

    fn pq_codes_for_cluster(&self, cluster: usize) -> Option<Vec<Vec<u8>>> {
        let pq = self.pq.as_ref()?;
        if !pq.trained {
            return None;
        }
        let list = &self.inverted_lists[cluster];
        let codes: Vec<Vec<u8>> = list
            .iter()
            .filter_map(|(_, v)| pq.encode(v).ok())
            .collect();
        if codes.len() != list.len() {
            return None;
        }
        Some(codes)
    }

    pub fn len(&self) -> usize {
        self.inverted_lists.iter().map(|l| l.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn memory_usage(&self) -> usize {
        let centroids_size = self.centroids.len() * self.dim * 4;
        let lists_size: usize = self
            .inverted_lists
            .iter()
            .map(|l| l.len() * (self.dim * 4 + 8))
            .sum();
        let pq_size = self
            .pq
            .as_ref()
            .map(|pq| pq.m * pq.k * pq.d * 4)
            .unwrap_or(0);
        centroids_size + lists_size + pq_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_dataset() -> Vec<(u64, Vec<f32>)> {
        (0..50)
            .map(|i| {
                let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
                (i as u64, v)
            })
            .collect()
    }

    #[test]
    fn test_ivf_build() {
        let data = small_dataset();
        let idx = IvfIndex::build(&data, 4, 2, 2, 8, 8).unwrap();
        assert!(idx.trained);
        assert_eq!(idx.nlist, 4);
        assert_eq!(idx.centroids.len(), 4);
        let total = idx.inverted_lists.iter().map(|l| l.len()).sum::<usize>();
        assert_eq!(total, 50);
    }

    #[test]
    fn test_ivf_search() {
        let data = small_dataset();
        let idx = IvfIndex::build(&data, 4, 2, 2, 8, 8).unwrap();
        let query = vec![0.0; 8];
        let (ids, dists) = idx.search(&query, 5, None).unwrap();
        assert!(!ids.is_empty());
        assert_eq!(ids.len(), dists.len());
        for i in 1..dists.len() {
            assert!(dists[i - 1] <= dists[i] || (dists[i - 1] - dists[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ivf_search_before_train() {
        let idx = IvfIndex::new(4, 2, 8);
        let result = idx.search(&[0.0; 8], 5, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ivf_search_dim_range() {
        let data = small_dataset();
        let idx = IvfIndex::build(&data, 4, 2, 2, 8, 8).unwrap();
        let query = vec![0.0; 8];
        let (ids, partials) = idx.search_dim_range(&query, 3, 0, 4, None).unwrap();
        assert!(!ids.is_empty());
        for &d in &partials {
            assert!(d >= 0.0);
        }
    }

    #[test]
    fn test_ivf_add() {
        let data = small_dataset();
        let mut idx = IvfIndex::build(&data, 4, 2, 2, 8, 8).unwrap();
        let before = idx.len();
        idx.add(100, vec![99.0; 8]);
        assert_eq!(idx.len(), before + 1);
    }

    #[test]
    fn test_ivf_nearest_centroids() {
        let data = small_dataset();
        let idx = IvfIndex::build(&data, 4, 2, 2, 8, 8).unwrap();
        let nearest = idx.nearest_centroids(&[0.0; 8], 2);
        assert_eq!(nearest.len(), 2);
        for &(cid, _) in &nearest {
            assert!(cid < idx.nlist);
        }
    }

    #[test]
    fn test_ivf_reassign() {
        let data = small_dataset();
        let mut idx = IvfIndex::build(&data, 4, 2, 2, 8, 8).unwrap();
        idx.reassign();
        assert!(idx.trained);
        assert_eq!(idx.centroids.len(), 4);
        let total = idx.len();
        assert_eq!(total, 50);
    }

    #[test]
    fn test_ivf_empty_build() {
        let result = IvfIndex::build(&[], 4, 2, 2, 8, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_ivf_memory_usage() {
        let data = small_dataset();
        let idx = IvfIndex::build(&data, 4, 2, 2, 8, 8).unwrap();
        let usage = idx.memory_usage();
        assert!(usage > 0);
    }

    #[test]
    fn test_ivf_is_empty() {
        let idx = IvfIndex::new(4, 2, 8);
        assert!(idx.is_empty());
    }

    #[test]
    fn test_ivf_search_with_pq() {
        let data = small_dataset();
        let idx = IvfIndex::build(&data, 4, 4, 2, 8, 8).unwrap();
        let query = vec![0.0; 8];
        let (ids, _) = idx.search_with_pq(&query, 5, None).unwrap();
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_ivf_nprobe_limited() {
        let data = small_dataset();
        let idx = IvfIndex::build(&data, 4, 2, 2, 8, 8).unwrap();
        let (ids_a, _) = idx.search(&[0.0; 8], 5, Some(4)).unwrap();
        let (ids_b, _) = idx.search(&[0.0; 8], 5, Some(1)).unwrap();
        assert!(!ids_a.is_empty());
        assert!(!ids_b.is_empty());
    }

    #[test]
    fn test_ivf_search_deduplicates() {
        let data = small_dataset();
        let idx = IvfIndex::build(&data, 4, 4, 2, 8, 8).unwrap();
        let (ids, _) = idx.search(&[0.0; 8], 50, None).unwrap();
        let mut unique = ids.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(ids.len(), unique.len());
    }
}
