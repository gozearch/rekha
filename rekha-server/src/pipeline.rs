use rekha_core::distance::l2_squared_partial;
use rekha_core::{RekhaError, ScoredPoint, SearchStats};
use rekha_index::IvfIndex;

use std::collections::{HashMap, HashSet};

pub struct DimensionPipeline {
    pub num_dim_groups: u32,
    pub dims_per_group: usize,
}

impl DimensionPipeline {
    pub fn new(num_dim_groups: u32, dim: usize) -> Self {
        let dims_per_group = if num_dim_groups > 0 {
            (dim / num_dim_groups as usize).max(1)
        } else {
            dim
        };
        Self {
            num_dim_groups,
            dims_per_group,
        }
    }

    pub fn execute(
        &self,
        query: &[f32],
        ivf: &IvfIndex,
        k: usize,
        nprobe: usize,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        let mut stats = SearchStats::default();
        let total_dim = query.len();

        let kth_best = self.prewarm(query, ivf, k);

        let mut survivors: HashMap<u64, f32> = HashMap::new();
        let mut all_ids: HashSet<u64> = HashSet::new();

        let centroid_dists = ivf.nearest_centroids(query, nprobe);
        for &(cid, _) in &centroid_dists {
            for (id, _) in &ivf.inverted_lists[cid] {
                all_ids.insert(*id);
                survivors.insert(*id, 0.0);
            }
        }

        let mut current_kth = kth_best;

        for g in 0..self.num_dim_groups {
            let ds = (g as usize) * self.dims_per_group;
            let de = (ds + self.dims_per_group).min(total_dim);

            let mut pruned: Vec<u64> = Vec::new();
            for (&id, running) in survivors.iter_mut() {
                let vec = self.find_vector(id, ivf);
                if let Some(v) = vec {
                    let partial = l2_squared_partial(query, v, ds, de);
                    *running += partial;
                    if *running > current_kth {
                        pruned.push(id);
                    }
                }
            }

            for id in &pruned {
                survivors.remove(id);
            }

            if !survivors.is_empty() {
                let mut sorted: Vec<(f32, u64)> =
                    survivors.iter().map(|(id, d)| (*d, *id)).collect();
                sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                if sorted.len() >= k {
                    current_kth = sorted[k - 1].0;
                }
            }
        }

        let mut candidates: Vec<ScoredPoint> = survivors
            .into_iter()
            .map(|(id, score)| ScoredPoint {
                id,
                score,
                payload: None,
            })
            .collect();
        candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        candidates.truncate(k);

        stats.vectors_scanned = all_ids.len() as u64;

        Ok((candidates, stats))
    }

    fn prewarm(&self, query: &[f32], ivf: &IvfIndex, k: usize) -> f32 {
        let centroid_dists = ivf.nearest_centroids(query, k.min(ivf.nlist));
        let mut dists: Vec<f32> = centroid_dists.iter().map(|&(_, d)| d).collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        dists.truncate(k);
        dists.last().copied().unwrap_or(f32::MAX)
    }

    fn find_vector<'a>(&self, id: u64, ivf: &'a IvfIndex) -> Option<&'a [f32]> {
        for list in &ivf.inverted_lists {
            if let Some(pos) = list.iter().position(|(vid, _)| *vid == id) {
                return Some(&list[pos].1);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rekha_index::IvfIndex;

    fn test_ivf() -> IvfIndex {
        let data: Vec<(u64, Vec<f32>)> = (0..40)
            .map(|i| {
                let v: Vec<f32> = (0..8).map(|d| ((i * 8 + d) as f32)).collect();
                (i as u64, v)
            })
            .collect();
        IvfIndex::build(&data, 4, 4, 2, 8, 8).unwrap()
    }

    #[test]
    fn test_pipeline_new() {
        let p = DimensionPipeline::new(4, 8);
        assert_eq!(p.num_dim_groups, 4);
        assert_eq!(p.dims_per_group, 2);
    }

    #[test]
    fn test_pipeline_execute() {
        let ivf = test_ivf();
        let p = DimensionPipeline::new(4, 8);
        let query = vec![0.0; 8];
        let (candidates, stats) = p.execute(&query, &ivf, 5, 4).unwrap();
        assert!(!candidates.is_empty());
        assert!(stats.vectors_scanned > 0);
        for i in 1..candidates.len() {
            assert!(
                candidates[i - 1].score <= candidates[i].score
                    || (candidates[i - 1].score - candidates[i].score).abs() < 1e-6
            );
        }
    }
}
