use rand::Rng;
use rand::SeedableRng;
use rekha_core::distance::l2_squared;
use rekha_core::RekhaError;

pub struct KMeans {
    pub k: usize,
    pub max_iter: usize,
    pub tolerance: f32,
    pub seed: u64,
}

impl KMeans {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iter: 20,
            tolerance: 1e-4,
            seed: 42,
        }
    }

    pub fn with_params(k: usize, max_iter: usize, tolerance: f32, seed: u64) -> Self {
        Self { k, max_iter, tolerance, seed }
    }

    pub fn train(&self, vectors: &[&[f32]], dim: usize) -> Result<Vec<Vec<f32>>, RekhaError> {
        if vectors.is_empty() || self.k == 0 {
            return Err(RekhaError::InvalidArgument(
                "k-means: no vectors or zero clusters".into(),
            ));
        }
        if self.k > vectors.len() {
            return Err(RekhaError::InvalidArgument(format!(
                "k-means: k={} > n={}", self.k, vectors.len()
            )));
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let n = vectors.len();

        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(self.k);

        let first_idx = rng.gen_range(0..n);
        centroids.push(vectors[first_idx].to_vec());

        let mut min_dists = vec![f32::MAX; n];
        for c in 1..self.k {
            let mut total = 0.0f32;
            for (i, v) in vectors.iter().enumerate() {
                let d = l2_squared(v, &centroids[c - 1]);
                if d < min_dists[i] {
                    min_dists[i] = d;
                }
                total += min_dists[i];
            }

            let threshold = rng.gen::<f32>() * total;
            let mut cumulative = 0.0f32;
            let mut chosen = 0;
            for (i, d) in min_dists.iter().enumerate() {
                cumulative += d;
                if cumulative >= threshold {
                    chosen = i;
                    break;
                }
            }
            centroids.push(vectors[chosen].to_vec());
        }

        let mut assignments = vec![0usize; n];

        for _iter in 0..self.max_iter {
            let mut changed = false;

            for (i, point) in vectors.iter().enumerate() {
                let nearest = (0..self.k)
                    .min_by(|&a, &b| {
                        let da = l2_squared(point, &centroids[a]);
                        let db = l2_squared(point, &centroids[b]);
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0);

                if assignments[i] != nearest {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            let mut sums = vec![vec![0.0f32; dim]; self.k];
            let mut counts = vec![0usize; self.k];

            for (i, point) in vectors.iter().enumerate() {
                let cluster = assignments[i];
                for d in 0..dim {
                    sums[cluster][d] += point[d];
                }
                counts[cluster] += 1;
            }

            let mut max_movement = 0.0f32;
            for c in 0..self.k {
                if counts[c] > 0 {
                    let new_centroid: Vec<f32> =
                        sums[c].iter().map(|s| s / counts[c] as f32).collect();
                    let movement = l2_squared(&centroids[c], &new_centroid);
                    if movement > max_movement {
                        max_movement = movement;
                    }
                    centroids[c] = new_centroid;
                }
            }

            if max_movement < self.tolerance {
                break;
            }
        }

        Ok(centroids)
    }

    pub fn assign(&self, vector: &[f32], centroids: &[Vec<f32>]) -> usize {
        self.compute_distances(vector, centroids)
            .into_iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    pub fn compute_distances(&self, vector: &[f32], centroids: &[Vec<f32>]) -> Vec<f32> {
        centroids
            .iter()
            .map(|c| l2_squared(vector, c))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_convergence() {
        let km = KMeans::new(2);
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.2],
            vec![5.0, 5.0], vec![5.1, 5.1], vec![5.2, 5.2],
        ];
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let centroids = km.train(&refs, 2).unwrap();
        assert_eq!(centroids.len(), 2);
        assert_eq!(centroids[0].len(), 2);
    }

    #[test]
    fn test_kmeans_assign() {
        let km = KMeans::new(2);
        let centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        assert_eq!(km.assign(&[1.0, 1.0], &centroids), 0);
        assert_eq!(km.assign(&[9.0, 9.0], &centroids), 1);
    }

    #[test]
    fn test_kmeans_empty_vectors() {
        let km = KMeans::new(3);
        let result = km.train(&[], 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans_k_too_large() {
        let km = KMeans::new(10);
        let data: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32; 2]).collect();
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let result = km.train(&refs, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let km = KMeans::new(1);
        let data: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32; 4]).collect();
        let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
        let centroids = km.train(&refs, 4).unwrap();
        assert_eq!(centroids.len(), 1);
    }

    #[test]
    fn test_kmeans_compute_distances() {
        let km = KMeans::new(2);
        let centroids = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let dists = km.compute_distances(&[0.0, 0.0], &centroids);
        assert!((dists[0] - 0.0).abs() < 1e-6);
        assert!((dists[1] - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_kmeans_with_params() {
        let km = KMeans::with_params(3, 5, 1e-2, 123);
        assert_eq!(km.k, 3);
        assert_eq!(km.max_iter, 5);
        assert!((km.tolerance - 1e-2).abs() < 1e-6);
        assert_eq!(km.seed, 123);
    }
}
