use rand::Rng;
use rekha_core::RekhaError;

pub struct KMeans {
    pub centroids: Vec<Vec<f32>>,
    pub n_clusters: usize,
    pub max_iter: usize,
    pub tolerance: f32,
    pub dim: usize,
}

impl KMeans {
    pub fn new(n_clusters: usize, dim: usize, max_iter: usize, tolerance: f32) -> Self {
        KMeans {
            centroids: Vec::new(),
            n_clusters,
            max_iter,
            tolerance,
            dim,
        }
    }

    pub fn is_trained(&self) -> bool {
        !self.centroids.is_empty()
    }

    pub fn fit(&mut self, data: &[Vec<f32>]) -> Result<(), RekhaError> {
        if data.is_empty() || data.len() < self.n_clusters {
            return Err(RekhaError::InvalidArgument(format!(
                "need at least {} samples for KMeans",
                self.n_clusters
            )));
        }
        let dim = data[0].len();
        if dim != self.dim {
            return Err(RekhaError::InvalidArgument(format!(
                "expected dim {}, got {}",
                self.dim, dim
            )));
        }

        self.centroids = Self::kmeans_plus_plus(data, self.n_clusters);
        let mut rng = rand::thread_rng();

        for _iter in 0..self.max_iter {
            let assignments = self.assign(data);
            let new_centroids = self.recompute(data, &assignments, &mut rng);
            let diff = self.centroid_shift(&new_centroids);
            self.centroids = new_centroids;
            if diff < self.tolerance {
                break;
            }
        }

        Ok(())
    }

    fn kmeans_plus_plus(data: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
        let first_idx = rng.gen_range(0..data.len());
        centroids.push(data[first_idx].clone());

        let mut min_dists = vec![f32::MAX; data.len()];

        for _ in 1..k {
            let mut total = 0.0f32;
            for (i, point) in data.iter().enumerate() {
                let d = Self::min_distance(point, &centroids);
                min_dists[i] = d;
                total += d;
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
            centroids.push(data[chosen].clone());
        }

        centroids
    }

    fn min_distance(point: &[f32], centroids: &[Vec<f32>]) -> f32 {
        centroids
            .iter()
            .map(|c| {
                point
                    .iter()
                    .zip(c.iter())
                    .map(|(x, y)| {
                        let d = x - y;
                        d * d
                    })
                    .sum::<f32>()
            })
            .fold(f32::MAX, f32::min)
    }

    fn assign(&self, data: &[Vec<f32>]) -> Vec<usize> {
        data.iter()
            .map(|point| {
                let mut best = 0;
                let mut best_dist = f32::MAX;
                for (j, centroid) in self.centroids.iter().enumerate() {
                    let d: f32 = point
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
            })
            .collect()
    }

    fn recompute(
        &self,
        data: &[Vec<f32>],
        assignments: &[usize],
        rng: &mut impl Rng,
    ) -> Vec<Vec<f32>> {
        let mut new_centroids = vec![vec![0.0f32; self.dim]; self.n_clusters];
        let mut counts = vec![0usize; self.n_clusters];

        for (i, &cluster) in assignments.iter().enumerate() {
            for (j, &val) in data[i].iter().enumerate() {
                new_centroids[cluster][j] += val;
            }
            counts[cluster] += 1;
        }

        for (i, centroid) in new_centroids.iter_mut().enumerate() {
            if counts[i] > 0 {
                let inv = 1.0 / counts[i] as f32;
                for val in centroid.iter_mut() {
                    *val *= inv;
                }
            } else {
                *centroid = data[rng.gen_range(0..data.len())].clone();
            }
        }

        new_centroids
    }

    fn centroid_shift(&self, new_centroids: &[Vec<f32>]) -> f32 {
        self.centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(old, new)| {
                old.iter()
                    .zip(new.iter())
                    .map(|(x, y)| {
                        let d = x - y;
                        d * d
                    })
                    .sum::<f32>()
            })
            .sum::<f32>()
            / self.n_clusters as f32
    }

    pub fn predict(&self, point: &[f32]) -> usize {
        let mut best = 0;
        let mut best_dist = f32::MAX;
        for (j, centroid) in self.centroids.iter().enumerate() {
            let d: f32 = point
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

    pub fn predict_with_distances(&self, point: &[f32]) -> Vec<(usize, f32)> {
        let mut distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(j, centroid)| {
                let d: f32 = point
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
        distances
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_fit_and_predict() {
        let data: Vec<Vec<f32>> = vec![
            vec![1.0, 1.0],
            vec![1.5, 1.5],
            vec![2.0, 2.0],
            vec![10.0, 10.0],
            vec![10.5, 10.5],
            vec![11.0, 11.0],
        ];
        let mut km = KMeans::new(2, 2, 20, 1e-4);
        km.fit(&data).unwrap();
        assert!(km.is_trained());
        assert_eq!(km.centroids.len(), 2);

        let c1 = km.predict(&[1.0, 1.0]);
        let c2 = km.predict(&[10.0, 10.0]);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_kmeans_not_enough_data() {
        let data = vec![vec![1.0, 1.0]];
        let mut km = KMeans::new(2, 2, 20, 1e-4);
        assert!(km.fit(&data).is_err());
    }

    #[test]
    fn test_predict_with_distances() {
        let data = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let mut km = KMeans::new(2, 2, 5, 1e-4);
        km.fit(&data).unwrap();
        let dists = km.predict_with_distances(&[0.0, 0.0]);
        assert_eq!(dists.len(), 2);
        assert!(dists[0].1 <= dists[1].1);
    }
}
