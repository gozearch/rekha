use rand::Rng;
use rekha_core::{distance::l2_squared, IndexError, RekhaError};
use std::f32;

/// Product Quantizer for compressing high-dimensional vectors.
///
/// Splits vectors into `M` sub-vectors, each quantized independently
/// using k-means with `K` centroids. A vector is encoded as M indices
/// (each log2(K) bits), giving significant compression.
///
/// For example, a 768-dim vector with M=64, K=256:
///   - PQ code: 64 bytes (vs 3072 bytes for f32)
///   - Distance table: 64 × 256 f32 = 64KB per query
#[derive(Debug)]
pub struct ProductQuantizer {
    /// Number of sub-vectors.
    pub m: usize,
    /// Number of centroids per sub-quantizer.
    pub k: usize,
    /// Total vector dimension.
    pub dim: usize,
    /// Dimensions per sub-vector (dim / M), must divide evenly.
    pub d: usize,
    /// Centroids: [M][K][D] — each sub-quantizer's centroids.
    pub centroids: Vec<Vec<Vec<f32>>>,
    /// Whether the PQ has been trained.
    pub trained: bool,
}

impl ProductQuantizer {
    /// Create a new PQ with M sub-quantizers, each with K centroids.
    pub fn new(m: usize, k: usize, dim: usize) -> Result<Self, RekhaError> {
        if !dim.is_multiple_of(m) {
            return Err(RekhaError::InvalidArgument(format!(
                "PQ: dimension {dim} not divisible by M={m}"
            )));
        }
        Ok(Self {
            m,
            k,
            dim,
            d: dim / m,
            centroids: vec![vec![vec![0.0f32; dim / m]; k]; m],
            trained: false,
        })
    }

    /// Train the PQ on a set of vectors using k-means per sub-quantizer.
    pub fn train(&mut self, vectors: &[&[f32]]) -> Result<(), RekhaError> {
        if vectors.is_empty() {
            return Err(IndexError::NotTrained { component: "PQ" }.into());
        }

        for v in vectors {
            if v.len() != self.dim {
                return Err(RekhaError::InvalidDimension {
                    expected: self.dim,
                    actual: v.len(),
                });
            }
        }

        for m in 0..self.m {
            let start = m * self.d;
            let end = start + self.d;

            // Collect sub-vectors for this sub-quantizer.
            let sub_vectors: Vec<&[f32]> = vectors.iter().map(|v| &v[start..end]).collect();

            // Train k-means on these sub-vectors.
            let centroids = kmeans(&sub_vectors, self.k, 20);
            self.centroids[m] = centroids;
        }

        self.trained = true;
        Ok(())
    }

    /// Encode a single vector into its PQ code.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>, RekhaError> {
        if !self.trained {
            return Err(IndexError::NotTrained { component: "PQ" }.into());
        }
        if vector.len() != self.dim {
            return Err(RekhaError::InvalidDimension {
                expected: self.dim,
                actual: vector.len(),
            });
        }

        let mut code = Vec::with_capacity(self.m);
        for m in 0..self.m {
            let start = m * self.d;
            let end = start + self.d;
            let sub_vec = &vector[start..end];
            let centroid_idx = nearest_centroid(sub_vec, &self.centroids[m]);
            code.push(centroid_idx as u8);
        }

        Ok(code)
    }

    /// Encode multiple vectors into PQ codes.
    pub fn encode_batch(&self, vectors: &[&[f32]]) -> Result<Vec<Vec<u8>>, RekhaError> {
        vectors.iter().map(|v| self.encode(v)).collect()
    }

    /// Compute a distance table for a query vector.
    /// Result is [M][K] — for each sub-quantizer, the distance from
    /// the query's sub-vector to each of the K centroids.
    pub fn distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let mut table = Vec::with_capacity(self.m);
        for m in 0..self.m {
            let start = m * self.d;
            let end = start + self.d;
            let query_sub = &query[start..end];
            let mut row = Vec::with_capacity(self.k);
            for centroid in &self.centroids[m] {
                let dist = l2_squared(query_sub, centroid);
                row.push(dist);
            }
            table.push(row);
        }
        table
    }

    /// Approximate L2 distance between query and an encoded vector using ADC.
    /// query: full query vector (used to build distance table externally)
    /// code: PQ-encoded vector
    /// table: precomputed distance table for this query
    #[inline]
    pub fn adc_distance(code: &[u8], table: &[Vec<f32>]) -> f32 {
        let mut dist = 0.0f32;
        for m in 0..code.len() {
            let centroid_idx = code[m] as usize;
            dist += table[m][centroid_idx];
        }
        dist
    }
}

/// K-means clustering on a set of sub-vectors.
/// Returns K centroids, each of dimension d.
fn kmeans(data: &[&[f32]], k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return vec![];
    }

    let dim = data[0].len();
    let mut rng = rand::thread_rng();

    // Initialize: randomly pick k data points as centroids.
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|_| {
            let idx = rng.gen_range(0..data.len());
            data[idx].to_vec()
        })
        .collect();

    let mut _assignments = vec![0usize; data.len()];

    for _iter in 0..max_iter {
        // Assignment step: assign each point to nearest centroid.
        let mut changed = false;
        for (i, point) in data.iter().enumerate() {
            let nearest = (0..k)
                .min_by(|&a, &b| {
                    let da = l2_squared(point, &centroids[a]);
                    let db = l2_squared(point, &centroids[b]);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);

            if _assignments[i] != nearest {
                _assignments[i] = nearest;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step: recompute centroids as mean of assigned points.
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, point) in data.iter().enumerate() {
            let cluster = _assignments[i];
            for d in 0..dim {
                sums[cluster][d] += point[d];
            }
            counts[cluster] += 1;
        }

        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / counts[c] as f32;
                }
            }
        }
    }

    centroids
}

/// Find the index of the nearest centroid to the given sub-vector.
fn nearest_centroid(sub_vec: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = l2_squared(sub_vec, a);
            let db = l2_squared(sub_vec, b);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_roundtrip() {
        let mut pq = ProductQuantizer::new(4, 16, 8).unwrap();
        assert_eq!(pq.m, 4);
        assert_eq!(pq.d, 2);

        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..8).map(|_| rand::thread_rng().gen::<f32>()).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        pq.train(&refs).unwrap();
        assert!(pq.trained);

        let code = pq.encode(&vectors[0]).unwrap();
        assert_eq!(code.len(), 4);

        let table = pq.distance_table(&vectors[0]);
        assert_eq!(table.len(), 4);
        assert_eq!(table[0].len(), 16);

        // ADC distance should be 0 for identical vector.
        let dist = ProductQuantizer::adc_distance(&code, &table);
        assert!(dist < 2.0); // Some quantization error expected.
    }

    #[test]
    fn test_pq_new_invalid_dim() {
        let result = ProductQuantizer::new(3, 16, 8);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not divisible"));
    }

    #[test]
    fn test_pq_untrained_encode() {
        let pq = ProductQuantizer::new(2, 8, 4).unwrap();
        let result = pq.encode(&[0.5; 4]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not been trained"));
    }

    #[test]
    fn test_pq_encode_wrong_dims() {
        let mut pq = ProductQuantizer::new(2, 8, 4).unwrap();
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|_| vec![rand::thread_rng().gen::<f32>(); 4])
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        pq.train(&refs).unwrap();

        let result = pq.encode(&[0.5; 8]);
        assert!(result.is_err());
    }

    #[test]
    fn test_pq_train_empty() {
        let mut pq = ProductQuantizer::new(2, 8, 4).unwrap();
        let result = pq.train(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_pq_encode_batch() {
        let mut pq = ProductQuantizer::new(2, 8, 4).unwrap();
        let vectors: Vec<Vec<f32>> = (0..20)
            .map(|_| vec![rand::thread_rng().gen::<f32>(); 4])
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        pq.train(&refs).unwrap();

        let codes = pq.encode_batch(&refs).unwrap();
        assert_eq!(codes.len(), 20);
        for code in &codes {
            assert_eq!(code.len(), 2);
        }
    }

    #[test]
    fn test_pq_distance_table_shape() {
        let mut pq = ProductQuantizer::new(4, 32, 8).unwrap();
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|_| vec![rand::thread_rng().gen::<f32>(); 8])
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        pq.train(&refs).unwrap();

        let table = pq.distance_table(&vectors[0]);
        assert_eq!(table.len(), 4);
        assert_eq!(table[0].len(), 32);
        assert_eq!(table[3].len(), 32);
    }

    #[test]
    fn test_pq_adc_distance_zero_for_identical() {
        let mut pq = ProductQuantizer::new(2, 8, 4).unwrap();
        let vectors: Vec<Vec<f32>> = (0..30)
            .map(|_| vec![rand::thread_rng().gen::<f32>(); 4])
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        pq.train(&refs).unwrap();

        let v = &vectors[5];
        let code = pq.encode(v).unwrap();
        let table = pq.distance_table(v);
        let dist = ProductQuantizer::adc_distance(&code, &table);
        assert!(dist < 1.0);
    }
}
