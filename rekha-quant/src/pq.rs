use std::fmt;

use rekha_core::RekhaError;

use crate::kmeans::KMeans;

#[derive(Clone)]
pub struct ProductQuantizer {
    pub m: usize,
    pub k: usize,
    pub sub_dim: usize,
    pub codebooks: Vec<Vec<Vec<f32>>>, // [m][k][sub_dim]
    pub trained: bool,
}

impl ProductQuantizer {
    pub fn new(m: usize, k: usize) -> Self {
        ProductQuantizer {
            m,
            k,
            sub_dim: 0,
            codebooks: Vec::new(),
            trained: false,
        }
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }

    pub fn train(&mut self, data: &[Vec<f32>]) -> Result<(), RekhaError> {
        if data.is_empty() {
            return Err(RekhaError::InvalidArgument(
                "empty training data for PQ".into(),
            ));
        }
        let dim = data[0].len();
        if !dim.is_multiple_of(self.m) {
            return Err(RekhaError::InvalidArgument(format!(
                "dimension {} must be divisible by m={}",
                dim, self.m
            )));
        }
        self.sub_dim = dim / self.m;

        let mut codebooks = Vec::with_capacity(self.m);
        for s in 0..self.m {
            let start = s * self.sub_dim;
            let end = start + self.sub_dim;
            let sub_vectors: Vec<Vec<f32>> = data.iter().map(|v| v[start..end].to_vec()).collect();

            let mut kmeans = KMeans::new(self.k, self.sub_dim, 20, 1e-4);
            kmeans.fit(&sub_vectors)?;
            codebooks.push(kmeans.centroids);
        }

        self.codebooks = codebooks;
        self.trained = true;
        Ok(())
    }

    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.m);
        for s in 0..self.m {
            let start = s * self.sub_dim;
            let end = start + self.sub_dim;
            let sub_vec = &vector[start..end];
            let mut best = 0u8;
            let mut best_dist = f32::MAX;
            for (j, centroid) in self.codebooks[s].iter().enumerate() {
                let d: f32 = sub_vec
                    .iter()
                    .zip(centroid.iter())
                    .map(|(x, y)| {
                        let diff = x - y;
                        diff * diff
                    })
                    .sum();
                if d < best_dist {
                    best_dist = d;
                    best = j as u8;
                }
            }
            codes.push(best);
        }
        codes
    }

    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Vec<Vec<u8>> {
        vectors.iter().map(|v| self.encode(v)).collect()
    }

    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.m * self.sub_dim);
        for (s, &code) in codes.iter().enumerate() {
            let idx = code as usize;
            if s < self.codebooks.len() && idx < self.codebooks[s].len() {
                result.extend_from_slice(&self.codebooks[s][idx]);
            }
        }
        result
    }

    pub fn distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let mut table = Vec::with_capacity(self.m);
        for s in 0..self.m {
            let start = s * self.sub_dim;
            let end = start + self.sub_dim;
            let sub_query = &query[start..end];
            let mut dists = Vec::with_capacity(self.k);
            for centroid in &self.codebooks[s] {
                let d: f32 = sub_query
                    .iter()
                    .zip(centroid.iter())
                    .map(|(x, y)| {
                        let diff = x - y;
                        diff * diff
                    })
                    .sum();
                dists.push(d);
            }
            table.push(dists);
        }
        table
    }

    pub fn adc_distance(&self, table: &[Vec<f32>], codes: &[u8]) -> f32 {
        let mut dist = 0.0f32;
        for (s, &code) in codes.iter().enumerate() {
            let idx = code as usize;
            if s < table.len() && idx < table[s].len() {
                dist += table[s][idx];
            }
        }
        dist
    }
}

impl fmt::Debug for ProductQuantizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProductQuantizer")
            .field("m", &self.m)
            .field("k", &self.k)
            .field("sub_dim", &self.sub_dim)
            .field("trained", &self.trained)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_train_encode_decode() {
        let dim = 8;
        let m = 4;
        let k = 16;
        let data: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..dim).map(|d| ((i * 10 + d) as f32) / 10.0).collect())
            .collect();

        let mut pq = ProductQuantizer::new(m, k);
        pq.train(&data).unwrap();
        assert!(pq.is_trained());
        assert_eq!(pq.sub_dim, dim / m);
        assert_eq!(pq.codebooks.len(), m);

        let codes = pq.encode(&data[0]);
        assert_eq!(codes.len(), m);

        let decoded = pq.decode(&codes);
        assert_eq!(decoded.len(), dim);
    }

    #[test]
    fn test_pq_distance_table() {
        let dim = 4;
        let m = 2;
        let k = 4;
        let data: Vec<Vec<f32>> = (0..20)
            .map(|i| (0..dim).map(|d| (i + d) as f32).collect())
            .collect();

        let mut pq = ProductQuantizer::new(m, k);
        pq.train(&data).unwrap();

        let query = vec![0.5, 0.5, 0.5, 0.5];
        let table = pq.distance_table(&query);
        assert_eq!(table.len(), m);
        assert_eq!(table[0].len(), k);

        let codes = pq.encode(&query);
        let d1 = pq.adc_distance(&table, &codes);
        assert!(d1.is_finite());
    }

    #[test]
    fn test_pq_invalid_dimension() {
        let mut pq = ProductQuantizer::new(3, 16);
        let data = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let result = pq.train(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_pq_empty_data() {
        let mut pq = ProductQuantizer::new(2, 16);
        let result = pq.train(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_batch() {
        let dim = 4;
        let m = 2;
        let k = 4;
        let data: Vec<Vec<f32>> = (0..20)
            .map(|i| (0..dim).map(|d| (i + d) as f32).collect())
            .collect();

        let mut pq = ProductQuantizer::new(m, k);
        pq.train(&data).unwrap();

        let batch_codes = pq.encode_batch(&data[..5]);
        assert_eq!(batch_codes.len(), 5);
        for codes in &batch_codes {
            assert_eq!(codes.len(), m);
        }
    }
}
