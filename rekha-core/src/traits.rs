use crate::error::RekhaError;
use crate::types::SearchParams;

/// The core vector index trait.
/// Implementations: VamanaGraph, (future) HNSW, Flat.
pub trait VectorIndex: Send + Sync {
    /// Insert a single vector into the index.
    fn insert(&self, id: u64, vector: &[f32]) -> Result<(), RekhaError>;

    /// Bulk insert vectors.
    fn insert_batch(&self, vectors: &[(u64, &[f32])]) -> Result<(), RekhaError>;

    /// Remove vectors by ID.
    fn delete(&self, ids: &[u64]) -> Result<(), RekhaError>;

    /// Search for the top-k approximate nearest neighbors.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>), RekhaError>;

    /// Dimension-aware search with early-stop pruning.
    /// Only considers dimensions in [dim_start, dim_end).
    fn search_dim_range(
        &self,
        query: &[f32],
        k: usize,
        dim_start: usize,
        dim_end: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>), RekhaError>;

    /// Number of vectors currently indexed.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimated memory usage in bytes.
    fn memory_usage(&self) -> usize;
}

/// Partition management trait.
/// Defines how vectors are distributed across nodes.
pub trait PartitionStrategy: Send + Sync {
    /// Assign a vector ID to a partition.
    fn assign(&self, id: u64) -> crate::PartitionKey;

    /// Number of vector shards configured.
    fn num_vector_shards(&self) -> u64;
}

/// A handle for pushing recent inserts to an index buffer.
/// Used by RaftNode to notify the index about committed inserts.
/// Avoids circular dependency between rekha-raft and rekha-index.
pub trait IndexBufferHandle: Send + Sync {
    /// Push a committed insert to the index buffer for immediate searchability.
    /// Persists both vector and payload atomically.
    fn buffer_insert(&self, id: u64, vector: Vec<f32>, payload: Option<Vec<u8>>);
    /// Mark committed deletes in the index buffer.
    fn buffer_delete(&self, ids: &[u64]);
}

/// Storage backend trait for persisting vectors and metadata.
pub trait VectorStoreBackend: Send + Sync {
    /// Store a vector by ID.
    fn put_vector(&self, id: u64, data: &[f32]) -> Result<(), RekhaError>;
    /// Retrieve a vector by ID.
    fn get_vector(&self, id: u64) -> Result<Option<Vec<f32>>, RekhaError>;
    /// Store payload metadata.
    fn put_payload(&self, id: u64, payload: &[u8]) -> Result<(), RekhaError>;
    /// Retrieve payload metadata.
    fn get_payload(&self, id: u64) -> Result<Option<Vec<u8>>, RekhaError>;
    /// Delete vectors and payloads.
    fn delete(&self, ids: &[u64]) -> Result<u64, RekhaError>;
    /// Iterate over all vector IDs.
    fn iter_ids(&self) -> Result<Vec<u64>, RekhaError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestIndex;

    impl VectorIndex for TestIndex {
        fn insert(&self, _id: u64, _vector: &[f32]) -> Result<(), RekhaError> {
            Ok(())
        }
        fn insert_batch(&self, _vectors: &[(u64, &[f32])]) -> Result<(), RekhaError> {
            Ok(())
        }
        fn delete(&self, _ids: &[u64]) -> Result<(), RekhaError> {
            Ok(())
        }
        fn search(
            &self,
            _query: &[f32],
            _k: usize,
            _params: &SearchParams,
        ) -> Result<(Vec<u64>, Vec<f32>), RekhaError> {
            Ok((vec![], vec![]))
        }
        fn search_dim_range(
            &self,
            _query: &[f32],
            _k: usize,
            _dim_start: usize,
            _dim_end: usize,
            _params: &SearchParams,
        ) -> Result<(Vec<u64>, Vec<f32>), RekhaError> {
            Ok((vec![], vec![]))
        }
        fn len(&self) -> usize {
            0
        }
        fn memory_usage(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_vector_index_is_empty_default() {
        let idx = TestIndex;
        assert!(idx.is_empty());
    }

    #[test]
    fn test_vector_index_is_empty_returns_false_when_non_empty() {
        struct NonEmpty;
        impl VectorIndex for NonEmpty {
            fn insert(&self, _id: u64, _vector: &[f32]) -> Result<(), RekhaError> {
                Ok(())
            }
            fn insert_batch(&self, _vectors: &[(u64, &[f32])]) -> Result<(), RekhaError> {
                Ok(())
            }
            fn delete(&self, _ids: &[u64]) -> Result<(), RekhaError> {
                Ok(())
            }
            fn search(
                &self,
                _query: &[f32],
                _k: usize,
                _params: &SearchParams,
            ) -> Result<(Vec<u64>, Vec<f32>), RekhaError> {
                Ok((vec![], vec![]))
            }
            fn search_dim_range(
                &self,
                _query: &[f32],
                _k: usize,
                _dim_start: usize,
                _dim_end: usize,
                _params: &SearchParams,
            ) -> Result<(Vec<u64>, Vec<f32>), RekhaError> {
                Ok((vec![], vec![]))
            }
            fn len(&self) -> usize {
                5
            }
            fn memory_usage(&self) -> usize {
                0
            }
        }
        assert!(!NonEmpty.is_empty());
    }
}
