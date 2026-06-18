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
    /// Assign a vector ID to a partition key.
    fn assign(&self, id: u64, num_dimensions: usize) -> crate::PartitionKey;

    /// Get the dimension range for a given dimension group.
    fn dim_group_range(&self, group: u32) -> Option<(usize, usize)>;

    /// Number of dimension groups configured.
    fn num_dim_groups(&self) -> u32;

    /// Number of vector shards configured.
    fn num_vector_shards(&self) -> u64;
}

/// A handle for pushing recent inserts to an index buffer.
/// Used by RaftNode to notify the index about committed inserts.
/// Avoids circular dependency between rekha-raft and rekha-index.
pub trait IndexBufferHandle: Send + Sync {
    /// Push a committed insert to the index buffer for immediate searchability.
    fn buffer_insert(&self, id: u64, vector: Vec<f32>);
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
