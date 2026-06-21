use crate::error::RekhaError;
use crate::types::SearchParams;

pub trait VectorIndex: Send + Sync {
    fn insert(&self, id: u64, vector: &[f32]) -> Result<(), RekhaError>;

    fn insert_batch(&self, vectors: &[(u64, &[f32])]) -> Result<(), RekhaError>;

    fn delete(&self, ids: &[u64]) -> Result<(), RekhaError>;

    fn search(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>), RekhaError>;

    fn search_dim_range(
        &self,
        query: &[f32],
        k: usize,
        dim_start: usize,
        dim_end: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>), RekhaError>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn memory_usage(&self) -> usize;

    fn centroids(&self) -> Vec<Vec<f32>>;

    fn num_clusters(&self) -> usize;
}

pub trait PartitionStrategy: Send + Sync {
    fn assign(&self, id: u64, num_dimensions: usize) -> crate::PartitionKey;

    fn dim_group_range(&self, group: u32) -> Option<(usize, usize)>;

    fn num_dim_groups(&self) -> u32;

    fn num_vector_shards(&self) -> u64;
}

pub trait IndexBufferHandle: Send + Sync {
    fn buffer_insert(&self, id: u64, vector: Vec<f32>);
    fn buffer_delete(&self, ids: &[u64]);
}

pub trait VectorStoreBackend: Send + Sync {
    fn put_vector(&self, id: u64, data: &[f32]) -> Result<(), RekhaError>;
    fn get_vector(&self, id: u64) -> Result<Option<Vec<f32>>, RekhaError>;
    fn put_payload(&self, id: u64, payload: &[u8]) -> Result<(), RekhaError>;
    fn get_payload(&self, id: u64) -> Result<Option<Vec<u8>>, RekhaError>;
    fn delete(&self, ids: &[u64]) -> Result<u64, RekhaError>;
    fn iter_ids(&self) -> Result<Vec<u64>, RekhaError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestIndex;

    impl VectorIndex for TestIndex {
        fn insert(&self, _id: u64, _vector: &[f32]) -> Result<(), RekhaError> { Ok(()) }
        fn insert_batch(&self, _vectors: &[(u64, &[f32])]) -> Result<(), RekhaError> { Ok(()) }
        fn delete(&self, _ids: &[u64]) -> Result<(), RekhaError> { Ok(()) }
        fn search(&self, _query: &[f32], _k: usize, _params: &SearchParams) -> Result<(Vec<u64>, Vec<f32>), RekhaError> { Ok((vec![], vec![])) }
        fn search_dim_range(&self, _query: &[f32], _k: usize, _dim_start: usize, _dim_end: usize, _params: &SearchParams) -> Result<(Vec<u64>, Vec<f32>), RekhaError> { Ok((vec![], vec![])) }
        fn len(&self) -> usize { 0 }
        fn memory_usage(&self) -> usize { 0 }
        fn centroids(&self) -> Vec<Vec<f32>> { vec![] }
        fn num_clusters(&self) -> usize { 0 }
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
            fn insert(&self, _id: u64, _vector: &[f32]) -> Result<(), RekhaError> { Ok(()) }
            fn insert_batch(&self, _vectors: &[(u64, &[f32])]) -> Result<(), RekhaError> { Ok(()) }
            fn delete(&self, _ids: &[u64]) -> Result<(), RekhaError> { Ok(()) }
            fn search(&self, _query: &[f32], _k: usize, _params: &SearchParams) -> Result<(Vec<u64>, Vec<f32>), RekhaError> { Ok((vec![], vec![])) }
            fn search_dim_range(&self, _query: &[f32], _k: usize, _dim_start: usize, _dim_end: usize, _params: &SearchParams) -> Result<(Vec<u64>, Vec<f32>), RekhaError> { Ok((vec![], vec![])) }
            fn len(&self) -> usize { 5 }
            fn memory_usage(&self) -> usize { 0 }
            fn centroids(&self) -> Vec<Vec<f32>> { vec![] }
            fn num_clusters(&self) -> usize { 0 }
        }
        assert!(!NonEmpty.is_empty());
    }
}
