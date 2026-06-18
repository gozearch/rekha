use rekha_core::{PartitionError, PartitionKey, PartitionStrategy, RekhaError};

/// Vector-ID based sharding strategy.
///
/// Vectors are distributed across shards by `id % num_vector_shards`.
/// Each shard is a self-contained index with full vector dimensions.
/// Shards are replicated across nodes via Raft groups.
pub struct ShardStrategy {
    num_vector_shards: u64,
}

impl ShardStrategy {
    pub fn new(num_vector_shards: u64) -> Result<Self, RekhaError> {
        if num_vector_shards == 0 {
            return Err(PartitionError::InvalidTopology {
                detail: "num_vector_shards must be > 0".into(),
            }
            .into());
        }
        Ok(Self { num_vector_shards })
    }

    pub fn num_vector_shards(&self) -> u64 {
        self.num_vector_shards
    }
}

impl PartitionStrategy for ShardStrategy {
    fn assign(&self, id: u64) -> PartitionKey {
        PartitionKey::VectorId(id % self.num_vector_shards)
    }

    fn num_vector_shards(&self) -> u64 {
        self.num_vector_shards
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_strategy_assign() {
        let strategy = ShardStrategy::new(4).unwrap();
        let key = strategy.assign(42);
        assert_eq!(key.vector_shard(4), 2);
    }

    #[test]
    fn test_strategy_zero_shards() {
        let result = ShardStrategy::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_strategy_assign_different_ids() {
        let strategy = ShardStrategy::new(4).unwrap();
        assert_eq!(strategy.assign(0).vector_shard(4), 0);
        assert_eq!(strategy.assign(4).vector_shard(4), 0);
        assert_eq!(strategy.assign(1).vector_shard(4), 1);
        assert_eq!(strategy.assign(7).vector_shard(2), 1);
    }

    #[test]
    fn test_strategy_num_shards() {
        let strategy = ShardStrategy::new(6).unwrap();
        assert_eq!(strategy.num_vector_shards(), 6);
    }

    #[test]
    fn test_single_shard() {
        let strategy = ShardStrategy::new(1).unwrap();
        assert_eq!(strategy.assign(42).vector_shard(1), 0);
        assert_eq!(strategy.assign(0).vector_shard(1), 0);
    }
}
