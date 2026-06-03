use rekha_core::{PartitionError, PartitionKey, PartitionStrategy, RekhaError};

/// The multi-granularity partition strategy, combining:
///
/// 1. **Vector-based partitioning** (horizontal sharding):
///    Vectors are split by ID range into `num_vector_shards` shards.
///    Each node owns a subset of full vectors.
///
/// 2. **Dimension-based partitioning** (vertical partitioning):
///    Within each vector shard, vectors are split across dimension groups.
///    Each dimension group handles a contiguous range of dimensions.
///
/// This 2D grid (vector_shard × dim_group) enables:
/// - Balanced distribution of compute: each node handles a manageable subset of vectors AND dims
/// - Early-stop pruning: partial distance computations can stop when exceeding the threshold
/// - Efficient parallel search: queries fan out to relevant (shard, dim_group) pairs
pub struct MultiGranularityStrategy {
    /// Number of vector shards (horizontal partitions).
    num_vector_shards: u64,
    /// Number of dimension groups (vertical partitions).
    /// Must divide total vector dimension evenly.
    num_dim_groups: u32,
    /// Total vector dimension.
    /// Dimensions per group (total_dim / num_dim_groups).
    dims_per_group: usize,
}

impl MultiGranularityStrategy {
    /// Create a new multi-granularity partition strategy.
    ///
    /// # Arguments
    /// * `num_vector_shards` - Number of vector-based partitions (shards)
    /// * `num_dim_groups` - Number of dimension-based groups
    /// * `total_dim` - Total vector dimensionality
    pub fn new(
        num_vector_shards: u64,
        num_dim_groups: u32,
        total_dim: usize,
    ) -> Result<Self, RekhaError> {
        if num_vector_shards == 0 {
            return Err(PartitionError::InvalidTopology {
                detail: "num_vector_shards must be > 0".into(),
            }.into());
        }
        if num_dim_groups == 0 {
            return Err(PartitionError::InvalidTopology {
                detail: "num_dim_groups must be > 0".into(),
            }.into());
        }
        if total_dim % num_dim_groups as usize != 0 {
            return Err(PartitionError::InvalidTopology {
                detail: format!(
                    "total_dim {total_dim} not divisible by num_dim_groups {num_dim_groups}"
                ),
            }.into());
        }

        Ok(Self {
            num_vector_shards,
            num_dim_groups,
            dims_per_group: total_dim / num_dim_groups as usize,
        })
    }

    /// Return the dimension range for a given dimension group.
    pub fn dim_group_range(&self, group: u32) -> Option<(usize, usize)> {
        if group >= self.num_dim_groups {
            return None;
        }
        let start = (group as usize) * self.dims_per_group;
        let end = start + self.dims_per_group;
        Some((start, end))
    }

    /// Enumerate all (vector_shard, dim_group) pairs in the grid.
    pub fn all_partitions(&self) -> Vec<(u64, u32)> {
        let mut partitions = Vec::with_capacity((self.num_vector_shards * self.num_dim_groups as u64) as usize);
        for s in 0..self.num_vector_shards {
            for g in 0..self.num_dim_groups {
                partitions.push((s, g));
            }
        }
        partitions
    }
}

impl PartitionStrategy for MultiGranularityStrategy {
    fn assign(&self, id: u64, _num_dimensions: usize) -> PartitionKey {
        PartitionKey::Hybrid {
            vector_shard: id % self.num_vector_shards,
            dim_group: 0, // All dim groups are queried for a full search
        }
    }

    fn dim_group_range(&self, group: u32) -> Option<(usize, usize)> {
        self.dim_group_range(group)
    }

    fn num_dim_groups(&self) -> u32 {
        self.num_dim_groups
    }

    fn num_vector_shards(&self) -> u64 {
        self.num_vector_shards
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_strategy() {
        let strategy = MultiGranularityStrategy::new(4, 4, 768).unwrap();
        assert_eq!(strategy.dims_per_group, 192);

        let range = strategy.dim_group_range(0).unwrap();
        assert_eq!(range, (0, 192));

        let range = strategy.dim_group_range(3).unwrap();
        assert_eq!(range, (576, 768));

        let key = strategy.assign(42, 768);
        assert_eq!(key.vector_shard(4), 2); // 42 % 4 = 2
    }

    #[test]
    fn test_strategy_zero_shards() {
        let result = MultiGranularityStrategy::new(0, 4, 768);
        assert!(result.is_err());
    }

    #[test]
    fn test_strategy_zero_dim_groups() {
        let result = MultiGranularityStrategy::new(4, 0, 768);
        assert!(result.is_err());
    }

    #[test]
    fn test_strategy_dim_not_divisible() {
        let result = MultiGranularityStrategy::new(4, 7, 768);
        assert!(result.is_err());
    }

    #[test]
    fn test_strategy_out_of_range_group() {
        let strategy = MultiGranularityStrategy::new(4, 4, 768).unwrap();
        assert!(strategy.dim_group_range(4).is_none());
    }

    #[test]
    fn test_strategy_assign_different_ids() {
        let strategy = MultiGranularityStrategy::new(4, 4, 768).unwrap();
        let key_a = strategy.assign(0, 768);
        let key_b = strategy.assign(4, 768);
        assert_eq!(key_a.vector_shard(4), 0);
        assert_eq!(key_b.vector_shard(4), 0);

        let key_c = strategy.assign(1, 768);
        assert_eq!(key_c.vector_shard(4), 1);
    }

    #[test]
    fn test_strategy_assign_non_standard_dims() {
        let strategy = MultiGranularityStrategy::new(2, 2, 512).unwrap();
        let key = strategy.assign(7, 512);
        assert_eq!(key.vector_shard(2), 1);
    }

    #[test]
    fn test_strategy_single_shard_single_group() {
        let strategy = MultiGranularityStrategy::new(1, 1, 128).unwrap();
        assert_eq!(strategy.dims_per_group, 128);
        assert_eq!(strategy.num_dim_groups(), 1);
        assert_eq!(strategy.num_vector_shards(), 1);

        let range = strategy.dim_group_range(0).unwrap();
        assert_eq!(range, (0, 128));
    }
}
