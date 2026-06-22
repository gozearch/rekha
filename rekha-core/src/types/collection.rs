use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub dim: u32,
    pub num_vector_shards: u64,
    pub replication_factor: u64,
    pub num_dim_groups: u32,
    pub dim_group_size: u32,
    pub nlist: u32,
    pub nprobe: u32,
    pub pq_num_sub_vectors: u32,
    pub pq_num_centroids: u32,
    pub re_rank_k: u32,
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            dim: 256,
            num_vector_shards: 6,
            replication_factor: 3,
            num_dim_groups: 1,
            dim_group_size: 256,
            nlist: 1024,
            nprobe: 16,
            pq_num_sub_vectors: 64,
            pq_num_centroids: 256,
            re_rank_k: 256,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub config: CollectionConfig,
    pub vector_count: u64,
    pub index_ready: bool,
    pub config_timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMeta {
    pub config: CollectionConfig,
    pub timestamp: u64,
    pub is_deleted: bool,
    #[serde(default)]
    pub vector_count: u64,
}
