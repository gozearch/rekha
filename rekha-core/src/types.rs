use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    pub id: u64,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedVector {
    pub id: u64,
    pub pq_code: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredPoint {
    pub id: u64,
    pub score: f32,
    pub payload: Option<Payload>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Payload {
    pub content_type: PayloadType,
    pub data: Vec<u8>,
}

impl Payload {
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            content_type: PayloadType::Text,
            data: text.into().into_bytes(),
        }
    }

    pub fn from_json<T: Serialize>(value: &T) -> Result<Self, serde_json::Error> {
        Ok(Self {
            content_type: PayloadType::Json,
            data: serde_json::to_vec(value)?,
        })
    }

    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self {
            content_type: PayloadType::Raw,
            data,
        }
    }

    pub fn as_text(&self) -> Option<String> {
        if matches!(self.content_type, PayloadType::Text) {
            String::from_utf8(self.data.clone()).ok()
        } else {
            None
        }
    }

    pub fn as_json<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        if matches!(self.content_type, PayloadType::Json) {
            serde_json::from_slice(&self.data).ok()
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PayloadType {
    Text,
    Json,
    Raw,
}

impl std::fmt::Display for PayloadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Json => write!(f, "json"),
            Self::Raw => write!(f, "raw"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DistanceMetric {
    L2,
    Cosine,
    InnerProduct,
}

impl DistanceMetric {
    pub fn name(&self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::Cosine => "cosine",
            Self::InnerProduct => "inner_product",
        }
    }

    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        crate::distance::distance(a, b, *self)
    }
}

impl std::str::FromStr for DistanceMetric {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "l2" | "euclidean" => Ok(Self::L2),
            "cosine" | "cos" => Ok(Self::Cosine),
            "ip" | "inner_product" => Ok(Self::InnerProduct),
            _ => Err(format!("unknown distance metric: {s}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PartitionKey {
    VectorId(u64),
    DimensionRange(u32, u32),
    Hybrid { vector_shard: u64, dim_group: u32 },
}

impl PartitionKey {
    pub fn vector_shard(&self, num_shards: u64) -> u64 {
        match self {
            Self::VectorId(id) => id % num_shards,
            Self::DimensionRange(_, _) => 0,
            Self::Hybrid { vector_shard, .. } => *vector_shard % num_shards,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub enum PlanType {
    #[default]
    VectorBased,
    DimensionBased,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    pub ef_search: usize,
    pub nprobe: usize,
    pub plan: PlanType,
    pub include_payloads: bool,
    pub local_only: bool,
    pub partition_hint: Option<u64>,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            ef_search: 128,
            nprobe: 16,
            plan: PlanType::default(),
            include_payloads: true,
            local_only: false,
            partition_hint: None,
        }
    }
}

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
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchStats {
    pub total_ms: f64,
    pub nodes_contacted: u32,
    pub vectors_scanned: u64,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub address: String,
    pub partition_id: u64,
    pub dim_groups: Vec<u32>,
    pub is_leader: bool,
    pub raft_term: u64,
    pub commit_index: u64,
    pub storage_bytes: u64,
    pub status: NodeStatus,
    #[serde(default)]
    pub last_heartbeat: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    Healthy,
    Degraded,
    Unreachable,
    Recovering,
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterTopology {
    pub cluster_id: String,
    pub nodes: HashMap<String, NodeInfo>,
    pub partition_map: HashMap<(u64, u32), Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwnedRange {
    pub vector_shard: u64,
    pub dim_start: usize,
    pub dim_end: usize,
}

impl OwnedRange {
    pub fn dim_count(&self) -> usize {
        self.dim_end.saturating_sub(self.dim_start)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct RaftIndex {
    pub term: u64,
    pub index: u64,
}

impl RaftIndex {
    pub fn zero() -> Self {
        Self { term: 0, index: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_from_text() {
        let p = Payload::from_text("hello");
        assert_eq!(p.content_type, PayloadType::Text);
        assert_eq!(p.as_text(), Some("hello".into()));
    }

    #[test]
    fn test_payload_from_json() {
        let p = Payload::from_json(&serde_json::json!({"key": "value"})).unwrap();
        assert_eq!(p.content_type, PayloadType::Json);
        let val: serde_json::Value = p.as_json().unwrap();
        assert_eq!(val["key"], "value");
    }

    #[test]
    fn test_payload_from_bytes() {
        let data = vec![1, 2, 3];
        let p = Payload::from_bytes(data.clone());
        assert_eq!(p.content_type, PayloadType::Raw);
        assert!(p.as_text().is_none());
        assert!(p.as_json::<serde_json::Value>().is_none());
    }

    #[test]
    fn test_distance_metric_name() {
        assert_eq!(DistanceMetric::L2.name(), "l2");
        assert_eq!(DistanceMetric::Cosine.name(), "cosine");
        assert_eq!(DistanceMetric::InnerProduct.name(), "inner_product");
    }

    #[test]
    fn test_distance_metric_from_str() {
        assert_eq!("l2".parse::<DistanceMetric>().unwrap(), DistanceMetric::L2);
        assert_eq!("euclidean".parse::<DistanceMetric>().unwrap(), DistanceMetric::L2);
        assert_eq!("cosine".parse::<DistanceMetric>().unwrap(), DistanceMetric::Cosine);
        assert_eq!("cos".parse::<DistanceMetric>().unwrap(), DistanceMetric::Cosine);
        assert_eq!("ip".parse::<DistanceMetric>().unwrap(), DistanceMetric::InnerProduct);
        assert!("unknown".parse::<DistanceMetric>().is_err());
    }

    #[test]
    fn test_partition_key_vector_shard() {
        let pk = PartitionKey::VectorId(42);
        assert_eq!(pk.vector_shard(4), 2);
        let pk = PartitionKey::DimensionRange(0, 128);
        assert_eq!(pk.vector_shard(4), 0);
        let pk = PartitionKey::Hybrid { vector_shard: 10, dim_group: 1 };
        assert_eq!(pk.vector_shard(4), 2);
    }

    #[test]
    fn test_search_params_default() {
        let p = SearchParams::default();
        assert_eq!(p.ef_search, 128);
        assert_eq!(p.nprobe, 16);
        assert_eq!(p.plan, PlanType::VectorBased);
        assert!(p.include_payloads);
        assert!(p.partition_hint.is_none());
    }

    #[test]
    fn test_owned_range_dim_count() {
        let r = OwnedRange { vector_shard: 0, dim_start: 0, dim_end: 128 };
        assert_eq!(r.dim_count(), 128);
        let r = OwnedRange { vector_shard: 1, dim_start: 128, dim_end: 64 };
        assert_eq!(r.dim_count(), 0);
    }

    #[test]
    fn test_raft_index_zero() {
        let ri = RaftIndex::zero();
        assert_eq!(ri.term, 0);
        assert_eq!(ri.index, 0);
    }

    #[test]
    fn test_raft_index_ordering() {
        let a = RaftIndex { term: 1, index: 5 };
        let b = RaftIndex { term: 2, index: 3 };
        assert!(a < b);
        let c = RaftIndex { term: 1, index: 10 };
        assert!(a < c);
    }

    #[test]
    fn test_plan_type_default() {
        assert_eq!(PlanType::default(), PlanType::VectorBased);
    }

    #[test]
    fn test_distance_metric_dispatch() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let d = DistanceMetric::L2.distance(&a, &b);
        assert!((d - 8.0).abs() < 1e-6);
        let d = DistanceMetric::Cosine.distance(&a, &b);
        assert!(d >= 0.0);
    }

    #[test]
    fn test_search_stats_default() {
        let s = SearchStats::default();
        assert_eq!(s.total_ms, 0.0);
        assert_eq!(s.nodes_contacted, 0);
        assert_eq!(s.vectors_scanned, 0);
        assert!(s.warnings.is_empty());
    }

    #[test]
    fn test_payload_type_display() {
        assert_eq!(PayloadType::Text.to_string(), "text");
        assert_eq!(PayloadType::Json.to_string(), "json");
        assert_eq!(PayloadType::Raw.to_string(), "raw");
    }
}
