use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ConsistencyLevel {
    One,
    Quorum,
    All,
}

impl ConsistencyLevel {
    pub fn to_i32(self) -> i32 {
        match self {
            Self::One => 1,
            Self::Quorum => 2,
            Self::All => 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: u64,
    pub timestamp: u64,
    pub data: Option<Vec<f32>>,
    pub is_tombstone: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    pub id: u64,
    pub data: Vec<f32>,
    #[serde(default)]
    pub timestamp: u64,
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
    #[serde(default)]
    pub timestamp: u64,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    pub ef_search: usize,
    pub nprobe: usize,
    pub include_payloads: bool,
    pub local_only: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            ef_search: 128,
            nprobe: 16,
            include_payloads: true,
            local_only: false,
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

pub fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

pub fn quorum(rf: usize) -> usize {
    rf / 2 + 1
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
    fn test_search_params_default() {
        let p = SearchParams::default();
        assert_eq!(p.ef_search, 128);
        assert_eq!(p.nprobe, 16);
        assert!(p.include_payloads);
    }

    #[test]
    fn test_owned_range_dim_count() {
        let r = OwnedRange { vector_shard: 0, dim_start: 0, dim_end: 128 };
        assert_eq!(r.dim_count(), 128);
        let r = OwnedRange { vector_shard: 1, dim_start: 128, dim_end: 64 };
        assert_eq!(r.dim_count(), 0);
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

    #[test]
    fn test_vector_timestamp_default() {
        let v = Vector { id: 1, data: vec![1.0, 2.0], timestamp: 0 };
        assert_eq!(v.timestamp, 0);
    }

    #[test]
    fn test_scored_point_timestamp_default() {
        let sp = ScoredPoint { id: 1, score: 0.5, payload: None, timestamp: 0 };
        assert_eq!(sp.timestamp, 0);
    }

    #[test]
    fn test_consistency_level_debug_clone_copy() {
        let c = ConsistencyLevel::Quorum;
        let _d = format!("{:?}", c);
        let _c = c;
        let _e = c;
    }

    #[test]
    fn test_vector_record_tombstone() {
        let r = VectorRecord { id: 42, timestamp: 100, data: None, is_tombstone: true };
        assert!(r.is_tombstone);
        assert!(r.data.is_none());
    }

    #[test]
    fn test_now_micros_nonzero() {
        let t = now_micros();
        assert!(t > 1_700_000_000_000_000); // must be past 2023
    }

    #[test]
    fn test_quorum() {
        assert_eq!(quorum(1), 1);
        assert_eq!(quorum(2), 2);
        assert_eq!(quorum(3), 2);
        assert_eq!(quorum(5), 3);
    }
}
