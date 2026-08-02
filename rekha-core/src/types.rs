use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: u64,
    pub data: Vec<f32>,
    pub timestamp: i64,
    pub is_tombstone: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IvfConfig {
    pub dim: u32,
    pub nlist: u32,
    pub nprobe: u32,
    #[serde(default = "default_pq_m")]
    pub pq_m: u32,
    #[serde(default = "default_pq_k")]
    pub pq_k: u16,
    #[serde(default = "default_replication_factor")]
    pub replication_factor: u32,
    #[serde(default)]
    pub distance_metric: DistanceMetric,
}

fn default_pq_m() -> u32 {
    4
}
fn default_pq_k() -> u16 {
    256
}

fn default_replication_factor() -> u32 {
    3
}

impl Default for IvfConfig {
    fn default() -> Self {
        IvfConfig {
            dim: 8,
            nlist: 256,
            nprobe: 16,
            pq_m: 4,
            pq_k: 256,
            replication_factor: 3,
            distance_metric: DistanceMetric::L2,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    #[default]
    L2,
    Cosine,
    InnerProduct,
}

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistanceMetric::L2 => write!(f, "l2"),
            DistanceMetric::Cosine => write!(f, "cosine"),
            DistanceMetric::InnerProduct => write!(f, "inner_product"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub address: String,
    pub is_alive: bool,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    One = 1,
    #[default]
    Quorum = 2,
    All = 3,
}

impl From<i32> for ConsistencyLevel {
    fn from(v: i32) -> Self {
        match v {
            1 => ConsistencyLevel::One,
            2 => ConsistencyLevel::Quorum,
            3 => ConsistencyLevel::All,
            _ => ConsistencyLevel::Quorum,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScoredPoint {
    pub id: u64,
    pub score: f32,
    pub payload: Option<Vec<u8>>,
    pub timestamp: i64,
}

impl PartialEq for ScoredPoint {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ScoredPoint {}

impl Ord for ScoredPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for ScoredPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
pub struct SearchParams {
    pub nprobe: u32,
    pub k: u32,
    pub include_payloads: bool,
    pub pre_filter: Option<String>,
    pub local_only: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            nprobe: 16,
            k: 10,
            include_payloads: false,
            pre_filter: None,
            local_only: false,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub config: IvfConfig,
    pub vector_count: u64,
    pub index_ready: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedVector {
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: Option<Vec<u8>>,
    pub timestamp: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_config_default() {
        let cfg = IvfConfig::default();
        assert_eq!(cfg.dim, 8);
        assert_eq!(cfg.nlist, 256);
        assert_eq!(cfg.nprobe, 16);
        assert_eq!(cfg.pq_m, 4);
        assert_eq!(cfg.pq_k, 256);
        assert_eq!(cfg.replication_factor, 3);
        assert_eq!(cfg.distance_metric, DistanceMetric::L2);
    }

    #[test]
    fn test_consistency_level_default() {
        assert_eq!(ConsistencyLevel::default(), ConsistencyLevel::Quorum);
    }

    #[test]
    fn test_search_params_default() {
        let p = SearchParams::default();
        assert_eq!(p.nprobe, 16);
        assert_eq!(p.k, 10);
        assert!(!p.include_payloads);
        assert!(p.pre_filter.is_none());
        assert!(!p.local_only);
    }

    #[test]
    fn test_distance_metric_display() {
        assert_eq!(DistanceMetric::L2.to_string(), "l2");
        assert_eq!(DistanceMetric::Cosine.to_string(), "cosine");
        assert_eq!(DistanceMetric::InnerProduct.to_string(), "inner_product");
    }

    #[test]
    fn test_vector_record_serde() {
        let v = VectorRecord {
            id: 42,
            data: vec![0.1, 0.2, 0.3],
            timestamp: 1000,
            is_tombstone: false,
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: VectorRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, 42);
        assert_eq!(back.data.len(), 3);
        assert!(!back.is_tombstone);
    }

    #[test]
    fn test_collection_info_serde() {
        let ci = CollectionInfo {
            name: "test".into(),
            config: IvfConfig {
                dim: 128,
                nlist: 1024,
                nprobe: 64,
                pq_m: 16,
                pq_k: 256,
                replication_factor: 3,
                distance_metric: DistanceMetric::L2,
            },
            vector_count: 1000,
            index_ready: true,
        };
        let json = serde_json::to_string(&ci).unwrap();
        let back: CollectionInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "test");
        assert_eq!(back.config.dim, 128);
        assert!(back.index_ready);
    }
}
