use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub cluster: ClusterConfig,
    pub tls: TlsConfig,
    pub observability: ObservabilityConfig,
    pub storage: StorageConfig,
    pub partition: PartitionConfig,
}

impl ServerConfig {
    pub fn from_file(path: impl Into<PathBuf>) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.into();
        let contents = std::fs::read_to_string(&path)?;
        let config: Self = serde_yaml::from_str(&contents)?;
        Ok(config)
    }

    pub fn dev_default(node_id: &str, data_dir: &str) -> Self {
        Self {
            cluster: ClusterConfig {
                node_id: node_id.to_string(),
                seed_nodes: vec![format!("127.0.0.1:50051")],
                bind_addr: "0.0.0.0:50051".into(),
                data_dir: data_dir.into(),
                default_write_consistency: "QUORUM".into(),
                hinted_handoff_enabled: true,
                max_hint_window_secs: 10800,
            },
            tls: TlsConfig::default(),
            observability: ObservabilityConfig {
                metrics: "prometheus".into(),
                tracing: "none".into(),
                logging: "structured".into(),
            },
            storage: StorageConfig {
                max_payload_size: 1_048_576,
                max_inline_size: 1_048_576,
                gc_grace_seconds: 864000,
            },
            partition: PartitionConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TlsConfig {
    #[serde(default)]
    pub enabled: bool,
    pub cert_path: Option<String>,
    pub key_path: Option<String>,
    pub ca_cert_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub node_id: String,
    pub seed_nodes: Vec<String>,
    pub bind_addr: String,
    pub data_dir: String,

    #[serde(default = "default_consistency")]
    pub default_write_consistency: String,

    #[serde(default = "default_true")]
    pub hinted_handoff_enabled: bool,

    #[serde(default = "default_hint_window")]
    pub max_hint_window_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    pub metrics: String,
    pub tracing: String,
    pub logging: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub max_payload_size: usize,
    pub max_inline_size: usize,

    #[serde(default = "default_gc_grace")]
    pub gc_grace_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    /// Number of dimension groups for partitioning
    #[serde(default = "default_num_dim_groups")]
    pub num_dim_groups: u32,
    /// Total dimension for partition layout
    #[serde(default = "default_total_dim")]
    pub total_dim: usize,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            num_dim_groups: default_num_dim_groups(),
            total_dim: default_total_dim(),
        }
    }
}

fn default_num_dim_groups() -> u32 { 4 }
fn default_total_dim() -> usize { 768 }

fn default_consistency() -> String {
    "QUORUM".into()
}

fn default_true() -> bool {
    true
}

fn default_hint_window() -> u64 {
    10800
}

fn default_gc_grace() -> u64 {
    864000
}

pub fn parse_consistency(s: &str) -> Option<rekha_core::ConsistencyLevel> {
    match s.to_uppercase().as_str() {
        "ONE" => Some(rekha_core::ConsistencyLevel::One),
        "QUORUM" => Some(rekha_core::ConsistencyLevel::Quorum),
        "ALL" => Some(rekha_core::ConsistencyLevel::All),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dev_default() {
        let config = ServerConfig::dev_default("node-1", "/tmp/rekha");
        assert_eq!(config.cluster.node_id, "node-1");
        assert_eq!(config.cluster.bind_addr, "0.0.0.0:50051");
        assert_eq!(config.cluster.default_write_consistency, "QUORUM");
        assert!(config.cluster.hinted_handoff_enabled);
        assert_eq!(config.cluster.max_hint_window_secs, 10800);
        assert_eq!(config.storage.gc_grace_seconds, 864000);
    }

    #[test]
    fn test_config_from_file_not_found() {
        let result = ServerConfig::from_file("/nonexistent/path.yaml");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_roundtrip() {
        let config = ServerConfig::dev_default("test-node", "/data");
        let yaml = serde_yaml::to_string(&config).unwrap();
        let config2: ServerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config.cluster.node_id, config2.cluster.node_id);
        assert_eq!(config.cluster.seed_nodes, config2.cluster.seed_nodes);
    }

    #[test]
    fn test_tls_config_default_disabled() {
        let tls = TlsConfig::default();
        assert!(!tls.enabled);
        assert!(tls.cert_path.is_none());
        assert!(tls.key_path.is_none());
        assert!(tls.ca_cert_path.is_none());
    }

    #[test]
    fn test_dev_default_tls_disabled() {
        let config = ServerConfig::dev_default("n1", "/tmp");
        assert!(!config.tls.enabled);
    }

    #[test]
    fn test_parse_consistency() {
        assert_eq!(parse_consistency("ONE"), Some(rekha_core::ConsistencyLevel::One));
        assert_eq!(parse_consistency("one"), Some(rekha_core::ConsistencyLevel::One));
        assert_eq!(parse_consistency("QUORUM"), Some(rekha_core::ConsistencyLevel::Quorum));
        assert_eq!(parse_consistency("ALL"), Some(rekha_core::ConsistencyLevel::All));
        assert_eq!(parse_consistency("INVALID"), None);
    }
}
