use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub cluster: ClusterConfig,
    pub partition: PartitionConfig,
    pub index: IndexConfig,
    pub raft: RaftConfig,
    pub tls: TlsConfig,
    pub observability: ObservabilityConfig,
    pub storage: StorageConfig,
}

impl ServerConfig {
    /// Load configuration from a YAML file.
    pub fn from_file(path: impl Into<PathBuf>) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.into();
        let contents = std::fs::read_to_string(&path)?;
        let config: Self = serde_yaml::from_str(&contents)?;
        Ok(config)
    }

    /// Default configuration for a single-node development setup.
    pub fn dev_default(node_id: &str, data_dir: &str) -> Self {
        Self {
            cluster: ClusterConfig {
                node_id: node_id.to_string(),
                seed_nodes: vec![format!("127.0.0.1:50051")],
                bind_addr: "0.0.0.0:50051".into(),
                data_dir: data_dir.into(),
            },
            partition: PartitionConfig {
                num_vector_shards: 1,
                replication_factor: 1,
                num_dim_groups: 4,
                dim_group_size: 64,
            },
            index: IndexConfig {
                index_type: "vamana".into(),
                graph_degree: 64,
                search_list_size: 128,
                pq_num_sub_vectors: 64,
                pq_num_centroids: 256,
                re_rank_k: 256,
            },
            raft: RaftConfig {
                heartbeat_interval_ms: 100,
                election_timeout_min_ms: 300,
                election_timeout_max_ms: 500,
                snapshot_interval: 10_000,
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
            },
        }
    }
}

/// TLS configuration for encrypted gRPC communication.
///
/// Supports server-side TLS (encryption) and optional mutual TLS (mTLS)
/// for node identity verification.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TlsConfig {
    /// Set to `true` to enable TLS on the gRPC server and client.
    /// When disabled, all communication is plaintext HTTP/2.
    #[serde(default)]
    pub enabled: bool,
    /// Path to the server TLS certificate (PEM format).
    /// Required when `enabled` is true.
    pub cert_path: Option<String>,
    /// Path to the server TLS private key (PEM format).
    /// Required when `enabled` is true.
    pub key_path: Option<String>,
    /// Optional CA certificate for verifying client certificates (mTLS).
    /// When set, the server will request and verify client certificates.
    pub ca_cert_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub node_id: String,
    pub seed_nodes: Vec<String>,
    pub bind_addr: String,
    pub data_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    pub num_vector_shards: u64,
    pub replication_factor: usize,
    pub num_dim_groups: u32,
    pub dim_group_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub index_type: String,
    pub graph_degree: usize,
    pub search_list_size: usize,
    pub pq_num_sub_vectors: usize,
    pub pq_num_centroids: usize,
    pub re_rank_k: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftConfig {
    pub heartbeat_interval_ms: u64,
    pub election_timeout_min_ms: u64,
    pub election_timeout_max_ms: u64,
    pub snapshot_interval: u64,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dev_default() {
        let config = ServerConfig::dev_default("node-1", "/tmp/rekha");
        assert_eq!(config.cluster.node_id, "node-1");
        assert_eq!(config.cluster.bind_addr, "0.0.0.0:50051");
        assert_eq!(config.partition.num_vector_shards, 1);
        assert_eq!(config.partition.replication_factor, 1);
        assert_eq!(config.partition.num_dim_groups, 4);
        assert_eq!(config.index.graph_degree, 64);
        assert_eq!(config.index.pq_num_sub_vectors, 64);
        assert_eq!(config.raft.heartbeat_interval_ms, 100);
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
        assert_eq!(
            config.partition.num_vector_shards,
            config2.partition.num_vector_shards
        );
        assert_eq!(
            config.index.pq_num_sub_vectors,
            config2.index.pq_num_sub_vectors
        );
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
    fn test_tls_config_enabled() {
        let tls = TlsConfig {
            enabled: true,
            cert_path: Some("/etc/certs/server.pem".into()),
            key_path: Some("/etc/certs/server.key".into()),
            ca_cert_path: None,
        };
        assert!(tls.enabled);
        assert_eq!(tls.cert_path.as_deref(), Some("/etc/certs/server.pem"));
        assert_eq!(tls.key_path.as_deref(), Some("/etc/certs/server.key"));
    }

    #[test]
    fn test_dev_default_tls_disabled() {
        let config = ServerConfig::dev_default("n1", "/tmp");
        assert!(!config.tls.enabled);
    }

    #[test]
    fn test_tls_config_serde_roundtrip() {
        let tls = TlsConfig {
            enabled: true,
            cert_path: Some("/certs/cert.pem".into()),
            key_path: Some("/certs/key.pem".into()),
            ca_cert_path: Some("/certs/ca.pem".into()),
        };
        let yaml = serde_yaml::to_string(&tls).unwrap();
        let tls2: TlsConfig = serde_yaml::from_str(&yaml).unwrap();
        assert!(tls2.enabled);
        assert_eq!(tls2.cert_path, tls.cert_path);
        assert_eq!(tls2.key_path, tls.key_path);
        assert_eq!(tls2.ca_cert_path, tls.ca_cert_path);
    }
}
