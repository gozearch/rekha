use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub cluster: ClusterConfig,
    pub tls: TlsConfig,
    pub observability: ObservabilityConfig,
    pub storage: StorageConfig,
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
}
