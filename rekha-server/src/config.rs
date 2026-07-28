use rekha_core::ConsistencyLevel;
use serde::{Deserialize, Deserializer, Serialize};

fn deserialize_consistency<'de, D>(deserializer: D) -> Result<ConsistencyLevel, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Ok(match s.to_lowercase().as_str() {
        "one" => ConsistencyLevel::One,
        "all" => ConsistencyLevel::All,
        _ => ConsistencyLevel::Quorum,
    })
}

fn default_consistency() -> ConsistencyLevel {
    ConsistencyLevel::Quorum
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    #[serde(
        default = "default_consistency",
        deserialize_with = "deserialize_consistency"
    )]
    pub default_write_consistency: ConsistencyLevel,
    #[serde(default = "default_true")]
    pub hinted_handoff_enabled: bool,
    #[serde(default = "default_max_hint_window")]
    pub max_hint_window_secs: i64,
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval_ms: u64,
    #[serde(default = "default_heartbeat_timeout")]
    pub heartbeat_timeout_ms: u64,
    #[serde(default)]
    pub seed_nodes: Vec<String>,
    #[serde(default = "default_rf")]
    pub default_rf: u32,
}

fn default_rf() -> u32 {
    3
}

fn default_true() -> bool {
    true
}

fn default_max_hint_window() -> i64 {
    3600
}

fn default_heartbeat_interval() -> u64 {
    1000
}

fn default_heartbeat_timeout() -> u64 {
    5000
}

impl Default for ClusterConfig {
    fn default() -> Self {
        ClusterConfig {
            default_write_consistency: default_consistency(),
            hinted_handoff_enabled: true,
            max_hint_window_secs: default_max_hint_window(),
            heartbeat_interval_ms: default_heartbeat_interval(),
            heartbeat_timeout_ms: default_heartbeat_timeout(),
            seed_nodes: Vec::new(),
            default_rf: default_rf(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    #[serde(default = "default_max_payload_size")]
    pub max_payload_size: usize,
    #[serde(default = "default_max_inline_size")]
    pub max_inline_size: usize,
    #[serde(default = "default_gc_grace")]
    pub gc_grace_seconds: i64, // Tombstone retention before compaction
    #[serde(default = "default_gc_interval")]
    pub gc_interval_secs: u64,
}

fn default_max_payload_size() -> usize {
    4 * 1024 * 1024
}

fn default_max_inline_size() -> usize {
    1024
}

fn default_gc_grace() -> i64 {
    86400
}

fn default_gc_interval() -> u64 {
    3600
}

impl Default for StorageConfig {
    fn default() -> Self {
        StorageConfig {
            max_payload_size: default_max_payload_size(),
            max_inline_size: default_max_inline_size(),
            gc_grace_seconds: default_gc_grace(),
            gc_interval_secs: default_gc_interval(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TlsConfig {
    pub enabled: bool,
    pub cert_path: Option<String>,
    pub key_path: Option<String>,
    #[serde(default)]
    pub client_ca_cert_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    #[serde(default)]
    pub enable_tracing: bool,
    #[serde(default)]
    pub enable_metrics: bool,
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,
}

fn default_metrics_port() -> u16 {
    9090
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        ObservabilityConfig {
            enable_tracing: false,
            enable_metrics: false,
            metrics_port: default_metrics_port(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_listen")]
    pub listen: String,
    pub advertise_address: Option<String>,
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
    #[serde(default = "default_node_id")]
    pub node_id: String,
    #[serde(default)]
    pub cluster: ClusterConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub tls: TlsConfig,
    #[serde(default)]
    pub observability: ObservabilityConfig,
}

impl ServerConfig {
    pub fn advertise(&self) -> &str {
        self.advertise_address.as_deref().unwrap_or(&self.listen)
    }
}

fn default_listen() -> String {
    "0.0.0.0:50051".to_string()
}

fn default_data_dir() -> String {
    "/tmp/rekha-data".to_string()
}

fn default_node_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            listen: default_listen(),
            advertise_address: None,
            data_dir: default_data_dir(),
            node_id: default_node_id(),
            cluster: ClusterConfig::default(),
            storage: StorageConfig::default(),
            tls: TlsConfig::default(),
            observability: ObservabilityConfig::default(),
        }
    }
}

impl ServerConfig {
    pub fn from_file(path: &str) -> Result<Self, anyhow::Error> {
        let content = std::fs::read_to_string(path)?;
        let config: ServerConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.listen, "0.0.0.0:50051");
        assert!(cfg.cluster.hinted_handoff_enabled);
        assert_eq!(cfg.cluster.max_hint_window_secs, 3600);
    }

    #[test]
    fn test_config_serde() {
        let cfg = ServerConfig::default();
        let yaml = serde_yaml::to_string(&cfg).unwrap();
        let parsed: ServerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.listen, cfg.listen);
        assert_eq!(
            parsed.cluster.max_hint_window_secs,
            cfg.cluster.max_hint_window_secs
        );
    }

    #[test]
    fn test_tls_default_disabled() {
        let cfg = ServerConfig::default();
        assert!(!cfg.tls.enabled);
        assert!(cfg.tls.cert_path.is_none());
    }

    #[test]
    fn test_config_yaml_parses_listen_and_seed_nodes() {
        let yaml = r#"
listen: "0.0.0.0:50051"
data_dir: "/data"
node_id: "node-1"
cluster:
  seed_nodes:
    - "node-1:50051"
    - "node-2:50051"
"#;
        let cfg: ServerConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.listen, "0.0.0.0:50051");
        assert_eq!(cfg.data_dir, "/data");
        assert_eq!(cfg.node_id, "node-1");
        assert_eq!(cfg.cluster.seed_nodes.len(), 2);
        assert!(cfg.cluster.seed_nodes.contains(&"node-1:50051".to_string()));
        assert!(cfg.cluster.seed_nodes.contains(&"node-2:50051".to_string()));
    }

    #[test]
    fn test_advertise_address_fallback() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.advertise(), cfg.listen);
    }

    #[test]
    fn test_advertise_address_override() {
        let yaml = r#"
listen: "0.0.0.0:50051"
advertise_address: "node-2:50051"
"#;
        let cfg: ServerConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.advertise(), "node-2:50051");
        assert_eq!(cfg.listen, "0.0.0.0:50051");
    }
}
