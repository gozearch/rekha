use rekha_core::ConsistencyLevel;

#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    pub node_id: String,
    pub bind_addr: String,
    pub seed_nodes: Vec<String>,
    pub default_write_consistency: String,
    pub hinted_handoff_enabled: bool,
    pub max_hint_window_secs: u64,
    pub gc_grace_seconds: u64,
    pub peer_timeout_ms: u64,
}

impl CoordinatorConfig {
    pub fn dev_default(node_id: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            bind_addr: "0.0.0.0:50051".into(),
            seed_nodes: vec!["127.0.0.1:50051".into()],
            default_write_consistency: "QUORUM".into(),
            hinted_handoff_enabled: true,
            max_hint_window_secs: 10800,
            gc_grace_seconds: 864000,
            peer_timeout_ms: 10000,
        }
    }
}

pub(super) fn parse_consistency(s: &str) -> Option<ConsistencyLevel> {
    match s.to_uppercase().as_str() {
        "ONE" => Some(ConsistencyLevel::One),
        "QUORUM" => Some(ConsistencyLevel::Quorum),
        "ALL" => Some(ConsistencyLevel::All),
        _ => None,
    }
}
