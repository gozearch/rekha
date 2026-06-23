use rekha_core::NodeInfo;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct PeerState {
    pub info: NodeInfo,
    pub last_seen: Instant,
}

impl PeerState {
    pub fn new(info: NodeInfo) -> Self {
        Self { info, last_seen: Instant::now() }
    }
}
