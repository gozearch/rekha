use rekha_core::{NodeInfo, now_epoch_secs};

use crate::Coordinator;

impl Coordinator {
    pub async fn register_peer(&self, mut info: NodeInfo) {
        info.last_heartbeat = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
        self.membership.write().await.register(info);
        self.refresh_peer_pool().await;
    }

    async fn refresh_peer_pool(&self) {
        let healthy = self.membership.read().await.healthy_peers();
        let mut pool = self.peer_pool.write().await;
        let external: Vec<NodeInfo> = healthy.into_iter()
            .filter(|p| p.node_id != self.config.node_id).collect();
        pool.refresh(&external).await;
    }

    pub async fn check_peer_health(&self) {
        let recovered_nodes = self.membership.write().await.check_health();
        self.sync_topology().await;

        if !recovered_nodes.is_empty() && self.handoff.is_enabled() {
            self.refresh_peer_pool().await;
            for peer_id in &recovered_nodes {
                let hint_store = self.store.hint_store();
                if let Ok(hints) = hint_store.iter_hints_for_node(peer_id) {
                    let max_age = now_epoch_secs().saturating_sub(self.config.max_hint_window_secs);
                    for hint in hints {
                        if hint.timestamp / 1_000_000 < max_age {
                            let _ = hint_store.delete_hint(&hint.target_node_id, &hint.collection, hint.id);
                            continue;
                        }
                        let mut pool = self.peer_pool.write().await;
                        if let Some(client) = pool.clients.get_mut(peer_id) {
                            if client.try_remote_insert(&hint.collection, hint.id, &hint.vector, &hint.payload, hint.timestamp).await.is_ok() {
                                let _ = hint_store.delete_hint(&hint.target_node_id, &hint.collection, hint.id);
                            }
                        }
                    }
                }
            }
        }
    }

    async fn sync_topology(&self) {
        let members = self.membership.read().await;
        let mut nodes = std::collections::HashMap::new();
        nodes.insert(self.config.node_id.clone(), self.local_node_info());
        for peer in members.all_peers() {
            nodes.insert(peer.node_id.clone(), peer);
        }
        let mut topo = self.topology.write().await;
        topo.nodes = nodes;
    }
}
