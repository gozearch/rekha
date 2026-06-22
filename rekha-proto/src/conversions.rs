use rekha_core::{ConsistencyLevel, NodeStatus, PayloadType};

// ── Helper functions for cross-crate conversions ──────────────

pub fn consistency_to_i32(cl: ConsistencyLevel) -> i32 {
    cl.to_i32()
}

pub fn consistency_from_i32(v: i32) -> ConsistencyLevel {
    match v {
        1 => ConsistencyLevel::One,
        2 => ConsistencyLevel::Quorum,
        3 => ConsistencyLevel::All,
        _ => ConsistencyLevel::Quorum,
    }
}

pub fn node_status_to_string(s: &NodeStatus) -> String {
    match s {
        NodeStatus::Healthy => "healthy".into(),
        NodeStatus::Degraded => "degraded".into(),
        NodeStatus::Unreachable => "unreachable".into(),
        NodeStatus::Recovering => "recovering".into(),
        NodeStatus::Offline => "offline".into(),
    }
}

pub fn node_status_from_str(s: &str) -> NodeStatus {
    match s.to_lowercase().as_str() {
        "healthy" => NodeStatus::Healthy,
        "degraded" => NodeStatus::Degraded,
        "unreachable" => NodeStatus::Unreachable,
        "recovering" => NodeStatus::Recovering,
        "offline" => NodeStatus::Offline,
        _ => NodeStatus::Healthy,
    }
}

// ── Payload (proto::Payload is local, so From impls work) ────

impl From<crate::proto::Payload> for rekha_core::Payload {
    fn from(p: crate::proto::Payload) -> Self {
        rekha_core::Payload {
            content_type: match p.content_type.as_str() {
                "text" => PayloadType::Text,
                "json" => PayloadType::Json,
                _ => PayloadType::Raw,
            },
            data: p.data,
        }
    }
}

impl From<rekha_core::Payload> for crate::proto::Payload {
    fn from(p: rekha_core::Payload) -> Self {
        crate::proto::Payload {
            content_type: p.content_type.to_string(),
            data: p.data,
        }
    }
}

// ── CollectionConfig (proto::CollectionConfig is local) ─────

impl From<crate::proto::CollectionConfig> for rekha_core::CollectionConfig {
    fn from(c: crate::proto::CollectionConfig) -> Self {
        rekha_core::CollectionConfig {
            dim: c.dim,
            num_vector_shards: c.num_vector_shards,
            replication_factor: c.replication_factor,
            num_dim_groups: c.num_dim_groups,
            dim_group_size: c.dim_group_size,
            nlist: c.nlist,
            nprobe: c.nprobe,
            pq_num_sub_vectors: c.pq_num_sub_vectors,
            pq_num_centroids: c.pq_num_centroids,
            re_rank_k: c.re_rank_k,
        }
    }
}

impl From<rekha_core::CollectionConfig> for crate::proto::CollectionConfig {
    fn from(c: rekha_core::CollectionConfig) -> Self {
        crate::proto::CollectionConfig {
            dim: c.dim,
            num_vector_shards: c.num_vector_shards,
            replication_factor: c.replication_factor,
            num_dim_groups: c.num_dim_groups,
            dim_group_size: c.dim_group_size,
            nlist: c.nlist,
            nprobe: c.nprobe,
            pq_num_sub_vectors: c.pq_num_sub_vectors,
            pq_num_centroids: c.pq_num_centroids,
            re_rank_k: c.re_rank_k,
        }
    }
}

// ── ScoredPoint (proto::ScoredPoint is local) ──────────────

impl From<crate::proto::ScoredPoint> for rekha_core::ScoredPoint {
    fn from(s: crate::proto::ScoredPoint) -> Self {
        rekha_core::ScoredPoint {
            id: s.id,
            score: s.score,
            payload: s.payload.map(rekha_core::Payload::from),
            timestamp: s.timestamp,
        }
    }
}

impl From<rekha_core::ScoredPoint> for crate::proto::ScoredPoint {
    fn from(s: rekha_core::ScoredPoint) -> Self {
        crate::proto::ScoredPoint {
            id: s.id,
            score: s.score,
            payload: s.payload.map(crate::proto::Payload::from),
            timestamp: s.timestamp,
        }
    }
}

// ── SearchParams (proto::SearchParams is local) ─────────────

impl From<crate::proto::SearchParams> for rekha_core::SearchParams {
    fn from(s: crate::proto::SearchParams) -> Self {
        rekha_core::SearchParams {
            ef_search: s.ef_search as usize,
            nprobe: s.nprobe as usize,
            include_payloads: s.include_payloads,
            local_only: false,
        }
    }
}

impl From<rekha_core::SearchParams> for crate::proto::SearchParams {
    fn from(s: rekha_core::SearchParams) -> Self {
        crate::proto::SearchParams {
            ef_search: s.ef_search as u32,
            nprobe: s.nprobe as u32,
            include_payloads: s.include_payloads,
        }
    }
}

// ── SearchStats (proto::SearchStats is local) ──────────────

impl From<crate::proto::SearchStats> for rekha_core::SearchStats {
    fn from(s: crate::proto::SearchStats) -> Self {
        rekha_core::SearchStats {
            total_ms: s.total_ms,
            nodes_contacted: s.nodes_contacted,
            vectors_scanned: s.vectors_scanned,
            warnings: s.warnings,
        }
    }
}

impl From<rekha_core::SearchStats> for crate::proto::SearchStats {
    fn from(s: rekha_core::SearchStats) -> Self {
        crate::proto::SearchStats {
            total_ms: s.total_ms,
            nodes_contacted: s.nodes_contacted,
            vectors_scanned: s.vectors_scanned,
            warnings: s.warnings,
        }
    }
}

// ── NodeInfo (proto::NodeInfo is local) ─────────────────────

impl From<crate::proto::NodeInfo> for rekha_core::NodeInfo {
    fn from(n: crate::proto::NodeInfo) -> Self {
        rekha_core::NodeInfo {
            node_id: n.node_id,
            address: n.address,
            partition_id: n.partition_id,
            dim_groups: n.dim_groups,
            is_leader: false,
            raft_term: 0,
            commit_index: 0,
            storage_bytes: n.storage_bytes,
            status: node_status_from_str(&n.status),
            last_heartbeat: 0,
        }
    }
}

impl From<rekha_core::NodeInfo> for crate::proto::NodeInfo {
    fn from(n: rekha_core::NodeInfo) -> Self {
        crate::proto::NodeInfo {
            node_id: n.node_id,
            address: n.address,
            partition_id: n.partition_id,
            dim_groups: n.dim_groups,
            storage_bytes: n.storage_bytes,
            status: node_status_to_string(&n.status),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rekha_core::*;

    #[test]
    fn test_consistency_level_roundtrip() {
        for cl in &[ConsistencyLevel::One, ConsistencyLevel::Quorum, ConsistencyLevel::All] {
            let i = consistency_to_i32(*cl);
            let back = consistency_from_i32(i);
            assert_eq!(*cl, back);
        }
    }

    #[test]
    fn test_consistency_default_for_unknown() {
        let cl = consistency_from_i32(99);
        assert_eq!(cl, ConsistencyLevel::Quorum);
    }

    #[test]
    fn test_payload_roundtrip() {
        let p = Payload::from_text("hello");
        let proto: crate::proto::Payload = p.clone().into();
        let back: Payload = proto.into();
        assert_eq!(back.content_type, PayloadType::Text);
        assert_eq!(back.data, b"hello");
    }

    #[test]
    fn test_payload_json_roundtrip() {
        let p = Payload::from_json(&serde_json::json!({"k": "v"})).unwrap();
        let proto: crate::proto::Payload = p.clone().into();
        let back: Payload = proto.into();
        assert_eq!(back.content_type, PayloadType::Json);
    }

    #[test]
    fn test_collection_config_roundtrip() {
        let cfg = CollectionConfig::default();
        let proto: crate::proto::CollectionConfig = cfg.clone().into();
        let back: CollectionConfig = proto.into();
        assert_eq!(cfg.dim, back.dim);
        assert_eq!(cfg.nlist, back.nlist);
    }

    #[test]
    fn test_node_info_roundtrip() {
        let node = NodeInfo {
            node_id: "n1".into(),
            address: "127.0.0.1:50051".into(),
            partition_id: 1,
            dim_groups: vec![0, 1],
            is_leader: false,
            raft_term: 0,
            commit_index: 0,
            storage_bytes: 1024,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        };
        let proto: crate::proto::NodeInfo = node.clone().into();
        let back: NodeInfo = proto.into();
        assert_eq!(back.node_id, "n1");
        assert_eq!(back.status, NodeStatus::Healthy);
    }

    #[test]
    fn test_search_params_roundtrip() {
        let params = SearchParams::default();
        let proto: crate::proto::SearchParams = params.clone().into();
        let back: SearchParams = proto.into();
        assert_eq!(params.ef_search, back.ef_search);
        assert_eq!(params.nprobe, back.nprobe);
        assert!(!back.local_only);
    }

    #[test]
    fn test_node_status_string() {
        assert_eq!(node_status_to_string(&NodeStatus::Healthy), "healthy");
        assert_eq!(node_status_from_str("unreachable"), NodeStatus::Unreachable);
        assert_eq!(node_status_from_str("unknown"), NodeStatus::Healthy);
    }
}
