use rekha_core::{ConsistencyLevel, DistanceMetric, IvfConfig, ScoredPoint};

use crate::proto;

impl From<ConsistencyLevel> for proto::ConsistencyLevel {
    fn from(cl: ConsistencyLevel) -> Self {
        match cl {
            ConsistencyLevel::One => proto::ConsistencyLevel::One,
            ConsistencyLevel::Quorum => proto::ConsistencyLevel::Quorum,
            ConsistencyLevel::All => proto::ConsistencyLevel::All,
        }
    }
}

impl From<proto::ConsistencyLevel> for ConsistencyLevel {
    fn from(cl: proto::ConsistencyLevel) -> Self {
        match cl {
            proto::ConsistencyLevel::One => ConsistencyLevel::One,
            proto::ConsistencyLevel::Quorum => ConsistencyLevel::Quorum,
            proto::ConsistencyLevel::All => ConsistencyLevel::All,
            _ => ConsistencyLevel::Quorum,
        }
    }
}

impl From<proto::CollectionConfig> for IvfConfig {
    fn from(cfg: proto::CollectionConfig) -> Self {
        IvfConfig {
            dim: cfg.dim,
            nlist: cfg.nlist,
            nprobe: cfg.nprobe,
            pq_m: cfg.pq_num_sub_vectors,
            pq_k: cfg.pq_num_centroids as u16,
            replication_factor: cfg.replication_factor as u32,
            distance_metric: DistanceMetric::L2,
        }
    }
}

impl From<IvfConfig> for proto::CollectionConfig {
    fn from(cfg: IvfConfig) -> Self {
        proto::CollectionConfig {
            dim: cfg.dim,
            nlist: cfg.nlist,
            nprobe: cfg.nprobe,
            pq_num_sub_vectors: cfg.pq_m,
            pq_num_centroids: cfg.pq_k as u32,
            num_vector_shards: 1,
            replication_factor: cfg.replication_factor as u64,
            num_dim_groups: 1,
            dim_group_size: cfg.dim,
            re_rank_k: 0,
        }
    }
}

impl From<proto::ScoredPoint> for ScoredPoint {
    fn from(sp: proto::ScoredPoint) -> Self {
        ScoredPoint {
            id: sp.id,
            score: sp.score,
            payload: sp.payload.map(|p| p.data),
            timestamp: sp.timestamp as i64,
        }
    }
}

impl From<ScoredPoint> for proto::ScoredPoint {
    fn from(sp: ScoredPoint) -> Self {
        proto::ScoredPoint {
            id: sp.id,
            score: sp.score,
            payload: sp.payload.map(|data| proto::Payload {
                content_type: "application/octet-stream".into(),
                data,
            }),
            timestamp: sp.timestamp as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistency_level_roundtrip() {
        let levels = vec![
            ConsistencyLevel::One,
            ConsistencyLevel::Quorum,
            ConsistencyLevel::All,
        ];
        for cl in levels {
            let proto: proto::ConsistencyLevel = cl.into();
            let back: ConsistencyLevel = proto.into();
            assert_eq!(cl, back);
        }
    }

    #[test]
    fn test_collection_config_roundtrip() {
        let cfg = IvfConfig {
            dim: 128,
            nlist: 1024,
            nprobe: 64,
            pq_m: 16,
            pq_k: 256,
            replication_factor: 3,
            distance_metric: DistanceMetric::L2,
        };
        let proto: proto::CollectionConfig = cfg.clone().into();
        let back: IvfConfig = proto.into();
        assert_eq!(cfg.dim, back.dim);
        assert_eq!(cfg.nlist, back.nlist);
        assert_eq!(cfg.pq_m, back.pq_m);
    }

    #[test]
    fn test_scored_point_roundtrip() {
        let sp = ScoredPoint {
            id: 42,
            score: 0.95,
            payload: Some(vec![1, 2, 3]),
            timestamp: 1000,
        };
        let proto_sp: proto::ScoredPoint = sp.clone().into();
        let back: ScoredPoint = proto_sp.into();
        assert_eq!(sp.id, back.id);
        assert!((sp.score - back.score).abs() < 1e-6);
        assert_eq!(sp.payload, back.payload);
    }

    #[test]
    fn test_default_consistency() {
        let proto_cl = proto::ConsistencyLevel::ConsistencyUnspecified;
        let cl: ConsistencyLevel = proto_cl.into();
        assert_eq!(cl, ConsistencyLevel::Quorum);
    }
}
