//! Collection and index configuration.
//!
//! `HnswConfig` carries the HNSW hyperparameters Chroma persists per segment
//! (`hnsw:M`, `hnsw:construction_ef`, `hnsw:ef_search`, `hnsw:space`,
//! `hnsw:batch_size`, `hnsw:sync_threshold`). The two thresholds drive the
//! index write pipeline: an in-memory brute-force buffer absorbs up to
//! `batch_size` pending records before they are merged into HNSW, and HNSW is
//! dumped to disk once pending records cross `sync_threshold`.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::{Distance, Metadata};

/// HNSW hyperparameters and write-buffering thresholds (Chroma defaults).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of bi-directional links per element (graph degree). Default 16.
    pub m: usize,
    /// Size of the dynamic candidate list during graph construction. Default 200.
    pub ef_construction: usize,
    /// Size of the dynamic candidate list during search. Default 100.
    pub ef_search: usize,
    /// Brute-force buffer flush threshold: merge this many buffered records
    /// into the HNSW graph at once. Default 100.
    pub batch_size: usize,
    /// Index-to-disk threshold: dump the HNSW graph to disk once this many
    /// records are pending. Default 1000.
    pub sync_threshold: usize,
    /// Brute-force fallback threshold: scan instead of graph-walk when the
    /// collection is this small. Default 10_000.
    pub max_scan: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            batch_size: 100,
            sync_threshold: 1000,
            max_scan: 10_000,
        }
    }
}

/// A fully-resolved collection definition: tenancy, naming, dimension, space,
/// optional metadata, HNSW tuning, and creation timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Globally-unique collection id (also the WAL topic and storage key).
    pub id: Uuid,
    /// Tenant namespace. Default `"default_tenant"`.
    pub tenant: String,
    /// Database namespace. Default `"default_database"`.
    pub database: String,
    /// Human-readable collection name.
    pub name: String,
    /// Embedding dimensionality. Every vector in the collection must match.
    pub dimension: usize,
    /// Distance space used for indexing and search.
    pub space: Distance,
    /// Collection-level metadata, if any.
    pub metadata: Option<Metadata>,
    /// Index tuning parameters.
    pub hnsw: HnswConfig,
    /// Creation timestamp, milliseconds since the Unix epoch.
    pub created_at_ms: u64,
}

impl CollectionConfig {
    /// Creates a collection config with a fresh v4 [`Uuid`], the default
    /// tenant/database, no metadata, default HNSW tuning, and the current
    /// wall-clock creation time.
    pub fn new(name: String, dimension: usize, space: Distance) -> Self {
        Self {
            id: Uuid::new_v4(),
            tenant: "default_tenant".to_owned(),
            database: "default_database".to_owned(),
            name,
            dimension,
            space,
            metadata: None,
            hnsw: HnswConfig::default(),
            created_at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hnsw_config_defaults() {
        let c = HnswConfig::default();
        assert_eq!(c.m, 16);
        assert_eq!(c.ef_construction, 200);
        assert_eq!(c.ef_search, 100);
        assert_eq!(c.batch_size, 100);
        assert_eq!(c.sync_threshold, 1000);
        assert_eq!(c.max_scan, 10_000);
    }

    #[test]
    fn collection_config_new_uses_defaults() {
        let c = CollectionConfig::new("test".into(), 128, Distance::Cosine);
        assert_eq!(c.name, "test");
        assert_eq!(c.dimension, 128);
        assert_eq!(c.space, Distance::Cosine);
        assert_eq!(c.tenant, "default_tenant");
        assert_eq!(c.database, "default_database");
        assert!(c.metadata.is_none());
        assert_eq!(c.hnsw.m, 16);
        assert_eq!(c.hnsw.batch_size, 100);
        assert!(c.created_at_ms > 0);
    }

    #[test]
    fn collection_config_ids_are_unique() {
        let a = CollectionConfig::new("a".into(), 4, Distance::L2);
        let b = CollectionConfig::new("b".into(), 4, Distance::L2);
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn configs_serialize_roundtrip() {
        let c = CollectionConfig::new("roundtrip".into(), 16, Distance::Ip);
        let json = serde_json::to_string(&c).unwrap();
        let back: CollectionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, c.id);
        assert_eq!(back.name, c.name);
        assert_eq!(back.space, c.space);
        assert_eq!(back.hnsw.sync_threshold, c.hnsw.sync_threshold);
    }
}
