#[non_exhaustive]
#[derive(Debug, Clone, thiserror::Error)]
pub enum StorageError {
    #[error("failed to open db at {path}: {msg}")]
    DbOpen { path: String, msg: String },
    #[error("column family {name}: {msg}")]
    ColumnFamily { name: String, msg: String },
    #[error("read error at key {key:?}: {msg}")]
    Read { key: Vec<u8>, msg: String },
    #[error("write error: {msg}")]
    Write { msg: String },
    #[error("batch write: committed {committed}, failed {failed}: {msg}")]
    BatchWrite { committed: usize, failed: usize, msg: String },
    #[error("corruption detected: {detail}")]
    Corruption { detail: String },
    #[error("serialization error: {detail}")]
    Serialization { detail: String },
    #[error("payload too large: {size} bytes (max {max})")]
    PayloadTooLarge { size: usize, max: usize },
}

#[non_exhaustive]
#[derive(Debug, Clone, thiserror::Error)]
pub enum IndexError {
    #[error("graph build failed: {detail}")]
    GraphBuild { detail: String },
    #[error("search failed: {detail}")]
    Search { detail: String },
    #[error("ef_search {ef} exceeds max {max}")]
    InvalidEfSearch { ef: usize, max: usize },
    #[error("index is empty")]
    EmptyIndex,
    #[error("{component} has not been trained")]
    NotTrained { component: &'static str },
    #[error("expected dimension {expected}, got {actual}")]
    IncompatibleDimension { expected: usize, actual: usize },
}

#[non_exhaustive]
#[derive(Debug, Clone, thiserror::Error)]
pub enum PartitionError {
    #[error("no available nodes for partition {partition_id}")]
    NoNodesAvailable { partition_id: u64 },
    #[error("invalid topology: {detail}")]
    InvalidTopology { detail: String },
    #[error("rebalance in progress for partition {partition_id}")]
    RebalanceInProgress { partition_id: u64 },
    #[error("dimension group mismatch: expected {expected}, got {actual}")]
    DimensionGroupMismatch { expected: u32, actual: u32 },
    #[error("shard {shard_id} not found")]
    ShardNotFound { shard_id: u64 },
}

#[non_exhaustive]
#[derive(Debug, Clone, thiserror::Error)]
pub enum RekhaError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("index full: capacity={capacity}, attempted={attempted}")]
    IndexFull { capacity: usize, attempted: usize },
    #[error("invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error(transparent)]
    Index(#[from] IndexError),
    #[error(transparent)]
    Partition(#[from] PartitionError),
    #[error("timeout on {operation} after {elapsed_ms}ms")]
    Timeout { operation: &'static str, elapsed_ms: u64 },
    #[error("cluster membership changed: {detail}")]
    ClusterChanged { detail: String },
    #[error("service unavailable: {detail}")]
    Unavailable { detail: String },
    #[error("internal error: {detail}")]
    Internal { detail: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_rekha_error_display_not_found() {
        let err = RekhaError::NotFound("vector 42".into());
        assert_eq!(err.to_string(), "not found: vector 42");
    }

    #[test]
    fn test_rekha_error_display_invalid_argument() {
        let err = RekhaError::InvalidArgument("bad dims".into());
        assert_eq!(err.to_string(), "invalid argument: bad dims");
    }

    #[test]
    fn test_rekha_error_display_index_full() {
        let err = RekhaError::IndexFull { capacity: 1000, attempted: 1001 };
        assert_eq!(err.to_string(), "index full: capacity=1000, attempted=1001");
    }

    #[test]
    fn test_rekha_error_display_invalid_dimension() {
        let err = RekhaError::InvalidDimension { expected: 768, actual: 64 };
        assert_eq!(err.to_string(), "invalid dimension: expected 768, got 64");
    }

    #[test]
    fn test_rekha_error_display_timeout() {
        let err = RekhaError::Timeout { operation: "search", elapsed_ms: 5000 };
        assert_eq!(err.to_string(), "timeout on search after 5000ms");
    }

    #[test]
    fn test_rekha_error_display_unavailable() {
        let err = RekhaError::Unavailable { detail: "node down".into() };
        assert_eq!(err.to_string(), "service unavailable: node down");
    }

    #[test]
    fn test_rekha_error_display_internal() {
        let err = RekhaError::Internal { detail: "oops".into() };
        assert_eq!(err.to_string(), "internal error: oops");
    }

    #[test]
    fn test_storage_error_display() {
        let err = StorageError::DbOpen { path: "/data/db".into(), msg: "permission denied".into() };
        assert_eq!(err.to_string(), "failed to open db at /data/db: permission denied");
        let err = StorageError::PayloadTooLarge { size: 2_000_000, max: 1_048_576 };
        assert_eq!(err.to_string(), "payload too large: 2000000 bytes (max 1048576)");
    }

    #[test]
    fn test_index_error_display() {
        let err = IndexError::EmptyIndex;
        assert_eq!(err.to_string(), "index is empty");
        let err = IndexError::InvalidEfSearch { ef: 200, max: 128 };
        assert_eq!(err.to_string(), "ef_search 200 exceeds max 128");
    }

    #[test]
    fn test_partition_error_display() {
        let err = PartitionError::NoNodesAvailable { partition_id: 3 };
        assert_eq!(err.to_string(), "no available nodes for partition 3");
    }

    #[test]
    fn test_error_source() {
        let err = StorageError::Corruption { detail: "bad block".into() };
        let re: RekhaError = err.clone().into();
        assert!(re.source().is_none(), "Corruption has no underlying source");

        let idx_err = IndexError::Search { detail: "failed".into() };
        let re2: RekhaError = idx_err.into();
        assert!(re2.source().is_none(), "Search error has no underlying source");
    }

    #[test]
    fn test_into_conversions() {
        let s = StorageError::Write { msg: "disk full".into() };
        let e: RekhaError = s.into();
        assert!(matches!(e, RekhaError::Storage(_)));

        let i = IndexError::EmptyIndex;
        let e: RekhaError = i.into();
        assert!(matches!(e, RekhaError::Index(_)));

        let p = PartitionError::ShardNotFound { shard_id: 5 };
        let e: RekhaError = p.into();
        assert!(matches!(e, RekhaError::Partition(_)));
    }

    #[test]
    fn test_rekha_error_cluster_changed() {
        let err = RekhaError::ClusterChanged { detail: "new node joined".into() };
        assert_eq!(err.to_string(), "cluster membership changed: new node joined");
    }

    #[test]
    fn test_storage_error_column_family() {
        let err = StorageError::ColumnFamily { name: "vectors".into(), msg: "handle not found".into() };
        assert_eq!(err.to_string(), "column family vectors: handle not found");
    }

    #[test]
    fn test_storage_error_read() {
        let err = StorageError::Read { key: vec![0, 0, 0, 0, 0, 0, 0, 42], msg: "IO error".into() };
        assert_eq!(err.to_string(), "read error at key [0, 0, 0, 0, 0, 0, 0, 42]: IO error");
    }

    #[test]
    fn test_storage_error_write() {
        let err = StorageError::Write { msg: "disk full".into() };
        assert_eq!(err.to_string(), "write error: disk full");
    }

    #[test]
    fn test_storage_error_batch_write() {
        let err = StorageError::BatchWrite { committed: 5, failed: 2, msg: "timeout".into() };
        assert_eq!(err.to_string(), "batch write: committed 5, failed 2: timeout");
    }

    #[test]
    fn test_storage_error_corruption() {
        let err = StorageError::Corruption { detail: "checksum mismatch".into() };
        assert_eq!(err.to_string(), "corruption detected: checksum mismatch");
    }

    #[test]
    fn test_storage_error_serialization() {
        let err = StorageError::Serialization { detail: "invalid encoding".into() };
        assert_eq!(err.to_string(), "serialization error: invalid encoding");
    }

    #[test]
    fn test_index_error_graph_build() {
        let err = IndexError::GraphBuild { detail: "out of memory".into() };
        assert_eq!(err.to_string(), "graph build failed: out of memory");
    }

    #[test]
    fn test_index_error_search() {
        let err = IndexError::Search { detail: "no results".into() };
        assert_eq!(err.to_string(), "search failed: no results");
    }

    #[test]
    fn test_index_error_not_trained() {
        let err = IndexError::NotTrained { component: "ProductQuantizer" };
        assert_eq!(err.to_string(), "ProductQuantizer has not been trained");
    }

    #[test]
    fn test_index_error_incompatible_dim() {
        let err = IndexError::IncompatibleDimension { expected: 768, actual: 64 };
        assert_eq!(err.to_string(), "expected dimension 768, got 64");
    }

    #[test]
    fn test_partition_error_invalid_topology() {
        let err = PartitionError::InvalidTopology { detail: "overlapping ranges".into() };
        assert_eq!(err.to_string(), "invalid topology: overlapping ranges");
    }

    #[test]
    fn test_partition_error_rebalance() {
        let err = PartitionError::RebalanceInProgress { partition_id: 7 };
        assert_eq!(err.to_string(), "rebalance in progress for partition 7");
    }

    #[test]
    fn test_partition_error_dim_group_mismatch() {
        let err = PartitionError::DimensionGroupMismatch { expected: 4, actual: 2 };
        assert_eq!(err.to_string(), "dimension group mismatch: expected 4, got 2");
    }

    #[test]
    fn test_partition_error_shard_not_found() {
        let err = PartitionError::ShardNotFound { shard_id: 5 };
        assert_eq!(err.to_string(), "shard 5 not found");
    }
}
