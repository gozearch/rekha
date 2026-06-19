use std::fmt;

/// Top-level error for the Rekha distributed vector database.
/// Every error in the system ultimately flows through this type.
#[derive(Debug, Clone)]
pub enum RekhaError {
    NotFound(String),
    InvalidArgument(String),
    IndexFull {
        capacity: usize,
        attempted: usize,
    },
    InvalidDimension {
        expected: usize,
        actual: usize,
    },
    Storage(StorageError),
    Index(IndexError),
    Partition(PartitionError),
    Consensus(RaftError),
    Timeout {
        operation: &'static str,
        elapsed_ms: u64,
    },
    ClusterChanged {
        detail: String,
    },
    CollectionNotFound(String),
    CollectionAlreadyExists(String),
    Unavailable {
        detail: String,
    },
    Internal {
        detail: String,
    },
}

impl fmt::Display for RekhaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(s) => write!(f, "not found: {s}"),
            Self::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
            Self::IndexFull {
                capacity,
                attempted,
            } => {
                write!(f, "index full: capacity={capacity}, attempted={attempted}")
            }
            Self::InvalidDimension { expected, actual } => {
                write!(f, "invalid dimension: expected {expected}, got {actual}")
            }
            Self::Storage(e) => write!(f, "storage error: {e}"),
            Self::Index(e) => write!(f, "index error: {e}"),
            Self::Partition(e) => write!(f, "partition error: {e}"),
            Self::Consensus(e) => write!(f, "consensus error: {e}"),
            Self::Timeout {
                operation,
                elapsed_ms,
            } => {
                write!(f, "timeout on {operation} after {elapsed_ms}ms")
            }
            Self::ClusterChanged { detail } => write!(f, "cluster membership changed: {detail}"),
            Self::CollectionNotFound(s) => write!(f, "collection not found: {s}"),
            Self::CollectionAlreadyExists(s) => write!(f, "collection already exists: {s}"),
            Self::Unavailable { detail } => write!(f, "service unavailable: {detail}"),
            Self::Internal { detail } => write!(f, "internal error: {detail}"),
        }
    }
}

impl std::error::Error for RekhaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Storage(e) => Some(e),
            Self::Index(e) => Some(e),
            Self::Partition(e) => Some(e),
            Self::Consensus(e) => Some(e),
            _ => None,
        }
    }
}

// Allow easy conversion from specific errors.
impl From<StorageError> for RekhaError {
    fn from(e: StorageError) -> Self {
        Self::Storage(e)
    }
}
impl From<IndexError> for RekhaError {
    fn from(e: IndexError) -> Self {
        Self::Index(e)
    }
}
impl From<PartitionError> for RekhaError {
    fn from(e: PartitionError) -> Self {
        Self::Partition(e)
    }
}
impl From<RaftError> for RekhaError {
    fn from(e: RaftError) -> Self {
        Self::Consensus(e)
    }
}

/// Storage-layer errors (RocksDB, serialization, etc.)
#[derive(Debug, Clone)]
pub enum StorageError {
    DbOpen {
        path: String,
        source: String,
    },
    ColumnFamily {
        name: String,
        source: String,
    },
    Read {
        key: Vec<u8>,
        source: String,
    },
    Write {
        source: String,
    },
    BatchWrite {
        committed: usize,
        failed: usize,
        source: String,
    },
    Corruption {
        detail: String,
    },
    Serialization {
        detail: String,
    },
    PayloadTooLarge {
        size: usize,
        max: usize,
    },
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DbOpen { path, source } => write!(f, "failed to open db at {path}: {source}"),
            Self::ColumnFamily { name, source } => write!(f, "column family {name}: {source}"),
            Self::Read { key, source } => write!(f, "read error at key {key:?}: {source}"),
            Self::Write { source } => write!(f, "write error: {source}"),
            Self::BatchWrite {
                committed,
                failed,
                source,
            } => {
                write!(
                    f,
                    "batch write: committed {committed}, failed {failed}: {source}"
                )
            }
            Self::Corruption { detail } => write!(f, "corruption detected: {detail}"),
            Self::Serialization { detail } => write!(f, "serialization error: {detail}"),
            Self::PayloadTooLarge { size, max } => {
                write!(f, "payload too large: {size} bytes (max {max})")
            }
        }
    }
}
impl std::error::Error for StorageError {}

/// Index-layer errors (Vamana graph, PQ, search)
#[derive(Debug, Clone)]
pub enum IndexError {
    GraphBuild { detail: String },
    Search { detail: String },
    InvalidEfSearch { ef: usize, max: usize },
    EmptyIndex,
    NotTrained { component: &'static str },
    IncompatibleDimension { expected: usize, actual: usize },
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GraphBuild { detail } => write!(f, "graph build failed: {detail}"),
            Self::Search { detail } => write!(f, "search failed: {detail}"),
            Self::InvalidEfSearch { ef, max } => write!(f, "ef_search {ef} exceeds max {max}"),
            Self::EmptyIndex => write!(f, "index is empty"),
            Self::NotTrained { component } => write!(f, "{component} has not been trained"),
            Self::IncompatibleDimension { expected, actual } => {
                write!(f, "expected dimension {expected}, got {actual}")
            }
        }
    }
}
impl std::error::Error for IndexError {}

/// Partition-layer errors
#[derive(Debug, Clone)]
pub enum PartitionError {
    NoNodesAvailable { partition_id: u64 },
    InvalidTopology { detail: String },
    RebalanceInProgress { partition_id: u64 },
    DimensionGroupMismatch { expected: u32, actual: u32 },
    ShardNotFound { shard_id: u64 },
}

impl fmt::Display for PartitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoNodesAvailable { partition_id } => {
                write!(f, "no available nodes for partition {partition_id}")
            }
            Self::InvalidTopology { detail } => write!(f, "invalid topology: {detail}"),
            Self::RebalanceInProgress { partition_id } => {
                write!(f, "rebalance in progress for partition {partition_id}")
            }
            Self::DimensionGroupMismatch { expected, actual } => {
                write!(
                    f,
                    "dimension group mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ShardNotFound { shard_id } => write!(f, "shard {shard_id} not found"),
        }
    }
}
impl std::error::Error for PartitionError {}

/// Raft consensus errors
#[derive(Debug, Clone)]
pub enum RaftError {
    NotLeader { leader_hint: Option<String> },
    ElectionTimeout,
    LogCompaction { detail: String },
    SnapshotFailed { detail: String },
    MembershipChange { detail: String },
    CommitFailed { index: u64, term: u64 },
    ReplicationFailed { detail: String },
}

impl fmt::Display for RaftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotLeader { leader_hint } => match leader_hint {
                Some(h) => write!(f, "not leader, try {h}"),
                None => write!(f, "not leader, leader unknown"),
            },
            Self::ElectionTimeout => write!(f, "election timed out"),
            Self::LogCompaction { detail } => write!(f, "log compaction: {detail}"),
            Self::SnapshotFailed { detail } => write!(f, "snapshot failed: {detail}"),
            Self::MembershipChange { detail } => write!(f, "membership change: {detail}"),
            Self::CommitFailed { index, term } => {
                write!(f, "commit failed at index {index} term {term}")
            }
            Self::ReplicationFailed { detail } => {
                write!(f, "replication failed: {detail}")
            }
        }
    }
}
impl std::error::Error for RaftError {}

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
        let err = RekhaError::IndexFull {
            capacity: 1000,
            attempted: 1001,
        };
        assert_eq!(err.to_string(), "index full: capacity=1000, attempted=1001");
    }

    #[test]
    fn test_rekha_error_display_invalid_dimension() {
        let err = RekhaError::InvalidDimension {
            expected: 768,
            actual: 64,
        };
        assert_eq!(err.to_string(), "invalid dimension: expected 768, got 64");
    }

    #[test]
    fn test_rekha_error_display_timeout() {
        let err = RekhaError::Timeout {
            operation: "search",
            elapsed_ms: 5000,
        };
        assert_eq!(err.to_string(), "timeout on search after 5000ms");
    }

    #[test]
    fn test_rekha_error_display_unavailable() {
        let err = RekhaError::Unavailable {
            detail: "node down".into(),
        };
        assert_eq!(err.to_string(), "service unavailable: node down");
    }

    #[test]
    fn test_rekha_error_display_internal() {
        let err = RekhaError::Internal {
            detail: "oops".into(),
        };
        assert_eq!(err.to_string(), "internal error: oops");
    }

    #[test]
    fn test_storage_error_display() {
        let err = StorageError::DbOpen {
            path: "/data/db".into(),
            source: "permission denied".into(),
        };
        assert_eq!(
            err.to_string(),
            "failed to open db at /data/db: permission denied"
        );

        let err = StorageError::PayloadTooLarge {
            size: 2_000_000,
            max: 1_048_576,
        };
        assert_eq!(
            err.to_string(),
            "payload too large: 2000000 bytes (max 1048576)"
        );
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
    fn test_raft_error_display() {
        let err = RaftError::NotLeader {
            leader_hint: Some("node-2".into()),
        };
        assert_eq!(err.to_string(), "not leader, try node-2");

        let err = RaftError::ElectionTimeout;
        assert_eq!(err.to_string(), "election timed out");

        let err = RaftError::ReplicationFailed {
            detail: "only got 1/3 acks".into(),
        };
        assert_eq!(err.to_string(), "replication failed: only got 1/3 acks");
    }

    #[test]
    fn test_error_source() {
        let storage_err = StorageError::Corruption {
            detail: "bad block".into(),
        };
        let rekha_err: RekhaError = storage_err.clone().into();
        assert!(rekha_err.source().is_some());
    }

    #[test]
    fn test_into_conversions() {
        let s = StorageError::Write {
            source: "disk full".into(),
        };
        let e: RekhaError = s.into();
        assert!(matches!(e, RekhaError::Storage(_)));

        let i = IndexError::EmptyIndex;
        let e: RekhaError = i.into();
        assert!(matches!(e, RekhaError::Index(_)));

        let p = PartitionError::ShardNotFound { shard_id: 5 };
        let e: RekhaError = p.into();
        assert!(matches!(e, RekhaError::Partition(_)));

        let r = RaftError::ElectionTimeout;
        let e: RekhaError = r.into();
        assert!(matches!(e, RekhaError::Consensus(_)));
    }

    #[test]
    fn test_raft_error_no_leader_hint() {
        let err = RaftError::NotLeader { leader_hint: None };
        assert_eq!(err.to_string(), "not leader, leader unknown");
    }

    #[test]
    fn test_rekha_error_cluster_changed() {
        let err = RekhaError::ClusterChanged {
            detail: "new node joined".into(),
        };
        assert_eq!(
            err.to_string(),
            "cluster membership changed: new node joined"
        );
    }

    #[test]
    fn test_rekha_error_consensus() {
        let err = RekhaError::Consensus(RaftError::ElectionTimeout);
        assert_eq!(err.to_string(), "consensus error: election timed out");
    }

    #[test]
    fn test_rekha_error_wrapped_storage() {
        let err = RekhaError::Storage(StorageError::Corruption {
            detail: "bad block".into(),
        });
        assert_eq!(
            err.to_string(),
            "storage error: corruption detected: bad block"
        );
    }

    #[test]
    fn test_rekha_error_wrapped_index() {
        let err = RekhaError::Index(IndexError::NotTrained { component: "PQ" });
        assert_eq!(err.to_string(), "index error: PQ has not been trained");
    }

    #[test]
    fn test_rekha_error_wrapped_partition() {
        let err = RekhaError::Partition(PartitionError::ShardNotFound { shard_id: 3 });
        assert_eq!(err.to_string(), "partition error: shard 3 not found");
    }

    #[test]
    fn test_storage_error_column_family() {
        let err = StorageError::ColumnFamily {
            name: "vectors".into(),
            source: "handle not found".into(),
        };
        assert_eq!(err.to_string(), "column family vectors: handle not found");
    }

    #[test]
    fn test_storage_error_read() {
        let err = StorageError::Read {
            key: vec![0, 0, 0, 0, 0, 0, 0, 42],
            source: "IO error".into(),
        };
        assert_eq!(
            err.to_string(),
            "read error at key [0, 0, 0, 0, 0, 0, 0, 42]: IO error"
        );
    }

    #[test]
    fn test_storage_error_write() {
        let err = StorageError::Write {
            source: "disk full".into(),
        };
        assert_eq!(err.to_string(), "write error: disk full");
    }

    #[test]
    fn test_storage_error_batch_write() {
        let err = StorageError::BatchWrite {
            committed: 5,
            failed: 2,
            source: "timeout".into(),
        };
        assert_eq!(
            err.to_string(),
            "batch write: committed 5, failed 2: timeout"
        );
    }

    #[test]
    fn test_storage_error_corruption() {
        let err = StorageError::Corruption {
            detail: "checksum mismatch".into(),
        };
        assert_eq!(err.to_string(), "corruption detected: checksum mismatch");
    }

    #[test]
    fn test_storage_error_serialization() {
        let err = StorageError::Serialization {
            detail: "invalid encoding".into(),
        };
        assert_eq!(err.to_string(), "serialization error: invalid encoding");
    }

    #[test]
    fn test_index_error_graph_build() {
        let err = IndexError::GraphBuild {
            detail: "out of memory".into(),
        };
        assert_eq!(err.to_string(), "graph build failed: out of memory");
    }

    #[test]
    fn test_index_error_search() {
        let err = IndexError::Search {
            detail: "no results".into(),
        };
        assert_eq!(err.to_string(), "search failed: no results");
    }

    #[test]
    fn test_index_error_not_trained() {
        let err = IndexError::NotTrained {
            component: "ProductQuantizer",
        };
        assert_eq!(err.to_string(), "ProductQuantizer has not been trained");
    }

    #[test]
    fn test_index_error_incompatible_dim() {
        let err = IndexError::IncompatibleDimension {
            expected: 768,
            actual: 64,
        };
        assert_eq!(err.to_string(), "expected dimension 768, got 64");
    }

    #[test]
    fn test_partition_error_invalid_topology() {
        let err = PartitionError::InvalidTopology {
            detail: "overlapping ranges".into(),
        };
        assert_eq!(err.to_string(), "invalid topology: overlapping ranges");
    }

    #[test]
    fn test_partition_error_rebalance() {
        let err = PartitionError::RebalanceInProgress { partition_id: 7 };
        assert_eq!(err.to_string(), "rebalance in progress for partition 7");
    }

    #[test]
    fn test_partition_error_dim_group_mismatch() {
        let err = PartitionError::DimensionGroupMismatch {
            expected: 4,
            actual: 2,
        };
        assert_eq!(
            err.to_string(),
            "dimension group mismatch: expected 4, got 2"
        );
    }

    #[test]
    fn test_partition_error_shard_not_found() {
        let err = PartitionError::ShardNotFound { shard_id: 5 };
        assert_eq!(err.to_string(), "shard 5 not found");
    }

    #[test]
    fn test_raft_error_log_compaction() {
        let err = RaftError::LogCompaction {
            detail: "snapshot too large".into(),
        };
        assert_eq!(err.to_string(), "log compaction: snapshot too large");
    }

    #[test]
    fn test_raft_error_snapshot_failed() {
        let err = RaftError::SnapshotFailed {
            detail: "disk full".into(),
        };
        assert_eq!(err.to_string(), "snapshot failed: disk full");
    }

    #[test]
    fn test_raft_error_membership_change() {
        let err = RaftError::MembershipChange {
            detail: "node already exists".into(),
        };
        assert_eq!(err.to_string(), "membership change: node already exists");
    }

    #[test]
    fn test_raft_error_commit_failed() {
        let err = RaftError::CommitFailed { index: 42, term: 5 };
        assert_eq!(err.to_string(), "commit failed at index 42 term 5");
    }
}
