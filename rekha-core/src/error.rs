use std::fmt;

/// Top-level error for the Rekha distributed vector database.
/// Every error in the system ultimately flows through this type.
#[derive(Debug, Clone)]
pub enum RekhaError {
    NotFound(String),
    InvalidArgument(String),
    IndexFull { capacity: usize, attempted: usize },
    InvalidDimension { expected: usize, actual: usize },
    Storage(StorageError),
    Index(IndexError),
    Partition(PartitionError),
    Consensus(RaftError),
    Timeout { operation: &'static str, elapsed_ms: u64 },
    ClusterChanged { detail: String },
    Unavailable { detail: String },
    Internal { detail: String },
}

impl fmt::Display for RekhaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(s) => write!(f, "not found: {s}"),
            Self::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
            Self::IndexFull { capacity, attempted } => {
                write!(f, "index full: capacity={capacity}, attempted={attempted}")
            }
            Self::InvalidDimension { expected, actual } => {
                write!(f, "invalid dimension: expected {expected}, got {actual}")
            }
            Self::Storage(e) => write!(f, "storage error: {e}"),
            Self::Index(e) => write!(f, "index error: {e}"),
            Self::Partition(e) => write!(f, "partition error: {e}"),
            Self::Consensus(e) => write!(f, "consensus error: {e}"),
            Self::Timeout { operation, elapsed_ms } => {
                write!(f, "timeout on {operation} after {elapsed_ms}ms")
            }
            Self::ClusterChanged { detail } => write!(f, "cluster membership changed: {detail}"),
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
    fn from(e: StorageError) -> Self { Self::Storage(e) }
}
impl From<IndexError> for RekhaError {
    fn from(e: IndexError) -> Self { Self::Index(e) }
}
impl From<PartitionError> for RekhaError {
    fn from(e: PartitionError) -> Self { Self::Partition(e) }
}
impl From<RaftError> for RekhaError {
    fn from(e: RaftError) -> Self { Self::Consensus(e) }
}

/// Storage-layer errors (RocksDB, serialization, etc.)
#[derive(Debug, Clone)]
pub enum StorageError {
    DbOpen { path: String, source: String },
    ColumnFamily { name: String, source: String },
    Read { key: Vec<u8>, source: String },
    Write { source: String },
    BatchWrite { committed: usize, failed: usize, source: String },
    Corruption { detail: String },
    Serialization { detail: String },
    PayloadTooLarge { size: usize, max: usize },
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DbOpen { path, source } => write!(f, "failed to open db at {path}: {source}"),
            Self::ColumnFamily { name, source } => write!(f, "column family {name}: {source}"),
            Self::Read { key, source } => write!(f, "read error at key {key:?}: {source}"),
            Self::Write { source } => write!(f, "write error: {source}"),
            Self::BatchWrite { committed, failed, source } => {
                write!(f, "batch write: committed {committed}, failed {failed}: {source}")
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
                write!(f, "dimension group mismatch: expected {expected}, got {actual}")
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
        let err = StorageError::DbOpen { path: "/data/db".into(), source: "permission denied".into() };
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
    fn test_raft_error_display() {
        let err = RaftError::NotLeader { leader_hint: Some("node-2".into()) };
        assert_eq!(err.to_string(), "not leader, try node-2");

        let err = RaftError::ElectionTimeout;
        assert_eq!(err.to_string(), "election timed out");
    }

    #[test]
    fn test_error_source() {
        let storage_err = StorageError::Corruption { detail: "bad block".into() };
        let rekha_err: RekhaError = storage_err.clone().into();
        assert!(rekha_err.source().is_some());
    }

    #[test]
    fn test_into_conversions() {
        let s = StorageError::Write { source: "disk full".into() };
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
}
