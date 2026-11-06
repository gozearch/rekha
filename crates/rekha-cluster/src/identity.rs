//! Durable node identity and cluster peer topology.
//!
//! Leadership is meaningless if identities move. Two rules are enforced here,
//! and both exist because their absence caused real bugs:
//!
//! 1. **A node id belongs to a data directory.** It is written once, on first
//!    start, to `node.id`, and startup refuses if `--node-id` later disagrees.
//!    Without this, a node whose disk is replaced (or whose data directory is
//!    copied) can come back campaigning under an id another node already holds.
//! 2. **Peer ids are data, not position.** `--peers` is `id@addr`, because
//!    deriving ids from list position means inserting a node anywhere but the
//!    end silently renumbers the cluster.
//!
//! Nothing in this module knows about leadership. It only fixes *who* the
//! participants are; *which one leads* comes from consensus, always.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Name of the file holding the durable node id.
pub const NODE_ID_FILE: &str = "node.id";

/// The identity of this node, as recorded on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeIdentity {
    node_id: u64,
}

impl NodeIdentity {
    /// The node id recorded in this data directory.
    pub fn node_id(&self) -> u64 {
        self.node_id
    }

    /// Read the identity recorded in `data_dir`, if any.
    pub fn read(data_dir: &Path) -> Result<Option<Self>, IdentityError> {
        let path = data_dir.join(NODE_ID_FILE);
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(IdentityError::Io(e)),
        };
        let trimmed = raw.trim();
        let node_id: u64 = trimmed.parse().map_err(|_| IdentityError::Malformed {
            path: path.display().to_string(),
            detail: format!("expected a decimal node id, found {trimmed:?}"),
        })?;
        if node_id == 0 {
            return Err(IdentityError::Malformed {
                path: path.display().to_string(),
                detail: "node id 0 is reserved as \"unknown\"".into(),
            });
        }
        Ok(Some(Self { node_id }))
    }

    /// Open the identity for `data_dir`, creating it on first start.
    ///
    /// Returns an error rather than a warning when the recorded id disagrees
    /// with `requested`: starting anyway would give the cluster two nodes
    /// claiming one identity, which is a split-brain with no recovery path.
    pub fn open_or_create(data_dir: &Path, requested: u64) -> Result<Self, IdentityError> {
        if requested == 0 {
            return Err(IdentityError::ZeroRequested);
        }
        match Self::read(data_dir)? {
            Some(existing) if existing.node_id != requested => Err(IdentityError::Mismatch {
                path: data_dir.join(NODE_ID_FILE).display().to_string(),
                existing: existing.node_id,
                requested,
            }),
            Some(existing) => Ok(existing),
            None => {
                let identity = Self { node_id: requested };
                identity.persist(data_dir)?;
                Ok(identity)
            }
        }
    }

    /// Write the id durably: temp file, fsync, rename, fsync directory.
    fn persist(&self, data_dir: &Path) -> Result<(), IdentityError> {
        std::fs::create_dir_all(data_dir)?;
        let final_path = data_dir.join(NODE_ID_FILE);
        let tmp_path = data_dir.join(format!("{NODE_ID_FILE}.tmp"));

        // A stale temp file from an interrupted first start is not authoritative.
        {
            let mut file = std::fs::File::create(&tmp_path)?;
            writeln!(file, "{}", self.node_id)?;
            file.sync_all()?;
        }
        std::fs::rename(&tmp_path, &final_path)?;
        sync_dir(data_dir)?;
        Ok(())
    }
}

/// fsync a directory so a rename survives a crash.
fn sync_dir(dir: &Path) -> std::io::Result<()> {
    // Opening a directory read-only and syncing it is the portable-enough way
    // to flush the directory entry on Unix. On platforms where this is a
    // no-op the file itself is already synced.
    let handle = std::fs::File::open(dir)?;
    handle.sync_all()
}

/// One member of the cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Peer {
    pub node_id: u64,
    pub addr: String,
}

/// Parse `id@addr` pairs, for example `1@10.0.0.1:8000,2@10.0.0.2:8000`.
///
/// Self may appear in the list; the returned order is the input order.
pub fn parse_peers(input: &str) -> Result<Vec<Peer>, PeerParseError> {
    let input = input.trim();
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut peers = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for raw in input.split(',') {
        let entry = raw.trim();
        if entry.is_empty() {
            return Err(PeerParseError::EmptyEntry);
        }
        let (id_part, addr_part) =
            entry
                .split_once('@')
                .ok_or_else(|| PeerParseError::MissingId {
                    entry: entry.to_string(),
                })?;
        let node_id: u64 = id_part.trim().parse().map_err(|_| PeerParseError::BadId {
            entry: entry.to_string(),
        })?;
        if node_id == 0 {
            return Err(PeerParseError::ZeroId {
                entry: entry.to_string(),
            });
        }
        let addr = addr_part.trim();
        if addr.is_empty() {
            return Err(PeerParseError::EmptyAddr {
                entry: entry.to_string(),
            });
        }
        if !seen.insert(node_id) {
            return Err(PeerParseError::DuplicateId { node_id });
        }
        peers.push(Peer {
            node_id,
            addr: addr.to_string(),
        });
    }
    Ok(peers)
}

/// Node-id → address book, with no opinion about leadership.
///
/// Shared rather than copied. Every component that needs to reach a node — the
/// Raft network factory, request forwarding, the state machine applying a
/// membership entry — holds the same directory, so learning an address in one
/// place teaches it everywhere. Per-component copies are what made adding a node
/// impossible without restarting the whole cluster with a new `--peers`: the
/// leader could not reach a node the flag did not mention, so there was nothing
/// to add.
#[derive(Debug, Clone, Default)]
pub struct PeerDirectory {
    peers: std::sync::Arc<parking_lot::RwLock<BTreeMap<u64, String>>>,
}

impl PeerDirectory {
    /// Build a directory from configured peers.
    pub fn new(peers: &[Peer]) -> Self {
        let directory = Self::default();
        directory.replace_all(peers.iter().map(|p| (p.node_id, p.addr.clone())));
        directory
    }

    /// Every known node id, ascending.
    pub fn ids(&self) -> Vec<u64> {
        self.peers.read().keys().copied().collect()
    }

    /// The configured address of `node_id`.
    pub fn addr(&self, node_id: u64) -> Option<String> {
        self.peers.read().get(&node_id).cloned()
    }

    /// The base URL of `node_id`, with a scheme applied.
    pub fn url(&self, node_id: u64) -> Option<String> {
        self.addr(node_id)
            .map(|addr| crate::network::normalize_peer_url(&addr))
    }

    pub fn contains(&self, node_id: u64) -> bool {
        self.peers.read().contains_key(&node_id)
    }

    pub fn len(&self) -> usize {
        self.peers.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.read().is_empty()
    }

    /// Record where `node_id` lives. Returns whether this changed the directory.
    pub fn insert(&self, node_id: u64, addr: impl Into<String>) -> bool {
        let addr = addr.into();
        let mut peers = self.peers.write();
        if peers.get(&node_id).is_some_and(|current| *current == addr) {
            return false;
        }
        peers.insert(node_id, addr);
        true
    }

    /// Forget `node_id`, returning the address it had.
    pub fn remove(&self, node_id: u64) -> Option<String> {
        self.peers.write().remove(&node_id)
    }

    /// Replace the whole directory.
    ///
    /// Used when a snapshot is installed: the snapshot's node set is the
    /// authority, so an address this node learned from a log entry that the
    /// snapshot supersedes must go, not linger.
    pub fn replace_all(&self, peers: impl IntoIterator<Item = (u64, String)>) {
        let replacement: BTreeMap<u64, String> = peers.into_iter().collect();
        *self.peers.write() = replacement;
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    #[error(
        "data directory already belongs to node {existing}, but --node-id is {requested} ({path}). \
         Two nodes sharing an identity is an unrecoverable split brain. \
         Start with --node-id {existing}, or point at a different --data-dir."
    )]
    Mismatch {
        path: String,
        existing: u64,
        requested: u64,
    },

    #[error(
        "node identity file {path} is malformed: {detail}. \
         Refusing to guess an identity: repair the file or use a fresh --data-dir."
    )]
    Malformed { path: String, detail: String },

    #[error("node id 0 is reserved as \"unknown\"; pick an id from 1 upward")]
    ZeroRequested,

    #[error("node identity io: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PeerParseError {
    #[error(
        "peer entry `{entry}` is missing its `id@` prefix. Peers are `id@host:port` \
         (for example `1@10.0.0.1:8000`). Positional ids were removed: deriving them \
         from list position renumbered the whole cluster whenever a node was inserted."
    )]
    MissingId { entry: String },

    #[error("peer entry `{entry}` has a non-numeric node id")]
    BadId { entry: String },

    #[error("peer entry `{entry}` addresses node 0, which is reserved as \"unknown\"")]
    ZeroId { entry: String },

    #[error("peer entry `{entry}` has an empty address")]
    EmptyAddr { entry: String },

    #[error("node id {node_id} appears more than once in --peers")]
    DuplicateId { node_id: u64 },

    #[error("--peers contains an empty entry; remove the stray comma")]
    EmptyEntry,
}

/// Where a data directory keeps its identity file.
pub fn node_id_path(data_dir: &Path) -> PathBuf {
    data_dir.join(NODE_ID_FILE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    #[test]
    fn identity_is_created_on_first_start_and_reused_after() {
        let dir = tmp();
        let first = NodeIdentity::open_or_create(dir.path(), 2).unwrap();
        assert_eq!(first.node_id(), 2);

        // A restart with the same id reuses it.
        let second = NodeIdentity::open_or_create(dir.path(), 2).unwrap();
        assert_eq!(second.node_id(), 2);

        // It is actually on disk, not just in memory.
        assert_eq!(
            std::fs::read_to_string(node_id_path(dir.path()))
                .unwrap()
                .trim(),
            "2"
        );
    }

    #[test]
    fn identity_mismatch_refuses_to_start() {
        let dir = tmp();
        NodeIdentity::open_or_create(dir.path(), 2).unwrap();

        let err = NodeIdentity::open_or_create(dir.path(), 3).unwrap_err();
        let msg = err.to_string();
        assert!(
            matches!(
                err,
                IdentityError::Mismatch {
                    existing: 2,
                    requested: 3,
                    ..
                }
            ),
            "expected Mismatch, got {err:?}"
        );
        assert!(
            msg.contains("split brain"),
            "message must name the risk: {msg}"
        );
        assert!(
            msg.contains("--node-id 2"),
            "message must name the fix: {msg}"
        );
    }

    #[test]
    fn malformed_identity_file_is_not_silently_overwritten() {
        let dir = tmp();
        std::fs::write(node_id_path(dir.path()), "not-a-number\n").unwrap();

        let err = NodeIdentity::open_or_create(dir.path(), 1).unwrap_err();
        assert!(matches!(err, IdentityError::Malformed { .. }));
        // The bad file must survive: guessing would erase the evidence.
        assert_eq!(
            std::fs::read_to_string(node_id_path(dir.path()))
                .unwrap()
                .trim(),
            "not-a-number"
        );
    }

    #[test]
    fn interrupted_first_start_leaves_no_authoritative_temp_file() {
        let dir = tmp();
        // Simulate a crash between create() and rename().
        std::fs::write(dir.path().join(format!("{NODE_ID_FILE}.tmp")), "9\n").unwrap();

        let identity = NodeIdentity::open_or_create(dir.path(), 1).unwrap();
        assert_eq!(identity.node_id(), 1, "a stale temp file must not win");
    }

    #[test]
    fn zero_is_rejected_both_on_disk_and_on_the_command_line() {
        let dir = tmp();
        assert!(matches!(
            NodeIdentity::open_or_create(dir.path(), 0).unwrap_err(),
            IdentityError::ZeroRequested
        ));

        std::fs::write(node_id_path(dir.path()), "0\n").unwrap();
        assert!(matches!(
            NodeIdentity::read(dir.path()).unwrap_err(),
            IdentityError::Malformed { .. }
        ));
    }

    #[test]
    fn peers_parse_id_at_addr() {
        let peers = parse_peers("1@10.0.0.1:8000, 2@10.0.0.2:8000 ,3@[::1]:8000").unwrap();
        assert_eq!(peers.len(), 3);
        assert_eq!(
            peers[0],
            Peer {
                node_id: 1,
                addr: "10.0.0.1:8000".into()
            }
        );
        assert_eq!(
            peers[1],
            Peer {
                node_id: 2,
                addr: "10.0.0.2:8000".into()
            }
        );
        assert_eq!(
            peers[2],
            Peer {
                node_id: 3,
                addr: "[::1]:8000".into()
            }
        );
    }

    #[test]
    fn peers_empty_is_single_node_mode() {
        assert!(parse_peers("").unwrap().is_empty());
        assert!(parse_peers("   ").unwrap().is_empty());
    }

    /// The old form silently renumbered the cluster; it must now be an error
    /// with a message that says what to write instead.
    #[test]
    fn positional_peer_form_is_rejected_with_guidance() {
        let err = parse_peers("host1:8000,host2:8001").unwrap_err();
        assert!(matches!(err, PeerParseError::MissingId { .. }));
        let msg = err.to_string();
        assert!(msg.contains("id@host:port"), "{msg}");
        assert!(msg.contains("renumbered"), "{msg}");
    }

    #[test]
    fn peers_reject_duplicates_zeros_and_empty_addresses() {
        assert!(matches!(
            parse_peers("1@a:1,1@b:2").unwrap_err(),
            PeerParseError::DuplicateId { node_id: 1 }
        ));
        assert!(matches!(
            parse_peers("0@a:1").unwrap_err(),
            PeerParseError::ZeroId { .. }
        ));
        assert!(matches!(
            parse_peers("1@").unwrap_err(),
            PeerParseError::EmptyAddr { .. }
        ));
        assert!(matches!(
            parse_peers("1@a:1,,").unwrap_err(),
            PeerParseError::EmptyEntry
        ));
        assert!(matches!(
            parse_peers("x@a:1").unwrap_err(),
            PeerParseError::BadId { .. }
        ));
    }

    #[test]
    fn directory_resolves_urls() {
        let peers = parse_peers("1@10.0.0.1:8000,2@10.0.0.2:8000").unwrap();
        let dir = PeerDirectory::new(&peers);

        assert_eq!(dir.url(2).as_deref(), Some("http://10.0.0.2:8000"));
        assert_eq!(dir.url(9), None);
        assert_eq!(dir.ids(), vec![1, 2]);
    }

    #[test]
    fn directory_keeps_an_explicit_scheme() {
        let peers = parse_peers("1@https://10.0.0.1:8443").unwrap();
        let dir = PeerDirectory::new(&peers);
        assert_eq!(dir.url(1).as_deref(), Some("https://10.0.0.1:8443"));
    }
}
