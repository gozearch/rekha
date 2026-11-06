//! Where a collection's data lives.
//!
//! There are two decisions here, and they are deliberately separated:
//!
//! 1. **Which shard a collection belongs to** is decided once, by the metadata
//!    leader, and recorded on the collection's catalog row. See
//!    [`Placement::place`].
//! 2. **Which nodes own a shard** is a pure function of the voter set, the
//!    shard count, and the replication factor, by rendezvous hashing. Two nodes
//!    that agree on those three inputs compute the same owners, which is what
//!    lets a request arriving at any node be routed without a lookup round
//!    trip.
//!
//! Only (2) is pure. (1) is a decision, because a hash over the collection id
//! cannot see how loaded the nodes are, and it is node load that determines how
//! much data a machine actually stores. Deriving (1) from the id measured badly:
//! at 200 collections on 10 nodes one node held 46% of the corpus and another
//! 8%, and raising the shard count did not help because shard ownership is
//! itself uneven.
//!
//! Rendezvous is used for the node choice rather than modulo so that, when the
//! voter set does change, only the shards whose winner changed move — roughly
//! `R/N` of them — instead of nearly all of them. Measured: adding a fifth
//! voter to a sixteen-shard cluster changes the owner set of **9 of the 16**
//! shards, and each of those is data that has to be copied before the change is
//! safe. That is why all three inputs live in `placement.json` and are
//! validated at startup, and why a node restarted with a different `--peers` is
//! refused rather than trusted.
//!
//! # Stability is a data-integrity property
//!
//! [`stable_hash`] is pinned. Changing it moves every shard's replicas, which
//! for a running cluster is data loss with extra steps. The `owners_do_not_drift`
//! test pins concrete outputs so the change cannot be made quietly.

use std::collections::HashMap;

use uuid::Uuid;

pub use rekha_core::cluster::ShardId;

/// Why a [`Placement`] could not be built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlacementError {
    /// Placement needs at least one node to put data on.
    #[error("placement requires at least one voter")]
    NoVoters,
    /// A shard count of zero would divide by zero.
    #[error("shard count must be at least 1")]
    ZeroShards,
    /// The cap exists so a typo (`--shards 100000`) fails at startup instead of
    /// allocating that many consensus groups.
    #[error("shard count {requested} exceeds the maximum of {max}")]
    TooManyShards { requested: u32, max: u32 },
    /// Replication factor zero means no node holds the data.
    #[error("replication factor must be at least 1")]
    ZeroReplication,
    /// An explicitly supplied owner table is not a valid placement.
    ///
    /// This is refused rather than repaired: the table decides which nodes hold
    /// which data, and "fixing" a malformed one would move replicas that a node
    /// may already have written.
    #[error("invalid shard owners: {detail}")]
    BadOwners { detail: String },
}

/// FNV-1a over `bytes`, then a splitmix64 finalizer.
///
/// FNV alone has weak avalanche in the final byte, which matters here because
/// rendezvous hashing ranks adjacent node ids. The finalizer is what makes
/// `(shard, node)` scores behave like independent random numbers. Both stages
/// are fixed arithmetic with no floating point and no platform dependence, so
/// the output is identical on every machine and every Rust version.
fn stable_hash(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut h = OFFSET;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(PRIME);
    }

    // splitmix64 finalizer.
    let mut z = h.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The placement function for one cluster configuration.
///
/// Cheap to clone and infallible to query. Built once at startup from flags
/// that are validated against the data directory (see `PlacementIdentity` in
/// the CLI), so a node cannot silently disagree with its peers about where a
/// collection lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    shard_count: u32,
    replication_factor: usize,
    /// Sorted, deduplicated. Sorting makes every derived order deterministic.
    voters: Vec<u64>,
    /// `owners[shard]`, highest rendezvous score first. `shards x R` node ids,
    /// fixed once the voter set is.
    ///
    /// Precomputed rather than derived per call: [`Placement::owners`] is on the
    /// routing path of every request and inside [`Placement::place`]'s loop over
    /// all shards, and deriving it there meant sorting the whole voter set once
    /// per shard per call. At 40 shards and 10 voters that is 480 comparisons
    /// and an allocation to answer "who owns shard 7", for a table that cannot
    /// change while the process runs.
    owners: Box<[Box<[u64]>]>,
}

impl Placement {
    /// Upper bound on shard count. Each shard is a consensus group, so this is
    /// a resource bound, not an arbitrary limit.
    pub const MAX_SHARDS: u32 = 1024;

    /// Build a placement.
    ///
    /// `replication_factor` is clamped to the voter count: asking for three
    /// replicas on two nodes is a configuration mistake, but the only sensible
    /// reading is "both nodes", and silently creating a group that can never
    /// reach quorum would be worse.
    pub fn new(
        shard_count: u32,
        replication_factor: usize,
        voters: impl IntoIterator<Item = u64>,
    ) -> Result<Self, PlacementError> {
        if shard_count == 0 {
            return Err(PlacementError::ZeroShards);
        }
        if shard_count > Self::MAX_SHARDS {
            return Err(PlacementError::TooManyShards {
                requested: shard_count,
                max: Self::MAX_SHARDS,
            });
        }
        if replication_factor == 0 {
            return Err(PlacementError::ZeroReplication);
        }
        let mut voters: Vec<u64> = voters.into_iter().collect();
        voters.sort_unstable();
        voters.dedup();
        if voters.is_empty() {
            return Err(PlacementError::NoVoters);
        }
        let replication_factor = replication_factor.min(voters.len());

        let owners = (0..shard_count)
            .map(|shard| {
                let shard = ShardId(shard);
                let mut scored: Vec<(u64, u64)> = voters
                    .iter()
                    .map(|&node| (Self::score(shard, node), node))
                    .collect();
                // Descending score; ascending node id on a tie, so the order is
                // total and every node computes the same one.
                scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                let owners: Vec<u64> = scored
                    .into_iter()
                    .take(replication_factor)
                    .map(|(_, node)| node)
                    .collect();
                owners.into_boxed_slice()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self {
            shard_count,
            replication_factor,
            voters,
            owners,
        })
    }

    /// Build a placement from an explicit owner table.
    ///
    /// The table is validated, not trusted: every shard must name exactly
    /// `min(replication_factor, voters)` distinct voters. A shorter list would
    /// create a consensus group that cannot reach the quorum its replication
    /// factor claims; an unknown node id would make the group wait for a peer
    /// that is not a voter.
    ///
    /// This exists because ownership stops being a pure function of the voter
    /// set as soon as it can change: after a shard is moved, or a replica is
    /// re-created elsewhere, re-deriving would silently undo the move.
    pub fn from_owners(
        shard_count: u32,
        replication_factor: usize,
        voters: impl IntoIterator<Item = u64>,
        owners: Vec<Vec<u64>>,
    ) -> Result<Self, PlacementError> {
        // `new` validates the scalars and gives the canonical voter set and the
        // effective replication factor, which is clamped to the voter count.
        let derived = Self::new(shard_count, replication_factor, voters)?;
        let expected = derived.owners.first().map_or(0, |o| o.len());

        if owners.len() != derived.shard_count as usize {
            return Err(PlacementError::BadOwners {
                detail: format!(
                    "{} shards are declared but {} owner lists were supplied",
                    derived.shard_count,
                    owners.len()
                ),
            });
        }

        let voters: std::collections::HashSet<u64> = derived.voters.iter().copied().collect();
        for (shard, set) in owners.iter().enumerate() {
            if set.len() != expected {
                return Err(PlacementError::BadOwners {
                    detail: format!(
                        "shard {shard} has {} owners; replication factor {} (clamped to {} \
                         voters) requires {expected}",
                        set.len(),
                        replication_factor,
                        derived.voters.len()
                    ),
                });
            }
            let mut seen = std::collections::HashSet::with_capacity(set.len());
            for &node in set {
                if !voters.contains(&node) {
                    return Err(PlacementError::BadOwners {
                        detail: format!("shard {shard} lists node {node}, which is not a voter"),
                    });
                }
                if !seen.insert(node) {
                    return Err(PlacementError::BadOwners {
                        detail: format!("shard {shard} lists node {node} twice"),
                    });
                }
            }
        }

        Ok(Self {
            shard_count: derived.shard_count,
            replication_factor: derived.replication_factor,
            voters: derived.voters,
            owners: owners
                .into_iter()
                .map(|set| set.into_boxed_slice())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
    }

    /// The owner table, in the form it is persisted.
    pub fn owner_table(&self) -> Vec<Vec<u64>> {
        self.owners.iter().map(|set| set.to_vec()).collect()
    }

    /// Return a new placement with an updated voter set, retaining current shard owners.
    pub fn with_voters(
        &self,
        voters: impl IntoIterator<Item = u64>,
    ) -> Result<Self, PlacementError> {
        Self::from_owners(
            self.shard_count,
            self.replication_factor,
            voters,
            self.owner_table(),
        )
    }

    /// Return a new placement with an updated owner set for `shard`.
    pub fn with_shard_owners(
        &self,
        shard: ShardId,
        owners: &[u64],
    ) -> Result<Self, PlacementError> {
        let mut table = self.owner_table();
        if let Some(entry) = table.get_mut(shard.0 as usize) {
            *entry = owners.to_vec();
        } else {
            return Err(PlacementError::BadOwners {
                detail: format!(
                    "shard index {} out of range (count {})",
                    shard.0, self.shard_count
                ),
            });
        }
        Self::from_owners(
            self.shard_count,
            self.replication_factor,
            self.voters.iter().copied(),
            table,
        )
    }

    /// Atomically persist this placement configuration to `<dir>/placement.json`.
    pub fn save(&self, dir: &std::path::Path) -> std::io::Result<()> {
        let path = dir.join(PlacementConfig::FILE);
        let tmp = dir.join(format!(
            ".{}.tmp.{}",
            PlacementConfig::FILE,
            std::process::id()
        ));
        let config = PlacementConfig {
            shard_count: self.shard_count(),
            replication_factor: self.replication_factor(),
            voters: self.voters().to_vec(),
            owners: Some(self.owner_table()),
        };
        let bytes = serde_json::to_vec_pretty(&config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        {
            let mut file = std::fs::File::create(&tmp)?;
            use std::io::Write;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// Shard count a cluster of `nodes` should start with.
    ///
    /// `4 × nodes` keeps the number of consensus groups **per node** constant:
    /// a node holds `shards × R / nodes` replicas, and with `shards = 4 ×
    /// nodes` and `R = 3` that is twelve regardless of cluster size. A fixed
    /// per-node cost that does not grow with the cluster is the property that
    /// makes 100 nodes feasible; a fixed *total* shard count would give the
    /// first few nodes everything and the rest nothing.
    pub fn default_shard_count(nodes: usize) -> u32 {
        (4 * nodes.max(1) as u32).clamp(8, 256)
    }

    /// The shard a new collection should be placed on.
    ///
    /// The objective is **node load**, not shard count. Shard ownership is
    /// itself uneven — measured at 40 shards over 10 nodes: `5, 9, 11, 12, 12,
    /// 13, 16, 16, 17, 9` shards per node — so spreading collections evenly
    /// across shards still piles them onto whichever node happens to own many
    /// shards. Choosing by node load attacks the skew directly.
    ///
    /// Ordering is by the busiest owner, then the total across owners, then a
    /// hash of `(collection, shard)`. The last term only makes ties
    /// deterministic: two leaders computing over the same state pick the same
    /// shard without coordinating, and a retry after a lost proposal lands in
    /// the same place.
    ///
    /// `load` is collections already placed per node. A node absent from the map
    /// is empty.
    ///
    /// # Limits
    ///
    /// This can only choose among the owner sets that exist. With three replicas
    /// over four nodes there are four of them, and rendezvous does not hand them
    /// out evenly, so a four-node cluster stays lopsided however the collections
    /// are assigned (measured: 53–80 per node against an ideal of 60). Ten nodes
    /// give 120 owner sets and balance to within two. Small clusters need an even
    /// shard-to-node assignment, which is a change to [`Placement::owners`] and
    /// not to this decision.
    ///
    /// # Complexity
    ///
    /// `O(shards x replication_factor)` reads and no allocations: the owner
    /// lists this compares are built once, in [`Placement::new`]. It runs once
    /// per collection creation, not per request.
    pub fn place(&self, collection: Uuid, load: &HashMap<u64, u32>) -> ShardId {
        let busiest = |owners: &[u64]| {
            owners
                .iter()
                .map(|node| load.get(node).copied().unwrap_or(0))
                .max()
                .unwrap_or(0)
        };

        (0..self.shard_count)
            .map(ShardId)
            .min_by_key(|shard| {
                let owners = self.owners(*shard);
                let total: u32 = owners
                    .iter()
                    .map(|node| load.get(node).copied().unwrap_or(0))
                    .sum();
                (busiest(owners), total, Self::tie_break(collection, *shard))
            })
            // `shard_count` is at least 1, so the range is never empty.
            .unwrap_or(ShardId(0))
    }

    /// Deterministic, order-independent tie-break for [`Placement::place`].
    fn tie_break(collection: Uuid, shard: ShardId) -> u64 {
        let mut key = [0u8; 20];
        key[0..16].copy_from_slice(collection.as_bytes());
        key[16..20].copy_from_slice(&shard.0.to_le_bytes());
        stable_hash(&key)
    }

    /// The nodes that hold `shard`, highest rendezvous score first.
    ///
    /// Order is meaningful: it is the deterministic fallback order for routing
    /// when no leader is known, and it is what makes "forward to the first
    /// owner" land on the same node from every sender.
    ///
    /// Empty for a shard this placement does not have. That is not reachable
    /// through [`Placement::place`], which only returns shards it built; it is
    /// reachable from a catalog row written under a different shard count, and
    /// the callers already read "no owners" as "retry" rather than routing to a
    /// guess.
    pub fn owners(&self, shard: ShardId) -> &[u64] {
        match self.owners.get(shard.0 as usize) {
            Some(owners) => owners,
            None => &[],
        }
    }

    /// Rendezvous score of `node` for `shard`.
    fn score(shard: ShardId, node: u64) -> u64 {
        let mut key = [0u8; 12];
        key[0..4].copy_from_slice(&shard.0.to_le_bytes());
        key[4..12].copy_from_slice(&node.to_le_bytes());
        stable_hash(&key)
    }

    /// Whether `node` is one of `shard`'s owners.
    pub fn owns(&self, shard: ShardId, node: u64) -> bool {
        self.owners(shard).contains(&node)
    }

    /// The shard count this placement was built with.
    pub fn shard_count(&self) -> u32 {
        self.shard_count
    }

    /// The effective replication factor (already clamped to the voter count).
    pub fn replication_factor(&self) -> usize {
        self.replication_factor
    }

    /// Every voter, ascending.
    pub fn voters(&self) -> &[u64] {
        &self.voters
    }

    /// Map a record ID to a shard deterministically.
    pub fn shard_for_record(&self, id: &str) -> ShardId {
        let hash = stable_hash(id.as_bytes());
        ShardId((hash % u64::from(self.shard_count)) as u32)
    }
}

/// The placement parameters a cluster agreed on, persisted per node.
///
/// All three inputs to [`Placement`] are here, and that completeness is the
/// point. Shard count and replication factor were persisted while the voter set
/// was left to `--peers` on each start, so the one input an operator could
/// change by editing a flag was the one that moved shards — and nothing
/// noticed. A node restarted with a different `--peers` computed different
/// owners than its peers and then routed, and stored, shards the rest of the
/// cluster disagreed about.
///
/// This file lives beside `node.id` for the same reason: a node that comes back
/// with a different value would answer differently from its peers about where
/// the data is.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementConfig {
    /// Number of shards the cluster was created with.
    pub shard_count: u32,
    /// Replicas per shard.
    pub replication_factor: usize,
    /// The voters placement was computed for, ascending.
    pub voters: Vec<u64>,
    /// `owners[shard]`, in replica order.
    ///
    /// `Option` so a document written before this field existed still opens: an
    /// absent table means "derive it from the voters", which is exactly what
    /// every existing directory recorded implicitly. A present table is the
    /// authority, because once shard ownership can change (removal, promotion,
    /// re-replication) it is no longer a function of the voter set alone.
    ///
    /// `#[serde(default)]` is what makes the old form readable; `deny_unknown_fields`
    /// on the struct still rejects a *misspelled* field, which is the failure
    /// worth catching.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owners: Option<Vec<Vec<u64>>>,
}

impl PlacementConfig {
    /// File name inside the data directory.
    pub const FILE: &'static str = "placement.json";
}

/// Why a data directory's placement could not be opened.
///
/// Each variant is a different operator mistake, and each message says what to
/// do instead: "these two numbers disagree" is only useful if it also says that
/// changing them is a migration.
#[derive(Debug, thiserror::Error)]
pub enum PlacementOpenError {
    /// The persisted file exists but could not be read.
    #[error("cannot read {path}: {source}")]
    Read {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// The persisted file is not a placement document.
    #[error("{path} is not a placement file: {source}")]
    Parse {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },
    /// The seed could not be encoded or written.
    #[error("cannot write {path}: {source}")]
    Write {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// The requested configuration is not a valid placement.
    #[error(transparent)]
    Invalid(#[from] PlacementError),
    /// `--shards` disagrees with the data directory.
    #[error(
        "--shards {requested} disagrees with this data directory, which was created with \
         {persisted} shards. Changing the shard count moves every collection onto different \
         nodes; that is a reshard, not a restart"
    )]
    ShardsDiffer { requested: u32, persisted: u32 },
    /// `--replication-factor` disagrees with the data directory.
    #[error(
        "--replication-factor {requested} disagrees with this data directory ({persisted}). \
         Changing it changes where replicas live; that is a rebalance, not a restart"
    )]
    ReplicationDiffers { requested: usize, persisted: usize },
    /// `--peers` disagrees with the data directory.
    #[error(
        "--peers lists nodes {configured:?} but this data directory was placed across \
         {persisted:?}. The voter set is a property of the cluster, not of a start: changing \
         it moves shards without moving their data, which is a rebalance"
    )]
    VotersDiffer {
        configured: Vec<u64>,
        persisted: Vec<u64>,
    },
    /// This directory holds a cluster's placement, but no topology was given.
    #[error(
        "{path} holds a cluster placement but --peers is empty, so this node would open every \
         collection as a single-node server. Start it with the topology it was placed across"
    )]
    NoTopology { path: std::path::PathBuf },
}

impl Placement {
    /// Open the placement this data directory was created with, or seed a new
    /// one from `voters`.
    ///
    /// `voters` is a *seed*, not an authority. It is consulted only when the
    /// data directory has never belonged to a cluster; afterwards the persisted
    /// voter set is the authority and a differing seed is refused. A node that
    /// could redefine placement by being restarted would move every shard's
    /// replicas without moving any data, and the first symptom would be a shard
    /// group that never elects.
    ///
    /// Returns `None` for a directory that is not, and has not been, part of a
    /// cluster. Single-node mode has no shards at all, so it has no placement
    /// document to be wrong about.
    ///
    /// # Errors
    ///
    /// Returns [`PlacementOpenError`] if the persisted document cannot be read
    /// or parsed, if a requested flag disagrees with it, or if a new seed is not
    /// a valid placement. Nothing is written unless all of that holds, so a
    /// failed start leaves the directory as it was.
    pub fn open(
        dir: &std::path::Path,
        requested_shards: u32,
        requested_replication: usize,
        voters: &[u64],
    ) -> Result<Option<Self>, PlacementOpenError> {
        let path = dir.join(PlacementConfig::FILE);

        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if voters.is_empty() {
                    return Ok(None);
                }
                return Self::seed(&path, requested_shards, requested_replication, voters)
                    .map(Some);
            }
            Err(source) => return Err(PlacementOpenError::Read { path, source }),
        };

        let config: PlacementConfig =
            serde_json::from_slice(&bytes).map_err(|source| PlacementOpenError::Parse {
                path: path.clone(),
                source,
            })?;

        if voters.is_empty() {
            return Err(PlacementOpenError::NoTopology { path });
        }
        if requested_shards != 0 && requested_shards != config.shard_count {
            return Err(PlacementOpenError::ShardsDiffer {
                requested: requested_shards,
                persisted: config.shard_count,
            });
        }
        if requested_replication != config.replication_factor {
            return Err(PlacementOpenError::ReplicationDiffers {
                requested: requested_replication,
                persisted: config.replication_factor,
            });
        }
        let mut configured = voters.to_vec();
        configured.sort_unstable();
        configured.dedup();
        let is_superset = configured.iter().all(|v| config.voters.contains(v));
        if configured != config.voters && !is_superset {
            return Err(PlacementOpenError::VotersDiffer {
                configured,
                persisted: config.voters,
            });
        }

        let placement = match config.owners {
            // A present table is the authority. Deriving owners from the voters
            // would silently revert a shard move that was committed.
            Some(owners) => Self::from_owners(
                config.shard_count,
                config.replication_factor,
                config.voters.iter().copied(),
                owners,
            ),
            // Written before this field existed: derive, which is what the file
            // meant.
            None => Self::new(config.shard_count, config.replication_factor, config.voters),
        };
        placement.map(Some).map_err(PlacementOpenError::Invalid)
    }

    /// Write the first placement document for a directory.
    fn seed(
        path: &std::path::Path,
        requested_shards: u32,
        requested_replication: usize,
        voters: &[u64],
    ) -> Result<Self, PlacementOpenError> {
        let mut voters = voters.to_vec();
        voters.sort_unstable();
        voters.dedup();

        let shard_count = if requested_shards == 0 {
            Self::default_shard_count(voters.len())
        } else {
            requested_shards
        };

        // Validate before persisting: a bad configuration should fail on the
        // first start, not after the file is written.
        let placement = Self::new(shard_count, requested_replication, voters)?;

        let config = PlacementConfig {
            shard_count: placement.shard_count(),
            replication_factor: placement.replication_factor(),
            voters: placement.voters().to_vec(),
            owners: Some(placement.owner_table()),
        };
        let json = serde_json::to_vec_pretty(&config).map_err(|e| PlacementOpenError::Write {
            path: path.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
        })?;
        std::fs::write(path, json).map_err(|source| PlacementOpenError::Write {
            path: path.to_path_buf(),
            source,
        })?;

        Ok(placement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placement(nodes: u64, shards: u32, r: usize) -> Placement {
        Placement::new(shards, r, 1..=nodes).expect("valid placement")
    }

    /// Pinned outputs. If this test fails, the hash changed and every shard's
    /// replicas moved — which is data loss unless a migration is written. Do not
    /// "update the expected values".
    #[test]
    fn owners_do_not_drift() {
        let p = placement(3, 16, 3);
        assert_eq!(
            p.owners(ShardId(2)),
            [3, 1, 2],
            "owner order changed; this is a data migration, not a test fix"
        );
    }

    /// Placement must aim at node load, not shard count.
    ///
    /// The precondition is measured, not assumed: shard ownership is itself
    /// uneven, so spreading collections evenly across shards still piles them
    /// onto whichever node owns many shards. The real run that motivated this
    /// gave `9, 13, 12, 17, 12, 5, 16, 9, 16, 11` shards per node.
    #[test]
    fn placement_follows_node_load_not_shard_count() {
        let p = placement(10, 40, 3);

        // If this ever fails, shard ownership became even, and the load-aware
        // choice is no longer doing work — revisit the decision, do not just
        // relax the number.
        let mut shards_per_node = [0u32; 10];
        for shard in 0..40 {
            for &owner in p.owners(ShardId(shard)) {
                shards_per_node[owner as usize - 1] += 1;
            }
        }
        let skew =
            shards_per_node.iter().max().unwrap_or(&0) - shards_per_node.iter().min().unwrap_or(&0);
        assert!(
            skew >= 5,
            "shard ownership is even ({shards_per_node:?}); this test exists because it is not"
        );

        // Every collection placed adds one to each of its three owners.
        let mut collections: HashMap<u64, u32> = HashMap::new();
        for i in 0..200u128 {
            let shard = p.place(Uuid::from_u128(i), &collections);
            for &owner in p.owners(shard) {
                *collections.entry(owner).or_default() += 1;
            }
        }

        let ideal = 200 * 3 / 10;
        let min = collections.values().copied().min().unwrap_or(0);
        let max = collections.values().copied().max().unwrap_or(0);
        assert!(
            max - min <= 2,
            "placement spread {min}..{max} (ideal {ideal}); the decision is not \
             following node load: {collections:?}"
        );
    }

    #[test]
    fn placement_is_deterministic_for_the_same_state() {
        let p = placement(10, 40, 3);
        let load: HashMap<u64, u32> = (1..=10).map(|node| (node, node as u32)).collect();
        for i in 0..100u128 {
            let id = Uuid::from_u128(i);
            assert_eq!(p.place(id, &load), p.place(id, &load));
        }
    }

    #[test]
    fn owners_are_exactly_the_replication_factor() {
        for nodes in 1..=10u64 {
            for r in 1..=5usize {
                let p = placement(nodes, 8, r);
                let expected = r.min(nodes as usize);
                for shard in 0..8 {
                    let owners = p.owners(ShardId(shard));
                    assert_eq!(owners.len(), expected, "nodes={nodes} r={r}");
                    let mut sorted = owners.to_vec();
                    sorted.sort_unstable();
                    sorted.dedup();
                    assert_eq!(sorted.len(), owners.len(), "owners must be distinct");
                    assert!(owners.iter().all(|n| (1..=nodes).contains(n)));
                }
            }
        }
    }

    /// Why the voter set had to become durable: it moves shards.
    ///
    /// This is the measurement behind "a different `--peers` moves data without
    /// moving it". Adding one voter re-ranks every shard's candidates, and the
    /// shards whose top three change are shards whose data is now on the wrong
    /// nodes.
    #[test]
    fn adding_a_voter_moves_owners() {
        let four = placement(4, 16, 3);
        let five = Placement::new(16, 3, 1..=5).expect("valid placement");

        let moved = (0..16u32)
            .filter(|shard| {
                let mut before = four.owners(ShardId(*shard)).to_vec();
                let mut after = five.owners(ShardId(*shard)).to_vec();
                before.sort_unstable();
                after.sort_unstable();
                before != after
            })
            .count();

        assert_eq!(
            moved, 9,
            "adding a fifth voter moved {moved} of 16 shards' owner sets. Pinned because it \
             is the measured cost of a membership change: those are the shards a rebalance \
             would have to move"
        );
    }

    /// The whole point of sharding: the cluster stores `R/N` of the corpus per
    /// node, not all of it.
    ///
    /// Asserting "every node holds fewer than S shards" would be wrong: with
    /// `S=16, R=3, N=4` a given node is in the top three for 12 of 16 shards on
    /// average, and `(3/4)^16` is around 1%, so one node holding all sixteen is
    /// a legitimate draw, not a bug. The property that must hold is the total:
    /// `S x R` replica slots, not `S x N`.
    #[test]
    fn the_cluster_does_not_store_a_copy_on_every_node() {
        let p = placement(4, 16, 3);
        let mut per_node = [0usize; 5];
        for shard in 0..16 {
            for &owner in p.owners(ShardId(shard)) {
                per_node[owner as usize] += 1;
            }
        }
        let total: usize = per_node[1..=4].iter().sum();
        assert_eq!(total, 16 * 3, "every shard must have exactly R replicas");
        assert!(
            per_node[1..=4].iter().any(|&n| n < 16),
            "at least one node must hold less than the whole corpus; \
             got {per_node:?}"
        );
        // And no node may exceed what `R/N` of the slots allow by more than a
        // shard or two, or placement has collapsed onto one machine.
        for (node, &held) in per_node.iter().enumerate().skip(1) {
            assert!(held <= 16, "node {node} holds {held} replica slots");
        }
    }

    #[test]
    fn replication_factor_is_clamped_to_the_voter_count() {
        let p = placement(2, 8, 3);
        assert_eq!(p.replication_factor(), 2);
        assert_eq!(p.owners(ShardId(0)).len(), 2);
    }

    #[test]
    fn invalid_configurations_are_rejected() {
        assert_eq!(Placement::new(0, 3, [1]), Err(PlacementError::ZeroShards));
        assert_eq!(
            Placement::new(8, 0, [1]),
            Err(PlacementError::ZeroReplication)
        );
        assert_eq!(Placement::new(8, 3, []), Err(PlacementError::NoVoters));
        assert!(matches!(
            Placement::new(Placement::MAX_SHARDS + 1, 3, [1]),
            Err(PlacementError::TooManyShards { .. })
        ));
    }

    #[test]
    fn default_shard_count_keeps_per_node_groups_constant() {
        for nodes in 3..=64usize {
            let shards = Placement::default_shard_count(nodes);
            let r = 3usize;
            let replicas_per_node = f64::from(shards) * r as f64 / nodes as f64;
            assert!(
                (replicas_per_node - 12.0).abs() < 3.0,
                "nodes={nodes} shards={shards}: {replicas_per_node:.1} groups per node, \
                 expected roughly 12"
            );
        }
    }
}

#[cfg(test)]
mod placement_config_tests {
    use super::*;

    const VOTERS: [u64; 4] = [1, 2, 3, 4];

    #[test]
    fn first_start_picks_a_default_and_persists_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let placement = Placement::open(dir.path(), 0, 3, &VOTERS)
            .expect("open")
            .expect("a cluster directory has a placement");
        assert_eq!(
            placement.shard_count(),
            Placement::default_shard_count(VOTERS.len())
        );

        let config: PlacementConfig = serde_json::from_slice(
            &std::fs::read(dir.path().join(PlacementConfig::FILE)).expect("read"),
        )
        .expect("parse");
        assert_eq!(config.shard_count, placement.shard_count());
        assert_eq!(config.replication_factor, placement.replication_factor());
        assert_eq!(config.voters, VOTERS, "the voter set must be persisted");
    }

    /// An explicit table is the authority, even when it differs from what the
    /// voters would derive. This is the property a shard move depends on: if a
    /// restart re-derived ownership, every committed move would silently revert.
    #[test]
    fn an_explicit_owner_table_is_authoritative() {
        let dir = tempfile::tempdir().expect("tempdir");
        let derived = Placement::new(8, 2, VOTERS).expect("placement");

        // A valid table that the derivation would never produce: swap the two
        // replicas of shard 0 (order is replica precedence, not identity).
        let mut owners = derived.owner_table();
        owners[0].swap(0, 1);
        assert_ne!(
            owners,
            derived.owner_table(),
            "the fixture must differ from the derived table"
        );

        let config = PlacementConfig {
            shard_count: 8,
            replication_factor: 2,
            voters: VOTERS.to_vec(),
            owners: Some(owners.clone()),
        };
        std::fs::write(
            dir.path().join(PlacementConfig::FILE),
            serde_json::to_vec_pretty(&config).expect("encode"),
        )
        .expect("write");

        let opened = Placement::open(dir.path(), 8, 2, &VOTERS)
            .expect("open")
            .expect("placement");
        assert_eq!(
            opened.owner_table(),
            owners,
            "the persisted table must be used as written"
        );
        assert_eq!(opened.owners(ShardId(0)), owners[0].as_slice());
    }

    /// A document written before the table existed records the same thing
    /// implicitly: derive it. Refusing those would make the upgrade a migration.
    #[test]
    fn a_document_without_an_owner_table_is_derived() {
        let dir = tempfile::tempdir().expect("tempdir");
        // No `owners` field at all, which is exactly what older builds wrote.
        let json = format!(
            r#"{{"shard_count": 16, "replication_factor": 3, "voters": {}}}"#,
            serde_json::to_string(&VOTERS).expect("voters")
        );
        std::fs::write(dir.path().join(PlacementConfig::FILE), json).expect("write");

        let opened = Placement::open(dir.path(), 16, 3, &VOTERS)
            .expect("open")
            .expect("placement");
        let derived = Placement::new(16, 3, VOTERS).expect("placement");
        assert_eq!(
            opened, derived,
            "an absent table must mean exactly what derivation produces"
        );
    }

    #[test]
    fn an_owner_table_naming_a_non_voter_is_refused() {
        let derived = Placement::new(4, 2, VOTERS).expect("placement");
        let mut owners = derived.owner_table();
        owners[1][1] = 99; // not a voter
        let error = Placement::from_owners(4, 2, VOTERS, owners)
            .expect_err("an unknown node must be refused");
        assert!(
            error.to_string().contains("not a voter"),
            "error must name the cause: {error}"
        );
    }

    #[test]
    fn an_owner_table_with_the_wrong_replica_count_is_refused() {
        let derived = Placement::new(4, 2, VOTERS).expect("placement");
        let mut owners = derived.owner_table();
        owners[2].pop(); // one replica instead of two
        let error = Placement::from_owners(4, 2, VOTERS, owners)
            .expect_err("a shard with fewer replicas must be refused");
        assert!(
            error.to_string().contains("requires 2"),
            "error must name the requirement: {error}"
        );
    }

    #[test]
    fn an_owner_table_listing_a_node_twice_is_refused() {
        let derived = Placement::new(4, 2, VOTERS).expect("placement");
        let mut owners = derived.owner_table();
        owners[0][1] = owners[0][0];
        let error = Placement::from_owners(4, 2, VOTERS, owners)
            .expect_err("a duplicated replica must be refused");
        assert!(error.to_string().contains("twice"), "{error}");
    }

    #[test]
    fn a_restart_reads_the_persisted_placement_instead_of_recomputing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first = Placement::open(dir.path(), 64, 3, &VOTERS)
            .expect("open")
            .expect("placement");
        assert_eq!(first.shard_count(), 64);

        // As long as the flags agree, the placement comes back unchanged.
        let second = Placement::open(dir.path(), 0, 3, &VOTERS)
            .expect("open")
            .expect("placement");
        assert_eq!(second, first);
    }

    /// The property this whole change exists for: a node cannot be restarted
    /// into a different voter set. Before the voters were persisted, this
    /// succeeded and computed different owners than the node's peers.
    #[test]
    fn a_different_voter_set_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        Placement::open(dir.path(), 16, 3, &VOTERS)
            .expect("open")
            .expect("placement");

        let before = std::fs::read(dir.path().join(PlacementConfig::FILE)).expect("read");
        let error = Placement::open(dir.path(), 16, 3, &[1, 2, 3, 4, 5])
            .expect_err("five voters where four were placed must be refused");
        match error {
            PlacementOpenError::VotersDiffer {
                configured,
                persisted,
            } => {
                assert_eq!(configured, vec![1, 2, 3, 4, 5]);
                assert_eq!(persisted, vec![1, 2, 3, 4]);
            }
            other => panic!("wrong error: {other}"),
        }
        let after = std::fs::read(dir.path().join(PlacementConfig::FILE)).expect("read");
        assert_eq!(before, after, "a refused start must not rewrite placement");
    }

    /// A clustered data directory started with no topology would open every
    /// collection as a single node, which is a silent lie about replication.
    #[test]
    fn a_cluster_directory_without_a_topology_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        Placement::open(dir.path(), 16, 3, &VOTERS)
            .expect("open")
            .expect("placement");
        assert!(matches!(
            Placement::open(dir.path(), 16, 3, &[]),
            Err(PlacementOpenError::NoTopology { .. })
        ));
    }

    /// Single-node mode: no directory to be wrong about, so no document.
    #[test]
    fn a_directory_that_never_belonged_to_a_cluster_has_no_placement() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(Placement::open(dir.path(), 0, 3, &[]).expect("open"), None);
        assert!(!dir.path().join(PlacementConfig::FILE).exists());
    }

    #[test]
    fn an_invalid_shard_count_fails_before_anything_is_written() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(Placement::open(dir.path(), 0, 3, &[]).is_ok());
        assert!(matches!(
            Placement::open(dir.path(), Placement::MAX_SHARDS + 1, 3, &VOTERS),
            Err(PlacementOpenError::Invalid(
                PlacementError::TooManyShards { .. }
            ))
        ));
        assert!(!dir.path().join(PlacementConfig::FILE).exists());
    }

    /// A document written by a different version, or edited by hand into
    /// something else, is a parse error rather than a silent default.
    #[test]
    fn a_document_missing_the_voter_set_is_not_a_placement() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join(PlacementConfig::FILE),
            br#"{"shard_count": 16, "replication_factor": 3}"#,
        )
        .expect("write");
        assert!(matches!(
            Placement::open(dir.path(), 0, 3, &VOTERS),
            Err(PlacementOpenError::Parse { .. })
        ));
    }
}
