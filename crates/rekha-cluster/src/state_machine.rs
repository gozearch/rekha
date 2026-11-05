//! Cluster metadata state machine — applies Raft-committed entries.

use std::collections::HashMap;
use std::sync::Arc;

use openraft::Entry;
use openraft::EntryPayload;
use openraft::LogId;
use openraft::Snapshot;
use openraft::SnapshotMeta;
use openraft::StorageError;
use openraft::StoredMembership;
use openraft::storage::RaftSnapshotBuilder;
use openraft::storage::RaftStateMachine;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::cluster::ClockMap;
use crate::placement::ShardId;
use crate::raft_types::{BatchResponse, ClusterOperation, NodeInfo, RaftTypeConfig};
use rekha_storage::Catalog;

/// Trait for engine snapshot/restore, avoiding circular dependency with rekha-engine.
///
/// The snapshot is an *archive*: the engine frames its own collection files
/// around an opaque `state` blob the cluster layer supplies. That keeps the
/// engine from knowing about Raft and keeps this crate from knowing how vectors
/// are stored.
pub trait EngineSnapshotProvider: Send + Sync + 'static {
    /// Encode a snapshot archive containing `state` plus the data of exactly
    /// `collections`.
    ///
    /// The scope is explicit because a snapshot belongs to one consensus group:
    /// a shard group must not ship another shard's data (that would materialize
    /// it on the receiver, defeating placement), and the metadata group holds no
    /// vectors at all.
    ///
    /// Every collection in scope must already have a published generation;
    /// callers run [`EngineSnapshotProvider::checkpoint_all`] first so the
    /// snapshot covers the boundary the log is about to be purged to.
    fn snapshot_bytes(&self, state: &[u8], collections: &[uuid::Uuid]) -> Result<Vec<u8>, String>;

    /// Replace local state from an archive, returning the embedded state blob.
    ///
    /// Only the collections the archive contains are replaced; a collection
    /// outside it is left alone, so restoring one shard cannot wipe another.
    fn restore_bytes(&self, archive: &[u8]) -> Result<Vec<u8>, String>;

    /// Checkpoint every collection in scope, so a snapshot is never behind its
    /// boundary.
    fn checkpoint_all(&self, collections: &[uuid::Uuid]) -> Result<usize, String>;

    /// Record a collection's metadata on this node without opening its data.
    ///
    /// Idempotent. Every node records every collection so it can authorize and
    /// route requests for one it does not hold.
    fn record_collection(
        &self,
        config: &rekha_core::config::CollectionConfig,
        shard: Option<ShardId>,
    ) -> Result<(), String>;

    /// Open a recorded collection's data on this node. Idempotent.
    fn materialize_collection(&self, collection_id: &uuid::Uuid) -> Result<(), String>;

    /// Remove a collection the log says is gone. Idempotent.
    fn drop_collection(&self, collection_id: &uuid::Uuid) -> Result<(), String>;

    /// Apply a committed data batch at `index`.
    ///
    /// Must be total: a committed entry cannot be rejected. Everything that can
    /// be refused (dimension, duplicate ids, batch size) is validated by the
    /// leader before the entry is proposed. An error here means this node has
    /// diverged or hit a bug, and the caller marks the node not-ready.
    fn apply_batch(
        &self,
        collection_id: &uuid::Uuid,
        ops: &[rekha_core::op::Operation],
        index: u64,
        dimension: usize,
    ) -> Result<(), String>;
}

/// Which consensus group this state machine belongs to.
///
/// The role changes what an `AddCollection` entry does and what a snapshot
/// contains, so it is recorded rather than inferred from "has an engine":
/// every group has an engine handle, but only shard groups own vectors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupRole {
    /// The metadata group: registry, RBAC, placement. Holds no vectors.
    #[default]
    Metadata,
    /// A shard group: holds the data of the collections placed on it.
    Shard(ShardId),
}

/// Whether a catalog error means "the request itself could never succeed".
///
/// These are deterministic verdicts: the same entry applied to the same state
/// yields the same error on every replica, so they are not evidence that this
/// node diverged from the log. Infrastructure failures are, and are handled
/// separately by the callers.
fn is_application_rejection(e: &rekha_storage::CatalogError) -> bool {
    matches!(
        e,
        rekha_storage::CatalogError::NotFound | rekha_storage::CatalogError::InvalidArgument(_)
    )
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct ClusterStateMachine {
    /// This node's id, used to decide whether a collection is materialized here.
    #[serde(default)]
    pub node_id: u64,
    /// What kind of group this is. See [`GroupRole`].
    #[serde(default)]
    pub role: GroupRole,
    pub nodes: HashMap<u64, NodeInfo>,
    /// Shard placement, as decided by the metadata leader at creation.
    ///
    /// Log-derived and replicated: it is what a shard group uses to decide whose
    /// data belongs in its snapshots, and what a node recovers after a restart
    /// without recomputing anything. The durable copy a request is routed by is
    /// the collection's catalog row; this map and that row are written from the
    /// same entry, so they cannot disagree about where a collection went.
    ///
    /// Not derived from the id: the leader chose the shard for node load, and a
    /// hash cannot reproduce a decision.
    pub placements: HashMap<Uuid, ShardId>,
    pub clock_map: ClockMap,
    pub last_applied: Option<LogId<u64>>,
    pub membership: StoredMembership<u64, u64>,
    #[serde(skip)]
    pub engine: Option<Arc<dyn EngineSnapshotProvider>>,
    #[serde(skip)]
    pub catalog: Option<Arc<dyn Catalog>>,
    /// Where the nodes in `nodes` can be reached, for the components that have
    /// to *talk* to them: the Raft network factory and request forwarding.
    ///
    /// `nodes` is the durable copy that travels in a snapshot; this is the live
    /// one every sender reads. They are written from the same entry, so they
    /// cannot disagree — the same pattern as `placements` and the collection
    /// rows.
    #[serde(skip)]
    pub addresses: Option<crate::identity::PeerDirectory>,
    /// Set when a committed entry could not be applied. A node in this state
    /// must stop answering: continuing would serve a state that disagrees with
    /// the log it claims to have.
    #[serde(skip)]
    pub apply_failure: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// Set once this node has the cluster's bootstrap credentials: the
    /// `BootstrapSuperadmin` entry has applied, or a snapshot carrying users
    /// has been installed.
    ///
    /// `/ready` gates on it. Without the gate, a node can report ready between
    /// "the user and key rows exist" and "the role assignment exists", and a
    /// client that waited for readiness still gets a 403 — measured as a
    /// flaking cluster test, not a hypothetical.
    #[serde(skip)]
    pub bootstrap_ready: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl std::fmt::Debug for ClusterStateMachine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusterStateMachine")
            .field("node_id", &self.node_id)
            .field("role", &self.role)
            .field("nodes", &self.nodes)
            .field("placements", &self.placements)
            .field("clock_map", &self.clock_map)
            .field("last_applied", &self.last_applied)
            .field("membership", &self.membership)
            .field("engine", &self.engine.as_ref().map(|_| "..."))
            .field("catalog", &self.catalog.as_ref().map(|_| "..."))
            .finish()
    }
}

impl ClusterStateMachine {
    pub fn new() -> Self {
        Self::default()
    }

    /// A state machine for a shard group on `node_id`.
    pub fn for_shard(node_id: u64, shard: ShardId) -> Self {
        Self {
            node_id,
            role: GroupRole::Shard(shard),
            ..Default::default()
        }
    }

    /// A state machine for the metadata group on `node_id`.
    pub fn for_metadata(node_id: u64) -> Self {
        Self {
            node_id,
            role: GroupRole::Metadata,
            ..Default::default()
        }
    }

    /// The collections whose data belongs in this group's snapshots.
    ///
    /// Empty for the metadata group: its state is the registry, and archiving
    /// vectors there would ship data to nodes that are not replicas of it.
    fn snapshot_scope(&self) -> Vec<Uuid> {
        match self.role {
            GroupRole::Metadata => Vec::new(),
            GroupRole::Shard(_) => self.placements.keys().copied().collect(),
        }
    }

    pub fn set_catalog(&mut self, catalog: Arc<dyn Catalog>) {
        self.catalog = Some(catalog);
    }

    /// Attach the engine used to produce and restore Raft snapshots.
    ///
    /// Without it `build_snapshot` embeds an empty engine payload, so a
    /// follower that joins after log compaction installs a snapshot with no
    /// records at all and reports a successful catch-up.
    pub fn set_engine(&mut self, engine: Arc<dyn EngineSnapshotProvider>) {
        self.engine = Some(engine);
    }

    /// Share the live address book, so a membership entry that commits can be
    /// used to reach the node it is about.
    pub fn set_addresses(&mut self, addresses: crate::identity::PeerDirectory) {
        self.addresses = Some(addresses);
    }

    /// Share the flag this state machine raises if a committed entry cannot be
    /// applied, so the HTTP layer can stop reporting readiness.
    pub fn set_apply_failure_flag(&mut self, flag: Arc<std::sync::atomic::AtomicBool>) {
        self.apply_failure = Some(flag);
    }

    /// Share the flag that means "the cluster's bootstrap credentials are
    /// applied here". `/ready` holds the node back until it is set.
    pub fn set_bootstrap_ready_flag(&mut self, flag: Arc<std::sync::atomic::AtomicBool>) {
        self.bootstrap_ready = Some(flag);
    }

    /// Resume from the boundary a purged log implies.
    ///
    /// openraft replays `[last_applied, committed]` at startup. A state machine
    /// that reports `last_applied = None` therefore asks for the whole log,
    /// including the prefix a snapshot already purged — which openraft then
    /// refuses (`expected index [0, 32), got [20, 31)`) and the node fails to
    /// start.
    ///
    /// The purge point *is* the snapshot boundary: openraft only purges up to
    /// `snapshot.last_log_id - max_in_snapshot_log_to_keep`. Resuming there
    /// says "everything at or below this has already been applied and is
    /// covered by durable state", which is what makes the remaining tail
    /// sufficient.
    ///
    /// The membership is seeded from the placement document rather than the
    /// log because the membership entry lives in the purged prefix. Callers
    /// must pass the voter set the placement says this group has; that
    /// document is the authority a starting node is already validated against.
    pub fn resume_from(
        &mut self,
        last_applied: Option<LogId<u64>>,
        membership: StoredMembership<u64, u64>,
    ) {
        self.last_applied = last_applied;
        self.membership = membership;
    }

    /// Rebuild `placements` from the catalog.
    ///
    /// The map is normally reconstructed by replaying `AddCollection` entries.
    /// After a purge those entries are gone, and an empty map is not benign: a
    /// shard group derives its snapshot scope from it, so the next snapshot
    /// would archive no collections at all and the following purge would delete
    /// the only remaining copy of the data's history.
    ///
    /// The catalog row is written by the same entry that fills this map, so it
    /// is an authoritative source rather than a guess.
    pub fn seed_placements(&mut self, catalog: &dyn Catalog) -> Result<(), String> {
        let rows = catalog
            .list_collections_all()
            .map_err(|e| format!("cannot rebuild placement from the catalog: {e}"))?;
        for row in rows {
            let Some(shard) = row.shard else { continue };
            // A shard group must seed only its own collections: the snapshot
            // scope is derived from this map, and checkpointing a collection
            // this node does not host would fail the snapshot build.
            let belongs = match self.role {
                GroupRole::Metadata => true,
                GroupRole::Shard(own) => shard == own,
            };
            if belongs {
                self.placements.insert(row.config.id, shard);
            }
        }
        Ok(())
    }

    /// Rebuild the address book this machine carries in snapshots.
    ///
    /// `nodes` is what a snapshot ships so a joining node can reach the rest of
    /// the group; `addresses` is the live copy used by the network factory.
    /// After a purge the `AddNode` entries are gone, so the seed is the peer
    /// list the process was started with.
    pub fn seed_nodes(&mut self, peers: impl IntoIterator<Item = (u64, String)>) {
        for (node_id, addr) in peers {
            self.nodes.insert(node_id, NodeInfo { node_id, addr });
        }
    }

    fn mark_bootstrap_ready(&self) {
        if let Some(flag) = &self.bootstrap_ready {
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Whether a committed entry has failed to apply on this node.
    pub fn apply_failed(&self) -> bool {
        self.apply_failure
            .as_ref()
            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::SeqCst))
    }

    fn note_apply_failure(&self) {
        if let Some(flag) = &self.apply_failure {
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Apply a catalog write that a committed entry asked for.
    ///
    /// Two kinds of failure reach here and they must not be conflated:
    ///
    /// * **The application refused the write** (`NotFound`, `InvalidArgument`).
    ///   This is deterministic — every replica has the same state and reaches
    ///   the same verdict — so it is a bad request that got committed, not
    ///   divergence. It is logged, and the node keeps serving: a replica that
    ///   agreed with the rest of the cluster is not a replica that lost data.
    /// * **The store failed** (`Storage`, `Serialization`). This node did not
    ///   take an entry the log committed, so it has diverged and must stop
    ///   answering.
    ///
    /// The previous version treated only `AlreadyExists` as expected, so a
    /// `POST /auth/users/{id}/keys` for a user that does not exist made every
    /// node report `ready:false` until someone restarted the cluster.
    fn apply_write(&self, what: &str, result: rekha_storage::CatalogResult<()>) {
        match result {
            Ok(()) | Err(rekha_storage::CatalogError::AlreadyExists) => {}
            Err(e) if is_application_rejection(&e) => {
                // `AlreadyExists` is the expected replay; the other two mean the
                // request could never have succeeded on any node. The leader
                // rejects these before proposing, so reaching here means a race
                // with a concurrent change or a replay of an older entry.
                eprintln!("cluster: {what} was rejected: {e}");
            }
            Err(e) => {
                eprintln!("cluster: {what} did not apply: {e}");
                self.note_apply_failure();
            }
        }
    }

    /// The same, for an entry that removes a row so a replayed removal is not a
    /// divergence. `InvalidArgument` is included for operations that refuse to
    /// remove something built in: that verdict is identical on every node.
    fn apply_removal(&self, what: &str, result: rekha_storage::CatalogResult<()>) {
        match result {
            Ok(()) | Err(rekha_storage::CatalogError::NotFound) => {}
            Err(e) if is_application_rejection(&e) => {
                eprintln!("cluster: {what} was rejected: {e}");
            }
            Err(e) => {
                eprintln!("cluster: {what} did not apply: {e}");
                self.note_apply_failure();
            }
        }
    }

    pub fn apply_operation(&mut self, op: &ClusterOperation) {
        match op {
            ClusterOperation::AddNode { node_id, addr } => {
                self.nodes.insert(
                    *node_id,
                    NodeInfo {
                        node_id: *node_id,
                        addr: addr.clone(),
                    },
                );
                // The live directory too, or a learner that just joined is still
                // unreachable and its replication never starts.
                if let Some(addresses) = &self.addresses {
                    addresses.insert(*node_id, addr.clone());
                }
            }
            ClusterOperation::RemoveNode { node_id } => {
                self.nodes.remove(node_id);
                if let Some(addresses) = &self.addresses {
                    addresses.remove(*node_id);
                }
            }
            ClusterOperation::AddCollection { config, shard } => {
                if let Some(shard) = shard {
                    self.placements.insert(config.id, *shard);
                }
                if let Some(engine) = &self.engine {
                    // Every node records the row: authorization and routing have
                    // to answer for a collection this node does not hold, and a
                    // 404 there would be a lie.
                    if let Err(e) = engine.record_collection(config, *shard) {
                        eprintln!(
                            "cluster: failed to record replicated collection {}: {e}",
                            config.id
                        );
                    }
                    // The group that owns the shard opens the data. Comparing
                    // the entry's owners against this node's id was the wrong
                    // test: a node is a replica of some shards and not others,
                    // and the group it applied the entry in already says which.
                    let owned = match (self.role, *shard) {
                        (GroupRole::Shard(hosted), Some(shard)) => hosted == shard,
                        // No shards at all: the single group holds everything.
                        (GroupRole::Shard(_), None) => true,
                        (GroupRole::Metadata, _) => false,
                    };
                    if owned && let Err(e) = engine.materialize_collection(&config.id) {
                        eprintln!(
                            "cluster: failed to materialize replicated collection {}: {e}",
                            config.id
                        );
                    }
                }
            }
            ClusterOperation::Batch { .. } => {
                // Data batches are applied by `apply()` because they need the
                // entry's log index as the dataset sequence. Reaching here means
                // a call path dropped the sequence, which would make the apply
                // non-idempotent and the snapshot boundary meaningless.
                eprintln!(
                    "cluster: a Batch operation reached apply_operation without a log index; \
                     refusing to apply it blind"
                );
                self.note_apply_failure();
            }
            ClusterOperation::CreateTenant { name } => {
                if let Some(catalog) = &self.catalog
                    && let Err(e) = catalog.create_tenant(name)
                    && !matches!(e, rekha_storage::CatalogError::AlreadyExists)
                {
                    // A duplicate is the expected replay; anything else means the
                    // node's catalog did not take a committed entry, and serving
                    // from it would report a namespace that half the cluster
                    // cannot see.
                    eprintln!("cluster: failed to create tenant {name}: {e}");
                    self.note_apply_failure();
                }
            }
            ClusterOperation::CreateDatabase { tenant, name } => {
                if let Some(catalog) = &self.catalog
                    && let Err(e) = catalog.create_database(tenant, name)
                    && !matches!(e, rekha_storage::CatalogError::AlreadyExists)
                {
                    eprintln!("cluster: failed to create database {tenant}/{name}: {e}");
                    self.note_apply_failure();
                }
            }
            ClusterOperation::RemoveCollection { collection_id } => {
                self.placements.remove(collection_id);
                if let Some(engine) = &self.engine
                    && let Err(e) = engine.drop_collection(collection_id)
                {
                    eprintln!("cluster: failed to drop replicated collection {collection_id}: {e}");
                }
            }
            ClusterOperation::CreateUser {
                user,
                key_hash,
                key_label,
            } => {
                if let Some(ref catalog) = self.catalog {
                    self.apply_write("create user", catalog.create_user(user));
                    self.apply_write(
                        "create api key",
                        catalog.create_api_key(&user.id, key_hash, key_label),
                    );
                }
            }
            ClusterOperation::BootstrapSuperadmin {
                user,
                key_hash,
                key_label,
                role_id,
            } => {
                // One entry, not three. Creating the user and then granting the
                // role in separate entries leaves a window in which an
                // authenticated user holds no permissions, and every request in
                // that window is a 403 — measured, not hypothetical.
                if let Some(ref catalog) = self.catalog {
                    self.apply_write("create bootstrap user", catalog.create_user(user));
                    self.apply_write(
                        "create bootstrap key",
                        catalog.create_api_key(&user.id, key_hash, key_label),
                    );
                    self.apply_write(
                        "grant bootstrap role",
                        catalog.assign_role(&user.id, role_id, &user.id),
                    );
                }
                self.mark_bootstrap_ready();
            }
            ClusterOperation::DeleteUser { user_id } => {
                if let Some(ref catalog) = self.catalog {
                    self.apply_removal("delete user", catalog.delete_user(user_id));
                }
            }
            ClusterOperation::SetUserEnabled { user_id, enabled } => {
                if let Some(ref catalog) = self.catalog {
                    self.apply_write(
                        "set user enabled",
                        catalog.set_user_enabled(user_id, *enabled),
                    );
                }
            }
            ClusterOperation::CreateApiKey {
                user_id,
                key_hash,
                label,
            } => {
                if let Some(ref catalog) = self.catalog {
                    self.apply_write(
                        "create api key",
                        catalog.create_api_key(user_id, key_hash, label),
                    );
                }
            }
            ClusterOperation::DeleteApiKey { user_id, key_hash } => {
                if let Some(ref catalog) = self.catalog {
                    self.apply_removal("delete api key", catalog.delete_api_key(user_id, key_hash));
                }
            }
            ClusterOperation::CreateRole { role } => {
                if let Some(ref catalog) = self.catalog {
                    self.apply_write("create role", catalog.create_role(role));
                }
            }
            ClusterOperation::DeleteRole { role_id } => {
                if let Some(ref catalog) = self.catalog {
                    self.apply_removal("delete role", catalog.delete_role(role_id));
                }
            }
            ClusterOperation::AddPermission {
                role_id,
                permission,
            } => {
                if let Some(ref catalog) = self.catalog {
                    self.apply_write(
                        "add permission",
                        catalog.add_permission(role_id, permission),
                    );
                }
            }
            ClusterOperation::RemovePermission { role_id, perm_id } => {
                if let Some(ref catalog) = self.catalog {
                    self.apply_removal(
                        "remove permission",
                        catalog.remove_permission(role_id, perm_id),
                    );
                }
            }
            ClusterOperation::AssignRole {
                user_id,
                role_id,
                assigned_by,
            } => {
                if let Some(ref catalog) = self.catalog {
                    self.apply_write(
                        "assign role",
                        catalog.assign_role(user_id, role_id, assigned_by),
                    );
                }
            }
            ClusterOperation::UnassignRole { user_id, role_id } => {
                if let Some(ref catalog) = self.catalog {
                    self.apply_removal("unassign role", catalog.unassign_role(user_id, role_id));
                }
            }
        }
    }
}

pub struct ClusterSnapshotBuilder {
    sm: ClusterStateMachine,
}

/// Assemble a snapshot from a state machine and its engine.
///
/// Shared by the builder and `get_current_snapshot` so there is one definition
/// of what a snapshot contains.
// The error type belongs to openraft and cannot be boxed without diverging from
// the trait signatures this feeds.
#[allow(clippy::result_large_err)]
/// What a snapshot's state blob contains.
///
/// The state machine alone is not enough for a node that joins after log
/// compaction: it carries the placement map but not the catalog, and routing
/// reads a collection's row while authorization reads the caller's role. A
/// metadata snapshot therefore carries the registry beside the machine; a shard
/// snapshot carries `None`, because the engine rebuilds its own collections'
/// rows from the archive.
#[derive(Serialize, Deserialize)]
struct SnapshotPayload {
    machine: ClusterStateMachine,
    #[serde(default)]
    registry: Option<crate::registry::RegistrySnapshot>,
}

// The error type is `openraft::StorageError`, imposed by the `RaftSnapshotBuilder`
// impl this feeds; boxing it here would only have to be unboxed one line later.
#[allow(clippy::result_large_err)]
fn assemble_snapshot(
    sm: &ClusterStateMachine,
) -> Result<Snapshot<RaftTypeConfig>, StorageError<u64>> {
    // Capture the boundary FIRST. The state machine is a clone, but the engine
    // sits behind an `Arc` and applies keep running while the archive is built.
    // A snapshot ahead of its boundary is harmless — replay re-applies those
    // entries and apply is idempotent — but a snapshot behind its boundary is
    // silent loss, because the missing entries may already have been purged.
    let boundary = sm.last_applied;

    let engine = sm.engine.clone().ok_or_else(|| StorageError::IO {
        source: openraft::StorageIOError::read(&std::io::Error::other(
            "cannot build a snapshot without an engine: the collection data is not held in \
             this state machine",
        )),
    })?;

    // The archive is a copy of each collection's checkpoint generation, so the
    // generations must cover the boundary before the archive exists. openraft
    // purges the log up to `last_log_id`; a stale generation would leave the
    // receiver needing entries that no longer exist.
    let scope = sm.snapshot_scope();
    engine
        .checkpoint_all(&scope)
        .map_err(|e| StorageError::IO {
            source: openraft::StorageIOError::read(&std::io::Error::other(format!(
                "checkpoint before snapshot failed: {e}"
            ))),
        })?;

    // Only the metadata group owns the registry. Capturing it for a shard group
    // would ship user records to a node that only needs vectors.
    let registry = match (sm.role, sm.catalog.as_ref()) {
        (GroupRole::Metadata, Some(catalog)) => Some(
            crate::registry::RegistrySnapshot::capture(catalog.as_ref()).map_err(|e| {
                StorageError::IO {
                    source: openraft::StorageIOError::read(&std::io::Error::other(format!(
                        "cannot capture the registry for a snapshot: {e}"
                    ))),
                }
            })?,
        ),
        _ => None,
    };

    let state = bincode::serialize(&SnapshotPayload {
        machine: sm.clone(),
        registry,
    })
    .map_err(|e| StorageError::IO {
        source: openraft::StorageIOError::read(&std::io::Error::other(format!(
            "cluster snapshot serialize failed: {e}"
        ))),
    })?;

    let archive = engine
        .snapshot_bytes(&state, &scope)
        .map_err(|e| StorageError::IO {
            source: openraft::StorageIOError::read(&std::io::Error::other(format!(
                "engine snapshot failed: {e}"
            ))),
        })?;

    let meta = SnapshotMeta {
        last_log_id: boundary,
        last_membership: sm.membership.clone(),
        snapshot_id: format!(
            "{}-{}",
            boundary.map(|id| id.index).unwrap_or(0),
            uuid::Uuid::new_v4()
        ),
    };

    Ok(Snapshot {
        meta,
        snapshot: Box::new(std::io::Cursor::new(archive)),
    })
}

impl RaftSnapshotBuilder<RaftTypeConfig> for ClusterSnapshotBuilder {
    async fn build_snapshot(&mut self) -> Result<Snapshot<RaftTypeConfig>, StorageError<u64>> {
        assemble_snapshot(&self.sm)
    }
}

impl RaftStateMachine<RaftTypeConfig> for ClusterStateMachine {
    type SnapshotBuilder = ClusterSnapshotBuilder;

    async fn applied_state(
        &mut self,
    ) -> Result<(Option<LogId<u64>>, StoredMembership<u64, u64>), StorageError<u64>> {
        Ok((self.last_applied, self.membership.clone()))
    }

    async fn apply<I>(&mut self, entries: I) -> Result<Vec<String>, StorageError<u64>>
    where
        I: IntoIterator<Item = Entry<RaftTypeConfig>> + Send,
        I::IntoIter: Send,
    {
        let mut responses = Vec::new();
        for entry in entries {
            self.last_applied = Some(entry.log_id);

            match &entry.payload {
                EntryPayload::Normal(ClusterOperation::Batch {
                    collection_id,
                    ops,
                    dimension,
                }) => {
                    // The log index is the dataset sequence. Applying with it
                    // means a replayed entry is idempotent and a snapshot's
                    // boundary is the same number on every node.
                    let index = entry.log_id.index;
                    let outcome = match self.engine.clone() {
                        Some(engine) => {
                            // Applying a batch drains the brute-force buffer into
                            // the HNSW graph, which is CPU-bound and can hold a
                            // worker for a long time. Raft's apply task shares
                            // the runtime with the heartbeat task, so doing that
                            // here would let a large batch delay heartbeats and
                            // trigger an election under load — the same failure
                            // the HTTP write path already offloads with
                            // `spawn_blocking`. `apply_batch` must stay total and
                            // ordered, so the next entry still waits for this
                            // one to finish.
                            let collection_id = *collection_id;
                            let ops: Arc<[rekha_core::op::Operation]> = Arc::from(ops.as_slice());
                            let dimension = *dimension;
                            match tokio::task::spawn_blocking(move || {
                                engine.apply_batch(&collection_id, &ops, index, dimension)
                            })
                            .await
                            {
                                Ok(result) => result,
                                Err(join) => {
                                    Err(format!("apply worker for {collection_id} failed: {join}"))
                                }
                            }
                        }
                        None => Err(format!(
                            "no engine is attached, so a batch for {collection_id} cannot be applied"
                        )),
                    };
                    let response = match outcome {
                        Ok(()) => BatchResponse::ok(ops.len(), index),
                        Err(e) => {
                            // A committed entry must apply. If this node cannot,
                            // it has diverged or hit a bug; serving on would
                            // advertise a log it did not honour. The failure is
                            // also reported in the response, so the client that
                            // wrote it is told rather than only the next reader
                            // of /ready.
                            eprintln!(
                                "cluster: FAILED to apply committed batch for {collection_id} \
                                 at index {index}: {e}; marking this node not-ready"
                            );
                            self.note_apply_failure();
                            BatchResponse::failed(index, e)
                        }
                    };
                    responses.push(serde_json::to_string(&response).unwrap_or_else(|e| {
                        format!("{{\"applied\":0,\"seq\":{index},\"error\":\"{e}\"}}")
                    }));
                }
                EntryPayload::Normal(op) => {
                    self.apply_operation(op);
                    responses.push(format!("applied: {:?}", op));
                }
                EntryPayload::Membership(m) => {
                    self.membership = StoredMembership::new(Some(entry.log_id), m.clone());
                    responses.push(format!("membership: {:?}", m));
                }
                EntryPayload::Blank => {
                    responses.push("blank".to_string());
                }
            }
        }
        Ok(responses)
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        ClusterSnapshotBuilder { sm: self.clone() }
    }

    async fn begin_receiving_snapshot(
        &mut self,
    ) -> Result<Box<std::io::Cursor<Vec<u8>>>, StorageError<u64>> {
        Ok(Box::new(std::io::Cursor::new(Vec::new())))
    }

    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta<u64, u64>,
        snapshot: Box<std::io::Cursor<Vec<u8>>>,
    ) -> Result<(), StorageError<u64>> {
        let data = snapshot.into_inner();

        let engine = self.engine.clone().ok_or_else(|| StorageError::IO {
            source: openraft::StorageIOError::read(&std::io::Error::other(
                "cannot install a snapshot without an engine",
            )),
        })?;

        // Restore the engine first. If it fails, this node's metadata is left
        // untouched rather than describing collections whose data did not land.
        let state = engine.restore_bytes(&data).map_err(|e| StorageError::IO {
            source: openraft::StorageIOError::read(&std::io::Error::other(format!(
                "engine restore failed: {e}"
            ))),
        })?;

        let restored: SnapshotPayload =
            bincode::deserialize(&state).map_err(|e| StorageError::IO {
                source: openraft::StorageIOError::read(&std::io::Error::other(format!(
                    "cluster snapshot deserialize failed: {e}"
                ))),
            })?;

        // Keep this node's handles: `#[serde(skip)]` means the deserialised copy
        // has none of them, and dropping the live ones would detach the state
        // machine from the engine it just restored into.
        let engine_handle = self.engine.clone();
        let catalog = self.catalog.clone();
        let addresses = self.addresses.clone();
        // Readiness is a property of *this process*, not of the snapshot. A node
        // that had already failed to apply a committed entry must not become
        // ready just because it installed one: the snapshot cannot prove the
        // divergence that raised the flag is gone. Preserving it keeps the
        // failure visible until an operator restarts the node.
        let apply_failure = self.apply_failure.clone();
        let bootstrap_ready = self.bootstrap_ready.clone();

        // The registry first, through the *local* handle. A node that joins from
        // a snapshot has none of these rows, and until they are written it can
        // route nothing and authorize nobody.
        if let (Some(registry), Some(catalog)) = (&restored.registry, catalog.as_ref()) {
            registry
                .restore(catalog.as_ref())
                .map_err(|e| StorageError::IO {
                    source: openraft::StorageIOError::write(&std::io::Error::other(format!(
                        "installing a snapshot could not rebuild the catalog: {e}"
                    ))),
                })?;
            // A snapshot may supersede the bootstrap entry, so the flag cannot
            // come from replay alone. The restored registry is the evidence.
            if !registry.users.is_empty() {
                self.mark_bootstrap_ready();
            }
        }
        let restored = restored.machine;

        // The snapshot's node set is the authority for where nodes live: an
        // address learned from an entry the snapshot supersedes must go, not
        // linger and shadow it.
        if let Some(addresses) = &addresses {
            addresses.replace_all(
                restored
                    .nodes
                    .iter()
                    .map(|(id, info)| (*id, info.addr.clone())),
            );
        }

        *self = restored;
        self.engine = engine_handle;
        self.catalog = catalog;
        self.addresses = addresses;
        self.apply_failure = apply_failure;
        self.bootstrap_ready = bootstrap_ready;

        self.last_applied = meta.last_log_id;
        self.membership = meta.last_membership.clone();
        Ok(())
    }

    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<Snapshot<RaftTypeConfig>>, StorageError<u64>> {
        // This state machine keeps no snapshot cache: it is rebuilt by replaying
        // the log at startup, and snapshots are produced on demand by the
        // builder when openraft needs to ship one.
        //
        // Returning a freshly built snapshot here would be both expensive and
        // wrong: openraft calls this during `Raft::new`, so every start would
        // checkpoint every collection and archive the whole corpus before the
        // node accepted a single request.
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openraft::CommittedLeaderId;

    fn make_log_id(index: u64) -> LogId<u64> {
        LogId::new(CommittedLeaderId::new(1, 0), index)
    }

    #[test]
    fn test_apply_add_node() {
        let mut sm = ClusterStateMachine::new();
        let op = ClusterOperation::AddNode {
            node_id: 1,
            addr: "127.0.0.1:8001".into(),
        };
        sm.apply_operation(&op);
        assert_eq!(sm.nodes.len(), 1);
        assert_eq!(sm.nodes[&1].addr, "127.0.0.1:8001");
    }

    #[test]
    fn test_apply_remove_node() {
        let mut sm = ClusterStateMachine::new();
        sm.apply_operation(&ClusterOperation::AddNode {
            node_id: 1,
            addr: "127.0.0.1:8001".into(),
        });
        sm.apply_operation(&ClusterOperation::RemoveNode { node_id: 1 });
        assert!(sm.nodes.is_empty());
    }

    /// A committed request the application rejects is not divergence.
    ///
    /// Every replica has the same state and reaches the same verdict, so the
    /// node has not lost an entry. Treating it as divergence made a single
    /// RBAC call referencing an unknown user or role mark the whole cluster
    /// not-ready until restart.
    #[tokio::test]
    async fn a_rejected_rbac_write_does_not_take_the_node_out_of_service() {
        use rekha_storage::RedbCatalog;

        let dir = tempfile::tempdir().expect("tempdir");
        let catalog =
            Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).expect("catalog"));

        let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut sm = ClusterStateMachine::for_metadata(1);
        sm.set_catalog(catalog);
        sm.set_apply_failure_flag(flag);

        // A key for a user that does not exist.
        sm.apply_operation(&ClusterOperation::CreateApiKey {
            user_id: "nobody".into(),
            key_hash: [7u8; 32],
            label: "test".into(),
        });
        assert!(
            !sm.apply_failed(),
            "a user-input rejection must not mark the node not-ready"
        );

        // A permission on a role that does not exist.
        sm.apply_operation(&ClusterOperation::AddPermission {
            role_id: "no-such-role".into(),
            permission: rekha_core::auth::Permission {
                id: "p1".into(),
                action: "collection.read".into(),
                tenant: None,
                database: None,
            },
        });
        assert!(
            !sm.apply_failed(),
            "a role-input rejection must not mark the node not-ready"
        );
    }

    #[test]
    fn test_apply_add_collection() {
        let mut sm = ClusterStateMachine::new();
        let cid = Uuid::new_v4();
        let mut config = rekha_core::config::CollectionConfig::new(
            "test".into(),
            128,
            rekha_core::types::Distance::L2,
        );
        config.id = cid;
        sm.apply_operation(&ClusterOperation::AddCollection {
            config,
            shard: Some(ShardId(7)),
        });
        assert_eq!(sm.placements[&cid], ShardId(7));
    }

    #[tokio::test]
    async fn test_apply_entries() {
        let mut sm = ClusterStateMachine::new();

        let op = ClusterOperation::AddNode {
            node_id: 1,
            addr: "127.0.0.1:8001".into(),
        };

        let entries = vec![Entry {
            log_id: make_log_id(0),
            payload: EntryPayload::Normal(op),
        }];

        let responses = sm.apply(entries).await.unwrap();
        assert_eq!(responses.len(), 1);
        assert!(responses[0].contains("applied"));
        assert_eq!(sm.nodes.len(), 1);
        assert_eq!(sm.last_applied, Some(make_log_id(0)));
    }

    /// A provider that records the order of snapshot operations and behaves
    /// like an identity codec, so the state machine's own logic is what is
    /// under test rather than the engine's archive format.
    #[derive(Default)]
    struct RecordingProvider {
        events: std::sync::Mutex<Vec<&'static str>>,
        fail_checkpoint: std::sync::atomic::AtomicBool,
        fail_apply: std::sync::atomic::AtomicBool,
        slow_apply: std::sync::atomic::AtomicBool,
    }

    impl RecordingProvider {
        fn events(&self) -> Vec<&'static str> {
            self.events.lock().unwrap().clone()
        }
        fn fail_checkpoint(&self) {
            self.fail_checkpoint
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
        fn fail_apply(&self) {
            self.fail_apply
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
        fn slow_apply(&self) {
            self.slow_apply
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl EngineSnapshotProvider for RecordingProvider {
        fn snapshot_bytes(
            &self,
            state: &[u8],
            _collections: &[uuid::Uuid],
        ) -> Result<Vec<u8>, String> {
            self.events.lock().unwrap().push("snapshot_bytes");
            Ok(state.to_vec())
        }
        fn restore_bytes(&self, archive: &[u8]) -> Result<Vec<u8>, String> {
            self.events.lock().unwrap().push("restore_bytes");
            Ok(archive.to_vec())
        }
        fn checkpoint_all(&self, _collections: &[uuid::Uuid]) -> Result<usize, String> {
            self.events.lock().unwrap().push("checkpoint_all");
            if self
                .fail_checkpoint
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                return Err("simulated checkpoint failure".into());
            }
            Ok(1)
        }
        fn record_collection(
            &self,
            _config: &rekha_core::config::CollectionConfig,
            _shard: Option<ShardId>,
        ) -> Result<(), String> {
            self.events.lock().unwrap().push("record_collection");
            Ok(())
        }

        fn materialize_collection(&self, _collection_id: &uuid::Uuid) -> Result<(), String> {
            self.events.lock().unwrap().push("materialize_collection");
            Ok(())
        }
        fn drop_collection(&self, _collection_id: &uuid::Uuid) -> Result<(), String> {
            Ok(())
        }
        fn apply_batch(
            &self,
            _collection_id: &uuid::Uuid,
            ops: &[rekha_core::op::Operation],
            _index: u64,
            _dimension: usize,
        ) -> Result<(), String> {
            self.events.lock().unwrap().push("apply_batch");
            let _ = ops;
            if self.slow_apply.load(std::sync::atomic::Ordering::SeqCst) {
                // Stand-in for the CPU-bound buffer flush into the HNSW graph.
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
            if self.fail_apply.load(std::sync::atomic::Ordering::SeqCst) {
                return Err("simulated apply failure".into());
            }
            Ok(())
        }
    }

    /// A committed batch is CPU-bound: applying it drains the brute-force
    /// buffer into the HNSW graph. It must run on the blocking pool, not on the
    /// async workers that carry Raft heartbeats.
    ///
    /// This is the same failure the HTTP write path fixed with `offload`: a
    /// 250 ms apply on a one-worker runtime delays a 50 ms timer to 250 ms,
    /// which is past the election floor, so sustained writes cause elections.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn a_slow_apply_does_not_starve_the_runtime() {
        let provider = Arc::new(RecordingProvider::default());
        provider.slow_apply();
        let mut sm = ClusterStateMachine::new();
        sm.set_engine(provider);

        let entries = vec![Entry {
            log_id: make_log_id(1),
            payload: EntryPayload::Normal(ClusterOperation::Batch {
                collection_id: Uuid::new_v4(),
                ops: Vec::new(),
                dimension: 4,
            }),
        }];

        let started = std::time::Instant::now();
        let apply = tokio::spawn(async move { sm.apply(entries).await });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let elapsed = started.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(200),
            "a slow apply delayed the runtime for {elapsed:?}; it must run on the blocking pool"
        );
        apply
            .await
            .expect("apply task must not panic")
            .expect("apply must succeed");
    }

    /// The log is purged up to `last_log_id`, so the generations must cover that
    /// boundary *before* the archive is built. Checkpointing after archiving, or
    /// not at all, leaves a receiver needing purged entries.
    #[tokio::test]
    async fn snapshot_checkpoints_before_archiving_and_freezes_the_boundary() {
        let provider = Arc::new(RecordingProvider::default());
        let mut sm = ClusterStateMachine::new();
        sm.set_engine(provider.clone());
        sm.last_applied = Some(make_log_id(7));

        let mut builder = sm.get_snapshot_builder().await;
        let snapshot = builder.build_snapshot().await.unwrap();

        assert_eq!(
            provider.events(),
            vec!["checkpoint_all", "snapshot_bytes"],
            "the checkpoint must happen before the archive is read"
        );
        assert_eq!(
            snapshot.meta.last_log_id,
            Some(make_log_id(7)),
            "the boundary is the applied index at build time"
        );
    }

    /// A snapshot that cannot cover its boundary must fail rather than ship and
    /// let openraft purge the entries it is missing.
    #[tokio::test]
    async fn a_failed_checkpoint_prevents_the_snapshot() {
        let provider = Arc::new(RecordingProvider::default());
        provider.fail_checkpoint();

        let mut sm = ClusterStateMachine::new();
        sm.set_engine(provider.clone());
        sm.last_applied = Some(make_log_id(4));

        let mut builder = sm.get_snapshot_builder().await;
        assert!(builder.build_snapshot().await.is_err());
        assert_eq!(
            provider.events(),
            vec!["checkpoint_all"],
            "nothing may be archived when the checkpoint failed"
        );
    }

    /// Installing restores the engine first and then adopts the metadata, and it
    /// must keep its own engine handle: the deserialised copy has none.
    /// A data batch is applied with the entry's log index as the sequence, so
    /// replay is idempotent and a snapshot boundary means the same number
    /// everywhere.
    #[tokio::test]
    async fn a_batch_is_applied_with_its_log_index() {
        let provider = Arc::new(RecordingProvider::default());
        let mut sm = ClusterStateMachine::new();
        sm.set_engine(provider.clone());
        let cid = Uuid::new_v4();

        let entry = Entry {
            log_id: make_log_id(9),
            payload: EntryPayload::Normal(ClusterOperation::Batch {
                collection_id: cid,
                ops: Vec::new(),
                dimension: 4,
            }),
        };
        let responses = sm.apply(vec![entry]).await.unwrap();

        assert_eq!(provider.events(), vec!["apply_batch"]);
        assert_eq!(responses.len(), 1);
        assert!(
            responses[0].contains("\"seq\":9"),
            "response must report the index: {}",
            responses[0]
        );
        assert!(!sm.apply_failed());
    }

    /// A committed entry that cannot be applied must stop the node answering
    /// rather than let it serve a state that disagrees with its own log.
    #[tokio::test]
    async fn a_failed_apply_marks_the_node_not_ready() {
        let provider = Arc::new(RecordingProvider::default());
        provider.fail_apply();
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let mut sm = ClusterStateMachine::new();
        sm.set_engine(provider.clone());
        sm.set_apply_failure_flag(flag.clone());

        let entry = Entry {
            log_id: make_log_id(2),
            payload: EntryPayload::Normal(ClusterOperation::Batch {
                collection_id: Uuid::new_v4(),
                ops: Vec::new(),
                dimension: 4,
            }),
        };
        // apply() itself must not return Err: openraft treats that as fatal for
        // the whole store, which would take every collection down.
        sm.apply(vec![entry]).await.unwrap();

        assert!(sm.apply_failed());
        assert!(
            flag.load(std::sync::atomic::Ordering::SeqCst),
            "the shared readiness flag must be raised"
        );
    }

    #[tokio::test]
    async fn install_restores_the_engine_then_adopts_metadata() {
        let source_provider = Arc::new(RecordingProvider::default());
        let mut source = ClusterStateMachine::new();
        source.set_engine(source_provider.clone());
        source.last_applied = Some(make_log_id(5));
        source.nodes.insert(
            1,
            NodeInfo {
                node_id: 1,
                addr: "127.0.0.1:8001".into(),
            },
        );
        source.placements.insert(Uuid::new_v4(), ShardId(3));

        let snapshot = source
            .get_snapshot_builder()
            .await
            .build_snapshot()
            .await
            .unwrap();

        let target_provider = Arc::new(RecordingProvider::default());
        let mut target = ClusterStateMachine::new();
        target.set_engine(target_provider.clone());

        target
            .install_snapshot(&snapshot.meta, snapshot.snapshot)
            .await
            .unwrap();

        assert_eq!(target_provider.events(), vec!["restore_bytes"]);
        assert_eq!(target.last_applied, Some(make_log_id(5)));
        assert_eq!(target.nodes.len(), 1);
        assert_eq!(target.placements.len(), 1);
        assert!(
            target.engine.is_some(),
            "install must keep this node's engine handle, not adopt a skipped one"
        );
    }

    /// A node that joins from a snapshot must be able to answer requests.
    ///
    /// Routing reads a collection's catalog row and authorization reads the
    /// caller's role, so a snapshot that carried only the placement map would
    /// leave the receiver unable to serve anything. Before the registry was
    /// part of the metadata snapshot, this target catalog stayed empty.
    #[tokio::test]
    async fn a_metadata_snapshot_rebuilds_the_registry_on_the_receiver() {
        use rekha_core::config::CollectionConfig;
        use rekha_core::types::Distance;
        use rekha_storage::{Catalog, RedbCatalog};

        let source_dir = tempfile::tempdir().expect("tempdir");
        let source_catalog =
            Arc::new(RedbCatalog::open(source_dir.path().join("catalog.redb")).expect("catalog"));
        source_catalog.create_tenant("acme").expect("tenant");
        source_catalog
            .create_database("acme", "prod")
            .expect("database");
        let mut config = CollectionConfig::new("docs".into(), 3, Distance::L2);
        config.tenant = "acme".into();
        config.database = "prod".into();
        source_catalog
            .record_collection(&config, Some(ShardId(2)))
            .expect("collection");
        let user = rekha_core::auth::UserRecord {
            id: "u1".into(),
            name: "operator".into(),
            enabled: true,
            created_at_ms: 1,
        };
        source_catalog.create_user(&user).expect("user");

        let mut source = ClusterStateMachine::for_metadata(1);
        source.set_catalog(source_catalog.clone());
        source.set_engine(Arc::new(RecordingProvider::default()));
        source.last_applied = Some(make_log_id(9));
        source.placements.insert(config.id, ShardId(2));

        let snapshot = source
            .get_snapshot_builder()
            .await
            .build_snapshot()
            .await
            .unwrap();

        // The receiver starts with nothing: no rows, no users.
        let target_dir = tempfile::tempdir().expect("tempdir");
        let target_catalog =
            Arc::new(RedbCatalog::open(target_dir.path().join("catalog.redb")).expect("catalog"));
        assert!(
            target_catalog
                .list_collections_all()
                .expect("empty")
                .is_empty(),
            "the receiver must start empty for this test to mean anything"
        );

        let mut target = ClusterStateMachine::for_metadata(2);
        target.set_catalog(target_catalog.clone());
        target.set_engine(Arc::new(RecordingProvider::default()));
        target
            .install_snapshot(&snapshot.meta, snapshot.snapshot)
            .await
            .unwrap();

        let rows = target_catalog.list_collections_all().expect("rows");
        assert_eq!(rows.len(), 1, "the collection row must be restored");
        assert_eq!(rows[0].shard, Some(ShardId(2)));
        assert_eq!(
            rows[0].config.name, "docs",
            "the row must carry the config, not just the id"
        );
        assert_eq!(
            target_catalog.list_tenants().expect("tenants"),
            vec!["acme".to_owned()]
        );
        assert_eq!(
            target_catalog.get_user("u1").expect("user").map(|u| u.name),
            Some("operator".to_owned()),
            "authorization needs the user, not only the collection"
        );
        // The placement map travels too — it is what a shard group uses to
        // scope its snapshots — but the *routing* answer is the row above.
        assert_eq!(target.placements.get(&rows[0].config.id), Some(&ShardId(2)));
    }

    #[tokio::test]
    async fn test_applied_state() {
        let mut sm = ClusterStateMachine::new();
        let (last_applied, _membership) = sm.applied_state().await.unwrap();
        assert_eq!(last_applied, None);
    }
}
