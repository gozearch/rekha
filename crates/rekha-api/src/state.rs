use crate::prelude::*;

// Shared state
// ---------------------------------------------------------------------------

/// Server-wide settings that are not owned by the engine.
pub struct ServerConfig {
    pub tenant: String,
    pub database: String,
    pub api_key: Option<String>,
    pub cluster_secret: Option<String>,
    pub pepper: Vec<u8>,
    pub allow_open_mode: bool,
}

pub(crate) type ShardWriteLockMap =
    HashMap<(Uuid, Option<rekha_cluster::ShardId>), Arc<tokio::sync::Mutex<()>>>;

pub struct AppState {
    pub engine: Arc<Engine>,
    pub tenant: String,
    pub database: String,
    pub raft: Option<Arc<openraft::Raft<rekha_cluster::RaftTypeConfig>>>,
    pub api_key: Option<String>,
    pub cluster_secret: Option<String>,
    pub pepper: Vec<u8>,
    pub allow_open_mode: bool,
    /// Where the other nodes live, for forwarding writes to the leader.
    ///
    /// Empty in single-node mode. Seeded from `--peers` and extended by
    /// committed membership entries; the addresses are the nodes' internal
    /// listeners, which is where forwarded writes land.
    ///
    /// Cloning it shares the same map, so there is one directory per process
    /// and no second copy to fall out of date.
    pub peers: rekha_cluster::PeerDirectory,
    /// Outbound client for forwarding writes.
    pub http: reqwest::Client,
    /// Leadership as observed from consensus, when this node has a Raft group.
    ///
    /// Used to *route* a write, never to authorise one: the log decides whether
    /// a write is legal, this only decides whether to forward it first.
    pub leader: Option<rekha_cluster::LeaderTracker>,
    /// Raised when a committed entry could not be applied. A node in this state
    /// must stop reporting readiness.
    pub apply_failure: Arc<AtomicBool>,
    /// Whether the cluster's bootstrap credentials are applied on this node.
    ///
    /// Until they are, a request can authenticate as the bootstrap user and
    /// still hold no permissions, which reads as a 403 to a client that waited
    /// for readiness. `/ready` stays false until this is set. Defaults to true
    /// for callers that never attach a flag (tests, single-node open mode).
    pub bootstrap_ready: Arc<AtomicBool>,
    /// Whether client data writes travel through the Raft log.
    ///
    /// Deliberately separate from `raft.is_some()`. A single-node deployment
    /// still holds a Raft group for metadata (RBAC, the collection registry),
    /// so using the presence of `raft` to decide the data path sent single-node
    /// writes into the log while the engine was configured to be its own
    /// durable record — and the apply refused, after the entry had committed.
    /// The client got 201 and the record was never stored.
    pub data_via_log: bool,
    /// The consensus groups this node runs, when it runs more than the
    /// metadata group.
    ///
    /// `None` in single-node mode and in a cluster started with one group, so
    /// every earlier behaviour is preserved without a flag.
    pub shards: Option<Arc<rekha_cluster::ShardCluster>>,
    /// Collections already placed per node, cached for this leadership term.
    ///
    /// Only the leader changes placement, so this is exactly accurate while this
    /// node leads, and rebuilt once when it starts leading. Rebuilding on every
    /// creation would make creating `n` collections `O(n^2)` — measured: 200
    /// collections took 29s before placement became a decision and 82s when the
    /// load was recomputed from the catalog each time.
    pub placement_cache: tokio::sync::Mutex<Option<(u64, std::collections::HashMap<u64, u32>)>>,
    /// One async lock per collection, held from prepare through propose.
    ///
    /// The single-node path validates and writes inside one collection lock, so
    /// Two metadata updates cannot merge from the same base. Splitting prepare
    /// from apply would reopen that window; this closes it again, and costs
    /// nothing in throughput because the log already totals-orders writes to a
    /// collection partition.
    write_locks: parking_lot::Mutex<ShardWriteLockMap>,
    /// Serialises snapshot chunk appends.
    ///
    /// Chunks of one transfer must land in order and without interleaving: the
    /// receiver appends at an exact offset, so two in-flight chunks would both
    /// see the same length and one would overwrite the other. Snapshot transfers
    /// are rare (a replica rejoining), so one lock for all of them is enough.
    pub snapshot_staging: tokio::sync::Mutex<()>,
    pub node_id: Option<u64>,
    pub raft_config: Option<Arc<openraft::Config>>,
}

/// The process-wide HTTP client.
///
/// Two problems, one setting:
///
/// * `reqwest::Client::new()` reads the system proxy configuration, which on
///   macOS goes through `SCDynamicStoreCreate`. That call deadlocks when several
///   threads construct a client at the same time (`_dispatch_once_wait` never
///   returns), observed as a test suite that reports every test passed and then
///   hangs.
/// * More importantly, this client talks to *peers*, not to the internet. A
///   system or `HTTP_PROXY` proxy would silently route Raft RPCs and forwarded
///   writes through an intermediary that cannot speak the protocol.
///
/// `no_proxy()` disables the matcher entirely, so neither applies. One client
/// for the process also means one connection pool rather than one per `AppState`.
pub(crate) fn shared_http_client() -> reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| match reqwest::Client::builder().no_proxy().build() {
            Ok(client) => client,
            Err(e) => {
                // TLS backend initialisation. There is no safer fallback that
                // does not reintroduce the proxy matcher; report and continue so
                // a node without TLS still serves plain HTTP.
                eprintln!("http client: proxy-free client could not be built ({e})");
                reqwest::Client::new()
            }
        })
        .clone()
}

impl AppState {
    pub fn new(
        engine: Arc<Engine>,
        raft: Option<Arc<openraft::Raft<rekha_cluster::RaftTypeConfig>>>,
        config: ServerConfig,
    ) -> Self {
        Self {
            engine,
            tenant: config.tenant,
            database: config.database,
            raft,
            api_key: config.api_key,
            cluster_secret: config.cluster_secret,
            pepper: config.pepper,
            allow_open_mode: config.allow_open_mode,
            peers: rekha_cluster::PeerDirectory::default(),
            http: shared_http_client(),
            leader: None,
            apply_failure: Arc::new(AtomicBool::new(false)),
            bootstrap_ready: Arc::new(AtomicBool::new(true)),
            data_via_log: false,
            shards: None,
            placement_cache: tokio::sync::Mutex::new(None),
            write_locks: parking_lot::Mutex::new(HashMap::new()),
            snapshot_staging: tokio::sync::Mutex::new(()),
            node_id: None,
            raft_config: None,
        }
    }

    pub fn with_node_id(mut self, node_id: u64) -> Self {
        self.node_id = Some(node_id);
        self
    }

    pub fn with_raft_config(mut self, config: Arc<openraft::Config>) -> Self {
        self.raft_config = Some(config);
        self
    }

    /// Attach the consensus leadership view, used to route writes.
    pub fn with_leader_tracker(mut self, tracker: rekha_cluster::LeaderTracker) -> Self {
        self.leader = Some(tracker);
        self
    }

    /// Declare whether client data writes go through the Raft log.
    ///
    /// Must agree with the engine's `ApplyMode`: `true` pairs with
    /// `ApplyMode::Committed`, `false` with `ApplyMode::Durable`. Startup
    /// asserts that, because the two disagreeing is how a node acknowledges
    /// writes it never stores.
    pub fn with_data_via_log(mut self, enabled: bool) -> Self {
        self.data_via_log = enabled;
        self
    }

    /// Attach the per-shard consensus groups this node participates in.
    pub fn with_shards(mut self, shards: Arc<rekha_cluster::ShardCluster>) -> Self {
        self.shards = Some(shards);
        self
    }

    /// The `Raft` instance for `group`.
    ///
    /// `METADATA_GROUP` is this node's metadata group; every other id is a
    /// shard the node hosts. `None` means "this node is not a member", which is
    /// a routing answer rather than an error.
    pub fn raft_for(
        &self,
        group: rekha_cluster::GroupId,
    ) -> Option<Arc<openraft::Raft<rekha_cluster::RaftTypeConfig>>> {
        if group == rekha_cluster::METADATA_GROUP {
            return self.raft.clone();
        }
        self.shards
            .as_ref()?
            .raft(rekha_cluster::ShardId(group - 1))
    }

    /// Whether this node hosts the group that holds `collection`.
    pub fn hosts_collection(&self, collection: &uuid::Uuid) -> bool {
        match self.shards.as_ref() {
            // No shard registry means one group holds everything.
            None => self.raft.is_some(),
            Some(shards) => match self.shard_of(collection) {
                Some(shard) => shards.hosts(shard),
                None => {
                    if let Ok(Some(rec)) = self.engine.get_collection_by_id(collection) {
                        rec.shard.is_none()
                    } else {
                        false
                    }
                }
            },
        }
    }

    /// Whether this node is a **voter** in the consensus group that holds
    /// `collection`.
    ///
    /// This is the condition under which a local count is a real answer rather
    /// than an artefact of having opened the files early: a node can hold a
    /// group and materialized partitions before it has replicated anything
    /// (`ensure_started` does exactly that while a promoted node catches up).
    /// A voter has the log up to its join point by joint consensus, so its
    /// applied state is authoritative for what it claims to hold.
    ///
    /// Single-node mode has no shard registry and no replication, so the local
    /// engine is authoritative: `true`. A clustered node with no group for the
    /// collection is not a voter: `false`, which callers must read as "forward
    /// or refuse", never as "zero".
    pub fn is_voter_for(&self, collection: &uuid::Uuid) -> bool {
        let Some(shards) = self.shards.as_ref() else {
            // One group holds everything: a single node, or a cluster started
            // with only the metadata group. This node's membership in that
            // group is the answer, and with no consensus at all the local
            // engine is the record.
            return match self.raft.as_ref() {
                Some(raft) => {
                    let metrics = raft.metrics();
                    let m = metrics.borrow();
                    m.membership_config
                        .membership()
                        .voter_ids()
                        .any(|id| id == m.id)
                }
                None => true,
            };
        };
        let Some(shard) = self.shard_of(collection) else {
            // An unsharded row inside a cluster: one group holds everything and
            // this node's membership in it is the answer.
            return match self.raft.as_ref() {
                Some(raft) => {
                    let metrics = raft.metrics();
                    let m = metrics.borrow();
                    m.membership_config
                        .membership()
                        .voter_ids()
                        .any(|id| id == m.id)
                }
                None => false,
            };
        };
        match shards.raft(shard) {
            Some(raft) => {
                let metrics = raft.metrics();
                let m = metrics.borrow();
                m.membership_config
                    .membership()
                    .voter_ids()
                    .any(|id| id == m.id)
            }
            // The group is not running here, so this node holds none of it.
            None => false,
        }
    }

    /// The shard that holds `collection`, from its catalog row.
    ///
    /// This is the only way a collection's shard is learned. It is decided once
    /// by the metadata leader and recorded; deriving it from the id let routing
    /// and storage disagree about where the bytes were, and the disagreement
    /// only showed up as reads returning empty from a node that thought it was
    /// a replica.
    ///
    /// `None` means the row is missing (the create has not been applied here
    /// yet) or unsharded (single node). Both are "this node cannot answer a
    /// placement question", which callers must treat as retry-later, never as
    /// "route somewhere plausible".
    pub fn shard_of(&self, collection: &uuid::Uuid) -> Option<rekha_cluster::ShardId> {
        match self.engine.get_collection_by_id(collection) {
            Ok(Some(record)) => record.shard,
            Ok(None) => None,
            Err(e) => {
                tracing::warn!(collection = %collection, %e, "could not read a collection's shard");
                None
            }
        }
    }

    /// Share the flag the state machine raises when a committed entry cannot be
    /// applied.
    pub fn with_apply_failure_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.apply_failure = flag;
        self
    }

    /// Share the flag that says the bootstrap credentials are applied here.
    pub fn with_bootstrap_ready_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.bootstrap_ready = flag;
        self
    }

    pub(crate) fn write_lock(
        &self,
        collection_id: Uuid,
        shard: Option<rekha_cluster::ShardId>,
    ) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.write_locks.lock();
        locks
            .entry((collection_id, shard))
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }

    /// Attach the cluster topology so writes can be forwarded to the leader.
    pub fn with_peer_directory(mut self, peers: rekha_cluster::PeerDirectory) -> Self {
        self.peers = peers;
        self
    }

    /// Return target nodes for `shard`: leader first if known, then remaining owners.
    pub(crate) fn shard_targets(&self, shard: rekha_cluster::ShardId) -> Vec<u64> {
        let Some(shards) = self.shards.as_ref() else {
            return Vec::new();
        };
        let mut targets = Vec::new();
        if let Some(leader) = shards.leader(shard) {
            targets.push(leader);
        }
        for owner in shards.owners(shard) {
            if !targets.contains(&owner) {
                targets.push(owner);
            }
        }
        targets
    }

    /// Build an internal HTTP POST request with cluster secret header attached.
    pub(crate) fn internal_post(&self, url: impl reqwest::IntoUrl) -> reqwest::RequestBuilder {
        let mut req = self.http.post(url);
        if let Some(ref secret) = self.cluster_secret {
            req = req.header(crate::leadership::SECRET_HEADER, secret);
        }
        req
    }
}
