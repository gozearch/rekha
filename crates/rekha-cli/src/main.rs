#![deny(clippy::unwrap_used, clippy::expect_used)]

use clap::{Parser, Subcommand};
use reqwest::Client;
use serde_json::Value;

#[derive(Subcommand)]
enum MembershipAction {
    /// List cluster members
    List,
    /// Add a node as learner
    Add {
        #[arg(long)]
        node_id: u64,
        #[arg(long)]
        addr: String,
    },
    /// Remove a node
    Remove {
        #[arg(long)]
        node_id: u64,
    },
}

#[derive(Subcommand)]
enum AuthAction {
    /// Create a new user (returns API key)
    CreateUser {
        /// User name
        name: String,
    },
    /// List all users
    ListUsers,
    /// Delete a user
    DeleteUser {
        /// User id
        id: String,
    },
    /// Create a new API key for a user
    CreateKey {
        /// User id
        user_id: String,
        /// Key label
        #[arg(long, default_value = "default")]
        label: String,
    },
    /// List keys for a user
    ListKeys {
        /// User id
        user_id: String,
    },
    /// Create a role
    CreateRole {
        /// Role name
        name: String,
        /// Description
        #[arg(long, default_value = "")]
        description: String,
    },
    /// List all roles
    ListRoles,
    /// Delete a role
    DeleteRole {
        /// Role id
        id: String,
    },
    /// Add a permission to a role
    AddPermission {
        /// Role id
        role_id: String,
        /// Action (e.g. record.query, collection.create, *)
        action: String,
        /// Tenant scope (optional)
        #[arg(long)]
        tenant: Option<String>,
        /// Database scope (optional)
        #[arg(long)]
        database: Option<String>,
    },
    /// Assign a role to a user
    AssignRole {
        /// User id
        user_id: String,
        /// Role id
        role_id: String,
    },
    /// Unassign a role from a user
    UnassignRole {
        /// User id
        user_id: String,
        /// Role id
        role_id: String,
    },
    /// Show current identity (whoami)
    Whoami,
}

#[derive(Parser)]
#[command(name = "rekha", about = "RekhaDB CLI")]
struct Cli {
    /// Base URL of the RekhaDB API server.
    #[arg(long, default_value = "http://localhost:8000", env = "REKHA_API_URL")]
    url: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the API server.
    Serve {
        /// Data directory path.
        #[arg(long, default_value = "rekha-data", env = "REKHA_DATA_DIR")]
        data_dir: String,
        /// Listen host.
        #[arg(long, default_value = "0.0.0.0", env = "REKHA_HOST")]
        host: String,
        /// Listen port.
        #[arg(long, default_value_t = 8000, env = "REKHA_PORT")]
        port: u16,
        /// Internal listener host (for Raft/WAL). Defaults to `host`.
        #[arg(long, env = "REKHA_INTERNAL_HOST")]
        internal_host: Option<String>,
        /// Internal listener port. If set, Raft/WAL routes are served on a
        /// separate listener for network isolation. Otherwise they share the
        /// public listener.
        #[arg(long, env = "REKHA_INTERNAL_PORT")]
        internal_port: Option<u16>,
        /// This node's unique id (must be unique across the cluster).
        ///
        /// On first start the id is written to `<data-dir>/node.id`; later
        /// starts must match it. Two nodes sharing an id cannot be recovered
        /// from, so a mismatch is refused rather than warned about.
        #[arg(long, default_value_t = 1, env = "REKHA_NODE_ID")]
        node_id: u64,
        /// Cluster topology as `id@addr` pairs, for example
        /// `1@10.0.0.1:8000,2@10.0.0.2:8000`.
        ///
        /// Every node must be listed, including this one, and the list must be
        /// identical on every node. The address is the node's *internal*
        /// listener, which is where Raft RPCs and WAL deltas go. Empty means
        /// single-node mode. Positional `host:port` entries are rejected: ids
        /// derived from list position renumber the cluster on insert.
        #[arg(long, default_value = "", env = "REKHA_PEERS")]
        peers: String,
        /// Initialize the cluster from this node.
        ///
        /// Exactly one node in a *new* cluster sets this; every other node
        /// starts and waits for a leader to reach it. Initializing from more
        /// than one node lets them disagree about the member set.
        #[arg(long, env = "REKHA_BOOTSTRAP", default_value_t = false)]
        bootstrap: bool,
        /// Join an existing cluster as a learner, by the address of any member.
        ///
        /// The node asks that member for the cluster's placement, so it routes
        /// by the same owner sets as everyone else, but it starts no shard
        /// groups and materializes no collections: hosting comes with promotion.
        /// The leader admits it with `rekha membership add`, and until then it
        /// is a node waiting for a leader.
        ///
        /// `--peers` must be empty. Inventing a voter set here is how a joining
        /// node used to compute a placement none of its peers agreed with.
        #[arg(long, env = "REKHA_JOIN")]
        join: Option<String>,
        /// Raft heartbeat interval in milliseconds.
        #[arg(long, env = "REKHA_HEARTBEAT_MS", default_value_t = 200)]
        heartbeat_ms: u64,
        /// Raft election timeout floor in milliseconds.
        ///
        /// Must be well above the heartbeat interval, or heartbeats cannot
        /// arrive fast enough to prevent elections.
        #[arg(long, env = "REKHA_ELECTION_MIN_MS", default_value_t = 800)]
        election_min_ms: u64,
        /// Raft election timeout ceiling in milliseconds.
        ///
        /// This also sets the leader lease, and a follower waits
        /// `lease + election_timeout` after the last heartbeat before
        /// campaigning, so the failover bound is roughly twice this value.
        #[arg(long, env = "REKHA_ELECTION_MAX_MS", default_value_t = 1600)]
        election_max_ms: u64,
        /// Path to TLS certificate file (PEM).
        #[arg(long, env = "REKHA_TLS_CERT")]
        tls_cert: Option<String>,
        /// Path to TLS private key file (PEM).
        #[arg(long, env = "REKHA_TLS_KEY")]
        tls_key: Option<String>,
        /// Number of placement shards.
        ///
        /// `0` chooses a default for the cluster size (4 shards per node). The
        /// value is persisted in the data directory on first start and cannot
        /// change: it is what collections were hashed by, so a different value
        /// would move data that has not moved. Choose it for the size the
        /// cluster is expected to reach, not its size today.
        #[arg(long, env = "REKHA_SHARDS", default_value_t = 0)]
        shards: u32,
        /// Replicas per shard. Three tolerates one node loss.
        ///
        /// A two-replica group cannot reach quorum with either member down, so
        /// this is a durability setting, not a cost setting.
        #[arg(long, env = "REKHA_REPLICATION_FACTOR", default_value_t = 3)]
        replication_factor: usize,
        /// What a successful write promises about surviving a crash.
        ///
        /// `durable` fsyncs every WAL batch before acknowledging it (default);
        /// `relaxed` acknowledges once the write reaches the OS, so a power
        /// loss can lose recently acknowledged writes.
        #[arg(
            long,
            env = "REKHA_WAL_DURABILITY",
            default_value = "durable",
            value_name = "durable|relaxed"
        )]
        wal_durability: rekha_engine::Durability,
    },
    /// Check a data directory for damage, without modifying it.
    ///
    /// Reports every problem found, not the first: an operator deciding whether
    /// to restore needs the whole list. Exits non-zero if anything would prevent
    /// the node from opening its data.
    Verify {
        /// Data directory to inspect.
        #[arg(long, default_value = "rekha-data", env = "REKHA_DATA_DIR")]
        data_dir: String,
        /// Emit the report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Copy a data directory into a sealed backup.
    ///
    /// Refuses to write into a directory that already holds a backup: the
    /// previous one is the thing you would need if this one were bad.
    Backup {
        /// Data directory to copy.
        #[arg(long, default_value = "rekha-data", env = "REKHA_DATA_DIR")]
        data_dir: String,
        /// Destination directory (must not exist, or be empty).
        #[arg(long)]
        out: String,
    },
    /// Restore a backup into an empty data directory.
    ///
    /// Stages the copy, verifies it against the backup's manifest, and only then
    /// moves it into place. A refused restore leaves the target untouched.
    Restore {
        /// Data directory to create.
        #[arg(long, default_value = "rekha-data", env = "REKHA_DATA_DIR")]
        data_dir: String,
        /// Backup directory produced by `backup`.
        #[arg(long)]
        from: String,
    },
    /// Collection operations.
    Collection {
        #[command(subcommand)]
        action: CollectionAction,
    },
    /// Cluster membership management.
    Membership {
        #[command(subcommand)]
        action: MembershipAction,
    },
    /// RBAC user and role management.
    Auth {
        #[command(subcommand)]
        action: AuthAction,
    },
    /// Add records to a collection.
    Add {
        /// Collection id (UUID).
        collection_id: String,
        /// JSON file with { ids, embeddings, metadatas?, documents? }.
        file: String,
    },
    /// Upsert records in a collection.
    Upsert {
        /// Collection id (UUID).
        collection_id: String,
        /// JSON file with { ids, embeddings, metadatas?, documents? }.
        file: String,
    },
    /// Update record metadata.
    Update {
        /// Collection id (UUID).
        collection_id: String,
        /// JSON file with { ids, metadatas?, documents? }.
        file: String,
    },
    /// Delete records from a collection.
    Delete {
        /// Collection id (UUID).
        collection_id: String,
        /// JSON array of ids: ["id1", "id2"]
        file: String,
    },
    /// Query a collection.
    Query {
        /// Collection id (UUID).
        collection_id: String,
        /// JSON file with { query_embeddings, n_results?, where_filter? }.
        file: String,
    },
    /// Get records from a collection.
    Get {
        /// Collection id (UUID).
        collection_id: String,
        /// JSON file with { ids: ["id1", "id2"] }.
        file: String,
    },
    /// Count records in a collection.
    Count {
        /// Collection id (UUID).
        collection_id: String,
    },
}

#[derive(Subcommand)]
enum CollectionAction {
    /// Create a new collection.
    Create {
        /// Collection name.
        name: String,
        /// Vector dimension.
        dimension: usize,
        /// Distance metric (l2, cosine, ip).
        #[arg(long, default_value = "l2")]
        distance: String,
    },
    /// List all collections.
    List,
    /// Get a collection by id.
    Get {
        /// Collection id (UUID).
        id: String,
    },
    /// Delete a collection.
    Delete {
        /// Collection id (UUID).
        id: String,
    },
}

/// Report a startup or command failure as a sentence, not as a Rust value.
///
/// `Termination for Result` prints the error with `Debug`, which for a typed
/// error is its struct dump: an operator who refused to start because their
/// `--peers` disagreed with the data directory would be shown
/// `VotersDiffer { configured: [1, 2, 3, 4, 5], .. }`. Every error that reaches
/// here already carries a `Display` message written for a human, so this is the
/// one place that decides how it is shown. The exit code is unchanged (1).
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// What this process is.
///
/// Three states rather than a placement and two booleans. `Option<Placement>`
/// used to mean "cluster or single node", which left a node joining an existing
/// cluster with nowhere to land: it either derived a voter set from `--peers`
/// and disagreed with every peer about where shards live, or was mistaken for a
/// single node and opened every collection.
enum Topology {
    /// One process owns everything. There are no shards.
    Single,
    /// A voting member of a sharded cluster, placed across these voters.
    Sharded(std::sync::Arc<rekha_cluster::Placement>),
    /// Told to join. Routes by the cluster's placement, hosts nothing.
    Joining(std::sync::Arc<rekha_cluster::Placement>),
}

impl Topology {
    /// Whether this process takes part in consensus, and therefore whether the
    /// engine writes through the log instead of its own WAL.
    fn is_clustered(&self) -> bool {
        !matches!(self, Self::Single)
    }

    /// Which catalog rows this process materializes.
    ///
    /// A joining node owns nothing until it is promoted, so it materializes
    /// nothing. It still holds the catalog, which is what lets it route and
    /// authorize while it catches up.
    fn engine_scope(&self, hosted: &[rekha_cluster::ShardId]) -> rekha_engine::EngineScope {
        match self {
            Self::Single => rekha_engine::EngineScope::All,
            Self::Sharded(_) | Self::Joining(_) => {
                rekha_engine::EngineScope::Shards(hosted.to_vec())
            }
        }
    }
}

/// Ask a cluster member for the placement this node must route by.
///
/// A joining node has no `--peers`, so the only node it can ask is the one it
/// was told to join. That node answers from its own persisted document, so the
/// joiner ends up with the same three inputs every member has.
///
/// Retried rather than attempted once: `--join` is normally given while the
/// cluster is still starting, and a node that refused to boot because a peer
/// was two seconds from listening would be a poor first experience.
///
/// # Errors
///
/// Returns a message naming the endpoint and the last thing it heard, because
/// "connection refused" from a shell script says nothing about which flag or
/// secret is wrong.
async fn fetch_placement(
    url: &str,
    secret: Option<&str>,
) -> Result<rekha_cluster::Placement, Box<dyn std::error::Error>> {
    let endpoint = format!(
        "{}/internal/cluster/placement",
        url.trim().trim_end_matches('/')
    );
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    // Assigned by every non-successful attempt before it is read below.
    let mut last;
    loop {
        let mut request = client.get(&endpoint);
        if let Some(secret) = secret {
            request = request.header("x-rekha-cluster-secret", secret);
        }
        match request.send().await {
            Ok(response) if response.status().is_success() => {
                let config: rekha_cluster::PlacementConfig = response.json().await?;
                return Ok(rekha_cluster::Placement::new(
                    config.shard_count,
                    config.replication_factor,
                    config.voters,
                )?);
            }
            Ok(response) => last = response.status().to_string(),
            Err(e) => last = e.to_string(),
        }
        if std::time::Instant::now() >= deadline {
            return Err(format!(
                "could not learn the cluster's placement from {endpoint} after 60s ({last}).                  --join takes the address of a running member; check it, and that                  REKHA_CLUSTER_SECRET matches"
            )
            .into());
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
}

/// The voter set a group starts with, before any membership entry is replayed.
///
/// Reconstructing membership from the placement document rather than the log is
/// only sound because the two cannot currently diverge: promotion and
/// `remove_member` are not implemented, so a group's voters are exactly the
/// owners the placement assigned. When membership changes land, this must be
/// replaced by a persisted `StoredMembership`.
fn seeded_membership(
    voters: impl IntoIterator<Item = u64>,
    log_id: Option<openraft::LogId<u64>>,
) -> openraft::StoredMembership<u64, u64> {
    let voters: std::collections::BTreeSet<u64> = voters.into_iter().collect();
    openraft::StoredMembership::new(log_id, openraft::Membership::new(vec![voters], ()))
}

#[tokio::main]
async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let client = Client::new();

    match cli.command {
        Command::Serve {
            data_dir,
            host,
            port,
            internal_host,
            internal_port,
            node_id,
            peers,
            bootstrap,
            join,
            heartbeat_ms,
            election_min_ms,
            election_max_ms,
            tls_cert,
            tls_key,
            shards,
            replication_factor,
            wal_durability,
        } => {
            use std::sync::Arc;

            use rekha_api::{AppState, internal_router, public_router, router};
            use std::sync::atomic::AtomicBool;

            use rekha_cluster::{
                ClusterStateMachine, LeaderTracker, NodeIdentity, PeerDirectory, Placement,
                RaftNetworkFactoryImpl, RedbLogStore, ShardCluster, ShardId, parse_peers,
            };
            use rekha_engine::{Engine, EngineConfig};
            use rekha_storage::{LocalStorage, RedbCatalog};

            tracing_subscriber::fmt::init();

            // Two listeners on one port fail with an `Address already in use`
            // after Raft has started, which leaves a half-initialized data
            // directory and a misleading error. It is a configuration mistake;
            // refuse it before anything opens.
            if internal_port == Some(port) {
                return Err(
                    "--internal-port must differ from --port: the internal listener would \
                     collide with the public one"
                        .into(),
                );
            }

            let wal_dir = std::path::PathBuf::from(&data_dir).join("wal");
            std::fs::create_dir_all(&wal_dir)?;

            // A node id belongs to a data directory. Refusing on mismatch keeps
            // a replaced or cloned disk from campaigning under an id another
            // node already holds.
            let identity = NodeIdentity::open_or_create(std::path::Path::new(&data_dir), node_id)?;
            let node_id = identity.node_id();

            // The pepper is optional. Keys are high-entropy, so a stolen
            // catalog yields hashes that cannot be cracked; the pepper is
            // defence in depth for deployments that want a secret the data
            // directory does not contain.
            let pepper = rekha_core::auth::pepper_from_env();
            let will_cluster = !peers.trim().is_empty() || join.is_some();

            // Internal routes fail closed without a secret. That includes a
            // joining node, whose `--peers` is empty: it would otherwise be
            // unable to ask a member for the placement, and Raft traffic to it
            // would be rejected, so the cluster would silently never form.
            // Validated before the catalog opens so a bad secret cannot leave
            // a half-initialized data directory behind.
            let cluster_secret = std::env::var("REKHA_CLUSTER_SECRET").ok();
            if will_cluster {
                match cluster_secret.as_deref() {
                    None => {
                        return Err(
                            "REKHA_CLUSTER_SECRET must be set on any node that takes part in \
                             consensus; the internal routes refuse unauthenticated calls, so \
                             without it Raft traffic is rejected and the cluster silently never \
                             forms"
                                .into(),
                        );
                    }
                    Some(secret) if secret.len() < 32 => {
                        return Err(format!(
                            "REKHA_CLUSTER_SECRET is {} bytes; use at least 32 random bytes. \
                             The secret authenticates every internal Raft and forwarding call, \
                             so a short value is a guessable cluster-wide credential.",
                            secret.len()
                        )
                        .into());
                    }
                    Some(_) => {}
                }
            }

            let catalog = Arc::new(RedbCatalog::open_with_pepper(
                std::path::PathBuf::from(&data_dir).join("catalog.redb"),
                pepper.clone(),
            )?);
            let storage = Arc::new(LocalStorage::new(
                std::path::PathBuf::from(&data_dir).join("objects"),
            ));
            // Cluster mode gets its durability from the Raft log, so the engine
            // must not also append to its own WAL: that would fsync every write
            // twice. Single-node mode is unchanged. Which mode this is comes
            // from the placement below, not from a flag: the flag is an input.

            // Topology: ids are data, not position.
            let peer_list = parse_peers(&peers)?;
            if !peer_list.is_empty() && !peer_list.iter().any(|p| p.node_id == node_id) {
                return Err(format!(
                    "--peers does not list this node (id {node_id}). Every node must be listed, \
                     including itself, so that the topology string is identical on every node: \
                     {peers}"
                )
                .into());
            }
            // One address book for the process. It is seeded from `--peers` and
            // extended by committed membership entries, and every component
            // that has to reach a node shares it: the Raft network factories,
            // request forwarding, and the state machine that applies the entry.
            let peer_dir = PeerDirectory::new(&peer_list);

            // A join is told where the cluster is by the cluster. Accepting
            // `--peers` here would make this node derive a voter set containing
            // itself, which is a placement none of the existing members share.
            if join.is_some() && !peers.trim().is_empty() {
                return Err(
                    "--join means the cluster tells this node where everyone is, so \
                            --peers must be empty. A joining node has no placement of its own."
                        .into(),
                );
            }
            if join.is_some() && bootstrap {
                return Err(
                    "--join and --bootstrap describe different clusters: a joining node \
                            is admitted by an existing one, not initialising a new one."
                        .into(),
                );
            }

            // Placement is a property of the cluster, not of a start. All three
            // of its inputs — shard count, replication factor, voter set — are
            // persisted in the data directory, so a node restarted with any of
            // them changed is refused instead of silently disagreeing with its
            // peers about where the data lives. `--peers` seeds that document
            // on a first start and is validated against it afterwards.
            let placement = Placement::open(
                std::path::Path::new(&data_dir),
                shards,
                replication_factor,
                &peer_dir.ids(),
            )?;

            let topology = match (join.as_deref(), placement) {
                (Some(_), Some(_)) => {
                    return Err(
                        "this data directory already has a placement, so it is already \
                                part of a cluster. Start it without --join."
                            .into(),
                    );
                }
                (Some(url), None) => {
                    let placement = fetch_placement(url, cluster_secret.as_deref()).await?;
                    Topology::Joining(std::sync::Arc::new(placement))
                }
                (None, Some(placement)) => Topology::Sharded(std::sync::Arc::new(placement)),
                (None, None) => Topology::Single,
            };
            match &topology {
                Topology::Sharded(placement) => println!(
                    "Placement: {} shards x {} replicas across {} nodes",
                    placement.shard_count(),
                    placement.replication_factor(),
                    placement.voters().len()
                ),
                Topology::Joining(placement) => println!(
                    "Node {node_id}: joining {} shards x {} replicas across {} voters as a \
                     learner; it hosts no shards until it is admitted and promoted",
                    placement.shard_count(),
                    placement.replication_factor(),
                    placement.voters().len()
                ),
                Topology::Single => {}
            }

            // The shards this node replicates, computed once and used for both
            // the engine's row filter and the consensus groups it runs. A node
            // that filtered with one set and voted with another would open data
            // it does not own, or refuse data it does.
            let hosted_shards: Vec<ShardId> = match &topology {
                Topology::Sharded(placement) => (0..placement.shard_count())
                    .map(ShardId)
                    .filter(|shard| placement.owns(*shard, node_id))
                    .collect(),
                // A joining node routes but must not host: whether it is
                // destined to own a shard is a decision that comes with
                // promotion, not with reading someone else's placement.
                Topology::Single | Topology::Joining(_) => Vec::new(),
            };
            let config = EngineConfig {
                // Which catalog rows are this node's data.
                node_id,
                scope: topology.engine_scope(&hosted_shards),
                durability: wal_durability,
                apply_mode: if topology.is_clustered() {
                    rekha_engine::ApplyMode::Committed
                } else {
                    rekha_engine::ApplyMode::Durable
                },
                ..EngineConfig::default()
            };
            if !wal_durability.is_durable() {
                eprintln!(
                    "WARN: WAL durability is `{}` — a power loss can discard writes that \
                     were already acknowledged. Use `--wal-durability durable` for production.",
                    wal_durability
                );
            }
            println!("WAL durability: {wal_durability}");
            let apply_mode = config.apply_mode;
            // Kept for the clustered bootstrap below, which asks the concrete
            // catalog "is this store fresh?" before proposing anything.
            let catalog_for_bootstrap = catalog.clone();
            let engine = Arc::new(Engine::open(catalog, storage, &wal_dir, config)?);

            // Built-in roles are materialised from one table on every node, with
            // ids derived from the role name, so the same role is the same row
            // everywhere. They are not log entries — which is exactly why the
            // ids must be deterministic rather than random per node.
            engine.catalog().ensure_builtin_roles()?;
            let env_key = std::env::var("REKHA_API_KEY").ok();
            // The bootstrap key is stored as a fast hash, so it has to be high
            // entropy. Length is a proxy, but it rejects the wordlist values
            // that actual deployments paste in.
            if let Some(key) = env_key.as_deref()
                && key.len() < 32
            {
                return Err(format!(
                    "REKHA_API_KEY is {} bytes; use at least 32 (for example \
                     `openssl rand -hex 16`). The key is stored as a hash, and a short value is \
                     recoverable from a stolen catalog.",
                    key.len()
                )
                .into());
            }
            if topology.is_clustered() && env_key.is_none() {
                return Err(
                    "REKHA_API_KEY must be set on every node in a cluster: without it each \
                     node mints its own superadmin and prints a key that authenticates only \
                     against the node that printed it"
                        .into(),
                );
            }
            // A clustered deployment writes its bootstrap user through the log
            // below, once a leader exists. Doing it here would mint a user on
            // every node, each with a different id, and only the leader's would
            // survive the next snapshot.
            if !topology.is_clustered()
                && let Some((uid, key)) = engine
                    .catalog()
                    .bootstrap_superadmin_if_empty(env_key.clone())?
            {
                println!("BOOTSTRAP: Created superadmin user {uid} with API key: {key}");
                eprintln!("BOOTSTRAP: Save this key — it will not be shown again.");
            }

            // Whether this node already holds the cluster's bootstrap
            // credentials. A fresh cluster starts false and the metadata apply
            // flips it; a restart, or a node restored from a snapshot with
            // users, starts true. `/ready` gates on it so a client cannot
            // authenticate before the role assignment has landed.
            let bootstrap_ready =
                Arc::new(AtomicBool::new(!engine.catalog().list_users()?.is_empty()));

            // Create Raft components.
            let mut log_store =
                RedbLogStore::open(std::path::PathBuf::from(&data_dir).join("raft_log.redb"))?;
            // If this log was previously compacted, the entries that carried
            // the membership and the applied boundary are gone. Both have to be
            // restored before Raft starts, or it asks for a prefix that no
            // longer exists and refuses to start.
            let purge_watermark = log_store.purge_watermark().await;
            let mut sm = ClusterStateMachine::new();
            sm.set_bootstrap_ready_flag(bootstrap_ready.clone());
            // Replicated RBAC/collection operations are applied by the state
            // machine, not the HTTP handlers, so it must hold the catalog or
            // every replicated administrative change silently no-ops.
            sm.set_catalog(engine.catalog());
            sm.set_engine(engine.clone() as Arc<dyn rekha_cluster::EngineSnapshotProvider>);
            // The state machine is what learns a new node's address when the
            // entry commits, so it has to write into the same directory the
            // network factory reads from.
            sm.set_addresses(peer_dir.clone());
            let apply_failure = Arc::new(AtomicBool::new(false));
            sm.set_apply_failure_flag(apply_failure.clone());
            // Where this group's snapshots live. A clustered node must have one:
            // openraft only asks for a snapshot when a follower is behind the
            // purged prefix, and a node that cannot answer that has no way to
            // recover the follower.
            // Under the engine's storage directory (beside the WALs and
            // checkpoints), so the archive and the staging area are on the same
            // filesystem as the data they describe.
            let snapshot_root = wal_dir.join("snapshots");
            sm.set_snapshot_store(Arc::new(rekha_cluster::SnapshotStore::open(
                snapshot_root.join("metadata"),
                sm.group_label(),
            )?));

            // Resume from the compaction boundary. Without this, the state
            // machine reports "applied nothing" and openraft tries to replay
            // the purged prefix.
            //
            // Everything here is gated on an actual purge. On a first start the
            // log is intact and replay reconstructs all of it; seeding it early
            // would give the node a voter set before the cluster has been
            // initialised, which changes election behaviour rather than
            // recovering it.
            if let Some(boundary) = purge_watermark {
                // The maps this machine carries in a snapshot are normally
                // rebuilt by replaying the log. Those entries are gone, so seed
                // them from the sources that are durable regardless: the
                // catalog for placement, the peer list for addresses.
                sm.seed_nodes(
                    peer_dir
                        .ids()
                        .into_iter()
                        .filter_map(|id| peer_dir.addr(id).map(|addr| (id, addr))),
                );
                if let Err(e) = sm.seed_placements(engine.catalog().as_ref()) {
                    // Not fatal: a node that routes by catalog row still serves,
                    // and this only affects what a future snapshot would contain.
                    eprintln!("placement could not be rebuilt from the catalog: {e}");
                }
                let voters: Vec<u64> = match &topology {
                    Topology::Sharded(placement) | Topology::Joining(placement) => {
                        placement.voters().to_vec()
                    }
                    Topology::Single => vec![node_id],
                };
                sm.resume_from(Some(boundary), seeded_membership(voters, Some(boundary)));
                println!(
                    "Resuming after log compaction at index {}: state before it is in \
                     durable storage, not in the log",
                    boundary.index
                );
            }
            let network = RaftNetworkFactoryImpl::new(peer_dir.clone())
                .with_cluster_secret(cluster_secret.clone());
            // Election timing is a durability-adjacent decision, so it is
            // validated rather than trusted. A heartbeat that cannot outpace
            // the election floor produces a cluster that elects under load for
            // no reason; a ceiling below the floor is a config error that
            // openraft would otherwise absorb silently.
            if election_min_ms <= heartbeat_ms * 2 {
                return Err(format!(
                    "--election-min-ms ({election_min_ms}) must exceed twice \
                     --heartbeat-ms ({heartbeat_ms}): heartbeats could not arrive fast \
                     enough to prevent elections"
                )
                .into());
            }
            if election_max_ms < election_min_ms {
                return Err(format!(
                    "--election-max-ms ({election_max_ms}) is below \
                     --election-min-ms ({election_min_ms})"
                )
                .into());
            }

            // Snapshotting is what purges the Raft log, and purging shortens the
            // history a node can replay. Two recoveries have to stay possible:
            //
            // * a node that **restarts** resumes from the purge watermark, since
            //   its applied state is in durable storage;
            // * a node that **fell behind** is repaired by installing a snapshot
            //   the leader persisted and streams in chunks.
            //
            // Both are implemented and tested. The default stays off anyway: with
            // purge enabled, membership is reconstructed from the placement
            // document at the watermark, which is sound only while membership is
            // static — promotion and removal are not implemented yet. An operator
            // who needs the disk bound back can enable it for a cluster that is
            // not being resized.
            let snapshot_every: u64 = match std::env::var("REKHA_SNAPSHOT_EVERY_N_LOGS") {
                Ok(raw) => raw.trim().parse().map_err(|e| {
                    format!(
                        "REKHA_SNAPSHOT_EVERY_N_LOGS must be a whole number of log entries \
                         (0 disables snapshotting): {e}"
                    )
                })?,
                Err(std::env::VarError::NotPresent) => 0,
                Err(e) => return Err(format!("REKHA_SNAPSHOT_EVERY_N_LOGS: {e}").into()),
            };
            let snapshot_policy = if snapshot_every == 0 {
                openraft::SnapshotPolicy::Never
            } else {
                openraft::SnapshotPolicy::LogsSinceLast(snapshot_every)
            };
            match snapshot_policy {
                openraft::SnapshotPolicy::Never => println!(
                    "Raft snapshots: disabled (the log is never purged). Set \
                     REKHA_SNAPSHOT_EVERY_N_LOGS to enable stream compaction."
                ),
                openraft::SnapshotPolicy::LogsSinceLast(n) => {
                    println!(
                        "Raft snapshots: every {n} log entries, keeping {n}; a replica that \
                         falls behind is repaired by a streamed snapshot"
                    );
                    if topology.is_clustered() {
                        eprintln!(
                            "warning: snapshotting is enabled. Membership is reconstructed from \
                             the placement document after a purge, which is correct only while \
                             nodes are not added or removed. Keep --shards and the peer list \
                             fixed for the life of this data directory."
                        );
                    }
                }
            }

            let raft_config = Arc::new(openraft::Config {
                heartbeat_interval: heartbeat_ms,
                election_timeout_min: election_min_ms,
                election_timeout_max: election_max_ms,
                replication_lag_threshold: 10000,
                snapshot_policy,
                // A snapshot is the one RPC whose cost is a function of corpus
                // size, and openraft's default is 200 ms. That value is a
                // sensible bound for a heartbeat and a permanent barrier to
                // recovery for an archive: a lagging replica would time out on
                // every attempt, forever. Ten minutes covers a multi-gigabyte
                // transfer on a slow link; the per-chunk requests still fail
                // fast if the peer is gone.
                install_snapshot_timeout: 10 * 60 * 1000,
                // openraft keeps 1,000 entries *before* a snapshot by default,
                // so a snapshot alone does not purge anything until the log is
                // far past the threshold. The same knob bounds that history:
                // "compact every N, keep N" is one number an operator can reason
                // about, and a small value lets the purge path be tested without
                // writing thousands of entries.
                max_in_snapshot_log_to_keep: snapshot_every.max(1),
                ..Default::default()
            });

            let raft = Arc::new(
                openraft::Raft::<rekha_cluster::RaftTypeConfig>::new(
                    node_id,
                    // Cloned because every shard group below builds its own
                    // `Raft` from the same timing configuration.
                    raft_config.clone(),
                    network,
                    log_store,
                    sm,
                )
                .await
                .map_err(|e| format!("Raft initialization failed: {e}"))?,
            );

            // Only the bootstrap node initializes. Every other node starts as a
            // follower with an empty log and waits for a leader to reach it;
            // letting each node initialize with its own member list is how a
            // cluster ends up with two disagreeing configurations.
            //
            // Single-node mode (no peers) bootstraps itself so the default
            // `rekha serve` still works with no flags. A joining node must not:
            // it is empty on purpose and waits to be admitted by a cluster that
            // already exists.
            let should_bootstrap = bootstrap || !topology.is_clustered();
            if should_bootstrap {
                let raft_clone = raft.clone();
                let members: std::collections::BTreeSet<u64> = peer_dir.ids().into_iter().collect();
                let members = if members.is_empty() {
                    // Single-node mode with no --peers.
                    std::collections::BTreeSet::from([node_id])
                } else {
                    members
                };
                // Where every member lives, to be written into the log below.
                let known: Vec<(u64, String)> = peer_dir
                    .ids()
                    .into_iter()
                    .filter_map(|id| peer_dir.addr(id).map(|addr| (id, addr)))
                    .collect();
                let clustered = topology.is_clustered();
                let env_key_for_bootstrap = env_key.clone();
                tokio::spawn(async move {
                    // initialize() waits for a quorum, which may not be up yet,
                    // so it must not block server startup.
                    match raft_clone.initialize(members).await {
                        Ok(_) => {
                            println!("Node {node_id}: initialized cluster");
                            // The address book has to live in the log, not only
                            // in each node's flags. A node that joins later
                            // replays this log with an empty `--peers`, and it
                            // cannot reach peers it never learned about.
                            for (member, addr) in known {
                                let mut attempts = 0;
                                loop {
                                    let entry = rekha_cluster::ClusterOperation::AddNode {
                                        node_id: member,
                                        addr: addr.clone(),
                                    };
                                    match raft_clone.client_write(entry).await {
                                        Ok(_) => break,
                                        Err(e) if attempts >= 50 => {
                                            eprintln!(
                                                "Node {node_id}: could not record where node \
                                                 {member} lives: {e}. A node joining later will \
                                                 not be able to reach it."
                                            );
                                            break;
                                        }
                                        Err(_) => {
                                            // initialize() returns before this
                                            // node is guaranteed to be the
                                            // leader that has to write.
                                            attempts += 1;
                                            tokio::time::sleep(std::time::Duration::from_millis(
                                                200,
                                            ))
                                            .await;
                                        }
                                    }
                                }
                            }
                            if clustered {
                                bootstrap_superadmin_via_log(
                                    &raft_clone,
                                    &catalog_for_bootstrap,
                                    env_key_for_bootstrap,
                                )
                                .await;
                            }
                        }
                        Err(e) => {
                            let msg = e.to_string();
                            if msg.contains("AlreadyInitialized") || msg.contains("leader") {
                                println!("Node {node_id}: cluster already initialized");
                            } else {
                                eprintln!("Node {node_id}: Raft init error: {e}");
                            }
                        }
                    }
                });
            } else {
                println!(
                    "Node {node_id}: not the bootstrap node; waiting for an elected leader. \
                     If no node was started with --bootstrap, the cluster will never elect one."
                );
            }

            // The single source of truth for leadership, read from consensus.
            let tracker = LeaderTracker::spawn(raft.clone(), peer_dir.clone());

            // One Raft per shard this node replicates. A node that is not an
            // owner has no instance, which makes "this is not my data"
            // structural rather than a check in a handler.
            let shard_cluster = match &topology {
                Topology::Sharded(placement) | Topology::Joining(placement) => {
                    let placement = placement.clone();
                    let log_dir = std::path::PathBuf::from(&data_dir).join("raft_log");
                    std::fs::create_dir_all(&log_dir)?;
                    let mut cluster = ShardCluster::new(placement.clone());
                    for shard in &hosted_shards {
                        let index = shard.0;
                        let mut log_store =
                            RedbLogStore::open(log_dir.join(format!("shard-{index}.redb")))?;
                        let purge_watermark = log_store.purge_watermark().await;
                        let mut shard_sm = ClusterStateMachine::for_shard(node_id, *shard);
                        shard_sm.set_catalog(engine.catalog());
                        shard_sm.set_engine(
                            engine.clone() as Arc<dyn rekha_cluster::EngineSnapshotProvider>
                        );
                        shard_sm.set_apply_failure_flag(apply_failure.clone());
                        let snapshot_root = wal_dir.join("snapshots");
                        shard_sm.set_snapshot_store(Arc::new(rekha_cluster::SnapshotStore::open(
                            snapshot_root.join(format!("shard-{index}")),
                            shard_sm.group_label(),
                        )?));
                        // Same resume logic as the metadata group, and gated the
                        // same way: only a purged log needs its membership and
                        // placement reconstructed rather than replayed.
                        if let Some(boundary) = purge_watermark {
                            shard_sm.seed_nodes(
                                peer_dir
                                    .ids()
                                    .into_iter()
                                    .filter_map(|id| peer_dir.addr(id).map(|addr| (id, addr))),
                            );
                            if let Err(e) = shard_sm.seed_placements(engine.catalog().as_ref()) {
                                eprintln!(
                                    "shard {index}: placement could not be rebuilt from the \
                                     catalog: {e}"
                                );
                            }
                            shard_sm.resume_from(
                                Some(boundary),
                                seeded_membership(
                                    placement.owners(*shard).iter().copied(),
                                    Some(boundary),
                                ),
                            );
                        }
                        let network = RaftNetworkFactoryImpl::new(peer_dir.clone())
                            .with_group(ShardCluster::group_id(*shard))
                            .with_cluster_secret(cluster_secret.clone());
                        let shard_raft = Arc::new(
                            openraft::Raft::<rekha_cluster::RaftTypeConfig>::new(
                                node_id,
                                raft_config.clone(),
                                network,
                                log_store,
                                shard_sm,
                            )
                            .await
                            .map_err(|e| {
                                format!("shard {shard} Raft initialization failed: {e}")
                            })?,
                        );

                        // Exactly one node initializes a group: its first
                        // owner. Every other owner starts uninitialized and is
                        // reached by the leader's append-entries once the group
                        // exists.
                        let owners = placement.owners(*shard);
                        if owners.first() == Some(&node_id) {
                            let raft = shard_raft.clone();
                            let members: std::collections::BTreeSet<u64> =
                                owners.iter().copied().collect();
                            tokio::spawn(async move {
                                // initialize() waits for a quorum, and the other
                                // owners may not be listening yet, so this cannot
                                // block startup. A group that is never
                                // initialized holds no data and never will, which
                                // is why it retries rather than giving up.
                                loop {
                                    match raft.initialize(members.clone()).await {
                                        Ok(()) => break,
                                        Err(e) if e.to_string().contains("AlreadyInitialized") => {
                                            break;
                                        }
                                        Err(e) => {
                                            eprintln!("shard {index} group init: {e}; retrying");
                                            tokio::time::sleep(std::time::Duration::from_millis(
                                                500,
                                            ))
                                            .await;
                                        }
                                    }
                                }
                            });
                        }
                        cluster.insert(*shard, shard_raft);
                    }
                    let hosted = cluster.shards();
                    println!(
                        "Shard groups: this node hosts {} of {} {hosted:?}",
                        hosted.len(),
                        placement.shard_count()
                    );
                    Some(Arc::new(cluster))
                }
                Topology::Single => None,
            };

            let api_key = env_key;
            if api_key.is_none() {
                // Only warn if RBAC has no users either — otherwise RBAC handles auth.
                let has_users = engine
                    .catalog()
                    .list_users()
                    .map(|u| !u.is_empty())
                    .unwrap_or(false);
                if !has_users {
                    eprintln!(
                        "WARN: REKHA_API_KEY is not set and no RBAC users exist — API is running without authentication."
                    );
                }
            }
            // The data path and the engine's durability model must agree. If
            // they do not, writes are acknowledged by one half and refused by
            // the other, which reads to a client as success and to a reader as
            // an empty collection.
            let data_via_log = topology.is_clustered();
            let engine_is_committed = apply_mode == rekha_engine::ApplyMode::Committed;
            if data_via_log != engine_is_committed {
                return Err(format!(
                    "inconsistent write path: data_via_log={data_via_log} but the engine is \
                     configured with apply_mode={:?}. Clustered deployments must use \
                     ApplyMode::Committed; single-node must use ApplyMode::Durable",
                    apply_mode
                )
                .into());
            }
            println!(
                "Write path: {}",
                if data_via_log {
                    "Raft log (cluster)"
                } else {
                    "local WAL (single node)"
                }
            );

            let allow_open = host == "127.0.0.1" || host == "localhost";
            let state = AppState::new(
                engine.clone(),
                Some(raft.clone()),
                rekha_api::ServerConfig {
                    tenant: "default_tenant".into(),
                    database: "default_database".into(),
                    api_key,
                    cluster_secret: cluster_secret.clone(),
                    pepper,
                    allow_open_mode: allow_open,
                },
            )
            .with_peer_directory(peer_dir.clone())
            .with_leader_tracker(tracker.clone())
            .with_apply_failure_flag(apply_failure.clone())
            .with_bootstrap_ready_flag(bootstrap_ready.clone())
            .with_data_via_log(topology.is_clustered());
            let state = match shard_cluster.clone() {
                Some(cluster) => state.with_shards(cluster),
                None => state,
            };
            let state = Arc::new(state);

            // Data reaches this node as committed log entries, so there is no
            // replication loop to run. What remains is housekeeping: a
            // checkpoint is no longer durability in cluster mode — the log is —
            // so it moves off the write path entirely and runs here instead.
            // That removes the checkpoint from the write tail, which is what
            // made p99 four hundred milliseconds while p50 was twenty.
            if topology.is_clustered() {
                let engine_clone = engine.clone();
                tokio::spawn(async move {
                    loop {
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        let engine = engine_clone.clone();
                        match tokio::task::spawn_blocking(move || engine.checkpoint_if_due()).await
                        {
                            Ok(Ok(0)) => {}
                            Ok(Ok(n)) => println!("checkpoint: {n} collection(s)"),
                            Ok(Err(e)) => eprintln!("background checkpoint failed: {e}"),
                            Err(e) => eprintln!("background checkpoint task failed: {e}"),
                        }
                    }
                });
            }

            if !peers.is_empty() {
                println!("Peers: {peers}");
            }

            // Initialize the metrics exporter if configured.
            //
            // Loopback only. The exporter's own listener cannot authenticate a
            // scrape, and this endpoint reports cluster topology (term, applied
            // index, quorum age) that a caller should not have to guess access
            // to. An operator scraping from another pod uses a sidecar, a
            // service mesh, or the pod network namespace; there is deliberately
            // no flag that puts an unauthenticated endpoint on every interface.
            if let Ok(metrics_port) = std::env::var("REKHA_METRICS_PORT") {
                if let Ok(host) = std::env::var("REKHA_METRICS_HOST") {
                    let parsed: std::net::IpAddr = host.parse().map_err(|_| {
                        format!("REKHA_METRICS_HOST must be an IP address, got `{host}`")
                    })?;
                    if !parsed.is_loopback() {
                        return Err(format!(
                            "REKHA_METRICS_HOST=`{host}` is not a loopback address, and the \
                             metrics listener cannot authenticate scrapes. Bind 127.0.0.1 and \
                             scrape through a sidecar or reverse proxy instead."
                        )
                        .into());
                    }
                }
                let metrics_addr: std::net::SocketAddr =
                    format!("127.0.0.1:{metrics_port}").parse()?;
                metrics_exporter_prometheus::PrometheusBuilder::new()
                    .with_http_listener(metrics_addr)
                    .install()?;
                println!("Metrics available at http://{metrics_addr}/metrics (loopback only)");
            }

            // After the exporter is installed: readings taken before then are
            // dropped, and a gauge that is never set afterwards never registers
            // with the recorder at all.
            spawn_raft_metrics(raft.clone());
            spawn_raft_log_size_metrics(std::path::PathBuf::from(&data_dir));

            // If --internal-port is set, run separate listeners for public and
            // internal routes. Otherwise use a single combined router.
            if let Some(internal_port) = internal_port {
                let ihost = internal_host.unwrap_or_else(|| host.clone());
                let public_app = public_router(state.clone());
                let internal_app = internal_router(state.clone());
                let public_addr = format!("{host}:{port}");
                let internal_addr = format!("{ihost}:{internal_port}");

                if let (Some(cert), Some(key)) = (&tls_cert, &tls_key) {
                    // TLS path: serve both listeners via axum-server.
                    let config =
                        axum_server::tls_rustls::RustlsConfig::from_pem_file(cert, key).await?;
                    // Clone config for the second listener (RustlsConfig is not Clone
                    // in older versions — reload from the same files).
                    let config2 =
                        axum_server::tls_rustls::RustlsConfig::from_pem_file(cert, key).await?;
                    let handle = axum_server::Handle::<std::net::SocketAddr>::new();
                    let handle_clone = handle.clone();
                    let handle2 = handle.clone();
                    tokio::spawn(async move {
                        shutdown_signal().await;
                        handle_clone.graceful_shutdown(None);
                        handle2.graceful_shutdown(None);
                    });
                    println!(
                        "RekhaDB node {node_id} public TLS on {public_addr}, internal TLS on {internal_addr}"
                    );
                    let h1 = handle.clone();
                    let public_fut = axum_server::bind_rustls(public_addr.parse()?, config)
                        .handle(h1)
                        .serve(public_app.into_make_service());
                    let h2 = handle.clone();
                    let internal_fut = axum_server::bind_rustls(internal_addr.parse()?, config2)
                        .handle(h2)
                        .serve(internal_app.into_make_service());
                    tokio::try_join!(public_fut, internal_fut)?;
                } else {
                    println!(
                        "RekhaDB node {node_id} public on {public_addr}, internal on {internal_addr}"
                    );
                    let pub_listener = tokio::net::TcpListener::bind(&public_addr).await?;
                    let int_listener = tokio::net::TcpListener::bind(&internal_addr).await?;
                    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
                    tokio::spawn(async move {
                        shutdown_signal().await;
                        let _ = shutdown_tx.send(true);
                    });
                    let pub_serve = axum::serve(pub_listener, public_app)
                        .with_graceful_shutdown(wait_for_shutdown(shutdown_rx.clone()));
                    let int_serve = axum::serve(int_listener, internal_app)
                        .with_graceful_shutdown(wait_for_shutdown(shutdown_rx));
                    tokio::try_join!(pub_serve, int_serve)?;
                }
            } else {
                let app = router(state);
                let addr = format!("{host}:{port}");
                if let (Some(cert), Some(key)) = (&tls_cert, &tls_key) {
                    let config =
                        axum_server::tls_rustls::RustlsConfig::from_pem_file(cert, key).await?;
                    println!("RekhaDB node {node_id} listening on {addr} (TLS)");
                    let handle = axum_server::Handle::<std::net::SocketAddr>::new();
                    let handle_clone = handle.clone();
                    tokio::spawn(async move {
                        shutdown_signal().await;
                        handle_clone.graceful_shutdown(None);
                    });
                    axum_server::bind_rustls(addr.parse()?, config)
                        .handle(handle)
                        .serve(app.into_make_service())
                        .await?;
                } else {
                    println!("RekhaDB node {node_id} listening on {addr}");
                    let listener = tokio::net::TcpListener::bind(&addr).await?;
                    axum::serve(listener, app)
                        .with_graceful_shutdown(shutdown_signal())
                        .await?;
                }
            }

            // Listeners have drained, so no new writes can arrive. Stopping the
            // Raft task now persists the log and vote state through its own
            // storage rather than leaving it to process teardown.
            if let Err(e) = raft.shutdown().await {
                eprintln!("raft shutdown: {e}");
            }
            if let Some(cluster) = shard_cluster.as_ref() {
                for shard in cluster.shards() {
                    if let Some(shard_raft) = cluster.raft(shard)
                        && let Err(e) = shard_raft.shutdown().await
                    {
                        eprintln!("shard {shard} shutdown: {e}");
                    }
                }
            }
            println!("Node {node_id} stopped");
        }

        Command::Verify { data_dir, json } => {
            use rekha_engine::verify::{Severity, verify_data_dir};

            let report = verify_data_dir(std::path::Path::new(&data_dir))?;
            let errors = report.count(Severity::Error);
            let warnings = report.count(Severity::Warning);

            if json {
                let findings: Vec<serde_json::Value> = report
                    .findings
                    .iter()
                    .map(|f| {
                        serde_json::json!({
                            "collection": f.scope,
                            "class": f.class.as_str(),
                            "severity": match f.class.severity() {
                                Severity::Error => "error",
                                Severity::Warning => "warning",
                            },
                            "detail": f.detail,
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::json!({
                        "data_dir": data_dir,
                        "collections": report.collections,
                        "wal_records": report.wal_records,
                        "errors": errors,
                        "warnings": warnings,
                        "clean": report.is_clean(),
                        "findings": findings,
                    })
                );
            } else {
                for finding in &report.findings {
                    let severity = match finding.class.severity() {
                        Severity::Error => "ERROR",
                        Severity::Warning => "WARN ",
                    };
                    println!(
                        "{severity} {:<22} {}: {}",
                        finding.class.as_str(),
                        finding.scope,
                        finding.detail
                    );
                }
                println!(
                    "{} collections, {} WAL records, {errors} error(s), {warnings} warning(s)",
                    report.collections, report.wal_records
                );
                if report.is_clean() {
                    println!("clean");
                }
            }

            if !report.is_clean() {
                std::process::exit(1);
            }
        }

        Command::Backup { data_dir, out } => {
            match rekha_engine::backup::create_backup(
                std::path::Path::new(&data_dir),
                std::path::Path::new(&out),
            ) {
                Ok(report) => println!(
                    "backup: {} files, {} bytes -> {out}",
                    report.files, report.bytes
                ),
                Err(e) => {
                    eprintln!("backup failed: {e}");
                    std::process::exit(1);
                }
            }
        }

        Command::Restore { data_dir, from } => {
            match rekha_engine::backup::restore_backup(
                std::path::Path::new(&from),
                std::path::Path::new(&data_dir),
            ) {
                Ok(report) => println!(
                    "restore: {} files, {} bytes -> {data_dir}",
                    report.files, report.bytes
                ),
                Err(e) => {
                    eprintln!("restore failed: {e}");
                    std::process::exit(1);
                }
            }
        }

        Command::Collection { action } => match action {
            CollectionAction::Create {
                name,
                dimension,
                distance,
            } => {
                let body = serde_json::json!({
                    "name": name,
                    "dimension": dimension,
                    "distance": distance,
                });
                let resp = client
                    .post(format!("{}/api/v2/collections", cli.url))
                    .json(&body)
                    .send()
                    .await?;
                print_response(resp).await?;
            }
            CollectionAction::List => {
                let resp = client
                    .get(format!("{}/api/v2/collections", cli.url))
                    .send()
                    .await?;
                print_response(resp).await?;
            }
            CollectionAction::Get { id } => {
                let resp = client
                    .get(format!("{}/api/v2/collections/{id}", cli.url))
                    .send()
                    .await?;
                print_response(resp).await?;
            }
            CollectionAction::Delete { id } => {
                let resp = client
                    .delete(format!("{}/api/v2/collections/{id}", cli.url))
                    .send()
                    .await?;
                print_response(resp).await?;
            }
        },

        Command::Add {
            collection_id,
            file,
        } => {
            let body = std::fs::read_to_string(&file)?;
            let body: Value = serde_json::from_str(&body)?;
            let resp = client
                .post(format!(
                    "{}/api/v2/collections/{collection_id}/add",
                    cli.url
                ))
                .json(&body)
                .send()
                .await?;
            print_response(resp).await?;
        }

        Command::Upsert {
            collection_id,
            file,
        } => {
            let body = std::fs::read_to_string(&file)?;
            let body: Value = serde_json::from_str(&body)?;
            let resp = client
                .post(format!(
                    "{}/api/v2/collections/{collection_id}/upsert",
                    cli.url
                ))
                .json(&body)
                .send()
                .await?;
            print_response(resp).await?;
        }

        Command::Update {
            collection_id,
            file,
        } => {
            let body = std::fs::read_to_string(&file)?;
            let body: Value = serde_json::from_str(&body)?;
            let resp = client
                .post(format!(
                    "{}/api/v2/collections/{collection_id}/update",
                    cli.url
                ))
                .json(&body)
                .send()
                .await?;
            print_response(resp).await?;
        }

        Command::Delete {
            collection_id,
            file,
        } => {
            let body = std::fs::read_to_string(&file)?;
            let body: Value = serde_json::from_str(&body)?;
            let resp = client
                .post(format!(
                    "{}/api/v2/collections/{collection_id}/delete",
                    cli.url
                ))
                .json(&body)
                .send()
                .await?;
            print_response(resp).await?;
        }

        Command::Query {
            collection_id,
            file,
        } => {
            let body = std::fs::read_to_string(&file)?;
            let body: Value = serde_json::from_str(&body)?;
            let resp = client
                .post(format!(
                    "{}/api/v2/collections/{collection_id}/query",
                    cli.url
                ))
                .json(&body)
                .send()
                .await?;
            print_response(resp).await?;
        }

        Command::Get {
            collection_id,
            file,
        } => {
            let body = std::fs::read_to_string(&file)?;
            let body: Value = serde_json::from_str(&body)?;
            let resp = client
                .post(format!(
                    "{}/api/v2/collections/{collection_id}/get",
                    cli.url
                ))
                .json(&body)
                .send()
                .await?;
            print_response(resp).await?;
        }

        Command::Count { collection_id } => {
            let resp = client
                .post(format!(
                    "{}/api/v2/collections/{collection_id}/count",
                    cli.url
                ))
                .send()
                .await?;
            print_response(resp).await?;
        }

        Command::Membership { action } => {
            // Membership lives on the internal listener and drives Raft, so it
            // is behind the cluster secret like every other internal route. A
            // command that omitted it would be rejected as unauthenticated on
            // exactly the deployments where the cluster is configured
            // correctly.
            let secret = std::env::var("REKHA_CLUSTER_SECRET").ok();
            let request = |builder: reqwest::RequestBuilder| match secret.as_deref() {
                Some(secret) => builder.header("x-rekha-cluster-secret", secret),
                None => builder,
            };
            match action {
                MembershipAction::List => {
                    let resp =
                        request(client.get(format!("{}/internal/cluster/membership", cli.url)))
                            .send()
                            .await?;
                    print_response(resp).await?;
                }
                MembershipAction::Add { node_id, addr } => {
                    let body = serde_json::json!({ "node_id": node_id, "addr": addr });
                    let resp =
                        request(client.post(format!("{}/internal/cluster/add_learner", cli.url)))
                            .json(&body)
                            .send()
                            .await?;
                    print_response(resp).await?;
                }
                MembershipAction::Remove { node_id } => {
                    let body = serde_json::json!({ "node_id": node_id });
                    let resp =
                        request(client.post(format!("{}/internal/cluster/remove_member", cli.url)))
                            .json(&body)
                            .send()
                            .await?;
                    print_response(resp).await?;
                }
            }
        }

        Command::Auth { action } => {
            // Attach auth header if REKHA_API_KEY is set
            let auth_header = std::env::var("REKHA_API_KEY").ok();
            let mut req_builder;
            match action {
                AuthAction::CreateUser { name } => {
                    let body = serde_json::json!({ "name": name });
                    req_builder = client.post(format!("{}/api/v2/auth/users", cli.url));
                    if let Some(ref key) = auth_header {
                        req_builder = req_builder.header("x-api-key", key);
                    }
                    let resp = req_builder.json(&body).send().await?;
                    print_response(resp).await?;
                }
                AuthAction::ListUsers => {
                    req_builder = client.get(format!("{}/api/v2/auth/users", cli.url));
                    if let Some(ref key) = auth_header {
                        req_builder = req_builder.header("x-api-key", key);
                    }
                    let resp = req_builder.send().await?;
                    print_response(resp).await?;
                }
                AuthAction::DeleteUser { id } => {
                    req_builder = client.delete(format!("{}/api/v2/auth/users/{id}", cli.url));
                    if let Some(ref key) = auth_header {
                        req_builder = req_builder.header("x-api-key", key);
                    }
                    let resp = req_builder.send().await?;
                    print_response(resp).await?;
                }
                AuthAction::CreateKey { user_id, label } => {
                    let body = serde_json::json!({ "name": label });
                    req_builder =
                        client.post(format!("{}/api/v2/auth/users/{user_id}/keys", cli.url));
                    if let Some(ref key) = auth_header {
                        req_builder = req_builder.header("x-api-key", key);
                    }
                    let resp = req_builder.json(&body).send().await?;
                    print_response(resp).await?;
                }
                AuthAction::ListKeys { user_id } => {
                    req_builder =
                        client.get(format!("{}/api/v2/auth/users/{user_id}/keys", cli.url));
                    if let Some(ref key) = auth_header {
                        req_builder = req_builder.header("x-api-key", key);
                    }
                    let resp = req_builder.send().await?;
                    print_response(resp).await?;
                }
                AuthAction::CreateRole { name, description } => {
                    let body = serde_json::json!({ "name": name, "description": description });
                    req_builder = client.post(format!("{}/api/v2/auth/roles", cli.url));
                    if let Some(ref key) = auth_header {
                        req_builder = req_builder.header("x-api-key", key);
                    }
                    let resp = req_builder.json(&body).send().await?;
                    print_response(resp).await?;
                }
                AuthAction::ListRoles => {
                    req_builder = client.get(format!("{}/api/v2/auth/roles", cli.url));
                    if let Some(ref key) = auth_header {
                        req_builder = req_builder.header("x-api-key", key);
                    }
                    let resp = req_builder.send().await?;
                    print_response(resp).await?;
                }
                AuthAction::DeleteRole { id } => {
                    req_builder = client.delete(format!("{}/api/v2/auth/roles/{id}", cli.url));
                    if let Some(ref key) = auth_header {
                        req_builder = req_builder.header("x-api-key", key);
                    }
                    let resp = req_builder.send().await?;
                    print_response(resp).await?;
                }
                AuthAction::AddPermission {
                    role_id,
                    action: perm_action,
                    tenant,
                    database,
                } => {
                    let mut body = serde_json::json!({ "action": perm_action });
                    if let Some(t) = tenant {
                        body["tenant"] = serde_json::Value::String(t);
                    }
                    if let Some(d) = database {
                        body["database"] = serde_json::Value::String(d);
                    }
                    req_builder = client.post(format!(
                        "{}/api/v2/auth/roles/{role_id}/permissions",
                        cli.url
                    ));
                    if let Some(ref key) = auth_header {
                        req_builder = req_builder.header("x-api-key", key);
                    }
                    let resp = req_builder.json(&body).send().await?;
                    print_response(resp).await?;
                }
                AuthAction::AssignRole { user_id, role_id } => {
                    let body = serde_json::json!({ "role_id": role_id });
                    req_builder =
                        client.post(format!("{}/api/v2/auth/users/{user_id}/roles", cli.url));
                    if let Some(ref key) = auth_header {
                        req_builder = req_builder.header("x-api-key", key);
                    }
                    let resp = req_builder.json(&body).send().await?;
                    print_response(resp).await?;
                }
                AuthAction::UnassignRole { user_id, role_id } => {
                    req_builder = client.delete(format!(
                        "{}/api/v2/auth/users/{user_id}/roles/{role_id}",
                        cli.url
                    ));
                    if let Some(ref key) = auth_header {
                        req_builder = req_builder.header("x-api-key", key);
                    }
                    let resp = req_builder.send().await?;
                    print_response(resp).await?;
                }
                AuthAction::Whoami => {
                    req_builder = client.get(format!("{}/api/v2/auth/whoami", cli.url));
                    if let Some(ref key) = auth_header {
                        req_builder = req_builder.header("x-api-key", key);
                    }
                    let resp = req_builder.send().await?;
                    print_response(resp).await?;
                }
            }
        }
    }

    Ok(())
}

async fn print_response(resp: reqwest::Response) -> Result<(), Box<dyn std::error::Error>> {
    let status = resp.status();
    let body = resp.text().await?;
    if !status.is_success() {
        eprintln!("Error ({status}): {body}");
        std::process::exit(1);
    }
    match serde_json::from_str::<Value>(&body) {
        Ok(v) => println!("{}", serde_json::to_string_pretty(&v)?),
        Err(_) => println!("{body}"),
    }
    Ok(())
}

/// Publish the consensus view as Prometheus gauges.
///
/// Without this the cluster is unobservable: `/ready` reports the same numbers
/// but nothing scrapes it, so there is no history of who led, when terms
/// changed, whether applies were keeping up, or how far behind a replica was.
/// Those are the signals an election problem shows up in first.
/// Create the bootstrap superadmin as a log entry, once this node leads.
///
/// Every node has to end up with the same `admin` row and the same key hash. A
/// row written locally on each node gives every node a different user id and,
/// without `REKHA_API_KEY`, a different key — and the key printed by one node
/// then authenticates only against that node, which is the failure this
/// replaces.
async fn bootstrap_superadmin_via_log(
    raft: &std::sync::Arc<openraft::Raft<rekha_cluster::RaftTypeConfig>>,
    catalog: &rekha_storage::RedbCatalog,
    env_key: Option<String>,
) {
    let pending = match catalog.pending_superadmin(env_key) {
        Ok(Some(pending)) => pending,
        // A store that already has a user is not fresh. A read that failed is
        // reported instead of guessed at, and the next restart retries it.
        Ok(None) => return,
        Err(e) => {
            eprintln!(
                "BOOTSTRAP: could not read the catalog to decide whether a superadmin is \
                 needed: {e}"
            );
            return;
        }
    };
    let user_id = pending.user.id.clone();
    let role_id = rekha_core::auth::builtin_role_id(rekha_core::auth::ROLE_SUPERADMIN);
    // A background task, so waiting is cheap. The bound exists to report a
    // cluster that never elects, not to time out a slow election: 50 attempts
    // (10s) was measurably too short under a loaded test run, and the failure it
    // produced was "the cluster has no users" rather than "the election never
    // finished".
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while std::time::Instant::now() < deadline {
        // One entry: user, key and role. See `BootstrapSuperadmin`.
        let op = rekha_cluster::ClusterOperation::BootstrapSuperadmin {
            user: pending.user.clone(),
            key_hash: pending.key_hash,
            key_label: "bootstrap".to_string(),
            role_id: role_id.clone(),
        };
        if raft.client_write(op).await.is_ok() {
            println!(
                "BOOTSTRAP: created superadmin {user_id} in the log; the REKHA_API_KEY from the \
                 environment authenticates against every node"
            );
            return;
        }
        // initialize() returns before this node is guaranteed to be the leader.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    eprintln!(
        "BOOTSTRAP: could not create the superadmin in the log; the metadata group never \
         accepted the write. The cluster serves with an empty user list until one exists."
    );
}

fn spawn_raft_metrics(raft: std::sync::Arc<openraft::Raft<rekha_cluster::RaftTypeConfig>>) {
    tokio::spawn(async move {
        let mut metrics = raft.metrics();
        loop {
            {
                let m = metrics.borrow_and_update();
                let last_log = m.last_log_index.unwrap_or(0);
                let applied = m.last_applied.map(|id| id.index).unwrap_or(0);

                metrics::gauge!("rekha_raft_term").set(m.current_term as f64);
                metrics::gauge!("rekha_raft_is_leader").set(if m.state.is_leader() {
                    1.0
                } else {
                    0.0
                });
                metrics::gauge!("rekha_raft_leader_id")
                    .set(m.current_leader.map(|id| id as f64).unwrap_or(-1.0));
                metrics::gauge!("rekha_raft_last_log_index").set(last_log as f64);
                metrics::gauge!("rekha_raft_last_applied_index").set(applied as f64);
                metrics::gauge!("rekha_raft_apply_lag_entries")
                    .set(last_log.saturating_sub(applied) as f64);
                metrics::gauge!("rekha_raft_quorum_age_ms")
                    .set(m.millis_since_quorum_ack.unwrap_or(0) as f64);
                metrics::gauge!("rekha_raft_voter_count")
                    .set(m.membership_config.membership().voter_ids().count() as f64);

                // Per-replica catch-up, which is the number that predicts a
                // failed election: a follower far behind cannot win one quickly.
                if let Some(replication) = &m.replication {
                    for (node, matched) in replication {
                        let matched_index = matched.map(|id| id.index).unwrap_or(0);
                        metrics::gauge!(
                            "rekha_raft_replication_lag_entries",
                            "node" => node.to_string()
                        )
                        .set(last_log.saturating_sub(matched_index) as f64);
                    }
                }
            }
            if metrics.changed().await.is_err() {
                break;
            }
        }
    });
}

/// Publish the on-disk size of every Raft log this node holds.
///
/// Snapshotting is off by default, so a log grows with every metadata
/// operation and every write to a hosted shard, and nothing truncates it. That
/// is a deliberate trade (see the snapshot policy in `serve`), and it is only
/// acceptable if the growth is observable: an unmeasured log becomes a disk-full
/// outage, and the failure mode of a full disk here is worse than a slow one.
fn spawn_raft_log_size_metrics(data_dir: std::path::PathBuf) {
    tokio::spawn(async move {
        let metadata_log = data_dir.join("raft_log.redb");
        let shard_logs = data_dir.join("raft_log");
        loop {
            if let Ok(metadata) = std::fs::metadata(&metadata_log) {
                metrics::gauge!("rekha_raft_log_bytes", "group" => "metadata")
                    .set(metadata.len() as f64);
            }
            if let Ok(entries) = std::fs::read_dir(&shard_logs) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let Some(rest) = name.to_str().and_then(|n| n.strip_prefix("shard-")) else {
                        continue;
                    };
                    let Some(shard) = rest.strip_suffix(".redb") else {
                        continue;
                    };
                    if let Ok(metadata) = entry.metadata() {
                        metrics::gauge!("rekha_raft_log_bytes", "group" => format!("shard-{shard}"))
                            .set(metadata.len() as f64);
                    }
                }
            }
            tokio::time::sleep(std::time::Duration::from_secs(15)).await;
        }
    });
}

/// Resolve on SIGINT or SIGTERM, then log the shutdown.
///
/// `ctrl_c()` alone only catches SIGINT. Containers stop with SIGTERM — `docker
/// stop`, a Kubernetes rollout, `systemctl stop` — and the default action for
/// SIGTERM is to terminate immediately: no drain, in-flight requests dropped
/// mid-write, and an election triggered by a node that vanished without saying
/// goodbye.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut terminate) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = terminate.recv() => {}
                }
            }
            Err(e) => {
                eprintln!(
                    "WARN: could not install a SIGTERM handler ({e}); stopping on SIGINT only"
                );
                tokio::signal::ctrl_c().await.ok();
            }
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await.ok();
    }
    println!("Shutting down...");
}

/// Resolve once `rx` reports shutdown.
async fn wait_for_shutdown(mut rx: tokio::sync::watch::Receiver<bool>) {
    while !*rx.borrow_and_update() {
        if rx.changed().await.is_err() {
            break;
        }
    }
}
