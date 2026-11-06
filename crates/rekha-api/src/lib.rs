#![deny(clippy::unwrap_used, clippy::expect_used)]

//! RekhaDB HTTP API — Chroma-compatible REST server.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use openraft::raft::{AppendEntriesRequest, AppendEntriesResponse, VoteRequest, VoteResponse};
use rekha_cluster::RaftTypeConfig;
use rekha_cluster::raft_types::ClusterOperation;
use rekha_core::config::CollectionConfig;
use rekha_core::filter::WhereFilter;
use rekha_core::types::{Distance, Embedding, Id, Metadata};
use rekha_engine::{Engine, EngineError, QueryOptions};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod auth_routes;
pub mod chroma;
pub mod leadership;

pub mod middleware {
    use axum::http::{HeaderMap, Request, StatusCode};
    use axum::middleware::Next;
    use axum::response::Response;
    use subtle::ConstantTimeEq;

    use rekha_core::auth::{RouteAction, has_permission, hash_api_key_with_pepper, route_action};

    /// Authenticated request context attached to request extensions after successful auth.
    #[derive(Debug, Clone)]
    pub struct AuthContext {
        pub user_id: String,
        pub user_name: String,
        pub permissions: Vec<rekha_core::auth::Permission>,
    }

    /// The identity the configured shared key authenticates as.
    ///
    /// Built in one place because two paths reach it — the legacy single-key
    /// deployment, and the window in which a clustered node has applied the
    /// bootstrap user but not yet its key row. Two constructions would be two
    /// permission sets that can drift.
    fn shared_key_context() -> AuthContext {
        AuthContext {
            user_id: "legacy".into(),
            user_name: "legacy".into(),
            permissions: vec![rekha_core::auth::Permission {
                id: "legacy-wide".into(),
                action: "*".into(),
                tenant: None,
                database: None,
            }],
        }
    }

    /// Public RBAC auth middleware.
    ///
    /// 1. Extract API key from `x-chroma-token` / `x-api-key` / `Authorization: Bearer`.
    /// 2. If RBAC users exist, require a valid RBAC key (lookup via catalog).
    /// 3. Else if legacy `api_key` is set, check via constant-time compare and treat as superadmin.
    /// 4. Else (no RBAC, no legacy key) allow through (bootstrap / open mode).
    /// 5. For matched routes, check `route_action` against the caller's permissions.
    pub async fn auth(
        mut request: Request<axum::body::Body>,
        next: Next,
    ) -> Result<Response, StatusCode> {
        let state = request
            .extensions()
            .get::<std::sync::Arc<super::AppState>>()
            .cloned()
            .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

        let path = request.uri().path().to_owned();
        let method = request.method().to_string();
        let is_public = matches!(
            path.as_str(),
            "/health"
                | "/ready"
                | "/api/v2/heartbeat"
                | "/api/v2/version"
                | "/api/v2/pre-flight-checks"
        );
        if is_public {
            return Ok(next.run(request).await);
        }

        let headers: HeaderMap = request.headers().clone();
        let provided = headers
            .get("x-chroma-token")
            .or_else(|| headers.get("x-api-key"))
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .or_else(|| {
                headers
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| {
                        if s.to_ascii_lowercase().starts_with("bearer ") {
                            Some(s[7..].trim().to_string())
                        } else {
                            None
                        }
                    })
            });

        let catalog = state.engine.catalog();
        // A failed read is not "there are no users". Treating it as one fell
        // through to the legacy shared-key path, so a storage error was a
        // fail-open. Ask once, and refuse the request if the answer cannot be
        // read.
        let users_exist = match catalog.list_users() {
            Ok(users) => !users.is_empty(),
            Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
        };

        let auth_ctx = if users_exist {
            let token = provided.ok_or(StatusCode::UNAUTHORIZED)?;
            let hash = hash_api_key_with_pepper(&token, &state.pepper);
            // Deliberately no permission cache. A cache here has to be
            // invalidated on every user, key and role change -- and those
            // changes are log entries that apply on every node -- while the
            // version this replaces was invalidated nowhere: a deleted user kept
            // working until the entry's TTL expired. The reads it saved are
            // in-process table lookups on the store this request already reads
            // to check collection access.
            let looked_up = catalog
                .lookup_api_key(&hash)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            match looked_up {
                Some((user_id, _meta)) => {
                    let user = catalog
                        .get_user(&user_id)
                        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
                        .ok_or(StatusCode::UNAUTHORIZED)?;
                    if !user.enabled {
                        return Err(StatusCode::UNAUTHORIZED);
                    }
                    let perms = catalog
                        .get_user_permissions(&user_id)
                        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                    AuthContext {
                        user_id: user.id.clone(),
                        user_name: user.name.clone(),
                        permissions: perms,
                    }
                }
                // One apply writes the bootstrap user, its key and its role, so a
                // reader can arrive after the user row lands and before the key
                // row does. The honest answer for that instant is not "401 for a
                // cluster that is still starting". The environment key is the
                // credential the bootstrap hashes, so accepting *it* here keeps
                // the node authenticable, and the constant-time comparison
                // against the configured secret means no other credential is
                // widened.
                None if state.api_key.as_deref().is_some_and(|expected| {
                    token.as_bytes().ct_eq(expected.as_bytes()).unwrap_u8() == 1
                }) =>
                {
                    shared_key_context()
                }
                None => return Err(StatusCode::UNAUTHORIZED),
            }
        } else if let Some(ref expected) = state.api_key {
            let token = provided.ok_or(StatusCode::UNAUTHORIZED)?;
            let authorized = token.as_bytes().ct_eq(expected.as_bytes()).unwrap_u8() == 1;
            if !authorized {
                return Err(StatusCode::UNAUTHORIZED);
            }
            shared_key_context()
        } else {
            if !state.allow_open_mode {
                return Err(StatusCode::UNAUTHORIZED);
            }
            AuthContext {
                user_id: "anonymous".into(),
                user_name: "anonymous".into(),
                permissions: vec![rekha_core::auth::Permission {
                    id: "open".into(),
                    action: "*".into(),
                    tenant: None,
                    database: None,
                }],
            }
        };

        // Authorization fails closed. The previous form checked permissions
        // `if let Some(..)` and skipped everything else, so a route the mapping
        // did not recognise was allowed for any authenticated caller — which is
        // how `/optimize` was unprotected until it was patched by hand.
        match route_action(method.as_str(), &path) {
            RouteAction::Public | RouteAction::AuthenticatedOnly => {}
            RouteAction::Permission {
                action,
                tenant,
                database,
            } => {
                // Native v2 collection routes carry no tenant in the path; the
                // handler resolves the collection's tenant and re-checks with
                // `check_collection_access`. Here the gate is coarse on purpose,
                // so a tenant-scoped operator is not denied before that check.
                let is_native = tenant.is_none()
                    && database.is_none()
                    && (action.starts_with("collection.") || action.starts_with("record."))
                    && !path.starts_with("/api/v2/auth/");
                let allowed = if is_native {
                    auth_ctx
                        .permissions
                        .iter()
                        .any(|p| p.action == "*" || p.action == action)
                } else {
                    has_permission(
                        &auth_ctx.permissions,
                        &action,
                        tenant.as_deref(),
                        database.as_deref(),
                    )
                };
                if !allowed {
                    return Err(StatusCode::FORBIDDEN);
                }
            }
            RouteAction::Unmatched => {
                // Not a route this build knows. Deny rather than skip, and do it
                // before the router so an unlisted path cannot reach a handler
                // that forgot to authorise itself.
                return Err(StatusCode::NOT_FOUND);
            }
        }

        request.extensions_mut().insert(auth_ctx);
        Ok(next.run(request).await)
    }

    pub async fn internal_auth(
        request: Request<axum::body::Body>,
        next: Next,
    ) -> Result<Response, StatusCode> {
        let state = request
            .extensions()
            .get::<std::sync::Arc<super::AppState>>()
            .cloned()
            .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

        // Fail closed. Internal routes drive Raft and WAL shipping; leaving them
        // open when no secret is configured would expose cluster administration
        // on any listener that mounts them. An *empty* secret is treated as
        // unset for the same reason: it would authenticate a request that sends
        // no header at all.
        let Some(ref expected) = state.cluster_secret else {
            return Err(StatusCode::UNAUTHORIZED);
        };
        if expected.is_empty() {
            return Err(StatusCode::UNAUTHORIZED);
        }
        let headers: HeaderMap = request.headers().clone();
        let provided = headers
            .get("x-rekha-cluster-secret")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let authorized = provided.as_bytes().ct_eq(expected.as_bytes()).unwrap_u8() == 1;
        if !authorized {
            return Err(StatusCode::UNAUTHORIZED);
        }
        Ok(next.run(request).await)
    }
}

// ---------------------------------------------------------------------------
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

type ShardWriteLockMap =
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
fn shared_http_client() -> reqwest::Client {
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

    fn write_lock(
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

/// Live record counts for `ids`, fetched from the nodes that hold the data.
///
/// A count is a data question: a node that is not a replica cannot answer it,
/// and answering `0` would look like an empty collection. This asks each
/// collection's first replica in one request per replica, not one per
/// collection, so a list of a thousand collections costs at most one round trip
/// per node.
///
/// A replica that cannot be reached is left out of the result rather than
/// reported as zero. Callers decide what to show for an unknown count; they must
/// not paper over it with a number that means "empty".
async fn counts_for(
    state: &AppState,
    ids: &[uuid::Uuid],
) -> Result<std::collections::HashMap<uuid::Uuid, u64>, ApiError> {
    let mut counts = std::collections::HashMap::new();
    let mut remote: std::collections::HashMap<u64, Vec<uuid::Uuid>> =
        std::collections::HashMap::new();

    for id in ids {
        match state.engine.count_local(id) {
            Ok(Some(local)) => {
                counts.insert(*id, local);
            }
            Ok(None) => {
                // A count belongs to a replica that holds the data, so the target
                // is the first owner of the collection's shard.
                let owner = state.shard_of(id).and_then(|shard| {
                    state
                        .shards
                        .as_ref()
                        .and_then(|shards| shards.owners(shard).first().copied())
                });
                if let Some(owner) = owner {
                    remote.entry(owner).or_default().push(*id);
                }
            }
            Err(e) => {
                tracing::warn!(collection = %id, %e, "could not count a collection");
            }
        }
    }

    for (owner, ids) in remote {
        let Some(url) = state.peers.url(owner) else {
            continue;
        };
        let mut request = state
            .http
            .post(format!("{url}/internal/stats/counts"))
            .json(&ids);
        if let Some(secret) = state.cluster_secret.as_deref() {
            request = request.header(crate::leadership::SECRET_HEADER, secret);
        }
        match request.send().await {
            Ok(response) => match response
                .json::<std::collections::HashMap<String, u64>>()
                .await
            {
                Ok(remote_counts) => {
                    for (id, count) in remote_counts {
                        if let Ok(id) = id.parse() {
                            counts.insert(id, count);
                        }
                    }
                }
                Err(e) => tracing::warn!(node = owner, %e, "unreadable count response"),
            },
            Err(e) => tracing::warn!(node = owner, %e, "count request failed"),
        }
    }

    // Every requested id must be answered. A missing entry means the replica
    // that holds it could not be reached, and the caller must not turn that into
    // a zero: "the collection is empty" and "I could not ask" are different
    // facts, and only one of them is true.
    if let Some(missing) = ids.iter().find(|id| !counts.contains_key(*id)) {
        return Err(ApiError::Engine(EngineError::Unavailable(format!(
            "no replica answered the count for collection {missing}; retry, or check that \
             the shard's owners are reachable"
        ))));
    }
    Ok(counts)
}

/// Counts this node holds, for a peer that needs them.
async fn stats_counts(
    State(state): State<Arc<AppState>>,
    Json(ids): Json<Vec<uuid::Uuid>>,
) -> Json<std::collections::HashMap<String, u64>> {
    let mut counts = std::collections::HashMap::new();
    for id in ids {
        if let Ok(Some(count)) = state.engine.count_local(&id) {
            counts.insert(id.to_string(), count);
        }
    }
    Json(counts)
}

/// Helper to check per-collection access for native v2 routes.
/// Loads the collection's tenant/database and verifies the caller's permissions.
pub(crate) fn check_collection_access(
    catalog: &dyn rekha_storage::Catalog,
    perms: &[rekha_core::auth::Permission],
    action: &str,
    collection_id: &uuid::Uuid,
) -> Result<(), EngineError> {
    let record = catalog
        .get_collection_by_id(collection_id)
        .map_err(EngineError::Catalog)?
        .ok_or_else(|| EngineError::CollectionNotFound(collection_id.to_string()))?;
    let tenant = record.config.tenant.as_str();
    let database = record.config.database.as_str();
    if rekha_core::auth::has_permission(perms, action, Some(tenant), Some(database))
        || rekha_core::auth::has_permission(perms, action, Some(tenant), None)
        || rekha_core::auth::has_permission(perms, action, None, None)
    {
        Ok(())
    } else {
        Err(EngineError::Forbidden(format!(
            "missing permission {action} on {tenant}/{database}"
        )))
    }
}

// ---------------------------------------------------------------------------
// Error mapping (newtype wrapper to satisfy orphan rules)
// ---------------------------------------------------------------------------

pub(crate) enum ApiError {
    Engine(EngineError),
    /// This node may not perform the write. `leader` names the node that can,
    /// so a client or proxy can retry against it.
    NotLeader {
        leader: Option<u64>,
    },
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        match self {
            ApiError::Engine(e) => {
                let status = engine_error_status(&e);
                let message = e.to_string();
                (status, Json(serde_json::json!({ "error": message }))).into_response()
            }
            ApiError::NotLeader { leader } => {
                let mut headers = axum::http::HeaderMap::new();
                if let Some(id) = leader
                    && let Ok(value) = axum::http::HeaderValue::from_str(&id.to_string())
                {
                    headers.insert(
                        axum::http::HeaderName::from_static(leadership::LEADER_HEADER),
                        value,
                    );
                }
                headers.insert(
                    axum::http::header::RETRY_AFTER,
                    axum::http::HeaderValue::from_static("1"),
                );
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    headers,
                    Json(serde_json::json!({
                        "error": "this node is not the leader for this write; retry against the leader",
                        "leader": leader,
                    })),
                )
                    .into_response()
            }
        }
    }
}

impl From<EngineError> for ApiError {
    fn from(e: EngineError) -> Self {
        ApiError::Engine(e)
    }
}

/// Create a collection and make it a cluster fact.
///
/// The log decides. In cluster mode the collection is proposed through Raft and
/// then materialized by the state machine on every node, including the leader;
/// How many collections are already placed on each node, read from the catalog.
///
/// This is the rebuild path, not the per-creation path: [`AppState::placement_cache`]
/// holds the incrementally-maintained answer and calls this only when the
/// leadership term changes, because a previous leader's placements are the only
/// ones in the catalog that this cache did not see happen.
fn load_from_catalog(
    state: &AppState,
    placement: &rekha_cluster::Placement,
) -> Result<std::collections::HashMap<u64, u32>, EngineError> {
    let mut load: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    for record in state.engine.catalog().list_collections_all()? {
        let Some(shard) = record.shard else {
            // An unsharded row in a sharded cluster: nothing known holds it, so
            // it contributes no load rather than a guessed one.
            continue;
        };
        for &node in placement.owners(shard) {
            *load.entry(node).or_default() += 1;
        }
    }
    Ok(load)
}

/// Decide where `config` goes and replicate the decision.
///
/// The leader decides because only it can see how loaded the nodes already are.
/// Recording the voter set instead made every node a replica, which put the data
/// back on every disk — the exact thing sharding removes — while the write path
/// had already started routing by shard, so the two disagreed about where the
/// data was.
async fn propose_with_placement(
    state: &AppState,
    raft: &Arc<openraft::Raft<rekha_cluster::RaftTypeConfig>>,
    shards: &Arc<rekha_cluster::ShardCluster>,
    config: &CollectionConfig,
) -> Result<(), EngineError> {
    let placement = shards.placement();

    // Held across the propose below on purpose: two creations choosing from the
    // same load picture would pick the same shard and land twice on one node.
    // Collection creation already serialises at the metadata leader, so this
    // costs nothing that was not already serialised.
    let mut cache = state.placement_cache.lock().await;
    let term = raft.metrics().borrow().current_term;
    let mut load = match cache.take() {
        Some((cached_term, load)) if cached_term == term => load,
        // A new term means a previous leader placed collections this process did
        // not witness. One rebuild per election, not one per creation.
        _ => load_from_catalog(state, &placement)?,
    };
    let shard = placement.place(config.id, &load);

    raft.client_write(ClusterOperation::AddCollection {
        config: config.clone(),
        shard: Some(shard),
    })
    .await
    .map_err(|e| EngineError::Recovery(format!("failed to replicate collection creation: {e}")))?;

    // Only after the entry commits: a rolled-back proposal must not leave load
    // counted for a collection that does not exist.
    for &owner in placement.owners(shard) {
        *load.entry(owner).or_default() += 1;
    }
    *cache = Some((term, load));
    Ok(())
}

/// Create a collection and make it a cluster fact.
///
/// The log decides. In cluster mode the collection is proposed through Raft and
/// then materialized by the state machine on every node, including the leader;
/// the leader does not create it locally first, because a create that is
/// interrupted before the log commits would leave one node holding a collection
/// the cluster has never heard of.
pub(crate) async fn create_collection_clustered(
    state: &AppState,
    config: &CollectionConfig,
    partitioned: bool,
) -> Result<rekha_storage::CollectionRecord, EngineError> {
    // Friendly early duplicate detection so the common case is a 409 without a
    // log round trip. The authoritative check is the catalog's own uniqueness
    // constraint inside the state machine.
    if state
        .engine
        .get_collection(&config.tenant, &config.database, &config.name)?
        .is_some()
    {
        return Err(EngineError::Catalog(
            rekha_storage::CatalogError::AlreadyExists,
        ));
    }

    let Some(raft) = state.raft.clone() else {
        let engine = state.engine.clone();
        let config = config.clone();
        return offload(move || engine.create_collection(&config)).await;
    };

    match state.shards.clone() {
        Some(shards) => {
            if partitioned {
                raft.client_write(ClusterOperation::AddCollection {
                    config: config.clone(),
                    shard: None,
                })
                .await
                .map_err(|e| {
                    EngineError::Recovery(format!("failed to replicate collection creation: {e}"))
                })?;
            } else {
                propose_with_placement(state, &raft, &shards, config).await?;
            }
        }
        // One group holds everything, and a single node has no shards at all.
        None => {
            raft.client_write(ClusterOperation::AddCollection {
                config: config.clone(),
                shard: None,
            })
            .await
            .map_err(|e| {
                EngineError::Recovery(format!("failed to replicate collection creation: {e}"))
            })?;
        }
    }

    let engine = state.engine.clone();
    let id = config.id;
    offload(move || engine.get_collection_by_id(&id))
        .await?
        .ok_or_else(|| {
            EngineError::Recovery(format!(
                "collection {id} committed to the log but was not materialized locally"
            ))
        })
}

/// Delete a collection through the log.
///
/// The removal is proposed before anything is deleted locally, so a failed
/// proposal leaves the collection intact everywhere rather than removing it
/// from one node.
pub(crate) async fn delete_collection_clustered(
    state: &AppState,
    collection_id: Uuid,
) -> Result<(), EngineError> {
    let Some(raft) = state.raft.clone() else {
        let engine = state.engine.clone();
        return offload(move || engine.delete_collection(&collection_id)).await;
    };
    raft.client_write(ClusterOperation::RemoveCollection { collection_id })
        .await
        .map_err(|e| {
            EngineError::Recovery(format!("failed to replicate collection deletion: {e}"))
        })?;
    Ok(())
}

/// Prepare a write on the leader and get it committed.
///
/// Two paths, one meaning:
///
/// * Single-node — append through the WAL and apply, exactly as before.
/// * Cluster — validate locally, then propose one entry. The state machine
///   applies it on every node, so the log index is the dataset sequence and
///   there is one replication mechanism instead of two.
///
/// The per-collection lock is held across prepare and propose so a metadata
/// patch cannot be resolved from a base that another write is about to replace.
pub(crate) async fn write_batch<F>(
    state: &AppState,
    collection_id: Uuid,
    prepare: F,
) -> Result<usize, ApiError>
where
    F: FnOnce() -> Result<rekha_engine::PreparedBatch, EngineError> + Send + 'static,
{
    // The guard is owned so it can be held across awaits without borrowing the
    // state borrow that produced it.
    let shard = state.shard_of(&collection_id);
    let lock = state.write_lock(collection_id, shard);
    let _guard = lock.lock_owned().await;

    let prepared = offload(prepare).await?;
    let ops = prepared.ops.len();
    commit_batch(state, collection_id, prepared).await?;
    Ok(ops)
}

pub(crate) async fn commit_batch_for_shard(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    collection_id: Uuid,
    prepared: rekha_engine::PreparedBatch,
) -> Result<(), ApiError> {
    if prepared.ops.is_empty() {
        return Ok(());
    }

    if !state.data_via_log {
        let engine = state.engine.clone();
        offload(move || engine.append_ops_partition(&collection_id, Some(shard), &prepared.ops))
            .await?;
        return Ok(());
    }

    let Some(shards) = state.shards.as_ref() else {
        return Err(ApiError::Engine(EngineError::Unavailable(
            "no shards configured".into(),
        )));
    };
    let Some(raft) = shards.raft(shard) else {
        return Err(ApiError::Engine(EngineError::NotReplica {
            collection: collection_id.to_string(),
            shard: Some(shard),
        }));
    };
    let record = state
        .engine
        .get_collection_by_id(&collection_id)?
        .ok_or_else(|| EngineError::CollectionNotFound(collection_id.to_string()))?;

    if !shards.is_ensured(collection_id, shard) {
        raft.client_write(ClusterOperation::AddCollection {
            config: record.config,
            shard: Some(shard),
        })
        .await
        .map_err(|e| {
            EngineError::Recovery(format!(
                "could not tell shard {shard} about collection {collection_id}: {e}"
            ))
        })?;
        shards.mark_ensured(collection_id, shard);
    }

    let operation = ClusterOperation::Batch {
        collection_id,
        ops: prepared.ops,
        dimension: prepared.dimension,
    };

    match raft.client_write(operation).await {
        Ok(response) => {
            match serde_json::from_str::<rekha_cluster::BatchResponse>(&response.data) {
                Ok(applied) => match applied.error {
                    None => Ok(()),
                    Some(error) => Err(ApiError::Engine(EngineError::Recovery(format!(
                        "the write committed at log index {} but this node could not apply it: \
                         {error}",
                        applied.seq
                    )))),
                },
                Err(e) => {
                    tracing::warn!(
                        %e,
                        "unreadable apply response; the entry committed but its outcome is unknown"
                    );
                    Ok(())
                }
            }
        }
        Err(openraft::error::RaftError::APIError(
            openraft::error::ClientWriteError::ForwardToLeader(forward),
        )) => Err(ApiError::NotLeader {
            leader: forward.leader_id.or(forward.leader_node),
        }),
        Err(e) => Err(ApiError::Engine(EngineError::Recovery(format!(
            "the write could not be committed to the log: {e}"
        )))),
    }
}

async fn commit_batch(
    state: &AppState,
    collection_id: Uuid,
    prepared: rekha_engine::PreparedBatch,
) -> Result<(), ApiError> {
    if !state.data_via_log {
        let engine = state.engine.clone();
        offload(move || engine.append_ops(&collection_id, &prepared.ops)).await?;
        return Ok(());
    }

    let shard = match state.shards.as_ref() {
        Some(_) => {
            let record = state
                .engine
                .get_collection_by_id(&collection_id)?
                .ok_or_else(|| EngineError::CollectionNotFound(collection_id.to_string()))?;
            record.shard
        }
        None => None,
    };

    match (state.shards.as_ref(), shard) {
        (Some(_), Some(shard)) => {
            commit_batch_for_shard(state, shard, collection_id, prepared).await
        }
        (Some(_), None) => {
            // Partitioned collection committed through single batch fallback
            commit_batch_for_shard(state, rekha_cluster::ShardId(0), collection_id, prepared).await
        }
        (None, _) => {
            let raft = state.raft.clone().ok_or_else(|| {
                EngineError::Recovery(
                    "this node is configured to write through the log but has no Raft group".into(),
                )
            })?;
            let operation = ClusterOperation::Batch {
                collection_id,
                ops: prepared.ops,
                dimension: prepared.dimension,
            };
            match raft.client_write(operation).await {
                Ok(response) => {
                    match serde_json::from_str::<rekha_cluster::BatchResponse>(&response.data) {
                        Ok(applied) => match applied.error {
                            None => Ok(()),
                            Some(error) => Err(ApiError::Engine(EngineError::Recovery(format!(
                                "the write committed at log index {} but this node could not apply it: \
                                 {error}",
                                applied.seq
                            )))),
                        },
                        Err(e) => {
                            tracing::warn!(
                                %e,
                                "unreadable apply response; the entry committed but its outcome is unknown"
                            );
                            Ok(())
                        }
                    }
                }
                Err(openraft::error::RaftError::APIError(
                    openraft::error::ClientWriteError::ForwardToLeader(forward),
                )) => Err(ApiError::NotLeader {
                    leader: forward.leader_id.or(forward.leader_node),
                }),
                Err(e) => Err(ApiError::Engine(EngineError::Recovery(format!(
                    "the write could not be committed to the log: {e}"
                )))),
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardWriteKind {
    Add {
        ids: Vec<Id>,
        embeddings: Vec<Vec<f32>>,
        metadatas: Option<Vec<Option<Metadata>>>,
        documents: Option<Vec<Option<String>>>,
    },
    Upsert {
        ids: Vec<Id>,
        embeddings: Vec<Vec<f32>>,
        metadatas: Option<Vec<Option<rekha_core::types::MetadataPatch>>>,
        documents: Option<Vec<Option<String>>>,
    },
    Update {
        ids: Vec<Id>,
        embeddings: Option<Vec<Vec<f32>>>,
        metadatas: Option<Vec<Option<rekha_core::types::MetadataPatch>>>,
        documents: Option<Vec<Option<String>>>,
    },
    Delete {
        ids: Vec<Id>,
    },
    DeleteWhere {
        ids: Option<Vec<Id>>,
        filter: Option<WhereFilter>,
        document: Option<rekha_core::filter::WhereDocument>,
        limit: Option<usize>,
    },
    Prepared {
        ops: Vec<rekha_core::op::Operation>,
        dimension: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardBatchRequest {
    pub collection_id: Uuid,
    pub kind: ShardWriteKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardPrepareRequest {
    pub tx_id: Uuid,
    pub collection_id: Uuid,
    pub kind: ShardWriteKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardCommitRequest {
    pub tx_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardAbortRequest {
    pub tx_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxCommitRequest {
    pub tx_id: Uuid,
    pub shards: Vec<rekha_cluster::ShardId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxAbortRequest {
    pub tx_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardBatchResponse {
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardQueryRequest {
    pub collection_id: Uuid,
    pub query_embeddings: Vec<Vec<f32>>,
    pub n_results: usize,
    pub where_filter: Option<WhereFilter>,
    pub where_document: Option<rekha_core::filter::WhereDocument>,
    pub with_embeddings: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardGetRequest {
    pub collection_id: Uuid,
    pub ids: Option<Vec<Id>>,
    pub where_filter: Option<WhereFilter>,
    pub where_document: Option<rekha_core::filter::WhereDocument>,
    pub limit: Option<usize>,
    pub offset: usize,
}

pub(crate) type SubBatchAdd = (
    Vec<Id>,
    Vec<Vec<f32>>,
    Option<Vec<Option<Metadata>>>,
    Option<Vec<Option<String>>>,
);
pub(crate) type SubBatchUpsert = (
    Vec<Id>,
    Vec<Vec<f32>>,
    Option<Vec<Option<rekha_core::types::MetadataPatch>>>,
    Option<Vec<Option<String>>>,
);
pub(crate) type SubBatchUpdate = (
    Vec<Id>,
    Option<Vec<Vec<f32>>>,
    Option<Vec<Option<rekha_core::types::MetadataPatch>>>,
    Option<Vec<Option<String>>>,
);

pub(crate) fn is_partitioned(state: &AppState, collection_id: &Uuid) -> bool {
    if state.shards.is_none() {
        return false;
    }
    match state.engine.get_collection_by_id(collection_id) {
        Ok(Some(record)) => record.shard.is_none(),
        _ => false,
    }
}

pub(crate) fn is_partitioned_requested(explicit: Option<bool>, meta: Option<&Metadata>) -> bool {
    if let Some(p) = explicit {
        return p;
    }
    if let Some(m) = meta
        && let Some(val) = m.get("partitioned")
    {
        let matches = match val {
            rekha_core::types::MetadataValue::Str(s) => s == "true",
            rekha_core::types::MetadataValue::Bool(b) => *b,
            _ => false,
        };
        if matches {
            return true;
        }
    }
    std::env::var("REKHA_INTRA_SHARDING")
        .map(|v| v == "always" || v == "1" || v == "true")
        .unwrap_or(false)
}

pub(crate) async fn check_shard_read_barrier(
    raft: &Arc<openraft::Raft<rekha_cluster::RaftTypeConfig>>,
) -> Result<(), ApiError> {
    let single_voter = {
        let metrics = raft.metrics();
        let m = metrics.borrow();
        m.membership_config.membership().voter_ids().count() <= 1
    };
    if single_voter {
        return Ok(());
    }

    match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        raft.ensure_linearizable(),
    )
    .await
    {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => {
            tracing::warn!(%err, "rejecting shard read: this node can no longer prove it leads");
            let leader = raft.metrics().borrow().current_leader;
            Err(ApiError::NotLeader { leader })
        }
        Err(_) => {
            tracing::warn!("rejecting shard read: no quorum confirmation within deadline");
            let leader = raft.metrics().borrow().current_leader;
            Err(ApiError::NotLeader { leader })
        }
    }
}

async fn ensure_collection_on_shard(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    collection_id: Uuid,
) -> Result<(), ApiError> {
    let record = match state.engine.get_collection_by_id(&collection_id)? {
        Some(r) => r,
        None => {
            return Err(ApiError::Engine(EngineError::Unavailable(format!(
                "collection {collection_id} metadata not yet applied on shard replica; retry shortly"
            ))));
        }
    };
    if let Some(shards) = state.shards.as_ref()
        && let Some(raft) = shards.raft(shard)
        && !shards.is_ensured(collection_id, shard)
    {
        raft.client_write(ClusterOperation::AddCollection {
            config: record.config.clone(),
            shard: Some(shard),
        })
        .await
        .map_err(|e| {
            EngineError::Recovery(format!(
                "could not tell shard {shard} about collection {collection_id}: {e}"
            ))
        })?;
        shards.mark_ensured(collection_id, shard);
    }
    Ok(())
}

async fn resolve_shard_write_kind(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    collection_id: Uuid,
    kind: ShardWriteKind,
) -> Result<(rekha_engine::PreparedBatch, usize), ApiError> {
    match kind {
        ShardWriteKind::Add {
            ids,
            embeddings,
            metadatas,
            documents,
        } => {
            let engine = state.engine.clone();
            let embs: Vec<Embedding> = embeddings.into_iter().map(Embedding::from).collect();
            let p = offload(move || {
                engine.prepare_add_partition(
                    &collection_id,
                    Some(shard),
                    &ids,
                    &embs,
                    metadatas.as_deref(),
                    documents.as_deref(),
                )
            })
            .await?;
            let n = p.ops.len();
            Ok((p, n))
        }
        ShardWriteKind::Upsert {
            ids,
            embeddings,
            metadatas,
            documents,
        } => {
            let engine = state.engine.clone();
            let embs: Vec<Embedding> = embeddings.into_iter().map(Embedding::from).collect();
            let p = offload(move || {
                engine.prepare_upsert_partition(
                    &collection_id,
                    Some(shard),
                    &ids,
                    &embs,
                    metadatas.as_deref(),
                    documents.as_deref(),
                )
            })
            .await?;
            let n = p.ops.len();
            Ok((p, n))
        }
        ShardWriteKind::Update {
            ids,
            embeddings,
            metadatas,
            documents,
        } => {
            let engine = state.engine.clone();
            let embs: Option<Vec<Embedding>> =
                embeddings.map(|e| e.into_iter().map(Embedding::from).collect());
            let p = offload(move || {
                engine.prepare_update_partition(
                    &collection_id,
                    Some(shard),
                    &ids,
                    embs.as_deref(),
                    metadatas.as_deref(),
                    documents.as_deref(),
                )
            })
            .await?;
            let n = p.ops.len();
            Ok((p, n))
        }
        ShardWriteKind::Delete { ids } => {
            let engine = state.engine.clone();
            let n = ids.len();
            let p =
                offload(move || engine.prepare_delete_partition(&collection_id, Some(shard), &ids))
                    .await?;
            Ok((p, n))
        }
        ShardWriteKind::DeleteWhere {
            ids,
            filter,
            document,
            limit,
        } => {
            let engine = state.engine.clone();
            let p = offload(move || {
                engine.prepare_delete_where_partition(
                    &collection_id,
                    Some(shard),
                    ids.as_deref(),
                    filter.as_ref(),
                    document.as_ref(),
                    limit,
                )
            })
            .await?;
            let n = p.ops.len();
            Ok((p, n))
        }
        ShardWriteKind::Prepared { ops, dimension } => {
            let n = ops.len();
            Ok((rekha_engine::PreparedBatch { ops, dimension }, n))
        }
    }
}

pub(crate) async fn commit_write_for_shard(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    collection_id: Uuid,
    kind: ShardWriteKind,
) -> Result<usize, ApiError> {
    let lock = state.write_lock(collection_id, Some(shard));
    let _guard = lock.lock_owned().await;

    ensure_collection_on_shard(state, shard, collection_id).await?;
    let (prepared, count) = resolve_shard_write_kind(state, shard, collection_id, kind).await?;
    commit_batch_for_shard(state, shard, collection_id, prepared).await?;
    Ok(count)
}

pub(crate) async fn prepare_write_for_shard(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    tx_id: Uuid,
    collection_id: Uuid,
    kind: ShardWriteKind,
) -> Result<usize, ApiError> {
    let lock = state.write_lock(collection_id, Some(shard));
    let _guard = lock.lock_owned().await;

    ensure_collection_on_shard(state, shard, collection_id).await?;
    let (prepared, count) = resolve_shard_write_kind(state, shard, collection_id, kind).await?;

    if prepared.ops.is_empty() {
        return Ok(0);
    }

    let ids: Vec<String> = prepared.ops.iter().map(|o| o.id().to_string()).collect();
    state
        .engine
        .reserve_tx_ids(&collection_id, Some(shard), tx_id, &ids)?;

    let Some(shards) = state.shards.as_ref() else {
        let _ = state
            .engine
            .release_tx_ids(&collection_id, Some(shard), &tx_id);
        return Err(ApiError::Engine(EngineError::Unavailable(
            "no shards configured".into(),
        )));
    };
    let Some(raft) = shards.raft(shard) else {
        let _ = state
            .engine
            .release_tx_ids(&collection_id, Some(shard), &tx_id);
        return Err(ApiError::Engine(EngineError::NotReplica {
            collection: collection_id.to_string(),
            shard: Some(shard),
        }));
    };

    let operation = ClusterOperation::PrepareBatch {
        tx_id,
        collection_id,
        ops: prepared.ops,
        dimension: prepared.dimension,
    };

    match raft.client_write(operation).await {
        Ok(response) => {
            match serde_json::from_str::<rekha_cluster::BatchResponse>(&response.data) {
                Ok(applied) => match applied.error {
                    None => {
                        shards.mark_tx_staged(shard, tx_id);
                        Ok(count)
                    }
                    Some(error) => {
                        let _ = state
                            .engine
                            .release_tx_ids(&collection_id, Some(shard), &tx_id);
                        Err(ApiError::Engine(EngineError::Recovery(format!(
                            "2PC prepare committed at log index {} but could not stage: {error}",
                            applied.seq
                        ))))
                    }
                },
                Err(_) => {
                    shards.mark_tx_staged(shard, tx_id);
                    Ok(count)
                }
            }
        }
        Err(openraft::error::RaftError::APIError(
            openraft::error::ClientWriteError::ForwardToLeader(forward),
        )) => {
            let _ = state
                .engine
                .release_tx_ids(&collection_id, Some(shard), &tx_id);
            Err(ApiError::NotLeader {
                leader: forward.leader_id.or(forward.leader_node),
            })
        }
        Err(e) => {
            let _ = state
                .engine
                .release_tx_ids(&collection_id, Some(shard), &tx_id);
            Err(ApiError::Engine(EngineError::Recovery(format!(
                "2PC prepare failed on shard {shard}: {e}"
            ))))
        }
    }
}

pub(crate) async fn commit_tx_for_shard(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    tx_id: Uuid,
) -> Result<(), ApiError> {
    let Some(shards) = state.shards.as_ref() else {
        return Err(ApiError::Engine(EngineError::Unavailable(
            "no shards configured".into(),
        )));
    };
    let Some(raft) = shards.raft(shard) else {
        return Err(ApiError::Engine(EngineError::Unavailable(format!(
            "shard {shard} not hosted here"
        ))));
    };

    let operation = ClusterOperation::CommitBatch { tx_id };
    match raft.client_write(operation).await {
        Ok(_) => {
            shards.mark_tx_resolved(&tx_id);
            Ok(())
        }
        Err(openraft::error::RaftError::APIError(
            openraft::error::ClientWriteError::ForwardToLeader(forward),
        )) => Err(ApiError::NotLeader {
            leader: forward.leader_id.or(forward.leader_node),
        }),
        Err(e) => Err(ApiError::Engine(EngineError::Recovery(format!(
            "commit 2PC tx {tx_id} failed on shard {shard}: {e}"
        )))),
    }
}

pub(crate) async fn abort_tx_for_shard(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    tx_id: Uuid,
) -> Result<(), ApiError> {
    let Some(shards) = state.shards.as_ref() else {
        return Ok(());
    };
    shards.mark_tx_resolved(&tx_id);
    let Some(raft) = shards.raft(shard) else {
        return Ok(());
    };

    let operation = ClusterOperation::AbortBatch { tx_id };
    let _ = raft.client_write(operation).await;
    Ok(())
}

async fn dispatch_shard_prepare(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    req: ShardPrepareRequest,
) -> Result<usize, ApiError> {
    let Some(shards) = state.shards.as_ref() else {
        return Err(ApiError::Engine(EngineError::Unavailable(
            "no shards configured".into(),
        )));
    };

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline {
        if let Some(raft) = shards.raft(shard)
            && raft.metrics().borrow().current_leader == Some(raft.metrics().borrow().id)
        {
            match prepare_write_for_shard(
                state,
                shard,
                req.tx_id,
                req.collection_id,
                req.kind.clone(),
            )
            .await
            {
                Ok(count) => return Ok(count),
                Err(ApiError::NotLeader { .. }) => {}
                Err(e) => return Err(e),
            }
        }

        let mut should_retry = false;
        for target in state.shard_targets(shard) {
            let Some(url) = state.peers.url(target) else {
                continue;
            };
            match state
                .internal_post(format!("{url}/internal/shard/{}/prepare", shard.0))
                .json(&req)
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    let res: ShardBatchResponse = resp.json().await.map_err(|e| {
                        ApiError::Engine(EngineError::Recovery(format!(
                            "invalid prepare response from shard {}: {e}",
                            shard.0
                        )))
                    })?;
                    return Ok(res.count);
                }
                Ok(resp) if resp.status() == StatusCode::SERVICE_UNAVAILABLE => {
                    should_retry = true;
                    if let Some(leader_id) = resp
                        .headers()
                        .get(crate::leadership::LEADER_HEADER)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok())
                        && let Some(leader_url) = state.peers.url(leader_id)
                        && let Ok(resp2) = state
                            .internal_post(format!(
                                "{leader_url}/internal/shard/{}/prepare",
                                shard.0
                            ))
                            .json(&req)
                            .send()
                            .await
                        && resp2.status().is_success()
                    {
                        let res: ShardBatchResponse = resp2.json().await.map_err(|e| {
                            ApiError::Engine(EngineError::Recovery(format!(
                                "invalid prepare response from shard {}: {e}",
                                shard.0
                            )))
                        })?;
                        return Ok(res.count);
                    }
                }
                Ok(resp) if resp.status().is_client_error() => {
                    let body = resp.text().await.unwrap_or_default();
                    return Err(ApiError::Engine(EngineError::Validation(body)));
                }
                _ => should_retry = true,
            }
        }

        if should_retry {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        } else {
            break;
        }
    }

    Err(ApiError::Engine(EngineError::Unavailable(format!(
        "could not reach leader for shard {} to prepare 2PC",
        shard.0
    ))))
}

async fn dispatch_shard_commit(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    tx_id: Uuid,
) -> Result<(), ApiError> {
    let Some(shards) = state.shards.as_ref() else {
        return Ok(());
    };

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline {
        let is_leader = if let Some(raft) = shards.raft(shard) {
            let m = raft.metrics();
            let b = m.borrow();
            b.current_leader == Some(b.id)
        } else {
            false
        };
        if is_leader {
            match commit_tx_for_shard(state, shard, tx_id).await {
                Ok(()) => return Ok(()),
                Err(ApiError::NotLeader { .. }) => {}
                Err(e) => return Err(e),
            }
        }

        let req = ShardCommitRequest { tx_id };
        let mut should_retry = false;
        for target in state.shard_targets(shard) {
            let Some(url) = state.peers.url(target) else {
                continue;
            };
            match state
                .internal_post(format!("{url}/internal/shard/{}/commit", shard.0))
                .json(&req)
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                Ok(resp) if resp.status() == StatusCode::SERVICE_UNAVAILABLE => should_retry = true,
                _ => should_retry = true,
            }
        }
        if should_retry {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        } else {
            break;
        }
    }
    Err(ApiError::Engine(EngineError::Unavailable(format!(
        "could not reach leader for shard {} to commit 2PC transaction {tx_id}",
        shard.0
    ))))
}

async fn dispatch_metadata_commit_tx(
    state: &AppState,
    tx_id: Uuid,
    shards: Vec<rekha_cluster::ShardId>,
) -> Result<(), ApiError> {
    let Some(raft) = state.raft.as_ref() else {
        return Ok(());
    };

    let op = ClusterOperation::CommitTx {
        tx_id,
        shards: shards.clone(),
    };

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline {
        let (is_leader, leader_id) = {
            let m = raft.metrics();
            let b = m.borrow();
            (b.current_leader == Some(b.id), b.current_leader)
        };
        if is_leader {
            match raft.client_write(op.clone()).await {
                Ok(_) => return Ok(()),
                Err(openraft::error::RaftError::APIError(
                    openraft::error::ClientWriteError::ForwardToLeader(_),
                )) => {}
                Err(e) => {
                    return Err(ApiError::Engine(EngineError::Recovery(format!(
                        "failed to commit 2PC transaction {tx_id} on metadata group: {e}"
                    ))));
                }
            }
        }

        // Try forwarding to the metadata leader
        if let Some(leader) = leader_id
            && let Some(url) = state.peers.url(leader)
        {
            let req = TxCommitRequest {
                tx_id,
                shards: shards.clone(),
            };
            if let Ok(resp) = state
                .internal_post(format!("{url}/internal/cluster/tx/commit"))
                .json(&req)
                .send()
                .await
                && resp.status().is_success()
            {
                return Ok(());
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    }

    Err(ApiError::Engine(EngineError::Unavailable(format!(
        "could not reach metadata leader to commit 2PC transaction {tx_id}"
    ))))
}

async fn dispatch_metadata_abort_tx(state: &AppState, tx_id: Uuid) {
    let Some(raft) = state.raft.as_ref() else {
        return;
    };
    let op = ClusterOperation::AbortTx { tx_id };
    let (is_leader, leader_id) = {
        let m = raft.metrics();
        let b = m.borrow();
        (b.current_leader == Some(b.id), b.current_leader)
    };
    if is_leader {
        let _ = raft.client_write(op).await;
        return;
    }
    if let Some(leader) = leader_id
        && let Some(url) = state.peers.url(leader)
    {
        let req = TxAbortRequest { tx_id };
        let _ = state
            .internal_post(format!("{url}/internal/cluster/tx/abort"))
            .json(&req)
            .send()
            .await;
    }
}

async fn dispatch_shard_abort(state: &AppState, shard: rekha_cluster::ShardId, tx_id: Uuid) {
    let Some(shards) = state.shards.as_ref() else {
        return;
    };
    let is_leader = if let Some(raft) = shards.raft(shard) {
        let m = raft.metrics();
        let b = m.borrow();
        b.current_leader == Some(b.id)
    } else {
        false
    };
    if is_leader {
        let _ = abort_tx_for_shard(state, shard, tx_id).await;
        return;
    }

    let req = ShardAbortRequest { tx_id };
    for target in state.shard_targets(shard) {
        let Some(url) = state.peers.url(target) else {
            continue;
        };
        if let Ok(resp) = state
            .internal_post(format!("{url}/internal/shard/{}/abort", shard.0))
            .json(&req)
            .send()
            .await
            && resp.status().is_success()
        {
            return;
        }
    }
}

pub(crate) async fn write_partitioned_2pc(
    state: &AppState,
    collection_id: Uuid,
    sub_batches: Vec<(rekha_cluster::ShardId, ShardWriteKind)>,
) -> Result<(), ApiError> {
    let tx_id = Uuid::new_v4();
    let prepare_futs = sub_batches.into_iter().map(|(shard, kind)| async move {
        let req = ShardPrepareRequest {
            tx_id,
            collection_id,
            kind,
        };
        dispatch_shard_prepare(state, shard, req)
            .await
            .map(|count| (shard, count))
    });

    let prepare_results = futures::future::join_all(prepare_futs).await;
    let mut prepared_shards = Vec::with_capacity(prepare_results.len());
    let mut first_err = None;

    for res in prepare_results {
        match res {
            Ok((shard, _)) => prepared_shards.push(shard),
            Err(e) => {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
    }

    if let Some(err) = first_err {
        let abort_futs = prepared_shards.into_iter().map(|shard| async move {
            dispatch_shard_abort(state, shard, tx_id).await;
        });
        futures::future::join_all(abort_futs).await;
        dispatch_metadata_abort_tx(state, tx_id).await;
        return Err(err);
    }

    if let Err(e) = dispatch_metadata_commit_tx(state, tx_id, prepared_shards.clone()).await {
        let abort_futs = prepared_shards.into_iter().map(|shard| async move {
            dispatch_shard_abort(state, shard, tx_id).await;
        });
        futures::future::join_all(abort_futs).await;
        dispatch_metadata_abort_tx(state, tx_id).await;
        return Err(e);
    }

    let commit_futs = prepared_shards
        .into_iter()
        .map(|shard| async move { dispatch_shard_commit(state, shard, tx_id).await });
    let commit_results = futures::future::join_all(commit_futs).await;
    for res in commit_results {
        res?;
    }
    Ok(())
}

async fn dispatch_shard_batch(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    req: ShardBatchRequest,
) -> Result<usize, ApiError> {
    let Some(shards) = state.shards.as_ref() else {
        return Err(ApiError::Engine(EngineError::Unavailable(
            "no shards configured".into(),
        )));
    };

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline {
        if let Some(raft) = shards.raft(shard)
            && raft.metrics().borrow().current_leader == Some(raft.metrics().borrow().id)
        {
            match commit_write_for_shard(state, shard, req.collection_id, req.kind.clone()).await {
                Ok(count) => return Ok(count),
                Err(ApiError::NotLeader { .. }) => {}
                Err(e) => return Err(e),
            }
        }

        let mut should_retry = false;
        for target in state.shard_targets(shard) {
            let Some(url) = state.peers.url(target) else {
                continue;
            };
            match state
                .internal_post(format!("{url}/internal/shard/{}/batch", shard.0))
                .json(&req)
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    let res: ShardBatchResponse = resp.json().await.map_err(|e| {
                        ApiError::Engine(EngineError::Recovery(format!(
                            "invalid response from shard {}: {e}",
                            shard.0
                        )))
                    })?;
                    return Ok(res.count);
                }
                Ok(resp) if resp.status() == StatusCode::SERVICE_UNAVAILABLE => {
                    should_retry = true;
                    if let Some(leader_id) = resp
                        .headers()
                        .get(crate::leadership::LEADER_HEADER)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok())
                        && let Some(leader_url) = state.peers.url(leader_id)
                        && let Ok(resp2) = state
                            .internal_post(format!("{leader_url}/internal/shard/{}/batch", shard.0))
                            .json(&req)
                            .send()
                            .await
                        && resp2.status().is_success()
                    {
                        let res: ShardBatchResponse = resp2.json().await.map_err(|e| {
                            ApiError::Engine(EngineError::Recovery(format!(
                                "invalid response from shard {}: {e}",
                                shard.0
                            )))
                        })?;
                        return Ok(res.count);
                    }
                }
                Ok(resp) if resp.status().is_client_error() => {
                    let body = resp.text().await.unwrap_or_default();
                    return Err(ApiError::Engine(EngineError::Validation(body)));
                }
                _ => should_retry = true,
            }
        }

        if should_retry {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        } else {
            break;
        }
    }

    Err(ApiError::Engine(EngineError::Unavailable(format!(
        "could not reach leader for shard {}",
        shard.0
    ))))
}

async fn dispatch_shard_query(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    req: ShardQueryRequest,
) -> Result<Vec<Vec<rekha_engine::ScoredPoint>>, ApiError> {
    let Some(shards) = state.shards.as_ref() else {
        return Err(ApiError::Engine(EngineError::Unavailable(
            "no shards configured".into(),
        )));
    };

    if let Some(raft) = shards.raft(shard) {
        let m = raft.metrics();
        let leads = m.borrow().current_leader == Some(m.borrow().id);
        if leads && check_shard_read_barrier(&raft).await.is_ok() {
            let engine = state.engine.clone();
            let collection_id = req.collection_id;
            let query_embeddings = req.query_embeddings;
            let n_results = req.n_results;
            let where_filter = req.where_filter;
            let where_document = req.where_document;
            let with_embeddings = req.with_embeddings;
            return offload(move || {
                let opts = QueryOptions {
                    ef: 0,
                    where_filter,
                    where_document,
                    oversampling: 4,
                };
                let embeddings_vec: Vec<Embedding> =
                    query_embeddings.into_iter().map(Embedding::from).collect();
                engine.query_many_partition(
                    &collection_id,
                    Some(shard),
                    &embeddings_vec,
                    n_results,
                    &opts,
                    with_embeddings,
                )
            })
            .await
            .map_err(ApiError::Engine);
        }
    }

    for target in state.shard_targets(shard) {
        let Some(url) = state.peers.url(target) else {
            continue;
        };
        if let Ok(resp) = state
            .internal_post(format!("{url}/internal/shard/{}/query", shard.0))
            .json(&req)
            .send()
            .await
            && resp.status().is_success()
            && let Ok(hits) = resp.json::<Vec<Vec<rekha_engine::ScoredPoint>>>().await
        {
            return Ok(hits);
        }
    }
    Err(ApiError::Engine(EngineError::Unavailable(format!(
        "could not query shard {}",
        shard.0
    ))))
}

async fn dispatch_shard_count(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    collection_id: Uuid,
) -> Result<u64, ApiError> {
    let Some(shards) = state.shards.as_ref() else {
        return Err(ApiError::Engine(EngineError::Unavailable(
            "no shards configured".into(),
        )));
    };

    if let Some(raft) = shards.raft(shard) {
        let m = raft.metrics();
        let leads = m.borrow().current_leader == Some(m.borrow().id);
        if leads && check_shard_read_barrier(&raft).await.is_ok() {
            return state
                .engine
                .count_partition(&collection_id, Some(shard))
                .map_err(ApiError::Engine);
        }
    }

    for target in state.shard_targets(shard) {
        let Some(url) = state.peers.url(target) else {
            continue;
        };
        if let Ok(resp) = state
            .internal_post(format!("{url}/internal/shard/{}/count", shard.0))
            .json(&collection_id)
            .send()
            .await
            && resp.status().is_success()
            && let Ok(val) = resp.json::<serde_json::Value>().await
            && let Some(c) = val.get("count").and_then(|v| v.as_u64())
        {
            return Ok(c);
        }
    }
    Err(ApiError::Engine(EngineError::Unavailable(format!(
        "could not count shard {}",
        shard.0
    ))))
}

async fn dispatch_shard_get(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    req: ShardGetRequest,
) -> Result<Vec<rekha_engine::Record>, ApiError> {
    let Some(shards) = state.shards.as_ref() else {
        return Err(ApiError::Engine(EngineError::Unavailable(
            "no shards configured".into(),
        )));
    };

    if let Some(raft) = shards.raft(shard) {
        let m = raft.metrics();
        let leads = m.borrow().current_leader == Some(m.borrow().id);
        if leads && check_shard_read_barrier(&raft).await.is_ok() {
            return state
                .engine
                .get_records_partition(
                    &req.collection_id,
                    Some(shard),
                    req.ids.as_deref(),
                    req.where_filter.as_ref(),
                    req.where_document.as_ref(),
                    req.limit,
                    req.offset,
                )
                .map_err(ApiError::Engine);
        }
    }

    for target in state.shard_targets(shard) {
        let Some(url) = state.peers.url(target) else {
            continue;
        };
        if let Ok(resp) = state
            .internal_post(format!("{url}/internal/shard/{}/get", shard.0))
            .json(&req)
            .send()
            .await
            && resp.status().is_success()
            && let Ok(records) = resp.json::<Vec<rekha_engine::Record>>().await
        {
            return Ok(records);
        }
    }
    Err(ApiError::Engine(EngineError::Unavailable(format!(
        "could not get records from shard {}",
        shard.0
    ))))
}

/// Run blocking engine work on the blocking pool.
///
/// The engine fsyncs WAL batches and runs checkpoint phases synchronously, and
/// queries scan and rank in memory. Those calls share a runtime with Raft,
/// which must send a heartbeat every `heartbeat_interval`. A few hundred
/// milliseconds of fsync on a worker thread delays that heartbeat, followers
/// time out at `election_timeout_max`, and the cluster elects a new leader.
///
/// The failure looks like a consensus bug and is a scheduling decision: under
/// sustained write load the cluster loses leadership precisely because it is
/// doing what it was asked to do. Moving engine work to the blocking pool keeps
/// the async workers free to answer heartbeats.
pub(crate) async fn offload<T, F>(work: F) -> Result<T, EngineError>
where
    F: FnOnce() -> Result<T, EngineError> + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(work).await {
        Ok(result) => result,
        Err(join) => Err(EngineError::Io(std::io::Error::other(format!(
            "engine worker task failed: {join}"
        )))),
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Maximum records accepted in a single write or delete batch.
/// Advertised by /pre-flight-checks and enforced by the write handlers.
pub const MAX_BATCH_SIZE: usize = 10_000;
pub(crate) const MAX_QUERY_EMBEDDINGS: usize = 100;
pub(crate) const MAX_N_RESULTS: usize = 1_000;
pub(crate) const MAX_ID_LENGTH: usize = 512;
pub(crate) const MAX_NAME_LENGTH: usize = 512;
pub(crate) const MAX_DIMENSION: usize = 65_536;

pub(crate) fn validation_err(msg: impl Into<String>) -> EngineError {
    EngineError::Validation(msg.into())
}

pub(crate) fn validate_name(name: &str) -> Result<(), EngineError> {
    if name.is_empty() {
        return Err(validation_err("name must not be empty"));
    }
    if name.len() > MAX_NAME_LENGTH {
        return Err(validation_err(format!(
            "name exceeds max length {MAX_NAME_LENGTH}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_ids(ids: &[Id]) -> Result<(), EngineError> {
    if ids.is_empty() {
        return Err(validation_err("ids must not be empty"));
    }
    if ids.len() > MAX_BATCH_SIZE {
        return Err(validation_err(format!(
            "batch size {} exceeds max {MAX_BATCH_SIZE}",
            ids.len()
        )));
    }
    for id in ids {
        if id.is_empty() {
            return Err(validation_err("id must not be empty"));
        }
        if id.len() > MAX_ID_LENGTH {
            return Err(validation_err(format!(
                "id exceeds max length {MAX_ID_LENGTH}: {id}"
            )));
        }
    }
    Ok(())
}

pub fn validate_embedding_slice(emb: &[f32]) -> Result<(), EngineError> {
    if emb.is_empty() {
        return Err(validation_err("embedding must not be empty"));
    }
    if emb.len() > MAX_DIMENSION {
        return Err(validation_err(format!(
            "embedding dimension {} exceeds max {MAX_DIMENSION}",
            emb.len()
        )));
    }
    if !emb.iter().all(|v| v.is_finite()) {
        return Err(validation_err(
            "embedding contains non-finite value (NaN or Infinity)",
        ));
    }
    Ok(())
}

pub(crate) fn validate_embeddings(embeddings: &[Vec<f32>]) -> Result<(), EngineError> {
    if embeddings.is_empty() {
        return Err(validation_err("embeddings must not be empty"));
    }
    for emb in embeddings {
        validate_embedding_slice(emb)?;
    }
    Ok(())
}

pub(crate) fn validate_add_request(req: &AddRequest) -> Result<(), EngineError> {
    validate_ids(&req.ids)?;
    validate_embeddings(&req.embeddings)?;
    if req.ids.len() != req.embeddings.len() {
        return Err(validation_err(format!(
            "ids len {} != embeddings len {}",
            req.ids.len(),
            req.embeddings.len()
        )));
    }
    if let Some(ref v) = req.metadatas
        && v.len() != req.ids.len()
    {
        return Err(validation_err(format!(
            "metadatas len {} != ids len {}",
            v.len(),
            req.ids.len()
        )));
    }
    if let Some(ref v) = req.documents
        && v.len() != req.ids.len()
    {
        return Err(validation_err(format!(
            "documents len {} != ids len {}",
            v.len(),
            req.ids.len()
        )));
    }
    Ok(())
}

pub(crate) fn validate_upsert_request(req: &UpsertRequest) -> Result<(), EngineError> {
    validate_ids(&req.ids)?;
    validate_embeddings(&req.embeddings)?;
    if req.ids.len() != req.embeddings.len() {
        return Err(validation_err(format!(
            "ids len {} != embeddings len {}",
            req.ids.len(),
            req.embeddings.len()
        )));
    }
    if let Some(ref v) = req.metadatas
        && v.len() != req.ids.len()
    {
        return Err(validation_err(format!(
            "metadatas len {} != ids len {}",
            v.len(),
            req.ids.len()
        )));
    }
    if let Some(ref v) = req.documents
        && v.len() != req.ids.len()
    {
        return Err(validation_err(format!(
            "documents len {} != ids len {}",
            v.len(),
            req.ids.len()
        )));
    }
    Ok(())
}

pub(crate) fn validate_query_request(req: &QueryRequest) -> Result<(), EngineError> {
    if req.query_embeddings.is_empty() {
        return Err(validation_err("query_embeddings must not be empty"));
    }
    if req.query_embeddings.len() > MAX_QUERY_EMBEDDINGS {
        return Err(validation_err(format!(
            "query_embeddings len {} exceeds max {MAX_QUERY_EMBEDDINGS}",
            req.query_embeddings.len()
        )));
    }
    if req.n_results == 0 {
        return Err(validation_err("n_results must be >= 1"));
    }
    if req.n_results > MAX_N_RESULTS {
        return Err(validation_err(format!(
            "n_results {} exceeds max {MAX_N_RESULTS}",
            req.n_results
        )));
    }
    for emb in &req.query_embeddings {
        validate_embedding_slice(emb)?;
    }
    Ok(())
}

pub(crate) fn validate_delete_request(req: &DeleteRequest) -> Result<(), EngineError> {
    validate_ids(&req.ids)
}

pub(crate) fn validate_update_request(req: &UpdateRequest) -> Result<(), EngineError> {
    validate_ids(&req.ids)?;
    if let Some(ref embs) = req.embeddings {
        if embs.len() != req.ids.len() {
            return Err(validation_err(format!(
                "embeddings len {} != ids len {}",
                embs.len(),
                req.ids.len()
            )));
        }
        for emb in embs {
            validate_embedding_slice(emb.as_ref())?;
        }
    }
    if let Some(ref v) = req.metadatas
        && v.len() != req.ids.len()
    {
        return Err(validation_err(format!(
            "metadatas len {} != ids len {}",
            v.len(),
            req.ids.len()
        )));
    }
    if let Some(ref v) = req.documents
        && v.len() != req.ids.len()
    {
        return Err(validation_err(format!(
            "documents len {} != ids len {}",
            v.len(),
            req.ids.len()
        )));
    }
    Ok(())
}

/// Shared helper: map EngineError variants to HTTP status.
pub(crate) fn engine_error_status(err: &EngineError) -> StatusCode {
    match err {
        EngineError::CollectionNotFound(_) => StatusCode::NOT_FOUND,
        // A catalog identity collision is the caller's conflict, not a server
        // fault. Chroma answers a duplicate collection name with 409; returning
        // 500 made a retriable race look like a crash.
        EngineError::Catalog(rekha_storage::CatalogError::AlreadyExists) => StatusCode::CONFLICT,
        EngineError::Catalog(rekha_storage::CatalogError::NotFound) => StatusCode::NOT_FOUND,
        EngineError::Catalog(rekha_storage::CatalogError::InvalidArgument(_)) => {
            StatusCode::BAD_REQUEST
        }
        EngineError::Validation(_) => StatusCode::BAD_REQUEST,
        EngineError::Forbidden(_) => StatusCode::FORBIDDEN,
        // Retryable, and never a number: a count that cannot be fetched is not
        // zero.
        EngineError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

// ---------------------------------------------------------------------------
// Request / Response DTOs
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dimension: usize,
    #[serde(default = "default_distance")]
    pub distance: Distance,
    #[serde(default)]
    pub partitioned: Option<bool>,
}

fn default_distance() -> Distance {
    Distance::L2
}

#[derive(Serialize)]
pub struct CollectionResponse {
    pub id: Uuid,
    pub name: String,
    pub dimension: usize,
    pub distance: Distance,
    /// Live record count, or `null` when no replica could answer.
    ///
    /// `null` rather than `0`: a list that reports zero for a collection it
    /// could not reach is a wrong answer that looks like a right one.
    pub count: Option<u64>,
    /// The shard this collection was placed on, if the deployment is sharded.
    ///
    /// Exposed because it is the answer to "where is my data", which an
    /// operator otherwise has to reconstruct by hashing — the thing this field
    /// exists to stop being the source of truth.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shard: Option<u32>,
}

#[derive(Deserialize)]
pub struct AddRequest {
    pub ids: Vec<Id>,
    pub embeddings: Vec<Vec<f32>>,
    pub metadatas: Option<Vec<Option<Metadata>>>,
    pub documents: Option<Vec<Option<String>>>,
}

#[derive(Deserialize)]
pub struct UpdateRequest {
    pub ids: Vec<Id>,
    /// Optional: a metadata-only update omits this (or sends null).
    #[serde(default)]
    pub embeddings: Option<Vec<Embedding>>,
    pub metadatas: Option<Vec<Option<Metadata>>>,
    pub documents: Option<Vec<Option<String>>>,
}

#[derive(Deserialize)]
pub struct UpsertRequest {
    pub ids: Vec<Id>,
    pub embeddings: Vec<Vec<f32>>,
    pub metadatas: Option<Vec<Option<Metadata>>>,
    pub documents: Option<Vec<Option<String>>>,
}

#[derive(Deserialize)]
pub struct DeleteRequest {
    pub ids: Vec<Id>,
}

#[derive(Deserialize)]
pub struct QueryRequest {
    pub query_embeddings: Vec<Vec<f32>>,
    #[serde(default = "default_k")]
    pub n_results: usize,
    pub where_filter: Option<WhereFilter>,
    /// Chroma's document-text eligibility filter. Absent means unfiltered.
    #[serde(default)]
    pub where_document: Option<rekha_core::filter::WhereDocument>,
    #[serde(default)]
    pub include: Vec<String>,
}

fn default_k() -> usize {
    10
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub ids: Vec<Vec<String>>,
    pub distances: Option<Vec<Vec<f32>>>,
    pub metadatas: Option<Vec<Vec<Option<Metadata>>>>,
    pub documents: Option<Vec<Vec<Option<String>>>>,
}

#[derive(Deserialize)]
pub struct GetRequest {
    pub ids: Option<Vec<Id>>,
    pub where_filter: Option<WhereFilter>,
    #[serde(default)]
    pub where_document: Option<rekha_core::filter::WhereDocument>,
    #[serde(default)]
    pub include: Vec<String>,
}

#[derive(Serialize)]
pub struct GetResponse {
    pub ids: Vec<String>,
    pub embeddings: Option<Vec<Vec<f32>>>,
    pub metadatas: Option<Vec<Option<Metadata>>>,
    pub documents: Option<Vec<Option<String>>>,
}

#[derive(Serialize)]
pub struct CountResponse {
    pub count: u64,
}

// ---------------------------------------------------------------------------
// Internal cluster endpoints (WAL shipping + Raft RPCs)
// ---------------------------------------------------------------------------

async fn raft_append_entries(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(group): axum::extract::Path<u32>,
    Json(req): Json<AppendEntriesRequest<RaftTypeConfig>>,
) -> Result<Json<AppendEntriesResponse<u64>>, StatusCode> {
    let raft = state.raft_for(group).ok_or(StatusCode::NOT_FOUND)?;
    raft.append_entries(req)
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn raft_vote(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(group): axum::extract::Path<u32>,
    Json(req): Json<VoteRequest<u64>>,
) -> Result<Json<VoteResponse<u64>>, StatusCode> {
    let raft = state.raft_for(group).ok_or(StatusCode::NOT_FOUND)?;
    raft.vote(req)
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

#[allow(clippy::type_complexity)]
async fn raft_install_snapshot(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(group): axum::extract::Path<u32>,
    Json(req): Json<(
        openraft::Vote<u64>,
        openraft::SnapshotMeta<u64, u64>,
        Vec<u8>,
    )>,
) -> Result<Json<openraft::raft::SnapshotResponse<u64>>, StatusCode> {
    let raft = state.raft_for(group).ok_or(StatusCode::NOT_FOUND)?;
    let (vote, meta, data) = req;
    let snapshot = openraft::Snapshot {
        meta,
        snapshot: Box::new(std::io::Cursor::new(data)),
    };
    raft.install_full_snapshot(vote, snapshot)
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// Receive one chunk of a snapshot and, on the last, install it.
///
/// Raw bytes rather than JSON: the previous endpoint took the whole archive in a
/// `Vec<u8>` field, base64-encoded, capped by the router's 32 MiB body limit,
/// with a 200 ms deadline. A replica that had fallen behind could not send one,
/// and this is the only path that repairs such a replica.
///
/// Staging is on disk under the data directory, at an exact offset, so a
/// half-transferred snapshot costs disk rather than memory and a retried chunk
/// is refused instead of being appended twice. The install itself still reads
/// the archive into memory (`SnapshotData` is a `Cursor<Vec<u8>>`); removing
/// that copy is the streaming follow-up, and is why this is chunked transport,
/// not chunked restore.
async fn raft_snapshot_chunk(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((group, snapshot_id)): axum::extract::Path<(u32, String)>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<openraft::raft::SnapshotResponse<u64>>, (StatusCode, Json<serde_json::Value>)> {
    fn bad(detail: impl Into<String>) -> (StatusCode, Json<serde_json::Value>) {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": detail.into() })),
        )
    }

    fn header_u64(
        headers: &axum::http::HeaderMap,
        name: &str,
    ) -> Result<u64, (StatusCode, Json<serde_json::Value>)> {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok())
            .ok_or_else(|| bad(format!("`{name}` is required and must be a whole number")))
    }

    // The id becomes part of a file name in the staging directory.
    if snapshot_id.is_empty()
        || snapshot_id.contains('/')
        || snapshot_id.contains('\\')
        || snapshot_id.contains("..")
    {
        return Err(bad("snapshot id is not a usable file name"));
    }

    let offset = header_u64(&headers, rekha_cluster::network::SNAPSHOT_OFFSET_HEADER)?;
    let total = header_u64(&headers, rekha_cluster::network::SNAPSHOT_TOTAL_HEADER)?;
    let is_final = headers
        .get(rekha_cluster::network::SNAPSHOT_FINAL_HEADER)
        .and_then(|v| v.to_str().ok())
        == Some("1");

    let vote: openraft::Vote<u64> = headers
        .get(rekha_cluster::network::SNAPSHOT_VOTE_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| serde_json::from_str(v).ok())
        .ok_or_else(|| bad("`x-rekha-snapshot-vote` is required and must be JSON"))?;
    let meta: openraft::SnapshotMeta<u64, u64> = headers
        .get(rekha_cluster::network::SNAPSHOT_META_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| serde_json::from_str(v).ok())
        .ok_or_else(|| bad("`x-rekha-snapshot-meta` is required and must be JSON"))?;

    let end = offset
        .checked_add(body.len() as u64)
        .ok_or_else(|| bad("chunk offset overflows"))?;
    if end > total {
        return Err(bad(format!(
            "chunk covers bytes {offset}..{end}, beyond the declared total of {total}"
        )));
    }
    if is_final && end != total {
        return Err(bad(format!(
            "the final chunk ends at byte {end}, but the snapshot is declared as {total} bytes"
        )));
    }

    let raft = state.raft_for(group).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("no Raft group {group} on this node") })),
        )
    })?;

    let staging_dir = state
        .engine
        .storage_dir()
        .join("snapshots")
        .join(".incoming");
    std::fs::create_dir_all(&staging_dir).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                serde_json::json!({ "error": format!("cannot create the staging directory: {e}") }),
            ),
        )
    })?;
    let staged = staging_dir.join(format!("{group}-{snapshot_id}.part"));

    // One transfer at a time: appends are offset-checked against the file, and
    // two concurrent chunks would both observe the same length.
    let _guard = state.snapshot_staging.lock().await;

    let existing = std::fs::metadata(&staged).map(|m| m.len()).unwrap_or(0);
    // Offset zero is a sender starting over, which is exactly what openraft does
    // after a failed transfer: discard the partial file rather than answering
    // 409 and costing a whole extra attempt.
    let existing = if offset == 0 && existing > 0 {
        let _ = std::fs::remove_file(&staged);
        0
    } else {
        existing
    };
    if offset != existing {
        // Start over rather than splice a chunk into the wrong place. The next
        // attempt begins at zero, which is what openraft does after a failure.
        let _ = std::fs::remove_file(&staged);
        return Err((
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": format!(
                    "chunk starts at byte {offset}, but {existing} bytes are already staged; \
                     the transfer must restart from zero"
                )
            })),
        ));
    }

    {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&staged)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": format!("cannot stage the chunk: {e}") })),
                )
            })?;
        file.write_all(&body).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("cannot write the chunk: {e}") })),
            )
        })?;
    }

    if !is_final {
        // No meta is returned for an intermediate chunk: openraft only reads the
        // response to the last one.
        return Ok(Json(openraft::raft::SnapshotResponse::new(vote)));
    }

    let read_path = staged.clone();
    let data = tokio::task::spawn_blocking(move || std::fs::read(&read_path))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("staging task failed: {e}") })),
            )
        })?
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    serde_json::json!({ "error": format!("cannot read the staged snapshot: {e}") }),
                ),
            )
        })?;

    // The staged file is only needed until it is installed. Leaving it would
    // double the disk a snapshot costs, permanently, because nothing else walks
    // this directory.
    let _ = std::fs::remove_file(&staged);

    let snapshot = openraft::Snapshot {
        meta,
        snapshot: Box::new(std::io::Cursor::new(data)),
    };
    raft.install_full_snapshot(vote, snapshot)
        .await
        .map(Json)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("snapshot install failed: {e}") })),
            )
        })
}

async fn shard_ensure_started(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
) -> Result<StatusCode, StatusCode> {
    let shard = rekha_cluster::ShardId(shard_num);
    let Some(shards) = state.shards.as_ref() else {
        return Err(StatusCode::NOT_FOUND);
    };
    if shards.hosts(shard) {
        return Ok(StatusCode::OK);
    }

    if let (Some(raft_config), Some(node_id)) = (&state.raft_config, state.node_id) {
        let wal_dir = state.engine.storage_dir();
        let data_dir = wal_dir.parent().unwrap_or(wal_dir);
        let log_dir = data_dir.join("raft_log");
        let _ = std::fs::create_dir_all(&log_dir);
        let mut log_store = match rekha_cluster::RedbLogStore::open(
            log_dir.join(format!("shard-{}.redb", shard.0)),
        ) {
            Ok(s) => s,
            Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
        };
        let purge_watermark = log_store.purge_watermark().await;
        let mut shard_sm = rekha_cluster::ClusterStateMachine::for_shard(node_id, shard);
        shard_sm.set_catalog(state.engine.catalog());
        shard_sm.set_engine(state.engine.clone() as Arc<dyn rekha_cluster::EngineSnapshotProvider>);
        shard_sm.set_apply_failure_flag(state.apply_failure.clone());
        let snapshot_root = wal_dir.join("snapshots");
        if let Ok(store) = rekha_cluster::SnapshotStore::open(
            snapshot_root.join(format!("shard-{}", shard.0)),
            shard_sm.group_label(),
        ) {
            shard_sm.set_snapshot_store(Arc::new(store));
        }

        if let Some(boundary) = purge_watermark {
            shard_sm.seed_nodes(
                state
                    .peers
                    .ids()
                    .into_iter()
                    .filter_map(|id| state.peers.addr(id).map(|addr| (id, addr))),
            );
            let _ = shard_sm.seed_placements(state.engine.catalog().as_ref());
            let voters = shards.owners(shard);
            shard_sm.resume_from(
                Some(boundary),
                openraft::StoredMembership::new(
                    Some(boundary),
                    openraft::Membership::new(vec![voters.into_iter().collect()], ()),
                ),
            );
        }

        let network = rekha_cluster::RaftNetworkFactoryImpl::new(state.peers.clone())
            .with_group(rekha_cluster::ShardCluster::group_id(shard))
            .with_cluster_secret(state.cluster_secret.clone());

        let shard_raft = match openraft::Raft::<rekha_cluster::RaftTypeConfig>::new(
            node_id,
            raft_config.clone(),
            network,
            log_store,
            shard_sm,
        )
        .await
        {
            Ok(r) => Arc::new(r),
            Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
        };

        shards.insert(shard, shard_raft);
        let _ = state.engine.materialize_shard(shard);
        return Ok(StatusCode::OK);
    }

    let _ = state.engine.materialize_shard(shard);
    Ok(StatusCode::OK)
}

async fn shard_drop(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
) -> Result<StatusCode, StatusCode> {
    let shard = rekha_cluster::ShardId(shard_num);
    if let Some(shards) = state.shards.as_ref()
        && let Some(raft) = shards.remove(shard)
    {
        let _ = raft.shutdown().await;
    }
    let _ = state.engine.drop_shard(shard);
    Ok(StatusCode::OK)
}

async fn shard_batch_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
    Json(req): Json<ShardBatchRequest>,
) -> Result<Json<ShardBatchResponse>, (StatusCode, Json<serde_json::Value>)> {
    let shard = rekha_cluster::ShardId(shard_num);
    let count = match commit_write_for_shard(&state, shard, req.collection_id, req.kind).await {
        Ok(c) => c,
        Err(ApiError::NotLeader { leader }) => {
            let mut headers = axum::http::HeaderMap::new();
            if let Some(id) = leader
                && let Ok(v) = axum::http::HeaderValue::from_str(&id.to_string())
            {
                headers.insert(
                    axum::http::HeaderName::from_static(leadership::LEADER_HEADER),
                    v,
                );
            }
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": "not shard leader", "leader": leader })),
            ));
        }
        Err(ApiError::Engine(e)) => {
            return Err((
                engine_error_status(&e),
                Json(serde_json::json!({ "error": e.to_string() })),
            ));
        }
    };
    Ok(Json(ShardBatchResponse { count }))
}

async fn shard_prepare_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
    Json(req): Json<ShardPrepareRequest>,
) -> Result<Json<ShardBatchResponse>, (StatusCode, Json<serde_json::Value>)> {
    let shard = rekha_cluster::ShardId(shard_num);
    let count = match prepare_write_for_shard(&state, shard, req.tx_id, req.collection_id, req.kind)
        .await
    {
        Ok(c) => c,
        Err(ApiError::NotLeader { leader }) => {
            let mut headers = axum::http::HeaderMap::new();
            if let Some(id) = leader
                && let Ok(v) = axum::http::HeaderValue::from_str(&id.to_string())
            {
                headers.insert(
                    axum::http::HeaderName::from_static(leadership::LEADER_HEADER),
                    v,
                );
            }
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": "not shard leader", "leader": leader })),
            ));
        }
        Err(ApiError::Engine(e)) => {
            return Err((
                engine_error_status(&e),
                Json(serde_json::json!({ "error": e.to_string() })),
            ));
        }
    };
    Ok(Json(ShardBatchResponse { count }))
}

async fn shard_commit_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
    Json(req): Json<ShardCommitRequest>,
) -> Result<StatusCode, StatusCode> {
    let shard = rekha_cluster::ShardId(shard_num);
    match commit_tx_for_shard(&state, shard, req.tx_id).await {
        Ok(()) => Ok(StatusCode::OK),
        Err(ApiError::NotLeader { .. }) => Err(StatusCode::SERVICE_UNAVAILABLE),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn shard_abort_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
    Json(req): Json<ShardAbortRequest>,
) -> StatusCode {
    let shard = rekha_cluster::ShardId(shard_num);
    let _ = abort_tx_for_shard(&state, shard, req.tx_id).await;
    StatusCode::OK
}

async fn tx_status_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(tx_id): axum::extract::Path<Uuid>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let Some(shards) = state.shards.as_ref() else {
        return Err(StatusCode::NOT_FOUND);
    };
    let committed = shards.is_tx_committed(&tx_id);
    Ok(Json(serde_json::json!({ "committed": committed })))
}

async fn tx_commit_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TxCommitRequest>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    dispatch_metadata_commit_tx(&state, req.tx_id, req.shards)
        .await
        .map(|_| StatusCode::OK)
        .map_err(|e| match e {
            ApiError::NotLeader { leader } => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": "not metadata leader", "leader": leader })),
            ),
            ApiError::Engine(e) => (
                engine_error_status(&e),
                Json(serde_json::json!({ "error": e.to_string() })),
            ),
        })
}

async fn tx_abort_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TxAbortRequest>,
) -> StatusCode {
    dispatch_metadata_abort_tx(&state, req.tx_id).await;
    StatusCode::OK
}

async fn shard_membership_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
) -> Result<Json<Vec<u64>>, StatusCode> {
    let shard = rekha_cluster::ShardId(shard_num);
    let Some(shards) = state.shards.as_ref() else {
        return Err(StatusCode::NOT_FOUND);
    };
    let Some(raft) = shards.raft(shard) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let m = raft
        .metrics()
        .borrow()
        .membership_config
        .membership()
        .clone();
    let voters: Vec<u64> = m.voter_ids().collect();
    Ok(Json(voters))
}

async fn shard_query_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
    Json(req): Json<ShardQueryRequest>,
) -> Result<Json<Vec<Vec<rekha_engine::ScoredPoint>>>, StatusCode> {
    let shard = rekha_cluster::ShardId(shard_num);
    let Some(shards) = state.shards.as_ref() else {
        return Err(StatusCode::NOT_FOUND);
    };
    let Some(raft) = shards.raft(shard) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let leads = raft.metrics().borrow().current_leader == Some(raft.metrics().borrow().id);
    if !leads {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    let _ = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        raft.ensure_linearizable(),
    )
    .await;
    let engine = state.engine.clone();
    let collection_id = req.collection_id;
    let query_embeddings = req.query_embeddings;
    let n_results = req.n_results;
    let where_filter = req.where_filter;
    let where_document = req.where_document;
    let with_embeddings = req.with_embeddings;

    let hits = offload(move || {
        let opts = QueryOptions {
            ef: 0,
            where_filter,
            where_document,
            oversampling: 4,
        };
        let embeddings_vec: Vec<Embedding> =
            query_embeddings.into_iter().map(Embedding::from).collect();
        engine.query_many_partition(
            &collection_id,
            Some(shard),
            &embeddings_vec,
            n_results,
            &opts,
            with_embeddings,
        )
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(hits))
}

async fn shard_count_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
    Json(collection_id): Json<Uuid>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let shard = rekha_cluster::ShardId(shard_num);
    let Some(shards) = state.shards.as_ref() else {
        return Err(StatusCode::NOT_FOUND);
    };
    let Some(raft) = shards.raft(shard) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let leads = raft.metrics().borrow().current_leader == Some(raft.metrics().borrow().id);
    let local = params
        .get("consistency")
        .map(|v| v == "local")
        .unwrap_or(false);
    if !leads && !local {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    if leads && !local {
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            raft.ensure_linearizable(),
        )
        .await;
    }
    let count = state
        .engine
        .count_partition(&collection_id, Some(shard))
        .map_err(|_| StatusCode::NOT_FOUND)?;
    Ok(Json(serde_json::json!({ "count": count })))
}

async fn shard_get_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
    Json(req): Json<ShardGetRequest>,
) -> Result<Json<Vec<rekha_engine::Record>>, StatusCode> {
    let shard = rekha_cluster::ShardId(shard_num);
    let Some(shards) = state.shards.as_ref() else {
        return Err(StatusCode::NOT_FOUND);
    };
    let Some(raft) = shards.raft(shard) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let leads = raft.metrics().borrow().current_leader == Some(raft.metrics().borrow().id);
    if !leads {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    let _ = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        raft.ensure_linearizable(),
    )
    .await;
    let records = state
        .engine
        .get_records_partition(
            &req.collection_id,
            Some(shard),
            req.ids.as_deref(),
            req.where_filter.as_ref(),
            req.where_document.as_ref(),
            req.limit,
            req.offset,
        )
        .map_err(|_| StatusCode::NOT_FOUND)?;
    Ok(Json(records))
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

async fn create_collection(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Json(req): Json<CreateCollectionRequest>,
) -> Result<Json<CollectionResponse>, ApiError> {
    // Per-tenant/database check for create
    if !rekha_core::auth::has_permission(
        &auth.permissions,
        "collection.create",
        Some(&state.tenant),
        Some(&state.database),
    ) && !rekha_core::auth::has_permission(
        &auth.permissions,
        "collection.create",
        Some(&state.tenant),
        None,
    ) && !rekha_core::auth::has_permission(&auth.permissions, "collection.create", None, None)
    {
        return Err(ApiError::Engine(EngineError::Forbidden(
            "missing permission collection.create".into(),
        )));
    }
    validate_name(&req.name)?;
    if req.dimension > MAX_DIMENSION {
        return Err(ApiError::Engine(validation_err(format!(
            "dimension {} exceeds max {MAX_DIMENSION}",
            req.dimension
        ))));
    }
    let mut config = CollectionConfig::new(req.name.clone(), req.dimension, req.distance);
    config.tenant = state.tenant.clone();
    config.database = state.database.clone();
    let partitioned = is_partitioned_requested(req.partitioned, None);
    let record = create_collection_clustered(&state, &config, partitioned).await?;
    // Known to be empty by construction, and a fresh collection answers `Some(0)`
    // rather than `None`: this is a fact the server just created, not a count it
    // failed to fetch.
    let count = Some(0);

    Ok(Json(CollectionResponse {
        id: record.config.id,
        name: record.config.name,
        dimension: record.config.dimension,
        distance: record.config.space,
        count,
        shard: record.shard.map(|shard| shard.0),
    }))
}

async fn list_collections(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
) -> Result<Json<Vec<CollectionResponse>>, ApiError> {
    if !rekha_core::auth::has_permission(
        &auth.permissions,
        "collection.list",
        Some(&state.tenant),
        Some(&state.database),
    ) && !rekha_core::auth::has_permission(
        &auth.permissions,
        "collection.list",
        Some(&state.tenant),
        None,
    ) && !rekha_core::auth::has_permission(&auth.permissions, "collection.list", None, None)
    {
        return Err(ApiError::Engine(EngineError::Forbidden(
            "missing permission collection.list".into(),
        )));
    }
    let records = state
        .engine
        .list_collections(&state.tenant, &state.database)?;
    let ids: Vec<uuid::Uuid> = records.iter().map(|r| r.config.id).collect();
    // One unreachable shard owner must not fail the whole list, so a missing
    // count is reported as `null` for that row instead of aborting the response
    // or inventing a zero.
    let counts = counts_for(&state, &ids).await.unwrap_or_default();
    let mut out = Vec::with_capacity(records.len());
    for record in records {
        let count = counts.get(&record.config.id).copied();
        out.push(CollectionResponse {
            id: record.config.id,
            name: record.config.name,
            dimension: record.config.dimension,
            distance: record.config.space,
            count,
            shard: record.shard.map(|shard| shard.0),
        });
    }
    Ok(Json(out))
}

async fn get_collection(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
) -> Result<Json<CollectionResponse>, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "collection.get",
        &id,
    )?;
    let record = state
        .engine
        .get_collection_by_id(&id)?
        .ok_or_else(|| EngineError::CollectionNotFound(id.to_string()))?;
    let count = counts_for(&state, &[id])
        .await?
        .get(&id)
        .copied()
        .ok_or_else(|| {
            ApiError::Engine(EngineError::Unavailable(format!(
                "no replica answered the count for collection {id}"
            )))
        })?;
    Ok(Json(CollectionResponse {
        id: record.config.id,
        name: record.config.name,
        dimension: record.config.dimension,
        distance: record.config.space,
        count: Some(count),
        shard: record.shard.map(|shard| shard.0),
    }))
}

async fn delete_collection(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "collection.delete",
        &id,
    )?;
    delete_collection_clustered(&state, id).await?;
    Ok(StatusCode::OK)
}

async fn add_records(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
    Json(req): Json<AddRequest>,
) -> Result<StatusCode, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "record.add",
        &id,
    )?;
    validate_add_request(&req)?;

    if is_partitioned(&state, &id) {
        let Some(shards) = state.shards.as_ref() else {
            return Err(ApiError::Engine(EngineError::Unavailable(
                "no shards configured".into(),
            )));
        };
        let placement = shards.placement();
        let mut sub_batches: HashMap<rekha_cluster::ShardId, SubBatchAdd> = HashMap::new();
        for (i, record_id) in req.ids.iter().enumerate() {
            let shard = placement.shard_for_record(record_id);
            let entry = sub_batches.entry(shard).or_insert_with(|| {
                (
                    Vec::new(),
                    Vec::new(),
                    req.metadatas.as_ref().map(|_| Vec::new()),
                    req.documents.as_ref().map(|_| Vec::new()),
                )
            });
            entry.0.push(record_id.clone());
            entry.1.push(req.embeddings[i].clone());
            if let Some(ref mut metas) = entry.2 {
                metas.push(
                    req.metadatas
                        .as_ref()
                        .and_then(|m| m.get(i).cloned())
                        .flatten(),
                );
            }
            if let Some(ref mut docs) = entry.3 {
                docs.push(
                    req.documents
                        .as_ref()
                        .and_then(|d| d.get(i).cloned())
                        .flatten(),
                );
            }
        }
        let batches: Vec<(rekha_cluster::ShardId, ShardWriteKind)> = sub_batches
            .into_iter()
            .map(|(shard, (ids, embs, metas, docs))| {
                (
                    shard,
                    ShardWriteKind::Add {
                        ids,
                        embeddings: embs,
                        metadatas: metas,
                        documents: docs,
                    },
                )
            })
            .collect();
        write_partitioned_2pc(&state, id, batches).await?;
        return Ok(StatusCode::OK);
    }

    let shard = state.shard_of(&id);
    let embeddings: Vec<Embedding> = req.embeddings.into_iter().map(|e| e.into()).collect();
    let engine = state.engine.clone();
    let ids = req.ids;
    let metadatas = req.metadatas;
    let documents = req.documents;
    write_batch(&state, id, move || {
        engine.prepare_add_partition(
            &id,
            shard,
            &ids,
            &embeddings,
            metadatas.as_deref(),
            documents.as_deref(),
        )
    })
    .await?;
    Ok(StatusCode::OK)
}

async fn upsert_records(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpsertRequest>,
) -> Result<StatusCode, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "record.upsert",
        &id,
    )?;
    validate_upsert_request(&req)?;

    if is_partitioned(&state, &id) {
        let Some(shards) = state.shards.as_ref() else {
            return Err(ApiError::Engine(EngineError::Unavailable(
                "no shards configured".into(),
            )));
        };
        let placement = shards.placement();
        let mut sub_batches: HashMap<rekha_cluster::ShardId, SubBatchUpsert> = HashMap::new();
        for (i, record_id) in req.ids.iter().enumerate() {
            let shard = placement.shard_for_record(record_id);
            let entry = sub_batches.entry(shard).or_insert_with(|| {
                (
                    Vec::new(),
                    Vec::new(),
                    req.metadatas.as_ref().map(|_| Vec::new()),
                    req.documents.as_ref().map(|_| Vec::new()),
                )
            });
            entry.0.push(record_id.clone());
            entry.1.push(req.embeddings[i].clone());
            if let Some(ref mut metas) = entry.2 {
                metas.push(
                    req.metadatas
                        .as_ref()
                        .and_then(|m| m.get(i).cloned())
                        .flatten()
                        .map(rekha_core::types::metadata_patch_from),
                );
            }
            if let Some(ref mut docs) = entry.3 {
                docs.push(
                    req.documents
                        .as_ref()
                        .and_then(|d| d.get(i).cloned())
                        .flatten(),
                );
            }
        }
        let batches: Vec<(rekha_cluster::ShardId, ShardWriteKind)> = sub_batches
            .into_iter()
            .map(|(shard, (ids, embs, metas, docs))| {
                (
                    shard,
                    ShardWriteKind::Upsert {
                        ids,
                        embeddings: embs,
                        metadatas: metas,
                        documents: docs,
                    },
                )
            })
            .collect();
        write_partitioned_2pc(&state, id, batches).await?;
        return Ok(StatusCode::OK);
    }

    let shard = state.shard_of(&id);
    let embeddings: Vec<Embedding> = req.embeddings.into_iter().map(|e| e.into()).collect();
    let metadatas = req.metadatas.map(|list| {
        list.into_iter()
            .map(|o| o.map(rekha_core::types::metadata_patch_from))
            .collect::<Vec<_>>()
    });
    let engine = state.engine.clone();
    let ids = req.ids;
    let documents = req.documents;
    write_batch(&state, id, move || {
        engine.prepare_upsert_partition(
            &id,
            shard,
            &ids,
            &embeddings,
            metadatas.as_deref(),
            documents.as_deref(),
        )
    })
    .await?;
    Ok(StatusCode::OK)
}

async fn update_records(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateRequest>,
) -> Result<StatusCode, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "record.update",
        &id,
    )?;
    validate_update_request(&req)?;

    if is_partitioned(&state, &id) {
        let Some(shards) = state.shards.as_ref() else {
            return Err(ApiError::Engine(EngineError::Unavailable(
                "no shards configured".into(),
            )));
        };
        let placement = shards.placement();
        let mut sub_batches: HashMap<rekha_cluster::ShardId, SubBatchUpdate> = HashMap::new();
        for (i, record_id) in req.ids.iter().enumerate() {
            let shard = placement.shard_for_record(record_id);
            let entry = sub_batches.entry(shard).or_insert_with(|| {
                (
                    Vec::new(),
                    req.embeddings.as_ref().map(|_| Vec::new()),
                    req.metadatas.as_ref().map(|_| Vec::new()),
                    req.documents.as_ref().map(|_| Vec::new()),
                )
            });
            entry.0.push(record_id.clone());
            if let (Some(embs), Some(src_embs)) = (&mut entry.1, &req.embeddings) {
                embs.push(src_embs[i].to_vec());
            }
            if let Some(ref mut metas) = entry.2 {
                metas.push(
                    req.metadatas
                        .as_ref()
                        .and_then(|m| m.get(i).cloned())
                        .flatten()
                        .map(rekha_core::types::metadata_patch_from),
                );
            }
            if let Some(ref mut docs) = entry.3 {
                docs.push(
                    req.documents
                        .as_ref()
                        .and_then(|d| d.get(i).cloned())
                        .flatten(),
                );
            }
        }
        let batches: Vec<(rekha_cluster::ShardId, ShardWriteKind)> = sub_batches
            .into_iter()
            .map(|(shard, (ids, embs, metas, docs))| {
                (
                    shard,
                    ShardWriteKind::Update {
                        ids,
                        embeddings: embs,
                        metadatas: metas,
                        documents: docs,
                    },
                )
            })
            .collect();
        write_partitioned_2pc(&state, id, batches).await?;
        return Ok(StatusCode::OK);
    }

    let shard = state.shard_of(&id);
    let metadatas = req.metadatas.map(|list| {
        list.into_iter()
            .map(|o| o.map(rekha_core::types::metadata_patch_from))
            .collect::<Vec<_>>()
    });
    let engine = state.engine.clone();
    let ids = req.ids;
    let embeddings = req.embeddings;
    let documents = req.documents;
    write_batch(&state, id, move || {
        engine.prepare_update_partition(
            &id,
            shard,
            &ids,
            embeddings.as_deref(),
            metadatas.as_deref(),
            documents.as_deref(),
        )
    })
    .await?;
    Ok(StatusCode::OK)
}

async fn delete_records(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
    Json(req): Json<DeleteRequest>,
) -> Result<StatusCode, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "record.delete",
        &id,
    )?;
    validate_delete_request(&req)?;

    if is_partitioned(&state, &id) {
        let Some(shards) = state.shards.as_ref() else {
            return Err(ApiError::Engine(EngineError::Unavailable(
                "no shards configured".into(),
            )));
        };
        let placement = shards.placement();
        let mut sub_batches: HashMap<rekha_cluster::ShardId, Vec<Id>> = HashMap::new();
        for record_id in req.ids {
            let shard = placement.shard_for_record(&record_id);
            sub_batches.entry(shard).or_default().push(record_id);
        }
        let batches: Vec<(rekha_cluster::ShardId, ShardWriteKind)> = sub_batches
            .into_iter()
            .map(|(shard, ids)| (shard, ShardWriteKind::Delete { ids }))
            .collect();
        write_partitioned_2pc(&state, id, batches).await?;
        return Ok(StatusCode::OK);
    }

    let shard = state.shard_of(&id);
    let engine = state.engine.clone();
    let ids = req.ids;
    write_batch(&state, id, move || {
        engine.prepare_delete_partition(&id, shard, &ids)
    })
    .await?;
    Ok(StatusCode::OK)
}

async fn query_records(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "record.query",
        &id,
    )?;
    validate_query_request(&req)?;
    let include_metadatas = req.include.iter().any(|s| s == "metadatas");
    let include_documents = req.include.iter().any(|s| s == "documents");
    let include_distances = req.include.iter().any(|s| s == "distances");

    let all_results: Vec<Vec<rekha_engine::ScoredPoint>> = if is_partitioned(&state, &id) {
        let Some(shards) = state.shards.as_ref() else {
            return Err(ApiError::Engine(EngineError::Unavailable(
                "no shards configured".into(),
            )));
        };
        let shard_count = shards.placement().shard_count();
        let mut futures = Vec::new();
        for s in 0..shard_count {
            let shard = rekha_cluster::ShardId(s);
            let state = state.clone();
            let query_req = ShardQueryRequest {
                collection_id: id,
                query_embeddings: req.query_embeddings.clone(),
                n_results: req.n_results,
                where_filter: req.where_filter.clone(),
                where_document: req.where_document.clone(),
                with_embeddings: false,
            };
            futures.push(async move { dispatch_shard_query(&state, shard, query_req).await });
        }
        let shard_results = futures::future::try_join_all(futures).await?;
        let n_queries = req.query_embeddings.len();
        let mut merged_queries = Vec::with_capacity(n_queries);
        for q in 0..n_queries {
            let mut hits = Vec::new();
            for shard_res in &shard_results {
                if let Some(shard_hits) = shard_res.get(q) {
                    hits.extend(shard_hits.iter().cloned());
                }
            }
            hits.sort_by(|a, b| {
                a.distance
                    .total_cmp(&b.distance)
                    .then_with(|| a.id.cmp(&b.id))
            });
            hits.truncate(req.n_results);
            merged_queries.push(hits);
        }
        merged_queries
    } else {
        let engine = state.engine.clone();
        let query_embeddings = req.query_embeddings.clone();
        let where_filter = req.where_filter.clone();
        let where_document = req.where_document.clone();
        let n_results = req.n_results;
        offload(move || {
            let mut results = Vec::with_capacity(query_embeddings.len());
            for query_emb in &query_embeddings {
                let embedding: Embedding = query_emb.clone().into();
                let opts = QueryOptions {
                    ef: 0,
                    where_filter: where_filter.clone(),
                    where_document: where_document.clone(),
                    oversampling: 4,
                };
                results.push(engine.query(&id, &embedding, n_results, &opts)?);
            }
            Ok(results)
        })
        .await?
    };

    let n_queries = all_results.len();
    let k = all_results.first().map_or(0, |r| r.len());

    let mut ids: Vec<Vec<String>> = Vec::with_capacity(n_queries);
    let mut distances: Option<Vec<Vec<f32>>> = if include_distances {
        Some(Vec::with_capacity(n_queries))
    } else {
        None
    };
    let mut metadatas: Option<Vec<Vec<Option<Metadata>>>> = if include_metadatas {
        Some(Vec::with_capacity(n_queries))
    } else {
        None
    };
    let mut documents: Option<Vec<Vec<Option<String>>>> = if include_documents {
        Some(Vec::with_capacity(n_queries))
    } else {
        None
    };

    for results in &all_results {
        ids.push(results.iter().map(|r| r.id.clone()).collect());
        if let Some(ref mut d) = distances {
            d.push(results.iter().map(|r| r.distance).collect());
        }
        if let Some(ref mut m) = metadatas {
            m.push(results.iter().map(|r| r.metadata.clone()).collect());
        }
        if let Some(ref mut doc) = documents {
            doc.push(results.iter().map(|r| r.document.clone()).collect());
        }
    }

    // Pad shorter result sets to k if needed (Chroma returns Vec<Vec<...>>
    // where inner vecs are all length k, with empty strings / None for missing).
    for vec in ids.iter_mut() {
        let len = vec.len();
        if len < k {
            vec.resize_with(k, String::new);
        }
    }
    if let Some(ref mut d) = distances {
        for vec in d.iter_mut() {
            let len = vec.len();
            if len < k {
                vec.resize_with(k, || f32::INFINITY);
            }
        }
    }
    if let Some(ref mut m) = metadatas {
        for vec in m.iter_mut() {
            let len = vec.len();
            if len < k {
                vec.resize_with(k, || None);
            }
        }
    }
    if let Some(ref mut doc) = documents {
        for vec in doc.iter_mut() {
            let len = vec.len();
            if len < k {
                vec.resize_with(k, || None);
            }
        }
    }

    Ok(Json(QueryResponse {
        ids,
        distances,
        metadatas,
        documents,
    }))
}

async fn get_records(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
    Json(req): Json<GetRequest>,
) -> Result<Json<GetResponse>, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "record.get",
        &id,
    )?;
    let ids = req.ids.unwrap_or_default();
    let include_embeddings = req.include.iter().any(|s| s == "embeddings");

    let records: Vec<Option<rekha_engine::Record>> = if is_partitioned(&state, &id) {
        if ids.is_empty() && req.where_filter.is_none() && req.where_document.is_none() {
            Vec::new()
        } else if !ids.is_empty() {
            let Some(shards) = state.shards.as_ref() else {
                return Err(ApiError::Engine(EngineError::Unavailable(
                    "no shards configured".into(),
                )));
            };
            let placement = shards.placement();
            let mut by_shard: HashMap<rekha_cluster::ShardId, Vec<Id>> = HashMap::new();
            for item_id in &ids {
                let shard = placement.shard_for_record(item_id);
                by_shard.entry(shard).or_default().push(item_id.clone());
            }
            let mut futures = Vec::new();
            for (shard, shard_ids) in by_shard {
                let state = state.clone();
                let get_req = ShardGetRequest {
                    collection_id: id,
                    ids: Some(shard_ids),
                    where_filter: req.where_filter.clone(),
                    where_document: req.where_document.clone(),
                    limit: None,
                    offset: 0,
                };
                futures.push(async move { dispatch_shard_get(&state, shard, get_req).await });
            }
            let shard_records = futures::future::try_join_all(futures).await?;
            let mut record_map: HashMap<Id, rekha_engine::Record> = HashMap::new();
            for list in shard_records {
                for r in list {
                    record_map.insert(r.id.clone(), r);
                }
            }
            ids.iter()
                .map(|item_id| record_map.remove(item_id))
                .collect()
        } else {
            let Some(shards) = state.shards.as_ref() else {
                return Err(ApiError::Engine(EngineError::Unavailable(
                    "no shards configured".into(),
                )));
            };
            let shard_count = shards.placement().shard_count();
            let mut futures = Vec::new();
            for s in 0..shard_count {
                let shard = rekha_cluster::ShardId(s);
                let state = state.clone();
                let get_req = ShardGetRequest {
                    collection_id: id,
                    ids: None,
                    where_filter: req.where_filter.clone(),
                    where_document: req.where_document.clone(),
                    limit: None,
                    offset: 0,
                };
                futures.push(async move { dispatch_shard_get(&state, shard, get_req).await });
            }
            let shard_records = futures::future::try_join_all(futures).await?;
            let mut all_recs = Vec::new();
            for list in shard_records {
                all_recs.extend(list);
            }
            all_recs.sort_by_key(|r| r.seq);
            all_recs.into_iter().map(Some).collect()
        }
    } else if ids.is_empty() && req.where_filter.is_none() && req.where_document.is_none() {
        Vec::new()
    } else if req.where_filter.is_none() && req.where_document.is_none() {
        state.engine.get(&id, &ids)?
    } else {
        state
            .engine
            .get_records(
                &id,
                (!ids.is_empty()).then_some(ids.as_slice()),
                req.where_filter.as_ref(),
                req.where_document.as_ref(),
                None,
                0,
            )?
            .into_iter()
            .map(Some)
            .collect()
    };

    let out_ids: Vec<String> = records
        .iter()
        .filter_map(|r| r.as_ref().map(|r| r.id.clone()))
        .collect();
    let out_embeddings = if include_embeddings {
        Some(
            records
                .iter()
                .filter_map(|r| {
                    r.as_ref()
                        .and_then(|r| r.embedding.as_ref().map(|e| e.iter().copied().collect()))
                })
                .collect(),
        )
    } else {
        None
    };
    let out_metadatas: Vec<Option<Metadata>> = records
        .iter()
        .map(|r| r.as_ref().and_then(|r| r.metadata.clone()))
        .collect();
    let out_documents: Vec<Option<String>> = records
        .iter()
        .map(|r| r.as_ref().and_then(|r| r.document.clone()))
        .collect();

    Ok(Json(GetResponse {
        ids: out_ids,
        embeddings: out_embeddings,
        metadatas: Some(out_metadatas),
        documents: Some(out_documents),
    }))
}

async fn count_records(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
) -> Result<Json<CountResponse>, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "record.count",
        &id,
    )?;
    if is_partitioned(&state, &id) {
        let Some(shards) = state.shards.as_ref() else {
            return Err(ApiError::Engine(EngineError::Unavailable(
                "no shards configured".into(),
            )));
        };
        let shard_count = shards.placement().shard_count();
        let mut futures = Vec::new();
        for s in 0..shard_count {
            let shard = rekha_cluster::ShardId(s);
            let state = state.clone();
            futures.push(async move { dispatch_shard_count(&state, shard, id).await });
        }
        let counts: Vec<u64> = futures::future::try_join_all(futures).await?;
        let total: u64 = counts.into_iter().sum();
        return Ok(Json(CountResponse { count: total }));
    }
    let count = state.engine.count(&id)?;
    Ok(Json(CountResponse { count }))
}

/// POST /api/v2/collections/{id}/optimize — force-flush, rebuild the ANN graph,
/// publish a checkpoint, and prune the WAL.
///
/// Operationally this is how an operator (or a test) forces a checkpoint
/// without waiting for `sync_threshold` writes, which is also the only way to
/// exercise the publication path over HTTP.
async fn optimize_collection(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "collection.optimize",
        &id,
    )?;
    // `optimize` compacts segments and rebuilds the index — measured at 6.5 s
    // for 40k records. Running that on an async worker thread would starve
    // Raft heartbeats for the whole duration and force an election.
    let engine = state.engine.clone();
    let stats = offload(move || engine.optimize(&id)).await?;
    Ok(Json(serde_json::json!({
        "records": stats.records,
        "indexed": stats.indexed,
        "wal_bytes_before": stats.wal_bytes_before,
        "wal_bytes_after": stats.wal_bytes_after,
        "flush_us": stats.flush_us,
        "rebuild_us": stats.rebuild_us,
        "checkpoint": {
            "records_us": stats.checkpoint.records_us,
            "index_us": stats.checkpoint.index_us,
            "vectors_us": stats.checkpoint.vectors_us,
            "index_map_us": stats.checkpoint.index_map_us,
            "manifest_us": stats.checkpoint.manifest_us,
            "validate_us": stats.checkpoint.validate_us,
            "publish_us": stats.checkpoint.publish_us,
            "cleanup_us": stats.checkpoint.cleanup_us,
            "total_us": stats.checkpoint.total_us,
        },
    })))
}

async fn raft_membership(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let raft = state.raft.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let metrics = raft.metrics().borrow().clone();
    let members: Vec<u64> = metrics
        .membership_config
        .membership()
        .voter_ids()
        .chain(metrics.membership_config.membership().learner_ids())
        .collect();
    // Per-shard leadership, so an operator can see which group is electing
    // without reading logs. `None` while a group has no leader, never a
    // default id.
    let shards: Option<Vec<serde_json::Value>> = state.shards.as_ref().map(|shards| {
        shards
            .shards()
            .into_iter()
            .map(|id| {
                serde_json::json!({
                    "shard": id.0,
                    "leader": shards.leader(id),
                })
            })
            .collect()
    });

    Ok(Json(serde_json::json!({
        "members": members,
        "leader": metrics.current_leader,
        "state": format!("{:?}", metrics.state),
        "shards": shards,
    })))
}

/// Membership changes are refused while shards exist.
///
/// Placement is a function of the voter set: a node is one of the inputs every
/// shard's owner set is computed from, so adding or removing one moves shards
/// without moving their data. The two halves would then disagree about where a
/// collection lives, which reads as a never-ending election for that shard
/// rather than as an error. Refusing is the only honest answer until a
/// rebalance exists.
///
/// The voter set is persisted in each node's placement document, and a node
/// whose `--peers` disagrees with that document refuses to start. So a
/// membership change that landed here would not merely be invisible: it would
/// leave every node unable to restart.
fn refuse_membership_change() -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    Err((
        StatusCode::CONFLICT,
        Json(serde_json::json!({
            "error": "membership cannot change while shards are placed: a node is one of the \
                      voters each shard's owner set is computed from, so adding or removing one \
                      moves shards without moving their data. Rebalance support is required \
                      first.",
        })),
    ))
}

/// Turn a Raft write failure into something an operator can act on.
fn membership_write_error(
    error: openraft::error::RaftError<u64, openraft::error::ClientWriteError<u64, u64>>,
    doing: &str,
) -> (StatusCode, Json<serde_json::Value>) {
    use openraft::error::{ClientWriteError, RaftError};

    if let RaftError::APIError(ClientWriteError::ForwardToLeader(forward)) = &error {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": format!(
                    "{doing} has to run on the metadata leader, and this node is not it"
                ),
                "leader": forward.leader_id.or(forward.leader_node),
            })),
        );
    }
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({ "error": format!("{doing} failed: {error}") })),
    )
}

/// How many times an internal admin call has been forwarded.
const ADMIN_HOPS: &str = "x-rekha-admin-hops";
const MAX_ADMIN_HOPS: u8 = 3;

fn admin_hops(headers: &axum::http::HeaderMap) -> u8 {
    headers
        .get(ADMIN_HOPS)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn too_many_admin_hops() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::LOOP_DETECTED,
        Json(serde_json::json!({
            "error": "the membership request was forwarded too many times; no stable metadata \
                      leader was found",
        })),
    )
}

/// The leader a Raft write pointed at, if it was refused for that reason.
fn leader_of(
    error: &openraft::error::RaftError<u64, openraft::error::ClientWriteError<u64, u64>>,
) -> Option<u64> {
    match error {
        openraft::error::RaftError::APIError(
            openraft::error::ClientWriteError::ForwardToLeader(forward),
        ) => forward.leader_id.or(forward.leader_node),
        _ => None,
    }
}

/// Replay an internal admin call on the metadata leader.
///
/// These endpoints live on the internal listener, which has no public
/// `require_leader` middleware. A CLI command can therefore land on any node,
/// and leadership can move between the caller's lookup and the command. The
/// Raft client already reported where the leader is; retrying there turns a
/// 503 that says "ask again" into a command that works. The hop header keeps a
/// leadership race from becoming a loop.
async fn forward_admin(
    state: &AppState,
    leader: u64,
    path: &str,
    body: &serde_json::Value,
    hops: u8,
) -> Option<Result<StatusCode, (StatusCode, Json<serde_json::Value>)>> {
    // Never forward to ourselves: that is a stale `ForwardToLeader` answer, and
    // replaying it here would turn a transient race into a loop.
    if state.raft.as_ref()?.metrics().borrow().id == leader {
        return None;
    }
    let url = state.peers.url(leader)?;
    let mut request = state
        .http
        .post(format!("{url}{path}"))
        .timeout(std::time::Duration::from_secs(15))
        .header(ADMIN_HOPS, (hops + 1).to_string())
        .json(body);
    if let Some(secret) = state.cluster_secret.as_ref() {
        request = request.header(leadership::SECRET_HEADER, secret.as_str());
    }
    let response = request.send().await.ok()?;
    let status = response.status();
    if status.is_success() {
        return Some(Ok(StatusCode::OK));
    }
    let body = response.json::<serde_json::Value>().await.unwrap_or_else(
        |_| serde_json::json!({"error": "the metadata leader refused the request"}),
    );
    Some(Err((status, Json(body))))
}

/// The placement this cluster was built with.
///
/// A node joining has no `--peers`, so it has nobody to ask except the node it
/// was told to join; this is what it asks. The answer is the same three inputs
/// every member persists, so the joiner routes by the same owner sets as
/// everyone else — while hosting nothing until it is promoted.
async fn raft_placement(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let shards = state.shards.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    let placement = shards.placement();
    Ok(Json(serde_json::json!({
        "shard_count": placement.shard_count(),
        "replication_factor": placement.replication_factor(),
        "voters": placement.voters(),
    })))
}

/// Admit a node to the metadata group as a learner.
///
/// A learner does not vote, so the voter set — and with it placement — is
/// unchanged. That is why this is admitted while shards are placed and
/// [`raft_remove_member`] is not: the membership guard exists to stop shards
/// moving, and nothing moves here.
///
/// The address is proposed as a committed entry first. The Raft client reaches
/// the learner through the shared address book, which until now knew only what
/// `--peers` said at each node's last start, so without this the leader has no
/// way to reach a node the flag does not mention — the reason adding a node
/// used to require restarting the whole cluster.
///
/// Admission is deliberately non-blocking. Waiting for a learner to catch up
/// before answering holds an operator's request open for as long as the data
/// takes — and forever if the learner is unreachable. The entry that admits it
/// is committed before this returns; `/ready` and the replication metrics are
/// where catch-up progress belongs.
async fn raft_add_learner(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<serde_json::Value>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    let hops = admin_hops(&headers);
    if hops >= MAX_ADMIN_HOPS {
        return Err(too_many_admin_hops());
    }
    let raft = state.raft.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({"error": "no metadata group"})),
    ))?;
    let node_id = req["node_id"].as_u64().ok_or((
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": "node_id is required"})),
    ))?;
    let addr = req["addr"]
        .as_str()
        .map(str::trim)
        .filter(|addr| !addr.is_empty())
        .ok_or((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "addr is required: it is how the leader reaches the new node, and no \
                          flag teaches it that address",
            })),
        ))?
        .to_owned();

    if state.peers.addr(node_id).as_deref() != Some(addr.as_str()) {
        // Committed, not local: every node has to be able to reach the new node,
        // and the address has to survive a restart and a change of leader.
        if let Err(e) = raft
            .client_write(ClusterOperation::AddNode {
                node_id,
                addr: addr.clone(),
            })
            .await
        {
            if let Some(leader) = leader_of(&e)
                && let Some(response) =
                    forward_admin(&state, leader, "/internal/cluster/add_learner", &req, hops).await
            {
                return response;
            }
            return Err(membership_write_error(
                e,
                "recording where the new node lives",
            ));
        }
    }

    if let Err(e) = raft.add_learner(node_id, node_id, false).await {
        if let Some(leader) = leader_of(&e)
            && let Some(response) =
                forward_admin(&state, leader, "/internal/cluster/add_learner", &req, hops).await
        {
            return response;
        }
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ));
    }
    Ok(StatusCode::OK)
}

async fn raft_remove_member(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<serde_json::Value>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    let hops = admin_hops(&headers);
    if hops >= MAX_ADMIN_HOPS {
        return Err(too_many_admin_hops());
    }
    if state.shards.is_some() {
        refuse_membership_change()?;
    }
    let raft = state.raft.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({"error": "no metadata group"})),
    ))?;
    let node_id = req["node_id"].as_u64().ok_or((
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": "node_id is required"})),
    ))?;
    let metrics = raft.metrics().borrow().clone();
    let new_members: std::collections::BTreeSet<u64> = metrics
        .membership_config
        .membership()
        .voter_ids()
        .chain(metrics.membership_config.membership().learner_ids())
        .filter(|&id| id != node_id)
        .collect();
    if new_members.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "the last member cannot be removed"})),
        ));
    }
    if let Err(e) = raft.change_membership(new_members, true).await {
        if let Some(leader) = leader_of(&e)
            && let Some(response) = forward_admin(
                &state,
                leader,
                "/internal/cluster/remove_member",
                &req,
                hops,
            )
            .await
        {
            return response;
        }
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ));
    }
    Ok(StatusCode::OK)
}

/// Health check — always returns 200 OK.
async fn health_handler() -> Result<StatusCode, StatusCode> {
    Ok(StatusCode::OK)
}

/// Readiness check — returns 200 if engine is accessible, 503 otherwise.
///
/// The body states the active durability level so an operator can confirm
/// remotely what a successful write promises, instead of inferring it from
/// deployment notes.
async fn ready_handler(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    // In cluster mode, readiness means "can this node serve a write", which
    // requires an elected leader this node can reach. Reporting ready while
    // there is no leader would route load at nodes that can only answer 503.
    let cluster = match state.raft.as_ref() {
        Some(raft) => {
            let m = raft.metrics().borrow().clone();
            let leader = m.current_leader;
            let role = format!("{:?}", m.state).to_lowercase();
            let has_leader = leader.is_some();
            Some(serde_json::json!({
                "node_id": m.id,
                "role": role,
                "term": m.current_term,
                "leader": leader,
                "applied_index": m.last_applied.map(|l| l.index),
                "quorum_age_ms": m.millis_since_quorum_ack,
            }))
            .map(|info| (info, has_leader))
        }
        None => None,
    };
    let has_leader = cluster.as_ref().map(|(_, ok)| *ok).unwrap_or(true);
    let cluster_info = cluster.map(|(info, _)| info);

    // Shard leadership is as load-bearing as metadata leadership. A node whose
    // shard group is mid-election answers 503 to every write for that shard, so
    // reporting ready routes load into a window already known to be bad. The
    // metadata group's leader was the only one consulted here, which is why a
    // node could be "ready" and unable to accept the data it was sent.
    let shards_without_leader: Vec<u32> = state
        .shards
        .as_ref()
        .map(|shards| {
            shards
                .shards()
                .into_iter()
                .filter(|shard| shards.leader(*shard).is_none())
                .map(|shard| shard.0)
                .collect()
        })
        .unwrap_or_default();
    let shards_ready = shards_without_leader.is_empty();
    let shards_info = state.shards.as_ref().map(|shards| {
        serde_json::json!({
            "hosted": shards.shards().len(),
            "without_leader": shards_without_leader,
        })
    });
    // A node whose bootstrap credentials have not fully applied can accept the
    // bootstrap key and still deny its requests, because the role assignment
    // lands in a separate catalog write. Treat that window as "not ready" so a
    // client that waits for readiness does not get a 403 for a healthy cluster.
    let bootstrap_ready = state.bootstrap_ready.load(Ordering::SeqCst);
    let is_ready = has_leader && shards_ready && bootstrap_ready;

    // A node that could not apply a committed entry has diverged from the log
    // it claims to have. It must not serve, whatever else it thinks.
    if state.apply_failure.load(Ordering::SeqCst) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "ready": false,
                "error": "a committed log entry could not be applied; this node is diverged \
                          and must be restarted or rebuilt from a snapshot",
                "cluster": cluster_info,
            })),
        );
    }

    match state
        .engine
        .list_collections(&state.tenant, &state.database)
    {
        Ok(collections) => {
            let status = if is_ready {
                StatusCode::OK
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            };
            (
                status,
                Json(serde_json::json!({
                    "ready": is_ready,
                    "durability": state.engine.durability().as_str(),
                    "tenant": state.tenant,
                    "database": state.database,
                    "collections": collections.len(),
                    "bootstrap_ready": bootstrap_ready,
                    "cluster": cluster_info,
                    "shards": shards_info,
                })),
            )
        }
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "ready": false,
                "durability": state.engine.durability().as_str(),
                "error": e.to_string(),
                "cluster": cluster_info,
            })),
        ),
    }
}

/// ChromaDB auth identity stub.
async fn auth_identity() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "user_id": "00000000-0000-0000-0000-000000000000"
    }))
}

// ---------------------------------------------------------------------------
// Tenants and databases
// ---------------------------------------------------------------------------

/// A namespace request that could not be served.
///
/// Separate from `ApiError` because these are not collection-scoped: the
/// failure is about whether a namespace exists, not about the data in it. It is
/// a struct rather than a tuple because a refusal also carries routing headers,
/// and a 136-byte tuple in every `Result` trips `result_large_err`.
struct NamespaceError {
    status: StatusCode,
    message: String,
    /// The node that can accept this write, when this one cannot.
    leader: Option<u64>,
}

impl NamespaceError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            leader: None,
        }
    }
}

impl IntoResponse for NamespaceError {
    fn into_response(self) -> axum::response::Response {
        let mut headers = axum::http::HeaderMap::new();
        if let Some(leader) = self.leader
            && let Ok(value) = axum::http::HeaderValue::from_str(&leader.to_string())
        {
            headers.insert(
                axum::http::HeaderName::from_static(leadership::LEADER_HEADER),
                value,
            );
            headers.insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static("1"),
            );
        }
        let mut body = serde_json::json!({ "error": self.message });
        if let Some(leader) = self.leader {
            body["leader"] = serde_json::json!(leader);
        }
        (self.status, headers, Json(body)).into_response()
    }
}

#[derive(Deserialize)]
struct NamespaceRequest {
    name: String,
}

fn namespace_error(status: StatusCode, message: impl Into<String>) -> NamespaceError {
    NamespaceError::new(status, message)
}

/// The same `503` + `x-rekha-leader` contract the collection path uses.
///
/// A node whose cached leadership is stale appends to a log it does not lead,
/// and `client_write` says so instead of forwarding. Reporting a bare error
/// would hide the one fact the caller needs: another node can accept this write.
fn not_leader(leader: Option<u64>) -> NamespaceError {
    NamespaceError {
        status: StatusCode::SERVICE_UNAVAILABLE,
        message: "this node is not the metadata leader; retry against the leader".into(),
        leader,
    }
}

/// Propose a metadata operation to the log.
///
/// Returns `Ok(false)` on a single-node deployment, where the caller applies the
/// change to its own catalog and the log has nothing to say about it.
async fn replicate_metadata(
    state: &AppState,
    op: ClusterOperation,
) -> Result<bool, NamespaceError> {
    let Some(raft) = state.raft.as_ref() else {
        if state.shards.is_some() {
            // A clustered node with no metadata group has not been admitted yet.
            // Writing its local catalog would create a namespace exactly one
            // node knows about.
            return Err(namespace_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "this node has not joined a cluster yet; retry once it is admitted",
            ));
        }
        return Ok(false);
    };
    use openraft::error::{ClientWriteError, RaftError};
    match raft.client_write(op).await {
        Ok(_) => Ok(true),
        Err(RaftError::APIError(ClientWriteError::ForwardToLeader(forward))) => {
            Err(not_leader(forward.leader_id))
        }
        Err(e) => Err(namespace_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("the metadata log refused this write: {e}"),
        )),
    }
}

fn missing_namespace(what: &str) -> NamespaceError {
    namespace_error(StatusCode::NOT_FOUND, format!("{what} does not exist"))
}

fn catalog_unavailable(e: rekha_storage::CatalogError) -> NamespaceError {
    namespace_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

/// Validate a tenant or database name and trim it.
fn namespace_name(raw: &str) -> Result<String, NamespaceError> {
    let name = raw.trim();
    if name.is_empty() {
        return Err(namespace_error(
            StatusCode::BAD_REQUEST,
            "name must not be empty",
        ));
    }
    validate_name(name).map_err(|e| {
        let status = engine_error_status(&e);
        namespace_error(status, e.to_string())
    })?;
    Ok(name.to_string())
}

/// Every database that exists under `tenant`.
///
/// A namespace exists if it was created, or if a collection already lives in
/// it. Both are catalog facts, so this is a union of two reads of one store
/// rather than a second source of truth — but it is a union because a
/// collection created before its database row existed must not make its own
/// database invisible. Reporting an empty list for a namespace that holds data
/// is the same class of lie as a create that persists nothing.
fn databases_of(
    catalog: &dyn rekha_storage::Catalog,
    tenant: &str,
) -> Result<std::collections::BTreeSet<String>, rekha_storage::CatalogError> {
    let mut names = std::collections::BTreeSet::new();
    for (owner, database) in catalog.list_databases()? {
        if owner == tenant {
            names.insert(database);
        }
    }
    for record in catalog.list_collections_all()? {
        if record.config.tenant == tenant {
            names.insert(record.config.database);
        }
    }
    Ok(names)
}

fn tenant_exists(
    catalog: &dyn rekha_storage::Catalog,
    tenant: &str,
) -> Result<bool, rekha_storage::CatalogError> {
    Ok(catalog.list_tenants()?.iter().any(|known| known == tenant)
        || !databases_of(catalog, tenant)?.is_empty())
}

/// Chroma's client requires a database id; the catalog keys by name.
fn database_id(tenant: &str, name: &str) -> Uuid {
    Uuid::new_v5(&Uuid::NAMESPACE_OID, format!("{tenant}/{name}").as_bytes())
}

/// GET /api/v2/tenants/{name} — does this tenant exist?
async fn get_tenant(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, NamespaceError> {
    let catalog = state.engine.catalog();
    // The configured namespace always exists: the server serves it, whether or
    // not a row has been written yet.
    if name != state.tenant
        && !tenant_exists(catalog.as_ref(), &name).map_err(catalog_unavailable)?
    {
        return Err(missing_namespace(&format!("tenant {name}")));
    }
    Ok(Json(serde_json::json!({ "name": name })))
}

/// POST /api/v2/tenants — create a tenant.
///
/// Idempotent, and the response is sent only once the namespace exists. In
/// cluster mode the row is a log entry, so a tenant created here is visible on
/// every node instead of only on the one that served the request.
async fn create_tenant(
    State(state): State<Arc<AppState>>,
    Json(req): Json<NamespaceRequest>,
) -> Result<Json<serde_json::Value>, NamespaceError> {
    let name = namespace_name(&req.name)?;
    if !replicate_metadata(
        &state,
        ClusterOperation::CreateTenant { name: name.clone() },
    )
    .await?
    {
        match state.engine.catalog().create_tenant(&name) {
            Ok(()) | Err(rekha_storage::CatalogError::AlreadyExists) => {}
            Err(e) => return Err(catalog_unavailable(e)),
        }
    }
    Ok(Json(serde_json::json!({ "name": name })))
}

/// GET /api/v2/tenants/{tenant}/databases — list databases under a tenant.
async fn list_databases(
    State(state): State<Arc<AppState>>,
    Path(tenant): Path<String>,
) -> Result<Json<serde_json::Value>, NamespaceError> {
    let catalog = state.engine.catalog();
    if tenant != state.tenant
        && !tenant_exists(catalog.as_ref(), &tenant).map_err(catalog_unavailable)?
    {
        return Err(missing_namespace(&format!("tenant {tenant}")));
    }
    let mut names = databases_of(catalog.as_ref(), &tenant).map_err(catalog_unavailable)?;
    if tenant == state.tenant {
        names.insert(state.database.clone());
    }
    let rows: Vec<serde_json::Value> = names
        .into_iter()
        .map(|name| {
            serde_json::json!({
                "id": database_id(&tenant, &name),
                "name": name,
                "tenant": tenant,
            })
        })
        .collect();
    Ok(Json(serde_json::json!(rows)))
}

/// GET /api/v2/tenants/{tenant}/databases/{name} — does this database exist?
async fn get_database(
    State(state): State<Arc<AppState>>,
    Path((tenant, name)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, NamespaceError> {
    let catalog = state.engine.catalog();
    let exists = (tenant == state.tenant && name == state.database)
        || databases_of(catalog.as_ref(), &tenant)
            .map_err(catalog_unavailable)?
            .contains(&name);
    if !exists {
        return Err(missing_namespace(&format!("database {tenant}/{name}")));
    }
    Ok(Json(serde_json::json!({
        "id": database_id(&tenant, &name),
        "name": name,
        "tenant": tenant,
    })))
}

/// POST /api/v2/tenants/{tenant}/databases — create a database.
///
/// A database is created inside an existing tenant; a missing tenant is a 404,
/// never an implicit create. Idempotent, and replicated in cluster mode for the
/// same reason as the tenant itself.
async fn create_database(
    State(state): State<Arc<AppState>>,
    Path(tenant): Path<String>,
    Json(req): Json<NamespaceRequest>,
) -> Result<Json<serde_json::Value>, NamespaceError> {
    let name = namespace_name(&req.name)?;
    let catalog = state.engine.catalog();
    if !tenant_exists(catalog.as_ref(), &tenant).map_err(catalog_unavailable)? {
        return Err(missing_namespace(&format!("tenant {tenant}")));
    }
    match replicate_metadata(
        &state,
        ClusterOperation::CreateDatabase {
            tenant: tenant.clone(),
            name: name.clone(),
        },
    )
    .await
    {
        Ok(true) => {}
        Ok(false) => match catalog.create_database(&tenant, &name) {
            Ok(()) | Err(rekha_storage::CatalogError::AlreadyExists) => {}
            Err(e) => return Err(catalog_unavailable(e)),
        },
        Err(e) => return Err(e),
    }
    Ok(Json(serde_json::json!({ "name": name, "tenant": tenant })))
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

fn public_routes() -> Router<Arc<AppState>> {
    let public: Router<Arc<AppState>> = Router::new()
        .route(
            "/api/v2/collections",
            post(create_collection).get(list_collections),
        )
        .route(
            "/api/v2/collections/{id}",
            get(get_collection).delete(delete_collection),
        )
        .route("/api/v2/collections/{id}/add", post(add_records))
        .route("/api/v2/collections/{id}/upsert", post(upsert_records))
        .route("/api/v2/collections/{id}/update", post(update_records))
        .route("/api/v2/collections/{id}/delete", post(delete_records))
        .route("/api/v2/collections/{id}/query", post(query_records))
        .route("/api/v2/collections/{id}/get", post(get_records))
        .route("/api/v2/collections/{id}/count", post(count_records))
        .route(
            "/api/v2/collections/{id}/optimize",
            post(optimize_collection),
        )
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/api/v2/auth/identity", get(auth_identity))
        .route("/api/v2/tenants", post(create_tenant))
        .route("/api/v2/tenants/{name}", get(get_tenant))
        .route(
            "/api/v2/tenants/{tenant}/databases",
            get(list_databases).post(create_database),
        )
        .route(
            "/api/v2/tenants/{tenant}/databases/{name}",
            get(get_database),
        );
    public
        .merge(chroma::chroma_router())
        .merge(auth_routes::auth_router())
}

fn cluster_admin_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/internal/cluster/membership", get(raft_membership))
        .route("/internal/cluster/placement", get(raft_placement))
        .route("/internal/cluster/add_learner", post(raft_add_learner))
        .route("/internal/cluster/remove_member", post(raft_remove_member))
}

fn internal_routes() -> Router<Arc<AppState>> {
    Router::new()
        // One process runs one `Raft` per group, so the group id travels in the
        // path. A vote addressed to the wrong group would corrupt two logs.
        .route(
            "/internal/raft/{group}/append_entries",
            post(raft_append_entries),
        )
        .route("/internal/raft/{group}/vote", post(raft_vote))
        .route(
            "/internal/raft/{group}/install_snapshot",
            post(raft_install_snapshot),
        )
        // Chunked upload, used by this build's transport. The single-request
        // endpoint above stays for older peers and for the in-process tests.
        .route(
            "/internal/raft/{group}/snapshot/{snapshot_id}",
            post(raft_snapshot_chunk),
        )
        // Membership is a property of a whole group, not of a single RPC, and
        // these operate on the metadata group.
        .route(
            "/internal/cluster/shard/{shard}/ensure_started",
            post(shard_ensure_started),
        )
        .route(
            "/internal/cluster/shard/{shard}/membership",
            get(shard_membership_handler),
        )
        .route("/internal/cluster/shard/{shard}/drop", post(shard_drop))
        .route("/internal/shard/{shard}/batch", post(shard_batch_handler))
        .route(
            "/internal/shard/{shard}/prepare",
            post(shard_prepare_handler),
        )
        .route("/internal/shard/{shard}/commit", post(shard_commit_handler))
        .route("/internal/shard/{shard}/abort", post(shard_abort_handler))
        .route("/internal/shard/{shard}/query", post(shard_query_handler))
        .route("/internal/shard/{shard}/count", post(shard_count_handler))
        .route("/internal/shard/{shard}/get", post(shard_get_handler))
        .route("/internal/stats/counts", post(stats_counts))
        .route(
            "/internal/cluster/tx/{tx_id}/status",
            get(tx_status_handler),
        )
        .route("/internal/cluster/tx/commit", post(tx_commit_handler))
        .route("/internal/cluster/tx/abort", post(tx_abort_handler))
        .merge(cluster_admin_routes())
        // Writes arriving at a follower are forwarded here, on the leader's
        // internal listener, and re-dispatched through the public pipeline so
        // the original caller's credentials are still what is authorized.
        .route(
            "/internal/forward/{*rest}",
            axum::routing::any(leadership::dispatch_forwarded),
        )
}

/// Build the full router (public + internal + Chroma) on a single listener.
/// For production, prefer separate listeners via [`public_router`] and
/// [`internal_router`] so internal endpoints are not exposed publicly.
/// Combined router for single-listener deployments.
///
/// Public and internal routes carry **separate** authentication layers:
/// public routes use RBAC ([`middleware::auth`]), internal routes use the
/// cluster secret ([`middleware::internal_auth`]). Applying the public layer
/// to everything meant an internal RPC had to hold a valid *API key* as well
/// as the cluster secret — so `REKHA_CLUSTER_SECRET` alone was not enough and a
/// clustered node could not talk to its peers once RBAC users existed.
pub fn router(app_state: Arc<AppState>) -> Router {
    let public = public_routes()
        .layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            leadership::require_leader,
        ))
        .layer(axum::middleware::from_fn(middleware::auth));
    let internal = internal_routes().layer(axum::middleware::from_fn(middleware::internal_auth));
    public
        .merge(internal)
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            32 * 1024 * 1024,
        ))
        .layer(axum::Extension(app_state.clone()))
        .with_state(app_state)
}

/// Public API router (collections, health, tenants, Chroma compat, and
/// cluster-secret-authenticated membership management).
/// Intended for the public listener.
pub fn public_router(app_state: Arc<AppState>) -> Router {
    let public = public_routes()
        // Inner layer: runs after authentication, so an unauthenticated write
        // is rejected as 401 rather than being told where the leader is.
        .layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            leadership::require_leader,
        ))
        .layer(axum::middleware::from_fn(middleware::auth));
    let cluster_admin =
        cluster_admin_routes().layer(axum::middleware::from_fn(middleware::internal_auth));
    public
        .merge(cluster_admin)
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            32 * 1024 * 1024,
        ))
        .layer(axum::Extension(app_state.clone()))
        .with_state(app_state)
}

/// Internal cluster router (Raft RPCs and forwarded writes).
/// Intended for a separate, network-isolated listener.
pub fn internal_router(app_state: Arc<AppState>) -> Router {
    internal_routes()
        .route("/health", get(health_handler))
        .layer(axum::middleware::from_fn(middleware::internal_auth))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            32 * 1024 * 1024,
        ))
        .layer(axum::Extension(app_state.clone()))
        .with_state(app_state)
}

#[cfg(test)]
mod offload_tests {
    use std::time::{Duration, Instant};

    use super::offload;
    use rekha_engine::EngineError;

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn engine_work_runs_on_a_blocking_thread() {
        let caller = std::thread::current().id();
        let worker = offload(move || Ok(std::thread::current().id()))
            .await
            .unwrap();
        assert_ne!(
            caller, worker,
            "engine work must not run on the async worker thread"
        );
    }

    /// The property that matters: a slow engine call must not stop the runtime
    /// from running other tasks. Raft heartbeats are such a task, and a delayed
    /// heartbeat is an election.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn slow_engine_work_does_not_starve_the_runtime() {
        let mut handles = Vec::new();
        for _ in 0..8 {
            handles.push(tokio::spawn(offload(|| {
                std::thread::sleep(Duration::from_millis(250));
                Ok(())
            })));
        }

        // A heartbeat-shaped timer must still fire on time while those run.
        let started = Instant::now();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(200),
            "a timer was delayed to {elapsed:?}: blocking work is running on the async worker"
        );

        for handle in handles {
            handle.await.unwrap().unwrap();
        }
    }

    #[tokio::test]
    async fn an_engine_error_propagates_unchanged() {
        let err = offload(|| Err::<(), _>(EngineError::CollectionNotFound("x".into())))
            .await
            .unwrap_err();
        assert!(matches!(err, EngineError::CollectionNotFound(_)));
    }
}
