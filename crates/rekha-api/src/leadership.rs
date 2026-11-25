//! Route every write to whoever consensus elected.
//!
//! A node that is not the leader for a write must not apply it. Before this
//! guard existed, a write to any node other than node 1 was acknowledged
//! locally and never replicated — an acknowledged write that existed on exactly
//! one node, until that node died.
//!
//! The decision is made by `Raft::is_leader()`, not by reading a cached leader
//! from application state. Leadership can move between a cached read and the
//! write, so the cached answer is a race; consensus answers about the term it
//! is actually in.
//!
//! Outcomes:
//!
//! - this node leads        → run the handler
//! - another node leads     → forward the request once, transparently
//! - no leader yet          → `503` with `Retry-After`; never a local apply
//! - already forwarded once → `503` with `x-rekha-leader`; never loop
//!
//! Reads follow the same rule as writes: they run on the group's leader, after
//! a barrier that confirms leadership and waits for the leader's applied state
//! to cover the read index. A replica can be behind by an unbounded amount, so
//! serving a read from one makes "write, then read on another node" return the
//! old value with no way for the client to know. `?consistency=local` opts out
//! for workloads that prefer a fast, possibly stale answer.
//!
//! Forwarding is preferred over rejecting because Chroma clients do not retry
//! against a different host on their own, so a `503` at the "wrong" node would
//! surface as a client error for a cluster that is entirely healthy.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{Method, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use openraft::error::{CheckIsLeaderError, RaftError};

use crate::AppState;

/// How many times this request has been forwarded.
///
/// A boolean "already forwarded" is too coarse once there are two reasons to
/// forward — reaching a replica, then reaching that shard's leader — because
/// the second hop is legitimate and the boolean refuses it. The budget keeps
/// termination without forbidding the second hop.
pub const HOPS_HEADER: &str = "x-rekha-hops";

/// One hop to reach a replica, one to reach that shard's leader, and one spare
/// for leadership moving between the check and the message.
const MAX_FORWARD_HOPS: u8 = 3;
/// Header carrying the shared cluster secret, as the internal listener expects.
pub const SECRET_HEADER: &str = "x-rekha-cluster-secret";
/// Tells a client which node leads, so a retry can be directed.
pub const LEADER_HEADER: &str = "x-rekha-leader";

/// Methods that can mutate. Everything else is never routed to the leader.
fn is_mutating(method: &Method) -> bool {
    matches!(
        *method,
        Method::POST | Method::PUT | Method::PATCH | Method::DELETE
    )
}

/// Whether this request must execute on the leader.
///
/// `POST` is used for reads as well (`/query`, `/get`, `/count`), so those are
/// excluded by suffix. Reads are allowed on any node: they may be stale, but
/// they cannot corrupt state.
pub fn requires_leader(method: &Method, path: &str) -> bool {
    if !is_mutating(method) {
        return false;
    }
    // Cluster-internal RPCs have their own authentication and their own rules.
    if path.starts_with("/internal/") {
        return false;
    }
    if path == "/health" || path == "/ready" {
        return false;
    }
    // Read endpoints that happen to be POST.
    //
    // They are still routed by [`require_leader`]; this function answers only
    // "is this a mutation", because the remaining decision — where a read runs —
    // is made by the same placement and leadership routing that handles writes.
    if path.ends_with("/query") || path.ends_with("/get") || path.ends_with("/count") {
        return false;
    }
    true
}

/// Record operations that touch one collection's data.
///
/// These are the paths a shard group owns. Everything else — collection
/// create/get/delete, tenants, RBAC — is a metadata operation, and the metadata
/// group owns it. The distinction is by path suffix because the same surface
/// exposes both, and a collection delete and a record delete differ only by
/// that suffix.
const DATA_OPS: [&str; 8] = [
    "add", "upsert", "update", "delete", "query", "get", "count", "optimize",
];

/// Resolve the collection a data operation targets, if this is one.
///
/// Two addressing schemes reach the same handler: native routes use the
/// collection UUID, Chroma-compatible routes use `(tenant, database, name)`.
/// The name form costs one catalog lookup, which is a local B-tree read on
/// every node — the catalog carries every collection precisely so a node that
/// holds none of the data can still route to it.
fn data_target(state: &AppState, path: &str) -> Option<uuid::Uuid> {
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    let at = segments.iter().position(|s| *s == "collections")?;
    let collection = *segments.get(at + 1)?;
    let op = *segments.get(at + 2)?;
    if !DATA_OPS.contains(&op) {
        return None;
    }

    if let Ok(id) = uuid::Uuid::parse_str(collection) {
        return Some(id);
    }

    // Chroma form: /api/v2/tenants/{tenant}/databases/{database}/collections/{name}/...
    let tenant = segments
        .iter()
        .position(|s| *s == "tenants")
        .and_then(|i| segments.get(i + 1))?;
    let database = segments
        .iter()
        .position(|s| *s == "databases")
        .and_then(|i| segments.get(i + 1))?;
    state
        .engine
        .get_collection(tenant, database, collection)
        .ok()
        .flatten()
        .map(|record| record.config.id)
}

/// Request header naming the node that actually answered, for data reads.
///
/// A read on any node is routed to the shard's leader, so a client that wants to
/// verify routing (or a test that wants evidence rather than a timing
/// assumption) can read this instead of inferring it.
pub const SERVED_BY_HEADER: &str = "x-rekha-served-by";

/// Whether this request asked to be served from the local replica.
///
/// `?consistency=local` is an explicit trade: the answer is whatever this node
/// has applied, which may be behind by an unbounded amount. Everything else is
/// linearizable.
fn reads_locally(request: &Request) -> bool {
    request
        .uri()
        .query()
        .map(|query| {
            query
                .split('&')
                .filter_map(|pair| pair.split_once('='))
                .any(|(key, value)| key == "consistency" && value == "local")
        })
        .unwrap_or(false)
}

/// How long a read may wait for the leadership check before it is refused.
///
/// A read that cannot confirm leadership must fail, not hang: the client can
/// retry against another node, but it cannot time out of a request that never
/// returns.
const READ_BARRIER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Confirm this node still leads the collection's group, and that its applied
/// state has reached the read index.
///
/// `ensure_linearizable` sends heartbeats to a quorum and waits for the state
/// machine to catch up, so a read that follows it observes every write that
/// committed before the request arrived. Without it a replica answers from
/// whatever it has applied, which is why a write followed by an immediate read
/// on another node could return the *old* value — the failure this replaces.
async fn read_barrier(state: &AppState, collection: &uuid::Uuid) -> Result<(), Response> {
    let raft = match state.shards.as_ref() {
        Some(shards) => state
            .shard_of(collection)
            .and_then(|shard| shards.raft(shard)),
        None => state.raft.clone(),
    };
    let Some(raft) = raft else {
        // No consensus at all: single-node, and the local state is the truth.
        return Ok(());
    };

    // One voter is its own quorum, and there is no second copy that could be
    // ahead of it, so the barrier would only add a round trip through the Raft
    // core to every read on a single-node deployment.
    let single_voter = {
        let metrics = raft.metrics();
        let m = metrics.borrow();
        m.membership_config.membership().voter_ids().count() <= 1
    };
    if single_voter {
        return Ok(());
    }

    match tokio::time::timeout(READ_BARRIER_TIMEOUT, raft.ensure_linearizable()).await {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => {
            // Leadership moved, or a quorum is unreachable. A retry will be
            // routed to the new leader; answering from this node would be the
            // stale read the barrier exists to prevent.
            tracing::warn!(%err, "rejecting a read: this node can no longer prove it leads");
            Err(no_leader())
        }
        Err(_) => {
            // openraft's check waits for a quorum ack, and a partition makes
            // that wait unbounded from the caller's point of view. A read has to
            // fail so the client can retry; a request that never returns cannot
            // be retried.
            tracing::warn!(
                timeout_ms = READ_BARRIER_TIMEOUT.as_millis(),
                "rejecting a read: no quorum confirmation within the deadline"
            );
            Err(no_leader())
        }
    }
}

fn tag_served_by(state: &AppState, mut response: Response) -> Response {
    let node_id = state.raft.as_ref().map(|raft| raft.metrics().borrow().id);
    if let Some(id) = node_id
        && let Ok(value) = axum::http::HeaderValue::from_str(&id.to_string())
    {
        response.headers_mut().insert(SERVED_BY_HEADER, value);
    }
    response
}

/// Middleware: send each request to the node that holds the data and, for a
/// write, to the leader of the group that owns it.
///
/// Two decisions, in this order:
///
/// 1. **Placement.** If this node holds no replica of the collection, forward.
///    This applies to reads as well as writes: a node with no replica cannot
///    answer, and answering "empty" would be a wrong answer rather than a
///    routing miss.
/// 2. **Leadership.** On a replica, a write must run on that group's leader.
///    Reads are served locally and may be stale, as before.
pub async fn require_leader(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: axum::middleware::Next,
) -> Response {
    let path = request.uri().path().to_owned();
    let method = request.method().clone();

    // No consensus at all: nothing to route to.
    if state.raft.is_none() {
        return next.run(request).await;
    }
    if path.starts_with("/internal/") || path == "/health" || path == "/ready" {
        return next.run(request).await;
    }

    let hops = hops_so_far(&request);

    if let Some(collection) = data_target(&state, &path) {
        if !state.hosts_collection(&collection) {
            if hops >= MAX_FORWARD_HOPS {
                return too_many_hops();
            }
            return forward_to_replica(&state, request, collection, hops + 1).await;
        }
        // A forwarded request names a collection the sender has in its catalog.
        // This node may not have applied that metadata entry yet, because the
        // metadata group and the shard group are different logs with no ordering
        // between them. Waiting here is a barrier between the two, and it only
        // applies to forwarded requests, so a genuinely unknown collection still
        // 404s immediately instead of stalling.
        if hops > 0 && !collection_is_visible(&state, &collection).await {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                [(header::RETRY_AFTER, "1")],
                Json(serde_json::json!({
                    "error": "the collection exists but this replica has not applied its \
                              creation yet; retry shortly",
                })),
            )
                .into_response();
        }

        let mutating = requires_leader(&method, &path);

        // An explicit opt-out, for workloads that would rather have a fast,
        // possibly stale answer (analytics over a large collection). It is not
        // the default because a replica has no bound on how stale it may be:
        // the client that wrote a record and immediately reads it can see it
        // missing, and there is no lag figure it could check to know.
        if !mutating && reads_locally(&request) {
            return tag_served_by(&state, next.run(request).await);
        }

        // Reads and writes both run on the group's leader. For a read the
        // leader then confirms its leadership with a quorum and waits for its
        // own applied state to reach the read index, which is what makes
        // read-after-write on any node return the write.
        return match shard_route(&state, collection).await {
            Route::Here => {
                if !mutating && let Err(response) = read_barrier(&state, &collection).await {
                    return response;
                }
                tag_served_by(&state, next.run(request).await)
            }
            Route::Forward(leader) => {
                if hops >= MAX_FORWARD_HOPS {
                    return too_many_hops();
                }
                forward_to(&state, request, leader, hops + 1).await
            }
            Route::Nowhere => no_leader(),
        };
    }

    if !requires_leader(&method, &path) {
        return next.run(request).await;
    }

    // Metadata group: routing only. The log decides whether a write is legal,
    // so this read decides *where*, never *whether*.
    let route = metadata_route(&state).await;

    match route {
        Route::Here => next.run(request).await,
        Route::Forward(leader) => {
            if hops >= MAX_FORWARD_HOPS {
                return too_many_hops();
            }
            forward_to(&state, request, leader, hops + 1).await
        }
        Route::Nowhere => no_leader(),
    }
}

/// Leadership of the shard group that holds `collection`, read from that
/// group's own metrics.
///
/// A group's term moves independently of the metadata group's, so this cannot
/// come from the shared `LeaderTracker`. A stale answer here costs one rejected
/// write — `client_write` returns `ForwardToLeader` — not a lost one.
async fn shard_route(state: &AppState, collection: uuid::Uuid) -> Route {
    let Some(shards) = state.shards.as_ref() else {
        // One group holds everything: the metadata leader is the leader.
        return metadata_route(state).await;
    };
    // A collection's shard comes from its catalog row, not from its id. The
    // metadata leader chose it for load, so no hash can reproduce it.
    let Some(shard) = state.shard_of(&collection) else {
        if let Ok(Some(rec)) = state.engine.get_collection_by_id(&collection)
            && rec.shard.is_none()
        {
            // An intra-collection partitioned collection: any node can coordinate
            // fan-out writes and scatter-gather queries across all shards.
            return Route::Here;
        }
        return Route::Nowhere;
    };
    let Some(raft) = shards.raft(shard) else {
        return Route::Nowhere;
    };
    let metrics = raft.metrics();
    let m = metrics.borrow();
    match m.current_leader {
        Some(leader) if leader == m.id => Route::Here,
        Some(leader) => Route::Forward(leader),
        None => Route::Nowhere,
    }
}

/// Wait for a forwarded request's collection to appear in this node's catalog.
///
/// The metadata group commits the collection's creation, and the request is
/// then forwarded to a shard replica. Applied state is per group, so the
/// replica can be a few entries behind. This is bounded: a client that waited
/// longer than this is better served by a retry than by a stall.
async fn collection_is_visible(state: &AppState, collection: &uuid::Uuid) -> bool {
    for _ in 0..100 {
        if let Ok(Some(_)) = state.engine.get_collection_by_id(collection) {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    false
}

/// Route a request to the metadata group's leader.
///
/// Prefers the `LeaderTracker` because it is a cheap cached read, and falls
/// back to asking consensus directly when no tracker is attached. A stale
/// cached answer costs one rejected write — `client_write` returns
/// `ForwardToLeader` — never a lost one.
async fn metadata_route(state: &AppState) -> Route {
    if let Some(tracker) = state.leader.as_ref() {
        let view = tracker.view();
        if view.is_leader()
            || view
                .leader
                .as_ref()
                .is_some_and(|l| l.node_id == view.node_id)
        {
            return Route::Here;
        }
        return match view.leader {
            Some(leader) => Route::Forward(leader.node_id),
            None => Route::Nowhere,
        };
    }

    let Some(raft) = state.raft.clone() else {
        // No consensus at all: single-node, serve locally.
        return Route::Here;
    };
    match raft.ensure_linearizable().await {
        Ok(_) => Route::Here,
        Err(RaftError::APIError(CheckIsLeaderError::ForwardToLeader(forward))) => {
            match forward.leader_id.or(forward.leader_node) {
                Some(id) => Route::Forward(id),
                None => Route::Nowhere,
            }
        }
        Err(err) => {
            tracing::warn!(%err, "rejecting write: no leader available");
            Route::Nowhere
        }
    }
}

/// Forward a request to one of the collection's replicas when this node holds
/// none.
///
/// The replicas are tried **in order**, not just the first. Placement order is a
/// deterministic routing choice, not a health signal: when the first owner is
/// down, a node that tried only it would fail every request for that collection
/// while two healthy replicas sat idle. Reproduced before this change: killing
/// one replica made the collection unwritable from every non-owner, even though
/// the shard had elected a new leader and never lost quorum.
///
/// A `503` from a replica is not terminal either — it may mean "this group is
/// electing" (another replica will know the same thing) or "I am not the
/// leader" (another replica is). Either way the next replica is worth a try,
/// and the last answer is returned if every one refuses.
async fn forward_to_replica(
    state: &Arc<AppState>,
    request: Request,
    collection: uuid::Uuid,
    next_hop: u8,
) -> Response {
    let owners_owned: Vec<u64> = match state.shards.as_ref() {
        // The owners of the shard the collection was placed on. Unknown shard
        // (the create has not reached this node yet) is not a guess: it is
        // "retry", which is what the caller's 503 means.
        Some(shards) => state
            .shard_of(&collection)
            .map_or(Vec::new(), |shard| shards.owners(shard)),
        // No placement information: one group, and `hosts_collection` already
        // said this node is not in it. That cannot be routed.
        None => Vec::new(),
    };
    let owners = owners_owned.as_slice();
    let Some(first) = owners.first().copied() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "no replica is known for this collection; every node must agree on \
                          the cluster membership",
            })),
        )
            .into_response();
    };

    let forwarded = match buffer(request).await {
        Ok(forwarded) => forwarded,
        Err(response) => return response,
    };

    let mut attempts: Vec<String> = Vec::with_capacity(owners.len());
    let mut last_refusal: Option<Response> = None;

    for &owner in owners {
        let Some(url) = state.peers.url(owner) else {
            attempts.push(format!("{owner}: no address configured"));
            continue;
        };
        match send_forward(state, &forwarded, &url, next_hop, None).await {
            Ok(response)
                if response.status() != StatusCode::SERVICE_UNAVAILABLE
                    && response.status() != StatusCode::BAD_GATEWAY =>
            {
                return response;
            }
            Ok(response) => {
                attempts.push(format!("{owner}: {}", response.status()));
                last_refusal = Some(response);
            }
            Err(e) => attempts.push(format!("{owner}: {e}")),
        }
    }

    // Every replica refused or was unreachable. Prefer a replica's own answer
    // when there is one: it knows whether it is electing, which tells the client
    // more than "nobody answered".
    if let Some(response) = last_refusal {
        return response;
    }
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, "1")],
        Json(serde_json::json!({
            "error": "no replica of this collection is reachable from this node",
            "replicas": owners,
            "attempts": attempts,
            "first_replica": first,
        })),
    )
        .into_response()
}

/// Forward to one specific node — the leader of the group this request belongs
/// to, or the metadata leader for a registry operation.
async fn forward_to(
    state: &Arc<AppState>,
    request: Request,
    target: u64,
    next_hop: u8,
) -> Response {
    let forwarded = match buffer(request).await {
        Ok(forwarded) => forwarded,
        Err(response) => return response,
    };
    let Some(url) = state.peers.url(target) else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "a leader is elected but this node has no address for it; every node \
                          must be listed in --peers",
                "leader": target,
            })),
        )
            .into_response();
    };
    match send_forward(state, &forwarded, &url, next_hop, Some(target)).await {
        Ok(response) => response,
        Err(e) => {
            tracing::warn!(%e, leader = target, "forwarding to the leader failed");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                [(header::RETRY_AFTER, "1")],
                Json(serde_json::json!({
                    "error": format!("could not forward to the leader: {e}"),
                    "leader": target,
                })),
            )
                .into_response()
        }
    }
}

/// A request body buffered once so it can be replayed against several replicas.
struct BufferedRequest {
    parts: axum::http::request::Parts,
    body: axum::body::Bytes,
}

async fn buffer(request: Request) -> Result<BufferedRequest, Response> {
    let (parts, body) = request.into_parts();
    match axum::body::to_bytes(body, 32 * 1024 * 1024).await {
        Ok(body) => Ok(BufferedRequest { parts, body }),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("unreadable body: {e}") })),
        )
            .into_response()),
    }
}

/// Send one forward and materialise the peer's answer.
///
/// `leader_hint` is set only when the target is known to be the leader; a
/// placement hop targets a replica, and claiming it is the leader would tell a
/// client to retry where it just failed.
async fn send_forward(
    state: &Arc<AppState>,
    forwarded: &BufferedRequest,
    base_url: &str,
    next_hop: u8,
    leader_hint: Option<u64>,
) -> Result<Response, reqwest::Error> {
    let parts = &forwarded.parts;
    let mut target = format!(
        "{}/internal/forward{}",
        base_url.trim_end_matches('/'),
        parts.uri.path()
    );
    if let Some(query) = parts.uri.query() {
        target.push('?');
        target.push_str(query);
    }

    let mut outbound = state
        .http
        .request(parts.method.clone(), &target)
        .body(forwarded.body.to_vec());

    // Copy the caller's headers so the peer can re-authenticate the original
    // request, but drop hop-by-hop and recomputed ones.
    for (name, value) in parts.headers.iter() {
        if name == header::HOST
            || name == header::CONTENT_LENGTH
            || name == header::CONNECTION
            || name == header::TRANSFER_ENCODING
        {
            continue;
        }
        outbound = outbound.header(name, value);
    }
    outbound = outbound.header(HOPS_HEADER, next_hop.to_string());
    if let Some(leader) = leader_hint {
        outbound = outbound.header(LEADER_HEADER, leader.to_string());
    }
    if let Some(secret) = state.cluster_secret.as_deref() {
        outbound = outbound.header(SECRET_HEADER, secret);
    }

    let response = outbound.send().await?;
    let status = response.status();
    let headers = response.headers().clone();
    let body = response.bytes().await.unwrap_or_default();

    let mut builder = Response::builder().status(status);
    for (name, value) in headers.iter() {
        if name == header::CONTENT_LENGTH || name == header::CONNECTION {
            continue;
        }
        builder = builder.header(name, value);
    }
    Ok(builder
        .body(Body::from(body))
        .unwrap_or_else(|e| (StatusCode::BAD_GATEWAY, e.to_string()).into_response()))
}

/// How many times this request has been forwarded already.
fn hops_so_far(request: &Request) -> u8 {
    request
        .headers()
        .get(HOPS_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u8>().ok())
        .unwrap_or(0)
}

/// The hop budget is exhausted, which means the cluster does not agree about
/// where this collection lives.
fn too_many_hops() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, "1")],
        Json(serde_json::json!({
            "error": "this request was forwarded more than the cluster allows; the nodes \
                      disagree about which node owns this collection (check that every node \
                      lists the same --peers and the same shard count)",
        })),
    )
        .into_response()
}

/// Where a write should run.
enum Route {
    /// This node leads.
    Here,
    /// Another node leads (or holds the data); forward to it.
    Forward(u64),
    /// No leader is known.
    Nowhere,
}

fn no_leader() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, "1")],
        Json(serde_json::json!({
            "error": "no leader is currently available for this data; retry shortly",
        })),
    )
        .into_response()
}

/// Hand a forwarded request back to the local public pipeline.
///
/// This runs on the leader's internal listener, which is already authenticated
/// by the cluster secret. The request is re-dispatched through the public
/// router so RBAC is evaluated against the *original* caller's credentials,
/// which were forwarded verbatim — a follower cannot launder permissions by
/// forwarding.
pub async fn dispatch_forwarded(
    state: State<Arc<AppState>>,
    axum::extract::Path(rest): axum::extract::Path<String>,
    request: Request,
) -> Response {
    use tower::ServiceExt;

    let query = request
        .uri()
        .query()
        .map(|q| format!("?{q}"))
        .unwrap_or_default();
    let uri = format!("/{rest}{query}");

    let (parts, body) = request.into_parts();
    let mut builder = Request::builder().method(parts.method).uri(&uri);
    for (name, value) in parts.headers.iter() {
        if name == header::HOST || name == header::CONTENT_LENGTH {
            continue;
        }
        builder = builder.header(name, value);
    }

    let rebuilt = match builder.body(body) {
        Ok(request) => request,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("malformed forward: {e}") })),
            )
                .into_response();
        }
    };

    // Rebuilding the public router per forwarded write is a known cost: it is
    // only paid when a write arrives at a follower, and caching a Router inside
    // AppState would create a reference cycle. Revisit if forward volume grows.
    let router: Router = crate::public_router(state.0);
    router.oneshot(rebuilt).await.unwrap_or_else(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("forward dispatch failed: {e}") })),
        )
            .into_response()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_are_never_routed_to_the_leader() {
        assert!(!requires_leader(
            &Method::POST,
            "/api/v2/collections/x/query"
        ));
        assert!(!requires_leader(&Method::POST, "/api/v2/collections/x/get"));
        assert!(!requires_leader(
            &Method::POST,
            "/api/v2/collections/x/count"
        ));
        assert!(!requires_leader(
            &Method::POST,
            "/api/v2/tenants/t/databases/d/collections/c/query"
        ));
        assert!(!requires_leader(&Method::GET, "/api/v2/collections"));
        assert!(!requires_leader(&Method::GET, "/health"));
    }

    #[test]
    fn writes_are_routed_to_the_leader() {
        assert!(requires_leader(&Method::POST, "/api/v2/collections/x/add"));
        assert!(requires_leader(
            &Method::POST,
            "/api/v2/collections/x/upsert"
        ));
        assert!(requires_leader(
            &Method::POST,
            "/api/v2/collections/x/update"
        ));
        assert!(requires_leader(
            &Method::POST,
            "/api/v2/collections/x/delete"
        ));
        assert!(requires_leader(&Method::DELETE, "/api/v2/collections/x"));
        assert!(requires_leader(&Method::POST, "/api/v2/collections"));
        assert!(requires_leader(&Method::PUT, "/anything"));
    }

    /// The internal surface has its own authentication and must not be gated on
    /// leadership, or Raft RPCs would be rejected by the node that needs them.
    #[test]
    fn internal_routes_are_exempt() {
        assert!(!requires_leader(
            &Method::POST,
            "/internal/raft/append_entries"
        ));
        assert!(!requires_leader(&Method::POST, "/internal/forward/x"));
    }

    /// A path that merely contains "expect" must not be mistaken for `/get`.
    #[test]
    fn suffix_exemptions_do_not_match_substrings() {
        assert!(requires_leader(
            &Method::POST,
            "/api/v2/collections/budget/getting"
        ));
    }
}

#[cfg(test)]
mod edge_tests {
    use super::*;

    #[test]
    fn an_exhausted_hop_budget_is_retryable_and_explains_itself() {
        let response = too_many_hops();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get(header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok()),
            Some("1")
        );
    }

    #[test]
    fn the_hop_counter_reads_only_a_valid_numeric_header() {
        let mut request = Request::new(Body::empty());
        assert_eq!(hops_so_far(&request), 0, "absent means first hop");
        request
            .headers_mut()
            .insert(HOPS_HEADER, axum::http::HeaderValue::from_static("2"));
        assert_eq!(hops_so_far(&request), 2);
        request.headers_mut().insert(
            HOPS_HEADER,
            axum::http::HeaderValue::from_static("not-a-number"),
        );
        assert_eq!(
            hops_so_far(&request),
            0,
            "a corrupt counter resets the count rather than trusting it"
        );
        request
            .headers_mut()
            .insert(HOPS_HEADER, axum::http::HeaderValue::from_static("300"));
        assert_eq!(hops_so_far(&request), 0, "u8 overflow is not a hop count");
    }
}
