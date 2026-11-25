use crate::prelude::*;
use crate::*;

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub(crate) fn public_routes() -> Router<Arc<AppState>> {
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

pub(crate) fn cluster_admin_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/internal/cluster/membership", get(raft_membership))
        .route("/internal/cluster/placement", get(raft_placement))
        .route("/internal/cluster/add_learner", post(raft_add_learner))
        .route("/internal/cluster/remove_member", post(raft_remove_member))
}

pub(crate) fn internal_routes() -> Router<Arc<AppState>> {
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
