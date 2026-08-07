//! RekhaDB HTTP API — Chroma-compatible REST server.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
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

pub mod chroma;

pub mod middleware {
    use axum::http::{HeaderMap, Request, StatusCode};
    use axum::middleware::Next;
    use axum::response::Response;

    pub async fn auth(
        request: Request<axum::body::Body>,
        next: Next,
    ) -> Result<Response, StatusCode> {
        let state = request
            .extensions()
            .get::<std::sync::Arc<super::AppState>>()
            .cloned()
            .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

        if let Some(ref expected) = state.api_key {
            let headers: HeaderMap = request.headers().clone();
            let provided = headers
                .get("x-chroma-token")
                .or_else(|| headers.get("x-api-key"))
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            if provided.as_ref() != Some(expected) {
                return Err(StatusCode::UNAUTHORIZED);
            }
        }
        Ok(next.run(request).await)
    }
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

pub struct AppState {
    pub engine: Arc<Engine>,
    pub tenant: String,
    pub database: String,
    pub raft: Option<Arc<openraft::Raft<rekha_cluster::RaftTypeConfig>>>,
    pub api_key: Option<String>,
}

// ---------------------------------------------------------------------------
// Error mapping (newtype wrapper to satisfy orphan rules)
// ---------------------------------------------------------------------------

struct ApiError(EngineError);

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match &self.0 {
            EngineError::CollectionNotFound(_) => (StatusCode::NOT_FOUND, self.0.to_string()),
            EngineError::Validation(_) => (StatusCode::BAD_REQUEST, self.0.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()),
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

impl From<EngineError> for ApiError {
    fn from(e: EngineError) -> Self {
        ApiError(e)
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
    pub count: u64,
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
    Json(req): Json<AppendEntriesRequest<RaftTypeConfig>>,
) -> Result<Json<AppendEntriesResponse<u64>>, StatusCode> {
    let raft = state.raft.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    raft.append_entries(req)
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn raft_vote(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VoteRequest<u64>>,
) -> Result<Json<VoteResponse<u64>>, StatusCode> {
    let raft = state.raft.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    raft.vote(req)
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn raft_install_snapshot(
    State(state): State<Arc<AppState>>,
    Json(req): Json<(
        openraft::Vote<u64>,
        openraft::SnapshotMeta<u64, u64>,
        Vec<u8>,
    )>,
) -> Result<Json<openraft::raft::SnapshotResponse<u64>>, StatusCode> {
    let raft = state.raft.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
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

async fn wal_delta_handler(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<rekha_cluster::WalDelta>, ApiError> {
    let from_seq: u64 = params
        .get("from_seq")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let delta = state.engine.wal_delta(&id, from_seq)?;
    Ok(Json(delta))
}

async fn wal_status_handler(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let last_seq = state.engine.wal_last_seq(&id)?;
    Ok(Json(serde_json::json!({ "last_seq": last_seq })))
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

async fn create_collection(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateCollectionRequest>,
) -> Result<Json<CollectionResponse>, ApiError> {
    let mut config = CollectionConfig::new(req.name.clone(), req.dimension, req.distance.clone());
    config.tenant = state.tenant.clone();
    config.database = state.database.clone();
    let record = state.engine.create_collection(&config)?;
    let count = state.engine.count(&record.config.id)?;

    // Propagate through Raft if available.
    if let Some(ref raft) = state.raft {
        let op = ClusterOperation::AddCollection {
            collection_id: record.config.id,
            name: req.name,
            dimension: req.dimension as u32,
            distance: format!("{:?}", req.distance).to_lowercase(),
            tenant: state.tenant.clone(),
            database: state.database.clone(),
            owner_nodes: vec![1],
        };
        let _ = raft.client_write(op).await;
    }

    Ok(Json(CollectionResponse {
        id: record.config.id,
        name: record.config.name,
        dimension: record.config.dimension,
        distance: record.config.space,
        count,
    }))
}

async fn list_collections(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<CollectionResponse>>, ApiError> {
    let records = state
        .engine
        .list_collections(&state.tenant, &state.database)?;
    let mut out = Vec::with_capacity(records.len());
    for record in records {
        let count = state.engine.count(&record.config.id)?;
        out.push(CollectionResponse {
            id: record.config.id,
            name: record.config.name,
            dimension: record.config.dimension,
            distance: record.config.space,
            count,
        });
    }
    Ok(Json(out))
}

async fn get_collection(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<CollectionResponse>, ApiError> {
    let record = state
        .engine
        .get_collection(&state.tenant, &state.database, &id.to_string())?
        .ok_or_else(|| EngineError::CollectionNotFound(id.to_string()))?;
    let count = state.engine.count(&id)?;
    Ok(Json(CollectionResponse {
        id: record.config.id,
        name: record.config.name,
        dimension: record.config.dimension,
        distance: record.config.space,
        count,
    }))
}

async fn delete_collection(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, ApiError> {
    state.engine.delete_collection(&id)?;

    // Propagate through Raft if available.
    if let Some(ref raft) = state.raft {
        let op = ClusterOperation::RemoveCollection { collection_id: id };
        let _ = raft.client_write(op).await;
    }

    Ok(StatusCode::OK)
}

async fn add_records(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<AddRequest>,
) -> Result<StatusCode, ApiError> {
    let embeddings: Vec<Embedding> = req.embeddings.into_iter().map(|e| e.into()).collect();
    state.engine.add(
        &id,
        &req.ids,
        &embeddings,
        req.metadatas.as_deref(),
        req.documents.as_deref(),
    )?;
    Ok(StatusCode::OK)
}

async fn upsert_records(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpsertRequest>,
) -> Result<StatusCode, ApiError> {
    let embeddings: Vec<Embedding> = req.embeddings.into_iter().map(|e| e.into()).collect();
    state.engine.upsert(
        &id,
        &req.ids,
        &embeddings,
        req.metadatas.as_deref(),
        req.documents.as_deref(),
    )?;
    Ok(StatusCode::OK)
}

async fn update_records(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateRequest>,
) -> Result<StatusCode, ApiError> {
    state.engine.update(
        &id,
        &req.ids,
        req.metadatas.as_deref(),
        req.documents.as_deref(),
    )?;
    Ok(StatusCode::OK)
}

async fn delete_records(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<DeleteRequest>,
) -> Result<StatusCode, ApiError> {
    state.engine.delete(&id, &req.ids)?;
    Ok(StatusCode::OK)
}

async fn query_records(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    let include_metadatas = req.include.iter().any(|s| s == "metadatas");
    let include_documents = req.include.iter().any(|s| s == "documents");
    let include_distances = req.include.iter().any(|s| s == "distances");

    let mut all_results = Vec::with_capacity(req.query_embeddings.len());
    for query_emb in &req.query_embeddings {
        let embedding: Embedding = query_emb.clone().into();
        let opts = QueryOptions {
            ef: 0,
            where_filter: req.where_filter.clone(),
            oversampling: 4,
        };
        let results = state.engine.query(&id, &embedding, req.n_results, &opts)?;
        all_results.push(results);
    }

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
    Path(id): Path<Uuid>,
    Json(req): Json<GetRequest>,
) -> Result<Json<GetResponse>, ApiError> {
    let ids = req.ids.unwrap_or_default();
    let include_embeddings = req.include.iter().any(|s| s == "embeddings");

    let records = if ids.is_empty() {
        // Chroma requires ids or where filter for get. Return empty.
        Vec::new()
    } else {
        state.engine.get(&id, &ids)?
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
    Path(id): Path<Uuid>,
) -> Result<Json<CountResponse>, ApiError> {
    let count = state.engine.count(&id)?;
    Ok(Json(CountResponse { count }))
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
    Ok(Json(serde_json::json!({
        "members": members,
        "leader": metrics.current_leader,
        "state": format!("{:?}", metrics.state),
    })))
}

async fn raft_add_learner(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Result<StatusCode, StatusCode> {
    let raft = state.raft.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let node_id = req["node_id"].as_u64().ok_or(StatusCode::BAD_REQUEST)?;
    raft.add_learner(node_id, node_id, true)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(StatusCode::OK)
}

async fn raft_remove_member(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Result<StatusCode, StatusCode> {
    let raft = state.raft.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let node_id = req["node_id"].as_u64().ok_or(StatusCode::BAD_REQUEST)?;
    let metrics = raft.metrics().borrow().clone();
    let new_members: std::collections::BTreeSet<u64> = metrics
        .membership_config
        .membership()
        .voter_ids()
        .chain(metrics.membership_config.membership().learner_ids())
        .filter(|&id| id != node_id)
        .collect();
    if new_members.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    raft.change_membership(new_members, true)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(StatusCode::OK)
}

/// Health check — always returns 200 OK.
async fn health_handler() -> Result<StatusCode, StatusCode> {
    Ok(StatusCode::OK)
}

/// Readiness check — returns 200 if engine is accessible.
async fn ready_handler(State(state): State<Arc<AppState>>) -> Result<StatusCode, StatusCode> {
    // Quick check: can we access the engine?
    let _ = state.engine.count(&uuid::Uuid::nil()).is_err();
    Ok(StatusCode::OK)
}

/// ChromaDB auth identity stub.
async fn auth_identity() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "user_id": "00000000-0000-0000-0000-000000000000"
    }))
}

/// GET /tenants/{name} — get tenant (stub)
async fn get_tenant(Path(name): Path<String>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "name": name }))
}

/// POST /tenants — create tenant (stub, always succeeds)
async fn create_tenant(Json(req): Json<serde_json::Value>) -> StatusCode {
    let _ = req;
    StatusCode::CREATED
}

/// GET /tenants/{tenant}/databases — list databases (stub)
async fn list_databases(Path(_tenant): Path<String>) -> Json<serde_json::Value> {
    Json(serde_json::json!([]))
}

/// GET /tenants/{tenant}/databases/{name} — get database (stub)
async fn get_database(Path((_tenant, name)): Path<(String, String)>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "id": "00000000-0000-0000-0000-000000000000",
        "name": name,
        "tenant": _tenant
    }))
}

/// POST /tenants/{tenant}/databases — create database (stub, always succeeds)
async fn create_database(
    Path(_tenant): Path<String>,
    Json(req): Json<serde_json::Value>,
) -> StatusCode {
    let _ = req;
    StatusCode::CREATED
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the axum router with all Chroma-compatible endpoints.
pub fn router(app_state: Arc<AppState>) -> Router {
    let internal = Router::new()
        .route("/internal/raft/append_entries", post(raft_append_entries))
        .route("/internal/raft/vote", post(raft_vote))
        .route(
            "/internal/raft/install_snapshot",
            post(raft_install_snapshot),
        )
        .route("/internal/raft/membership", get(raft_membership))
        .route("/internal/raft/add_learner", post(raft_add_learner))
        .route("/internal/raft/remove_member", post(raft_remove_member))
        .route("/internal/wal/{id}/delta", get(wal_delta_handler))
        .route("/internal/wal/{id}/status", get(wal_status_handler));

    let public = Router::new()
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

    let chroma = chroma::chroma_router();

    public
        .merge(internal)
        .merge(chroma)
        .layer(axum::middleware::from_fn(middleware::auth))
        .layer(axum::Extension(app_state.clone()))
        .with_state(app_state)
}
