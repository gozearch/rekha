use std::sync::Arc;

use axum_test::TestServer;
use rekha_api::{AppState, ServerConfig, router};
use rekha_engine::{Engine, EngineConfig};
use rekha_storage::{LocalStorage, RedbCatalog};
use tempfile::TempDir;

fn test_app(api_key: Option<String>) -> (TempDir, TestServer) {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    let storage = Arc::new(LocalStorage::new(dir.path().join("objects")));
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    let engine =
        Arc::new(Engine::open(catalog, storage, &wal_dir, EngineConfig::default()).unwrap());
    let state = Arc::new(AppState::new(
        engine,
        None,
        ServerConfig {
            tenant: "default_tenant".into(),
            database: "default_database".into(),
            api_key,
            cluster_secret: None,
            pepper: Vec::new(),
            allow_open_mode: true,
        },
    ));
    let app = router(state);
    let server = TestServer::new(app);
    (dir, server)
}

fn test_app_no_auth() -> (TempDir, TestServer) {
    test_app(None)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn create_collection(server: &TestServer, name: &str, dim: usize) -> serde_json::Value {
    let resp = server
        .post("/api/v2/collections")
        .json(&serde_json::json!({
            "name": name,
            "dimension": dim,
            "distance": "l2"
        }))
        .await;
    let code = resp.status_code();
    assert!(
        code == 200 || code == 201,
        "create_collection failed: {} (code {code})",
        resp.text()
    );
    resp.json::<serde_json::Value>()
}

fn collection_id(v: &serde_json::Value) -> String {
    v["id"].as_str().unwrap().to_owned()
}

// ---------------------------------------------------------------------------
// Health / ready
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_returns_200() {
    let (_dir, server) = test_app_no_auth();
    let resp = server.get("/health").await;
    assert_eq!(resp.status_code(), 200);
}

#[tokio::test]
async fn ready_returns_200_when_healthy() {
    let (_dir, server) = test_app_no_auth();
    let resp = server.get("/ready").await;
    assert_eq!(resp.status_code(), 200);
}

#[tokio::test]
async fn heartbeat_and_version() {
    let (_dir, server) = test_app_no_auth();
    let resp = server.get("/api/v2/heartbeat").await;
    assert_eq!(resp.status_code(), 200);
    let v: serde_json::Value = resp.json();
    assert!(v.get("nanosecond heartbeat").is_some());

    let resp = server.get("/api/v2/version").await;
    assert_eq!(resp.status_code(), 200);

    let resp = server.get("/api/v2/pre-flight-checks").await;
    assert_eq!(resp.status_code(), 200);
    let v: serde_json::Value = resp.json();
    // Must advertise the limit the write handlers actually enforce.
    assert_eq!(
        v["max_batch_size"].as_u64(),
        Some(rekha_api::MAX_BATCH_SIZE as u64)
    );
}

// ---------------------------------------------------------------------------
// Collection CRUD
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_and_get_collection() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "my_coll", 3).await;
    assert_eq!(created["name"], "my_coll");
    assert_eq!(created["dimension"], 3);
    let id = collection_id(&created);

    let resp = server.get(&format!("/api/v2/collections/{id}")).await;
    assert_eq!(resp.status_code(), 200);
    let got: serde_json::Value = resp.json();
    assert_eq!(got["name"], "my_coll");
}

#[tokio::test]
async fn create_collection_empty_name_rejected() {
    let (_dir, server) = test_app_no_auth();
    let resp = server
        .post("/api/v2/collections")
        .json(&serde_json::json!({"name": "", "dimension": 3}))
        .await;
    assert_eq!(resp.status_code(), 400);
}

#[tokio::test]
async fn create_collection_name_too_long_rejected() {
    let (_dir, server) = test_app_no_auth();
    let long = "x".repeat(600);
    let resp = server
        .post("/api/v2/collections")
        .json(&serde_json::json!({"name": long, "dimension": 3}))
        .await;
    assert_eq!(resp.status_code(), 400);
}

#[tokio::test]
async fn create_collection_dimension_too_large_rejected() {
    let (_dir, server) = test_app_no_auth();
    let resp = server
        .post("/api/v2/collections")
        .json(&serde_json::json!({"name": "big_dim", "dimension": 99999}))
        .await;
    assert_eq!(resp.status_code(), 400);
}

#[tokio::test]
async fn get_missing_collection_returns_404() {
    let (_dir, server) = test_app_no_auth();
    let fake = uuid::Uuid::new_v4().to_string();
    let resp = server.get(&format!("/api/v2/collections/{fake}")).await;
    assert_eq!(resp.status_code(), 404);
}

#[tokio::test]
async fn list_collections_returns_created() {
    let (_dir, server) = test_app_no_auth();
    create_collection(&server, "c1", 2).await;
    create_collection(&server, "c2", 2).await;
    let resp = server.get("/api/v2/collections").await;
    assert_eq!(resp.status_code(), 200);
    let list: Vec<serde_json::Value> = resp.json();
    assert_eq!(list.len(), 2);
}

#[tokio::test]
async fn delete_collection() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "to_delete", 2).await;
    let id = collection_id(&created);
    let resp = server.delete(&format!("/api/v2/collections/{id}")).await;
    assert_eq!(resp.status_code(), 200);
    let resp = server.get(&format!("/api/v2/collections/{id}")).await;
    assert_eq!(resp.status_code(), 404);
}

// ---------------------------------------------------------------------------
// Add / Upsert / Update / Delete / Get / Count / Query
// ---------------------------------------------------------------------------

#[tokio::test]
async fn add_and_count() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "vecs", 3).await;
    let id = collection_id(&created);

    let resp = server
        .post(&format!("/api/v2/collections/{id}/add"))
        .json(&serde_json::json!({
            "ids": ["a", "b"],
            "embeddings": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "metadatas": [{"x": 1}, {"x": 2}],
            "documents": ["doc a", "doc b"]
        }))
        .await;
    assert_eq!(resp.status_code(), 200);

    let resp = server
        .post(&format!("/api/v2/collections/{id}/count"))
        .await;
    assert_eq!(resp.status_code(), 200);
    let v: serde_json::Value = resp.json();
    assert_eq!(v["count"], 2);
}

#[tokio::test]
async fn add_mismatched_ids_embeddings_rejected() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "bad_batch", 3).await;
    let id = collection_id(&created);
    let resp = server
        .post(&format!("/api/v2/collections/{id}/add"))
        .json(&serde_json::json!({
            "ids": ["a", "b"],
            "embeddings": [[1.0, 0.0, 0.0]]
        }))
        .await;
    assert_eq!(resp.status_code(), 400);
}

#[tokio::test]
async fn add_empty_ids_rejected() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "empty_ids", 3).await;
    let id = collection_id(&created);
    let resp = server
        .post(&format!("/api/v2/collections/{id}/add"))
        .json(&serde_json::json!({
            "ids": [],
            "embeddings": []
        }))
        .await;
    assert_eq!(resp.status_code(), 400);
}

#[tokio::test]
async fn add_oversized_batch_rejected() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "oversized", 2).await;
    let id = collection_id(&created);
    // 10001 > MAX_BATCH_SIZE (10000)
    let ids: Vec<String> = (0..10001).map(|i| format!("id_{i}")).collect();
    let embeddings: Vec<Vec<f32>> = (0..10001).map(|_| vec![1.0, 0.0]).collect();
    let resp = server
        .post(&format!("/api/v2/collections/{id}/add"))
        .json(&serde_json::json!({"ids": ids, "embeddings": embeddings}))
        .await;
    assert_eq!(resp.status_code(), 400);
}

#[tokio::test]
async fn query_returns_results() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "qcoll", 3).await;
    let id = collection_id(&created);

    server
        .post(&format!("/api/v2/collections/{id}/add"))
        .json(&serde_json::json!({
            "ids": ["a", "b", "c"],
            "embeddings": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        }))
        .await;

    let resp = server
        .post(&format!("/api/v2/collections/{id}/query"))
        .json(&serde_json::json!({
            "query_embeddings": [[1.0, 0.0, 0.0]],
            "n_results": 2
        }))
        .await;
    assert_eq!(resp.status_code(), 200);
    let v: serde_json::Value = resp.json();
    let ids = &v["ids"][0];
    assert_eq!(ids[0], "a");
}

#[tokio::test]
async fn query_invalid_n_results_rejected() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "qbad", 3).await;
    let id = collection_id(&created);
    let resp = server
        .post(&format!("/api/v2/collections/{id}/query"))
        .json(&serde_json::json!({
            "query_embeddings": [[1.0, 0.0, 0.0]],
            "n_results": 0
        }))
        .await;
    assert_eq!(resp.status_code(), 400);

    let resp = server
        .post(&format!("/api/v2/collections/{id}/query"))
        .json(&serde_json::json!({
            "query_embeddings": [[1.0, 0.0, 0.0]],
            "n_results": 2000
        }))
        .await;
    assert_eq!(resp.status_code(), 400);
}

#[tokio::test]
async fn query_empty_embeddings_rejected() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "qempty", 3).await;
    let id = collection_id(&created);
    let resp = server
        .post(&format!("/api/v2/collections/{id}/query"))
        .json(&serde_json::json!({
            "query_embeddings": [],
            "n_results": 5
        }))
        .await;
    assert_eq!(resp.status_code(), 400);
}

#[tokio::test]
async fn get_records_and_delete() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "getdel", 2).await;
    let id = collection_id(&created);

    server
        .post(&format!("/api/v2/collections/{id}/add"))
        .json(&serde_json::json!({
            "ids": ["x", "y"],
            "embeddings": [[1.0, 0.0], [0.0, 1.0]]
        }))
        .await;

    // Get
    let resp = server
        .post(&format!("/api/v2/collections/{id}/get"))
        .json(&serde_json::json!({"ids": ["x"]}))
        .await;
    assert_eq!(resp.status_code(), 200);
    let v: serde_json::Value = resp.json();
    assert_eq!(v["ids"][0], "x");

    // Delete
    let resp = server
        .post(&format!("/api/v2/collections/{id}/delete"))
        .json(&serde_json::json!({"ids": ["x"]}))
        .await;
    assert_eq!(resp.status_code(), 200);

    let resp = server
        .post(&format!("/api/v2/collections/{id}/count"))
        .await;
    let v: serde_json::Value = resp.json();
    assert_eq!(v["count"], 1);
}

#[tokio::test]
async fn delete_empty_ids_rejected() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "delempty", 2).await;
    let id = collection_id(&created);
    let resp = server
        .post(&format!("/api/v2/collections/{id}/delete"))
        .json(&serde_json::json!({"ids": []}))
        .await;
    assert_eq!(resp.status_code(), 400);
}

#[tokio::test]
async fn update_records() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "upd", 2).await;
    let id = collection_id(&created);
    server
        .post(&format!("/api/v2/collections/{id}/add"))
        .json(&serde_json::json!({
            "ids": ["u1"],
            "embeddings": [[1.0, 0.0]],
            "metadatas": [{"a": 1}]
        }))
        .await;
    let resp = server
        .post(&format!("/api/v2/collections/{id}/update"))
        .json(&serde_json::json!({
            "ids": ["u1"],
            "metadatas": [{"a": 2}],
            "documents": ["new doc"]
        }))
        .await;
    assert_eq!(resp.status_code(), 200);
}

#[tokio::test]
async fn upsert_replaces() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "up", 2).await;
    let id = collection_id(&created);
    server
        .post(&format!("/api/v2/collections/{id}/add"))
        .json(&serde_json::json!({
            "ids": ["k"],
            "embeddings": [[1.0, 0.0]]
        }))
        .await;
    let resp = server
        .post(&format!("/api/v2/collections/{id}/upsert"))
        .json(&serde_json::json!({
            "ids": ["k"],
            "embeddings": [[0.0, 1.0]]
        }))
        .await;
    assert_eq!(resp.status_code(), 200);
    // Query should return the updated vector's nearest
    let resp = server
        .post(&format!("/api/v2/collections/{id}/query"))
        .json(&serde_json::json!({
            "query_embeddings": [[0.0, 1.0]],
            "n_results": 1
        }))
        .await;
    let v: serde_json::Value = resp.json();
    assert_eq!(v["ids"][0][0], "k");
}

#[tokio::test]
async fn query_with_where_filter() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "filtered", 2).await;
    let id = collection_id(&created);
    server
        .post(&format!("/api/v2/collections/{id}/add"))
        .json(&serde_json::json!({
            "ids": ["a", "b"],
            "embeddings": [[1.0, 0.0], [1.0, 0.1]],
            "metadatas": [{"color": "red"}, {"color": "blue"}]
        }))
        .await;
    let resp = server
        .post(&format!("/api/v2/collections/{id}/query"))
        .json(&serde_json::json!({
            "query_embeddings": [[1.0, 0.0]],
            "n_results": 10,
            "where_filter": {"color": "red"}
        }))
        .await;
    assert_eq!(resp.status_code(), 200);
    let v: serde_json::Value = resp.json();
    let ids = v["ids"][0].as_array().unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], "a");
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

#[tokio::test]
async fn auth_required_when_key_set() {
    let (_dir, server) = test_app(Some("secret123".into()));
    // No token -> 401
    let resp = server.get("/api/v2/collections").await;
    assert_eq!(resp.status_code(), 401);

    // Wrong token -> 401
    let resp = server
        .get("/api/v2/collections")
        .add_header(
            axum::http::HeaderName::from_static("x-chroma-token"),
            axum::http::HeaderValue::from_static("wrong"),
        )
        .await;
    assert_eq!(resp.status_code(), 401);

    // Correct token via x-chroma-token -> 200
    let resp = server
        .get("/api/v2/collections")
        .add_header(
            axum::http::HeaderName::from_static("x-chroma-token"),
            axum::http::HeaderValue::from_static("secret123"),
        )
        .await;
    assert_eq!(resp.status_code(), 200);

    // Correct token via x-api-key -> 200
    let resp = server
        .get("/api/v2/collections")
        .add_header(
            axum::http::HeaderName::from_static("x-api-key"),
            axum::http::HeaderValue::from_static("secret123"),
        )
        .await;
    assert_eq!(resp.status_code(), 200);
}

#[tokio::test]
async fn auth_not_required_when_no_key() {
    let (_dir, server) = test_app_no_auth();
    let resp = server.get("/api/v2/collections").await;
    assert_eq!(resp.status_code(), 200);
}

// ---------------------------------------------------------------------------
// Chroma compat
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chroma_create_and_query() {
    let (_dir, server) = test_app_no_auth();
    let resp = server
        .post("/api/v2/tenants/default_tenant/databases/default_database/collections")
        .json(&serde_json::json!({"name": "chroma_coll", "dimension": 2}))
        .await;
    assert!(resp.status_code() == 200 || resp.status_code() == 201);

    let resp = server
        .post(
            "/api/v2/tenants/default_tenant/databases/default_database/collections/chroma_coll/add",
        )
        .json(&serde_json::json!({
            "ids": ["c1"],
            "embeddings": [[1.0, 0.0]]
        }))
        .await;
    // Chroma answers `add` with 201 Created.
    assert_eq!(resp.status_code(), 201);

    let resp = server
        .post("/api/v2/tenants/default_tenant/databases/default_database/collections/chroma_coll/query")
        .json(&serde_json::json!({
            "query_embeddings": [[1.0, 0.0]],
            "n_results": 1
        }))
        .await;
    assert_eq!(resp.status_code(), 200);
    let v: serde_json::Value = resp.json();
    assert_eq!(v["ids"][0][0], "c1");
}

#[tokio::test]
async fn chroma_get_or_create() {
    let (_dir, server) = test_app_no_auth();
    // First create
    let resp = server
        .post("/api/v2/tenants/default_tenant/databases/default_database/collections")
        .json(&serde_json::json!({"name": "goc", "dimension": 2, "get_or_create": true}))
        .await;
    assert!(resp.status_code() == 200 || resp.status_code() == 201);
    // Second with same name and get_or_create true should return same
    let resp2 = server
        .post("/api/v2/tenants/default_tenant/databases/default_database/collections")
        .json(&serde_json::json!({"name": "goc", "dimension": 2, "get_or_create": true}))
        .await;
    assert_eq!(resp2.status_code(), 200);
    let v1: serde_json::Value = resp.json();
    let v2: serde_json::Value = resp2.json();
    assert_eq!(v1["id"], v2["id"]);
}

#[tokio::test]
async fn chroma_validation_rejects_oversized_batch() {
    let (_dir, server) = test_app_no_auth();
    server
        .post("/api/v2/tenants/default_tenant/databases/default_database/collections")
        .json(&serde_json::json!({"name": "big", "dimension": 2}))
        .await;
    let ids: Vec<String> = (0..10001).map(|i| format!("id_{i}")).collect();
    let embeddings: Vec<Vec<f32>> = (0..10001).map(|_| vec![1.0, 0.0]).collect();
    let resp = server
        .post("/api/v2/tenants/default_tenant/databases/default_database/collections/big/add")
        .json(&serde_json::json!({"ids": ids, "embeddings": embeddings}))
        .await;
    assert_eq!(resp.status_code(), 400);
}

// ---------------------------------------------------------------------------
// Collection not found for record ops
// ---------------------------------------------------------------------------

#[tokio::test]
async fn record_ops_on_missing_collection_returns_404_or_500() {
    let (_dir, server) = test_app_no_auth();
    let fake = uuid::Uuid::new_v4().to_string();
    let resp = server
        .post(&format!("/api/v2/collections/{fake}/add"))
        .json(&serde_json::json!({"ids": ["x"], "embeddings": [[1.0, 0.0]]}))
        .await;
    // Engine returns CollectionNotFound which maps to 404
    assert_eq!(resp.status_code(), 404);
}

// ---------------------------------------------------------------------------
// Body size limit is not enforced for normal payloads
// ---------------------------------------------------------------------------

#[tokio::test]
async fn body_within_limit_succeeds() {
    let (_dir, server) = test_app_no_auth();
    let created = create_collection(&server, "body_ok", 2).await;
    let id = collection_id(&created);
    let resp = server
        .post(&format!("/api/v2/collections/{id}/add"))
        .json(&serde_json::json!({
            "ids": ["a"],
            "embeddings": [[1.0, 0.0]]
        }))
        .await;
    assert_eq!(resp.status_code(), 200);
}

// ---------------------------------------------------------------------------
// Chroma compatibility: metric, filters, pagination, include defaults
// ---------------------------------------------------------------------------

const CHROMA_BASE: &str = "/api/v2/tenants/default_tenant/databases/default_database/collections";

async fn chroma_create(
    server: &TestServer,
    name: &str,
    dimension: Option<usize>,
    metadata: Option<serde_json::Value>,
) -> serde_json::Value {
    let mut body = serde_json::json!({"name": name});
    if let Some(d) = dimension {
        body["dimension"] = serde_json::json!(d);
    }
    if let Some(m) = metadata {
        body["metadata"] = m;
    }
    let resp = server.post(CHROMA_BASE).json(&body).await;
    assert!(
        resp.status_code().is_success(),
        "chroma create failed: {}",
        resp.text()
    );
    resp.json()
}

async fn chroma_add_tagged(server: &TestServer, name: &str) {
    let resp = server
        .post(&format!("{CHROMA_BASE}/{name}/add"))
        .json(&serde_json::json!({
            "ids": ["a0", "b0", "a1", "b1", "a2"],
            "embeddings": [
                [1.0, 0.0], [0.0, 1.0], [1.1, 0.0], [0.0, 1.1], [1.2, 0.0]
            ],
            "metadatas": [
                {"tag": "a"}, {"tag": "b"}, {"tag": "a"}, {"tag": "b"}, {"tag": "a"}
            ],
            "documents": ["doc0", "doc1", "doc2", "doc3", "doc4"]
        }))
        .await;
    assert!(resp.status_code().is_success(), "{}", resp.text());
}

/// `hnsw:space` must select the metric; it used to be hardcoded to L2, which
/// silently changed reported distances for cosine collections.
#[tokio::test]
async fn chroma_create_honors_cosine_space() {
    let (_dir, server) = test_app_no_auth();
    let created = chroma_create(
        &server,
        "cos",
        Some(2),
        Some(serde_json::json!({"hnsw:space": "cosine"})),
    )
    .await;
    assert_eq!(created["configuration_json"]["hnsw"]["space"], "cosine");

    let resp = server
        .post(&format!("{CHROMA_BASE}/cos/add"))
        .json(&serde_json::json!({
            "ids": ["x", "y"],
            "embeddings": [[1.0, 0.0], [0.0, 1.0]]
        }))
        .await;
    assert!(resp.status_code().is_success(), "{}", resp.text());

    let resp = server
        .post(&format!("{CHROMA_BASE}/cos/query"))
        .json(&serde_json::json!({
            "query_embeddings": [[1.0, 0.0]],
            "n_results": 2,
            "include": ["distances"]
        }))
        .await;
    assert!(resp.status_code().is_success(), "{}", resp.text());
    let v: serde_json::Value = resp.json();
    let d = v["distances"][0].as_array().unwrap();
    assert_eq!(d[0].as_f64().unwrap(), 0.0);
    // Orthogonal unit vectors: cosine distance 1, L2-squared would be 2.
    assert!(
        (d[1].as_f64().unwrap() - 1.0).abs() < 1e-5,
        "cosine distance should be 1, got {d:?}"
    );
}

/// An unknown `hnsw:space` must be rejected rather than silently defaulting.
#[tokio::test]
async fn chroma_create_rejects_unknown_space() {
    let (_dir, server) = test_app_no_auth();
    let resp = server
        .post(CHROMA_BASE)
        .json(&serde_json::json!({
            "name": "bogus",
            "dimension": 2,
            "metadata": {"hnsw:space": "manhattan"}
        }))
        .await;
    assert_eq!(resp.status_code(), 400, "{}", resp.text());
}

/// `get` must honour `where` (it used to be parsed and ignored) and paginate.
#[tokio::test]
async fn chroma_get_applies_where_and_pagination() {
    let (_dir, server) = test_app_no_auth();
    chroma_create(&server, "gets", Some(2), None).await;
    chroma_add_tagged(&server, "gets").await;

    let resp = server
        .post(&format!("{CHROMA_BASE}/gets/get"))
        .json(&serde_json::json!({"where": {"tag": {"$eq": "a"}}}))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());
    let v: serde_json::Value = resp.json();
    assert_eq!(
        v["ids"].as_array().unwrap().len(),
        3,
        "filter-only get: {v}"
    );

    let resp = server
        .post(&format!("{CHROMA_BASE}/gets/get"))
        .json(&serde_json::json!({"limit": 2}))
        .await;
    let v: serde_json::Value = resp.json();
    assert_eq!(v["ids"].as_array().unwrap().len(), 2);

    let resp = server
        .post(&format!("{CHROMA_BASE}/gets/get"))
        .json(&serde_json::json!({"limit": 2, "offset": 2}))
        .await;
    let v2: serde_json::Value = resp.json();
    assert_eq!(v2["ids"].as_array().unwrap().len(), 2);
    assert_ne!(v["ids"], v2["ids"], "offset must advance the window");
}

/// `delete` must support a metadata filter, and combine it with `ids`.
#[tokio::test]
async fn chroma_delete_by_where() {
    let (_dir, server) = test_app_no_auth();
    chroma_create(&server, "dels", Some(2), None).await;
    chroma_add_tagged(&server, "dels").await;

    let resp = server
        .post(&format!("{CHROMA_BASE}/dels/delete"))
        .json(&serde_json::json!({"where": {"tag": {"$eq": "b"}}}))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());

    let resp = server.get(&format!("{CHROMA_BASE}/dels/count")).await;
    let v: serde_json::Value = resp.json();
    assert_eq!(v.as_u64(), Some(3), "only tag=a records should remain");

    // ids ∩ where: b0 is already gone, so nothing should be removed.
    let resp = server
        .post(&format!("{CHROMA_BASE}/dels/delete"))
        .json(&serde_json::json!({
            "ids": ["a0"],
            "where": {"tag": {"$eq": "b"}}
        }))
        .await;
    assert_eq!(resp.status_code(), 200);
    let resp = server.get(&format!("{CHROMA_BASE}/dels/count")).await;
    let v: serde_json::Value = resp.json();
    assert_eq!(v.as_u64(), Some(3), "intersection must not delete tag=a");
}

/// `get` and `query` must fall back to Chroma's documented defaults when
/// `include` is omitted, instead of returning ids only.
#[tokio::test]
async fn chroma_include_defaults() {
    let (_dir, server) = test_app_no_auth();
    chroma_create(&server, "incl", Some(2), None).await;
    chroma_add_tagged(&server, "incl").await;

    let resp = server
        .post(&format!("{CHROMA_BASE}/incl/get"))
        .json(&serde_json::json!({}))
        .await;
    let v: serde_json::Value = resp.json();
    assert!(!v["metadatas"].is_null(), "get defaults to metadatas: {v}");
    assert!(!v["documents"].is_null(), "get defaults to documents: {v}");

    let resp = server
        .post(&format!("{CHROMA_BASE}/incl/query"))
        .json(&serde_json::json!({
            "query_embeddings": [[1.0, 0.0]],
            "n_results": 2
        }))
        .await;
    let v: serde_json::Value = resp.json();
    assert!(
        !v["distances"].is_null(),
        "query defaults to distances: {v}"
    );
    assert!(
        !v["metadatas"].is_null(),
        "query defaults to metadatas: {v}"
    );
    assert!(
        !v["documents"].is_null(),
        "query defaults to documents: {v}"
    );
}

/// Unknown `include` fields must be rejected, not silently dropped.
#[tokio::test]
async fn chroma_unknown_include_rejected() {
    let (_dir, server) = test_app_no_auth();
    chroma_create(&server, "badinc", Some(2), None).await;
    chroma_add_tagged(&server, "badinc").await;

    let resp = server
        .post(&format!("{CHROMA_BASE}/badinc/get"))
        .json(&serde_json::json!({"include": ["nonsense"]}))
        .await;
    assert_eq!(resp.status_code(), 400, "{}", resp.text());
}

/// `update` from chromadb >= 1.5 sends `"embeddings": null` for a
/// metadata-only edit; the request type must accept it (it used to be a
/// required array and the whole update 422'd).
#[tokio::test]
async fn chroma_update_accepts_null_embeddings() {
    let (_dir, server) = test_app_no_auth();
    chroma_create(&server, "nullupd", Some(2), None).await;
    let resp = server
        .post(&format!("{CHROMA_BASE}/nullupd/add"))
        .json(&serde_json::json!({
            "ids": ["u0"],
            "embeddings": [[1.0, 0.0]],
            "metadatas": [{"k": 1}]
        }))
        .await;
    assert!(resp.status_code().is_success(), "{}", resp.text());

    let resp = server
        .post(&format!("{CHROMA_BASE}/nullupd/update"))
        .json(&serde_json::json!({
            "ids": ["u0"],
            "embeddings": null,
            "metadatas": [{"k": 2}]
        }))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());

    let resp = server
        .post(&format!("{CHROMA_BASE}/nullupd/get"))
        .json(&serde_json::json!({"ids": ["u0"], "include": ["metadatas"]}))
        .await;
    let v: serde_json::Value = resp.json();
    assert_eq!(v["metadatas"][0]["k"], 2);
}

/// chromadb >= 1.5 sets the width via `configuration.hnsw.dimension`, not a
/// top-level `dimension` field. Ignoring it silently inferred the width from
/// the first write instead.
#[tokio::test]
async fn chroma_create_reads_dimension_from_configuration() {
    let (_dir, server) = test_app_no_auth();
    let resp = server
        .post(CHROMA_BASE)
        .json(&serde_json::json!({
            "name": "cfgdim",
            "configuration": {"hnsw": {"dimension": 4}}
        }))
        .await;
    assert!(resp.status_code().is_success(), "{}", resp.text());
    let v: serde_json::Value = resp.json();
    assert_eq!(v["dimension"], 4, "dimension must come from configuration");

    let resp = server
        .post(&format!("{CHROMA_BASE}/cfgdim/add"))
        .json(&serde_json::json!({
            "ids": ["c0"],
            "embeddings": [[0.1, 0.1, 0.1, 0.1]]
        }))
        .await;
    assert!(resp.status_code().is_success(), "{}", resp.text());

    // Wrong width must now be rejected because the dimension is real.
    let resp = server
        .post(&format!("{CHROMA_BASE}/cfgdim/add"))
        .json(&serde_json::json!({
            "ids": ["c1"],
            "embeddings": [[0.1, 0.1]]
        }))
        .await;
    assert_eq!(resp.status_code(), 400, "{}", resp.text());
}

/// `update` may also carry a new vector, which must take effect.
#[tokio::test]
async fn chroma_update_with_embedding_replaces_vector() {
    let (_dir, server) = test_app_no_auth();
    chroma_create(&server, "updvec", Some(2), None).await;
    server
        .post(&format!("{CHROMA_BASE}/updvec/add"))
        .json(&serde_json::json!({
            "ids": ["u0"],
            "embeddings": [[1.0, 0.0]]
        }))
        .await;

    let resp = server
        .post(&format!("{CHROMA_BASE}/updvec/update"))
        .json(&serde_json::json!({
            "ids": ["u0"],
            "embeddings": [[0.0, 1.0]]
        }))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());

    let resp = server
        .post(&format!("{CHROMA_BASE}/updvec/get"))
        .json(&serde_json::json!({"ids": ["u0"], "include": ["embeddings"]}))
        .await;
    let v: serde_json::Value = resp.json();
    assert_eq!(v["embeddings"][0], serde_json::json!([0.0, 1.0]));
}

/// A collection UUID must not be a skeleton key into another tenant.
///
/// Chroma addresses a collection as `.../tenants/{t}/databases/{d}/collections/{name}`
/// where `{name}` may be a UUID. Resolution used to parse the UUID and return
/// the collection without comparing its owning tenant/database, so anyone who
/// knew an id could read, write, or delete another tenant's collection.
#[tokio::test]
async fn chroma_uuid_does_not_bypass_tenant_scope() {
    let (_dir, server) = test_app_no_auth();
    let created = chroma_create(&server, "scoped", Some(2), None).await;
    let uuid = created["id"].as_str().unwrap().to_owned();

    // Same tenant/database: the UUID is a valid reference.
    let resp = server.get(&format!("{CHROMA_BASE}/{uuid}")).await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());

    // Same tenant, different database.
    let resp = server
        .get(&format!(
            "/api/v2/tenants/default_tenant/databases/other_database/collections/{uuid}"
        ))
        .await;
    assert_eq!(
        resp.status_code(),
        404,
        "UUID resolved across databases: {}",
        resp.text()
    );

    // Different tenant.
    let resp = server
        .get(&format!(
            "/api/v2/tenants/other_tenant/databases/default_database/collections/{uuid}"
        ))
        .await;
    assert_eq!(
        resp.status_code(),
        404,
        "UUID resolved across tenants: {}",
        resp.text()
    );

    // Reads through the same bypass must be closed too.
    let resp = server
        .post(&format!(
            "/api/v2/tenants/other_tenant/databases/default_database/collections/{uuid}/get"
        ))
        .json(&serde_json::json!({}))
        .await;
    assert_eq!(resp.status_code(), 404, "{}", resp.text());

    // And a destructive request must not reach the collection.
    let resp = server
        .delete(&format!(
            "/api/v2/tenants/other_tenant/databases/default_database/collections/{uuid}"
        ))
        .await;
    assert_eq!(resp.status_code(), 404, "{}", resp.text());

    let resp = server.get(&format!("{CHROMA_BASE}/{uuid}")).await;
    assert_eq!(
        resp.status_code(),
        200,
        "collection was deleted cross-tenant"
    );
}

/// A delete with neither `ids` nor `where` is a no-op in Chroma.
///
/// Rekha used to reject it with 400 on the theory that it would wipe the
/// collection. Chroma returns `{"deleted": 0}` and deletes nothing, which is
/// both safer and the behaviour clients are written against.
#[tokio::test]
async fn chroma_delete_without_selector_is_a_noop() {
    let (_dir, server) = test_app_no_auth();
    chroma_create(&server, "noop", Some(2), None).await;
    chroma_add_tagged(&server, "noop").await;

    let resp = server
        .post(&format!("{CHROMA_BASE}/noop/delete"))
        .json(&serde_json::json!({}))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());
    let v: serde_json::Value = resp.json();
    assert_eq!(v["deleted"], 0);

    let resp = server.get(&format!("{CHROMA_BASE}/noop/count")).await;
    let v: serde_json::Value = resp.json();
    assert_eq!(
        v.as_u64(),
        Some(5),
        "records must survive a selector-free delete"
    );
}

/// Chroma reports the number of submitted ids, including ones that do not
/// exist, so callers can account for their own request.
#[tokio::test]
async fn chroma_delete_reports_submitted_id_count() {
    let (_dir, server) = test_app_no_auth();
    chroma_create(&server, "cnt", Some(2), None).await;
    chroma_add_tagged(&server, "cnt").await;

    let resp = server
        .post(&format!("{CHROMA_BASE}/cnt/delete"))
        .json(&serde_json::json!({"ids": ["a0", "does-not-exist"]}))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());
    let v: serde_json::Value = resp.json();
    assert_eq!(v["deleted"], 2, "count describes the request: {v}");

    // `where` reports the number actually matched.
    let resp = server
        .post(&format!("{CHROMA_BASE}/cnt/delete"))
        .json(&serde_json::json!({"where": {"tag": {"$eq": "b"}}}))
        .await;
    let v: serde_json::Value = resp.json();
    assert_eq!(v["deleted"], 2, "two tag=b records remain: {v}");
}

/// Creating a collection whose name already exists is a conflict, not a crash.
///
/// The catalog returns `AlreadyExists`, which mapped to a bare 500. Chroma
/// answers 409, and a 500 tells the caller to retry a request that can never
/// succeed.
#[tokio::test]
async fn chroma_duplicate_collection_is_conflict() {
    let (_dir, server) = test_app_no_auth();
    chroma_create(&server, "dup", Some(2), None).await;

    let resp = server
        .post(CHROMA_BASE)
        .json(&serde_json::json!({"name": "dup", "dimension": 2}))
        .await;
    assert_eq!(resp.status_code(), 409, "{}", resp.text());

    // `get_or_create` must still succeed on the same name.
    let resp = server
        .post(CHROMA_BASE)
        .json(&serde_json::json!({"name": "dup", "dimension": 2, "get_or_create": true}))
        .await;
    assert!(resp.status_code().is_success(), "{}", resp.text());
}

/// `include: []` means ids only; an omitted `include` means the defaults.
///
/// Both were treated as "empty", so an explicit empty list silently returned
/// metadatas and documents.
#[tokio::test]
async fn chroma_empty_include_is_not_the_default() {
    let (_dir, server) = test_app_no_auth();
    chroma_create(&server, "inc0", Some(2), None).await;
    chroma_add_tagged(&server, "inc0").await;

    // Explicit empty list -> ids only.
    let resp = server
        .post(&format!("{CHROMA_BASE}/inc0/get"))
        .json(&serde_json::json!({"include": []}))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());
    let v: serde_json::Value = resp.json();
    assert!(
        v["metadatas"].is_null(),
        "explicit [] must not include metadatas: {v}"
    );
    assert!(
        v["documents"].is_null(),
        "explicit [] must not include documents: {v}"
    );
    assert!(!v["ids"].as_array().unwrap().is_empty());

    // Omitted -> documented defaults.
    let resp = server
        .post(&format!("{CHROMA_BASE}/inc0/get"))
        .json(&serde_json::json!({}))
        .await;
    let v: serde_json::Value = resp.json();
    assert!(
        !v["metadatas"].is_null(),
        "omitted include uses defaults: {v}"
    );
    assert!(
        !v["documents"].is_null(),
        "omitted include uses defaults: {v}"
    );
}

/// `include: null` is rejected, as the reference does.
#[tokio::test]
async fn chroma_null_include_is_rejected() {
    let (_dir, server) = test_app_no_auth();
    chroma_create(&server, "incnull", Some(2), None).await;
    chroma_add_tagged(&server, "incnull").await;

    let resp = server
        .post(&format!("{CHROMA_BASE}/incnull/get"))
        .json(&serde_json::json!({"include": null}))
        .await;
    assert_eq!(resp.status_code(), 422, "{}", resp.text());
}

/// The response echoes the include list that was applied.
#[tokio::test]
async fn chroma_query_echoes_include() {
    let (_dir, server) = test_app_no_auth();
    chroma_create(&server, "incecho", Some(2), None).await;
    chroma_add_tagged(&server, "incecho").await;

    let resp = server
        .post(&format!("{CHROMA_BASE}/incecho/query"))
        .json(&serde_json::json!({
            "query_embeddings": [[1.0, 0.0]],
            "n_results": 1
        }))
        .await;
    let v: serde_json::Value = resp.json();
    let echoed: Vec<String> = v["include"]
        .as_array()
        .expect("include is echoed")
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect();
    let mut sorted = echoed.clone();
    sorted.sort();
    assert_eq!(sorted, vec!["distances", "documents", "metadatas"]);
}
