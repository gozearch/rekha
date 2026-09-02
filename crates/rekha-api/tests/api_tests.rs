use std::sync::Arc;

use axum_test::TestServer;
use rekha_api::{router, AppState};
use rekha_engine::{Engine, EngineConfig};
use rekha_storage::{LocalStorage, RedbCatalog};
use tempfile::TempDir;

fn test_app(api_key: Option<String>) -> (TempDir, TestServer) {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(
        RedbCatalog::open(dir.path().join("catalog.redb")).unwrap(),
    );
    let storage = Arc::new(LocalStorage::new(dir.path().join("objects")));
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    let engine = Arc::new(
        Engine::open(catalog, storage, &wal_dir, EngineConfig::default()).unwrap(),
    );
    let state = Arc::new(AppState {
        engine,
        tenant: "default_tenant".into(),
        database: "default_database".into(),
        raft: None,
        api_key,
    });
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
    assert_eq!(v["max_batch_size"], 1000);
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
        .post("/api/v2/tenants/default_tenant/databases/default_database/collections/chroma_coll/add")
        .json(&serde_json::json!({
            "ids": ["c1"],
            "embeddings": [[1.0, 0.0]]
        }))
        .await;
    assert_eq!(resp.status_code(), 200);

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
