use std::sync::Arc;

use axum_test::TestServer;
use rekha_api::{AppState, ServerConfig, router};
use rekha_core::auth::{ROLE_READER, ROLE_WRITER, generate_api_key, hash_api_key_with_pepper};
use rekha_engine::{Engine, EngineConfig};
use rekha_storage::{Catalog, LocalStorage, RedbCatalog};
use tempfile::TempDir;

fn test_app_rbac() -> (TempDir, TestServer, String) {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    catalog.ensure_builtin_roles().unwrap();
    // Create superadmin user
    let super_key = generate_api_key();
    let hash = hash_api_key_with_pepper(&super_key, &[]);
    let user = rekha_core::auth::UserRecord {
        id: uuid::Uuid::new_v4().to_string(),
        name: "superadmin".into(),
        enabled: true,
        created_at_ms: 0,
    };
    catalog.create_user(&user).unwrap();
    catalog.create_api_key(&user.id, &hash, "test").unwrap();
    // Assign superadmin role
    let role = catalog.get_role_by_name("superadmin").unwrap().unwrap();
    catalog.assign_role(&user.id, &role.id, &user.id).unwrap();

    let storage = Arc::new(LocalStorage::new(dir.path().join("objects")));
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    let engine = Arc::new(
        Engine::open(catalog.clone(), storage, &wal_dir, EngineConfig::default()).unwrap(),
    );
    let state = Arc::new(AppState::new(
        engine,
        None,
        ServerConfig {
            tenant: "default_tenant".into(),
            database: "default_database".into(),
            api_key: None,
            cluster_secret: None,
            pepper: Vec::new(),
            allow_open_mode: true,
        },
    ));
    let server = TestServer::new(router(state));
    (dir, server, super_key)
}

fn api_key_header(key: &str) -> (axum::http::HeaderName, axum::http::HeaderValue) {
    (
        axum::http::HeaderName::from_static("x-api-key"),
        axum::http::HeaderValue::from_str(key).unwrap(),
    )
}

#[tokio::test]
async fn rbac_superadmin_can_create_user() {
    let (_dir, server, super_key) = test_app_rbac();
    let (k, v) = api_key_header(&super_key);
    let resp = server
        .post("/api/v2/auth/users")
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "name": "alice" }))
        .await;
    assert_eq!(
        resp.status_code(),
        200,
        "create user failed: {}",
        resp.text()
    );
    let body: serde_json::Value = resp.json();
    assert_eq!(body["name"], "alice");
    assert!(body["api_key"].as_str().unwrap().starts_with("rekha_sk_"));
}

#[tokio::test]
async fn rbac_unauthenticated_is_denied_when_rbac_active() {
    let (_dir, server, _super_key) = test_app_rbac();
    // No key provided, RBAC is active (users exist), should be 401
    let resp = server
        .post("/api/v2/auth/users")
        .json(&serde_json::json!({ "name": "bob" }))
        .await;
    assert_eq!(resp.status_code(), 401);
}

#[tokio::test]
async fn rbac_reader_cannot_write() {
    let (_dir, server, super_key) = test_app_rbac();
    let (k, v) = api_key_header(&super_key);

    // Create reader user
    let resp = server
        .post("/api/v2/auth/users")
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "name": "reader_user" }))
        .await;
    let body: serde_json::Value = resp.json();
    let user_id = body["id"].as_str().unwrap().to_owned();
    let reader_key = body["api_key"].as_str().unwrap().to_owned();

    // Assign reader role
    let roles_resp = server
        .get("/api/v2/auth/roles")
        .add_header(k.clone(), v.clone())
        .await;
    let roles: Vec<serde_json::Value> = roles_resp.json();
    let reader_role = roles.iter().find(|r| r["name"] == ROLE_READER).unwrap();
    let role_id = reader_role["id"].as_str().unwrap();

    let resp = server
        .post(&format!("/api/v2/auth/users/{user_id}/roles"))
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "role_id": role_id }))
        .await;
    assert_eq!(resp.status_code(), 200);

    // Reader tries to create a collection — should be forbidden (reader has no collection.create)
    let (rk, rv) = api_key_header(&reader_key);
    let resp = server
        .post("/api/v2/collections")
        .add_header(rk, rv)
        .json(&serde_json::json!({ "name": "test_coll", "dimension": 2 }))
        .await;
    assert_eq!(
        resp.status_code(),
        403,
        "reader should not be able to create collection: {}",
        resp.text()
    );
}

#[tokio::test]
async fn rbac_writer_can_write_and_query() {
    let (_dir, server, super_key) = test_app_rbac();
    let (k, v) = api_key_header(&super_key);

    // Create writer user
    let resp = server
        .post("/api/v2/auth/users")
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "name": "writer_user" }))
        .await;
    let body: serde_json::Value = resp.json();
    let user_id = body["id"].as_str().unwrap().to_owned();
    let writer_key = body["api_key"].as_str().unwrap().to_owned();

    // Assign writer role
    let roles_resp = server
        .get("/api/v2/auth/roles")
        .add_header(k.clone(), v.clone())
        .await;
    let roles: Vec<serde_json::Value> = roles_resp.json();
    let writer_role = roles.iter().find(|r| r["name"] == ROLE_WRITER).unwrap();
    let role_id = writer_role["id"].as_str().unwrap();

    server
        .post(&format!("/api/v2/auth/users/{user_id}/roles"))
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "role_id": role_id }))
        .await;

    // Writer creates collection — should succeed
    let (wk, wv) = api_key_header(&writer_key);
    let resp = server
        .post("/api/v2/collections")
        .add_header(wk.clone(), wv.clone())
        .json(&serde_json::json!({ "name": "writer_coll", "dimension": 2 }))
        .await;
    assert!(
        resp.status_code() == 200 || resp.status_code() == 201,
        "writer should create collection: {}",
        resp.text()
    );
    let coll: serde_json::Value = resp.json();
    let coll_id = coll["id"].as_str().unwrap().to_owned();

    // Writer adds records
    let resp = server
        .post(&format!("/api/v2/collections/{coll_id}/add"))
        .add_header(wk.clone(), wv.clone())
        .json(&serde_json::json!({
            "ids": ["a"],
            "embeddings": [[1.0, 0.0]]
        }))
        .await;
    assert_eq!(
        resp.status_code(),
        200,
        "writer add failed: {}",
        resp.text()
    );

    // Writer queries
    let resp = server
        .post(&format!("/api/v2/collections/{coll_id}/query"))
        .add_header(wk, wv)
        .json(&serde_json::json!({
            "query_embeddings": [[1.0, 0.0]],
            "n_results": 1
        }))
        .await;
    assert_eq!(resp.status_code(), 200);
}

#[tokio::test]
async fn rbac_tenant_scoped_permission() {
    let (_dir, server, super_key) = test_app_rbac();
    let (k, v) = api_key_header(&super_key);

    // Create a custom role with tenant-scoped permission
    let resp = server
        .post("/api/v2/auth/roles")
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "name": "acme_reader", "description": "reader for acme" }))
        .await;
    assert_eq!(resp.status_code(), 200);
    let role: serde_json::Value = resp.json();
    let role_id = role["id"].as_str().unwrap().to_owned();

    // Add permission for acme/prod only
    let resp = server
        .post(&format!("/api/v2/auth/roles/{role_id}/permissions"))
        .add_header(k.clone(), v.clone())
        .json(
            &serde_json::json!({ "action": "record.query", "tenant": "acme", "database": "prod" }),
        )
        .await;
    assert_eq!(resp.status_code(), 200);

    // Create user and assign scoped role
    let resp = server
        .post("/api/v2/auth/users")
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "name": "scoped_user" }))
        .await;
    let body: serde_json::Value = resp.json();
    let user_id = body["id"].as_str().unwrap().to_owned();
    let scoped_key = body["api_key"].as_str().unwrap().to_owned();

    server
        .post(&format!("/api/v2/auth/users/{user_id}/roles"))
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "role_id": role_id }))
        .await;

    // Try to query a collection in acme/prod via Chroma path — should succeed (middleware checks tenant/database)
    // First create a collection in acme/prod via superadmin
    let resp = server
        .post("/api/v2/tenants/acme/databases/prod/collections")
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "name": "scoped_coll", "dimension": 2 }))
        .await;
    assert!(
        resp.status_code() == 200 || resp.status_code() == 201,
        "create scoped coll failed: {}",
        resp.text()
    );

    // Scoped user queries via Chroma path — should be allowed
    let (sk, sv) = api_key_header(&scoped_key);
    let resp = server
        .post("/api/v2/tenants/acme/databases/prod/collections/scoped_coll/query")
        .add_header(sk.clone(), sv.clone())
        .json(&serde_json::json!({ "query_embeddings": [[1.0, 0.0]], "n_results": 1 }))
        .await;
    // Even with 0 vectors, query should be 200 (not 403)
    assert_eq!(
        resp.status_code(),
        200,
        "scoped query should be allowed: {}",
        resp.text()
    );

    // Same user tries to query other tenant — should be forbidden
    let resp = server
        .post("/api/v2/tenants/other/databases/prod/collections/scoped_coll/query")
        .add_header(sk, sv)
        .json(&serde_json::json!({ "query_embeddings": [[1.0, 0.0]], "n_results": 1 }))
        .await;
    assert_eq!(
        resp.status_code(),
        403,
        "cross-tenant should be forbidden: {}",
        resp.text()
    );
}

#[tokio::test]
async fn rbac_whoami() {
    let (_dir, server, super_key) = test_app_rbac();
    let (k, v) = api_key_header(&super_key);
    let resp = server.get("/api/v2/auth/whoami").add_header(k, v).await;
    assert_eq!(resp.status_code(), 200);
    let body: serde_json::Value = resp.json();
    assert_eq!(body["user_name"], "superadmin");
    assert!(
        body["permissions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["action"] == "*")
    );
}

#[tokio::test]
async fn rbac_internal_secret() {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    catalog.ensure_builtin_roles().unwrap();
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
            api_key: None,
            cluster_secret: Some("s3cr3t".into()),
            pepper: Vec::new(),
            allow_open_mode: true,
        },
    ));
    let app = rekha_api::internal_router(state);
    let server = TestServer::new(app);

    // No secret -> 401
    let resp = server.get("/internal/raft/membership").await;
    assert_eq!(resp.status_code(), 401);

    // Wrong secret -> 401
    let resp = server
        .get("/internal/raft/membership")
        .add_header(
            axum::http::HeaderName::from_static("x-rekha-cluster-secret"),
            axum::http::HeaderValue::from_static("wrong"),
        )
        .await;
    assert_eq!(resp.status_code(), 401);

    // Correct secret -> 200 or 503 (no raft, but auth passes, then handler returns 503)
    let resp = server
        .get("/internal/raft/membership")
        .add_header(
            axum::http::HeaderName::from_static("x-rekha-cluster-secret"),
            axum::http::HeaderValue::from_static("s3cr3t"),
        )
        .await;
    // Membership handler returns 503 when no raft, but not 401
    assert_ne!(resp.status_code(), 401);
}

/// A bootstrap key must authenticate when a pepper is configured.
///
/// The catalog used to hash the bootstrap superadmin key *unpeppered* while the
/// auth middleware verified with the server pepper, so the key printed at
/// startup could never be used once `REKHA_KEY_PEPPER` was set. This asserts the
/// two sides hash identically.
#[tokio::test]
async fn bootstrap_key_authenticates_with_pepper() {
    let dir = TempDir::new().unwrap();
    let pepper = b"test-pepper".to_vec();

    let catalog = Arc::new(
        RedbCatalog::open_with_pepper(dir.path().join("catalog.redb"), pepper.clone()).unwrap(),
    );
    let storage = Arc::new(LocalStorage::new(dir.path().join("objects")));
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    let engine =
        Arc::new(Engine::open(catalog, storage, &wal_dir, EngineConfig::default()).unwrap());
    engine.catalog().ensure_builtin_roles().unwrap();
    let (_, key) = engine
        .catalog()
        .bootstrap_superadmin_if_empty(None)
        .unwrap()
        .expect("bootstrap should create a superadmin");

    let state = Arc::new(AppState::new(
        engine,
        None,
        ServerConfig {
            tenant: "default_tenant".into(),
            database: "default_database".into(),
            api_key: None,
            cluster_secret: None,
            pepper,
            allow_open_mode: false,
        },
    ));
    let server = TestServer::new(router(state));

    // RBAC is active and no key was supplied -> unauthorized.
    let resp = server.get("/api/v2/auth/whoami").await;
    assert_eq!(resp.status_code(), 401);

    // The printed bootstrap key must be accepted.
    let (k, v) = api_key_header(&key);
    let resp = server.get("/api/v2/auth/whoami").add_header(k, v).await;
    assert_eq!(
        resp.status_code(),
        200,
        "bootstrap key was hashed without the pepper and cannot authenticate"
    );
    let body: serde_json::Value = resp.json();
    assert_eq!(body["user_name"], "admin");
}

/// Build an `AppState` with a chosen cluster secret (no RBAC users).
fn open_state(dir: &TempDir, cluster_secret: Option<&str>) -> Arc<AppState> {
    let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    let storage = Arc::new(LocalStorage::new(dir.path().join("objects")));
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    let engine =
        Arc::new(Engine::open(catalog, storage, &wal_dir, EngineConfig::default()).unwrap());
    Arc::new(AppState::new(
        engine,
        None,
        ServerConfig {
            tenant: "default_tenant".into(),
            database: "default_database".into(),
            api_key: None,
            cluster_secret: cluster_secret.map(str::to_owned),
            pepper: Vec::new(),
            allow_open_mode: true,
        },
    ))
}

/// The combined single-listener router must not expose cluster internals.
///
/// It used to mount `/internal/*` behind only the public RBAC layer, so any
/// authenticated user could drive Raft or pull collection WAL deltas. Internal
/// routes now additionally require the cluster secret and fail closed when none
/// is configured.
#[tokio::test]
async fn combined_router_requires_cluster_secret_for_internal_routes() {
    let secret_header = || {
        (
            axum::http::HeaderName::from_static("x-rekha-cluster-secret"),
            axum::http::HeaderValue::from_static("s3cr3t"),
        )
    };

    // Secret configured.
    let dir = TempDir::new().unwrap();
    let server = TestServer::new(router(open_state(&dir, Some("s3cr3t"))));

    let resp = server.get("/internal/raft/membership").await;
    assert_eq!(
        resp.status_code(),
        401,
        "internal route reachable without the cluster secret"
    );

    let (k, v) = secret_header();
    let resp = server
        .get("/internal/raft/membership")
        .add_header(k, v)
        .await;
    assert_ne!(
        resp.status_code(),
        401,
        "the configured cluster secret should be accepted"
    );

    // No secret configured: must fail closed rather than allow through.
    let dir2 = TempDir::new().unwrap();
    let server2 = TestServer::new(router(open_state(&dir2, None)));
    let resp = server2.get("/internal/raft/membership").await;
    assert_eq!(
        resp.status_code(),
        401,
        "internal routes must fail closed when no cluster secret is configured"
    );
}

/// Build an `AppState` that has BOTH RBAC users and a cluster secret.
fn rbac_state_with_secret(dir: &TempDir, cluster_secret: Option<&str>) -> (Arc<AppState>, String) {
    let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    catalog.ensure_builtin_roles().unwrap();
    let super_key = generate_api_key();
    let user = rekha_core::auth::UserRecord {
        id: uuid::Uuid::new_v4().to_string(),
        name: "superadmin".into(),
        enabled: true,
        created_at_ms: 0,
    };
    catalog.create_user(&user).expect("create the RBAC user");
    catalog
        .create_api_key(&user.id, &hash_api_key_with_pepper(&super_key, &[]), "test")
        .expect("create the API key");
    let role = catalog.get_role_by_name("superadmin").unwrap().unwrap();
    catalog.assign_role(&user.id, &role.id, &user.id).unwrap();

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
            api_key: None,
            cluster_secret: cluster_secret.map(str::to_owned),
            pepper: Vec::new(),
            allow_open_mode: false,
        },
    ));
    (state, super_key)
}

/// Internal RPCs authenticate with the cluster secret, not an API key.
///
/// The combined router applied the public RBAC middleware to `/internal/*` as
/// well, so once RBAC users existed every peer RPC was rejected unless it also
/// carried a valid API key — and the Raft/WAL clients only send the cluster
/// secret. A cluster therefore could not form after the first user was created.
#[tokio::test]
async fn internal_routes_use_cluster_secret_while_rbac_is_active() {
    let dir = TempDir::new().unwrap();
    let (state, super_key) = rbac_state_with_secret(&dir, Some("s3cr3t"));
    let server = TestServer::new(router(state));

    let secret = (
        axum::http::HeaderName::from_static("x-rekha-cluster-secret"),
        axum::http::HeaderValue::from_static("s3cr3t"),
    );

    // Cluster secret alone: accepted (the handler then reports no raft, which is
    // a 503 — anything but 401/403 means authentication passed).
    let resp = server
        .get("/internal/raft/membership")
        .add_header(secret.0.clone(), secret.1.clone())
        .await;
    assert!(
        resp.status_code() != 401 && resp.status_code() != 403,
        "cluster secret alone must authenticate internal routes, got {}",
        resp.status_code()
    );

    // An API key is not a substitute for the cluster secret.
    let (k, v) = api_key_header(&super_key);
    let resp = server
        .get("/internal/raft/membership")
        .add_header(k, v)
        .await;
    assert_eq!(
        resp.status_code(),
        401,
        "a user API key must not reach internal routes"
    );

    // No credentials at all.
    let resp = server.get("/internal/raft/membership").await;
    assert_eq!(resp.status_code(), 401);

    // Public routes still require RBAC.
    let resp = server.get("/api/v2/auth/whoami").await;
    assert_eq!(
        resp.status_code(),
        401,
        "public routes must still require authentication"
    );
}

/// An empty cluster secret must not authenticate anything.
///
/// `Some("")` used to compare equal to a request that sent no header at all,
/// silently turning internal routes into open endpoints.
#[tokio::test]
async fn empty_cluster_secret_fails_closed() {
    let dir = TempDir::new().unwrap();
    let server = TestServer::new(router(open_state(&dir, Some(""))));

    let resp = server.get("/internal/raft/membership").await;
    assert_eq!(
        resp.status_code(),
        401,
        "an empty cluster secret must not authenticate a header-less request"
    );

    let resp = server
        .get("/internal/raft/membership")
        .add_header(
            axum::http::HeaderName::from_static("x-rekha-cluster-secret"),
            axum::http::HeaderValue::from_static(""),
        )
        .await;
    assert_eq!(
        resp.status_code(),
        401,
        "empty secret must not match itself"
    );
}

/// A collection's *name* must never decide which permission a request needs.
///
/// The classifier used to ask `path.contains("/get")` over the whole path, and
/// the name is part of the path — so `POST .../collections/get/add` was checked
/// as `record.get`. A caller holding only read permission could write, and a
/// collection named `update` let `record.update` authorise a delete.
#[tokio::test]
async fn a_collection_name_cannot_elevate_a_permission() {
    let (_dir, server, super_key) = test_app_rbac();
    let (k, v) = api_key_header(&super_key);

    // A collection literally named `get`, the shape that broke the classifier.
    let resp = server
        .post("/api/v2/tenants/acme/databases/prod/collections")
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "name": "get", "dimension": 2 }))
        .await;
    assert!(
        resp.status_code() == 200 || resp.status_code() == 201,
        "create failed: {}",
        resp.text()
    );

    // A role holding read permission and nothing else, scoped to acme/prod.
    let role: serde_json::Value = server
        .post("/api/v2/auth/roles")
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "name": "reader_only", "description": "read only" }))
        .await
        .json();
    let role_id = role["id"].as_str().unwrap().to_owned();

    server
        .post(&format!("/api/v2/auth/roles/{role_id}/permissions"))
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({
            "action": "record.get", "tenant": "acme", "database": "prod"
        }))
        .await;

    let user: serde_json::Value = server
        .post("/api/v2/auth/users")
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "name": "name_reader" }))
        .await
        .json();
    let user_id = user["id"].as_str().unwrap().to_owned();
    let user_key = user["api_key"].as_str().unwrap().to_owned();

    server
        .post(&format!("/api/v2/auth/users/{user_id}/roles"))
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "role_id": role_id }))
        .await;

    let (uk, uv) = api_key_header(&user_key);

    // Reading is allowed.
    let resp = server
        .post("/api/v2/tenants/acme/databases/prod/collections/get/get")
        .add_header(uk.clone(), uv.clone())
        .json(&serde_json::json!({}))
        .await;
    assert_eq!(
        resp.status_code(),
        200,
        "the read the role permits was denied: {}",
        resp.text()
    );

    // Writing is not, even though the path contains "/get".
    let resp = server
        .post("/api/v2/tenants/acme/databases/prod/collections/get/add")
        .add_header(uk.clone(), uv.clone())
        .json(&serde_json::json!({ "ids": ["a"], "embeddings": [[1.0, 2.0]] }))
        .await;
    assert_eq!(
        resp.status_code(),
        403,
        "a read-only role wrote to a collection named `get`: {}",
        resp.text()
    );
}
