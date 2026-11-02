//! Tenants and databases are namespaces, so they have to exist.
//!
//! Before this suite, `POST /api/v2/tenants` answered `201` and wrote nothing,
//! `GET /api/v2/tenants/{name}` answered `200` for any name, and
//! `GET .../databases` answered `[]` regardless. A client could not tell an
//! existing namespace from a mistyped one — which is the failure mode that
//! matters, because the wrong answer here is "yes, that tenant exists".

use std::sync::Arc;

use axum_test::TestServer;
use rekha_api::{AppState, ServerConfig, router};
use rekha_engine::{Engine, EngineConfig};
use rekha_storage::{LocalStorage, RedbCatalog};
use tempfile::TempDir;

fn test_app() -> (TempDir, TestServer) {
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
            api_key: None,
            cluster_secret: None,
            pepper: Vec::new(),
            allow_open_mode: true,
        },
    ));
    let server = TestServer::new(router(state));
    (dir, server)
}

async fn create_tenant(server: &TestServer, name: &str) -> u16 {
    server
        .post("/api/v2/tenants")
        .json(&serde_json::json!({ "name": name }))
        .await
        .status_code()
        .as_u16()
}

async fn create_database(server: &TestServer, tenant: &str, name: &str) -> u16 {
    server
        .post(&format!("/api/v2/tenants/{tenant}/databases"))
        .json(&serde_json::json!({ "name": name }))
        .await
        .status_code()
        .as_u16()
}

#[tokio::test]
async fn a_missing_tenant_is_not_reported_as_existing() {
    let (_dir, server) = test_app();
    let resp = server.get("/api/v2/tenants/no_such_tenant").await;
    assert_eq!(
        resp.status_code().as_u16(),
        404,
        "a tenant that was never created must not answer 200: {}",
        resp.text()
    );
}

#[tokio::test]
async fn a_created_tenant_can_be_read_back() {
    let (_dir, server) = test_app();
    assert_eq!(create_tenant(&server, "acme").await, 200);

    let resp = server.get("/api/v2/tenants/acme").await;
    assert_eq!(resp.status_code().as_u16(), 200, "{}", resp.text());
    let body: serde_json::Value = resp.json();
    assert_eq!(body["name"], "acme");
}

#[tokio::test]
async fn creating_the_same_tenant_twice_is_idempotent() {
    let (_dir, server) = test_app();
    assert_eq!(create_tenant(&server, "acme").await, 200);
    assert_eq!(
        create_tenant(&server, "acme").await,
        200,
        "create-then-create must settle on one namespace, not fail"
    );
    assert_eq!(
        server
            .get("/api/v2/tenants/acme")
            .await
            .status_code()
            .as_u16(),
        200
    );
}

#[tokio::test]
async fn an_empty_tenant_name_is_refused() {
    let (_dir, server) = test_app();
    assert_eq!(create_tenant(&server, "  ").await, 400);
}

#[tokio::test]
async fn a_database_under_a_missing_tenant_is_refused() {
    let (_dir, server) = test_app();
    assert_eq!(
        create_database(&server, "no_such_tenant", "prod").await,
        404,
        "creating a database must not silently create its tenant"
    );
}

#[tokio::test]
async fn a_created_database_is_listed_under_its_tenant() {
    let (_dir, server) = test_app();
    assert_eq!(create_tenant(&server, "acme").await, 200);
    assert_eq!(create_database(&server, "acme", "prod").await, 200);

    let got = server.get("/api/v2/tenants/acme/databases/prod").await;
    assert_eq!(got.status_code().as_u16(), 200, "{}", got.text());

    let listed = server.get("/api/v2/tenants/acme/databases").await;
    assert_eq!(listed.status_code().as_u16(), 200, "{}", listed.text());
    let rows: Vec<serde_json::Value> = listed.json();
    let names: Vec<&str> = rows.iter().filter_map(|r| r["name"].as_str()).collect();
    assert_eq!(names, vec!["prod"], "exactly the database that exists");
}

#[tokio::test]
async fn listing_databases_of_a_missing_tenant_is_404() {
    let (_dir, server) = test_app();
    let resp = server.get("/api/v2/tenants/ghost/databases").await;
    assert_eq!(resp.status_code().as_u16(), 404, "{}", resp.text());
}

#[tokio::test]
async fn a_tenant_holding_data_exists_even_without_an_explicit_create() {
    let (_dir, server) = test_app();
    let created = server
        .post("/api/v2/tenants/acme/databases/prod/collections")
        .json(&serde_json::json!({ "name": "docs" }))
        .await;
    assert!(
        created.status_code().is_success(),
        "collection create failed: {}",
        created.text()
    );

    let tenant = server.get("/api/v2/tenants/acme").await;
    assert_eq!(
        tenant.status_code().as_u16(),
        200,
        "a namespace with a collection in it exists; reporting 404 would make \
         the data unreachable through the namespace API"
    );

    let dbs: Vec<serde_json::Value> = server.get("/api/v2/tenants/acme/databases").await.json();
    assert_eq!(dbs.len(), 1);
    assert_eq!(dbs[0]["name"], "prod");
}

#[tokio::test]
async fn collections_stay_inside_their_namespace() {
    let (_dir, server) = test_app();
    let created = server
        .post("/api/v2/tenants/acme/databases/prod/collections")
        .json(&serde_json::json!({ "name": "secret" }))
        .await;
    assert!(created.status_code().is_success(), "{}", created.text());

    let other: Vec<serde_json::Value> = server
        .get("/api/v2/tenants/other/databases/prod/collections")
        .await
        .json();
    assert!(
        other.is_empty(),
        "a collection created in acme/prod must not appear in other/prod: {other:?}"
    );
}
