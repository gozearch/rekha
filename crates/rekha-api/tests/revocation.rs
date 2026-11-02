//! Revocation takes effect on the next request, not 30 seconds later.
//!
//! `AppState` carried a permission cache keyed by API-key hash with a 30-second
//! TTL, and a function called `invalidate_perm_cache` that nothing called. A
//! deleted user, a deleted key, or an unassigned role therefore kept working
//! until the entry expired — per node, because the cache was per process and the
//! mutation was a log entry applied on every node.
//!
//! The cache is gone. These tests are the reason it cannot come back without an
//! invalidation path that is correct across nodes.

use std::sync::Arc;

use axum_test::TestServer;
use rekha_api::{AppState, ServerConfig, router};
use rekha_core::auth::{ROLE_SUPERADMIN, generate_api_key, hash_api_key_with_pepper};
use rekha_engine::{Engine, EngineConfig};
use rekha_storage::{Catalog, LocalStorage, RedbCatalog};
use tempfile::TempDir;

fn rbac_app() -> (TempDir, TestServer, String) {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    catalog.ensure_builtin_roles().unwrap();

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
    let role = catalog.get_role_by_name(ROLE_SUPERADMIN).unwrap().unwrap();
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
    (dir, TestServer::new(router(state)), super_key)
}

fn key_header(key: &str) -> (axum::http::HeaderName, axum::http::HeaderValue) {
    (
        axum::http::HeaderName::from_static("x-api-key"),
        axum::http::HeaderValue::from_str(key).unwrap(),
    )
}

/// A user created through the API, with the key it was issued.
async fn create_user(server: &TestServer, super_key: &str, name: &str) -> (String, String) {
    let (k, v) = key_header(super_key);
    let resp = server
        .post("/api/v2/auth/users")
        .add_header(k, v)
        .json(&serde_json::json!({ "name": name }))
        .await;
    assert_eq!(resp.status_code().as_u16(), 200, "{}", resp.text());
    let body: serde_json::Value = resp.json();
    (
        body["id"].as_str().expect("id").to_string(),
        body["api_key"].as_str().expect("key").to_string(),
    )
}

#[tokio::test]
async fn a_deleted_user_is_denied_on_the_very_next_request() {
    let (_dir, server, super_key) = rbac_app();
    let (id, alice_key) = create_user(&server, &super_key, "alice").await;

    let (k, v) = key_header(&alice_key);
    let before = server
        .get("/api/v2/auth/whoami")
        .add_header(k.clone(), v.clone())
        .await;
    assert_eq!(
        before.status_code().as_u16(),
        200,
        "the new key must work before the user is deleted"
    );

    let (sk, sv) = key_header(&super_key);
    let deleted = server
        .delete(&format!("/api/v2/auth/users/{id}"))
        .add_header(sk, sv)
        .await;
    assert_eq!(deleted.status_code().as_u16(), 200, "{}", deleted.text());

    // No sleep: a cache with a TTL would still answer 200 here, which is a
    // deleted user still holding access.
    let after = server.get("/api/v2/auth/whoami").add_header(k, v).await;
    assert_eq!(
        after.status_code().as_u16(),
        401,
        "a deleted user's key must stop working immediately: {}",
        after.text()
    );
}

#[tokio::test]
async fn a_deleted_key_is_denied_on_the_very_next_request() {
    let (_dir, server, super_key) = rbac_app();
    let (id, _) = create_user(&server, &super_key, "bob").await;

    let (sk, sv) = key_header(&super_key);
    let issued = server
        .post(&format!("/api/v2/auth/users/{id}/keys"))
        .add_header(sk, sv)
        .json(&serde_json::json!({ "name": "second" }))
        .await;
    assert_eq!(issued.status_code().as_u16(), 200, "{}", issued.text());
    let issued: serde_json::Value = issued.json();
    let key = issued["api_key"].as_str().expect("key").to_string();
    let hash_hex = issued["key_hash_hex"].as_str().expect("hash").to_string();

    let (k, v) = key_header(&key);
    assert_eq!(
        server
            .get("/api/v2/auth/whoami")
            .add_header(k.clone(), v.clone())
            .await
            .status_code()
            .as_u16(),
        200
    );

    let (sk, sv) = key_header(&super_key);
    let deleted = server
        .delete(&format!("/api/v2/auth/users/{id}/keys/{hash_hex}"))
        .add_header(sk, sv)
        .await;
    assert_eq!(deleted.status_code().as_u16(), 200, "{}", deleted.text());

    assert_eq!(
        server
            .get("/api/v2/auth/whoami")
            .add_header(k, v)
            .await
            .status_code()
            .as_u16(),
        401,
        "a deleted key must stop working immediately"
    );
}
