//! The RBAC administration surface: users, keys, roles, permissions, and the
//! assignments that tie them together.
//!
//! These endpoints are the only way an operator creates the credentials every
//! other request authenticates with, so each one is asserted for both the happy
//! path and the refusal a caller can actually reach: a missing user, a built-in
//! role, a malformed key hash, or a caller without the permission.

use std::sync::Arc;

use axum_test::TestServer;
use rekha_api::{AppState, ServerConfig, router};
use rekha_core::auth::{generate_api_key, hash_api_key_with_pepper};
use rekha_engine::{Engine, EngineConfig};
use rekha_storage::{Catalog, LocalStorage, RedbCatalog};
use tempfile::TempDir;

fn test_app_rbac() -> (TempDir, TestServer, String) {
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

fn key_header(key: &str) -> (axum::http::HeaderName, axum::http::HeaderValue) {
    (
        axum::http::HeaderName::from_static("x-api-key"),
        axum::http::HeaderValue::from_str(key).unwrap(),
    )
}

async fn create_user(server: &TestServer, key: &str, name: &str) -> serde_json::Value {
    let (k, v) = key_header(key);
    let resp = server
        .post("/api/v2/auth/users")
        .add_header(k, v)
        .json(&serde_json::json!({ "name": name }))
        .await;
    assert_eq!(resp.status_code(), 200, "create user: {}", resp.text());
    resp.json()
}

async fn create_role(server: &TestServer, key: &str, name: &str) -> serde_json::Value {
    let (k, v) = key_header(key);
    let resp = server
        .post("/api/v2/auth/roles")
        .add_header(k, v)
        .json(&serde_json::json!({ "name": name, "description": "test role" }))
        .await;
    assert_eq!(resp.status_code(), 200, "create role: {}", resp.text());
    resp.json()
}

async fn builtin_role_id(server: &TestServer, key: &str, name: &str) -> String {
    let (k, v) = key_header(key);
    let roles: Vec<serde_json::Value> = server
        .get("/api/v2/auth/roles")
        .add_header(k, v)
        .await
        .json();
    roles
        .into_iter()
        .find(|r| r["name"] == name)
        .unwrap_or_else(|| panic!("built-in role {name} missing"))["id"]
        .as_str()
        .unwrap()
        .to_owned()
}

// ---------------------------------------------------------------------------
// Users
// ---------------------------------------------------------------------------

#[tokio::test]
async fn users_can_be_listed_read_and_deleted() {
    let (_dir, server, key) = test_app_rbac();
    let alice = create_user(&server, &key, "alice").await;
    let id = alice["id"].as_str().unwrap();

    let (k, v) = key_header(&key);
    let users: Vec<serde_json::Value> = server
        .get("/api/v2/auth/users")
        .add_header(k.clone(), v.clone())
        .await
        .json();
    assert!(users.iter().any(|u| u["id"] == id));

    let fetched: serde_json::Value = server
        .get(&format!("/api/v2/auth/users/{id}"))
        .add_header(k.clone(), v.clone())
        .await
        .json();
    assert_eq!(fetched["name"], "alice");
    assert_eq!(fetched["enabled"], true);

    let resp = server
        .delete(&format!("/api/v2/auth/users/{id}"))
        .add_header(k.clone(), v.clone())
        .await;
    assert_eq!(resp.status_code(), 200);

    let resp = server
        .get(&format!("/api/v2/auth/users/{id}"))
        .add_header(k, v)
        .await;
    assert_eq!(resp.status_code(), 404);
}

#[tokio::test]
async fn a_missing_user_is_404_not_an_empty_record() {
    let (_dir, server, key) = test_app_rbac();
    let (k, v) = key_header(&key);
    let resp = server
        .get("/api/v2/auth/users/does-not-exist")
        .add_header(k, v)
        .await;
    assert_eq!(resp.status_code(), 404);
    assert_eq!(resp.json::<serde_json::Value>()["error"], "user not found");
}

#[tokio::test]
async fn a_user_without_permissions_is_forbidden_not_unauthenticated() {
    let (_dir, server, key) = test_app_rbac();
    let alice = create_user(&server, &key, "alice").await;
    let alice_key = alice["api_key"].as_str().unwrap();

    let (k, v) = key_header(alice_key);
    let resp = server.get("/api/v2/auth/users").add_header(k, v).await;
    // The auth middleware answers with a bare 403: it decides before the
    // handler runs, so there is no handler-produced body to read.
    assert_eq!(resp.status_code(), 403);
}

// ---------------------------------------------------------------------------
// API keys
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_key_can_be_created_listed_and_revoked() {
    let (_dir, server, key) = test_app_rbac();
    let alice = create_user(&server, &key, "alice").await;
    let id = alice["id"].as_str().unwrap();
    let (k, v) = key_header(&key);

    let created: serde_json::Value = server
        .post(&format!("/api/v2/auth/users/{id}/keys"))
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "name": "ci" }))
        .await
        .json();
    assert!(
        created["api_key"]
            .as_str()
            .unwrap()
            .starts_with("rekha_sk_")
    );
    let hash_hex = created["key_hash_hex"].as_str().unwrap();
    assert_eq!(hash_hex.len(), 64);

    let keys: Vec<serde_json::Value> = server
        .get(&format!("/api/v2/auth/users/{id}/keys"))
        .add_header(k.clone(), v.clone())
        .await
        .json();
    assert!(keys.iter().any(|k| k["name"] == "ci"));

    let resp = server
        .delete(&format!("/api/v2/auth/users/{id}/keys/{hash_hex}"))
        .add_header(k.clone(), v.clone())
        .await;
    assert_eq!(resp.status_code(), 200);

    let keys: Vec<serde_json::Value> = server
        .get(&format!("/api/v2/auth/users/{id}/keys"))
        .add_header(k, v)
        .await
        .json();
    assert!(!keys.iter().any(|k| k["name"] == "ci"));
}

#[tokio::test]
async fn a_key_name_must_not_be_empty_and_the_user_must_exist() {
    let (_dir, server, key) = test_app_rbac();
    let alice = create_user(&server, &key, "alice").await;
    let id = alice["id"].as_str().unwrap();
    let (k, v) = key_header(&key);

    let resp = server
        .post(&format!("/api/v2/auth/users/{id}/keys"))
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "name": "  " }))
        .await;
    assert_eq!(resp.status_code(), 400);

    let resp = server
        .post("/api/v2/auth/users/missing/keys")
        .add_header(k, v)
        .json(&serde_json::json!({ "name": "ci" }))
        .await;
    assert_eq!(resp.status_code(), 404);
}

#[tokio::test]
async fn a_malformed_key_hash_is_rejected_before_it_reaches_the_catalog() {
    let (_dir, server, key) = test_app_rbac();
    let alice = create_user(&server, &key, "alice").await;
    let id = alice["id"].as_str().unwrap();
    let (k, v) = key_header(&key);

    let short = server
        .delete(&format!("/api/v2/auth/users/{id}/keys/abc123"))
        .add_header(k.clone(), v.clone())
        .await;
    assert_eq!(short.status_code(), 400);

    let non_hex = "z".repeat(64);
    let resp = server
        .delete(&format!("/api/v2/auth/users/{id}/keys/{non_hex}"))
        .add_header(k, v)
        .await;
    assert_eq!(resp.status_code(), 400);
    assert_eq!(resp.json::<serde_json::Value>()["error"], "invalid hex");
}

// ---------------------------------------------------------------------------
// Roles
// ---------------------------------------------------------------------------

#[tokio::test]
async fn custom_roles_can_be_created_listed_and_deleted() {
    let (_dir, server, key) = test_app_rbac();
    let role = create_role(&server, &key, "analyst").await;
    let id = role["id"].as_str().unwrap();
    assert_eq!(role["built_in"], false);
    let (k, v) = key_header(&key);

    let roles: Vec<serde_json::Value> = server
        .get("/api/v2/auth/roles")
        .add_header(k.clone(), v.clone())
        .await
        .json();
    assert!(roles.iter().any(|r| r["id"] == id));

    let resp = server
        .delete(&format!("/api/v2/auth/roles/{id}"))
        .add_header(k, v)
        .await;
    assert_eq!(resp.status_code(), 200);
}

#[tokio::test]
async fn role_names_are_unique_and_must_not_be_empty() {
    let (_dir, server, key) = test_app_rbac();
    let (k, v) = key_header(&key);

    let empty = server
        .post("/api/v2/auth/roles")
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "name": " " }))
        .await;
    assert_eq!(empty.status_code(), 400);

    create_role(&server, &key, "analyst").await;
    let duplicate = server
        .post("/api/v2/auth/roles")
        .add_header(k, v)
        .json(&serde_json::json!({ "name": "analyst" }))
        .await;
    assert_eq!(duplicate.status_code(), 409);
}

#[tokio::test]
async fn a_builtin_role_cannot_be_deleted_and_an_unknown_one_is_404() {
    let (_dir, server, key) = test_app_rbac();
    let reader = builtin_role_id(&server, &key, "reader").await;
    let (k, v) = key_header(&key);

    let builtin = server
        .delete(&format!("/api/v2/auth/roles/{reader}"))
        .add_header(k.clone(), v.clone())
        .await;
    assert_eq!(builtin.status_code(), 409);

    let missing = server
        .delete("/api/v2/auth/roles/does-not-exist")
        .add_header(k, v)
        .await;
    assert_eq!(missing.status_code(), 404);
}

// ---------------------------------------------------------------------------
// Permissions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn permissions_can_be_added_listed_and_removed() {
    let (_dir, server, key) = test_app_rbac();
    let role = create_role(&server, &key, "analyst").await;
    let role_id = role["id"].as_str().unwrap();
    let (k, v) = key_header(&key);

    let perm: serde_json::Value = server
        .post(&format!("/api/v2/auth/roles/{role_id}/permissions"))
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({
            "action": "collection.get",
            "tenant": "default_tenant",
            "database": "default_database"
        }))
        .await
        .json();
    let perm_id = perm["id"].as_str().unwrap();

    let listed: Vec<serde_json::Value> = server
        .get(&format!("/api/v2/auth/roles/{role_id}/permissions"))
        .add_header(k.clone(), v.clone())
        .await
        .json();
    assert!(listed.iter().any(|p| p["id"] == perm_id));

    let resp = server
        .delete(&format!(
            "/api/v2/auth/roles/{role_id}/permissions/{perm_id}"
        ))
        .add_header(k.clone(), v.clone())
        .await;
    assert_eq!(resp.status_code(), 200);

    let listed: Vec<serde_json::Value> = server
        .get(&format!("/api/v2/auth/roles/{role_id}/permissions"))
        .add_header(k, v)
        .await
        .json();
    assert!(listed.is_empty());
}

#[tokio::test]
async fn a_permission_needs_an_action_and_an_existing_role() {
    let (_dir, server, key) = test_app_rbac();
    let role = create_role(&server, &key, "analyst").await;
    let role_id = role["id"].as_str().unwrap();
    let (k, v) = key_header(&key);

    let empty = server
        .post(&format!("/api/v2/auth/roles/{role_id}/permissions"))
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "action": "" }))
        .await;
    assert_eq!(empty.status_code(), 400);

    let missing = server
        .post("/api/v2/auth/roles/does-not-exist/permissions")
        .add_header(k, v)
        .json(&serde_json::json!({ "action": "collection.get" }))
        .await;
    assert_eq!(missing.status_code(), 404);
}

// ---------------------------------------------------------------------------
// Assignments
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_role_can_be_assigned_listed_and_unassigned() {
    let (_dir, server, key) = test_app_rbac();
    let alice = create_user(&server, &key, "alice").await;
    let user_id = alice["id"].as_str().unwrap();
    let role = create_role(&server, &key, "analyst").await;
    let role_id = role["id"].as_str().unwrap();
    let (k, v) = key_header(&key);

    let resp = server
        .post(&format!("/api/v2/auth/users/{user_id}/roles"))
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "role_id": role_id }))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());

    let roles: Vec<serde_json::Value> = server
        .get(&format!("/api/v2/auth/users/{user_id}/roles"))
        .add_header(k.clone(), v.clone())
        .await
        .json();
    assert!(roles.iter().any(|r| r["id"] == role_id));

    let resp = server
        .delete(&format!("/api/v2/auth/users/{user_id}/roles/{role_id}"))
        .add_header(k.clone(), v.clone())
        .await;
    assert_eq!(resp.status_code(), 200);

    let roles: Vec<serde_json::Value> = server
        .get(&format!("/api/v2/auth/users/{user_id}/roles"))
        .add_header(k, v)
        .await
        .json();
    assert!(roles.is_empty());
}

#[tokio::test]
async fn assigning_a_role_requires_both_sides_to_exist() {
    let (_dir, server, key) = test_app_rbac();
    let alice = create_user(&server, &key, "alice").await;
    let user_id = alice["id"].as_str().unwrap();
    let role = create_role(&server, &key, "analyst").await;
    let role_id = role["id"].as_str().unwrap();
    let (k, v) = key_header(&key);

    let missing_user = server
        .post("/api/v2/auth/users/missing/roles")
        .add_header(k.clone(), v.clone())
        .json(&serde_json::json!({ "role_id": role_id }))
        .await;
    assert_eq!(missing_user.status_code(), 404);

    let missing_role = server
        .post(&format!("/api/v2/auth/users/{user_id}/roles"))
        .add_header(k, v)
        .json(&serde_json::json!({ "role_id": "does-not-exist" }))
        .await;
    assert_eq!(missing_role.status_code(), 404);
}
