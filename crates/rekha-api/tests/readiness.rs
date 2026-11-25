//! What `/ready` promises, and the states in which it must refuse.
//!
//! Readiness is the contract a load balancer and a test suite both rely on, and
//! every false "ready" is a request routed into a node that cannot serve it.
//! These tests pin the two local refusal reasons the handler can decide without
//! a cluster: an apply divergence and bootstrap credentials that have not
//! finished applying.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use axum_test::TestServer;
use rekha_api::{AppState, ServerConfig, router};
use rekha_engine::{Engine, EngineConfig};
use rekha_storage::{LocalStorage, RedbCatalog};
use tempfile::TempDir;

fn app_state() -> (TempDir, AppState) {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    let storage = Arc::new(LocalStorage::new(dir.path().join("objects")));
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    let engine =
        Arc::new(Engine::open(catalog, storage, &wal_dir, EngineConfig::default()).unwrap());
    (
        dir,
        AppState::new(
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
        ),
    )
}

async fn ready(state: AppState) -> (u16, serde_json::Value) {
    let server = TestServer::new(router(Arc::new(state)));
    let resp = server.get("/ready").await;
    (resp.status_code().as_u16(), resp.json())
}

#[tokio::test]
async fn a_healthy_node_reports_ready_with_its_durability() {
    let (_dir, state) = app_state();
    let (status, body) = ready(state).await;
    assert_eq!(status, 200);
    assert_eq!(body["ready"], true);
    assert_eq!(body["bootstrap_ready"], true);
    assert_eq!(body["tenant"], "default_tenant");
    assert_eq!(body["database"], "default_database");
    assert!(body["durability"].is_string());
    assert!(
        body["cluster"].is_null(),
        "no raft group in single-node mode"
    );
    assert!(body["shards"].is_null());
}

#[tokio::test]
async fn a_diverged_node_is_not_ready_whatever_else_it_reports() {
    let (_dir, state) = app_state();
    let flag = Arc::new(AtomicBool::new(true));
    let (status, body) = ready(state.with_apply_failure_flag(flag)).await;
    assert_eq!(status, 503);
    assert_eq!(body["ready"], false);
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("could not be applied"),
        "{body}"
    );
}

#[tokio::test]
async fn a_node_whose_bootstrap_credentials_have_not_applied_is_not_ready() {
    let (_dir, state) = app_state();
    let bootstrap = Arc::new(AtomicBool::new(false));
    let flag = Arc::new(AtomicBool::new(false));
    let (status, body) = ready(
        state
            .with_bootstrap_ready_flag(bootstrap)
            .with_apply_failure_flag(flag),
    )
    .await;
    assert_eq!(status, 503);
    assert_eq!(body["ready"], false);
    assert_eq!(body["bootstrap_ready"], false);
    assert!(
        body["error"].is_null(),
        "the node is healthy but still bootstrapping, not diverged"
    );
}
