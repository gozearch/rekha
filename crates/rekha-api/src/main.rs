use std::sync::Arc;

use rekha_api::{AppState, router};
use rekha_engine::{Engine, EngineConfig};
use rekha_storage::{LocalStorage, RedbCatalog};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let data_dir = std::env::var("REKHA_DATA_DIR").unwrap_or_else(|_| "rekha-data".into());
    let host = std::env::var("REKHA_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port = std::env::var("REKHA_PORT").unwrap_or_else(|_| "8000".into());

    let wal_dir = std::path::PathBuf::from(&data_dir).join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();

    let catalog = Arc::new(
        RedbCatalog::open(std::path::PathBuf::from(&data_dir).join("catalog.redb")).unwrap(),
    );
    let storage = Arc::new(LocalStorage::new(
        std::path::PathBuf::from(&data_dir).join("objects"),
    ));
    let config = EngineConfig::default();

    let engine = Arc::new(Engine::open(catalog, storage, &wal_dir, config).unwrap());

    let state = Arc::new(AppState {
        engine,
        tenant: "default_tenant".into(),
        database: "default_database".into(),
        raft: None,
        api_key: std::env::var("REKHA_API_KEY").ok(),
    });

    let app = router(state);
    let addr = format!("{host}:{port}");
    println!("RekhaDB API listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
