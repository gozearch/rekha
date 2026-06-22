use rekha_index::RekhaIndex;
use rekha_partition::PartitionManager;
use rekha_storage::RocksVectorStore;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tonic::transport::server::ServerTlsConfig;
use tonic::transport::{Identity, Server};
use tracing::{info, warn};

use crate::config::ServerConfig;
use crate::proto::rekha_server::RekhaServer as RekhaGrpcServer;
use crate::proto::HeartbeatRequest;
use crate::service::RekhaService;
use rekha_coordinator::Coordinator;

pub struct ServerInstance {
    config: ServerConfig,
    coordinator: Arc<Coordinator>,
}

impl ServerInstance {
    pub async fn from_config_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config = ServerConfig::from_file(path)?;
        Self::from_config(config).await
    }

    pub async fn from_config(config: ServerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let _ = tracing_subscriber::fmt().with_target(false).try_init();

        info!(
            "Starting Rekha server: node={}, bind={}",
            config.cluster.node_id, config.cluster.bind_addr
        );

        let store = Arc::new(
            RocksVectorStore::open(&config.cluster.data_dir)
                .map_err(|e| format!("failed to open storage: {e}"))?,
        );
        info!("Storage opened at {}", config.cluster.data_dir);

        let partition_manager = Arc::new(RwLock::new(PartitionManager::new(
            HashMap::new(),
            config.partition.num_dim_groups,
            config.partition.total_dim,
        )));

        let coord_config = rekha_coordinator::CoordinatorConfig {
            node_id: config.cluster.node_id.clone(),
            bind_addr: config.cluster.bind_addr.clone(),
            seed_nodes: config.cluster.seed_nodes.clone(),
            default_write_consistency: config.cluster.default_write_consistency.clone(),
            hinted_handoff_enabled: config.cluster.hinted_handoff_enabled,
            max_hint_window_secs: config.cluster.max_hint_window_secs,
            gc_grace_seconds: config.storage.gc_grace_seconds,
        };
        let coordinator = Arc::new(Coordinator::new(
            coord_config,
            store.clone(),
            partition_manager,
        ));

        let index = RekhaIndex::new()?;
        coordinator.initialize(index).await;

        let server = Self { config, coordinator };
        info!("Server initialized {}/{}", server.config.cluster.node_id, server.config.cluster.bind_addr);
        Ok(server)
    }

    pub async fn with_index(self, index: RekhaIndex) -> Self {
        self.coordinator.initialize(index).await;
        self
    }

    fn spawn_heartbeat_loop(&self) {
        let coordinator = self.coordinator.clone();
        let heartbeat_ms = 100;
        let node_id = self.config.cluster.node_id.clone();
        let seed_nodes = self.config.cluster.seed_nodes.clone();
        let bind_addr = self.config.cluster.bind_addr.clone();

        let my_addr = seed_nodes
            .iter()
            .find(|s| s.starts_with(&node_id))
            .cloned()
            .unwrap_or_else(|| bind_addr.clone());

        tokio::spawn(async move {
            let mut health_tick = 0u64;
            loop {
                tokio::time::sleep(Duration::from_millis(heartbeat_ms)).await;

                for seed in &seed_nodes {
                    if seed.starts_with(&node_id) {
                        continue;
                    }
                    let endpoint = format!("http://{}", seed);
                    match tonic::transport::Channel::from_shared(endpoint) {
                        Ok(ch) => match ch.connect().await {
                            Ok(ch) => {
                                let mut client = crate::proto::rekha_client::RekhaClient::new(ch);
                                let storage_bytes =
                                    coordinator.store().get_storage_estimate().unwrap_or(0);
                                let req = tonic::Request::new(HeartbeatRequest {
                                    node_id: node_id.clone(),
                                    address: my_addr.clone(),
                                    storage_bytes,
                                });
                                match client.heartbeat(req).await {
                                    Ok(resp) => { let _ = resp.into_inner(); }
                                    Err(e) => { warn!("Heartbeat to {} failed: {}", seed, e); }
                                }
                            }
                            Err(e) => { warn!("Cannot connect to seed node {}: {}", seed, e); }
                        },
                        Err(e) => { warn!("Invalid seed URI {}: {}", seed, e); }
                    }
                }

                health_tick += 1;
                if health_tick >= 10 {
                    health_tick = 0;
                    coordinator.check_peer_health().await;
                }
            }
        });
    }

    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        let addr: std::net::SocketAddr = self.config.cluster.bind_addr.parse()?;
        info!("Starting gRPC server on {}", addr);

        let service = RekhaService::new(self.coordinator.clone());

        self.spawn_heartbeat_loop();

        let mut builder = Server::builder();
        if self.config.tls.enabled {
            if let (Some(cert), Some(key)) = (
                self.config.tls.cert_path.as_ref(),
                self.config.tls.key_path.as_ref(),
            ) {
                let cert = tokio::fs::read(cert).await?;
                let key = tokio::fs::read(key).await?;
                let identity = Identity::from_pem(cert, key);
                let tls = ServerTlsConfig::new().identity(identity);
                if let Some(ca) = self.config.tls.ca_cert_path.as_ref() {
                    let ca = tokio::fs::read(ca).await?;
                    let ca = tonic::transport::Certificate::from_pem(ca);
                    let tls = tls.client_ca_root(ca);
                    builder = builder.tls_config(tls)?;
                } else {
                    builder = builder.tls_config(tls)?;
                }
            } else {
                warn!("TLS enabled but cert or key path missing — falling back to plaintext");
            }
        }

        builder
            .add_service(RekhaGrpcServer::new(service))
            .serve(addr)
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(deprecated)]
    fn temp_dir() -> String {
        tempfile::TempDir::new().unwrap().into_path().to_string_lossy().to_string()
    }

    #[tokio::test]
    async fn test_server_initializes() {
        let config = ServerConfig::dev_default("test-node", &temp_dir());
        let server = ServerInstance::from_config(config).await.unwrap();
        assert!(server.coordinator.is_initialized().await);
    }

    #[tokio::test]
    async fn test_with_index() {
        let config = ServerConfig::dev_default("test-node", &temp_dir());
        let server = ServerInstance::from_config(config).await.unwrap();
        let index =
            rekha_index::RekhaIndex::new().unwrap();
        index.create_collection("default", 8, 4, 2).unwrap();
        for i in 0..5 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            index.insert("default", i, 0, &v).unwrap();
        }
        index.flush_buffer("default").unwrap();
        let server = server.with_index(index).await;
        assert!(server.coordinator.is_initialized().await);
    }

    #[tokio::test]
    async fn test_server_config_refs() {
        let config = ServerConfig::dev_default("test-node", &temp_dir());
        let server = ServerInstance::from_config(config).await.unwrap();
        assert_eq!(server.config.cluster.node_id, "test-node");
    }

    #[tokio::test]
    async fn test_from_config_invalid_path() {
        let config = ServerConfig::dev_default("test-node", "/nonexistent_dir_xyz/data");
        let result = ServerInstance::from_config(config).await;
        assert!(result.is_err());
    }
}
