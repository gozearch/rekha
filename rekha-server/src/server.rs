use rekha_index::RekhaIndex;
use rekha_partition::PartitionManager;
use rekha_storage::RocksVectorStore;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::server::ServerTlsConfig;
use tonic::transport::{Identity, Server};
use tracing::info;

use crate::config::ServerConfig;
use crate::coordinator::Coordinator;
use crate::proto::rekha_server::RekhaServer as RekhaGrpcServer;
use crate::service::RekhaService;

/// The Rekha distributed vector database server.
///
/// Orchestrates:
/// - gRPC service for client RPCs
/// - Coordinator for query routing
/// - Local index for vector search
/// - Raft consensus for replication
/// - Partition management for topology
pub struct ServerInstance {
    config: ServerConfig,
    coordinator: Arc<Coordinator>,
}

impl ServerInstance {
    /// Create and initialize a new Rekha server from a configuration file.
    pub async fn from_config_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config = ServerConfig::from_file(path)?;
        Self::from_config(config).await
    }

    /// Create and initialize a new Rekha server from a config struct.
    pub async fn from_config(config: ServerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize tracing/logging (ignore errors if already initialized).
        let _ = tracing_subscriber::fmt().with_target(false).try_init();

        info!(
            "Starting Rekha server: node={}, bind={}",
            config.cluster.node_id, config.cluster.bind_addr
        );

        // Open storage.
        let store = Arc::new(
            RocksVectorStore::open(&config.cluster.data_dir)
                .map_err(|e| format!("failed to open storage: {e}"))?,
        );
        info!("Storage opened at {}", config.cluster.data_dir);

        // Create partition manager.
        let partition_manager = Arc::new(RwLock::new(PartitionManager::new(
            HashMap::new(),
            config.partition.num_dim_groups,
            config.partition.dim_group_size * config.partition.num_dim_groups as usize,
        )));

        // Create coordinator.
        let coordinator = Arc::new(Coordinator::new(config.clone(), store, partition_manager));

        Ok(Self {
            config,
            coordinator,
        })
    }

    /// Set the index on the coordinator.
    pub async fn with_index(self, index: RekhaIndex) -> Self {
        self.coordinator.initialize(index).await;
        self
    }

    /// Run the server (blocking).
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let addr = self.config.cluster.bind_addr.parse()?;
        let service = RekhaService::new(self.coordinator.clone());

        if self.config.tls.enabled {
            let cert_path = self
                .config
                .tls
                .cert_path
                .as_ref()
                .ok_or_else(|| "TLS enabled but cert_path not configured".to_string())?;
            let key_path = self
                .config
                .tls
                .key_path
                .as_ref()
                .ok_or_else(|| "TLS enabled but key_path not configured".to_string())?;

            let cert = std::fs::read(cert_path).map_err(|e| format!("failed to read cert: {e}"))?;
            let key = std::fs::read(key_path).map_err(|e| format!("failed to read key: {e}"))?;
            let identity = Identity::from_pem(cert, key);

            let mut tls_config = ServerTlsConfig::new().identity(identity);
            if let Some(ca_path) = &self.config.tls.ca_cert_path {
                let ca_cert =
                    std::fs::read(ca_path).map_err(|e| format!("failed to read CA cert: {e}"))?;
                tls_config =
                    tls_config.client_ca_root(tonic::transport::Certificate::from_pem(ca_cert));
            }

            info!("gRPC server listening on {addr} with TLS");

            Server::builder()
                .tls_config(tls_config)?
                .add_service(RekhaGrpcServer::new(service))
                .serve(addr)
                .await?;
        } else {
            info!("gRPC server listening on {addr} (plaintext)");

            Server::builder()
                .add_service(RekhaGrpcServer::new(service))
                .serve(addr)
                .await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> String {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        format!("/tmp/rekha_server_test_{}", n)
    }

    #[tokio::test]
    async fn test_server_from_config() {
        let config = ServerConfig::dev_default("test-node", &temp_dir());
        let result = ServerInstance::from_config(config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_server_from_config_file_not_found() {
        let result = ServerInstance::from_config_file("/nonexistent/config.yaml").await;
        assert!(result.is_err());
    }
}
