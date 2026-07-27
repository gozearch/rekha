use std::sync::Arc;
use std::time::Duration;

use std::sync::atomic::{AtomicBool, Ordering};

use rekha_cluster::chord::{ChordNode, hash_to_chord_id};
use rekha_cluster::Membership;
use rekha_coordinator::{Coordinator, PeerPool};
use rekha_core::ConsistencyLevel;
use rekha_proto::proto::rekha_server::RekhaServer;
use rekha_storage::RekhaStore;
use tokio::sync::RwLock;
use tonic::transport::{Certificate, Identity};
use tracing::{info, warn};

use crate::config::ServerConfig;
use crate::service::RekhaService;

pub struct ServerInstance {
    pub config: ServerConfig,
    pub store: Arc<RekhaStore>,
    pub coordinator: Arc<Coordinator>,
    pub chord: Arc<ChordNode>,
    pub shutdown_flag: Arc<AtomicBool>,
}

impl ServerInstance {
    pub async fn new(config: ServerConfig) -> anyhow::Result<Self> {
        let store = Arc::new(RekhaStore::open(&config.data_dir)?);

        let node_id = config.node_id.clone();
        let membership = Arc::new(RwLock::new(Membership::new(
            &node_id,
            config.cluster.heartbeat_timeout_ms,
        )));

        let node_id_num: u64 = farmhash(&node_id);

        let consistency = match config.cluster.default_write_consistency.to_lowercase().as_str() {
            "one" => ConsistencyLevel::One,
            "all" => ConsistencyLevel::All,
            _ => ConsistencyLevel::Quorum,
        };

        let chord_id = hash_to_chord_id(format!("{}:{}", node_id, config.listen).as_bytes());
        let chord = Arc::new(ChordNode::new(chord_id, &config.listen));
        chord.set_successor(&node_id, &config.listen);

        let peer_pool = Arc::new(PeerPool::new());

        let coordinator = Arc::new(Coordinator::new(
            store.clone(),
            membership,
            node_id_num,
            node_id.clone(),
            config.cluster.hinted_handoff_enabled,
            config.cluster.max_hint_window_secs,
            consistency,
            config.cluster.default_rf,
            chord.clone(),
            peer_pool.clone(),
            config.storage.gc_grace_seconds,
        ));

        coordinator.initialize().await?;

        let shutdown_flag = Arc::new(AtomicBool::new(false));

        Ok(ServerInstance {
            config,
            store,
            coordinator,
            chord,
            shutdown_flag,
        })
    }

    pub async fn run(self) -> anyhow::Result<()> {
        if self.config.observability.enable_tracing {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::builder()
                        .with_default_directive("rekha=info".parse().unwrap())
                        .from_env_lossy(),
                )
                .init();
            info!("tracing initialized");
        }

        let addr = self.config.listen.parse()?;
        let service = RekhaService::new(self.coordinator.clone());

        let hb_coordinator = self.coordinator.clone();
        let hb_seeds = self.config.cluster.seed_nodes.clone();
        let hb_interval = self.config.cluster.heartbeat_interval_ms;
        let hb_self_id = self.config.node_id.clone();
        let hb_listen = self.config.listen.clone();
        tokio::spawn(async move {
            heartbeat_loop(hb_coordinator, &hb_seeds, hb_interval, &hb_self_id, &hb_listen).await;
        });

        let gc_coordinator = self.coordinator.clone();
        let gc_interval = self.config.storage.gc_interval_secs;
        tokio::spawn(async move {
            gc_loop(gc_coordinator, gc_interval).await;
        });

        let hr_coordinator = self.coordinator.clone();
        tokio::spawn(async move {
            hint_replay_loop(hr_coordinator).await;
        });

        let chord_stab = self.chord.clone();
        let shutdown_stab = self.shutdown_flag.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(500)).await;
                if shutdown_stab.load(Ordering::Relaxed) { break; }
                // stabilize: ask successor for predecessor, update if needed
                if let Some((succ_id, succ_addr)) = chord_stab.successor() {
                    chord_stab.stabilize_with(&succ_id, &succ_addr);
                }
            }
        });

        let chord_fix = self.chord.clone();
        let shutdown_fix = self.shutdown_flag.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                if shutdown_fix.load(Ordering::Relaxed) { break; }
                chord_fix.fix_next_finger(|_start| {
                    None
                });
            }
        });

        let chord_check = self.chord.clone();
        let shutdown_check = self.shutdown_flag.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                if shutdown_check.load(Ordering::Relaxed) { break; }
                chord_check.check_predecessor(false);
            }
        });

        let reaper_membership = self.coordinator.membership.clone();
        let reaper_timeout = self.config.cluster.heartbeat_timeout_ms;
        let shutdown_rap = self.shutdown_flag.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;
                if shutdown_rap.load(Ordering::Relaxed) { break; }
                let memb = reaper_membership.read().await;
                memb.check_timeouts(reaper_timeout);
            }
        });

        let mut builder = tonic::transport::Server::builder();

        if self.config.tls.enabled {
            if let (Some(cert_path), Some(key_path)) =
                (&self.config.tls.cert_path, &self.config.tls.key_path)
            {
                let cert = tokio::fs::read(cert_path).await?;
                let key = tokio::fs::read(key_path).await?;
                let identity = Identity::from_pem(cert, key);
                let mut tls = tonic::transport::ServerTlsConfig::new().identity(identity);

                if let Some(ca_path) = &self.config.tls.client_ca_cert_path {
                    let ca_cert = tokio::fs::read(ca_path).await?;
                    tls = tls.client_ca_root(Certificate::from_pem(ca_cert));
                    info!("mTLS enabled with client CA verification");
                } else {
                    info!("TLS enabled (no client cert verification)");
                }

                builder = builder.tls_config(tls)?;
            } else {
                warn!("TLS enabled but cert_path or key_path missing, skipping TLS");
            }
        }

        if self.config.observability.enable_metrics {
            let addr = format!("0.0.0.0:{}", self.config.observability.metrics_port);
            if let Ok(_exporter) = metrics_exporter_prometheus::PrometheusBuilder::new()
                .with_http_listener(addr.parse::<std::net::SocketAddr>().unwrap())
                .install()
            {
                info!("Prometheus metrics endpoint on {}", addr);
            } else {
                warn!("metrics already installed");
            }
        }

        info!("Rekha server listening on {}", addr);

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            tokio::signal::ctrl_c().await.ok();
            info!("shutdown signal received, stopping server...");
            let _ = tx.send(());
        });

        builder
            .add_service(RekhaServer::new(service))
            .serve_with_shutdown(addr, async {
                rx.await.ok();
            })
            .await?;

        info!("flushing RocksDB...");
        self.store.flush()?;
        self.shutdown_flag.store(true, Ordering::Relaxed);
        info!("shutdown complete");

        Ok(())
    }
}

async fn heartbeat_loop(
    coordinator: Arc<Coordinator>,
    seed_nodes: &[String],
    interval_ms: u64,
    self_id: &str,
    listen_addr: &str,
) {
    let interval = Duration::from_millis(interval_ms.max(100));
    let mut tick = 0u64;
    loop {
        tokio::time::sleep(interval).await;
        tick += 1;

        for seed in seed_nodes {
            match rekha_client::Client::connect(seed).await {
                Ok(mut client) => {
                    if let Err(e) = client.send_heartbeat(self_id, listen_addr).await {
                        warn!("heartbeat to {} failed: {}", seed, e);
                    }
                }
                Err(e) => {
                    warn!("cannot connect to seed {}: {}", seed, e);
                }
            }
        }

        if tick.is_multiple_of(10) {
            coordinator.membership.write().await.rebuild_ring().await;
        }
    }
}

async fn gc_loop(coordinator: Arc<Coordinator>, interval_secs: u64) {
    let interval = Duration::from_secs(interval_secs.max(60));
    loop {
        tokio::time::sleep(interval).await;
        if let Ok(collections) = coordinator.list_collections().await {
            for name in &collections {
                if let Ok(count) = coordinator.gc_collection(name).await {
                    if count > 0 {
                        info!("GC'd {} tombstones from {}", count, name);
                    }
                }
            }
        }
    }
}

async fn hint_replay_loop(coordinator: Arc<Coordinator>) {
    let interval = Duration::from_secs(300);
    loop {
        tokio::time::sleep(interval).await;
        if let Ok(count) = coordinator.replay_hints() {
            if count > 0 {
                info!("replayed {} hints", count);
            }
        }
    }
}

fn farmhash(data: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}
