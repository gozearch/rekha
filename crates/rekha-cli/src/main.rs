use clap::{Parser, Subcommand};
use reqwest::Client;
use serde_json::Value;

#[derive(Subcommand)]
enum MembershipAction {
    /// List cluster members
    List,
    /// Add a node as learner
    Add {
        #[arg(long)]
        node_id: u64,
        #[arg(long)]
        addr: String,
    },
    /// Remove a node
    Remove {
        #[arg(long)]
        node_id: u64,
    },
}

#[derive(Parser)]
#[command(name = "rekha", about = "RekhaDB CLI")]
struct Cli {
    /// Base URL of the RekhaDB API server.
    #[arg(long, default_value = "http://localhost:8000", env = "REKHA_API_URL")]
    url: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the API server.
    Serve {
        /// Data directory path.
        #[arg(long, default_value = "rekha-data", env = "REKHA_DATA_DIR")]
        data_dir: String,
        /// Listen host.
        #[arg(long, default_value = "0.0.0.0", env = "REKHA_HOST")]
        host: String,
        /// Listen port.
        #[arg(long, default_value_t = 8000, env = "REKHA_PORT")]
        port: u16,
        /// Internal listener host (for Raft/WAL). Defaults to `host`.
        #[arg(long, env = "REKHA_INTERNAL_HOST")]
        internal_host: Option<String>,
        /// Internal listener port. If set, Raft/WAL routes are served on a
        /// separate listener for network isolation. Otherwise they share the
        /// public listener.
        #[arg(long, env = "REKHA_INTERNAL_PORT")]
        internal_port: Option<u16>,
        /// This node's unique id (must be unique across the cluster).
        #[arg(long, default_value_t = 1, env = "REKHA_NODE_ID")]
        node_id: u64,
        /// Comma-separated list of peer addresses (e.g. "host1:8001,host2:8002").
        /// Empty for single-node mode.
        #[arg(long, default_value = "", env = "REKHA_PEERS")]
        peers: String,
        /// Path to TLS certificate file (PEM).
        #[arg(long, env = "REKHA_TLS_CERT")]
        tls_cert: Option<String>,
        /// Path to TLS private key file (PEM).
        #[arg(long, env = "REKHA_TLS_KEY")]
        tls_key: Option<String>,
    },
    /// Collection operations.
    Collection {
        #[command(subcommand)]
        action: CollectionAction,
    },
    /// Cluster membership management.
    Membership {
        #[command(subcommand)]
        action: MembershipAction,
    },
    /// Add records to a collection.
    Add {
        /// Collection id (UUID).
        collection_id: String,
        /// JSON file with { ids, embeddings, metadatas?, documents? }.
        file: String,
    },
    /// Upsert records in a collection.
    Upsert {
        /// Collection id (UUID).
        collection_id: String,
        /// JSON file with { ids, embeddings, metadatas?, documents? }.
        file: String,
    },
    /// Update record metadata.
    Update {
        /// Collection id (UUID).
        collection_id: String,
        /// JSON file with { ids, metadatas?, documents? }.
        file: String,
    },
    /// Delete records from a collection.
    Delete {
        /// Collection id (UUID).
        collection_id: String,
        /// JSON array of ids: ["id1", "id2"]
        file: String,
    },
    /// Query a collection.
    Query {
        /// Collection id (UUID).
        collection_id: String,
        /// JSON file with { query_embeddings, n_results?, where_filter? }.
        file: String,
    },
    /// Get records from a collection.
    Get {
        /// Collection id (UUID).
        collection_id: String,
        /// JSON file with { ids: ["id1", "id2"] }.
        file: String,
    },
    /// Count records in a collection.
    Count {
        /// Collection id (UUID).
        collection_id: String,
    },
}

#[derive(Subcommand)]
enum CollectionAction {
    /// Create a new collection.
    Create {
        /// Collection name.
        name: String,
        /// Vector dimension.
        dimension: usize,
        /// Distance metric (l2, cosine, ip).
        #[arg(long, default_value = "l2")]
        distance: String,
    },
    /// List all collections.
    List,
    /// Get a collection by id.
    Get {
        /// Collection id (UUID).
        id: String,
    },
    /// Delete a collection.
    Delete {
        /// Collection id (UUID).
        id: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let client = Client::new();

    match cli.command {
        Command::Serve {
            data_dir,
            host,
            port,
            internal_host,
            internal_port,
            node_id,
            peers,
            tls_cert,
            tls_key,
        } => {
            use std::collections::HashMap;
            use std::sync::Arc;

            use rekha_api::{AppState, internal_router, public_router, router};
            use rekha_cluster::{
                ClusterStateMachine, RaftNetworkFactoryImpl, RedbLogStore, WalReplication,
            };
            use rekha_engine::{Engine, EngineConfig};
            use rekha_storage::{LocalStorage, RedbCatalog};

            tracing_subscriber::fmt::init();

            let wal_dir = std::path::PathBuf::from(&data_dir).join("wal");
            std::fs::create_dir_all(&wal_dir)?;

            let catalog = Arc::new(RedbCatalog::open(
                std::path::PathBuf::from(&data_dir).join("catalog.redb"),
            )?);
            let storage = Arc::new(LocalStorage::new(
                std::path::PathBuf::from(&data_dir).join("objects"),
            ));
            let config = EngineConfig::default();
            let engine = Arc::new(Engine::open(catalog, storage, &wal_dir, config)?);

            // Build peer address map.
            let mut peer_map: HashMap<u64, String> = HashMap::new();
            let mut peer_ids: Vec<u64> = Vec::new();
            if !peers.is_empty() {
                for (i, addr) in peers.split(',').enumerate() {
                    let addr = addr.trim().to_string();
                    let peer_id = (i as u64) + 1;
                    if peer_id != node_id {
                        peer_map.insert(peer_id, addr);
                        peer_ids.push(peer_id);
                    }
                }
            }

            // Create Raft components.
            let log_store =
                RedbLogStore::open(std::path::PathBuf::from(&data_dir).join("raft_log.redb"))?;
            let sm = ClusterStateMachine::new();
            let network = RaftNetworkFactoryImpl::new(peer_map.clone());
            let raft_config = Arc::new(openraft::Config {
                heartbeat_interval: 500,
                election_timeout_min: 1500,
                election_timeout_max: 3000,
                replication_lag_threshold: 10000,
                ..Default::default()
            });

            let raft = openraft::Raft::<rekha_cluster::RaftTypeConfig>::new(
                node_id,
                raft_config,
                network,
                log_store,
                sm,
            )
            .await
            .expect("Raft initialization failed");

            // Build full member set (all nodes including self)
            let mut members = std::collections::BTreeSet::new();
            members.insert(node_id);
            for &pid in &peer_ids {
                members.insert(pid);
            }

            // Spawn initialize in background so it doesn't block server startup.
            // initialize() waits for a quorum which may not be available yet.
            let raft_clone = raft.clone();
            tokio::spawn(async move {
                match raft_clone.initialize(members).await {
                    Ok(_) => println!("Node {node_id}: initialized cluster"),
                    Err(e) => {
                        let msg = e.to_string();
                        if msg.contains("AlreadyInitialized") || msg.contains("leader") {
                            println!("Node {node_id}: cluster already initialized");
                        } else {
                            eprintln!("Node {node_id}: Raft init error: {e}");
                        }
                    }
                }
            });

            let api_key = std::env::var("REKHA_API_KEY").ok();
            if api_key.is_none() {
                eprintln!(
                    "WARN: REKHA_API_KEY is not set — API is running without authentication. \
                     Set REKHA_API_KEY to enable token auth."
                );
            }
            let state = Arc::new(AppState {
                engine: engine.clone(),
                tenant: "default_tenant".into(),
                database: "default_database".into(),
                raft: Some(Arc::new(raft)),
                api_key,
            });

            // Start follower replication if we have peers.
            if !peers.is_empty() && node_id != 1 {
                let leader_url = peer_map
                    .get(&1)
                    .cloned()
                    .unwrap_or_else(|| "http://127.0.0.1:8000".into());
                let mut replication =
                    WalReplication::new(leader_url, std::time::Duration::from_secs(1));
                let engine_clone = engine.clone();
                tokio::spawn(async move {
                    loop {
                        if let Ok(collections) =
                            engine_clone.list_collections("default_tenant", "default_database")
                        {
                            for record in &collections {
                                let cid = record.config.id;
                                let from_seq = replication.next_seq(&cid);
                                match replication.fetch_delta_with_retry(&cid, from_seq, 3).await {
                                    Ok(delta) => {
                                        for entry in &delta.records {
                                            if let Ok(op) =
                                                bincode::deserialize::<rekha_core::op::Operation>(
                                                    &entry.payload,
                                                )
                                                && let Err(e) = engine_clone
                                                    .apply_remote_ops(&cid, vec![(entry.seq, op)])
                                            {
                                                eprintln!(
                                                    "Failed to apply remote op for {cid}: {e}"
                                                );
                                            }
                                            replication.mark_applied(cid, entry.seq);
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("WAL replication error for {cid}: {e}");
                                    }
                                }
                            }
                        }
                        tokio::time::sleep(replication.poll_interval()).await;
                    }
                });
            }

            if !peers.is_empty() {
                println!("Peers: {peers}");
            }

            // Initialize metrics exporter if configured.
            if let Ok(metrics_port) = std::env::var("REKHA_METRICS_PORT") {
                let metrics_addr: std::net::SocketAddr =
                    format!("0.0.0.0:{metrics_port}").parse()?;
                metrics_exporter_prometheus::PrometheusBuilder::new()
                    .with_http_listener(metrics_addr)
                    .install()?;
                println!("Metrics available at http://{metrics_addr}/metrics");
            }

            // If --internal-port is set, run separate listeners for public and
            // internal routes. Otherwise use a single combined router.
            if let Some(internal_port) = internal_port {
                let ihost = internal_host.unwrap_or_else(|| host.clone());
                let public_app = public_router(state.clone());
                let internal_app = internal_router(state.clone());
                let public_addr = format!("{host}:{port}");
                let internal_addr = format!("{ihost}:{internal_port}");

                if tls_cert.is_some() && tls_key.is_some() {
                    // TLS path: serve both listeners via axum-server.
                    let cert = tls_cert.unwrap();
                    let key = tls_key.unwrap();
                    let config =
                        axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert, &key)
                            .await?;
                    // Clone config for the second listener (RustlsConfig is not Clone
                    // in older versions — reload from the same files).
                    let config2 =
                        axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert, &key)
                            .await?;
                    let handle = axum_server::Handle::<std::net::SocketAddr>::new();
                    let handle_clone = handle.clone();
                    let handle2 = handle.clone();
                    tokio::spawn(async move {
                        tokio::signal::ctrl_c().await.ok();
                        println!("Shutting down...");
                        handle_clone.graceful_shutdown(None);
                        handle2.graceful_shutdown(None);
                    });
                    println!(
                        "RekhaDB node {node_id} public TLS on {public_addr}, internal TLS on {internal_addr}"
                    );
                    let h1 = handle.clone();
                    let public_fut = axum_server::bind_rustls(public_addr.parse()?, config)
                        .handle(h1)
                        .serve(public_app.into_make_service());
                    let h2 = handle.clone();
                    let internal_fut = axum_server::bind_rustls(internal_addr.parse()?, config2)
                        .handle(h2)
                        .serve(internal_app.into_make_service());
                    tokio::try_join!(public_fut, internal_fut)?;
                } else {
                    println!(
                        "RekhaDB node {node_id} public on {public_addr}, internal on {internal_addr}"
                    );
                    let pub_listener = tokio::net::TcpListener::bind(&public_addr).await?;
                    let int_listener = tokio::net::TcpListener::bind(&internal_addr).await?;
                    let shutdown = async {
                        tokio::signal::ctrl_c().await.ok();
                        println!("Shutting down...");
                    };
                    let pub_serve =
                        axum::serve(pub_listener, public_app).with_graceful_shutdown(shutdown);
                    // Internal listener shares the same shutdown signal via a channel.
                    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
                    tokio::spawn(async move {
                        tokio::signal::ctrl_c().await.ok();
                        let _ = tx.send(());
                    });
                    let int_serve = axum::serve(int_listener, internal_app)
                        .with_graceful_shutdown(async { rx.await.ok(); });
                    tokio::try_join!(pub_serve, int_serve)?;
                }
            } else {
                let app = router(state);
                let addr = format!("{host}:{port}");
                if let (Some(cert), Some(key)) = (&tls_cert, &tls_key) {
                    let config =
                        axum_server::tls_rustls::RustlsConfig::from_pem_file(cert, key).await?;
                    println!("RekhaDB node {node_id} listening on {addr} (TLS)");
                    let handle = axum_server::Handle::<std::net::SocketAddr>::new();
                    let handle_clone = handle.clone();
                    tokio::spawn(async move {
                        tokio::signal::ctrl_c().await.ok();
                        println!("Shutting down...");
                        handle_clone.graceful_shutdown(None);
                    });
                    axum_server::bind_rustls(addr.parse()?, config)
                        .handle(handle)
                        .serve(app.into_make_service())
                        .await?;
                } else {
                    println!("RekhaDB node {node_id} listening on {addr}");
                    let listener = tokio::net::TcpListener::bind(&addr).await?;
                    let shutdown = async {
                        tokio::signal::ctrl_c().await.ok();
                        println!("Shutting down...");
                    };
                    axum::serve(listener, app)
                        .with_graceful_shutdown(shutdown)
                        .await?;
                }
            }
        }

        Command::Collection { action } => match action {
            CollectionAction::Create {
                name,
                dimension,
                distance,
            } => {
                let body = serde_json::json!({
                    "name": name,
                    "dimension": dimension,
                    "distance": distance,
                });
                let resp = client
                    .post(format!("{}/api/v2/collections", cli.url))
                    .json(&body)
                    .send()
                    .await?;
                print_response(resp).await?;
            }
            CollectionAction::List => {
                let resp = client
                    .get(format!("{}/api/v2/collections", cli.url))
                    .send()
                    .await?;
                print_response(resp).await?;
            }
            CollectionAction::Get { id } => {
                let resp = client
                    .get(format!("{}/api/v2/collections/{id}", cli.url))
                    .send()
                    .await?;
                print_response(resp).await?;
            }
            CollectionAction::Delete { id } => {
                let resp = client
                    .delete(format!("{}/api/v2/collections/{id}", cli.url))
                    .send()
                    .await?;
                print_response(resp).await?;
            }
        },

        Command::Add {
            collection_id,
            file,
        } => {
            let body = std::fs::read_to_string(&file)?;
            let body: Value = serde_json::from_str(&body)?;
            let resp = client
                .post(format!(
                    "{}/api/v2/collections/{collection_id}/add",
                    cli.url
                ))
                .json(&body)
                .send()
                .await?;
            print_response(resp).await?;
        }

        Command::Upsert {
            collection_id,
            file,
        } => {
            let body = std::fs::read_to_string(&file)?;
            let body: Value = serde_json::from_str(&body)?;
            let resp = client
                .post(format!(
                    "{}/api/v2/collections/{collection_id}/upsert",
                    cli.url
                ))
                .json(&body)
                .send()
                .await?;
            print_response(resp).await?;
        }

        Command::Update {
            collection_id,
            file,
        } => {
            let body = std::fs::read_to_string(&file)?;
            let body: Value = serde_json::from_str(&body)?;
            let resp = client
                .post(format!(
                    "{}/api/v2/collections/{collection_id}/update",
                    cli.url
                ))
                .json(&body)
                .send()
                .await?;
            print_response(resp).await?;
        }

        Command::Delete {
            collection_id,
            file,
        } => {
            let body = std::fs::read_to_string(&file)?;
            let body: Value = serde_json::from_str(&body)?;
            let resp = client
                .post(format!(
                    "{}/api/v2/collections/{collection_id}/delete",
                    cli.url
                ))
                .json(&body)
                .send()
                .await?;
            print_response(resp).await?;
        }

        Command::Query {
            collection_id,
            file,
        } => {
            let body = std::fs::read_to_string(&file)?;
            let body: Value = serde_json::from_str(&body)?;
            let resp = client
                .post(format!(
                    "{}/api/v2/collections/{collection_id}/query",
                    cli.url
                ))
                .json(&body)
                .send()
                .await?;
            print_response(resp).await?;
        }

        Command::Get {
            collection_id,
            file,
        } => {
            let body = std::fs::read_to_string(&file)?;
            let body: Value = serde_json::from_str(&body)?;
            let resp = client
                .post(format!(
                    "{}/api/v2/collections/{collection_id}/get",
                    cli.url
                ))
                .json(&body)
                .send()
                .await?;
            print_response(resp).await?;
        }

        Command::Count { collection_id } => {
            let resp = client
                .post(format!(
                    "{}/api/v2/collections/{collection_id}/count",
                    cli.url
                ))
                .send()
                .await?;
            print_response(resp).await?;
        }

        Command::Membership { action } => match action {
            MembershipAction::List => {
                let resp = client
                    .get(format!("{}/internal/raft/membership", cli.url))
                    .send()
                    .await?;
                print_response(resp).await?;
            }
            MembershipAction::Add { node_id, addr } => {
                let body = serde_json::json!({ "node_id": node_id, "addr": addr });
                let resp = client
                    .post(format!("{}/internal/raft/add_learner", cli.url))
                    .json(&body)
                    .send()
                    .await?;
                print_response(resp).await?;
            }
            MembershipAction::Remove { node_id } => {
                let body = serde_json::json!({ "node_id": node_id });
                let resp = client
                    .post(format!("{}/internal/raft/remove_member", cli.url))
                    .json(&body)
                    .send()
                    .await?;
                print_response(resp).await?;
            }
        },
    }

    Ok(())
}

async fn print_response(resp: reqwest::Response) -> Result<(), Box<dyn std::error::Error>> {
    let status = resp.status();
    let body = resp.text().await?;
    if !status.is_success() {
        eprintln!("Error ({status}): {body}");
        std::process::exit(1);
    }
    match serde_json::from_str::<Value>(&body) {
        Ok(v) => println!("{}", serde_json::to_string_pretty(&v)?),
        Err(_) => println!("{body}"),
    }
    Ok(())
}
