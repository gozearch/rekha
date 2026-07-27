use std::io::Read;

use clap::{Parser, Subcommand};
use rekha_core::{ConsistencyLevel, IvfConfig, SearchParams};
use rekha_proto::proto;
use tokio::io::AsyncBufReadExt;

#[derive(Parser)]
#[command(name = "rekha", about = "Rekha — distributed vector database")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Server {
        #[arg(long, default_value = "config.yaml")]
        config: String,
    },
    CreateCollection {
        #[arg(short = 'c')]
        collection: String,
        #[arg(long, default_value = "1")]
        rf: usize,
        #[arg(
            long,
            default_value = "{\"dim\":256,\"nlist\":4096,\"nprobe\":32,\"pq_m\":64,\"pq_k\":256}"
        )]
        config: String,
    },
    ListCollections,
    CollectionExists {
        #[arg(short = 'c')]
        collection: String,
    },
    Insert {
        #[arg(short = 'c')]
        collection: String,
        #[arg(short = 'i')]
        id: Option<u64>,
    },
    Search {
        #[arg(short = 'c')]
        collection: String,
        #[arg(short = 'k', default_value = "10")]
        k: u32,
        #[arg(short = 'n', default_value = "16")]
        nprobe: u32,
    },
    Delete {
        ids: Vec<u64>,
    },
    Health,
    Import {
        #[arg(short = 'c')]
        collection: String,
        #[arg(short = 'i')]
        input: String,
    },
    Export {
        #[arg(short = 'c')]
        collection: String,
        #[arg(short = 'o')]
        output: String,
        #[arg(long, default_value = "0")]
        offset: u64,
        #[arg(long, default_value = "1000")]
        limit: u64,
    },
}

fn read_vectors_from_stdin() -> Vec<f32> {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("failed to read stdin");
    input
        .split_whitespace()
        .filter_map(|s| s.parse::<f32>().ok())
        .collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Server { config } => {
            let cfg = rekha_server::ServerConfig::from_file(&config)?;
            let instance = rekha_server::ServerInstance::new(cfg).await?;
            instance.run().await?;
        }
        Command::CreateCollection {
            collection,
            rf: _,
            config,
        } => {
            let ivf_config: IvfConfig = serde_json::from_str(&config)?;
            let address =
                std::env::var("REKHA_ADDRESS").unwrap_or_else(|_| "http://0.0.0.0:50051".into());
            let mut client = rekha_client::Client::connect(&address).await?;
            client.create_collection(&collection, ivf_config).await?;
            println!("collection '{}' created", collection);
        }
        Command::ListCollections => {
            let address =
                std::env::var("REKHA_ADDRESS").unwrap_or_else(|_| "http://0.0.0.0:50051".into());
            let mut client = rekha_client::Client::connect(&address).await?;
            let names = client.list_collections().await?;
            for name in &names {
                println!("{}", name);
            }
        }
        Command::CollectionExists { collection } => {
            let address =
                std::env::var("REKHA_ADDRESS").unwrap_or_else(|_| "http://0.0.0.0:50051".into());
            let mut client = rekha_client::Client::connect(&address).await?;
            let exists = client.collection_exists(&collection).await?;
            println!("{}", exists);
        }
        Command::Insert { collection, id } => {
            let vector = read_vectors_from_stdin();
            let id = id.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64
            });
            let address =
                std::env::var("REKHA_ADDRESS").unwrap_or_else(|_| "http://0.0.0.0:50051".into());
            let mut client = rekha_client::Client::connect(&address).await?;
            client
                .insert(
                    &collection,
                    id,
                    vector,
                    None,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    ConsistencyLevel::One,
                )
                .await?;
            println!("inserted {}", id);
        }
        Command::Search {
            collection,
            k,
            nprobe,
        } => {
            let query = read_vectors_from_stdin();
            let address =
                std::env::var("REKHA_ADDRESS").unwrap_or_else(|_| "http://0.0.0.0:50051".into());
            let mut client = rekha_client::Client::connect(&address).await?;
            let params = SearchParams {
                nprobe,
                k,
                include_payloads: false,
                pre_filter: None,
                local_only: false,
            };
            let results = client.search(&collection, query, k, params).await?;
            for r in &results {
                println!("id={} score={}", r.id, r.score);
            }
        }
        Command::Delete { ids } => {
            let address =
                std::env::var("REKHA_ADDRESS").unwrap_or_else(|_| "http://0.0.0.0:50051".into());
            let mut client = rekha_client::Client::connect(&address).await?;
            let collection = std::env::var("REKHA_COLLECTION").unwrap_or_else(|_| "default".into());
            client
                .delete(
                    &collection,
                    &ids,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    ConsistencyLevel::One,
                )
                .await?;
            println!("deleted {} ids", ids.len());
        }
        Command::Health => {
            let address =
                std::env::var("REKHA_ADDRESS").unwrap_or_else(|_| "http://0.0.0.0:50051".into());
            let mut client = rekha_client::Client::connect(&address).await?;
            let ok = client.health().await?;
            println!("health: {}", if ok { "ok" } else { "unhealthy" });
        }
        Command::Import { collection, input } => {
            let address = std::env::var("REKHA_ADDRESS")
                .unwrap_or_else(|_| "http://0.0.0.0:50051".to_string());
            let mut client = rekha_client::Client::connect(&address).await?;

            let file = tokio::fs::File::open(&input).await?;
            let reader = tokio::io::BufReader::new(file);

            let (batch_tx, batch_rx) = tokio::sync::mpsc::channel::<Vec<proto::InsertRequest>>(4);
            let send_lines_collection = collection.clone();

            tokio::spawn(async move {
                let mut lines = reader.lines();
                let mut batch = Vec::new();
                loop {
                    match lines.next_line().await {
                        Ok(Some(line)) => {
                            if line.trim().is_empty() { continue; }
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&line) {
                                let id = parsed["id"].as_u64().unwrap_or(0);
                                let vector: Vec<f32> = parsed["vector"].as_array()
                                    .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
                                    .unwrap_or_default();
                                let payload_str = parsed["payload"].as_str().map(|s| s.to_string());
                                let ts = parsed["timestamp"].as_u64().unwrap_or(0);
                                batch.push(proto::InsertRequest {
                                    id,
                                    vector,
                                    payload: payload_str.map(|data| proto::Payload {
                                        content_type: "text/plain".into(),
                                        data: data.into_bytes(),
                                    }),
                                    collection_name: send_lines_collection.clone(),
                                    is_replication: false,
                                    timestamp: ts,
                                    consistency: proto::ConsistencyLevel::One as i32,
                                    origin_node_id: "cli".to_string(),
                                });
                            }
                            if batch.len() >= 500
                                && batch_tx.send(std::mem::take(&mut batch)).await.is_err() { return; }
                        }
                        Ok(None) => break,
                        Err(_) => break,
                    }
                }
                if !batch.is_empty() {
                    let _ = batch_tx.send(batch).await;
                }
            });

            let resp = client.import_stream(tokio_stream::wrappers::ReceiverStream::new(batch_rx)).await?;
            println!("imported {} vectors ({} errors)", resp.inserted_count, resp.errors.len());
        }
        Command::Export { collection, output, offset, limit } => {
            let address = std::env::var("REKHA_ADDRESS")
                .unwrap_or_else(|_| "http://0.0.0.0:50051".to_string());
            let mut client = rekha_client::Client::connect(&address).await?;
            let mut stream = client.export_stream(&collection, offset, limit, true, true).await?;

            let file = tokio::fs::File::create(&output).await?;
            let mut writer = tokio::io::BufWriter::new(file);
            use tokio::io::AsyncWriteExt;
            use tokio_stream::StreamExt;

            let mut count = 0u64;
            while let Some(result) = stream.next().await {
                let v = result?;
                let line = format!(
                    r#"{{"id":{},"vector":[{}],"payload":{},"timestamp":{}}}"#,
                    v.id,
                    v.vector.iter().map(|x| format!("{:.6}", x)).collect::<Vec<_>>().join(","),
                    v.payload.as_ref().map(|p| format!("\"{}\"", String::from_utf8_lossy(p)))
                        .unwrap_or_else(|| "null".to_string()),
                    v.timestamp,
                );
                writer.write_all(line.as_bytes()).await?;
                writer.write_all(b"\n").await?;
                count += 1;
            }
            writer.flush().await?;
            println!("exported {} vectors to {}", count, output);
        }
    }

    Ok(())
}
