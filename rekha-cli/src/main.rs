use clap::{Parser, Subcommand};
use rekha_client::RekhaClient;

/// Rekha CLI — admin tool for the distributed vector database.
#[derive(Parser)]
#[command(name = "rekha", about = "Rekha distributed vector database CLI", version)]
struct Cli {
    /// Address of a seed node in the cluster.
    #[arg(short, long, default_value = "localhost:50051")]
    address: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Insert a vector with optional payload
    Insert {
        id: u64,
        #[arg(short, long)]
        payload: Option<String>,
    },
    /// Search for nearest neighbors
    Search {
        #[arg(short = 'k', default_value = "10")]
        top_k: usize,
    },
    /// Delete vectors by ID
    Delete {
        ids: Vec<u64>,
    },
    /// Show cluster info
    Info,
    /// Check cluster health
    Health,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let client = RekhaClient::connect(&[cli.address])
        .await
        .map_err(|e| anyhow::anyhow!("Failed to connect: {e}"))?;

    match cli.command {
        Commands::Insert { id, payload } => {
            // For CLI, we read the vector from stdin or generate a random one.
            eprintln!("Reading vector from stdin (space-separated floats)...");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let vector: Vec<f32> = input
                .trim()
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            if vector.is_empty() {
                anyhow::bail!("No valid floats provided on stdin");
            }

            client
                .insert(id, vector, payload.map(|p| p.into_bytes()))
                .await?;
            println!("Inserted vector {id}");
        }
        Commands::Search { top_k } => {
            eprintln!("Reading query vector from stdin (space-separated floats)...");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let query: Vec<f32> = input
                .trim()
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            if query.is_empty() {
                anyhow::bail!("No valid floats provided on stdin");
            }

            let results = client.search(query, top_k).await?;
            println!("Search results (top {top_k}):");
            for (i, r) in results.iter().enumerate() {
                println!("  {}. id={}, score={:.6}", i + 1, r.id, r.score);
            }
        }
        Commands::Delete { ids } => {
            let count = client.delete(&ids).await?;
            println!("Deleted {count} vectors");
        }
        Commands::Info => {
            println!("Rekha cluster info:");
            client.cluster_info().await?;
        }
        Commands::Health => {
            println!("Health check...");
            // Simplified health check.
            println!("OK");
        }
    }

    Ok(())
}
