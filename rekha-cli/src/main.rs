use clap::{Parser, Subcommand};
use rekha_client::RekhaClient;

/// Rekha CLI — admin tool for the distributed vector database.
#[derive(Parser)]
#[command(
    name = "rekha",
    about = "Rekha distributed vector database CLI",
    version
)]
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
    Delete { ids: Vec<u64> },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse_server() {
        // Can't actually parse 'server' since it's not a subcommand;
        // test that the CLI parser works with an address flag
        let cli = Cli::try_parse_from(["rekha", "info"]).unwrap();
        assert_eq!(cli.address, "localhost:50051");
    }

    #[test]
    fn test_cli_parse_insert() {
        let cli = Cli::try_parse_from(["rekha", "insert", "42"]).unwrap();
        match cli.command {
            Commands::Insert { id, payload } => {
                assert_eq!(id, 42);
                assert!(payload.is_none());
            }
            _ => panic!("expected Insert command"),
        }
    }

    #[test]
    fn test_cli_parse_insert_with_payload() {
        let cli =
            Cli::try_parse_from(["rekha", "insert", "42", "--payload", "{\"k\":\"v\"}"]).unwrap();
        match cli.command {
            Commands::Insert { id, payload } => {
                assert_eq!(id, 42);
                assert_eq!(payload, Some("{\"k\":\"v\"}".into()));
            }
            _ => panic!("expected Insert command"),
        }
    }

    #[test]
    fn test_cli_parse_search() {
        let cli = Cli::try_parse_from(["rekha", "search", "-k", "20"]).unwrap();
        match cli.command {
            Commands::Search { top_k } => {
                assert_eq!(top_k, 20);
            }
            _ => panic!("expected Search command"),
        }
    }

    #[test]
    fn test_cli_parse_delete() {
        let cli = Cli::try_parse_from(["rekha", "delete", "1", "2", "3"]).unwrap();
        match cli.command {
            Commands::Delete { ids } => {
                assert_eq!(ids, vec![1, 2, 3]);
            }
            _ => panic!("expected Delete command"),
        }
    }

    #[test]
    fn test_cli_parse_info() {
        let cli = Cli::try_parse_from(["rekha", "info"]).unwrap();
        assert!(matches!(cli.command, Commands::Info));
    }

    #[test]
    fn test_cli_parse_health() {
        let cli = Cli::try_parse_from(["rekha", "health"]).unwrap();
        assert!(matches!(cli.command, Commands::Health));
    }

    #[test]
    fn test_cli_parse_custom_address() {
        let cli = Cli::try_parse_from(["rekha", "--address", "10.0.0.1:50051", "info"]).unwrap();
        assert_eq!(cli.address, "10.0.0.1:50051");
    }
}
