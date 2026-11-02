//! RekhaDB interactive REPL — psql-style terminal client.

use clap::Parser;

#[derive(Parser)]
#[command(name = "rekha-repl", about = "RekhaDB interactive REPL")]
struct Cli {
    /// RekhaDB server URL.
    #[arg(long, default_value = "http://localhost:8000", env = "REKHA_API_URL")]
    url: String,

    /// API key for authentication.
    #[arg(long, env = "REKHA_API_KEY")]
    api_key: Option<String>,

    /// Initial collection to use.
    #[arg(long)]
    collection: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    rekha_repl::run(&cli.url, cli.api_key.as_deref(), cli.collection.as_deref()).await
}
