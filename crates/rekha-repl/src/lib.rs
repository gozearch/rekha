//! RekhaDB interactive REPL — psql-style terminal client.

pub mod client;
pub mod commands;
pub mod display;
pub mod query;
pub mod repl;

use client::RekhaClient;
use repl::Repl;

/// Run the REPL.
pub async fn run(
    url: &str,
    api_key: Option<&str>,
    initial_collection: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = RekhaClient::new(url, api_key);

    match client.heartbeat().await {
        Ok(ns) => println!("Connected to RekhaDB (heartbeat: {ns})"),
        Err(e) => {
            eprintln!("Warning: Could not reach server: {e}");
            eprintln!("Starting in offline mode...");
        }
    }

    let mut repl = Repl::new(client, initial_collection);
    repl.run().await?;
    Ok(())
}
