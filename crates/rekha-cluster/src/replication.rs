//! Follower-side WAL replication — polls the leader for WAL deltas
//! and applies them locally.

use std::collections::HashMap;
use std::time::Duration;

use crate::cluster::WalDelta;
use uuid::Uuid;

/// Tracks replication state for one follower node.
#[derive(Debug, Clone)]
pub struct WalReplication {
    /// HTTP client for calling the leader's WAL delta endpoint.
    client: reqwest::Client,
    /// Leader's base URL (e.g. "http://127.0.0.1:8000").
    leader_url: String,
    /// Per-collection: highest seq applied from the leader's WAL.
    last_applied: HashMap<Uuid, u64>,
    /// Poll interval.
    poll_interval: Duration,
}

impl WalReplication {
    /// Create a new replication handle.
    pub fn new(leader_url: String, poll_interval: Duration) -> Self {
        Self {
            client: reqwest::Client::new(),
            leader_url,
            last_applied: HashMap::new(),
            poll_interval,
        }
    }

    /// Get the next seq to request from the leader for a collection.
    pub fn next_seq(&self, collection_id: &Uuid) -> u64 {
        self.last_applied.get(collection_id).copied().unwrap_or(0) + 1
    }

    /// Fetch WAL delta from the leader for a collection.
    pub async fn fetch_delta(
        &self,
        collection_id: &Uuid,
        from_seq: u64,
    ) -> Result<WalDelta, String> {
        let url = format!(
            "{}/internal/wal/{collection_id}/delta?from_seq={from_seq}",
            self.leader_url
        );
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("HTTP error: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("HTTP {status}: {body}"));
        }

        resp.json::<WalDelta>()
            .await
            .map_err(|e| format!("JSON decode error: {e}"))
    }

    /// Record that we've applied up to `seq` for a collection.
    pub fn mark_applied(&mut self, collection_id: Uuid, seq: u64) {
        self.last_applied.insert(collection_id, seq);
    }

    /// Get the poll interval.
    pub fn poll_interval(&self) -> Duration {
        self.poll_interval
    }

    /// Fetch delta with retry and exponential backoff.
    pub async fn fetch_delta_with_retry(
        &self,
        collection_id: &Uuid,
        from_seq: u64,
        max_retries: u32,
    ) -> Result<WalDelta, String> {
        let mut last_err = String::new();
        for attempt in 0..max_retries {
            match self.fetch_delta(collection_id, from_seq).await {
                Ok(delta) => return Ok(delta),
                Err(e) => {
                    last_err = e;
                    if attempt < max_retries - 1 {
                        let delay = std::time::Duration::from_millis(100 * 2u64.pow(attempt));
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
        Err(format!("Failed after {max_retries} retries: {last_err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replication_next_seq_defaults_to_1() {
        let repl = WalReplication::new("http://127.0.0.1:8000".into(), Duration::from_secs(1));
        let id = Uuid::new_v4();
        assert_eq!(repl.next_seq(&id), 1);
    }

    #[test]
    fn replication_mark_applied_advances_next_seq() {
        let mut repl = WalReplication::new("http://127.0.0.1:8000".into(), Duration::from_secs(1));
        let id = Uuid::new_v4();
        assert_eq!(repl.next_seq(&id), 1);
        repl.mark_applied(id, 10);
        assert_eq!(repl.next_seq(&id), 11);
    }
}
