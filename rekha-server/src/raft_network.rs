use async_trait::async_trait;
use rekha_core::RekhaError;
use rekha_raft::{RaftCommand, RaftLogEntry, RaftPeerNetwork};
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::RwLock;

/// A gRPC-based Raft network that sends AppendEntries and RequestVote to peers.
///
/// Peer addresses are used directly (RaftNode.peers stores addresses like
/// `"node-2:50051"`, not logical node IDs). Connections are cached.
pub struct GrpcRaftNetwork {
    channels: RwLock<HashMap<String, tonic::transport::Channel>>,
}

impl GrpcRaftNetwork {
    pub fn new() -> Self {
        Self {
            channels: RwLock::new(HashMap::new()),
        }
    }

    async fn get_client(
        &self,
        addr: &str,
    ) -> Result<crate::proto::rekha_client::RekhaClient<tonic::transport::Channel>, RekhaError>
    {
        // Check channel cache.
        {
            let cache = self.channels.read().await;
            if let Some(ch) = cache.get(addr) {
                let client = crate::proto::rekha_client::RekhaClient::new(ch.clone());
                return Ok(client);
            }
        }

        // Connect and cache.
        let endpoint = format!("http://{addr}");
        let ch = tonic::transport::Channel::from_shared(endpoint)
            .map_err(|e| RekhaError::Internal {
                detail: format!("invalid peer URI: {e}"),
            })?
            .connect_timeout(Duration::from_secs(5))
            .connect()
            .await
            .map_err(|e| RekhaError::Internal {
                detail: format!("cannot connect to peer {addr}: {e}"),
            })?;

        {
            let mut cache = self.channels.write().await;
            cache.insert(addr.to_string(), ch.clone());
        }

        let client = crate::proto::rekha_client::RekhaClient::new(ch);
        Ok(client)
    }
}

impl Default for GrpcRaftNetwork {
    fn default() -> Self {
        Self::new()
    }
}

fn raft_entry_to_proto(entry: &RaftLogEntry) -> crate::proto::RaftEntry {
    let cmd = match &entry.command {
        RaftCommand::Insert {
            collection_name,
            id,
            vector,
            payload,
        } => Some(crate::proto::raft_command::Cmd::Insert(
            crate::proto::InsertRequest {
                id: *id,
                vector: vector.clone(),
                payload: payload.clone().map(|data| crate::proto::Payload {
                    content_type: "raw".into(),
                    data,
                }),
                collection_name: collection_name.clone(),
            },
        )),
        RaftCommand::Delete {
            collection_name,
            ids,
        } => Some(crate::proto::raft_command::Cmd::Delete(
            crate::proto::DeleteRequest {
                ids: ids.clone(),
                collection_name: collection_name.clone(),
            },
        )),
        RaftCommand::CreateCollection { .. }
        | RaftCommand::DropCollection { .. }
        | RaftCommand::NoOp => None,
    };

    crate::proto::RaftEntry {
        term: entry.term,
        index: entry.index,
        command: cmd.map(|c| crate::proto::RaftCommand { cmd: Some(c) }),
    }
}

#[allow(clippy::too_many_arguments)]
#[async_trait]
impl RaftPeerNetwork for GrpcRaftNetwork {
    async fn append_entries(
        &self,
        peer_addr: &str,
        partition_id: u64,
        leader_term: u64,
        leader_id: &str,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<RaftLogEntry>,
        leader_commit: u64,
    ) -> Result<(bool, u64), RekhaError> {
        let mut client = self.get_client(peer_addr).await?;

        let proto_entries: Vec<crate::proto::RaftEntry> =
            entries.iter().map(raft_entry_to_proto).collect();

        let req = tonic::Request::new(crate::proto::AppendEntriesRequest {
            collection_name: "default".into(),
            partition_id,
            leader_term,
            leader_id: leader_id.to_string(),
            prev_log_index,
            prev_log_term,
            entries: proto_entries,
            leader_commit,
        });

        let resp = client
            .raft_append_entries(req)
            .await
            .map_err(|s| RekhaError::Internal {
                detail: format!("AppendEntries RPC failed: {s}"),
            })?;

        let ack = resp.into_inner();
        Ok((ack.success, ack.current_term))
    }

    async fn request_vote(
        &self,
        peer_addr: &str,
        partition_id: u64,
        term: u64,
        candidate_id: &str,
        last_log_index: u64,
        last_log_term: u64,
    ) -> Result<(bool, u64), RekhaError> {
        let mut client = self.get_client(peer_addr).await?;

        let req = tonic::Request::new(crate::proto::RaftVoteRequest {
            collection_name: "default".into(),
            term,
            candidate_id: candidate_id.to_string(),
            last_log_index,
            last_log_term,
            partition_id,
        });

        let resp = client
            .raft_request_vote(req)
            .await
            .map_err(|s| RekhaError::Internal {
                detail: format!("RequestVote RPC failed: {s}"),
            })?;

        let vote = resp.into_inner();
        Ok((vote.vote_granted, vote.term))
    }
}
