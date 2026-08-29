//! HTTP-based Raft network transport using reqwest.

use std::fmt::Display;
use std::sync::Arc;
use std::time::Duration;

use openraft::error::{
    Fatal, NetworkError, RPCError, RaftError, ReplicationClosed, StreamingError, Unreachable,
};
use openraft::network::{Backoff, RPCOption, RaftNetwork, RaftNetworkFactory};
use openraft::raft::{AppendEntriesRequest, AppendEntriesResponse, VoteRequest, VoteResponse};
use serde::{Deserialize, Serialize};

use crate::raft_types::RaftTypeConfig as RC;

/// HTTP-based network client for a single peer.
#[derive(Debug)]
pub struct RaftNetworkImpl {
    addr: String,
    client: reqwest::Client,
}

impl RaftNetworkImpl {
    fn new(_target: u64, addr: String) -> Self {
        Self {
            addr,
            client: reqwest::Client::new(),
        }
    }

    #[allow(clippy::result_large_err)]
    async fn send_rpc<Req, Resp>(
        &self,
        path: &str,
        req: Req,
    ) -> Result<Resp, RPCError<u64, u64, RaftError<u64>>>
    where
        Req: Serialize,
        Resp: for<'de> Deserialize<'de>,
    {
        let url = format!("http://{}/internal/raft/{}", self.addr, path);
        let resp = self
            .client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| RPCError::Unreachable(Unreachable::new(&e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(RPCError::Network(NetworkError::new(&RpcHttpError {
                status: status.as_u16(),
                body,
            })));
        }

        resp.json::<Resp>()
            .await
            .map_err(|e| RPCError::Network(NetworkError::new(&e)))
    }
}

/// Simple error type for HTTP failures.
#[derive(Debug)]
struct RpcHttpError {
    status: u16,
    body: String,
}

impl Display for RpcHttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HTTP {}: {}", self.status, self.body)
    }
}

impl std::error::Error for RpcHttpError {}

impl RaftNetwork<RC> for RaftNetworkImpl {
    async fn append_entries(
        &mut self,
        req: AppendEntriesRequest<RC>,
        _opt: RPCOption,
    ) -> Result<AppendEntriesResponse<u64>, RPCError<u64, u64, RaftError<u64>>> {
        self.send_rpc("append_entries", req).await
    }

    async fn vote(
        &mut self,
        req: VoteRequest<u64>,
        _opt: RPCOption,
    ) -> Result<VoteResponse<u64>, RPCError<u64, u64, RaftError<u64>>> {
        self.send_rpc("vote", req).await
    }

    async fn full_snapshot(
        &mut self,
        vote: openraft::Vote<u64>,
        snapshot: openraft::Snapshot<RC>,
        _cancel: impl std::future::Future<Output = ReplicationClosed> + Send + 'static,
        _opt: RPCOption,
    ) -> Result<openraft::raft::SnapshotResponse<u64>, StreamingError<RC, Fatal<u64>>> {
        let meta = snapshot.meta.clone();
        let data: Vec<u8> = (*snapshot.snapshot).into_inner();
        self.send_rpc::<_, openraft::raft::SnapshotResponse<u64>>(
            "raft/install_snapshot",
            (vote, meta, data),
        )
        .await
        .map_err(|e| match e {
            RPCError::Network(n) => StreamingError::Network(n),
            RPCError::Unreachable(u) => StreamingError::Unreachable(u),
            RPCError::Timeout(t) => StreamingError::Timeout(t),
            RPCError::PayloadTooLarge(p) => StreamingError::Network(NetworkError::new(&p)),
            RPCError::RemoteError(r) => StreamingError::Unreachable(Unreachable::new(&r)),
        })
    }

    fn backoff(&self) -> Backoff {
        Backoff::new(
            (0..).map(|i| {
                Duration::from_millis(500).min(Duration::from_secs(5) * 2u32.pow(i.min(3)))
            }),
        )
    }
}

/// Factory that creates network clients for each peer.
#[derive(Debug, Clone)]
pub struct RaftNetworkFactoryImpl {
    /// Maps node_id → address.
    peers: Arc<std::collections::HashMap<u64, String>>,
}

impl RaftNetworkFactoryImpl {
    pub fn new(peers: std::collections::HashMap<u64, String>) -> Self {
        Self {
            peers: Arc::new(peers),
        }
    }
}

impl RaftNetworkFactory<RC> for RaftNetworkFactoryImpl {
    type Network = RaftNetworkImpl;

    async fn new_client(&mut self, target: u64, _node: &u64) -> Self::Network {
        let addr = self
            .peers
            .get(&target)
            .cloned()
            .unwrap_or_else(|| format!("127.0.0.1:{}", 8000 + target));
        RaftNetworkImpl::new(target, addr)
    }
}

/// In-memory channel-based network for testing.
pub mod channel {
    use std::collections::HashMap;
    use std::fmt;
    use std::sync::Arc;
    use std::time::Duration;

    use openraft::error::{
        Fatal, InstallSnapshotError, RPCError, RaftError, ReplicationClosed, StreamingError,
        Unreachable,
    };
    use openraft::network::{Backoff, RPCOption, RaftNetwork, RaftNetworkFactory};
    use openraft::raft::{
        AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest,
        InstallSnapshotResponse, SnapshotResponse, VoteRequest, VoteResponse,
    };
    use tokio::sync::{Mutex, mpsc, oneshot};

    use crate::raft_types::RaftTypeConfig as RC;

    /// Error returned when the channel to a target node is closed.
    #[derive(Debug)]
    struct ChannelClosed;

    impl fmt::Display for ChannelClosed {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "channel closed")
        }
    }

    impl std::error::Error for ChannelClosed {}

    /// Message variants sent through the channel hub.
    #[derive(Debug)]
    #[allow(clippy::type_complexity)]
    pub enum ChannelMessage {
        AppendEntries {
            req: AppendEntriesRequest<RC>,
            resp_tx: oneshot::Sender<
                Result<AppendEntriesResponse<u64>, RPCError<u64, u64, RaftError<u64>>>,
            >,
        },
        Vote {
            req: VoteRequest<u64>,
            resp_tx: oneshot::Sender<Result<VoteResponse<u64>, RPCError<u64, u64, RaftError<u64>>>>,
        },
        InstallSnapshot {
            req: InstallSnapshotRequest<RC>,
            resp_tx: oneshot::Sender<
                Result<
                    InstallSnapshotResponse<u64>,
                    RPCError<u64, u64, RaftError<u64, InstallSnapshotError>>,
                >,
            >,
        },
        FullSnapshot {
            vote: openraft::Vote<u64>,
            meta: openraft::SnapshotMeta<u64, u64>,
            data: Vec<u8>,
            resp_tx: oneshot::Sender<Result<SnapshotResponse<u64>, StreamingError<RC, Fatal<u64>>>>,
        },
    }

    /// Shared hub that routes messages between in-process nodes.
    #[derive(Clone)]
    pub struct ChannelHub {
        senders: Arc<Mutex<HashMap<u64, mpsc::Sender<ChannelMessage>>>>,
    }

    impl ChannelHub {
        pub fn new() -> Self {
            Self {
                senders: Arc::new(Mutex::new(HashMap::new())),
            }
        }

        pub async fn register(&self, node_id: u64, sender: mpsc::Sender<ChannelMessage>) {
            self.senders.lock().await.insert(node_id, sender);
        }

        async fn send(&self, target: u64, msg: ChannelMessage) {
            if let Some(s) = self.senders.lock().await.get(&target) {
                let _ = s.send(msg).await;
            }
        }
    }

    impl Default for ChannelHub {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Network implementation that routes RPCs through in-memory channels.
    #[derive(Clone)]
    pub struct ChannelNetwork {
        hub: ChannelHub,
        target: u64,
    }

    impl ChannelNetwork {
        pub fn new(hub: ChannelHub, target: u64) -> Self {
            Self { hub, target }
        }
    }

    impl RaftNetwork<RC> for ChannelNetwork {
        async fn append_entries(
            &mut self,
            req: AppendEntriesRequest<RC>,
            _opt: RPCOption,
        ) -> Result<AppendEntriesResponse<u64>, RPCError<u64, u64, RaftError<u64>>> {
            let (tx, rx) = oneshot::channel();
            self.hub
                .send(
                    self.target,
                    ChannelMessage::AppendEntries { req, resp_tx: tx },
                )
                .await;
            rx.await
                .map_err(|_| RPCError::Unreachable(Unreachable::new(&ChannelClosed)))?
        }

        async fn vote(
            &mut self,
            req: VoteRequest<u64>,
            _opt: RPCOption,
        ) -> Result<VoteResponse<u64>, RPCError<u64, u64, RaftError<u64>>> {
            let (tx, rx) = oneshot::channel();
            self.hub
                .send(self.target, ChannelMessage::Vote { req, resp_tx: tx })
                .await;
            rx.await
                .map_err(|_| RPCError::Unreachable(Unreachable::new(&ChannelClosed)))?
        }

        async fn install_snapshot(
            &mut self,
            req: InstallSnapshotRequest<RC>,
            _opt: RPCOption,
        ) -> Result<
            InstallSnapshotResponse<u64>,
            RPCError<u64, u64, RaftError<u64, InstallSnapshotError>>,
        > {
            let (tx, rx) = oneshot::channel();
            self.hub
                .send(
                    self.target,
                    ChannelMessage::InstallSnapshot { req, resp_tx: tx },
                )
                .await;
            rx.await
                .map_err(|_| RPCError::Unreachable(Unreachable::new(&ChannelClosed)))?
        }

        async fn full_snapshot(
            &mut self,
            vote: openraft::Vote<u64>,
            snapshot: openraft::Snapshot<RC>,
            _cancel: impl std::future::Future<Output = ReplicationClosed> + Send + 'static,
            _opt: RPCOption,
        ) -> Result<SnapshotResponse<u64>, StreamingError<RC, Fatal<u64>>> {
            let meta = snapshot.meta.clone();
            let data: Vec<u8> = (*snapshot.snapshot).into_inner();
            let (tx, rx) = oneshot::channel();
            self.hub
                .send(
                    self.target,
                    ChannelMessage::FullSnapshot {
                        vote,
                        meta,
                        data,
                        resp_tx: tx,
                    },
                )
                .await;
            rx.await
                .map_err(|_| StreamingError::Unreachable(Unreachable::new(&ChannelClosed)))?
        }

        fn backoff(&self) -> Backoff {
            Backoff::new((0..).map(|i| {
                Duration::from_millis(10).min(Duration::from_millis(50) * 2u32.pow(i.min(3)))
            }))
        }
    }

    /// Factory that creates `ChannelNetwork` instances for each target node.
    #[derive(Clone)]
    pub struct ChannelNetworkFactory {
        hub: ChannelHub,
    }

    impl ChannelNetworkFactory {
        pub fn new(hub: ChannelHub) -> Self {
            Self { hub }
        }
    }

    impl RaftNetworkFactory<RC> for ChannelNetworkFactory {
        type Network = ChannelNetwork;

        async fn new_client(&mut self, target: u64, _node: &u64) -> Self::Network {
            ChannelNetwork::new(self.hub.clone(), target)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpc_http_error_display() {
        let e = RpcHttpError {
            status: 500,
            body: "internal error".into(),
        };
        assert_eq!(format!("{e}"), "HTTP 500: internal error");
    }
}
