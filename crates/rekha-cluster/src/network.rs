//! HTTP-based Raft network transport using reqwest.

use std::fmt::Display;
use std::time::Duration;

use openraft::error::{
    Fatal, NetworkError, RPCError, RaftError, ReplicationClosed, StreamingError, Unreachable,
};
use openraft::network::{Backoff, RPCOption, RaftNetwork, RaftNetworkFactory};
use openraft::raft::{AppendEntriesRequest, AppendEntriesResponse, VoteRequest, VoteResponse};
use serde::{Deserialize, Serialize};

use crate::raft_types::RaftTypeConfig as RC;

/// Consent group id sent in the RPC path.
///
/// `0` is the metadata group; `1..=shard_count` are shard groups. The id has to
/// travel with every message because one process runs one `Raft` instance per
/// group it votes on, and a vote or append-entries addressed to the wrong group
/// would corrupt two logs instead of one.
pub type GroupId = u32;

/// The metadata group's id. Shard groups are `1..=shard_count`.
pub const METADATA_GROUP: GroupId = 0;

/// HTTP-based network client for a single peer and a single consensus group.
#[derive(Debug)]
pub struct RaftNetworkImpl {
    target: u64,
    /// `None` when this node was not configured with an address for the peer.
    ///
    /// This used to fall back to `127.0.0.1:{8000 + target}`, which turned a
    /// missing peer into a connection to whatever happened to be listening on a
    /// local port. With several groups per process, a message addressed to the
    /// wrong group is already hard enough to diagnose.
    addr: Option<String>,
    group: GroupId,
    client: reqwest::Client,
    /// Shared secret required by the peer's `/internal/*` routes.
    cluster_secret: Option<String>,
}

impl RaftNetworkImpl {
    fn new(
        target: u64,
        addr: Option<String>,
        group: GroupId,
        cluster_secret: Option<String>,
    ) -> Self {
        Self {
            target,
            addr: addr.map(|a| normalize_peer_url(&a)),
            group,
            client: reqwest::Client::new(),
            cluster_secret,
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
        let Some(addr) = self.addr.as_deref() else {
            return Err(RPCError::Unreachable(Unreachable::new(&UnknownPeer {
                target: self.target,
            })));
        };
        let url = format!("{}/internal/raft/{}/{}", addr, self.group, path);
        let mut request = self.client.post(&url).json(&req);
        if let Some(ref secret) = self.cluster_secret {
            request = request.header("x-rekha-cluster-secret", secret);
        }
        let resp = request
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

/// A peer with no configured address.
#[derive(Debug)]
struct UnknownPeer {
    target: u64,
}

impl Display for UnknownPeer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "no address is configured for node {}; every node must be listed in --peers",
            self.target
        )
    }
}

impl std::error::Error for UnknownPeer {}

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

/// The RPC did not finish inside the deadline openraft handed us.
#[derive(Debug)]
struct TimedOut(Duration);

impl Display for TimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "raft rpc exceeded its {:?} deadline", self.0)
    }
}

impl std::error::Error for TimedOut {}

/// Run one RPC with the deadline `RPCOption` asks for.
///
/// openraft's documentation asks the implementation to cancel at `soft_ttl()`
/// so that its own `hard_ttl()` cutoff is not the first thing that happens. This
/// transport ignored the option and built its client with no timeout at all, so
/// a peer that accepted a connection and then never answered held that
/// replication stream open until the socket itself failed: the leader never
/// retried, and nothing reported the follower as behind.
async fn with_deadline<T, F>(
    opt: &RPCOption,
    fut: F,
) -> Result<T, RPCError<u64, u64, RaftError<u64>>>
where
    F: std::future::Future<Output = Result<T, RPCError<u64, u64, RaftError<u64>>>>,
{
    let deadline = opt.soft_ttl();
    match tokio::time::timeout(deadline, fut).await {
        Ok(result) => result,
        Err(_) => Err(RPCError::Unreachable(Unreachable::new(&TimedOut(deadline)))),
    }
}

impl RaftNetwork<RC> for RaftNetworkImpl {
    async fn append_entries(
        &mut self,
        req: AppendEntriesRequest<RC>,
        opt: RPCOption,
    ) -> Result<AppendEntriesResponse<u64>, RPCError<u64, u64, RaftError<u64>>> {
        with_deadline(&opt, self.send_rpc("append_entries", req)).await
    }

    async fn vote(
        &mut self,
        req: VoteRequest<u64>,
        opt: RPCOption,
    ) -> Result<VoteResponse<u64>, RPCError<u64, u64, RaftError<u64>>> {
        with_deadline(&opt, self.send_rpc("vote", req)).await
    }

    async fn full_snapshot(
        &mut self,
        vote: openraft::Vote<u64>,
        snapshot: openraft::Snapshot<RC>,
        _cancel: impl std::future::Future<Output = ReplicationClosed> + Send + 'static,
        opt: RPCOption,
    ) -> Result<openraft::raft::SnapshotResponse<u64>, StreamingError<RC, Fatal<u64>>> {
        let meta = snapshot.meta.clone();
        let data: Vec<u8> = (*snapshot.snapshot).into_inner();
        // A snapshot has its own, longer deadline; the option is the caller's
        // bound, not this transport's guess.
        let deadline = opt.hard_ttl();
        let transfer = tokio::time::timeout(
            deadline,
            self.send_rpc::<_, openraft::raft::SnapshotResponse<u64>>(
                // `send_rpc` already prefixes `/internal/raft/`; passing
                // `raft/install_snapshot` produced `/internal/raft/raft/install_snapshot`,
                // which never matches a route, so snapshot installs always 404'd.
                "install_snapshot",
                (vote, meta, data),
            ),
        );
        match transfer.await {
            // The deadline elapsed: report it the way openraft reports an
            // unreachable peer, which is retryable.
            Err(_) => Err(StreamingError::Unreachable(Unreachable::new(&TimedOut(
                deadline,
            )))),
            Ok(Ok(response)) => Ok(response),
            Ok(Err(e)) => Err(match e {
                RPCError::Network(n) => StreamingError::Network(n),
                RPCError::Unreachable(u) => StreamingError::Unreachable(u),
                RPCError::Timeout(t) => StreamingError::Timeout(t),
                RPCError::PayloadTooLarge(p) => StreamingError::Network(NetworkError::new(&p)),
                RPCError::RemoteError(r) => StreamingError::Unreachable(Unreachable::new(&r)),
            }),
        }
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
    /// The live address book. Shared with every other component that has to
    /// reach a node, so an address learned from a committed membership entry is
    /// usable immediately — including for the `add_learner` that produced it.
    peers: crate::identity::PeerDirectory,
    /// Which consensus group this factory's clients speak for.
    group: GroupId,
    /// Shared secret sent on every `/internal/*` request.
    cluster_secret: Option<String>,
}

impl RaftNetworkFactoryImpl {
    pub fn new(peers: crate::identity::PeerDirectory) -> Self {
        Self {
            peers,
            group: METADATA_GROUP,
            cluster_secret: None,
        }
    }

    /// Speak for `group` instead of the metadata group.
    ///
    /// One factory per group: a `Raft` instance holds its factory, and the
    /// group id is what lets a peer dispatch the message to the right log.
    pub fn with_group(mut self, group: GroupId) -> Self {
        self.group = group;
        self
    }

    /// Attach the cluster secret required by peer internal routes.
    ///
    /// Without it, every Raft RPC is rejected with 401 as soon as the peers
    /// configure `REKHA_CLUSTER_SECRET`, so the cluster silently never forms.
    pub fn with_cluster_secret(mut self, secret: Option<String>) -> Self {
        self.cluster_secret = secret;
        self
    }
}

impl RaftNetworkFactory<RC> for RaftNetworkFactoryImpl {
    type Network = RaftNetworkImpl;

    async fn new_client(&mut self, target: u64, _node: &u64) -> Self::Network {
        RaftNetworkImpl::new(
            target,
            self.peers.url(target),
            self.group,
            self.cluster_secret.clone(),
        )
    }
}

/// Accept a peer address with or without a scheme.
///
/// Operators configure peers as `host:port` in the common case, and as
/// `https://host:port` when the internal listener terminates TLS. Hardcoding
/// `http://` in front of the stored address produced `http://https://host:port`
/// for the TLS form, so the scheme, when present, is preserved.
pub fn normalize_peer_url(addr: &str) -> String {
    let addr = addr.trim().trim_end_matches('/');
    if addr.contains("://") {
        addr.to_string()
    } else {
        format!("http://{addr}")
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

    /// Every Raft RPC must target exactly one route segment after
    /// `/internal/raft/`.
    ///
    /// `send_rpc` prefixes the path itself, so passing `raft/install_snapshot`
    /// produced `/internal/raft/raft/install_snapshot`, which matches no route
    /// and 404s — meaning a lagging follower could never install a snapshot.
    #[tokio::test]
    async fn all_raft_rpcs_target_mounted_routes() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        for (label, path) in [
            ("vote", "vote"),
            ("append_entries", "append_entries"),
            ("install_snapshot", "install_snapshot"),
        ] {
            let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
                .await
                .unwrap();
            let port = listener.local_addr().unwrap().port();

            let handle = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buf = vec![0u8; 8192];
                let n = socket.read(&mut buf).await.unwrap();
                let head = String::from_utf8_lossy(&buf[..n]).to_string();
                let body = br#"{"error":"probe"}"#;
                let resp = format!(
                    "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
                    body.len()
                );
                socket.write_all(resp.as_bytes()).await.unwrap();
                socket.write_all(body).await.unwrap();
                socket.flush().await.unwrap();
                head
            });

            // The group id is part of the path: one process runs one log per group.
            let request_line = format!("POST /internal/raft/0/{path}");
            let opt = || RPCOption::new(Duration::from_secs(1));
            let mut net = RaftNetworkImpl::new(1, Some(format!("127.0.0.1:{port}")), 0, None);
            // Each call must fail (the probe replies 400); the assertion is on
            // the request target the client sent, not the outcome.
            match label {
                "vote" => {
                    let _ = net
                        .vote(
                            openraft::raft::VoteRequest {
                                vote: openraft::Vote::new(1, 1u64),
                                last_log_id: None,
                            },
                            opt(),
                        )
                        .await;
                }
                "append_entries" => {
                    use openraft::raft::AppendEntriesRequest;
                    let _ = net
                        .append_entries(
                            AppendEntriesRequest {
                                vote: openraft::Vote::new(1, 1u64),
                                prev_log_id: None,
                                entries: Vec::new(),
                                leader_commit: None,
                            },
                            opt(),
                        )
                        .await;
                }
                _ => {
                    let snapshot = openraft::Snapshot {
                        meta: openraft::SnapshotMeta {
                            last_log_id: None,
                            last_membership: openraft::StoredMembership::default(),
                            snapshot_id: "probe".to_string(),
                        },
                        snapshot: Box::new(std::io::Cursor::new(Vec::<u8>::new())),
                    };
                    let _ = net
                        .full_snapshot(
                            openraft::Vote::new(1, 1u64),
                            snapshot,
                            std::future::pending::<openraft::error::ReplicationClosed>(),
                            opt(),
                        )
                        .await;
                }
            }

            let head = handle.await.unwrap();
            assert!(
                head.starts_with(&request_line),
                "{label}: expected `{request_line}`, got:\n{head}"
            );
            assert!(
                !head.contains("/internal/raft/raft/"),
                "{label}: doubled route prefix in request target:\n{head}"
            );
        }
    }

    use openraft::{Vote, raft::VoteRequest};

    /// A peer that accepts the connection and then never answers must fail the
    /// RPC inside the deadline openraft gave us.
    ///
    /// This covers a real hang, not a hypothetical one: the client was built
    /// with `reqwest::Client::new()`, which has no timeout, and the deadline in
    /// `RPCOption` was ignored. A peer in that state held its replication stream
    /// open until the socket itself failed, so the leader never retried it.
    #[tokio::test]
    async fn an_unanswering_peer_fails_within_the_deadline() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a stand-in peer");
        let addr = listener.local_addr().expect("addr");
        let held = tokio::spawn(async move {
            // Hold the connection open without answering. Dropping it would make
            // the failure a reset, which is a different case.
            if let Ok((stream, _)) = listener.accept().await {
                tokio::time::sleep(Duration::from_secs(30)).await;
                drop(stream);
            }
        });

        let mut network = RaftNetworkImpl::new(
            2,
            Some(format!("127.0.0.1:{}", addr.port())),
            METADATA_GROUP,
            None,
        );
        let option = RPCOption::new(Duration::from_millis(400));
        let request = VoteRequest::new(Vote::new(1, 2), None);

        let started = std::time::Instant::now();
        let answer = network.vote(request, option).await;
        let elapsed = started.elapsed();

        assert!(
            answer.is_err(),
            "a peer that never answers must not look like a successful RPC"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "the RPC took {elapsed:?}, past the deadline it was given"
        );
        held.abort();
    }
}
