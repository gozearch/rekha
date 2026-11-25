use crate::prelude::*;
use crate::*;

/// One learner's catch-up position, for `membership` output.
///
/// `matched_index` is `None` when the learner never replicated (infinite lag);
/// `lag` is entries behind `last_log_index`, `None` for the same reason.
pub(crate) fn learner_entry(
    node: u64,
    matched_index: Option<u64>,
    last_log_index: u64,
) -> serde_json::Value {
    let lag = matched_index.map(|matched| last_log_index.saturating_sub(matched));
    serde_json::json!({
        "node": node,
        "matched_index": matched_index,
        "last_log_index": last_log_index,
        "lag": lag,
    })
}

/// Voters plus each learner's replication position for one Raft group.
///
/// Shared by the cluster-wide and per-shard membership routes so both answer
/// the same shape. `matched` comes from [`replication_matched`]: `None`
/// means the peer never replicated (infinite lag), never zero.
/// One Raft group's voter set plus each learner's replication position and
/// the leader's last log index.
pub(crate) type GroupLearners<NID> = (Vec<NID>, Vec<(NID, Option<u64>)>, u64);

pub(crate) fn group_learners<NID, N>(metrics: &openraft::RaftMetrics<NID, N>) -> GroupLearners<NID>
where
    NID: openraft::NodeId + Clone,
    N: openraft::Node,
{
    use rekha_cluster::replication_matched;
    let membership = metrics.membership_config.membership().clone();
    let voters: Vec<NID> = membership.voter_ids().collect();
    let mut learners = Vec::new();
    for node in membership.learner_ids() {
        learners.push((node.clone(), replication_matched(metrics, &node)));
    }
    (voters, learners, metrics.last_log_index.unwrap_or(0))
}

/// The voters/learners split for one Raft group's metrics.
///
/// Both membership routes answer this shape so operators and tests read one
/// format. `members` is a deprecated alias (voters followed by learners),
/// kept so existing `membership list` parsers keep working while callers
/// migrate to `voters`/`learners`.
pub(crate) fn membership_body(
    voters: Vec<u64>,
    learner_matched: Vec<(u64, Option<u64>)>,
    last_log_index: u64,
    last_promotion: serde_json::Value,
) -> serde_json::Value {
    let mut learners = Vec::with_capacity(learner_matched.len());
    let mut members = voters.clone();
    for (node, matched_index) in learner_matched {
        members.push(node);
        learners.push(learner_entry(node, matched_index, last_log_index));
    }
    serde_json::json!({
        "voters": voters,
        "learners": learners,
        "last_promotion": last_promotion,
        "members": members,
    })
}
/// The most recent promotion attempt for `group`, if any.
pub(crate) fn last_promotion_for(
    shards: Option<&Arc<rekha_cluster::ShardCluster>>,
    group: &str,
) -> serde_json::Value {
    let Some(shards) = shards else {
        return serde_json::Value::Null;
    };
    let mut best: Option<(String, u64, rekha_cluster::shard::PromotionAttempt)> = None;
    for (g, node, attempt) in shards.last_promotions() {
        if g != group {
            continue;
        }
        let replace = match &best {
            None => true,
            Some((_, _, prev)) => attempt.at_ms >= prev.at_ms,
        };
        if replace {
            best = Some((g, node, attempt));
        }
    }
    match best {
        None => serde_json::Value::Null,
        Some((group, node, attempt)) => serde_json::json!({
            "group": group,
            "node": node,
            "at_ms": attempt.at_ms,
            "outcome": attempt.outcome,
            "lag": attempt.lag,
        }),
    }
}

pub(crate) async fn raft_membership(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let raft = state.raft.as_ref().ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let metrics = raft.metrics().borrow().clone();
    let (voters, learner_matched, last_log_index) = group_learners(&metrics);
    // Per-shard leadership, so an operator can see which group is electing
    // without reading logs. `None` while a group has no leader, never a
    // default id.
    let shards: Option<Vec<serde_json::Value>> = state.shards.as_ref().map(|shards| {
        shards
            .shards()
            .into_iter()
            .map(|id| {
                serde_json::json!({
                    "shard": id.0,
                    "leader": shards.leader(id),
                })
            })
            .collect()
    });

    let mut body = membership_body(
        voters,
        learner_matched,
        last_log_index,
        last_promotion_for(state.shards.as_ref(), "metadata"),
    );
    body["leader"] = serde_json::json!(metrics.current_leader);
    body["state"] = serde_json::json!(format!("{:?}", metrics.state));
    body["shards"] = serde_json::json!(shards);
    Ok(Json(body))
}

/// Membership changes are refused while shards exist.
///
/// Placement is a function of the voter set: a node is one of the inputs every
/// shard's owner set is computed from, so adding or removing one moves shards
/// without moving their data. The two halves would then disagree about where a
/// collection lives, which reads as a never-ending election for that shard
/// rather than as an error. Refusing is the only honest answer until a
/// rebalance exists.
///
/// This guard applies to the HTTP routes only (`add_learner` beyond the
/// initial admit, `remove_member`). The background reconciler calls
/// `Raft::change_membership` directly and is the sole bypass: it promotes a
/// learner only after the lag check passes, one group at a time. During joint
/// consensus `remove_member` still returns 409; the reconciler never removes a
/// voter in v1.
///
/// The voter set is persisted in each node's placement document, and a node
/// whose `--peers` disagrees with that document refuses to start. So a
/// membership change that landed here would not merely be invisible: it would
/// leave every node unable to restart.
pub(crate) fn refuse_membership_change() -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    Err((
        StatusCode::CONFLICT,
        Json(serde_json::json!({
            "error": "membership cannot change while shards are placed: a node is one of the \
                      voters each shard's owner set is computed from, so adding or removing one \
                      moves shards without moving their data. Rebalance support is required \
                      first.",
        })),
    ))
}

/// Turn a Raft write failure into something an operator can act on.
pub(crate) fn membership_write_error(
    error: openraft::error::RaftError<u64, openraft::error::ClientWriteError<u64, u64>>,
    doing: &str,
) -> (StatusCode, Json<serde_json::Value>) {
    use openraft::error::{ClientWriteError, RaftError};

    if let RaftError::APIError(ClientWriteError::ForwardToLeader(forward)) = &error {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": format!(
                    "{doing} has to run on the metadata leader, and this node is not it"
                ),
                "leader": forward.leader_id.or(forward.leader_node),
            })),
        );
    }
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({ "error": format!("{doing} failed: {error}") })),
    )
}

/// How many times an internal admin call has been forwarded.
pub(crate) const ADMIN_HOPS: &str = "x-rekha-admin-hops";
pub(crate) const MAX_ADMIN_HOPS: u8 = 3;

pub(crate) fn admin_hops(headers: &axum::http::HeaderMap) -> u8 {
    headers
        .get(ADMIN_HOPS)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

pub(crate) fn too_many_admin_hops() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::LOOP_DETECTED,
        Json(serde_json::json!({
            "error": "the membership request was forwarded too many times; no stable metadata \
                      leader was found",
        })),
    )
}

/// The leader a Raft write pointed at, if it was refused for that reason.
pub(crate) fn leader_of(
    error: &openraft::error::RaftError<u64, openraft::error::ClientWriteError<u64, u64>>,
) -> Option<u64> {
    match error {
        openraft::error::RaftError::APIError(
            openraft::error::ClientWriteError::ForwardToLeader(forward),
        ) => forward.leader_id.or(forward.leader_node),
        _ => None,
    }
}

/// Replay an internal admin call on the metadata leader.
///
/// These endpoints live on the internal listener, which has no public
/// `require_leader` middleware. A CLI command can therefore land on any node,
/// and leadership can move between the caller's lookup and the command. The
/// Raft client already reported where the leader is; retrying there turns a
/// 503 that says "ask again" into a command that works. The hop header keeps a
/// leadership race from becoming a loop.
pub(crate) async fn forward_admin(
    state: &AppState,
    leader: u64,
    path: &str,
    body: &serde_json::Value,
    hops: u8,
) -> Option<Result<StatusCode, (StatusCode, Json<serde_json::Value>)>> {
    // Never forward to ourselves: that is a stale `ForwardToLeader` answer, and
    // replaying it here would turn a transient race into a loop.
    if state.raft.as_ref()?.metrics().borrow().id == leader {
        return None;
    }
    let url = state.peers.url(leader)?;
    let mut request = state
        .http
        .post(format!("{url}{path}"))
        .timeout(std::time::Duration::from_secs(15))
        .header(ADMIN_HOPS, (hops + 1).to_string())
        .json(body);
    if let Some(secret) = state.cluster_secret.as_ref() {
        request = request.header(leadership::SECRET_HEADER, secret.as_str());
    }
    let response = request.send().await.ok()?;
    let status = response.status();
    if status.is_success() {
        return Some(Ok(StatusCode::OK));
    }
    let body = response.json::<serde_json::Value>().await.unwrap_or_else(
        |_| serde_json::json!({"error": "the metadata leader refused the request"}),
    );
    Some(Err((status, Json(body))))
}

/// The placement this cluster was built with.
///
/// A node joining has no `--peers`, so it has nobody to ask except the node it
/// was told to join; this is what it asks. The answer is the same three inputs
/// every member persists, so the joiner routes by the same owner sets as
/// everyone else — while hosting nothing until it is promoted.
pub(crate) async fn raft_placement(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let shards = state.shards.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    let placement = shards.placement();
    Ok(Json(serde_json::json!({
        "shard_count": placement.shard_count(),
        "replication_factor": placement.replication_factor(),
        "voters": placement.voters(),
    })))
}

/// Admit a node to the metadata group as a learner.
///
/// A learner does not vote, so the voter set — and with it placement — is
/// unchanged. That is why this is admitted while shards are placed and
/// [`raft_remove_member`] is not: the membership guard exists to stop shards
/// moving, and nothing moves here.
///
/// The address is proposed as a committed entry first. The Raft client reaches
/// the learner through the shared address book, which until now knew only what
/// `--peers` said at each node's last start, so without this the leader has no
/// way to reach a node the flag does not mention — the reason adding a node
/// used to require restarting the whole cluster.
///
/// Admission is deliberately non-blocking. Waiting for a learner to catch up
/// before answering holds an operator's request open for as long as the data
/// takes — and forever if the learner is unreachable. The entry that admits it
/// is committed before this returns; `/ready` and the replication metrics are
/// where catch-up progress belongs.
pub(crate) async fn raft_add_learner(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<serde_json::Value>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    let hops = admin_hops(&headers);
    if hops >= MAX_ADMIN_HOPS {
        return Err(too_many_admin_hops());
    }
    let raft = state.raft.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({"error": "no metadata group"})),
    ))?;
    let node_id = req["node_id"].as_u64().ok_or((
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": "node_id is required"})),
    ))?;
    let addr = req["addr"]
        .as_str()
        .map(str::trim)
        .filter(|addr| !addr.is_empty())
        .ok_or((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "addr is required: it is how the leader reaches the new node, and no \
                          flag teaches it that address",
            })),
        ))?
        .to_owned();

    if state.peers.addr(node_id).as_deref() != Some(addr.as_str()) {
        // Committed, not local: every node has to be able to reach the new node,
        // and the address has to survive a restart and a change of leader.
        if let Err(e) = raft
            .client_write(ClusterOperation::AddNode {
                node_id,
                addr: addr.clone(),
            })
            .await
        {
            if let Some(leader) = leader_of(&e)
                && let Some(response) =
                    forward_admin(&state, leader, "/internal/cluster/add_learner", &req, hops).await
            {
                return response;
            }
            return Err(membership_write_error(
                e,
                "recording where the new node lives",
            ));
        }
    }

    if let Err(e) = raft.add_learner(node_id, node_id, false).await {
        if let Some(leader) = leader_of(&e)
            && let Some(response) =
                forward_admin(&state, leader, "/internal/cluster/add_learner", &req, hops).await
        {
            return response;
        }
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ));
    }
    Ok(StatusCode::OK)
}

pub(crate) async fn raft_remove_member(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<serde_json::Value>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    let hops = admin_hops(&headers);
    if hops >= MAX_ADMIN_HOPS {
        return Err(too_many_admin_hops());
    }
    if state.shards.is_some() {
        refuse_membership_change()?;
    }
    let raft = state.raft.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({"error": "no metadata group"})),
    ))?;
    let node_id = req["node_id"].as_u64().ok_or((
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": "node_id is required"})),
    ))?;
    let metrics = raft.metrics().borrow().clone();
    let new_members: std::collections::BTreeSet<u64> = metrics
        .membership_config
        .membership()
        .voter_ids()
        .chain(metrics.membership_config.membership().learner_ids())
        .filter(|&id| id != node_id)
        .collect();
    if new_members.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "the last member cannot be removed"})),
        ));
    }
    if let Err(e) = raft.change_membership(new_members, true).await {
        if let Some(leader) = leader_of(&e)
            && let Some(response) = forward_admin(
                &state,
                leader,
                "/internal/cluster/remove_member",
                &req,
                hops,
            )
            .await
        {
            return response;
        }
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ));
    }
    Ok(StatusCode::OK)
}
