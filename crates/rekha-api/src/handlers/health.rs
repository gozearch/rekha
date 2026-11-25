use crate::prelude::*;

pub(crate) async fn health_handler() -> Result<StatusCode, StatusCode> {
    Ok(StatusCode::OK)
}

/// Readiness check — returns 200 if engine is accessible, 503 otherwise.
///
/// The body states the active durability level so an operator can confirm
/// remotely what a successful write promises, instead of inferring it from
/// deployment notes.
pub(crate) async fn ready_handler(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    // In cluster mode, readiness means "can this node serve a write", which
    // requires an elected leader this node can reach. Reporting ready while
    // there is no leader would route load at nodes that can only answer 503.
    let cluster = match state.raft.as_ref() {
        Some(raft) => {
            let m = raft.metrics().borrow().clone();
            let leader = m.current_leader;
            let role = format!("{:?}", m.state).to_lowercase();
            let has_leader = leader.is_some();
            Some(serde_json::json!({
                "node_id": m.id,
                "role": role,
                "term": m.current_term,
                "leader": leader,
                "applied_index": m.last_applied.map(|l| l.index),
                "quorum_age_ms": m.millis_since_quorum_ack,
            }))
            .map(|info| (info, has_leader))
        }
        None => None,
    };
    let has_leader = cluster.as_ref().map(|(_, ok)| *ok).unwrap_or(true);
    let cluster_info = cluster.map(|(info, _)| info);

    // Shard leadership is as load-bearing as metadata leadership. A node whose
    // shard group is mid-election answers 503 to every write for that shard, so
    // reporting ready routes load into a window already known to be bad. The
    // metadata group's leader was the only one consulted here, which is why a
    // node could be "ready" and unable to accept the data it was sent.
    let shards_without_leader: Vec<u32> = state
        .shards
        .as_ref()
        .map(|shards| {
            shards
                .shards()
                .into_iter()
                .filter(|shard| shards.leader(*shard).is_none())
                .map(|shard| shard.0)
                .collect()
        })
        .unwrap_or_default();
    let shards_ready = shards_without_leader.is_empty();
    let shards_info = state.shards.as_ref().map(|shards| {
        serde_json::json!({
            "hosted": shards.shards().len(),
            "without_leader": shards_without_leader,
        })
    });
    // A node whose bootstrap credentials have not fully applied can accept the
    // bootstrap key and still deny its requests, because the role assignment
    // lands in a separate catalog write. Treat that window as "not ready" so a
    // client that waits for readiness does not get a 403 for a healthy cluster.
    let bootstrap_ready = state.bootstrap_ready.load(Ordering::SeqCst);
    let is_ready = has_leader && shards_ready && bootstrap_ready;

    // A node that could not apply a committed entry has diverged from the log
    // it claims to have. It must not serve, whatever else it thinks.
    if state.apply_failure.load(Ordering::SeqCst) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "ready": false,
                "error": "a committed log entry could not be applied; this node is diverged \
                          and must be restarted or rebuilt from a snapshot",
                "cluster": cluster_info,
            })),
        );
    }

    match state
        .engine
        .list_collections(&state.tenant, &state.database)
    {
        Ok(collections) => {
            let status = if is_ready {
                StatusCode::OK
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            };
            (
                status,
                Json(serde_json::json!({
                    "ready": is_ready,
                    "durability": state.engine.durability().as_str(),
                    "tenant": state.tenant,
                    "database": state.database,
                    "collections": collections.len(),
                    "bootstrap_ready": bootstrap_ready,
                    "cluster": cluster_info,
                    "shards": shards_info,
                })),
            )
        }
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "ready": false,
                "durability": state.engine.durability().as_str(),
                "error": e.to_string(),
                "cluster": cluster_info,
            })),
        ),
    }
}

/// ChromaDB auth identity stub.
pub(crate) async fn auth_identity() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "user_id": "00000000-0000-0000-0000-000000000000"
    }))
}
