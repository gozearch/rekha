use crate::prelude::*;
use crate::*;

/// Live record counts for `ids`, fetched from the nodes that hold the data.
///
/// A count is a data question: a node that is not a replica cannot answer it,
/// and answering `0` would look like an empty collection. This asks each
/// collection's first replica in one request per replica, not one per
/// collection, so a list of a thousand collections costs at most one round trip
/// per node.
///
/// A replica that cannot be reached is left out of the result rather than
/// reported as zero. Callers decide what to show for an unknown count; they must
/// not paper over it with a number that means "empty".
pub(crate) async fn counts_for(
    state: &AppState,
    ids: &[uuid::Uuid],
) -> Result<std::collections::HashMap<uuid::Uuid, u64>, ApiError> {
    let mut counts = std::collections::HashMap::new();
    let mut remote: std::collections::HashMap<u64, Vec<uuid::Uuid>> =
        std::collections::HashMap::new();

    for id in ids {
        // A local count is only an answer when this node is a voter of the
        // group that holds the collection. A node that has merely opened the
        // files — `ensure_started` materializes an empty partition for a shard
        // that is still being replicated — would otherwise report `0` for
        // records that exist elsewhere, which is indistinguishable from an
        // empty collection. Treat it as "cannot answer" and forward instead.
        let local = if state.is_voter_for(id) {
            state.engine.count_local(id)
        } else {
            Ok(None)
        };
        match local {
            Ok(Some(local)) => {
                counts.insert(*id, local);
            }
            Ok(None) => {
                // A count belongs to a replica that holds the data, so the target
                // is the first owner of the collection's shard.
                let owner = state.shard_of(id).and_then(|shard| {
                    state
                        .shards
                        .as_ref()
                        .and_then(|shards| shards.owners(shard).first().copied())
                });
                if let Some(owner) = owner {
                    remote.entry(owner).or_default().push(*id);
                }
            }
            Err(e) => {
                tracing::warn!(collection = %id, %e, "could not count a collection");
            }
        }
    }

    for (owner, ids) in remote {
        let Some(url) = state.peers.url(owner) else {
            continue;
        };
        let mut request = state
            .http
            .post(format!("{url}/internal/stats/counts"))
            .json(&ids);
        if let Some(secret) = state.cluster_secret.as_deref() {
            request = request.header(crate::leadership::SECRET_HEADER, secret);
        }
        match request.send().await {
            Ok(response) => match response
                .json::<std::collections::HashMap<String, u64>>()
                .await
            {
                Ok(remote_counts) => {
                    for (id, count) in remote_counts {
                        if let Ok(id) = id.parse() {
                            counts.insert(id, count);
                        }
                    }
                }
                Err(e) => tracing::warn!(node = owner, %e, "unreadable count response"),
            },
            Err(e) => tracing::warn!(node = owner, %e, "count request failed"),
        }
    }

    // Every requested id must be answered. A missing entry means the replica
    // that holds it could not be reached, and the caller must not turn that into
    // a zero: "the collection is empty" and "I could not ask" are different
    // facts, and only one of them is true.
    if let Some(missing) = ids.iter().find(|id| !counts.contains_key(*id)) {
        return Err(ApiError::Engine(EngineError::Unavailable(format!(
            "no replica answered the count for collection {missing}; retry, or check that \
             the shard's owners are reachable"
        ))));
    }
    Ok(counts)
}

/// Counts this node holds, for a peer that needs them.
///
/// Only a voter answers: a peer may hold a group and empty partitions while it
/// catches up, and answering `0` there would turn "still replicating" into
/// "empty collection" one forwarding hop away from the client.
pub(crate) async fn stats_counts(
    State(state): State<Arc<AppState>>,
    Json(ids): Json<Vec<uuid::Uuid>>,
) -> Json<std::collections::HashMap<String, u64>> {
    let mut counts = std::collections::HashMap::new();
    for id in ids {
        if state.is_voter_for(&id)
            && let Ok(Some(count)) = state.engine.count_local(&id)
        {
            counts.insert(id.to_string(), count);
        }
    }
    Json(counts)
}
