//! Automatic cluster reconciler and rebalancer.
//!
//! Runs on every node:
//! - Local shard leader reconciles shard group membership towards target placement.
//! - Metadata leader monitors cluster membership, learner catch-up, and commits
//!   global topology updates (`SetVoters`, `SetShardOwners`).

use std::sync::Arc;
use std::time::Duration;

use openraft::Raft;

use crate::identity::PeerDirectory;
use crate::placement::{Placement, ShardId};
use crate::raft_types::{ClusterOperation, RaftTypeConfig};
use crate::shard::ShardCluster;

/// Entries behind `last_log_index` that still allow promotion.
///
/// A learner whose replication `matched` index is at least this close to the
/// leader's last log is caught up enough to become a voter. `None` matched
/// (never replicated) is treated as infinite lag, never as zero, so an idle
/// cluster does not promote a node that has replicated nothing.
pub const PROMOTION_LAG_THRESHOLD: u64 = 50;

/// Attach the cluster secret to an internal request, if one is configured.
///
/// Internal routes fail closed without it; the three reconciler call sites
/// share this rather than repeating the conditional header.
fn with_secret(
    req: reqwest::RequestBuilder,
    cluster_secret: Option<&str>,
) -> reqwest::RequestBuilder {
    match cluster_secret {
        Some(secret) => req.header("x-rekha-cluster-secret", secret),
        None => req,
    }
}

/// Record one promotion attempt for observability.
fn record_promotion_attempt(
    shards: &ShardCluster,
    group: &str,
    node: u64,
    lag: Option<u64>,
    outcome: &str,
) {
    metrics::counter!(
        "rekha_promotion_attempts_total",
        "group" => group.to_string(),
        "node" => node.to_string(),
        "outcome" => outcome.to_string()
    )
    .increment(1);
    if let Some(lag) = lag {
        metrics::gauge!(
            "rekha_learner_lag_entries",
            "group" => group.to_string(),
            "node" => node.to_string()
        )
        .set(lag as f64);
    }
    shards.record_promotion_attempt(group, node, lag, outcome);
    tracing::info!(group, node, lag, outcome, "promotion attempt");
}

/// Reconcile shard groups led by this local node towards target placement.
pub async fn reconcile_local_shard_leaders(node_id: u64, shards: &Arc<ShardCluster>) {
    let placement = shards.placement();
    let target_placement = match Placement::new(
        placement.shard_count(),
        placement.replication_factor(),
        placement.voters().iter().copied(),
    ) {
        Ok(p) => p,
        Err(_) => return,
    };

    let mut shards_sorted = shards.shards();
    shards_sorted.sort_unstable_by_key(|s| s.0);
    for shard in shards_sorted {
        let Some(shard_raft) = shards.raft(shard) else {
            continue;
        };
        let m = shard_raft.metrics().borrow().clone();
        if m.current_leader != Some(node_id) {
            continue;
        }

        let target_owners = target_placement.owners(shard).to_vec();
        let current_voters: std::collections::BTreeSet<u64> =
            m.membership_config.membership().voter_ids().collect();
        let target_set: std::collections::BTreeSet<u64> = target_owners.iter().copied().collect();

        if current_voters == target_set {
            continue;
        }

        // Add missing target nodes as learners
        let existing_learners: std::collections::BTreeSet<u64> =
            m.membership_config.membership().learner_ids().collect();
        for &node in &target_owners {
            if !current_voters.contains(&node) && !existing_learners.contains(&node) {
                let _ = shard_raft.add_learner(node, node, false).await;
            }
        }

        // At most one promotion attempt per group per tick. A learner that
        // never replicated (matched None) is infinite lag and skips.
        let shard_last_log = m.last_log_index.unwrap_or(0);
        let mut lag_of: Option<u64> = Some(0);
        let mut pending: Option<u64> = None;
        for &node in &target_owners {
            if current_voters.contains(&node) {
                continue;
            }
            match crate::replication_matched(&m, &node) {
                None => {
                    lag_of = None;
                    pending = Some(node);
                    break;
                }
                Some(matched) => {
                    let lag = shard_last_log.saturating_sub(matched);
                    lag_of = Some(lag_of.map_or(lag, |prev: u64| prev.max(lag)));
                    if lag > PROMOTION_LAG_THRESHOLD {
                        pending = Some(node);
                    }
                }
            }
        }
        let group_label = format!("shard:{}", shard.0);
        match (lag_of, pending) {
            (None, Some(node)) => {
                record_promotion_attempt(shards, &group_label, node, None, "skipped_lag");
                continue;
            }
            (Some(_), Some(node)) => {
                if let Some(lag) = lag_of {
                    record_promotion_attempt(shards, &group_label, node, Some(lag), "skipped_lag");
                }
                continue;
            }
            _ => {}
        }

        if shard_raft
            .change_membership(target_set, false)
            .await
            .is_ok()
        {
            for &node in &target_owners {
                if !current_voters.contains(&node) {
                    record_promotion_attempt(shards, &group_label, node, lag_of, "promoted");
                }
            }
            tracing::info!(
                shard = shard.0,
                ?target_owners,
                "shard leader transitioned membership to target owners"
            );
        } else {
            for &node in &target_owners {
                if !current_voters.contains(&node) {
                    record_promotion_attempt(shards, &group_label, node, lag_of, "error");
                }
            }
        }
    }

    // Reconcile any in-flight 2PC transactions older than 15s on shards led by this node
    let stale = shards.stale_staged_txs(Duration::from_secs(15));
    for (tx_id, shard) in stale {
        let Some(shard_raft) = shards.raft(shard) else {
            continue;
        };
        let m = shard_raft.metrics().borrow().clone();
        if m.current_leader != Some(node_id) {
            continue;
        }
        if shards.is_tx_committed(&tx_id) {
            tracing::info!(shard = shard.0, %tx_id, "reconciler committing stranded 2PC transaction");
            let _ = shard_raft
                .client_write(ClusterOperation::CommitBatch { tx_id })
                .await;
        } else {
            tracing::info!(shard = shard.0, %tx_id, "reconciler aborting stranded 2PC transaction");
            let _ = shard_raft
                .client_write(ClusterOperation::AbortBatch { tx_id })
                .await;
        }
        shards.mark_tx_resolved(&tx_id);
    }
}

/// Execute one reconciliation tick.
pub async fn run_rebalancer_tick(
    node_id: u64,
    raft: &Arc<Raft<RaftTypeConfig>>,
    shards: &Arc<ShardCluster>,
    peers: &PeerDirectory,
    http: &reqwest::Client,
    cluster_secret: Option<&str>,
) {
    // 1. Shards led by this node reconcile their Raft membership
    reconcile_local_shard_leaders(node_id, shards).await;

    // 2. Only the metadata group leader performs global topology reconciliation
    let m = raft.metrics().borrow().clone();
    if !m.state.is_leader() {
        return;
    }

    let last_log = m.last_log_index.unwrap_or(0);
    let current_voters: std::collections::BTreeSet<u64> =
        m.membership_config.membership().voter_ids().collect();

    // 2a. Learner auto-promotion in the metadata group. At most one
    // promotion attempt per tick; groups proceed independently. Falls through
    // to 2b either way: shard reconciliation must not starve while a learner
    // exists.
    let mut learners: Vec<u64> = m.membership_config.membership().learner_ids().collect();
    learners.sort_unstable();
    if let Some(learner) = learners.into_iter().next() {
        let mut attempt: Option<(Option<u64>, &str)> = None;
        let mut lag = None;
        match crate::replication_matched(&m, &learner) {
            // Never replicated: infinite lag, never promote on an idle cluster.
            None => attempt = Some((None, "skipped_lag")),
            Some(matched) => {
                lag = Some(last_log.saturating_sub(matched));
                if lag.is_some_and(|lag| lag > PROMOTION_LAG_THRESHOLD) {
                    attempt = Some((lag, "skipped_lag"));
                }
            }
        }
        if let Some((lag, outcome)) = attempt {
            record_promotion_attempt(shards, "metadata", learner, lag, outcome);
        } else {
            let mut new_voters = current_voters.clone();
            new_voters.insert(learner);
            if raft
                .change_membership(new_voters.clone(), true)
                .await
                .is_ok()
            {
                record_promotion_attempt(shards, "metadata", learner, lag, "promoted");
                tracing::info!(learner, "auto-promoted learner to voter in metadata group");
                let voter_vec: Vec<u64> = new_voters.into_iter().collect();
                if let Err(e) = raft
                    .client_write(ClusterOperation::SetVoters {
                        voters: voter_vec.clone(),
                    })
                    .await
                {
                    tracing::error!(learner, %e, "failed to record SetVoters in consensus log");
                }
            } else {
                record_promotion_attempt(shards, "metadata", learner, lag, "error");
            }
        }
    }

    // 2b. Per-shard owner reconciliation
    let placement = shards.placement();
    let target_placement = match Placement::new(
        placement.shard_count(),
        placement.replication_factor(),
        placement.voters().iter().copied(),
    ) {
        Ok(p) => p,
        Err(_) => return,
    };

    for s in 0..placement.shard_count() {
        let shard = ShardId(s);
        let current_owners = shards.owners(shard);
        let target_owners = target_placement.owners(shard).to_vec();
        if current_owners != target_owners {
            // Instruct new nodes to ensure shard group is started
            for &new_node in &target_owners {
                if !current_owners.contains(&new_node)
                    && let Some(url) = peers.url(new_node)
                {
                    let req = with_secret(
                        http.post(format!(
                            "{url}/internal/cluster/shard/{}/ensure_started",
                            shard.0
                        )),
                        cluster_secret,
                    );
                    let _ = req.send().await;
                }
            }

            // Check if shard's voters have transitioned to target_owners
            let is_rebalanced = if let Some(shard_raft) = shards.raft(shard) {
                let voters: std::collections::BTreeSet<u64> = shard_raft
                    .metrics()
                    .borrow()
                    .membership_config
                    .membership()
                    .voter_ids()
                    .collect();
                voters == target_owners.iter().copied().collect()
            } else {
                let mut match_found = false;
                for &owner in &target_owners {
                    if let Some(url) = peers.url(owner) {
                        let req = with_secret(
                            http.get(format!(
                                "{url}/internal/cluster/shard/{}/membership",
                                shard.0
                            )),
                            cluster_secret,
                        );
                        if let Ok(resp) = req.send().await
                            && let Ok(body) = resp.json::<serde_json::Value>().await
                        {
                            // New shape is an object with a `voters` array;
                            // accept a bare array from an older binary too.
                            let voters: Option<Vec<u64>> = body
                                .get("voters")
                                .and_then(|v| serde_json::from_value::<Vec<u64>>(v.clone()).ok())
                                .or_else(|| serde_json::from_value::<Vec<u64>>(body.clone()).ok());
                            if let Some(voters) = voters {
                                let v_set: std::collections::BTreeSet<u64> =
                                    voters.into_iter().collect();
                                if v_set == target_owners.iter().copied().collect() {
                                    match_found = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                match_found
            };

            if is_rebalanced {
                tracing::info!(
                    shard = shard.0,
                    ?target_owners,
                    "auto-rebalanced shard owners"
                );
                let _ = raft
                    .client_write(ClusterOperation::SetShardOwners {
                        shard,
                        owners: target_owners.clone(),
                    })
                    .await;

                // Drop old replicas that are no longer owners
                for &old_node in &current_owners {
                    if !target_owners.contains(&old_node)
                        && let Some(url) = peers.url(old_node)
                    {
                        let req = with_secret(
                            http.post(format!("{url}/internal/cluster/shard/{}/drop", shard.0)),
                            cluster_secret,
                        );
                        let _ = req.send().await;
                    }
                }
            }
        }
    }
}

/// Start the background reconciler task on this node.
pub fn spawn_auto_rebalancer(
    node_id: u64,
    raft: Arc<Raft<RaftTypeConfig>>,
    shards: Arc<ShardCluster>,
    peers: PeerDirectory,
    http: reqwest::Client,
    cluster_secret: Option<String>,
) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(1500)).await;
            run_rebalancer_tick(
                node_id,
                &raft,
                &shards,
                &peers,
                &http,
                cluster_secret.as_deref(),
            )
            .await;
        }
    });
}
