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

/// What one reconciliation tick should do about a learner.
///
/// Split out of the two serving loops because the decision has six cases
/// (never replicated, behind by more than the threshold, within the threshold,
/// already a voter, nothing to promote, several learners) and inlining it made
/// each loop read like a state machine with the metrics calls on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromotionDecision {
    /// Membership may be widened to the target owners.
    Promote { lag: Option<u64> },
    /// `node` is not caught up; record the attempt and try again next tick.
    Skip { node: u64, lag: Option<u64> },
}

/// Decide whether `targets` can be promoted against `voters`.
///
/// `matched` reports how far a node has replicated, or `None` when it has
/// replicated nothing. `None` is *infinite* lag, never zero: promoting a node
/// that promised nothing on an idle cluster is what removed a voter's ability
/// to answer for its own data.
pub(crate) fn decide_promotion(
    targets: &[u64],
    voters: &std::collections::BTreeSet<u64>,
    last_log: u64,
    matched: impl Fn(u64) -> Option<u64>,
) -> PromotionDecision {
    let mut lag_of: Option<u64> = Some(0);
    let mut blocked: Option<u64> = None;
    for &node in targets {
        if voters.contains(&node) {
            continue;
        }
        match matched(node) {
            None => return PromotionDecision::Skip { node, lag: None },
            Some(matched) => {
                let lag = last_log.saturating_sub(matched);
                lag_of = Some(lag_of.map_or(lag, |prev| prev.max(lag)));
                if lag > PROMOTION_LAG_THRESHOLD {
                    blocked = Some(node);
                }
            }
        }
    }
    match blocked {
        Some(node) => PromotionDecision::Skip { node, lag: lag_of },
        None => PromotionDecision::Promote { lag: lag_of },
    }
}

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
        let group_label = format!("shard:{}", shard.0);
        let lag_of = match decide_promotion(
            &target_owners,
            &current_voters,
            m.last_log_index.unwrap_or(0),
            |node| crate::replication_matched(&m, &node),
        ) {
            PromotionDecision::Skip { node, lag } => {
                record_promotion_attempt(shards, &group_label, node, lag, "skipped_lag");
                continue;
            }
            PromotionDecision::Promote { lag } => lag,
        };

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
        match decide_promotion(&[learner], &current_voters, last_log, |node| {
            crate::replication_matched(&m, &node)
        }) {
            PromotionDecision::Skip { lag, .. } => {
                record_promotion_attempt(shards, "metadata", learner, lag, "skipped_lag");
                // Falls through to 2b: shard reconciliation must not starve
                // while a learner waits for its next attempt.
            }
            PromotionDecision::Promote { lag } => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn voters(ids: &[u64]) -> BTreeSet<u64> {
        ids.iter().copied().collect()
    }

    #[test]
    fn a_learner_within_the_threshold_is_promoted() {
        let decision = decide_promotion(&[1, 2], &voters(&[1]), 1_000, |node| {
            (node == 2).then_some(1_000 - PROMOTION_LAG_THRESHOLD)
        });
        assert_eq!(
            decision,
            PromotionDecision::Promote {
                lag: Some(PROMOTION_LAG_THRESHOLD)
            },
            "exactly at the threshold is caught up"
        );
    }

    #[test]
    fn one_entry_behind_the_threshold_skips() {
        let decision = decide_promotion(&[1, 2], &voters(&[1]), 1_000, |node| {
            (node == 2).then_some(1_000 - PROMOTION_LAG_THRESHOLD - 1)
        });
        assert_eq!(
            decision,
            PromotionDecision::Skip {
                node: 2,
                lag: Some(PROMOTION_LAG_THRESHOLD + 1)
            }
        );
    }

    #[test]
    fn a_learner_that_never_replicated_is_infinite_lag_not_zero() {
        let decision = decide_promotion(&[1, 2], &voters(&[1]), 1_000, |_| None);
        assert_eq!(
            decision,
            PromotionDecision::Skip { node: 2, lag: None },
            "an idle cluster must not promote a node with no replicated entries"
        );
    }

    #[test]
    fn already_matching_voters_promote_with_zero_lag() {
        let decision = decide_promotion(&[1, 2], &voters(&[1, 2]), 1_000, |_| None);
        assert_eq!(decision, PromotionDecision::Promote { lag: Some(0) });
    }

    #[test]
    fn several_behind_learners_report_the_last_and_the_maximum_lag() {
        let decision = decide_promotion(&[1, 2, 3], &voters(&[1]), 1_000, |node| match node {
            2 => Some(900),
            3 => Some(800),
            _ => None,
        });
        assert_eq!(
            decision,
            PromotionDecision::Skip {
                node: 3,
                lag: Some(200)
            }
        );
    }

    #[test]
    fn one_caught_up_and_one_behind_still_skips() {
        let decision = decide_promotion(&[1, 2, 3], &voters(&[1]), 1_000, |node| match node {
            2 => Some(1_000),
            3 => Some(100),
            _ => None,
        });
        assert_eq!(
            decision,
            PromotionDecision::Skip {
                node: 3,
                lag: Some(900)
            },
            "a caught-up sibling does not make the behind learner a voter"
        );
    }
}
