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

    for shard in shards.shards() {
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

        // Check if all target owners have caught up within 50 entries
        let shard_last_log = m.last_log_index.unwrap_or(0);
        let all_caught_up = target_owners.iter().all(|&node| {
            if current_voters.contains(&node) {
                return true;
            }
            let matched = m
                .replication
                .as_ref()
                .and_then(|r| r.get(&node))
                .and_then(|id| id.map(|l| l.index))
                .unwrap_or(0);
            matched >= shard_last_log.saturating_sub(50)
        });

        if all_caught_up
            && shard_raft
                .change_membership(target_set, false)
                .await
                .is_ok()
        {
            tracing::info!(
                shard = shard.0,
                ?target_owners,
                "shard leader transitioned membership to target owners"
            );
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
    let learners: Vec<u64> = m.membership_config.membership().learner_ids().collect();

    // 2a. Learner auto-promotion in the metadata group
    for learner in learners {
        let matched = m
            .replication
            .as_ref()
            .and_then(|r| r.get(&learner))
            .and_then(|id| id.map(|l| l.index))
            .unwrap_or(0);

        // Once learner caught up within 50 entries of last log, promote to voter
        if matched >= last_log.saturating_sub(50) {
            let mut new_voters = current_voters.clone();
            new_voters.insert(learner);
            if raft
                .change_membership(new_voters.clone(), true)
                .await
                .is_ok()
            {
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
                    let mut req = http.post(format!(
                        "{url}/internal/cluster/shard/{}/ensure_started",
                        shard.0
                    ));
                    if let Some(secret) = cluster_secret {
                        req = req.header("x-rekha-cluster-secret", secret);
                    }
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
                        let mut req = http.get(format!(
                            "{url}/internal/cluster/shard/{}/membership",
                            shard.0
                        ));
                        if let Some(secret) = cluster_secret {
                            req = req.header("x-rekha-cluster-secret", secret);
                        }
                        if let Ok(resp) = req.send().await
                            && let Ok(voters) = resp.json::<Vec<u64>>().await
                        {
                            let v_set: std::collections::BTreeSet<u64> =
                                voters.into_iter().collect();
                            if v_set == target_owners.iter().copied().collect() {
                                match_found = true;
                                break;
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
                        let mut req =
                            http.post(format!("{url}/internal/cluster/shard/{}/drop", shard.0));
                        if let Some(secret) = cluster_secret {
                            req = req.header("x-rekha-cluster-secret", secret);
                        }
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
