//! Automatic cluster reconciler and rebalancer.
//!
//! Runs on the metadata group leader and monitors cluster membership, learner
//! catch-up progress, dead-node timeouts, and shard owner distribution.

use std::sync::Arc;
use std::time::Duration;

use openraft::Raft;

use crate::identity::PeerDirectory;
use crate::placement::{Placement, ShardId};
use crate::raft_types::{ClusterOperation, RaftTypeConfig};
use crate::shard::ShardCluster;

/// Execute one reconciliation tick on the metadata leader.
pub async fn run_rebalancer_tick(
    node_id: u64,
    raft: &Arc<Raft<RaftTypeConfig>>,
    shards: &Arc<ShardCluster>,
    peers: &PeerDirectory,
    http: &reqwest::Client,
    cluster_secret: Option<&str>,
) {
    let m = raft.metrics().borrow().clone();
    if !m.state.is_leader() {
        return;
    }

    let last_log = m.last_log_index.unwrap_or(0);
    let current_voters: std::collections::BTreeSet<u64> =
        m.membership_config.membership().voter_ids().collect();
    let learners: Vec<u64> = m.membership_config.membership().learner_ids().collect();

    // 1. Learner auto-promotion in the metadata group
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
                let _ = raft
                    .client_write(ClusterOperation::SetVoters {
                        voters: voter_vec.clone(),
                    })
                    .await;
                let current_p = shards.placement();
                if let Ok(new_p) = Placement::new(
                    current_p.shard_count(),
                    current_p.replication_factor(),
                    voter_vec.iter().copied(),
                ) {
                    shards.set_placement(Arc::new(new_p));
                }
            }
        }
    }

    // 2. Per-shard owner reconciliation
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
            // Reconcile shard s to target_owners
            for &new_node in &target_owners {
                if !current_owners.contains(&new_node) {
                    // Instruct new_node to ensure shard group is started
                    if let Some(url) = peers.url(new_node) {
                        let mut req = http.post(format!(
                            "{url}/internal/cluster/shard/{}/ensure_started",
                            shard.0
                        ));
                        if let Some(secret) = cluster_secret {
                            req = req.header("x-rekha-cluster-secret", secret);
                        }
                        let _ = req.send().await;
                    }
                    // Add learner to shard Raft group if this node leads shard s
                    if let Some(shard_raft) = shards.raft(shard)
                        && shard_raft.metrics().borrow().current_leader == Some(node_id)
                    {
                        let _ = shard_raft.add_learner(new_node, new_node, false).await;
                    }
                }
            }

            // If this node leads shard s, check if target_owners caught up
            if let Some(shard_raft) = shards.raft(shard)
                && shard_raft.metrics().borrow().current_leader == Some(node_id)
            {
                let sm_metrics = shard_raft.metrics().borrow().clone();
                let shard_last_log = sm_metrics.last_log_index.unwrap_or(0);
                let all_caught_up = target_owners.iter().all(|&node| {
                    if current_owners.contains(&node) {
                        return true;
                    }
                    let matched = sm_metrics
                        .replication
                        .as_ref()
                        .and_then(|r| r.get(&node))
                        .and_then(|id| id.map(|l| l.index))
                        .unwrap_or(0);
                    matched >= shard_last_log.saturating_sub(50)
                });

                if all_caught_up {
                    let new_members: std::collections::BTreeSet<u64> =
                        target_owners.iter().copied().collect();
                    if shard_raft
                        .change_membership(new_members, false)
                        .await
                        .is_ok()
                    {
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
                                let mut req = http
                                    .post(format!("{url}/internal/cluster/shard/{}/drop", shard.0));
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
