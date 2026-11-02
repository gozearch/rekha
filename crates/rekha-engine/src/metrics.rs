//! Process-wide instrumentation.
//!
//! The server installs a Prometheus recorder when `REKHA_METRICS_PORT` is set.
//! Before this module existed the exporter was installed but nothing was ever
//! recorded, so `/metrics` served an empty registry: every scrape succeeded and
//! said nothing.
//!
//! Recording is cheap and a no-op when no recorder is installed, so library
//! users and tests pay nothing for it.

use crate::CheckpointTimings;

/// Record one checkpoint's phase breakdown and outcome.
pub(crate) fn record_checkpoint(timings: &CheckpointTimings, ok: bool) {
    for (phase, micros) in timings.phases() {
        metrics::histogram!("rekha_checkpoint_phase_us", "phase" => phase).record(micros as f64);
    }
    metrics::histogram!("rekha_checkpoint_total_us").record(timings.total_us as f64);
    metrics::counter!(
        "rekha_checkpoint_total",
        "outcome" => if ok { "ok" } else { "failed" }
    )
    .increment(1);
}

/// Record a committed write batch.
pub(crate) fn record_write(ops: usize, wal_bytes: u64, records: usize, synced: bool) {
    metrics::counter!("rekha_wal_ops_total").increment(ops as u64);
    metrics::counter!("rekha_write_batches_total").increment(1);
    if synced {
        metrics::counter!("rekha_wal_syncs_total").increment(1);
    }
    metrics::gauge!("rekha_wal_bytes").set(wal_bytes as f64);
    metrics::gauge!("rekha_records").set(records as f64);
}

/// Record a checkpoint that compacted instead of appending.
pub(crate) fn record_compaction() {
    metrics::counter!("rekha_checkpoint_compactions_total").increment(1);
}
