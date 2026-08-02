use std::time::Instant;

pub fn record_insert() {
    metrics::counter!("rekha_inserts_total", 1);
}

pub fn record_search() {
    metrics::counter!("rekha_searches_total", 1);
}

pub fn record_delete() {
    metrics::counter!("rekha_deletes_total", 1);
}

pub fn record_latency(op: &str, secs: f64) {
    metrics::histogram!("rekha_latency_seconds", secs, "op" => op.to_string());
}

pub fn set_peer_count(n: f64) {
    metrics::gauge!("rekha_peers", n);
}

pub fn set_hint_backlog(n: f64) {
    metrics::gauge!("rekha_hint_backlog", n);
}

pub struct TimerGuard {
    op: &'static str,
    start: Instant,
}

impl TimerGuard {
    pub fn new(op: &'static str) -> Self {
        TimerGuard {
            op,
            start: Instant::now(),
        }
    }
}

impl Drop for TimerGuard {
    fn drop(&mut self) {
        record_latency(self.op, self.start.elapsed().as_secs_f64());
    }
}
