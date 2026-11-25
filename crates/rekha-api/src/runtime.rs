use crate::prelude::*;

/// Run blocking engine work on the blocking pool.
///
/// The engine fsyncs WAL batches and runs checkpoint phases synchronously, and
/// queries scan and rank in memory. Those calls share a runtime with Raft,
/// which must send a heartbeat every `heartbeat_interval`. A few hundred
/// milliseconds of fsync on a worker thread delays that heartbeat, followers
/// time out at `election_timeout_max`, and the cluster elects a new leader.
///
/// The failure looks like a consensus bug and is a scheduling decision: under
/// sustained write load the cluster loses leadership precisely because it is
/// doing what it was asked to do. Moving engine work to the blocking pool keeps
/// the async workers free to answer heartbeats.
pub(crate) async fn offload<T, F>(work: F) -> Result<T, EngineError>
where
    F: FnOnce() -> Result<T, EngineError> + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(work).await {
        Ok(result) => result,
        Err(join) => Err(EngineError::Io(std::io::Error::other(format!(
            "engine worker task failed: {join}"
        )))),
    }
}
#[cfg(test)]
mod offload_tests {
    use std::time::{Duration, Instant};

    use super::offload;
    use rekha_engine::EngineError;

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn engine_work_runs_on_a_blocking_thread() {
        let caller = std::thread::current().id();
        let worker = offload(move || Ok(std::thread::current().id()))
            .await
            .unwrap();
        assert_ne!(
            caller, worker,
            "engine work must not run on the async worker thread"
        );
    }

    /// The property that matters: a slow engine call must not stop the runtime
    /// from running other tasks. Raft heartbeats are such a task, and a delayed
    /// heartbeat is an election.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn slow_engine_work_does_not_starve_the_runtime() {
        let mut handles = Vec::new();
        for _ in 0..8 {
            handles.push(tokio::spawn(offload(|| {
                std::thread::sleep(Duration::from_millis(250));
                Ok(())
            })));
        }

        // A heartbeat-shaped timer must still fire on time while those run.
        let started = Instant::now();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(200),
            "a timer was delayed to {elapsed:?}: blocking work is running on the async worker"
        );

        for handle in handles {
            handle.await.unwrap().unwrap();
        }
    }

    #[tokio::test]
    async fn an_engine_error_propagates_unchanged() {
        let err = offload(|| Err::<(), _>(EngineError::CollectionNotFound("x".into())))
            .await
            .unwrap_err();
        assert!(matches!(err, EngineError::CollectionNotFound(_)));
    }
}
