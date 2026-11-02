#![deny(clippy::unwrap_used, clippy::expect_used)]
//! Benchmark harness for RekhaDB.
//!
//! Measures what the durability and checkpoint design actually costs, instead
//! of assuming it. Two questions it exists to answer:
//!
//! 1. How much does [`Durability::Durable`] cost relative to
//!    [`Durability::Relaxed`] at batch sizes that matter?
//! 2. Where does a checkpoint spend its time, and how does that grow with the
//!    size of the collection?
//!
//! The second question is the one that produced the current roadmap: a
//! checkpoint is whole-corpus work, so its cost grows with the collection, not
//! with the number of changed records.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use clap::Parser;
use rekha_core::config::{CollectionConfig, HnswConfig};
use rekha_core::types::{Distance, Embedding};
use rekha_engine::{Durability, Engine, EngineConfig};
use rekha_storage::{Catalog, LocalStorage, RedbCatalog, Storage};

#[derive(Parser, Debug)]
#[command(
    name = "rekha-bench",
    about = "Measure RekhaDB's write path, durability cost, and checkpoint phases"
)]
struct Args {
    /// Directory for the catalog, WAL, and segments. Created if absent.
    #[arg(long)]
    data_dir: PathBuf,
    /// What a successful write promises about surviving a crash.
    #[arg(long, default_value = "durable")]
    durability: Durability,
    /// Total records to ingest across all collections.
    #[arg(long, default_value_t = 20_000)]
    records: usize,
    /// Embedding width.
    #[arg(long, default_value_t = 128)]
    dim: usize,
    /// Records per `add` call.
    #[arg(long, default_value_t = 100)]
    batch: usize,
    /// Concurrent writer threads.
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Number of collections to spread writes across.
    #[arg(long, default_value_t = 1)]
    collections: usize,
    /// Distance space: l2, cosine, or ip.
    #[arg(long, default_value = "l2")]
    space: String,
    /// `hnsw.sync_threshold`: checkpoint after this many writes. Zero disables
    /// automatic checkpoints so WAL cost can be isolated from checkpoint cost.
    #[arg(long, default_value_t = 1000)]
    sync_threshold: usize,
    /// Skip the post-ingest `optimize` (and therefore the checkpoint phase
    /// breakdown), measuring only the ingest path.
    #[arg(long)]
    skip_optimize: bool,
    /// Write the JSON result here instead of stdout.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("rekha-bench: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.records == 0 || args.batch == 0 || args.threads == 0 || args.collections == 0 {
        return Err("records, batch, threads and collections must all be non-zero".into());
    }

    std::fs::create_dir_all(&args.data_dir)?;
    let space = match args.space.as_str() {
        "l2" => Distance::L2,
        "cosine" => Distance::Cosine,
        "ip" => Distance::Ip,
        other => return Err(format!("unknown space `{other}` (expected l2, cosine, ip)").into()),
    };

    let catalog: Arc<dyn Catalog> =
        Arc::new(RedbCatalog::open(args.data_dir.join("catalog.redb"))?);
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(args.data_dir.join("objects")));
    let engine = Arc::new(Engine::open(
        catalog,
        storage,
        args.data_dir.join("wal"),
        EngineConfig {
            durability: args.durability,
            ..EngineConfig::default()
        },
    )?);

    let hnsw = HnswConfig {
        sync_threshold: args.sync_threshold,
        ..HnswConfig::default()
    };

    let mut ids = Vec::with_capacity(args.collections);
    for c in 0..args.collections {
        let mut config = CollectionConfig::new(format!("bench-{c}"), args.dim, space);
        config.hnsw = hnsw.clone();
        ids.push(engine.create_collection(&config)?.config.id);
    }

    let ingest_started = Instant::now();
    let next = AtomicUsize::new(0);
    let latencies: Vec<Vec<u64>> = std::thread::scope(|scope| -> Result<Vec<Vec<u64>>, String> {
        let mut handles = Vec::with_capacity(args.threads);
        for _ in 0..args.threads {
            let engine = Arc::clone(&engine);
            let ids = &ids;
            let next = &next;
            handles.push(scope.spawn(move || -> Result<Vec<u64>, String> {
                let mut samples = Vec::new();
                loop {
                    let start = next.fetch_add(args.batch, Ordering::Relaxed);
                    if start >= args.records {
                        break;
                    }
                    let n = args.batch.min(args.records - start);
                    let collection = &ids[start % ids.len()];
                    let batch_ids: Vec<String> = (start..start + n).map(ids_of).collect();
                    let embeddings: Vec<Embedding> = (start..start + n)
                        .map(|i| {
                            Embedding::from(
                                (0..args.dim)
                                    .map(|d| (((i * 31 + d * 17) % 997) as f32) / 997.0)
                                    .collect::<Vec<f32>>(),
                            )
                        })
                        .collect();
                    let t = Instant::now();
                    // A failed write does not produce a slow sample, it
                    // produces no sample at all: abort the run rather than
                    // report a latency distribution for a workload that did
                    // not happen.
                    engine
                        .add(collection, &batch_ids, &embeddings, None, None)
                        .map_err(|e| format!("ingest write failed: {e}"))?;
                    samples.push(t.elapsed().as_micros() as u64);
                }
                Ok(samples)
            }));
        }
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| "an ingest thread panicked".to_string())?
            })
            .collect()
    })?;

    let ingest_ms = ingest_started.elapsed().as_millis() as u64;
    let mut samples: Vec<u64> = latencies.into_iter().flatten().collect();
    samples.sort_unstable();

    let mut report = BTreeMap::new();
    report.insert("durability", serde_json::json!(args.durability.to_string()));
    report.insert("records", serde_json::json!(args.records));
    report.insert("dim", serde_json::json!(args.dim));
    report.insert("batch", serde_json::json!(args.batch));
    report.insert("threads", serde_json::json!(args.threads));
    report.insert("collections", serde_json::json!(args.collections));
    report.insert("sync_threshold", serde_json::json!(args.sync_threshold));
    report.insert(
        "writes",
        serde_json::json!((args.records as f64 / args.batch as f64).ceil() as u64),
    );
    report.insert("syncs", serde_json::json!(engine.sync_count()));
    report.insert("ingest_ms", serde_json::json!(ingest_ms));
    report.insert(
        "records_per_sec",
        serde_json::json!((args.records as f64 / (ingest_ms as f64 / 1000.0)).round() as u64),
    );
    report.insert(
        "latency_us",
        serde_json::json!({
            "p50": percentile(&samples, 0.50),
            "p95": percentile(&samples, 0.95),
            "p99": percentile(&samples, 0.99),
            "max": samples.last().copied().unwrap_or(0),
        }),
    );

    // A routine checkpoint with nothing new shows the fixed per-checkpoint cost
    // (`records.bin`, the graph, the manifest). Then one batch of new records
    // shows the incremental cost of the vectors themselves.
    let mut checkpoints = Vec::new();
    for id in &ids {
        engine.checkpoint(id)?;
        let idle = engine.last_checkpoint_timings(id)?;
        let start = samples_total(&engine, id)?;
        let batch_ids: Vec<String> = (0..args.batch).map(|i| ids_of(start + i)).collect();
        let embeddings: Vec<Embedding> = (0..args.batch)
            .map(|i| {
                let n = start + i;
                Embedding::from(
                    (0..args.dim)
                        .map(|d| (((n * 31 + d * 17) % 997) as f32) / 997.0)
                        .collect::<Vec<f32>>(),
                )
            })
            .collect();
        engine.add(id, &batch_ids, &embeddings, None, None)?;
        engine.checkpoint(id)?;
        let after = engine.last_checkpoint_timings(id)?;
        checkpoints.push(serde_json::json!({
            "idle": phases(&idle),
            "after_writes": phases(&after),
        }));
    }
    report.insert("checkpoints", serde_json::json!(checkpoints));

    let mut optimizes = Vec::new();
    if !args.skip_optimize {
        for id in &ids {
            let stats = engine.optimize(id)?;
            optimizes.push(serde_json::json!({
                "records": stats.records,
                "indexed": stats.indexed,
                "flush_us": stats.flush_us,
                "rebuild_us": stats.rebuild_us,
                "checkpoint": {
                    "records_us": stats.checkpoint.records_us,
                    "index_us": stats.checkpoint.index_us,
                    "vectors_us": stats.checkpoint.vectors_us,
                    "index_map_us": stats.checkpoint.index_map_us,
                    "manifest_us": stats.checkpoint.manifest_us,
                    "validate_us": stats.checkpoint.validate_us,
                    "publish_us": stats.checkpoint.publish_us,
                    "cleanup_us": stats.checkpoint.cleanup_us,
                    "total_us": stats.checkpoint.total_us,
                },
            }));
        }
    }
    report.insert("optimize", serde_json::json!(optimizes));
    report.insert(
        "data_dir_bytes",
        serde_json::json!(dir_size(&args.data_dir)),
    );

    let json = serde_json::to_string_pretty(&report)?;
    match args.out {
        Some(path) => std::fs::write(&path, &json)?,
        None => println!("{json}"),
    }
    Ok(())
}

fn phases(t: &rekha_engine::CheckpointTimings) -> serde_json::Value {
    serde_json::json!({
        "records_us": t.records_us,
        "index_us": t.index_us,
        "vectors_us": t.vectors_us,
        "index_map_us": t.index_map_us,
        "manifest_us": t.manifest_us,
        "validate_us": t.validate_us,
        "publish_us": t.publish_us,
        "cleanup_us": t.cleanup_us,
        "total_us": t.total_us,
    })
}

/// Total live records, used to keep the extra checkpoint batches distinct.
fn samples_total(engine: &Engine, id: &uuid::Uuid) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(engine.count(id)? as usize)
}

fn ids_of(i: usize) -> String {
    format!("id-{i:012}")
}

fn percentile(sorted: &[u64], q: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0;
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            total += dir_size(&entry.path());
        } else {
            total += meta.len();
        }
    }
    total
}
