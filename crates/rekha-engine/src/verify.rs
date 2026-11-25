//! Read-only integrity check of a data directory.
//!
//! `rekha verify` answers one question before an operator does anything
//! irreversible: **is what is on this disk the thing the metadata says it is.**
//! It opens nothing for writing, repairs nothing, and reports every problem it
//! finds rather than the first, because the useful answer to "is this disk
//! healthy" is the full list.
//!
//! # What it checks
//!
//! For every `<collection>.checkpoint` directory:
//!
//! - `CURRENT` names a generation that exists ([`FindingClass::StaleCurrent`]);
//! - `checkpoint.json` parses and declares the version this build writes
//!   ([`FindingClass::BadManifest`]);
//! - the generation belongs to the collection it sits under, so a copied
//!   directory is not silently adopted ([`FindingClass::ForeignGeneration`]);
//! - every inventoried file exists, has the recorded length, and hashes to the
//!   recorded CRC ([`FindingClass::MissingFile`], [`FindingClass::WrongLength`],
//!   [`FindingClass::ChecksumMismatch`]);
//! - `records.bin` holds exactly the number of records the manifest declares.
//!
//! For every `<collection>.wal`:
//!
//! - the batch chain verifies ([`FindingClass::WalCorrupt`]);
//! - an incomplete final batch is a warning, because recovery discards it
//!   ([`FindingClass::WalTruncated`]);
//! - the log does not start after the checkpoint's replay boundary, which would
//!   mean the records in between exist nowhere ([`FindingClass::WalHole`]).
//!
//! # What it cannot check
//!
//! The catalog. redb takes an exclusive lock, so verifying a *running* node
//! would either fail to open it or have to skip it. Cross-checking the registry
//! against the files is a backup-time concern, and is named here rather than
//! silently omitted.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rekha_core::types::Id;

use crate::Record;
use crate::collection::{
    CHECKPOINT_FORMAT_VERSION, CURRENT_POINTER, CheckpointFile, CheckpointMeta, GENERATIONS_DIR,
    active_generation,
};

/// How bad a finding is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Data is missing, wrong, or unreadable. The node would refuse to open the
    /// collection, or open a smaller one than was acknowledged.
    Error,
    /// Recovery handles this without loss — an incomplete final batch, which is
    /// discarded. Worth reporting; not a reason to refuse to start.
    Warning,
}

/// One class of problem, named so a test can assert on it rather than on prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingClass {
    /// `CURRENT` is missing, empty, escapes the directory, or names a
    /// generation that is not there.
    StaleCurrent,
    /// `checkpoint.json` is missing, unparseable, or a format this build cannot
    /// read.
    BadManifest,
    /// The generation's manifest names a different collection.
    ForeignGeneration,
    /// A file the manifest inventories is not on disk.
    MissingFile,
    /// A file's length disagrees with the manifest.
    WrongLength,
    /// A file's CRC-32 disagrees with the manifest.
    ChecksumMismatch,
    /// `records.bin` does not deserialize to the declared record count.
    RecordCountMismatch,
    /// The WAL's batch chain is broken with data after it: not a tail.
    WalCorrupt,
    /// The WAL's final batch is incomplete. Recovery discards it.
    WalTruncated,
    /// The WAL starts after the checkpoint's replay boundary, so the records in
    /// between exist nowhere.
    WalHole,
}

impl FindingClass {
    /// Severity this class carries.
    pub fn severity(self) -> Severity {
        match self {
            Self::WalTruncated => Severity::Warning,
            Self::StaleCurrent
            | Self::BadManifest
            | Self::ForeignGeneration
            | Self::MissingFile
            | Self::WrongLength
            | Self::ChecksumMismatch
            | Self::RecordCountMismatch
            | Self::WalCorrupt
            | Self::WalHole => Severity::Error,
        }
    }

    /// Stable name, for `--json` and for tests.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StaleCurrent => "stale_current",
            Self::BadManifest => "bad_manifest",
            Self::ForeignGeneration => "foreign_generation",
            Self::MissingFile => "missing_file",
            Self::WrongLength => "wrong_length",
            Self::ChecksumMismatch => "checksum_mismatch",
            Self::RecordCountMismatch => "record_count_mismatch",
            Self::WalCorrupt => "wal_corrupt",
            Self::WalTruncated => "wal_truncated",
            Self::WalHole => "wal_hole",
        }
    }
}

/// One problem found on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// Collection id the damage belongs to.
    pub scope: String,
    /// What kind of problem.
    pub class: FindingClass,
    /// Human-readable detail, including the file and the numbers involved.
    pub detail: String,
}

/// Result of verifying one data directory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VerifyReport {
    /// Collections that have any data on disk.
    pub collections: usize,
    /// WAL records verified across every collection.
    pub wal_records: u64,
    /// Every problem found, ordered by collection id.
    pub findings: Vec<Finding>,
}

impl VerifyReport {
    /// Whether the directory is safe to open: no errors, warnings allowed.
    pub fn is_clean(&self) -> bool {
        self.count(Severity::Error) == 0
    }

    /// Number of findings at `severity`.
    pub fn count(&self, severity: Severity) -> usize {
        self.findings
            .iter()
            .filter(|f| f.class.severity() == severity)
            .count()
    }
}

/// The files one collection may have.
#[derive(Default)]
struct CollectionFiles {
    checkpoint: Option<PathBuf>,
    /// Presence only: `Wal::inspect` takes the directory and reports whether the
    /// file exists, so the path itself carries no extra information.
    has_wal: bool,
}

/// Verify every collection under `data_dir`.
///
/// Read-only. An absent `<data_dir>/wal` is an empty database, not an error: a
/// node that has never written anything has nothing to be wrong.
///
/// # Errors
///
/// Returns an error only when the directory cannot be listed. Damage *inside* a
/// collection is a [`Finding`], not an error — the caller decides what to do
/// with it.
pub fn verify_data_dir(data_dir: &Path) -> std::io::Result<VerifyReport> {
    // A missing *data directory* is a wrong path, not an empty database. The
    // old behaviour reported it as clean with zero collections, so a typo in
    // `--data-dir` produced a green check on data that was never read.
    if !data_dir.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("data directory {} does not exist", data_dir.display()),
        ));
    }
    let wal_dir = data_dir.join("wal");
    let entries = match std::fs::read_dir(&wal_dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(VerifyReport::default());
        }
        Err(e) => return Err(e),
    };

    // Both files for one collection are checked together: the log's first
    // sequence only means "hole" relative to that collection's checkpoint.
    let mut by_id: BTreeMap<String, CollectionFiles> = BTreeMap::new();
    for entry in entries {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()).map(str::to_owned) else {
            continue;
        };
        if let Some(id) = name.strip_suffix(".checkpoint") {
            by_id.entry(id.to_owned()).or_default().checkpoint = Some(path);
        } else if let Some(id) = name.strip_suffix(".wal") {
            by_id.entry(id.to_owned()).or_default().has_wal = true;
        }
    }

    let mut report = VerifyReport {
        collections: by_id.len(),
        ..Default::default()
    };
    for (id, files) in by_id {
        verify_one(&wal_dir, &id, &files, &mut report);
    }
    Ok(report)
}

/// Verify one collection: its published generation, its log, and the boundary
/// between them.
fn verify_one(wal_dir: &Path, id: &str, files: &CollectionFiles, report: &mut VerifyReport) {
    let boundary = files
        .checkpoint
        .as_deref()
        .and_then(|base| verify_checkpoint(base, id, report));
    let inspected = files
        .has_wal
        .then(|| verify_wal(wal_dir, id, report))
        .flatten();

    // The generation covers operations up to `snapshot_seq`; the log must resume
    // at `snapshot_seq + 1`. A log that starts later means the batches in
    // between were pruned or lost, and nothing can reconstruct them.
    if let (Some(snapshot_seq), Some(inspection)) = (boundary, inspected)
        && inspection.exists
        && inspection.record_count > 0
        && inspection.first_seq > snapshot_seq + 1
    {
        report.findings.push(Finding {
            scope: id.to_owned(),
            class: FindingClass::WalHole,
            detail: format!(
                "checkpoint covers seq {snapshot_seq} but the log starts at {}; \
                 the records in between are gone",
                inspection.first_seq
            ),
        });
    }
}

/// Check one collection's published generation, returning its replay boundary.
fn verify_checkpoint(base: &Path, id: &str, report: &mut VerifyReport) -> Option<u64> {
    let generations = base.join(GENERATIONS_DIR);
    let dir = match active_generation(base) {
        Some(dir) => dir,
        None => {
            // No pointer and no generations means "never checkpointed", which is
            // normal. Generations with no usable pointer means the publication
            // was interrupted or the pointer was damaged: the data is there and
            // unreachable.
            let has_generations = std::fs::read_dir(&generations)
                .map(|mut it| it.next().is_some())
                .unwrap_or(false);
            if base.join(CURRENT_POINTER).exists() || has_generations {
                report.findings.push(Finding {
                    scope: id.to_owned(),
                    class: FindingClass::StaleCurrent,
                    detail: format!(
                        "`{CURRENT_POINTER}` does not name a readable generation under {}",
                        generations.display()
                    ),
                });
            }
            return None;
        }
    };

    let manifest = dir.join("checkpoint.json");
    let meta: CheckpointMeta = match std::fs::read(&manifest) {
        Ok(bytes) => match serde_json::from_slice(&bytes) {
            Ok(meta) => meta,
            Err(e) => {
                report.findings.push(Finding {
                    scope: id.to_owned(),
                    class: FindingClass::BadManifest,
                    detail: format!("{}: {e}", manifest.display()),
                });
                return None;
            }
        },
        Err(e) => {
            report.findings.push(Finding {
                scope: id.to_owned(),
                class: FindingClass::BadManifest,
                detail: format!("{}: {e}", manifest.display()),
            });
            return None;
        }
    };

    if meta.format_version != CHECKPOINT_FORMAT_VERSION {
        report.findings.push(Finding {
            scope: id.to_owned(),
            class: FindingClass::BadManifest,
            detail: format!(
                "format version {} but this build writes {CHECKPOINT_FORMAT_VERSION}",
                meta.format_version
            ),
        });
        return None;
    }

    let expected_collection_id = id
        .split_once(".shard-")
        .map(|(coll_id, _)| coll_id)
        .unwrap_or(id);
    if meta.collection_id != expected_collection_id {
        report.findings.push(Finding {
            scope: id.to_owned(),
            class: FindingClass::ForeignGeneration,
            detail: format!(
                "generation belongs to collection {}, and sits under {id}",
                meta.collection_id
            ),
        });
        return None;
    }

    verify_files(&dir, id, &meta.files, report);
    verify_record_count(&dir, id, meta.record_count, report);
    Some(meta.snapshot_seq)
}

/// Verify every file the manifest inventories.
fn verify_files(dir: &Path, id: &str, files: &[CheckpointFile], report: &mut VerifyReport) {
    for entry in files {
        if entry.path.is_empty()
            || entry.path.starts_with('/')
            || entry.path.contains("..")
            || entry.path.contains('\\')
        {
            report.findings.push(Finding {
                scope: id.to_owned(),
                class: FindingClass::BadManifest,
                detail: format!("manifest lists an unsafe path `{}`", entry.path),
            });
            continue;
        }
        let path = dir.join(&entry.path);
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(e) => {
                report.findings.push(Finding {
                    scope: id.to_owned(),
                    class: FindingClass::MissingFile,
                    detail: format!("`{}`: {e}", entry.path),
                });
                continue;
            }
        };
        if metadata.len() != entry.len {
            report.findings.push(Finding {
                scope: id.to_owned(),
                class: FindingClass::WrongLength,
                detail: format!(
                    "`{}` is {} bytes, manifest declares {}",
                    entry.path,
                    metadata.len(),
                    entry.len
                ),
            });
            continue;
        }
        // No checksum means the file carries its own integrity check or is
        // derived and rebuildable.
        let Some(expected) = entry.crc32 else {
            // A vector segment is the first case, and its check lives in the
            // segment header — which only the segment reader validates. Not
            // opening it here would let `verify` report a directory clean and the
            // node then refuse to open the collection, which is the opposite of
            // what this command is for.
            if entry.path.starts_with("segments/")
                && let Err(e) = crate::segment::Segment::open(&path)
            {
                report.findings.push(Finding {
                    scope: id.to_owned(),
                    class: FindingClass::ChecksumMismatch,
                    detail: format!("`{}`: {e}", entry.path),
                });
            }
            // `index.bin` is the second case and is deliberately not
            // inventoried: a lost graph must not reject a generation whose
            // authoritative data is intact, and recovery rebuilds it.
            continue;
        };
        match std::fs::read(&path) {
            Ok(bytes) => {
                let actual = crc32fast::hash(&bytes);
                if actual != expected {
                    report.findings.push(Finding {
                        scope: id.to_owned(),
                        class: FindingClass::ChecksumMismatch,
                        detail: format!(
                            "`{}`: expected {expected:#010x}, got {actual:#010x}",
                            entry.path
                        ),
                    });
                }
            }
            Err(e) => report.findings.push(Finding {
                scope: id.to_owned(),
                class: FindingClass::MissingFile,
                detail: format!("`{}`: {e}", entry.path),
            }),
        }
    }
}

/// `records.bin` must deserialize to the count the manifest declares.
///
/// The CRC covers the bytes, not their meaning: a valid `records.bin` from a
/// different snapshot satisfies it. The count is the cheap statement that the
/// two still describe the same generation.
fn verify_record_count(dir: &Path, id: &str, declared: u64, report: &mut VerifyReport) {
    let Ok(bytes) = std::fs::read(dir.join("records.bin")) else {
        return; // Already reported as a missing file.
    };
    let Ok(records) = bincode::deserialize::<std::collections::HashMap<Id, Record>>(&bytes) else {
        report.findings.push(Finding {
            scope: id.to_owned(),
            class: FindingClass::RecordCountMismatch,
            detail: "records.bin does not deserialize".to_owned(),
        });
        return;
    };
    if records.len() as u64 != declared {
        report.findings.push(Finding {
            scope: id.to_owned(),
            class: FindingClass::RecordCountMismatch,
            detail: format!(
                "records.bin holds {} records, manifest declares {declared}",
                records.len()
            ),
        });
    }
}

/// Check one collection's write-ahead log, returning its inspection for the
/// boundary check.
fn verify_wal(
    wal_dir: &Path,
    id: &str,
    report: &mut VerifyReport,
) -> Option<rekha_wal::WalInspection> {
    // A `.wal` whose name is not a collection id or partitioned id
    // (`{uuid}.shard-{n}`) is not this engine's file.
    let base = id
        .split_once(".shard-")
        .map(|(u, s)| s.parse::<u32>().is_ok().then_some(u));
    let is_valid = match base {
        Some(Some(u)) => u.parse::<uuid::Uuid>().is_ok(),
        Some(None) => false,
        None => id.parse::<uuid::Uuid>().is_ok(),
    };
    if !is_valid {
        return None;
    }

    let inspection = match rekha_wal::Wal::inspect_named(wal_dir, id) {
        Ok(inspection) => inspection,
        Err(e) => {
            report.findings.push(Finding {
                scope: id.to_owned(),
                class: FindingClass::WalCorrupt,
                detail: format!("{}: {e}", wal_dir.join(format!("{id}.wal")).display()),
            });
            return None;
        }
    };
    if !inspection.exists {
        return None;
    }
    report.wal_records += inspection.record_count;

    match inspection.tail {
        rekha_wal::WalTail::Clean => {}
        rekha_wal::WalTail::Suspect { reason, at_seq } => {
            report.findings.push(Finding {
                scope: id.to_owned(),
                class: FindingClass::WalTruncated,
                detail: format!(
                    "incomplete final batch at seq {at_seq} ({}); {} of {} bytes verified, \
                     recovery discards the rest",
                    reason.describe(),
                    inspection.valid_len,
                    inspection.file_len
                ),
            });
        }
        rekha_wal::WalTail::Corrupt { reason, at_offset } => {
            report.findings.push(Finding {
                scope: id.to_owned(),
                class: FindingClass::WalCorrupt,
                detail: format!(
                    "damage at offset {at_offset} ({}), followed by more data",
                    reason.describe()
                ),
            });
        }
    }
    Some(inspection)
}
