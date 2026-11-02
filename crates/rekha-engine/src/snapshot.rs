//! The replica snapshot archive.
//!
//! A node that falls behind the Raft log cannot be repaired by replaying it, so
//! it is given a snapshot instead. A snapshot must be **complete**: every
//! committed write up to its boundary, including every vector. The previous
//! implementation serialised the in-memory record map, which carries `None`
//! for every embedding already flushed into a segment file — so a restored node
//! had ids, metadata and documents but could not answer a single query
//! correctly.
//!
//! The fix is to stop describing the data and start copying it. A checkpoint
//! generation directory is already self-contained: `manifest.json` lists every
//! authoritative file with its length and CRC, `CURRENT` names the generation,
//! and the loader already validates a staged generation before trusting it. So
//! a snapshot is a framed, checksummed copy of that directory plus the
//! collection's catalog row.
//!
//! Two consequences worth stating:
//!
//! * This archive does **not** know how records or segments are laid out. It
//!   copies bytes. A future storage format does not need a new snapshot format.
//! * The check is doubled on purpose. Each file is verified against the
//!   archive's own CRC on arrival, and then the reconstructed generation is
//!   verified again by the checkpoint loader. Neither check trusts the other.

use std::path::{Path, PathBuf};

use rekha_storage::CollectionRecord;
use serde::{Deserialize, Serialize};

use crate::{EngineError, EngineResult};

/// Magic at the start of an archive.
pub(crate) const MAGIC: &[u8; 4] = b"RKSN";
/// Framing version. Bumped on any change to the layout below.
pub(crate) const VERSION: u32 = 1;
/// Magic at the end. Distinguishes a truncated archive from a valid one.
const TRAILER: &[u8; 4] = b"RKSE";

/// Bytes: magic + version + meta length.
const HEADER_LEN: usize = 12;

/// Everything a restore needs that is not raw file bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ArchiveMeta {
    /// Opaque state blob supplied by the caller (the cluster state machine's
    /// own metadata). The engine does not interpret it.
    pub state: Vec<u8>,
    pub collections: Vec<ArchiveCollection>,
}

/// One collection's contribution to the archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ArchiveCollection {
    /// Catalog row: the collection definition plus its sequence watermarks.
    pub record: CollectionRecord,
    /// Generation directory name, as named by `CURRENT`.
    pub generation: String,
    /// Files inside the generation, in the order their bytes appear.
    pub files: Vec<ArchiveFile>,
}

/// A file in the archive, relative to the generation directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ArchiveFile {
    pub path: String,
    pub len: u64,
    pub crc32: u32,
}

/// A decoded archive, borrowing from the input buffer.
#[derive(Debug)]
pub(crate) struct Decoded<'a> {
    pub state: Vec<u8>,
    pub collections: Vec<DecodedCollection<'a>>,
}

#[derive(Debug)]
pub(crate) struct DecodedCollection<'a> {
    pub record: CollectionRecord,
    pub generation: String,
    /// Relative path (owned, because the header is decoded into a local) and
    /// the file's bytes, borrowed from the archive buffer.
    pub files: Vec<(String, &'a [u8])>,
}

/// Build an archive from `state` and a set of `(catalog row, generation dir)`.
///
/// Every collection must already have a published generation. That is not a
/// limitation, it is the invariant that makes the snapshot safe to pair with a
/// log purge: a generation behind the snapshot boundary would leave the
/// follower needing entries that are no longer in the log.
pub(crate) fn encode(
    state: &[u8],
    collections: Vec<(CollectionRecord, PathBuf)>,
) -> EngineResult<Vec<u8>> {
    let mut meta = ArchiveMeta {
        state: state.to_vec(),
        collections: Vec::with_capacity(collections.len()),
    };
    let mut payload: Vec<u8> = Vec::new();

    for (record, generation_dir) in collections {
        let generation = generation_dir
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                EngineError::Serialization(format!(
                    "generation directory {} has no usable name",
                    generation_dir.display()
                ))
            })?
            .to_string();

        let mut paths = Vec::new();
        collect_files(&generation_dir, Path::new(""), &mut paths)?;
        // Deterministic order so the same state produces the same archive.
        paths.sort();

        let mut files = Vec::with_capacity(paths.len());
        for relative in paths {
            let bytes = std::fs::read(generation_dir.join(&relative)).map_err(|e| {
                EngineError::Io(std::io::Error::new(
                    e.kind(),
                    format!("snapshot: reading `{relative}`: {e}"),
                ))
            })?;
            files.push(ArchiveFile {
                path: relative,
                len: bytes.len() as u64,
                crc32: crc32fast::hash(&bytes),
            });
            payload.extend_from_slice(&bytes);
        }

        meta.collections.push(ArchiveCollection {
            record,
            generation,
            files,
        });
    }

    let meta_bytes =
        bincode::serialize(&meta).map_err(|e| EngineError::Serialization(e.to_string()))?;

    let mut out = Vec::with_capacity(HEADER_LEN + meta_bytes.len() + payload.len() + 4);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&meta_bytes);
    out.extend_from_slice(&payload);
    out.extend_from_slice(TRAILER);
    Ok(out)
}

/// Decode and verify an archive.
///
/// Every file's length and CRC is checked before anything is returned, so a
/// caller can never observe a partially-valid archive.
pub(crate) fn decode(archive: &[u8]) -> EngineResult<Decoded<'_>> {
    if archive.len() < HEADER_LEN + 4 {
        return Err(malformed(format!(
            "{} bytes is shorter than the minimum archive",
            archive.len()
        )));
    }
    if &archive[0..4] != MAGIC {
        return Err(malformed(
            "not a snapshot archive (bad magic); this is likely a WAL delta or a \
             snapshot from a different product"
                .into(),
        ));
    }
    let version = u32_at(archive, 4)?;
    if version != VERSION {
        return Err(EngineError::Recovery(format!(
            "snapshot archive is format version {version}, but this build implements \
             version {VERSION}; refusing to guess at its layout"
        )));
    }
    let meta_len = u32_at(archive, 8)? as usize;
    let meta_end = HEADER_LEN
        .checked_add(meta_len)
        .filter(|end| *end + 4 <= archive.len())
        .ok_or_else(|| {
            malformed(format!(
                "declared header length {meta_len} overruns the archive"
            ))
        })?;

    let meta: ArchiveMeta = bincode::deserialize(&archive[HEADER_LEN..meta_end])
        .map_err(|e| malformed(format!("header is unreadable: {e}")))?;

    let body = &archive[meta_end..archive.len() - 4];
    if &archive[archive.len() - 4..] != TRAILER {
        return Err(malformed(
            "archive is truncated: the end marker is missing".into(),
        ));
    }

    let mut collections = Vec::with_capacity(meta.collections.len());
    let mut cursor = 0usize;
    for entry in &meta.collections {
        let mut files = Vec::with_capacity(entry.files.len());
        for file in &entry.files {
            verify_path(&file.path)?;
            let start = cursor;
            let end = start
                .checked_add(file.len as usize)
                .filter(|end| *end <= body.len())
                .ok_or_else(|| {
                    malformed(format!(
                        "`{}` declares {} bytes, which overruns the archive body",
                        file.path, file.len
                    ))
                })?;
            let bytes = &body[start..end];
            let actual = crc32fast::hash(bytes);
            if actual != file.crc32 {
                return Err(malformed(format!(
                    "`{}` checksum mismatch: header records {:#010x}, body holds {:#010x}",
                    file.path, file.crc32, actual
                )));
            }
            files.push((file.path.clone(), bytes));
            cursor = end;
        }
        collections.push(DecodedCollection {
            record: entry.record.clone(),
            generation: entry.generation.clone(),
            files,
        });
    }

    if cursor != body.len() {
        return Err(malformed(format!(
            "archive body is {} bytes but the header accounts for {cursor}",
            body.len()
        )));
    }

    Ok(Decoded {
        state: meta.state,
        collections,
    })
}

fn collect_files(dir: &Path, prefix: &Path, out: &mut Vec<String>) -> EngineResult<()> {
    let entries = std::fs::read_dir(dir).map_err(|e| {
        EngineError::Io(std::io::Error::new(
            e.kind(),
            format!("snapshot: reading {}: {e}", dir.display()),
        ))
    })?;
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let relative = prefix.join(entry.file_name());
        if file_type.is_dir() {
            collect_files(&entry.path(), &relative, out)?;
        } else if file_type.is_file() {
            let relative = relative
                .to_str()
                .ok_or_else(|| {
                    EngineError::Serialization(format!(
                        "file name in {} is not valid UTF-8",
                        entry.path().display()
                    ))
                })?
                .to_string();
            out.push(relative);
        }
        // Symlinks are deliberately skipped: a generation must be a plain tree
        // of regular files, and following links out of it would make the
        // snapshot depend on paths the archive does not describe.
    }
    Ok(())
}

/// Reject a path that could escape the generation directory on restore.
fn verify_path(path: &str) -> EngineResult<()> {
    if path.is_empty()
        || path.starts_with('/')
        || path.starts_with('\\')
        || path.split(['/', '\\']).any(|part| part == "..")
    {
        return Err(malformed(format!("archive lists an unsafe path `{path}`")));
    }
    Ok(())
}

/// Read a little-endian `u32` from a byte buffer.
///
/// Archive bytes arrive from a peer or from a restore file, so a short header is
/// malformed input to report, never a broken invariant to panic on.
fn u32_at(bytes: &[u8], at: usize) -> EngineResult<u32> {
    let slice = bytes
        .get(at..at + 4)
        .ok_or_else(|| malformed(format!("archive header ends before byte {}", at + 4)))?;
    Ok(u32::from_le_bytes(slice.try_into().map_err(|_| {
        malformed("archive header field is not four bytes".into())
    })?))
}

fn malformed(detail: String) -> EngineError {
    EngineError::Serialization(format!("snapshot archive: {detail}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rekha_core::config::CollectionConfig;
    use rekha_core::types::Distance;

    fn record() -> CollectionRecord {
        CollectionRecord {
            config: CollectionConfig::new("docs".into(), 3, Distance::L2),
            max_seq_id: 42,
            total_elements: 7,
            shard: None,
        }
    }

    /// Write a fake generation directory so the archive has real files.
    fn generation(dir: &Path, name: &str, files: &[(&str, &[u8])]) -> PathBuf {
        let dir = dir.join("generations").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        for (path, bytes) in files {
            std::fs::write(dir.join(path), bytes).unwrap();
        }
        dir
    }

    #[test]
    fn roundtrip_preserves_bytes_state_and_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let gen_dir = generation(
            tmp.path(),
            "gen-1",
            &[("manifest.json", b"{\"a\":1}"), ("records.bin", b"records")],
        );

        let archive = encode(b"cluster-state", vec![(record(), gen_dir)]).unwrap();
        let decoded = decode(&archive).unwrap();

        assert_eq!(decoded.state, b"cluster-state");
        assert_eq!(decoded.collections.len(), 1);
        let entry = &decoded.collections[0];
        assert_eq!(entry.generation, "gen-1");
        assert_eq!(entry.record.max_seq_id, 42);
        assert_eq!(entry.files.len(), 2);
        let names: Vec<&str> = entry.files.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, vec!["manifest.json", "records.bin"]);
        assert_eq!(entry.files[1].1, b"records");
    }

    #[test]
    fn an_unknown_version_is_refused_with_a_useful_message() {
        let tmp = tempfile::tempdir().unwrap();
        let gen_dir = generation(tmp.path(), "gen-1", &[("manifest.json", b"{}")]);
        let mut archive = encode(b"", vec![(record(), gen_dir)]).unwrap();
        archive[4..8].copy_from_slice(&(VERSION + 1).to_le_bytes());

        let err = decode(&archive).unwrap_err().to_string();
        assert!(err.contains("format version"), "{err}");
        assert!(err.contains("refusing"), "{err}");
    }

    #[test]
    fn a_flipped_payload_byte_is_detected() {
        let tmp = tempfile::tempdir().unwrap();
        let gen_dir = generation(tmp.path(), "gen-1", &[("records.bin", b"0123456789")]);
        let mut archive = encode(b"", vec![(record(), gen_dir)]).unwrap();

        // Corrupt the final payload byte, just before the trailer.
        let index = archive.len() - 5;
        archive[index] ^= 0xff;

        let err = decode(&archive).unwrap_err().to_string();
        assert!(err.contains("checksum mismatch"), "{err}");
        assert!(err.contains("records.bin"), "{err}");
    }

    #[test]
    fn a_truncated_archive_is_detected_rather_than_partially_accepted() {
        let tmp = tempfile::tempdir().unwrap();
        let gen_dir = generation(tmp.path(), "gen-1", &[("records.bin", b"0123456789")]);
        let archive = encode(b"", vec![(record(), gen_dir)]).unwrap();

        let truncated = &archive[..archive.len() - 2];
        let err = decode(truncated).unwrap_err().to_string();
        assert!(
            err.contains("truncated") || err.contains("overruns"),
            "{err}"
        );
    }

    #[test]
    fn a_header_length_larger_than_the_archive_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let gen_dir = generation(tmp.path(), "gen-1", &[("records.bin", b"x")]);
        let mut archive = encode(b"", vec![(record(), gen_dir)]).unwrap();
        archive[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode(&archive).is_err());
    }

    #[test]
    fn foreign_bytes_are_rejected_by_magic() {
        let err = decode(b"NOPE-and-then-some-more-bytes")
            .unwrap_err()
            .to_string();
        assert!(err.contains("bad magic"), "{err}");
    }

    #[test]
    fn traversal_paths_in_the_header_are_rejected() {
        assert!(verify_path("../escape").is_err());
        assert!(verify_path("/abs").is_err());
        assert!(verify_path("a/../../b").is_err());
        assert!(verify_path("").is_err());
        assert!(verify_path("gen/segment-0.bin").is_ok());
    }

    #[test]
    fn nested_directories_are_copied_with_their_relative_path() {
        let tmp = tempfile::tempdir().unwrap();
        let gen_dir = generation(tmp.path(), "gen-1", &[("manifest.json", b"{}")]);
        std::fs::create_dir_all(gen_dir.join("sub")).unwrap();
        std::fs::write(gen_dir.join("sub").join("part"), b"nested").unwrap();

        let archive = encode(b"", vec![(record(), gen_dir)]).unwrap();
        let decoded = decode(&archive).unwrap();
        let names: Vec<&str> = decoded.collections[0]
            .files
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        assert!(names.contains(&"sub/part"), "{names:?}");
    }
}
