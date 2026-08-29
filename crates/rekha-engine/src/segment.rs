//! Immutable mmap'd vector segment files.
//!
//! A [`Segment`] is a flat binary file holding a batch of f32 vectors with a
//! 32-byte header.  Segments are written once ([`SegmentWriter`]) and
//! memory-mapped on read for zero-copy random access.  They are the
//! persistence layer for vector data: on checkpoint the engine writes segment
//! files; on reopen it mmaps them for fast brute-force scan and usearch
//! rebuild.
//!
//! File layout (all integers little-endian):
//! ```text
//! [header: 32 bytes]
//!   magic:      u8[4]  = b"RKSG"
//!   version:    u32    = 1
//!   dimension:  u32
//!   count:      u32
//!   checksum:   u32    (CRC-32 of the vector data block)
//!   _reserved:  u8[16] = 0
//! [vector data: count * dimension * 4 bytes]
//!   Row-major f32 array - each vector is `dimension` contiguous floats.
//! ```

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::{EngineError, EngineResult};

/// Magic bytes identifying a RekhaDB segment file.
const MAGIC: &[u8; 4] = b"RKSG";
/// Current segment format version.
const VERSION: u32 = 1;
/// Fixed header size in bytes.
const HEADER_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// SegmentHeader
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SegmentHeader {
    dimension: u32,
    count: u32,
    checksum: u32,
}

impl SegmentHeader {
    fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(MAGIC);
        buf[4..8].copy_from_slice(&VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&self.dimension.to_le_bytes());
        buf[12..16].copy_from_slice(&self.count.to_le_bytes());
        buf[16..20].copy_from_slice(&self.checksum.to_le_bytes());
        buf
    }

    fn decode(buf: &[u8; HEADER_SIZE]) -> EngineResult<Self> {
        if &buf[0..4] != MAGIC {
            return Err(EngineError::Serialization(format!(
                "segment bad magic: expected {:?}, got {:?}",
                MAGIC,
                &buf[0..4]
            )));
        }
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(EngineError::Serialization(format!(
                "segment unsupported version {version}"
            )));
        }
        let dimension = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let count = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let checksum = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        Ok(Self {
            dimension,
            count,
            checksum,
        })
    }
}

// ---------------------------------------------------------------------------
// Segment (mmap'd read handle)
// ---------------------------------------------------------------------------

/// An immutable, memory-mapped vector segment.
#[derive(Debug)]
pub struct Segment {
    #[allow(dead_code)]
    path: PathBuf,
    _file: File,
    mmap: memmap2::Mmap,
    header: SegmentHeader,
}

impl Segment {
    /// Open a segment file from the local filesystem.
    pub fn open(path: impl AsRef<Path>) -> EngineResult<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        Self::from_file(path, file)
    }

    fn from_file(path: PathBuf, mut file: File) -> EngineResult<Self> {
        let mut hdr_buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut hdr_buf)?;
        let header = SegmentHeader::decode(&hdr_buf)?;

        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Verify checksum of the vector data block.
        let data_start = HEADER_SIZE;
        let data_len = header.count as usize * header.dimension as usize * 4;
        if data_start + data_len > mmap.len() {
            return Err(EngineError::Serialization(format!(
                "segment truncated: header says {} bytes of vector data, file has {}",
                data_len,
                mmap.len() - data_start,
            )));
        }
        let actual_crc = crc32fast::hash(&mmap[data_start..data_start + data_len]);
        if actual_crc != header.checksum {
            return Err(EngineError::Serialization(format!(
                "segment checksum mismatch: expected {:#010x}, got {:#010x}",
                header.checksum, actual_crc,
            )));
        }

        Ok(Self {
            path,
            _file: file,
            mmap,
            header,
        })
    }

    /// Number of vectors in this segment.
    pub fn len(&self) -> usize {
        self.header.count as usize
    }

    /// Whether the segment is empty.
    pub fn is_empty(&self) -> bool {
        self.header.count == 0
    }

    /// Vector dimension.
    pub fn dimension(&self) -> usize {
        self.header.dimension as usize
    }

    /// Get a single vector by its sequential index (0-based).
    /// Returns a slice of `dimension` f32 values.
    pub fn get_vector(&self, index: usize) -> Option<&[f32]> {
        if index >= self.len() {
            return None;
        }
        let dim = self.dimension();
        let offset = HEADER_SIZE + index * dim * 4;
        let end = offset + dim * 4;
        Some(bytemuck::cast_slice(&self.mmap[offset..end]))
    }

    /// Iterate over all vectors as `&[f32]` slices.
    pub fn iter_vectors(&self) -> impl Iterator<Item = &[f32]> {
        let dim = self.dimension();
        let count = self.len();
        let base = HEADER_SIZE;
        (0..count).map(move |i| {
            let offset = base + i * dim * 4;
            let end = offset + dim * 4;
            bytemuck::cast_slice(&self.mmap[offset..end])
        })
    }
}

// ---------------------------------------------------------------------------
// SegmentWriter
// ---------------------------------------------------------------------------

/// Builds a segment file by accumulating vectors, then writes it atomically.
pub struct SegmentWriter {
    dimension: u32,
    vectors: Vec<f32>,
    count: u32,
}

impl SegmentWriter {
    /// Create a writer for vectors of the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension: dimension as u32,
            vectors: Vec::new(),
            count: 0,
        }
    }

    /// Append a vector. Panics if `embedding.len() != self.dimension`.
    pub fn push(&mut self, embedding: &[f32]) {
        assert_eq!(
            embedding.len(),
            self.dimension as usize,
            "dimension mismatch: expected {}, got {}",
            self.dimension,
            embedding.len()
        );
        self.vectors.extend_from_slice(embedding);
        self.count += 1;
    }

    /// Number of vectors accumulated.
    pub fn len(&self) -> u32 {
        self.count
    }

    /// Returns `true` if no vectors have been accumulated.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Write the segment to `path` atomically (temp file + rename).
    /// Returns the number of bytes written.
    pub fn write(self, path: impl AsRef<Path>) -> EngineResult<u64> {
        let path = path.as_ref();
        let data_bytes = self.vectors.len() * 4;
        let checksum = crc32fast::hash(bytemuck::cast_slice(&self.vectors));

        let header = SegmentHeader {
            dimension: self.dimension,
            count: self.count,
            checksum,
        };

        // Atomic write: temp file, write, rename.
        let tmp = {
            let mut os = path.as_os_str().to_os_string();
            os.push(".tmp");
            PathBuf::from(os)
        };

        let total = HEADER_SIZE + data_bytes;
        let mut buf = Vec::with_capacity(total);
        buf.extend_from_slice(&header.encode());
        buf.extend_from_slice(bytemuck::cast_slice(&self.vectors));

        std::fs::write(&tmp, buf)?;
        std::fs::rename(&tmp, path)?;

        Ok(total as u64)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_reader_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.seg");

        let mut writer = SegmentWriter::new(4);
        writer.push(&[1.0, 2.0, 3.0, 4.0]);
        writer.push(&[5.0, 6.0, 7.0, 8.0]);
        writer.push(&[9.0, 10.0, 11.0, 12.0]);
        assert_eq!(writer.len(), 3);

        let bytes_written = writer.write(&path).unwrap();
        assert_eq!(bytes_written, 32 + 3 * 4 * 4); // header + 3 vectors * 4 dims * 4 bytes

        let seg = Segment::open(&path).unwrap();
        assert_eq!(seg.len(), 3);
        assert_eq!(seg.dimension(), 4);

        let v0 = seg.get_vector(0).unwrap();
        assert_eq!(v0, &[1.0, 2.0, 3.0, 4.0]);

        let v2 = seg.get_vector(2).unwrap();
        assert_eq!(v2, &[9.0, 10.0, 11.0, 12.0]);

        assert!(seg.get_vector(3).is_none());
    }

    #[test]
    fn iter_vectors_matches_input() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("iter.seg");

        let mut writer = SegmentWriter::new(2);
        writer.push(&[1.0, 2.0]);
        writer.push(&[3.0, 4.0]);
        writer.write(&path).unwrap();

        let seg = Segment::open(&path).unwrap();
        let vecs: Vec<&[f32]> = seg.iter_vectors().collect();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0], &[1.0, 2.0]);
        assert_eq!(vecs[1], &[3.0, 4.0]);
    }

    #[test]
    fn empty_segment() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.seg");

        let writer = SegmentWriter::new(8);
        writer.write(&path).unwrap();

        let seg = Segment::open(&path).unwrap();
        assert_eq!(seg.len(), 0);
        assert_eq!(seg.dimension(), 8);
        assert!(seg.is_empty());
        assert!(seg.get_vector(0).is_none());
    }

    #[test]
    fn bad_magic_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.seg");
        // Write 32 bytes with wrong magic so header decode runs but rejects.
        let mut buf = [0u8; 32];
        buf[0..4].copy_from_slice(b"XXXX");
        std::fs::write(&path, buf).unwrap();
        let err = Segment::open(&path).unwrap_err();
        assert!(format!("{err}").contains("bad magic"));
    }

    #[test]
    fn truncated_file_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trunc.seg");

        let mut writer = SegmentWriter::new(4);
        writer.push(&[1.0, 2.0, 3.0, 4.0]);
        writer.write(&path).unwrap();

        // Truncate the file to remove the last few bytes.
        let meta = std::fs::metadata(&path).unwrap();
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(meta.len() - 8).unwrap();

        let err = Segment::open(&path).unwrap_err();
        assert!(format!("{err}").contains("truncated") || format!("{err}").contains("checksum"));
    }
}
