//! Local-filesystem object store.
//!
//! Keys are forward-slash paths mapped to files under a root directory. Every
//! key is strictly sanitized before touching the filesystem: absolute paths,
//! `.` / `..` segments, empty segments, backslashes, and NUL bytes are all
//! rejected with [`StorageError::InvalidKey`], so a key can never escape the
//! root (no `../etc/passwd`).
//!
//! `put` writes to a uniquely-named temp file in the target directory and then
//! `rename`s it into place, so concurrent readers always observe either the
//! old or the new object, never a torn write.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::storage::{Storage, StorageError};

/// Temp-file suffix marker, used both for naming and to filter stale temp files
/// out of `list`.
const TMP_SUFFIX: &str = ".tmp-";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Object store rooted at a local directory.
#[derive(Debug, Clone)]
pub struct LocalStorage {
    root: PathBuf,
}

impl LocalStorage {
    /// Create an object store rooted at `root`. The directory does not need to
    /// exist yet; it is created lazily on the first `put`.
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Map a forward-slash object key to a root-relative `PathBuf`, rejecting
    /// anything that could escape the root.
    fn sanitize_key(key: &str) -> Result<PathBuf, StorageError> {
        if key.is_empty() || key.starts_with('/') {
            return Err(StorageError::InvalidKey(key.to_owned()));
        }
        let mut parts = Vec::new();
        for segment in key.split('/') {
            if segment.is_empty() || segment == "." || segment == ".." {
                return Err(StorageError::InvalidKey(key.to_owned()));
            }
            if segment.contains('\\') || segment.contains('\0') {
                return Err(StorageError::InvalidKey(key.to_owned()));
            }
            parts.push(segment);
        }
        if parts.is_empty() {
            return Err(StorageError::InvalidKey(key.to_owned()));
        }
        Ok(parts.into_iter().collect())
    }

    fn full_path(&self, key: &str) -> Result<PathBuf, StorageError> {
        Ok(self.root.join(Self::sanitize_key(key)?))
    }

    /// Recursively collect the keys of all regular files under `dir`, skipping
    /// stale temp files.
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<(), StorageError> {
        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        for entry in entries {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let path = entry.path();
            if file_type.is_dir() {
                Self::walk(root, &path, out)?;
            } else if file_type.is_file() {
                let name = entry.file_name();
                if name.to_string_lossy().contains(TMP_SUFFIX) {
                    continue;
                }
                let relative = path.strip_prefix(root).map_err(|_| {
                    StorageError::InvalidKey(format!("path escapes root: {:?}", path))
                })?;
                let key = relative.to_string_lossy().replace('\\', "/");
                out.push(key);
            }
        }
        Ok(())
    }
}

fn sync_dir(dir: &Path) -> io::Result<()> {
    let f = fs::File::open(dir)?;
    f.sync_all()
}

impl Storage for LocalStorage {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<(), StorageError> {
        let full = self.full_path(key)?;
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent)?;
        }
        let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let file_name = full
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "object".to_owned());
        let tmp = full.with_file_name(format!(
            "{file_name}{TMP_SUFFIX}{}-{nonce}",
            std::process::id()
        ));
        {
            let mut file = fs::File::create(&tmp)?;
            file.write_all(bytes)?;
            file.sync_all()?;
        }
        fs::rename(&tmp, &full)?;
        if let Some(parent) = full.parent() {
            let _ = sync_dir(parent);
        }
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, StorageError> {
        let full = self.full_path(key)?;
        match fs::read(&full) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        let trimmed = prefix.trim_end_matches('/');
        let dir = if trimmed.is_empty() {
            self.root.clone()
        } else {
            self.root.join(Self::sanitize_key(trimmed)?)
        };
        if dir.is_file() {
            return Ok(vec![trimmed.to_owned()]);
        }
        let mut keys = Vec::new();
        Self::walk(&self.root, &dir, &mut keys)?;
        keys.sort();
        Ok(keys)
    }

    fn delete(&self, key: &str) -> Result<(), StorageError> {
        let full = self.full_path(key)?;
        match fs::remove_file(&full) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                Err(StorageError::NotFound(key.to_owned()))
            }
            Err(e) => Err(e.into()),
        }
    }

    fn exists(&self, key: &str) -> Result<bool, StorageError> {
        Ok(self.full_path(key)?.is_file())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    fn new_store() -> (TempDir, LocalStorage) {
        let dir = TempDir::new().unwrap();
        let store = LocalStorage::new(dir.path().to_path_buf());
        (dir, store)
    }

    #[test]
    fn put_get_roundtrip() {
        let (_dir, store) = new_store();
        store
            .put("col/abc/segments/seg-0.bin", b"hello world")
            .unwrap();
        let got = store.get("col/abc/segments/seg-0.bin").unwrap().unwrap();
        assert_eq!(got, b"hello world");
        assert!(store.exists("col/abc/segments/seg-0.bin").unwrap());
    }

    #[test]
    fn put_overwrites_atomically() {
        let (_dir, store) = new_store();
        store.put("key", b"v1").unwrap();
        store.put("key", b"v2-longer").unwrap();
        assert_eq!(store.get("key").unwrap().unwrap(), b"v2-longer");
    }

    #[test]
    fn get_missing_returns_none() {
        let (_dir, store) = new_store();
        assert!(store.get("does/not/exist").unwrap().is_none());
        assert!(!store.exists("does/not/exist").unwrap());
    }

    #[test]
    fn list_prefix_and_empty() {
        let (_dir, store) = new_store();
        store.put("col/a/s1", b"1").unwrap();
        store.put("col/a/s2", b"2").unwrap();
        store.put("col/b/s1", b"3").unwrap();
        store.put("other/x", b"4").unwrap();

        let mut all = store.list("").unwrap();
        assert_eq!(all.len(), 4);
        all.sort();
        assert_eq!(
            all,
            vec![
                "col/a/s1".to_owned(),
                "col/a/s2".to_owned(),
                "col/b/s1".to_owned(),
                "other/x".to_owned(),
            ]
        );

        let col_a = store.list("col/a").unwrap();
        assert_eq!(col_a, vec!["col/a/s1".to_owned(), "col/a/s2".to_owned()]);
        assert_eq!(store.list("col/a/").unwrap(), col_a);
        assert!(store.list("col/z").unwrap().is_empty());
        assert_eq!(store.list("col/a/s1").unwrap(), vec!["col/a/s1".to_owned()]);
    }

    #[test]
    fn delete_removes_object() {
        let (_dir, store) = new_store();
        store.put("gone", b"x").unwrap();
        store.delete("gone").unwrap();
        assert!(store.get("gone").unwrap().is_none());
        assert!(matches!(
            store.delete("gone"),
            Err(StorageError::NotFound(_))
        ));
    }

    #[test]
    fn sanitization_rejects_unsafe_keys() {
        let (_dir, store) = new_store();
        for bad in [
            "../etc/passwd",
            "a/../evil",
            "../evil",
            "/abs/path",
            "a//b",
            "a/b/",
            "a/./b",
            "",
            "a\\..\\evil",
            "trailing\\",
        ] {
            assert!(
                matches!(store.put(bad, b"x"), Err(StorageError::InvalidKey(_))),
                "key `{bad}` should be rejected"
            );
            assert!(
                matches!(store.get(bad), Err(StorageError::InvalidKey(_))),
                "get key `{bad}` should be rejected"
            );
            assert!(matches!(
                store.delete(bad),
                Err(StorageError::InvalidKey(_))
            ));
        }
    }

    #[test]
    fn sanitized_keys_stay_inside_root() {
        let (_dir, store) = new_store();
        store.put("safe/key.bin", b"data").unwrap();
        let root = store.root.clone();
        assert!(root.join("safe/key.bin").is_file());
        // The traversal attempt must not have written anywhere outside root.
        assert!(!root.join("..").join("safe").join("key.bin").exists());
        let listed = store.list("").unwrap();
        assert!(listed.contains(&"safe/key.bin".to_owned()));
    }

    #[test]
    fn list_skips_leftover_temp_files() {
        let (_dir, store) = new_store();
        store.put("obj", b"z").unwrap();
        let root = store.root.clone();
        fs::create_dir_all(root.join("dir")).unwrap();
        fs::write(root.join("dir/stale.tmp-123-456"), b"stale").unwrap();
        let listed = store.list("").unwrap();
        assert_eq!(listed, vec!["obj".to_owned()]);
    }
}
