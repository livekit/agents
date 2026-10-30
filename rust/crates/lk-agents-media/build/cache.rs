//! Content-addressed native artifacts, shared across Cargo targets and worktrees.

use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

/// Platform conventions use the build machine's paths, independent of Cargo's target.
pub fn default_directory(
    platform: &str,
    home: Option<PathBuf>,
    xdg_cache: Option<PathBuf>,
    local_app_data: Option<PathBuf>,
) -> Option<PathBuf> {
    let home = home.filter(|path| path.is_absolute());
    let base = match platform {
        "windows" => local_app_data.filter(|path| path.is_absolute()),
        "macos" => home.map(|path| path.join("Library").join("Caches")),
        _ => xdg_cache
            .filter(|path| path.is_absolute())
            .or_else(|| home.map(|path| path.join(".cache"))),
    };
    base.map(|path| path.join("livekit-agents").join("native"))
}

/// An unavailable default user cache must not prevent compilation in a writable target.
/// Explicit overrides are validated and reported to the caller instead of silently ignored.
pub fn directory(
    explicit: Option<&Path>,
    preferred: Option<&Path>,
    fallback: &Path,
) -> io::Result<PathBuf> {
    fn writable(path: &Path) -> io::Result<()> {
        fs::create_dir_all(path)?;
        let _probe = tempfile::tempfile_in(path)?;
        Ok(())
    }
    if let Some(path) = explicit {
        if !path.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "LK_AGENTS_NATIVE_CACHE_DIR must be an absolute path",
            ));
        }
        writable(path)?;
        return Ok(path.to_owned());
    }
    if let Some(path) = preferred.filter(|path| path.is_absolute())
        && writable(path).is_ok()
    {
        return Ok(path.to_owned());
    }
    writable(fallback)?;
    Ok(fallback.to_owned())
}

pub struct Key(Sha256);

impl Key {
    pub fn new() -> Self {
        let mut key = Self(Sha256::new());
        key.input("format", b"livekit-agents-native-v1");
        key
    }

    pub fn input(&mut self, name: &str, value: &[u8]) {
        self.0.update((name.len() as u64).to_le_bytes());
        self.0.update(name.as_bytes());
        self.0.update((value.len() as u64).to_le_bytes());
        self.0.update(value);
    }

    pub fn tree(&mut self, root: &Path) -> io::Result<()> {
        fn visit(key: &mut Key, root: &Path, path: &Path) -> io::Result<()> {
            if path.is_dir() {
                let mut entries = fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
                entries.sort_by_key(|entry| entry.file_name());
                for entry in entries {
                    visit(key, root, &entry.path())?;
                }
            } else {
                let relative = path.strip_prefix(root).expect("entry belongs to source tree");
                key.input("path", relative.as_os_str().as_encoded_bytes());
                key.input("contents", &fs::read(path)?);
            }
            Ok(())
        }
        visit(self, root, root)
    }

    pub fn finish(self) -> String {
        format!("{:x}", self.0.finalize())
    }
}

/// Build under an OS lock. A failed or interrupted build is never a cache hit.
/// Build in a unique staging directory and atomically publish it. The builder also
/// receives the final path so native pkg-config files can embed a stable prefix.
pub fn get_or_build(
    root: &Path,
    name: &str,
    key: &str,
    valid: impl Fn(&Path) -> bool,
    build: impl FnOnce(&Path, &Path) -> io::Result<()>,
) -> io::Result<PathBuf> {
    fs::create_dir_all(root)?;
    let entry = root.join(format!("{name}-{key}"));
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.join(format!("{name}-{key}.lock")))?;
    lock.lock()?;

    let marker = entry.join(".complete");
    if fs::read_to_string(&marker).is_ok_and(|contents| contents == key) && valid(&entry) {
        return Ok(entry);
    }
    if entry.exists() {
        fs::remove_dir_all(&entry)?;
    }
    let staging = tempfile::Builder::new().prefix(&format!(".{name}-")).tempdir_in(root)?;
    build(staging.path(), &entry)?;
    if !valid(staging.path()) {
        return Err(io::Error::other(format!(
            "{name} build did not produce its expected artifacts"
        )));
    }
    fs::write(staging.path().join(".complete"), key)?;
    fs::rename(staging.path(), &entry)?;
    Ok(entry)
}
