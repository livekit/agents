use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::Arc;

use crate::StorageError;

use crate::sql::{Bind, SqlValue, first_cell};
use lk_agents_abi::Clock;

use super::connection::{batch, connect, query};
use super::{DeleteRegistryThread, RegistryThreadIsLive};

// The lock file is permanent: replacing its inode would let two processes lock different files.

pub(super) struct Lock<'a>(&'a File);

impl Drop for Lock<'_> {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

fn lock_file(directory: &Path) -> Result<File, StorageError> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(directory.join("lifecycle.lock"))
        .map_err(io_error)
}

fn io_error(error: std::io::Error) -> StorageError {
    match error.kind() {
        std::io::ErrorKind::PermissionDenied => StorageError::Unauthorized(error.to_string()),
        _ => StorageError::Unavailable(format!("storage lifecycle: {error}")),
    }
}

/// Each thread worker checks current metadata while holding a shared lifecycle lock.
pub(super) struct Access {
    lock: File,
    registry: rusqlite::Connection,
    thread_id: String,
}

impl Access {
    pub(super) fn open(
        directory: &Path,
        thread_id: String,
        clock: &Arc<dyn Clock>,
    ) -> Result<Self, StorageError> {
        let lock = lock_file(directory)?;
        let registry = connect(&directory.join("registry.sqlite"), false, clock)?;
        Ok(Self { lock, registry, thread_id })
    }

    pub(super) fn lock(&self) -> Result<Lock<'_>, StorageError> {
        self.lock.lock_shared().map_err(io_error)?;
        Ok(Lock(&self.lock))
    }

    fn enter(&self) -> Result<Lock<'_>, StorageError> {
        let lock = self.lock()?;
        let live = RegistryThreadIsLive { thread_id: &self.thread_id }.bind();
        if first_cell(&query(&self.registry, &live)?).and_then(SqlValue::as_i64) != Some(1) {
            return Err(StorageError::NotFound(self.thread_id.clone()));
        }
        Ok(lock)
    }
}

pub(super) fn with_access<T>(
    access: Option<&Access>,
    run: impl FnOnce() -> Result<T, StorageError>,
) -> Result<T, StorageError> {
    let _lock = access.map(Access::enter).transpose()?;
    run()
}

pub(super) fn close(connection: rusqlite::Connection, access: Option<&Access>) {
    let _lock = access.and_then(|access| access.lock().ok());
    drop(connection);
}

/// Wait for in-flight operations, hide the thread, then reclaim files. Existing workers
/// cannot access their open SQLite connection again after the registry row is gone.
pub(super) fn delete(
    registry: &mut rusqlite::Connection,
    directory: &Path,
    thread_id: &str,
) -> Result<(), StorageError> {
    let file = lock_file(directory)?;
    file.lock().map_err(io_error)?;
    let _lock = Lock(&file);
    batch(registry, &[DeleteRegistryThread { thread_id }.bind()])?;
    let path = directory.join("threads").join(format!("{thread_id}.sqlite"));
    for suffix in ["", "-wal", "-shm"] {
        let mut path = path.clone().into_os_string();
        path.push(suffix);
        match std::fs::remove_file(path) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                return Err(io_error(error));
            }
            _ => {}
        }
    }
    Ok(())
}
