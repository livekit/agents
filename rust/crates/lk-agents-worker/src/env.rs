//! What the hosting process lends the worker's host-constructed classes (decision 88): its
//! runtime through the bounded worker I/O built on it.

use crate::io::WorkerIo;

pub trait Env: Send + Sync + 'static {
    /// The process's worker I/O, on the runtime the worker's tasks and sockets run on.
    fn worker_io() -> Result<WorkerIo, String>;
}

/// A Rust caller with no host: worker I/O on the current runtime.
pub struct Standalone;

impl Env for Standalone {
    fn worker_io() -> Result<WorkerIo, String> {
        let runtime = tokio::runtime::Handle::try_current().map_err(|error| error.to_string())?;
        WorkerIo::new(runtime, 4).map_err(|error| error.to_string())
    }
}
