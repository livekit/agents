//! What the hosting process lends the worker's host-constructed classes (decision 88): its
//! runtime through the bounded worker I/O built on it.

use crate::io::WorkerIo;

pub trait Env: Send + Sync + 'static {
    /// The process's worker I/O, on the runtime the worker's tasks and sockets run on.
    fn worker_io() -> Result<WorkerIo, String>;
}

