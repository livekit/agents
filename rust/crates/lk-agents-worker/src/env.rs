//! What the hosting process lends the worker's host-constructed classes (decision 88): the shared
//! process environment plus the bounded worker I/O built on its runtime.

use crate::io::WorkerIo;

pub trait Env: lk_agents_net::env::Env {
    /// The process's worker I/O, on the runtime the worker's tasks and sockets run on.
    fn worker_io() -> Result<WorkerIo, String>;
}

