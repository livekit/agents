//! What the hosting process lends the async crates' clients: its Tokio runtime,
//! the executor its host awaits on, the host's clock and its connection pool, as one
//! [`ProcessEnv`] value. Internal constructors take the handles they need;
//! a class the host constructs receives the runtime's value from the generated bindings.

use std::sync::Arc;

use lk_agents_abi::{Clock, Spawner};

use crate::http::GatewayHttp;

/// What the hosting process lends a declared class: the generated bindings pass
/// it to each declared constructor that takes it, and Rust callers and tests build their own.
/// Handles bound to the process runtime are read on use, so constructing a client starts
/// nothing and a fork-inherited runtime is refused rather than used.
#[derive(Clone)]
pub struct ProcessEnv {
    /// The runtime this crate's tasks and sockets run on.
    pub runtime: fn() -> Result<tokio::runtime::Handle, String>,
    /// The executor the host awaits a chain's futures on.
    pub spawner: fn() -> Result<Arc<dyn Spawner>, String>,
    /// The host's clock (virtual in a virtual-time test).
    pub clock: Arc<dyn Clock>,
    /// The process's shared connection pool.
    pub http: fn() -> Result<Arc<GatewayHttp>, String>,
}

impl ProcessEnv {
    /// The runtime the caller is on, else the process runtime.
    pub fn caller_runtime(&self) -> Result<tokio::runtime::Handle, String> {
        tokio::runtime::Handle::try_current().or_else(|_| (self.runtime)())
    }

    /// A Rust caller with no host: the caller's Tokio runtime, the wall clock, a connection pool
    /// per client (a pool's connections belong to the runtime that opened them, and the
    /// caller's runtime may not outlive the client).
    pub fn current_runtime() -> Self {
        fn runtime() -> Result<tokio::runtime::Handle, String> {
            tokio::runtime::Handle::try_current().map_err(|error| error.to_string())
        }
        ProcessEnv {
            runtime,
            spawner: || Ok(Arc::new(crate::task::TokioSpawner::new(runtime()?))),
            clock: Arc::new(crate::clock::RealClock::new()),
            http: || GatewayHttp::new().map(Arc::new),
        }
    }
}
