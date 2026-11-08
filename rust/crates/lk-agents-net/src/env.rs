//! What the hosting process lends the async crates' clients: its Tokio runtime,
//! the executor its host awaits on, the host's clock and its connection pool, as one
//! [`ProcessEnv`] value. Internal constructors take the handles they need;
//! a class the host constructs receives the runtime's value from the generated bindings.

use std::sync::Arc;

use lk_agents_abi::{Clock, Spawner};

use crate::http::GatewayHttp;

/// Why the process cannot lend a runtime-bound handle. Nothing retries it: a client that cannot
/// get one fails for good.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProcessError {
    #[error("the native runtime failed to start: {0}")]
    Start(String),
    #[error(
        "livekit.agents._core was inherited across a fork. The Rust core cannot be used in a \
         forked child without re-initialising it: call _core.init(role) in the child."
    )]
    Inherited,
    /// A Rust caller with no runtime on its thread.
    #[error("no runtime to run on: {0}")]
    NoRuntime(String),
}

/// A client that cannot get its process handles fails for good, as a non-retryable `APIError`.
impl From<ProcessError> for lk_agents_abi::errors::ApiError {
    fn from(error: ProcessError) -> Self {
        Self::new(error.to_string()).with_retryable(false)
    }
}

/// What the hosting process lends a declared class: the generated bindings pass
/// it to each declared constructor that takes it, and Rust callers and tests build their own.
/// Handles bound to the process runtime are read on use, so constructing a client starts
/// nothing and a fork-inherited runtime is refused rather than used.
#[derive(Clone)]
pub struct ProcessEnv {
    /// The runtime this crate's tasks and sockets run on.
    pub runtime: fn() -> Result<tokio::runtime::Handle, ProcessError>,
    /// The executor the host awaits a chain's futures on.
    pub spawner: fn() -> Result<Arc<dyn Spawner>, ProcessError>,
    /// The host's clock (virtual in a virtual-time test).
    pub clock: Arc<dyn Clock>,
    /// The process's shared connection pool.
    pub http: fn() -> Result<Arc<GatewayHttp>, ProcessError>,
    /// The process's span recording, timed by `clock`.
    pub tracer: lk_agents_telemetry::Tracer,
}

impl ProcessEnv {
    /// `<hostname>-<pid>`: this process's default storage lease holder. Read on each call, so a
    /// job process names itself, not the parent it was forked from.
    pub fn lease_holder(&self) -> String {
        let host = sysinfo::System::host_name().unwrap_or_default();
        format!("{host}-{}", std::process::id())
    }

    /// The runtime the caller is on, else the process runtime.
    pub fn caller_runtime(&self) -> Result<tokio::runtime::Handle, ProcessError> {
        tokio::runtime::Handle::try_current().or_else(|_| (self.runtime)())
    }

    /// A Rust caller with no host: the caller's Tokio runtime, the wall clock, no span recording,
    /// a connection pool per client (a pool's connections belong to the runtime that opened them, and the
    /// caller's runtime may not outlive the client).
    pub fn current_runtime() -> Self {
        fn runtime() -> Result<tokio::runtime::Handle, ProcessError> {
            tokio::runtime::Handle::try_current()
                .map_err(|error| ProcessError::NoRuntime(error.to_string()))
        }
        let clock: Arc<dyn Clock> = Arc::new(crate::clock::RealClock::new());
        ProcessEnv {
            runtime,
            spawner: || Ok(Arc::new(crate::task::TokioSpawner::new(runtime()?))),
            tracer: lk_agents_telemetry::Tracer::disabled(clock.clone()),
            clock,
            http: || {
                GatewayHttp::new().map(Arc::new).map_err(|e| ProcessError::Start(e.to_string()))
            },
        }
    }
}
