//! What the hosting process lends the async crates' clients (decision 88): its Tokio runtime, the
//! executor its host awaits on, the host's clock and process admissions. Every client takes these
//! values in its constructor; a class the host constructs (a declared `new`) reads them through
//! its [`Env`] type parameter, which `lk-agents-runtime` binds to its process.

use std::sync::{Arc, LazyLock};

use lk_agents_abi::{Clock, Spawner};

use crate::http::GatewayHttp;
use crate::session_report::ReportAdmission;

/// What the hosting process lends a declared class (decision 100): the generated bindings pass
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
    /// The process's session report admission.
    pub report_admission: fn() -> Result<ReportAdmission, String>,
}

impl ProcessEnv {
    /// The runtime the caller is on, else the process runtime.
    pub fn caller_runtime(&self) -> Result<tokio::runtime::Handle, String> {
        tokio::runtime::Handle::try_current().or_else(|_| (self.runtime)())
    }

    /// A Rust caller with no host: the caller's Tokio runtime, the wall clock, a connection pool
    /// per client (a pool's connections belong to the runtime that opened them, and the
    /// caller's runtime may not outlive the client) and one report admission.
    pub fn current_runtime() -> Self {
        fn runtime() -> Result<tokio::runtime::Handle, String> {
            tokio::runtime::Handle::try_current().map_err(|error| error.to_string())
        }
        ProcessEnv {
            runtime,
            spawner: || Ok(Arc::new(lk_agents_abi::TokioSpawner::new(runtime()?))),
            clock: Arc::new(lk_agents_abi::RealClock::new()),
            http: || GatewayHttp::new().map(Arc::new),
            report_admission: || {
                static ADMISSION: LazyLock<ReportAdmission> =
                    LazyLock::new(ReportAdmission::default);
                Ok(ADMISSION.clone())
            },
        }
    }
}

pub trait Env: Send + Sync + 'static {
    /// The process's shared connection pool, supplied to each client constructor.
    fn http() -> Result<Arc<GatewayHttp>, String>;

    /// The runtime this crate's tasks and sockets run on.
    fn runtime() -> Result<tokio::runtime::Handle, String>;
    /// The runtime the caller is on, else [`Env::runtime`].
    fn caller_runtime() -> Result<tokio::runtime::Handle, String> {
        tokio::runtime::Handle::try_current().or_else(|_| Self::runtime())
    }
    /// The executor the host awaits a chain's futures on.
    fn spawner() -> Result<Arc<dyn Spawner>, String> {
        Ok(Arc::new(lk_agents_abi::TokioSpawner::new(Self::runtime()?)))
    }
    /// The host's clock (virtual in a virtual-time test).
    fn clock() -> Arc<dyn Clock>;
    /// The process's session report admission.
    fn report_admission() -> Result<ReportAdmission, String>;
}

/// A Rust caller with no host: the current Tokio runtime and the wall clock.
pub struct Standalone;

impl Env for Standalone {
    /// A pool per client: a pool's connections belong to the runtime that opened them, and a
    /// standalone caller's runtime may not outlive the client (a test's).
    fn http() -> Result<Arc<GatewayHttp>, String> {
        GatewayHttp::new().map(Arc::new)
    }

    fn runtime() -> Result<tokio::runtime::Handle, String> {
        tokio::runtime::Handle::try_current().map_err(|error| error.to_string())
    }

    fn clock() -> Arc<dyn Clock> {
        Arc::new(lk_agents_abi::RealClock::new())
    }

    fn report_admission() -> Result<ReportAdmission, String> {
        static ADMISSION: LazyLock<ReportAdmission> = LazyLock::new(ReportAdmission::default);
        Ok(ADMISSION.clone())
    }
}
