//! What the hosting process lends the async crates' clients (decision 88): its Tokio runtime, the
//! executor its host awaits on, the host's clock and process admissions. Every client takes these
//! values in its constructor; a class the host constructs (a declared `new`) reads them through
//! its [`Env`] type parameter, which `lk-agents-runtime` binds to its process.

use std::sync::{Arc, LazyLock};

use lk_agents_abi::{Clock, Spawner};

use crate::session_report::ReportAdmission;

pub trait Env: Send + Sync + 'static {
    /// Subject-specific services supplied by the process, without an edge to their crate.
    type Services: Send + Sync + 'static;
    fn services() -> Self::Services;

    /// The process's shared connection pool, supplied to each client constructor.
    fn http() -> Result<Arc<crate::http::GatewayHttp>, String> {
        crate::http::GatewayHttp::new().map(Arc::new)
    }

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
pub struct Standalone<S = ()>(std::marker::PhantomData<fn() -> S>);

impl<S: Default + Send + Sync + 'static> Env for Standalone<S> {
    type Services = S;

    fn services() -> S {
        S::default()
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
