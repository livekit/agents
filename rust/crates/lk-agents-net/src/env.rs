//! What the hosting process lends the async crates' clients (decision 88): its Tokio runtime, the
//! executor its host awaits on and the host's clock. Every client takes these values in its
//! constructor; a class the host constructs (a declared `new`) reads them through its [`Env`]
//! type parameter, which `lk-agents-runtime` binds to its process.

use std::sync::Arc;

use lk_agents_abi::{Clock, Spawner};

pub trait Env: Send + Sync + 'static {
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
}

/// A Rust caller with no host: the current Tokio runtime and the wall clock.
pub struct Standalone;

impl Env for Standalone {
    fn runtime() -> Result<tokio::runtime::Handle, String> {
        tokio::runtime::Handle::try_current().map_err(|error| error.to_string())
    }

    fn clock() -> Arc<dyn Clock> {
        Arc::new(lk_agents_abi::RealClock::new())
    }
}
