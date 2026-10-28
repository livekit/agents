//! What the hosting process lends this crate's clients (decision 88): its Tokio runtime, the
//! executor its host awaits on, the host's clock and the media worker pool. Every client takes
//! these values in its constructor; a class the host constructs (a declared `new`) reads them
//! through its [`Env`] type parameter, which `lk-agents-runtime` binds to its process.

use std::sync::Arc;

use lk_agents_abi::{Clock, Spawner};
use lk_agents_media::workers::MediaWorkers;

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
    /// The process's admitted CPU workers for model, codec and image work.
    fn workers() -> Result<MediaWorkers, String>;
}

/// A Rust caller with no host: the current Tokio runtime, the wall clock and one worker pool.
pub struct Standalone;

impl Env for Standalone {
    fn runtime() -> Result<tokio::runtime::Handle, String> {
        tokio::runtime::Handle::try_current().map_err(|error| error.to_string())
    }

    fn clock() -> Arc<dyn Clock> {
        Arc::new(lk_agents_abi::RealClock::new())
    }

    fn workers() -> Result<MediaWorkers, String> {
        static WORKERS: std::sync::OnceLock<Result<MediaWorkers, String>> =
            std::sync::OnceLock::new();
        WORKERS
            .get_or_init(|| {
                MediaWorkers::new(Default::default()).map_err(|error| error.to_string())
            })
            .clone()
    }
}
