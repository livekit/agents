use crate::ipc::telemetry;
use std::sync::Arc;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::worker::ipc_host;
use lk_agents_net::signal;

use super::ipc_channel::{self, Channel};
use super::owner::ThreadLaunch;
use super::wire::{ToChild, ToParent};

/// A local failure, as a launch reports it.
pub(super) fn host_error(error: impl std::fmt::Display) -> HostError {
    HostError::new(error.to_string())
}

/// The worker's end of a runner's channel.
pub type Connection = Channel<ToChild, ToParent>;

/// Launches one job runner.
#[async_trait::async_trait]
pub trait RunnerExecutor: Send + Sync + 'static {
    /// Starts a runner and returns it with the worker's end of its channel; an error (the host's
    /// own, when the host failed) means nothing was started.
    async fn launch(&self) -> Result<(Box<dyn Runner>, Connection), HostError>;
}

/// One launched runner.
#[async_trait::async_trait]
pub trait Runner: Send + Sync {
    /// Resolves once it ended and nothing of it is left (a process reaped, a thread returned);
    /// `Err(Ended)` once nothing can report its end any more. Cancel safe.
    async fn finished(&self) -> Result<Exit, signal::Ended>;
    /// Ends it where it can be forced: a process is killed; a thread cannot be, it ends once it
    /// did what it was asked. Returns whether it forced the end. Idempotent.
    fn stop(&self) -> bool;
    /// Its own memory footprint in MB, `None` when it has none or it cannot be read. Blocking.
    fn memory_mb(&self) -> Option<f64>;
}

/// How a runner ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exit {
    /// Negative: the signal that ended a Unix process.
    pub code: i64,
    /// `stop()` killed it.
    pub killed: bool,
}

impl Exit {
    /// A runner nothing can report the end of any more: failed.
    pub const LOST: Exit = Exit { code: -1, killed: false };
}

/// A job thread of the worker's own process: the launcher starts it on the job's end of the
/// channel and reports its return on the launch.
pub struct ThreadExecutor {
    launcher: Arc<dyn ipc_host::JobLauncher>,
    context: CallContext,
}

impl ThreadExecutor {
    pub fn new(launcher: Arc<dyn ipc_host::JobLauncher>, context: CallContext) -> Self {
        Self { launcher, context }
    }
}

#[async_trait::async_trait]
impl RunnerExecutor for ThreadExecutor {
    async fn launch(&self) -> Result<(Box<dyn Runner>, Connection), HostError> {
        let (parent, child) = ipc_channel::pair().map_err(host_error)?;
        let connection = Channel::from_endpoint(parent).map_err(host_error)?;
        // The host holds the producer: dropping the launch unreported ends the wait.
        let (exited, returned) = signal::signal(false);
        let launch = Arc::new(ThreadLaunch(exited));
        self.launcher.launch(&telemetry::Cx::NONE, &self.context, child, launch).await?;
        Ok((Box::new(JobThread(returned)), connection))
    }
}

struct JobThread(signal::Cell<bool>);

#[async_trait::async_trait]
impl Runner for JobThread {
    async fn finished(&self) -> Result<Exit, signal::Ended> {
        self.0.wait_for(|exited| *exited).await?;
        Ok(Exit { code: 0, killed: false })
    }

    fn stop(&self) -> bool {
        false
    }

    fn memory_mb(&self) -> Option<f64> {
        None
    }
}
