//! What a job runs in, as supervision sees it: a [`RunnerExecutor`] launches a [`Runner`] and hands
//! back the worker's end of its channel. The process executor ([`super::process`]) and the thread
//! executor ([`ThreadExecutor`]) are the two; supervision never asks which one it has.

use crate::ipc::telemetry;
use std::sync::Arc;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::worker::ipc_host;
use tokio::sync::watch;

use super::ipc_channel::{self, Channel};
use super::owner::{JobLaunch, Report};
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
    /// Resolves once it ended and nothing of it is left (a process reaped, a thread returned).
    /// Cancel safe.
    async fn finished(&self) -> Exit;
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

/// A job thread of the worker's own process: the launcher starts it on the job's end of the
/// channel and reports its return on the launch.
pub struct ThreadExecutor {
    launcher: Arc<dyn ipc_host::JobLauncher>,
    context: CallContext,
    exited: Arc<watch::Sender<bool>>,
}

impl ThreadExecutor {
    pub fn new(launcher: Arc<dyn ipc_host::JobLauncher>, context: CallContext) -> Self {
        Self { launcher, context, exited: Arc::new(watch::channel(false).0) }
    }
}

#[async_trait::async_trait]
impl RunnerExecutor for ThreadExecutor {
    async fn launch(&self) -> Result<(Box<dyn Runner>, Connection), HostError> {
        let (parent, child) = ipc_channel::pair().map_err(host_error)?;
        let connection = Channel::from_endpoint(parent).map_err(host_error)?;
        let launch = JobLaunch::report(Report::Thread(self.exited.clone()));
        self.launcher.launch(&telemetry::Cx::NONE, &self.context, child, launch).await?;
        Ok((Box::new(JobThread(self.exited.subscribe())), connection))
    }
}

struct JobThread(watch::Receiver<bool>);

#[async_trait::async_trait]
impl Runner for JobThread {
    async fn finished(&self) -> Exit {
        // Without its executor nobody can report the return: the thread stays unaccounted for.
        if self.0.clone().wait_for(|exited| *exited).await.is_err() {
            std::future::pending::<()>().await;
        }
        Exit { code: 0, killed: false }
    }

    fn stop(&self) -> bool {
        false
    }

    fn memory_mb(&self) -> Option<f64> {
        None
    }
}
