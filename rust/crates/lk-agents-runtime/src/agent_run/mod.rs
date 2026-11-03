//! One agent invocation executed: the core's `AgentRun` stepped one event at a time, its model
//! requests run against the agent's `Llm`, its host callbacks through [`HostInvoker`](crate::host::HostInvoker), its
//! deadline on the clock. Updates (stream events, then the settled result) reach the consumer
//! through a bounded queue the driver waits on; nothing is dropped.

use lk_agents_net::task::{TaskSet, TokioSpawner};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::agent::AgentDefinition;
use lk_agents_abi::agent::run::{Output, RunError, RunSettled, RunUpdate, ValidationId};
use lk_agents_abi::agent::{AgentCallbacks, Invocation, PreparedArguments};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::tool_executor::Approval;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::telemetry::spans::text::AgentRun as AgentRunSpan;
use lk_agents_abi::usage::Usage;
use lk_agents_abi::utils::host_ref::HostRef;
use lk_agents_abi::{ApiConnectOptions, ChatRequest, Clock, FunctionCall, Llm, Ts};
use lk_agents_core::agent_run::{
    AgentRunEffect, AgentRunEvent, CallId, CheckpointId, OutputVerdict, RenderId, RequestId,
};
use lk_agents_core::tool_calls::PrepareFailure;
use lk_agents_core::{AgentError, AgentRun, RunInput};
use lk_agents_net::telemetry::ScopedSpan;
use tokio::sync::{mpsc, oneshot, watch};

pub(crate) mod execution;
pub mod resources;
use execution::{Execution, Reporter};

/// Events the work in flight may report ahead of the driver.
const EVENT_QUEUE: usize = 64;
/// Progress updates not yet taken by the driver; it takes them ahead of every other report.
const PROGRESS_QUEUE: usize = 32;

/// What a run executes against.
#[derive(Clone)]
pub struct RunSetup {
    pub llm: Arc<dyn Llm>,
    pub tools_host: Option<HostRef>,
    pub host: Arc<dyn AgentCallbacks>,
    pub clock: Arc<dyn Clock>,
    pub conn_options: ApiConnectOptions,
    /// Captured at the `run()` entry.
    pub entry: Invocation,
}

/// The running invocation. Dropping it cancels the run; its work still settles.
pub struct ActiveRun {
    cancel: Option<oneshot::Sender<()>>,
    progress: Progress,
    usage: watch::Receiver<Usage>,
    /// The driver's owner: dropping the handle cancels the run through `cancel`, and the driver
    /// ends after its cleanup.
    _driver: Arc<TaskSet>,
}

/// Where a tool body's `ctx.update(message)` reaches its run: the host operation running the body
/// and the message. The host's loop never waits on it: an update finding the bounded queue full is
/// refused (the driver is behind by [`PROGRESS_QUEUE`] updates), and one whose call settled first
/// is dropped by the driver (the update raced the call's end).
#[derive(Clone)]
pub struct Progress(mpsc::Sender<(u64, String)>);

/// Why an update did not reach its run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressRefused {
    /// [`PROGRESS_QUEUE`] updates are waiting: this one is dropped.
    Full,
    Ended,
}

impl Progress {
    pub fn report(&self, operation: u64, message: String) -> Result<(), ProgressRefused> {
        self.0.try_send((operation, message)).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => ProgressRefused::Full,
            mpsc::error::TrySendError::Closed(_) => ProgressRefused::Ended,
        })
    }

    pub fn ended(&self) -> bool {
        self.0.is_closed()
    }
}

impl ActiveRun {
    /// Where the run's tool bodies report their progress.
    pub fn progress(&self) -> Progress {
        self.progress.clone()
    }

    /// The core's latest usage snapshot of the run, kept after it ended.
    pub fn usage(&self) -> watch::Receiver<Usage> {
        self.usage.clone()
    }

    /// Requests cancellation; idempotent. The run settles once its work reported back.
    pub fn cancel(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
    }
}

impl Drop for ActiveRun {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Starts one invocation on `runtime`; its updates arrive on the receiver.
pub fn start(
    runtime: &tokio::runtime::Handle,
    agent: AgentDefinition,
    input: RunInput,
    setup: RunSetup,
    span: ScopedSpan<AgentRunSpan>,
    updates: usize,
) -> Result<(ActiveRun, mpsc::Receiver<RunUpdate>), AgentError> {
    let input_id = input.invocation_id.clone();
    let run = AgentRun::new(agent, input)?.with_checkpoints();
    let (cancel, cancelled) = oneshot::channel();
    let (update_tx, update_rx) = mpsc::channel(updates.max(1));
    let (events_tx, events) = mpsc::channel(EVENT_QUEUE);
    let (progress_tx, progress) = mpsc::channel(PROGRESS_QUEUE);
    let (usage_tx, usage) = watch::channel(run.usage());
    let driver = Driver {
        usage: usage_tx,
        run,
        work: Execution::new(
            setup,
            runtime.clone(),
            Reporter { id: input_id, tx: events_tx },
            span,
        ),
        updates: update_tx,
        published: 0,
        finished: false,
    };
    let driving = Arc::new(TaskSet::default());
    let spawner = TokioSpawner::new(runtime.clone());
    driving.spawn(&spawner, "agent run", driver.drive(events, progress, cancelled));
    let handle = ActiveRun {
        cancel: Some(cancel),
        progress: Progress(progress_tx),
        usage,
        _driver: driving,
    };
    Ok((handle, update_rx))
}

struct Driver {
    run: AgentRun,
    /// Republished whenever the core's ledger moved.
    usage: watch::Sender<Usage>,
    work: Execution,
    updates: mpsc::Sender<RunUpdate>,
    /// The ledger revision `usage` holds.
    published: u64,
    finished: bool,
}

impl Driver {
    async fn drive(
        mut self,
        mut events: mpsc::Receiver<(String, execution::Report)>,
        mut progress: mpsc::Receiver<(u64, String)>,
        mut cancelled: oneshot::Receiver<()>,
    ) {
        let mut cancel_seen = false;
        let mut effects = self.run.handle(AgentRunEvent::Start, self.work.setup.clock.now());
        loop {
            if self.run.ledger().revision() != self.published {
                self.published = self.run.ledger().revision();
                self.usage.send_replace(self.run.usage());
            }
            for effect in effects.drain(..) {
                if let Some(update) = self.work.execute(effect).await {
                    self.finished |=
                        matches!(update, RunUpdate::Finished(_) | RunUpdate::Failed { .. });
                    let _ = self.updates.send(update).await;
                }
            }
            if self.finished && self.work.idle() {
                return;
            }
            let wake = self.run.wake_at();
            let clock = self.work.setup.clock.clone();
            let event = tokio::select! {
                biased;
                () = wait(clock.as_ref(), wake) => AgentRunEvent::Wake,
                _ = &mut cancelled, if !cancel_seen => { cancel_seen = true; AgentRunEvent::Cancel }
                Some((operation, message)) = progress.recv() => {
                    match self.work.progress(operation, message) {
                        Some(event) => event,
                        None => continue,
                    }
                }
                Some((_, report)) = events.recv() => self.work.report(report).0,
                _ = self.work.invoker.settled() => continue,
            };
            effects = self.run.handle(event, self.work.setup.clock.now());
        }
    }
}

pub(crate) async fn wait(clock: &dyn Clock, at: Option<Ts>) {
    match at {
        Some(at) => clock.wait_until(at).await,
        None => std::future::pending().await,
    }
}

mod runner;
#[cfg(test)]
#[path = "../agent_run_tests.rs"]
mod tests;
/// `result.expect`'s events: the assertions are the core's own, nothing runs.
pub use lk_agents_core::assertions::RunEvents;
pub use runner::{Runner, Updates};
