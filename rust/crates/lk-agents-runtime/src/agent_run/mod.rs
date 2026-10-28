//! One agent invocation executed: the core's `AgentRun` stepped one event at a time, its model
//! requests run against the agent's `Llm`, its host callbacks through [`HostInvoker`], its
//! deadline on the clock. Updates (stream events, then the settled result) reach the consumer
//! through a bounded queue the driver waits on; nothing is dropped.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::agent::AgentDefinition;
use lk_agents_abi::agent::run::{Output, RunError, RunSettled, RunUpdate, ValidationId};
use lk_agents_abi::agent::{AgentHost, Invocation, PreparedArguments, ValidatedOutput};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::tool_executor::Approval;
use lk_agents_abi::llm::tool_executor::INTERNAL_ERROR_OUTPUT;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::utils::host_ref::HostRef;
use lk_agents_abi::{ApiConnectOptions, ChatRequest, Clock, FunctionCall, Llm, Ts};
use lk_agents_core::agent_run::{
    CallId, CheckpointId, Effect, Event, OutputVerdict, RenderId, RequestId,
};
use lk_agents_core::{AgentError, AgentRun, RunInput, approval};
use tokio::sync::{mpsc, oneshot};

pub(crate) mod execution;
use execution::{Execution, Reporter};

/// Events the work in flight may report ahead of the driver.
const EVENT_QUEUE: usize = 64;
/// Progress updates not yet taken by the driver; it takes them ahead of every other report.
const PROGRESS_QUEUE: usize = 32;

/// What a run executes against.
#[derive(Clone)]
pub struct Ports {
    pub llm: Arc<dyn Llm>,
    pub host: Arc<dyn AgentHost>,
    pub clock: Arc<dyn Clock>,
    pub conn_options: ApiConnectOptions,
    /// Captured at the `run()` entry.
    pub entry: Invocation,
}

/// The running invocation. Dropping it cancels the run; its work still settles.
pub struct AgentRunHandle {
    cancel: Option<oneshot::Sender<()>>,
    progress: Progress,
    model_error: oneshot::Receiver<ProviderError>,
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

impl AgentRunHandle {
    /// Where the run's tool bodies report their progress.
    pub fn progress(&self) -> Progress {
        self.progress.clone()
    }

    /// The provider's failure a run that failed on its model ended with, sent before its
    /// `Failed` update: the caller gets the provider's own error, not only its message.
    pub fn model_error(&mut self) -> Option<ProviderError> {
        self.model_error.try_recv().ok()
    }

    /// Requests cancellation; idempotent. The run settles once its work reported back.
    pub fn cancel(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
    }
}

impl Drop for AgentRunHandle {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Starts one invocation on `runtime`; its updates arrive on the receiver.
pub fn start(
    runtime: &tokio::runtime::Handle,
    agent: AgentDefinition,
    input: RunInput,
    ports: Ports,
    updates: usize,
) -> Result<(AgentRunHandle, mpsc::Receiver<RunUpdate>), AgentError> {
    let input_id = input.invocation_id.clone();
    let run = AgentRun::new(agent, input)?.with_checkpoints();
    let (cancel, cancelled) = oneshot::channel();
    let (update_tx, update_rx) = mpsc::channel(updates.max(1));
    let (events_tx, events) = mpsc::channel(EVENT_QUEUE);
    let (progress_tx, progress) = mpsc::channel(PROGRESS_QUEUE);
    let (model_error_tx, model_error) = oneshot::channel();
    let driver = Driver {
        run,
        work: Execution::new(
            ports,
            runtime.clone(),
            Reporter {
                id: input_id,
                tx: events_tx,
            },
        ),
        updates: update_tx,
        model_error: Some(model_error_tx),
        finished: false,
    };
    runtime.spawn(driver.drive(events, progress, cancelled));
    let handle = AgentRunHandle {
        cancel: Some(cancel),
        progress: Progress(progress_tx),
        model_error,
    };
    Ok((handle, update_rx))
}

struct Driver {
    run: AgentRun,
    work: Execution,
    updates: mpsc::Sender<RunUpdate>,
    model_error: Option<oneshot::Sender<ProviderError>>,
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
        let mut effects = self.run.handle(Event::Start, self.work.ports.clock.now());
        loop {
            for effect in effects.drain(..) {
                if let Some(update) = self.work.execute(effect).await {
                    self.finished |=
                        matches!(update, RunUpdate::Finished(_) | RunUpdate::Failed(_));
                    if let RunUpdate::Failed(RunError::Model(_)) = &update
                        && let (Some(error), Some(tx)) =
                            (self.work.failure.take(), self.model_error.take())
                    {
                        let _ = tx.send(error);
                    }
                    let _ = self.updates.send(update).await;
                }
            }
            if self.finished && self.work.idle() {
                return;
            }
            let wake = self.run.wake_at();
            let clock = self.work.ports.clock.clone();
            let event = tokio::select! {
                biased;
                () = wait(clock.as_ref(), wake) => Event::Wake,
                _ = &mut cancelled, if !cancel_seen => { cancel_seen = true; Event::Cancel }
                Some((operation, message)) = progress.recv() => {
                    match self.work.progress(operation, message) {
                        Some(event) => event,
                        None => continue,
                    }
                }
                Some((_, report)) = events.recv() => self.work.report(report).0,
                _ = self.work.invoker.settled() => continue,
            };
            effects = self.run.handle(event, self.work.ports.clock.now());
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
mod tests;
/// `result.expect`'s events: the assertions are the core's own, nothing runs.
pub use lk_agents_core::assertions::RunEvents;
pub use runner::{Runner, Updates};
