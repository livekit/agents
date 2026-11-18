use lk_agents_abi::a2a::turn::{HandlerOutput, HandlerPort, TurnEvent, TurnHost};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{BoxFuture, Clock, Ts};
use lk_agents_net::outbox::{Outbox, outbox};
use lk_agents_net::task::Cancellation;
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::Weak;
use tokio::sync::{mpsc, oneshot};

// The turn reads reports in order and sees the handler's output only once every report it
// awaited before returning was read.

/// Reports a handler may have awaited and the turn not yet read.
pub const REPORT_QUEUE: usize = 64;

enum Report {
    Checkpoint(oneshot::Sender<bool>),
    Event(TurnEvent),
    Session(SessionWrite),
    Live(ManagedWrite),
    Exact(ExactOutput),
}

/// One command capability under the partition lease, shared by every handler it opens.
/// This is deliberately local to the execution owner rather than a process-global registry.
#[derive(Default)]
pub(super) struct ManagedOwner {
    commands: Mutex<Option<Weak<dyn lk_agents_abi::session::SessionCommands>>>,
    execution: Mutex<Option<lk_agents_abi::OwnedTask>>,
    checkpoint: Option<Arc<super::checkpoint::Port>>,
}

impl ManagedOwner {
    pub(super) fn with_checkpoint(checkpoint: Arc<super::checkpoint::Port>) -> Self {
        Self { checkpoint: Some(checkpoint), ..Default::default() }
    }

    pub(super) fn commands(&self) -> Option<crate::session::CommandPort> {
        self.commands.lock().clone().and_then(|commands| commands.upgrade()).and_then(|commands| {
            (commands.as_ref() as &dyn std::any::Any)
                .downcast_ref::<crate::session::CommandPort>()
                .cloned()
        })
    }

    pub(super) async fn settled(&self, cancel: bool) -> bool {
        let task = self.execution.lock().take();
        let Some(task) = task else { return true };
        let commands = self.commands.lock().clone().and_then(|commands| commands.upgrade());
        let port = commands.as_ref().and_then(|commands| {
            (commands.as_ref() as &dyn std::any::Any)
                .downcast_ref::<crate::session::CommandPort>()
                .cloned()
        });
        let close = match port {
            Some(port) if cancel => {
                if let Some(live) = &port.live {
                    live.cancel.cancel();
                }
                true
            }
            Some(port) => port.close_live_owner().await.is_ok(),
            None => true,
        };
        task.await.is_ok() && close
    }
}

pub struct ManagedWrite {
    pub report: lk_agents_abi::text_session::LiveTurnReport,
    pub reply: oneshot::Sender<Result<(), lk_agents_abi::storage::StorageError>>,
}

enum PendingReplies {
    None,
    Checkpoint(oneshot::Sender<bool>),
    Managed(ManagedWrite),
    Both { checkpoint: oneshot::Sender<bool>, managed: ManagedWrite },
}

/// A private receipt travels with the existing report until its output handoff. The source
/// retains canonical history ownership; dropping this boundary ends every receipt reader.
pub struct ExactOutput {
    pub task_id: String,
    pub message: Box<lk_agents_abi::ChatMessage>,
    pub delivered:
        lk_agents_net::signal::Producer<Option<Result<(), lk_agents_abi::errors::HostError>>>,
}

pub struct SessionWrite {
    pub state: serde_json::Value,
    pub history: Vec<lk_agents_abi::ChatItem>,
    pub reply: oneshot::Sender<Result<(), lk_agents_abi::storage::StorageError>>,
}

/// What the handler reports through.
pub struct HandlerTurn {
    task_id: String,
    /// A turn that ended reads no more reports: a checkpoint then counts as a go-ahead and a
    /// session write as refused, through their dropped reply.
    reports: Outbox<Report>,
    policy: lk_agents_abi::a2a::options::Policy,
    origin: lk_agents_abi::text_session::LiveTurnOrigin,
    managed: Weak<ManagedOwner>,
    initial_history: Vec<lk_agents_abi::storage::ItemRow>,
    /// Admission work belongs to the handler, not its optional Python receipt reader.
    outputs: Weak<lk_agents_net::task::TaskSet>,
    spawner: Arc<dyn lk_agents_abi::Spawner>,
    /// The last ordered admission, including ones waiting for bounded report capacity.
    output_tail: Mutex<Option<lk_agents_net::signal::Cell<Option<()>>>>,
}

impl HandlerTurn {
    fn managed(&self) -> Result<Arc<ManagedOwner>, lk_agents_abi::storage::StorageError> {
        self.managed.upgrade().ok_or_else(|| {
            lk_agents_abi::storage::StorageError::Invalid(
                "the live partition owner has ended".into(),
            )
        })
    }
}

impl HandlerPort for HandlerTurn {
    fn inbox_policy(&self) -> lk_agents_abi::a2a::options::Policy {
        self.policy
    }

    fn live_origin(&self) -> Option<lk_agents_abi::text_session::LiveTurnOrigin> {
        (self.policy == lk_agents_abi::a2a::options::Policy::Live).then(|| self.origin.clone())
    }

    fn session_commands(
        &self,
    ) -> Result<
        Option<Arc<dyn lk_agents_abi::session::SessionCommands>>,
        lk_agents_abi::storage::StorageError,
    > {
        let managed = self.managed()?;
        let owner = managed.commands.lock().clone();
        Ok(owner.and_then(|owner| owner.upgrade()))
    }

    fn session_checkpoint_port(
        &self,
    ) -> Result<
        Option<Arc<dyn lk_agents_abi::text_session::SessionCheckpointPort>>,
        lk_agents_abi::storage::StorageError,
    > {
        if self.policy != lk_agents_abi::a2a::options::Policy::Live {
            return Ok(None);
        }
        let managed = self.managed()?;
        let Some(port) = &managed.checkpoint else { return Ok(None) };
        port.initialize(&self.initial_history)?;
        Ok(Some(port.clone()))
    }

    fn attach_session(
        &self,
        commands: Weak<dyn lk_agents_abi::session::SessionCommands>,
    ) -> Result<(), lk_agents_abi::storage::StorageError> {
        if self.policy != lk_agents_abi::a2a::options::Policy::Live {
            return Ok(());
        }
        let managed = self.managed()?;
        let mut active = managed.commands.lock();
        if let Some(previous) = active.as_ref()
            && previous.upgrade().is_some()
            && !Weak::ptr_eq(previous, &commands)
        {
            return Err(lk_agents_abi::storage::StorageError::Invalid(
                "the live partition already has a managed conversation owner".into(),
            ));
        }
        *active = Some(commands);
        Ok(())
    }

    fn retain_session(
        &self,
        task: lk_agents_abi::OwnedTask,
    ) -> Result<(), lk_agents_abi::storage::StorageError> {
        let managed = self.managed()?;
        let mut execution = managed.execution.lock();
        if execution.as_ref().is_some_and(|task| !task.is_finished()) {
            return Err(lk_agents_abi::storage::StorageError::Invalid(
                "the live partition already owns its managed execution".into(),
            ));
        }
        let previous = execution.replace(task);
        drop(execution);
        drop(previous);
        Ok(())
    }

    fn live_turn(
        &self,
        report: lk_agents_abi::text_session::LiveTurnReport,
    ) -> BoxFuture<
        '_,
        Result<lk_agents_abi::text_session::LiveTurnCommit, lk_agents_abi::storage::StorageError>,
    > {
        Box::pin(async move {
            let (reply, received) = oneshot::channel();
            self.wait_output_admissions().await;
            self.reports.send(Report::Live(ManagedWrite { report, reply })).await;
            let committed: lk_agents_abi::text_session::LiveTurnCommit = Box::pin(async move {
                received.await.map_err(|_| {
                    lk_agents_abi::storage::StorageError::Invalid(
                        "the live partition ended before committing its managed turn".into(),
                    )
                })?
            });
            Ok(committed)
        })
    }

    fn checkpoint(&self) -> BoxFuture<'_, bool> {
        Box::pin(async move {
            let (verdict, answered) = oneshot::channel();
            self.wait_output_admissions().await;
            self.reports.send(Report::Checkpoint(verdict)).await;
            answered.await.unwrap_or(true)
        })
    }
    fn save_session(
        &self,
        state: serde_json::Value,
        history: Vec<lk_agents_abi::ChatItem>,
    ) -> BoxFuture<'_, Result<serde_json::Value, lk_agents_abi::storage::StorageError>> {
        Box::pin(async move {
            let (reply, received) = oneshot::channel();
            self.wait_output_admissions().await;
            self.reports
                .send(Report::Session(SessionWrite { state: state.clone(), history, reply }))
                .await;
            received.await.map_err(|_| {
                lk_agents_abi::storage::StorageError::Invalid("the turn has ended".into())
            })??;
            Ok(state)
        })
    }
    fn progress(&self, text: String) -> BoxFuture<'_, ()> {
        self.progress_with_policy(text, true)
    }
    fn progress_with_policy(&self, text: String, reply_required: bool) -> BoxFuture<'_, ()> {
        let event = TurnEvent::Progress { task_id: self.task_id.clone(), text, reply_required };
        Box::pin(async move {
            self.wait_output_admissions().await;
            self.reports.send(Report::Event(event)).await;
        })
    }
    fn say(
        &self,
        message: lk_agents_abi::ChatMessage,
    ) -> BoxFuture<'static, Result<(), lk_agents_abi::errors::HostError>> {
        self.enqueue_exact(message)
    }
    fn token(&self, delta: String) -> BoxFuture<'_, ()> {
        let event = TurnEvent::Token { task_id: self.task_id.clone(), delta };
        Box::pin(async move {
            self.wait_output_admissions().await;
            self.reports.send(Report::Event(event)).await;
        })
    }
}

impl lk_agents_abi::a2a::turn::HandlerTurn for HandlerTurn {
    async fn checkpoint(&self) -> bool {
        HandlerPort::checkpoint(self).await
    }
    async fn save_session(
        &self,
        state: serde_json::Value,
        history: Vec<lk_agents_abi::ChatItem>,
    ) -> Result<(), lk_agents_abi::storage::StorageError> {
        HandlerPort::save_session(self, state, history).await.map(drop)
    }
    async fn progress(&self, text: String) {
        HandlerPort::progress(self, text).await
    }
    async fn token(&self, delta: String) {
        HandlerPort::token(self, delta).await
    }
}

/// What the turn reads next from its handler.
pub enum Read {
    Session(SessionWrite),
    Live(lk_agents_abi::text_session::LiveTurnReport),
    Checkpoint,
    Event(TurnEvent),
    Exact(ExactOutput),
    Output(Box<Result<HandlerOutput, ProviderError>>),
}

/// One open handler: its host call, its unread reports, the checkpoint awaiting a verdict.
pub struct Handler {
    call: Option<BoxFuture<'static, Result<HandlerOutput, ProviderError>>>,
    /// The call's result, until the reports it awaited before returning were read.
    output: Option<Result<HandlerOutput, ProviderError>>,
    reports: mpsc::Receiver<Report>,
    pending: PendingReplies,
    cancel: Arc<Cancellation>,
    _managed: Arc<ManagedOwner>,
    outputs: Arc<lk_agents_net::task::TaskSet>,
}

impl Handler {
    pub(super) fn cancel(&mut self) {
        self.cancel.cancel();
    }

    pub(super) fn returned(&self) -> bool {
        self.call.is_none() && self.output.is_none()
    }
    pub(super) fn awaiting_live_commit(&self) -> bool {
        matches!(self.pending, PendingReplies::Managed(_) | PendingReplies::Both { .. })
    }

    /// The handler's host call on `ctx`; it starts when first read.
    pub fn open(
        host: Arc<dyn TurnHost>,
        cx: crate::telemetry::Cx,
        ctx: lk_agents_abi::a2a::turn::TurnContext,
        spawner: Arc<dyn lk_agents_abi::Spawner>,
    ) -> Self {
        Self::open_managed(
            host,
            cx,
            ctx,
            lk_agents_abi::a2a::options::Policy::Queue,
            Arc::new(ManagedOwner::default()),
            spawner,
        )
    }

    pub(super) fn open_managed(
        host: Arc<dyn TurnHost>,
        cx: crate::telemetry::Cx,
        ctx: lk_agents_abi::a2a::turn::TurnContext,
        policy: lk_agents_abi::a2a::options::Policy,
        managed: Arc<ManagedOwner>,
        spawner: Arc<dyn lk_agents_abi::Spawner>,
    ) -> Self {
        let (reports, unread) = outbox(REPORT_QUEUE);
        let outputs = Arc::new(lk_agents_net::task::TaskSet::default());
        let turn = Arc::new(HandlerTurn {
            task_id: ctx.task_id.clone(),
            reports,
            policy,
            managed: Arc::downgrade(&managed),
            initial_history: ctx.history.clone(),
            outputs: Arc::downgrade(&outputs),
            spawner,
            output_tail: Mutex::new(None),
            origin: lk_agents_abi::text_session::LiveTurnOrigin {
                request_id: ctx.message.id().to_owned(),
                execution_id: ctx.task_id.clone(),
                task_ids: vec![ctx.task_id.clone()],
            },
        });
        let cancel = Arc::new(Cancellation::default());
        let signal = cancel.clone();
        let admitted = turn.clone();
        let call = Box::pin(async move {
            let result = host.handle(&cx, &ctx, turn, signal).await;
            // Unawaited exact outputs were still enqueued by this callback. Read them before
            // its final output, including admissions waiting for bounded report capacity.
            admitted.wait_output_admissions().await;
            result
        });
        Self {
            call: Some(call),
            output: None,
            reports: unread,
            pending: PendingReplies::None,
            cancel,
            _managed: managed,
            outputs,
        }
    }

    /// Answers the checkpoint the handler waits at, if any.
    pub fn answer(&mut self, should_yield: bool) {
        let old = std::mem::replace(&mut self.pending, PendingReplies::None);
        let (checkpoint, pending) = match old {
            PendingReplies::Checkpoint(checkpoint) => (Some(checkpoint), PendingReplies::None),
            PendingReplies::Both { checkpoint, managed } => {
                (Some(checkpoint), PendingReplies::Managed(managed))
            }
            pending => (None, pending),
        };
        self.pending = pending;
        if let Some(checkpoint) = checkpoint {
            let _ = checkpoint.send(should_yield);
        }
    }

    pub(super) fn commit_live(
        &mut self,
        turn: u64,
        result: Result<(), lk_agents_abi::storage::StorageError>,
    ) {
        let old = std::mem::replace(&mut self.pending, PendingReplies::None);
        self.pending = match old {
            PendingReplies::Managed(managed) if managed.report.turn == turn => {
                let _ = managed.reply.send(result);
                PendingReplies::None
            }
            PendingReplies::Both { checkpoint, managed } if managed.report.turn == turn => {
                let _ = managed.reply.send(result);
                PendingReplies::Checkpoint(checkpoint)
            }
            pending => pending,
        };
    }

    /// The handler's next report, or its output once every report before it was read; `None`
    /// once the output was read.
    pub async fn next(&mut self) -> Option<Read> {
        if let Some(call) = self.call.as_mut() {
            tokio::select! {
                biased;
                Some(report) = self.reports.recv() => return Some(self.read(report)),
                output = call => {
                    self.call = None;
                    self.output = Some(output);
                }
            }
        }
        // The call may return in the same poll as its last reports: those come first.
        let output = self.output.take()?;
        match self.reports.try_recv() {
            Ok(report) => {
                self.output = Some(output);
                Some(self.read(report))
            }
            Err(_) => Some(Read::Output(Box::new(output))),
        }
    }

    fn read(&mut self, report: Report) -> Read {
        match report {
            Report::Session(write) => Read::Session(write),
            Report::Live(write) => {
                let report = write.report.clone();
                let old = std::mem::replace(&mut self.pending, PendingReplies::None);
                self.pending = match old {
                    PendingReplies::Checkpoint(checkpoint) => {
                        PendingReplies::Both { checkpoint, managed: write }
                    }
                    PendingReplies::None => PendingReplies::Managed(write),
                    pending => {
                        let _ =
                            write.reply.send(Err(lk_agents_abi::storage::StorageError::Invalid(
                                "a managed turn is already awaiting its exact commit".into(),
                            )));
                        pending
                    }
                };
                Read::Live(report)
            }
            Report::Checkpoint(verdict) => {
                let old = std::mem::replace(&mut self.pending, PendingReplies::None);
                self.pending = match old {
                    PendingReplies::Managed(managed) => {
                        PendingReplies::Both { checkpoint: verdict, managed }
                    }
                    PendingReplies::None => PendingReplies::Checkpoint(verdict),
                    pending => {
                        let _ = verdict.send(true);
                        pending
                    }
                };
                Read::Checkpoint
            }
            Report::Event(event) => Read::Event(event),
            Report::Exact(output) => Read::Exact(output),
        }
    }

    /// Cancels a running handler and waits for its cleanup until `deadline`; `false` when it
    /// was still running then.
    pub async fn close(mut self, clock: &dyn Clock, deadline: Ts) -> bool {
        self.outputs.close();
        self.reports.close();
        // Receipts in unread reports must end before waiting for the host that may await one.
        while let Ok(report) = self.reports.try_recv() {
            drop(report);
        }
        self.pending = PendingReplies::None;
        let outputs_settled = self.outputs.wait_empty(clock, Some(deadline)).await;
        let Some(call) = self.call.take() else {
            return outputs_settled;
        };
        self.cancel.cancel();
        tokio::select! {
            biased;
            _ = call => outputs_settled,
            () = clock.wait_until(deadline) => false,
        }
    }
}

#[path = "output.rs"]
mod output;

#[cfg(test)]
#[path = "handler_tests.rs"]
mod tests;
