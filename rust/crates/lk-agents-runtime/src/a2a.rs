//! The A2A text door's turns: [`Turn::run`] takes one incoming message to its answer by executing
//! the operations of `lk_agents_core::a2a::task::TaskMachine` against storage and the host's
//! handler, and feeding each typed result back. Every decision (deduplication, arrival policy,
//! claim, drain, the pending question, checkpoints, settlement, commit and failure outcomes) is
//! the machine's; this file only executes.

pub mod door;

use std::sync::Arc;

use serde_json::Value;
use tokio_util::task::TaskTracker;

use lk_agents_abi::storage::StreamInfo;
use lk_agents_abi::{Clock, Closed, HOST_TEARDOWN_TIMEOUT, StreamSink, StreamSource};
use lk_agents_core::a2a::task::{
    Answer, Completion, Effect, Event, HandlerEvent, Operation, TaskMachine,
};
use lk_agents_core::a2a::{
    HandlerOutput, Message, Policy, TurnContext, TurnEvent, TurnHost, TurnOptions, TurnResult,
    Verdict, caller_history_of, failed_message, host_error, resolve_policy, result_for,
    unpersisted_failed,
};
use lk_agents_storage::db::{Receive, ReceiveInput};
use lk_agents_storage::run::{
    ItemRow, REASON_STORAGE_UNAVAILABLE, REASON_WORKER_LOST, TaskRow, TaskState,
};
use lk_agents_storage::{StorageError, TaskWrite};

use crate::storage::{
    CommitOptions, Lane, Lease, Run, RunOptions, Thread, now_ms, now_secs, shortuuid,
};

/// Who watches a turn from outside: its response stream.
#[async_trait::async_trait]
pub trait Observer: Send + Sync {
    /// The message's task is durable, before admission waits or application work.
    async fn accepted(&self, task: &TaskRow) -> Result<(), StorageError>;
    /// A handler's progress or answer token.
    async fn event(&self, event: &TurnEvent) -> Result<(), StorageError>;
}

/// How many task rows the handler's context carries.
const TASKS_IN_CONTEXT: i64 = 50;
/// How many runs a refusal reads to name the running task.
const RUNS_FOR_REJECT: usize = 50;

/// The handler's turn: its checkpoint verdicts in, its events out.
type Handler = (Box<dyn StreamSink<Verdict>>, Box<dyn StreamSource<TurnEvent>>);

/// One message's live resources while its machine runs: the lane, the open run, a kept lease,
/// the handler's stream.
struct Call {
    lane: Lane,
    run: Option<Run>,
    lease: Option<Lease>,
    handler: Option<Handler>,
    /// A delegating caller's conversation, when the message carried one.
    caller_history: Option<Value>,
}

impl Call {
    fn run(&mut self) -> Result<&mut Run, StorageError> {
        self.run.as_mut().ok_or_else(|| StorageError::Invalid("no A2A run is open".into()))
    }
}

/// One text door's turns: its lane, its options, its host.
#[derive(Clone)]
pub struct Turn {
    thread: Thread,
    lane: String,
    host: Arc<dyn TurnHost>,
    /// The host stream's clock, for the bound on closing it.
    clock: Arc<dyn Clock>,
    opts: TurnOptions,
    stream: Option<StreamInfo>,
    observer: Option<Arc<dyn Observer>>,
    host_context: Option<lk_agents_abi::HostRef>,
    /// The resolvers of commits left uncertain, owned by the door.
    resolvers: TaskTracker,
}

fn failed(task_id: &str, reason: &str) -> a2a::Message {
    failed_message(task_id, reason, shortuuid("msg_"))
}

impl Turn {
    pub fn new(
        thread: Thread,
        lane: &str,
        host: Arc<dyn TurnHost>,
        clock: Arc<dyn Clock>,
        options: TurnOptions,
    ) -> Self {
        Self {
            thread,
            lane: lane.to_owned(),
            host,
            clock,
            opts: options,
            stream: None,
            observer: None,
            host_context: None,
            resolvers: TaskTracker::new(),
        }
    }

    pub fn with_observer(mut self, observer: Arc<dyn Observer>) -> Self {
        self.observer = Some(observer);
        self
    }

    pub fn with_owner(mut self, owner: TaskTracker) -> Self {
        self.resolvers = owner;
        self
    }

    /// Each invocation supplies its own host execution context; the endpoint proxy is shared.
    pub fn with_context(mut self, context: lk_agents_abi::HostRef) -> Self {
        self.host_context = Some(context);
        self
    }

    /// The response stream is independent of task identity and is recorded with submission.
    pub fn with_stream(mut self, stream: StreamInfo) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn lane(&self) -> &str {
        &self.lane
    }

    pub fn thread(&self) -> &Thread {
        &self.thread
    }

    pub fn options(&self) -> &TurnOptions {
        &self.opts
    }

    pub fn policy_for(&self, channel: &str) -> Policy {
        resolve_policy(&self.opts.policy, channel)
    }

    /// One incoming message through its policy. `task_id` is the answer stream's id when the
    /// caller created one. [`StorageError::Rejected`] under `reject`,
    /// [`StorageError::Cancelled`] when the host cancelled the handler.
    pub async fn run(
        &self,
        message: Value,
        task_id: Option<String>,
    ) -> Result<TurnResult, StorageError> {
        let message: a2a::Message = serde_json::from_value(message)
            .map_err(|e| StorageError::Invalid(format!("an A2A v1 message: {e}")))?;
        self.run_message(message, task_id).await
    }

    /// Execute an already decoded native protocol message.
    pub async fn run_message(
        &self,
        message: a2a::Message,
        task_id: Option<String>,
    ) -> Result<TurnResult, StorageError> {
        // the delegation's `chat_ctx` part, which `Message::from_a2a` strips
        let caller_history = caller_history_of(&message);
        let epoch = tokio::time::Instant::now();
        let wall = now_secs();
        let msg = Message::from_a2a(&message, wall, None)?;
        let task_id = task_id.unwrap_or_else(|| shortuuid("AST_"));
        let mut machine = TaskMachine::new(
            self.lane.clone(),
            msg.clone(),
            task_id,
            self.stream.clone(),
            &self.opts,
            wall,
        );
        let lane = self.thread.lane(&self.lane);
        let mut call = Call { lane, run: None, lease: None, handler: None, caller_history };
        let mut effects = machine.handle(Event::Start, epoch.elapsed());
        loop {
            match effects.pop() {
                Some(Effect::Execute { id, operation }) => {
                    let result = self.execute(&mut call, &machine, operation).await;
                    if result.is_err() {
                        self.close_handler(&mut call).await;
                    }
                    effects = machine.handle(Event::Completed { id, result }, epoch.elapsed());
                }
                Some(Effect::Finished(answer)) => {
                    return answer.map(|answer| self.result(&msg, answer));
                }
                None => {
                    let at = machine.wake_at().ok_or_else(|| {
                        StorageError::Invalid("the A2A task stalled with nothing to wait on".into())
                    })?;
                    tokio::time::sleep_until(epoch + at).await;
                    effects = machine.handle(Event::Wake, epoch.elapsed());
                }
            }
        }
    }

    /// The machine's answer as the caller's result: a row it could not read is the task as this
    /// worker left it, failed with the machine's reason or still `working` while it resolves.
    fn result(&self, msg: &Message, answer: Answer) -> TurnResult {
        let task = answer.task.unwrap_or_else(|| match answer.reason {
            Some(reason) => unpersisted_failed(
                &answer.task_id,
                &self.lane,
                Some(msg),
                reason,
                now_ms(),
                shortuuid("msg_"),
            ),
            None => TaskRow {
                status: TaskState::Working,
                ..self.submitted(msg, &answer.task_id).row(&self.lane)
            },
        });
        let mut result = result_for(task, self.thread.id(), answer.run_id.as_deref());
        if let Some(answered_by) = answer.answered_by {
            result.answered_by = answered_by;
            result.output = answer.output;
        }
        result
    }

    fn submitted(&self, msg: &Message, task_id: &str) -> TaskWrite {
        TaskWrite {
            task_id: task_id.to_owned(),
            status: TaskState::Submitted,
            message_id: Some(msg.message_id.clone()),
            name: None,
            data: lk_agents_storage::schema::TaskData {
                history: vec![msg.to_a2a()],
                ..Default::default()
            },
            internal: self.stream.clone().map(|stream| lk_agents_storage::schema::TaskInternal {
                ask: None,
                streams: vec![stream],
            }),
        }
    }

    async fn execute(
        &self,
        call: &mut Call,
        machine: &TaskMachine,
        operation: Operation,
    ) -> Result<Completion, StorageError> {
        let db = call.lane.db().clone();
        Ok(match operation {
            Operation::FindReceipt(message_id) => {
                Completion::Task(db.task_for_message(&message_id).await?)
            }
            Operation::FindTarget(task_id) => Completion::Task(self.thread.task(&task_id).await?),
            Operation::ReadTask(task_id) => Completion::Task(call.lane.task(&task_id).await?),
            Operation::Submit(task) => Completion::Submitted(db.submit(&task).await?),
            Operation::Receive { message, task } => {
                let input = ReceiveInput {
                    task_id: &task.task_id,
                    message: &message,
                    data: &task.data,
                    internal: task.internal.as_ref(),
                };
                let (received, lease) = db.receive(self.thread.holder(), input).await?;
                call.lease = lease;
                match received {
                    Receive::Duplicate(existing) => Completion::Received {
                        duplicate: Some(existing.unwrap_or(task.task_id)),
                        leased: false,
                    },
                    Receive::Received(_) => {
                        Completion::Received { duplicate: None, leased: call.lease.is_some() }
                    }
                }
            }
            Operation::Accepted(task_id) => {
                if let Some(observer) = &self.observer {
                    let task = call.lane.task(&task_id).await?.ok_or_else(|| {
                        StorageError::Unavailable("an accepted task was not durable".into())
                    })?;
                    observer.accepted(&task).await?;
                }
                Completion::Done
            }
            Operation::Open {
                run_id,
                task_id,
                channel,
                exclusive,
                wait,
                interrupt,
                pickup,
                resume,
            } => {
                let options = RunOptions {
                    exclusive,
                    wait,
                    interrupt,
                    channel,
                    pickup,
                    wait_timeout: machine.wait_timeout(),
                    ..RunOptions::new(run_id, task_id)
                };
                let run = match call.lease.take().filter(|_| resume) {
                    Some(lease) => call.lane.resume(options, lease).await?,
                    None => call.lane.open(options).await?,
                };
                let history_end = run.history.iter().map(ItemRow::created_at).fold(0.0, f64::max);
                let opened = Completion::Opened {
                    closed: run.closed,
                    picked_up: run.picked_up,
                    history_end,
                };
                call.run = Some(run);
                opened
            }
            Operation::ReadInbox => Completion::Inbox(db.inbox(true).await?),
            Operation::Pickup(task_id) => Completion::Picked(db.pickup(&task_id).await?),
            Operation::FailTask { task_id, reason } => {
                if db.pickup(&task_id).await?.is_some() {
                    db.fail_orphan(&task_id, &failed(&task_id, reason)).await?;
                    log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = task_id.as_str(); "the task fails {reason}");
                }
                Completion::Done
            }
            Operation::RecoverOrphans { skip } => {
                let run = call.run()?;
                for row in run.tasks.working().await? {
                    if !skip.contains(&row.task_id) {
                        let failed = failed(&row.task_id, REASON_WORKER_LOST);
                        run.tasks.lane().fail_orphan(&row.task_id, &failed).await?;
                        log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = row.task_id.as_str(); "a task whose worker died fails worker-lost");
                    }
                }
                Completion::Done
            }
            Operation::ReadPending(named) => {
                Completion::Task(call.run()?.tasks.pending_ask(named.as_deref()).await?)
            }
            Operation::ReadTasks(task_ids) => {
                let mut rows = Vec::new();
                for task_id in task_ids {
                    rows.extend(call.lane.task(&task_id).await?);
                }
                Completion::Tasks(rows)
            }
            Operation::Stage(staged) => {
                let run = call.run()?;
                for edit in staged.edits {
                    edit.apply(&mut run.history);
                }
                for t in staged.transitions {
                    run.tasks.write(&t.task_id, t.status, t.patch).await?;
                }
                for (ty, payload) in staged.events {
                    run.events.append(&ty, payload);
                }
                if let Some(end) = staged.end {
                    if end.state.is_some() {
                        run.state = end.state;
                    }
                    run.scope_writes = end.scope_writes;
                    run.usage.add(&end.usage);
                    run.outcome = end.outcome;
                    run.reason = end.reason;
                }
                Completion::Done
            }
            Operation::Handle(handle) => {
                let lane = call.lane.name().to_owned();
                let caller_history = call.caller_history.clone();
                let run = call.run()?;
                let ctx = TurnContext {
                    thread_id: self.thread.id().to_owned(),
                    context_scope_id: format!("root:{lane}"),
                    lane,
                    message: handle.message.payload,
                    batch: handle.batch,
                    history: run.history.clone(),
                    state: run.state.clone(),
                    context_scopes: run.context_scopes.clone(),
                    read_only: run.lease().is_none(),
                    task_id: handle.task_id,
                    tasks: run.tasks.recent(TASKS_IN_CONTEXT).await?,
                    caller_history,
                    answer_tool: handle.answer_tool,
                };
                let mut host_ctx = lk_agents_abi::a2a::turn::TurnContext::try_from(ctx)?;
                host_ctx.host_context = self.host_context.clone();
                call.handler = Some(self.host.handle(&host_ctx).await.map_err(host_error)?);
                Completion::Done
            }
            Operation::Next(verdict) => {
                let event = self.next_event(call, verdict).await?;
                if !matches!(event, HandlerEvent::Checkpoint) {
                    self.close_handler(call).await;
                }
                Completion::Event(event)
            }
            Operation::Checkpoint(task_ids) => {
                let holder = call.run()?.holder.clone();
                let (mut other_waiting, mut canceled) = (false, false);
                for task_id in &task_ids {
                    let (waiter, status) = call.lane.db().checkpoint(task_id).await?;
                    other_waiting |= waiter.is_some_and(|w| !w.is_empty() && w != holder);
                    canceled |= status == Some(TaskState::Canceled);
                }
                Completion::Checkpoint { other_waiting, canceled }
            }
            Operation::Commit { answer_task, keep_lease, drained_id } => {
                let run = call.run()?;
                let committed = run
                    .commit(CommitOptions {
                        answer_task,
                        keep_lease,
                        drained_id,
                        ..Default::default()
                    })
                    .await?;
                let run_id = run.run_id.clone();
                if keep_lease {
                    call.lease = run.lease().cloned();
                }
                call.run = None;
                Completion::Committed { task: committed.task, run_id }
            }
            Operation::Abandon { outcome, reason } => {
                self.close_handler(call).await;
                if let Some(mut run) = call.run.take() {
                    run.abandon(outcome, reason).await?;
                }
                Completion::Done
            }
            Operation::ResolveLater { task_id } => {
                let run = call
                    .run
                    .take()
                    .ok_or_else(|| StorageError::Invalid("no A2A run is open".into()))?;
                self.resolve_later(call.lane.clone(), run, task_id);
                Completion::Done
            }
            Operation::CloseRun => {
                if let Some(mut run) = call.run.take() {
                    run.commit(CommitOptions::default()).await?;
                }
                Completion::Done
            }
            Operation::ReleaseDrained => {
                let released = match &call.lease {
                    Some(lease) => lease.release_if_drained().await?,
                    None => true,
                };
                if released {
                    call.lease = None;
                }
                Completion::Released(released)
            }
            Operation::RunningTask => {
                let runs = call.lane.runs(RUNS_FOR_REJECT).await?;
                Completion::RunningTask(
                    runs.into_iter().find(|r| r.interrupted()).and_then(|r| r.task_id),
                )
            }
        })
    }

    /// Hands a run whose commit stayed uncertain to a background resolver: an authoritative
    /// rejection fails its task `storage-unavailable`; a commit found committed needs nothing.
    fn resolve_later(&self, lane: Lane, run: Run, answer: String) {
        self.resolvers.spawn(async move {
            match run.resolve_later().await {
                Some(Err(StorageError::VersionConflict { .. })) => {
                    let failed = failed(&answer, REASON_STORAGE_UNAVAILABLE);
                    if let Err(e) = lane.db().fail_orphan(&answer, &failed).await {
                        log::warn!(target: "livekit.agents", taskId = answer.as_str(); "a rejected commit's task could not be failed: {e}");
                    }
                }
                Some(Err(e)) => {
                    log::warn!(target: "livekit.agents", taskId = answer.as_str(); "a commit stayed unresolved: {e}");
                }
                Some(Ok(_)) | None => {}
            }
        });
    }

    /// Answers the handler's last checkpoint, then reads its next checkpoint or its output; a
    /// lost lease ends the read with `LeaseLost`.
    async fn next_event(
        &self,
        call: &mut Call,
        verdict: Option<Verdict>,
    ) -> Result<HandlerEvent, StorageError> {
        let lease = call.run.as_ref().and_then(Run::lease).cloned();
        let (verdicts, events) = call
            .handler
            .as_mut()
            .ok_or_else(|| StorageError::Invalid("no A2A handler is open".into()))?;
        if let Some(verdict) = verdict {
            verdicts.push(verdict).map_err(host_error)?;
        }
        let read = async {
            loop {
                match events.next().await.map_err(host_error)? {
                    Some(TurnEvent::Checkpoint) => return Ok(HandlerEvent::Checkpoint),
                    Some(event @ (TurnEvent::Progress { .. } | TurnEvent::Token { .. })) => {
                        if let Some(observer) = &self.observer {
                            observer.event(&event).await?;
                        }
                    }
                    Some(TurnEvent::Output { output }) => {
                        let out = HandlerOutput::try_from(output)?;
                        if let Some(err) = &out.error {
                            log::error!(target: "livekit.agents", lane = self.lane.as_str(); "the a2a_session handler raised; the task fails: {err}");
                        }
                        return Ok(HandlerEvent::Output(Box::new(out)));
                    }
                    None => {
                        return Err(StorageError::Invalid(
                            "the handler's stream ended without its output".into(),
                        ));
                    }
                }
            }
        };
        match lease {
            Some(lease) => tokio::select! {
                event = read => event,
                e = lease.lost() => Err(e),
            },
            None => read.await,
        }
    }

    /// Closes the handler's stream, if one is open, within the host teardown bound.
    async fn close_handler(&self, call: &mut Call) {
        let Some((_, events)) = call.handler.take() else { return };
        let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        if events.close(deadline).await == Closed::TimedOut {
            log::warn!(target: "livekit.agents", lane = self.lane.as_str(); "the handler's stream did not close within its deadline; its host task is still running");
        }
    }
}
