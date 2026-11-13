pub mod door;
pub mod handler;

use crate::telemetry;
use handler::{Handler, Read};
use lk_agents_abi::a2a::metadata::REASON_KEY;
use lk_agents_abi::a2a::turn::Turn as TurnContract;
use lk_agents_abi::storage::thread::{Backoff, TaskReason, TaskWrite};
use lk_agents_abi::{Clock, HOST_TEARDOWN_TIMEOUT, Spawner};
use lk_agents_core::a2a::task::{
    A2aTaskEffect, A2aTaskEvent, Access, Completion, Handle, HandlerEvent, Operation, PartitionRun,
    RunEnd, Stage, TaskMachine, rejected_commit,
};
use lk_agents_core::a2a::{
    HandlerOutput, Message, Policy, TurnContext, TurnHost, TurnOptions, TurnResult,
    caller_history_of, host_error, resolve_policy,
};
use lk_agents_net::env::{ProcessEnv, ProcessError};
use lk_agents_net::task::{Leave, TaskSet};
use lk_agents_storage::thread::RESOLVE_LATER;
use lk_agents_storage::thread::db::{Receive, ReceiveInput};
use lk_agents_storage::thread::run::ItemRow;
use lk_agents_storage::{
    CommitOptions, Lease, LeaseAfterCommit, Partition, PartitionAccess, Run, RunOptions, Thread,
    WhenHeld, shortuuid,
};
use lk_agents_storage::{Holder, StorageError};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// One message's live resources while its machine runs: the partition, the open run, a kept lease,
/// the open handler, the task it submitted.
struct Call {
    partition_key: Partition,
    /// The request's lease owner: its receive, runs and drain carry it.
    holder: Holder,
    run: Option<Run>,
    lease: Option<Lease>,
    handler: Option<Handler>,
    submitted: Option<Submitted>,
    /// Where the turn's background work (a cancel, a commit resolver) runs.
    spawner: Arc<dyn Spawner>,
}

/// The `submitted` row a turn writes, owned from before its insert until the turn answers it.
/// Dropped unanswered (the client dropped the turn, or an error ended it) it is canceled in the
/// background, through outages: no task is left submitted with nobody to run it.
struct Submitted {
    partition_key: Partition,
    task_id: String,
    /// Keeps the thread's connection open for the cancel.
    open: Option<Leave>,
    clock: Arc<dyn Clock>,
    resolvers: Arc<TaskSet>,
    spawner: Arc<dyn Spawner>,
    answered: bool,
}

impl Drop for Submitted {
    fn drop(&mut self) {
        if self.answered {
            return;
        }
        let (partition_key, task_id, clock) =
            (self.partition_key.clone(), self.task_id.clone(), self.clock.clone());
        let open = self.open.take();
        self.resolvers.spawn(&*self.spawner, "a2a task cancel", async move {
            let _open = open;
            cancel_unanswered(partition_key, task_id, clock).await;
        });
    }
}

/// Cancels a task its turn left unanswered, `caller-canceled`; a terminal row stays as it is. A
/// database that cannot be reached is tried again for up to [`RESOLVE_LATER`].
async fn cancel_unanswered(partition_key: Partition, task_id: String, clock: Arc<dyn Clock>) {
    let reason = Value::String(TaskReason::CallerCanceled.as_str().to_owned());
    let metadata = serde_json::Map::from_iter([(REASON_KEY.to_owned(), reason)]);
    let epoch = clock.now();
    let mut backoff = Backoff::new(Duration::ZERO, RESOLVE_LATER);
    loop {
        let error = match partition_key.cancel_task(&task_id, metadata.clone()).await {
            Ok(_) => return,
            Err(error) => error,
        };
        let retry = matches!(error, StorageError::Unavailable(_) | StorageError::Uncertain(_));
        let next = backoff.next(clock.now().saturating_duration_since(epoch)).filter(|_| retry);
        let Some(at) = next else {
            log::warn!(target: lk_agents_abi::LOG_TARGET, taskId = task_id.as_str(); "an unanswered task could not be canceled: {error}");
            return;
        };
        clock.wait_until(epoch.saturating_add(at)).await;
    }
}

impl Call {
    fn run(&mut self) -> Result<&mut Run, StorageError> {
        self.run.as_mut().ok_or_else(|| StorageError::Invalid("no A2A run is open".into()))
    }

    /// Opens the run as `access` takes the partition; a kept lease resumes it.
    async fn open_run(
        &mut self,
        access: Access,
        options: RunOptions,
    ) -> Result<Completion, StorageError> {
        let partition_access = match access {
            Access::Concurrent => PartitionAccess::Concurrent,
            Access::Exclusive(when_held) => PartitionAccess::Exclusive(when_held),
            Access::Kept => PartitionAccess::Exclusive(WhenHeld::Reject),
        };
        let options = RunOptions { access: partition_access, ..options };
        let run = match (access, self.lease.take()) {
            (Access::Kept, Some(lease)) => self.partition_key.resume(options, lease).await?,
            _ => self.partition_key.open(options, &self.holder).await?,
        };
        let history_end = run.history().iter().map(ItemRow::created_at).fold(0.0, f64::max);
        let opened =
            Completion::Opened { closed: run.closed(), picked_up: run.picked_up(), history_end };
        self.run = Some(run);
        Ok(opened)
    }

    /// Applies a turn's staged edits, task rows, journal entries and end to the open run.
    async fn stage(&mut self, staged: Stage) -> Result<Completion, StorageError> {
        let run = self.run()?;
        for edit in staged.edits {
            edit.apply(run.history_mut());
        }
        for t in staged.transitions {
            run.tasks().write(&t.task_id, t.status, t.patch).await?;
        }
        for event in staged.events {
            run.events().append(event);
        }
        if let Some(end) = staged.end {
            end_run(run, end);
        }
        Ok(Completion::Done)
    }

    async fn checkpoint(&mut self, task_ids: &[String]) -> Result<Completion, StorageError> {
        let mut rows = Vec::with_capacity(task_ids.len());
        for task_id in task_ids {
            rows.push(self.partition_key.checkpoint(task_id).await?);
        }
        Ok(Completion::Checkpoint { statuses: rows })
    }

    /// Commits the open run; a kept lease stays on the call.
    async fn commit(&mut self, options: CommitOptions) -> Result<Completion, StorageError> {
        let lease = options.lease;
        let run = self.run()?;
        let committed = run.commit(options).await?;
        let run_id = run.id().to_owned();
        if lease == LeaseAfterCommit::Keep {
            self.lease = run.lease().cloned();
        }
        self.run = None;
        Ok(Completion::Committed { task: committed.task, run_id })
    }

    async fn release_drained(&mut self) -> Result<Completion, StorageError> {
        let released = match &self.lease {
            Some(lease) => lease.release_if_drained().await?,
            None => true,
        };
        if released {
            self.lease = None;
        }
        Ok(Completion::Released(released))
    }
}

/// What a settled turn leaves for the commit; a `None` state keeps the loaded one.
fn end_run(run: &mut Run, end: RunEnd) {
    if let Some(state) = end.state {
        run.set_state(state);
    }
    run.stage_scopes(end.scope_writes);
    run.stage_statements(end.app_writes);
    run.set_usage(end.usage);
    run.end_as(end.outcome, end.reason);
}

/// One text door's turns: its partition, its options, its host.
#[derive(Clone)]
pub struct Turn {
    server: Option<std::sync::Weak<dyn lk_agents_abi::worker::server_host::ServerConfiguration>>,
    namespace: Option<Arc<lk_agents_storage::storage::Namespace>>,
    endpoint: Option<String>,
    thread: Thread,
    /// The label of each request's lease owner: this process's lease holder id.
    holder: String,
    partition_key: String,
    host: Arc<dyn TurnHost>,
    /// The turn's clock: the machine's waits and the bound on closing the host stream.
    clock: Arc<dyn Clock>,
    opts: TurnOptions,
    /// The response stream, which watches the turn from outside; its identity is independent
    /// of the task's and recorded with submission.
    publisher: Option<Arc<door::stream::Publisher>>,
    host_context: Option<lk_agents_abi::HostRef>,
    /// The span the handler runs in: the request's `a2a_request`.
    span: Option<telemetry::SpanContext>,
    /// The resolvers of commits left uncertain, owned by the door.
    resolvers: Arc<TaskSet>,
    /// The executor each run's background work spawns on, read when the run starts.
    spawner: fn() -> Result<Arc<dyn Spawner>, ProcessError>,
}

impl Turn {
    pub fn new(
        thread: Thread,
        holder: &str,
        partition_key: &str,
        host: Arc<dyn TurnHost>,
        clock: Arc<dyn Clock>,
        spawner: fn() -> Result<Arc<dyn Spawner>, ProcessError>,
        options: TurnOptions,
    ) -> Self {
        Self {
            server: None,
            namespace: None,
            endpoint: None,
            thread,
            holder: holder.to_owned(),
            partition_key: partition_key.to_owned(),
            host,
            clock,
            opts: options,
            publisher: None,
            host_context: None,
            span: None,
            resolvers: Arc::default(),
            spawner,
        }
    }

    /// Each invocation supplies its own host execution context; the endpoint proxy is shared.
    pub fn with_context(mut self, context: lk_agents_abi::HostRef) -> Self {
        self.host_context = Some(context);
        self
    }

    pub fn partition_key(&self) -> &str {
        &self.partition_key
    }

    pub fn policy_for(&self, channel: &str) -> Policy {
        resolve_policy(&self.opts.policy, channel)
    }

    /// One incoming message through its policy. `task_id` is supplied by the caller when known. [`StorageError::Rejected`] under `reject`,
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
        let _ = caller_history_of(&message)?;
        let epoch = self.clock.now();
        let elapsed = || self.clock.now().saturating_duration_since(epoch);
        // the run's clock, as every time the machine reads: the DST's is virtual
        let wall = self.clock.wall().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        let wall = wall.as_secs_f64();
        let msg = Message::from_a2a(&message, wall, None)?;
        let task_id = task_id.unwrap_or_else(|| shortuuid("AST_"));
        let mut machine =
            TaskMachine::new(self.partition_key.clone(), msg, task_id, &self.opts, wall)
                .with_endpoint(self.endpoint.clone());
        let partition_key = self.thread.partition(&self.partition_key);
        let holder = Holder::new(&self.holder);
        let spawner = (self.spawner)()
            .map_err(|error| StorageError::Unavailable(format!("A2A runtime: {error}")))?;
        let mut call = Call {
            partition_key,
            holder,
            run: None,
            lease: None,
            handler: None,
            submitted: None,
            spawner,
        };
        let mut effects = machine.handle(A2aTaskEvent::Start, elapsed());
        loop {
            match effects.pop() {
                Some(A2aTaskEffect::Execute { id, operation }) => {
                    let result = self.execute(&mut call, &machine, operation).await;
                    if result.is_err() {
                        self.close_handler(&mut call).await;
                    }
                    effects = machine.handle(
                        A2aTaskEvent::Completed { id, result: Box::new(result) },
                        elapsed(),
                    );
                }
                Some(A2aTaskEffect::Finished(answer)) => {
                    let now = elapsed();
                    let Call { holder, run, lease, mut submitted, .. } = call;
                    if let (Ok(_), Some(submitted)) = (&answer, submitted.as_mut()) {
                        submitted.answered = true;
                    }
                    drop((run, lease, submitted));
                    if !holder.wait_released(&*self.clock).await {
                        log::warn!(target: lk_agents_abi::LOG_TARGET, partition_key = self.partition_key.as_str(); "an A2A turn ended with its lease release in flight");
                    }
                    return answer.map(|answer| machine.result(answer, self.thread.id(), now));
                }
                None => {
                    let at = machine.wake_at().ok_or_else(|| {
                        StorageError::Invalid("the A2A task stalled with nothing to wait on".into())
                    })?;
                    self.clock.wait_until(epoch.saturating_add(at)).await;
                    effects = machine.handle(A2aTaskEvent::Wake, elapsed());
                }
            }
        }
    }

    async fn execute(
        &self,
        call: &mut Call,
        machine: &TaskMachine,
        operation: Operation,
    ) -> Result<Completion, StorageError> {
        Ok(match operation {
            Operation::FindReceipt(message_id) => {
                Completion::Task(call.partition_key.task_for_message(&message_id).await?)
            }
            Operation::FindTarget(task_id) => Completion::Task(self.thread.task(&task_id).await?),
            Operation::ReadTask(task_id) => {
                Completion::Task(call.partition_key.task(&task_id).await?)
            }
            Operation::Submit(task) => {
                call.submitted = Some(Submitted {
                    partition_key: call.partition_key.clone(),
                    task_id: task.task_id.clone(),
                    open: Some(self.thread.keep_open("a2a task cancel")),
                    clock: self.clock.clone(),
                    resolvers: self.resolvers.clone(),
                    spawner: call.spawner.clone(),
                    answered: false,
                });
                let inserted =
                    call.partition_key.submit_at(&task, self.endpoint.as_deref()).await?;
                if let (false, Some(submitted)) = (inserted, call.submitted.as_mut()) {
                    // another send of the message owns its row
                    submitted.answered = true;
                }
                Completion::Submitted(inserted)
            }
            Operation::Receive { message, task } => self.receive(call, &message, task).await?,
            Operation::Accepted(task_id) => {
                self.announce_accepted(&call.partition_key, &task_id).await?
            }
            Operation::Open { run_id, task_id, channel, access, pickup } => {
                let options = RunOptions {
                    channel,
                    pickup,
                    wait_timeout: machine.wait_timeout(),
                    ..RunOptions::new(run_id, task_id)
                };
                call.open_run(access, options).await?
            }
            Operation::ReadInbox => Completion::Inbox(call.partition_key.routed_inbox().await?),
            Operation::Pickup(task_id) => {
                Completion::Picked(call.partition_key.pickup(&task_id).await?)
            }
            Operation::ReadWorking => Completion::Tasks(call.run()?.tasks().working().await?),
            Operation::FailOrphans { reason, failed } => {
                self.fail_orphans(&call.partition_key, reason, failed).await?
            }
            Operation::ReadPending(named) => {
                Completion::Task(call.run()?.tasks().pending_ask(named.as_deref()).await?)
            }
            Operation::Stage(staged) => call.stage(*staged).await?,
            Operation::Handle(handle) => self.open_handler(call, *handle).await?,
            Operation::Next(should_yield) => {
                let event = self.next_event(call, should_yield).await?;
                if !matches!(event, HandlerEvent::Checkpoint) {
                    self.close_handler(call).await;
                }
                Completion::Event(event)
            }
            Operation::Checkpoint(task_ids) => call.checkpoint(&task_ids).await?,
            Operation::Commit { answer_task, lease, drained_id } => {
                call.commit(CommitOptions { answer_task, lease, drained_id }).await?
            }
            Operation::Abandon { outcome, reason } => {
                self.close_handler(call).await;
                if let Some(mut run) = call.run.take() {
                    run.abandon(outcome, reason).await?;
                }
                Completion::Done
            }
            Operation::ResolveLater { task_id, failed } => {
                let run = call
                    .run
                    .take()
                    .ok_or_else(|| StorageError::Invalid("no A2A run is open".into()))?;
                self.resolve_later(call, run, task_id, failed);
                Completion::Done
            }
            Operation::CloseRun => {
                if let Some(run) = call.run.as_mut() {
                    run.commit(CommitOptions::default()).await?;
                }
                call.run = None;
                Completion::Done
            }
            Operation::ReleaseDrained => call.release_drained().await?,
            Operation::ReadRuns(limit) => Completion::Runs(
                call.partition_key
                    .runs(limit)
                    .await?
                    .into_iter()
                    .map(|r| PartitionRun { interrupted: r.interrupted(), task_id: r.task_id })
                    .collect(),
            ),
        })
    }

    /// The `batch` policy's receipt; a free partition's lease is kept on the call.
    async fn receive(
        &self,
        call: &mut Call,
        message: &a2a::Message,
        task: TaskWrite,
    ) -> Result<Completion, StorageError> {
        let input = ReceiveInput {
            task_id: &task.task_id,
            message,
            data: &task.data,
            internal: task.internal.as_ref(),
        };
        let (received, lease) =
            call.partition_key.receive_at(&call.holder, input, self.endpoint.as_deref()).await?;
        call.lease = lease;
        Ok(match received {
            Receive::Duplicate(existing) => Completion::Received {
                duplicate: Some(existing.unwrap_or(task.task_id)),
                leased: false,
            },
            Receive::Received(_) => {
                Completion::Received { duplicate: None, leased: call.lease.is_some() }
            }
        })
    }

    /// Announces a durable receipt on the response stream, when one watches the turn.
    async fn announce_accepted(
        &self,
        partition_key: &Partition,
        task_id: &str,
    ) -> Result<Completion, StorageError> {
        let Some(publisher) = &self.publisher else { return Ok(Completion::Done) };
        let task = partition_key
            .task(task_id)
            .await?
            .ok_or_else(|| StorageError::Unavailable("an accepted task was not durable".into()))?;
        publisher.accepted(&task).await;
        Ok(Completion::Done)
    }

    async fn fail_orphans(
        &self,
        partition_key: &Partition,
        reason: lk_agents_abi::storage::thread::TaskReason,
        failed: Vec<(String, a2a::Message)>,
    ) -> Result<Completion, StorageError> {
        for (task_id, message) in failed {
            partition_key.fail_orphan(&task_id, &message).await?;
            log::warn!(target: lk_agents_abi::LOG_TARGET, partition_key = self.partition_key.as_str(), taskId = task_id.as_str(); "the task fails {reason}");
        }
        Ok(Completion::Done)
    }

    /// Opens the host handler's turn over the open run; the handler starts when first read. A
    /// lease already lost ends the open with `LeaseLost`.
    async fn open_handler(
        &self,
        call: &mut Call,
        handle: Handle,
    ) -> Result<Completion, StorageError> {
        let host = if handle.endpoint == self.endpoint {
            self.host.clone()
        } else {
            use lk_agents_abi::storage::Storage as _;
            let endpoint = handle.endpoint.as_deref().ok_or_else(|| {
                StorageError::Invalid("queued request has no configured endpoint".into())
            })?;
            let namespace = self.namespace.as_ref().ok_or_else(|| {
                StorageError::Invalid("endpoint recovery requires a server registry".into())
            })?;
            self.server
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
                .and_then(|server| server.text_endpoint(endpoint))
                .and_then(|door| {
                    door.recovery_handler(
                        &namespace.location(),
                        &namespace.project_id(),
                        &self.partition_key,
                    )
                })
                .ok_or_else(|| {
                    StorageError::Invalid(format!(
                        "queued endpoint {endpoint:?} is unavailable for this storage partition"
                    ))
                })?
        };
        let caller_history = handle
            .batch
            .last()
            .map(|message| caller_history_of(&message.raw))
            .transpose()?
            .flatten();
        let run = call.run()?;
        let lease = run.lease().cloned();
        let ctx = TurnContext {
            thread_id: self.thread.id().to_owned(),
            context_scope_id: handle.context_scope_id,
            partition_key: handle.partition_key,
            message: handle.message.payload,
            batch: handle.batch,
            history: run.history().to_vec(),
            state: run.state().cloned(),
            context_scopes: run.context_scopes().to_vec(),
            read_only: handle.read_only,
            task_id: handle.task_id,
            tasks: run.tasks().recent(handle.recent_tasks).await?,
            caller_history,
            answer_tool: handle.answer_tool,
        };
        let mut host_ctx = lk_agents_abi::a2a::turn::TurnContext::try_from(ctx)?;
        host_ctx.host_context = self.host_context.clone();
        if let Some(lease) = &lease {
            lease.check()?;
        }
        let cx = telemetry::Cx::NONE.continuing(self.span);
        call.handler = Some(Handler::open(host, cx, host_ctx));
        Ok(Completion::Done)
    }

    /// Hands a run whose commit stayed uncertain to a background resolver, which fails its task
    /// with `failed` when the commit was rejected.
    fn resolve_later(&self, call: &Call, run: Run, answer: String, failed: a2a::Message) {
        let open = self.thread.keep_open("a2a commit resolver");
        let partition_key = call.partition_key.clone();
        self.resolvers.spawn(&*call.spawner, "a2a commit resolver", async move {
            let _open = open;
            match run.resolve_later().await {
                Some(Err(e)) if rejected_commit(&e) => {
                    if let Err(e) = partition_key.fail_orphan(&answer, &failed).await {
                        log::warn!(target: lk_agents_abi::LOG_TARGET, taskId = answer.as_str(); "a rejected commit's task could not be failed: {e}");
                    }
                }
                Some(Err(e)) => {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, taskId = answer.as_str(); "a commit stayed unresolved: {e}");
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
        should_yield: Option<bool>,
    ) -> Result<HandlerEvent, StorageError> {
        let lease = call.run.as_ref().and_then(Run::lease).cloned();
        let handler = call
            .handler
            .as_mut()
            .ok_or_else(|| StorageError::Invalid("no A2A handler is open".into()))?;
        self.read_handler_event(handler, &mut call.run, lease.as_ref(), should_yield).await
    }

    /// Answers the checkpoint, then publishes the handler's progress until its next checkpoint
    /// or output; the lease is checked before the answer and after every report.
    async fn read_handler_event(
        &self,
        handler: &mut Handler,
        run: &mut Option<Run>,
        lease: Option<&Lease>,
        should_yield: Option<bool>,
    ) -> Result<HandlerEvent, StorageError> {
        if let Some(lease) = lease {
            lease.check()?;
        }
        if let Some(should_yield) = should_yield {
            handler.answer(should_yield);
        }
        loop {
            // Race only the retained handler read. Once a checkpoint write starts, retain
            // its retry identity and await its outcome instead of dropping it on lease loss.
            let read = match lease {
                Some(lease) => tokio::select! {
                    biased;
                    read = handler.next() => read,
                    error = lease.lost() => return Err(error),
                },
                None => handler.next().await,
            };
            if let Some(lease) = lease {
                lease.check()?;
            }
            match read {
                Some(Read::Session(write)) => {
                    let result = match run.as_mut() {
                        Some(run) => {
                            let history = write
                                .history
                                .into_iter()
                                .map(|item| {
                                    lk_agents_abi::llm::chat_context::protocol::encode_chat_item(
                                        &item,
                                    )
                                })
                                .collect::<Vec<_>>();
                            run.checkpoint_session(write.state, history).await
                        }
                        None => Err(StorageError::Invalid("no session run is active".into())),
                    };
                    let _ = write.reply.send(result.clone());
                    result?;
                }
                Some(Read::Checkpoint) => return Ok(HandlerEvent::Checkpoint),
                Some(Read::Event(event)) => {
                    if let Some(publisher) = &self.publisher {
                        publisher.event(&event).await;
                    }
                }
                Some(Read::Output(output)) => {
                    return self.handler_output((*output).map_err(host_error)?);
                }
                None => {
                    return Err(StorageError::Invalid("the handler already returned".into()));
                }
            }
        }
    }

    fn handler_output(
        &self,
        output: lk_agents_abi::a2a::turn::HandlerOutput,
    ) -> Result<HandlerEvent, StorageError> {
        let out = HandlerOutput::try_from(output)?;
        if let Some(err) = &out.error {
            log::error!(target: lk_agents_abi::LOG_TARGET, partition_key = self.partition_key.as_str(); "the a2a_session handler raised; the task fails: {err}");
        }
        Ok(HandlerEvent::Output(Box::new(out)))
    }

    /// Closes the handler, if one is open: a running one is cancelled and its cleanup awaited
    /// within the host teardown bound.
    async fn close_handler(&self, call: &mut Call) {
        let Some(handler) = call.handler.take() else { return };
        let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        if !handler.close(self.clock.as_ref(), deadline).await {
            log::warn!(target: lk_agents_abi::LOG_TARGET, partition_key = self.partition_key.as_str(); "the handler did not settle within its deadline; its host task is still running");
        }
    }
}

impl TurnContract for Turn {
    type Env = ProcessEnv;
    type Thread = Thread;

    fn new(
        env: &ProcessEnv,
        thread: Arc<Thread>,
        partition_key: String,
        host: Arc<dyn TurnHost>,
        options: Option<TurnOptions>,
    ) -> Self {
        let holder = thread.holder().map_or_else(|| env.lease_holder(), str::to_owned);
        Self::new(
            thread.as_ref().clone(),
            &holder,
            &partition_key,
            host,
            env.clock.clone(),
            env.spawner,
            options.unwrap_or_default(),
        )
    }
    fn partition_key(&self) -> String {
        self.partition_key().to_owned()
    }
    fn policy_for(&self, channel: String) -> Policy {
        self.policy_for(&channel)
    }
    async fn run(
        &self,
        message: serde_json::Value,
        task_id: Option<String>,
        context: lk_agents_abi::invoke::CallContext,
    ) -> Result<lk_agents_abi::a2a::turn::TurnResult, StorageError> {
        let mut turn = self.clone();
        if let Some(context) = context.host_context {
            turn = turn.with_context(context);
        }
        Ok(turn.run(message, task_id).await?.into())
    }
}

#[cfg(test)]
#[path = "a2a_tests.rs"]
mod tests;

#[cfg(test)]
mod lease_tests {
    use super::*;
    use lk_agents_abi::BoxFuture;
    use lk_agents_abi::a2a::turn::HandlerPort;
    use lk_agents_abi::errors::ProviderError;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Counts the handler calls the host was entered for.
    #[derive(Clone)]
    struct Probe(Arc<AtomicUsize>);

    impl TurnHost for Probe {
        fn handle<'a>(
            &'a self,
            _cx: &telemetry::Cx,
            _: &'a lk_agents_abi::a2a::turn::TurnContext,
            turn: Arc<dyn HandlerPort>,
            _cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
        ) -> BoxFuture<'a, Result<lk_agents_abi::a2a::turn::HandlerOutput, ProviderError>> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move {
                turn.checkpoint().await;
                Ok(Default::default())
            })
        }
    }

    fn context() -> lk_agents_abi::a2a::turn::TurnContext {
        lk_agents_abi::a2a::turn::TurnContext {
            thread_id: "AT_loss".into(),
            partition_key: "desk".into(),
            message: lk_agents_abi::ChatItem::Message(lk_agents_abi::ChatMessage::text(
                "message",
                lk_agents_abi::Role::User,
                "",
                std::time::SystemTime::UNIX_EPOCH,
            )),
            batch: Vec::new(),
            delegated: false,
            history: Vec::new(),
            state: None,
            context_scope_id: String::new(),
            context_scopes: Vec::new(),
            read_only: false,
            task_id: "task".into(),
            tasks: Vec::new(),
            caller_history: None,
            answer_tool: None,
            host_context: None,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn expired_ownership_precedes_a_ready_handler_event_and_checkpoint_verdict() {
        let clock = Arc::new(lk_agents_net::clock::RealClock::new());
        let file = lk_agents_storage::testing::TempDatabase::new();
        let connection = Arc::new(file.connect().await);
        let background = ProcessEnv::current_runtime().background().unwrap();
        let thread =
            Thread::new(connection, "AT_loss", Default::default(), Default::default(), background)
                .await
                .unwrap();
        let partition_key = thread.partition("desk");
        let holder = Holder::new("worker");
        let run = partition_key.open(RunOptions::new("run", "task"), &holder).await.unwrap();
        run.lease().unwrap().stop();
        tokio::time::advance(Duration::from_secs(10)).await;
        let probe = Probe(Arc::default());
        let host: Arc<dyn TurnHost> = Arc::new(probe.clone());
        let spawner = ProcessEnv::current_runtime().spawner;
        let turn =
            Turn::new(thread, "worker", "desk", host.clone(), clock, spawner, Default::default());
        let handler = Handler::open(host, telemetry::Cx::NONE, context());
        let mut call = Call {
            partition_key,
            holder,
            run: Some(run),
            lease: None,
            handler: Some(handler),
            submitted: None,
            spawner: spawner().unwrap(),
        };
        let result = turn.next_event(&mut call, Some(false)).await;
        assert!(matches!(result, Err(StorageError::LeaseLost(_))));
        assert_eq!(
            probe.0.load(Ordering::Relaxed),
            0,
            "neither resume nor poll the handler after loss"
        );
    }
}
