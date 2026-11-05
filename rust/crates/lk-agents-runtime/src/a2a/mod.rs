//! The A2A text door's turns: [`Turn::run`] takes one incoming message to its answer by executing
//! the operations of `lk_agents_core::a2a::task::TaskMachine` against storage and the host's
//! handler, and feeding each typed result back. Every decision (deduplication, arrival policy,
//! claim, drain, the pending question, checkpoints, settlement, commit and failure outcomes) is
//! the machine's; this file only executes.

pub mod door;

use crate::telemetry;
use lk_agents_abi::a2a::turn::Turn as TurnContract;
use lk_agents_abi::storage::thread::TaskWrite;
use lk_agents_abi::{Clock, Closed, HOST_TEARDOWN_TIMEOUT, StreamInput, StreamOutput};
use lk_agents_core::a2a::task::{
    A2aTaskEffect, A2aTaskEvent, Access, Completion, Handle, HandlerEvent, LaneRun, Operation,
    RunEnd, Stage, TaskMachine, rejected_commit,
};
use lk_agents_core::a2a::{
    HandlerOutput, Message, Policy, TurnContext, TurnEvent, TurnHost, TurnOptions, TurnResult,
    Verdict, caller_history_of, host_error, resolve_policy,
};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::task::{TaskSet, TokioSpawner};
use lk_agents_storage::StorageError;
use lk_agents_storage::thread::db::{Receive, ReceiveInput};
use lk_agents_storage::thread::run::ItemRow;
use lk_agents_storage::{CommitOptions, Lane, Lease, Run, RunOptions, Thread, shortuuid};
use serde_json::Value;
use std::sync::Arc;

/// The handler's turn: its checkpoint verdicts in, its events out.
type Handler =
    lk_agents_abi::DuplexStream<Box<dyn StreamInput<Verdict>>, Box<dyn StreamOutput<TurnEvent>>>;

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

    /// Opens the run as `access` takes the lane; a kept lease resumes it.
    async fn open_run(
        &mut self,
        access: Access,
        options: RunOptions,
    ) -> Result<Completion, StorageError> {
        let (exclusive, wait, interrupt) = match access {
            Access::Concurrent => (false, false, false),
            Access::Exclusive { wait, interrupt } => (true, wait, interrupt),
            Access::Kept => (true, false, false),
        };
        let options = RunOptions { exclusive, wait, interrupt, ..options };
        let run = match (access, self.lease.take()) {
            (Access::Kept, Some(lease)) => self.lane.resume(options, lease).await?,
            _ => self.lane.open(options).await?,
        };
        let history_end = run.history.iter().map(ItemRow::created_at).fold(0.0, f64::max);
        let opened =
            Completion::Opened { closed: run.closed, picked_up: run.picked_up, history_end };
        self.run = Some(run);
        Ok(opened)
    }

    /// Applies a turn's staged edits, task rows, journal entries and end to the open run.
    async fn stage(&mut self, staged: Stage) -> Result<Completion, StorageError> {
        let run = self.run()?;
        for edit in staged.edits {
            edit.apply(&mut run.history);
        }
        for t in staged.transitions {
            run.tasks.write(&t.task_id, t.status, t.patch).await?;
        }
        for event in staged.events {
            run.events.append(event);
        }
        if let Some(end) = staged.end {
            end_run(run, end);
        }
        Ok(Completion::Done)
    }

    async fn checkpoint(&mut self, task_ids: &[String]) -> Result<Completion, StorageError> {
        let holder = self.run()?.holder.clone();
        let mut rows = Vec::with_capacity(task_ids.len());
        for task_id in task_ids {
            rows.push(self.lane.checkpoint(task_id).await?);
        }
        Ok(Completion::Checkpoint { holder, rows })
    }

    /// Commits the open run; its lease stays on the call when `keep_lease`.
    async fn commit(&mut self, options: CommitOptions) -> Result<Completion, StorageError> {
        let keep_lease = options.keep_lease;
        let run = self.run()?;
        let committed = run.commit(options).await?;
        let run_id = run.run_id.clone();
        if keep_lease {
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
    if end.state.is_some() {
        run.state = end.state;
    }
    run.scope_writes = end.scope_writes;
    run.app_writes = end.app_writes;
    run.usage = end.usage;
    run.outcome = end.outcome;
    run.reason = end.reason;
}

/// One text door's turns: its lane, its options, its host.
#[derive(Clone)]
pub struct Turn {
    thread: Thread,
    lane: String,
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
            publisher: None,
            host_context: None,
            span: None,
            resolvers: Arc::default(),
        }
    }

    /// Each invocation supplies its own host execution context; the endpoint proxy is shared.
    pub fn with_context(mut self, context: lk_agents_abi::HostRef) -> Self {
        self.host_context = Some(context);
        self
    }

    pub fn lane(&self) -> &str {
        &self.lane
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
        let caller_history = caller_history_of(&message)?;
        let epoch = self.clock.now();
        let elapsed = || self.clock.now().saturating_duration_since(epoch);
        // the run's clock, as every time the machine reads: the DST's is virtual
        let wall = self.clock.wall().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        let wall = wall.as_secs_f64();
        let msg = Message::from_a2a(&message, wall, None)?;
        let task_id = task_id.unwrap_or_else(|| shortuuid("AST_"));
        let mut machine = TaskMachine::new(self.lane.clone(), msg, task_id, &self.opts, wall);
        let lane = self.thread.lane(&self.lane);
        let mut call = Call { lane, run: None, lease: None, handler: None, caller_history };
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
                Completion::Task(call.lane.task_for_message(&message_id).await?)
            }
            Operation::FindTarget(task_id) => Completion::Task(self.thread.task(&task_id).await?),
            Operation::ReadTask(task_id) => Completion::Task(call.lane.task(&task_id).await?),
            Operation::Submit(task) => Completion::Submitted(call.lane.submit(&task).await?),
            Operation::Receive { message, task } => self.receive(call, &message, task).await?,
            Operation::Accepted(task_id) => self.announce_accepted(&call.lane, &task_id).await?,
            Operation::Open { run_id, task_id, channel, access, pickup } => {
                let options = RunOptions {
                    channel,
                    pickup,
                    wait_timeout: machine.wait_timeout(),
                    ..RunOptions::new(run_id, task_id)
                };
                call.open_run(access, options).await?
            }
            Operation::ReadInbox => Completion::Inbox(call.lane.inbox(true).await?),
            Operation::Pickup(task_id) => Completion::Picked(call.lane.pickup(&task_id).await?),
            Operation::ReadWorking => Completion::Tasks(call.run()?.tasks.working().await?),
            Operation::FailOrphans { reason, failed } => {
                self.fail_orphans(&call.lane, reason, failed).await?
            }
            Operation::ReadPending(named) => {
                Completion::Task(call.run()?.tasks.pending_ask(named.as_deref()).await?)
            }
            Operation::Stage(staged) => call.stage(*staged).await?,
            Operation::Handle(handle) => self.open_handler(call, *handle).await?,
            Operation::Next(verdict) => {
                let event = self.next_event(call, verdict).await?;
                if !matches!(event, HandlerEvent::Checkpoint) {
                    self.close_handler(call).await;
                }
                Completion::Event(event)
            }
            Operation::Checkpoint(task_ids) => call.checkpoint(&task_ids).await?,
            Operation::Commit { answer_task, keep_lease, drained_id } => {
                call.commit(CommitOptions { answer_task, keep_lease, drained_id }).await?
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
                self.resolve_later(call.lane.clone(), run, task_id, failed);
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
                call.lane
                    .runs(limit)
                    .await?
                    .into_iter()
                    .map(|r| LaneRun { interrupted: r.interrupted(), task_id: r.task_id })
                    .collect(),
            ),
        })
    }

    /// The `batch` policy's receipt; a free lane's lease is kept on the call.
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
        let (received, lease) = call.lane.receive(self.thread.holder(), input).await?;
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
        lane: &Lane,
        task_id: &str,
    ) -> Result<Completion, StorageError> {
        let Some(publisher) = &self.publisher else { return Ok(Completion::Done) };
        let task = lane
            .task(task_id)
            .await?
            .ok_or_else(|| StorageError::Unavailable("an accepted task was not durable".into()))?;
        publisher.accepted(&task).await;
        Ok(Completion::Done)
    }

    async fn fail_orphans(
        &self,
        lane: &Lane,
        reason: &str,
        failed: Vec<(String, a2a::Message)>,
    ) -> Result<Completion, StorageError> {
        for (task_id, message) in failed {
            lane.fail_orphan(&task_id, &message).await?;
            log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = task_id.as_str(); "the task fails {reason}");
        }
        Ok(Completion::Done)
    }

    /// Opens the host handler's turn over the open run. A lease lost before the host answers
    /// ends the open with `LeaseLost`.
    async fn open_handler(
        &self,
        call: &mut Call,
        handle: Handle,
    ) -> Result<Completion, StorageError> {
        let caller_history = call.caller_history.clone();
        let run = call.run()?;
        let lease = run.lease().cloned();
        let ctx = TurnContext {
            thread_id: self.thread.id().to_owned(),
            context_scope_id: handle.context_scope_id,
            lane: handle.lane,
            message: handle.message.payload,
            batch: handle.batch,
            history: run.history.clone(),
            state: run.state.clone(),
            context_scopes: run.context_scopes.clone(),
            read_only: handle.read_only,
            task_id: handle.task_id,
            tasks: run.tasks.recent(handle.recent_tasks).await?,
            caller_history,
            answer_tool: handle.answer_tool,
        };
        let mut host_ctx = lk_agents_abi::a2a::turn::TurnContext::try_from(ctx)?;
        host_ctx.host_context = self.host_context.clone();
        if let Some(lease) = &lease {
            lease.check()?;
        }
        let handler = self.host.handle(&telemetry::Cx::NONE.continuing(self.span), &host_ctx);
        call.handler = Some(match lease {
            Some(lease) => tokio::select! {
                biased;
                error = lease.lost() => return Err(error),
                handler = handler => handler.map_err(host_error)?,
            },
            None => handler.await.map_err(host_error)?,
        });
        Ok(Completion::Done)
    }

    /// Hands a run whose commit stayed uncertain to a background resolver, which fails its task
    /// with `failed` when the commit was rejected.
    fn resolve_later(&self, lane: Lane, run: Run, answer: String, failed: a2a::Message) {
        self.resolvers.spawn(&TokioSpawner::current(), "a2a commit resolver", async move {
            match run.resolve_later().await {
                Some(Err(e)) if rejected_commit(&e) => {
                    if let Err(e) = lane.fail_orphan(&answer, &failed).await {
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
            .ok_or_else(|| StorageError::Invalid("no A2A handler is open".into()))?
            .split_mut();
        let read = self.read_handler_event(verdicts, events, lease.as_ref(), verdict);
        match &lease {
            Some(lease) => tokio::select! {
                biased;
                e = lease.lost() => Err(e),
                event = read => event,
            },
            None => read.await,
        }
    }

    /// Pushes the verdict, then publishes the handler's progress until its next checkpoint or
    /// output; the lease is checked before the push and after every event.
    async fn read_handler_event(
        &self,
        verdicts: &mut Box<dyn StreamInput<Verdict>>,
        events: &mut Box<dyn StreamOutput<TurnEvent>>,
        lease: Option<&Lease>,
        verdict: Option<Verdict>,
    ) -> Result<HandlerEvent, StorageError> {
        if let Some(lease) = lease {
            lease.check()?;
        }
        if let Some(verdict) = verdict {
            verdicts.push(verdict).map_err(host_error)?;
        }
        loop {
            let event = events.next().await.map_err(host_error)?;
            if let Some(lease) = lease {
                lease.check()?;
            }
            match event {
                Some(TurnEvent::Checkpoint) => return Ok(HandlerEvent::Checkpoint),
                Some(event @ (TurnEvent::Progress { .. } | TurnEvent::Token { .. })) => {
                    if let Some(publisher) = &self.publisher {
                        publisher.event(&event).await;
                    }
                }
                Some(TurnEvent::Output { output }) => return self.handler_output(*output),
                None => {
                    return Err(StorageError::Invalid(
                        "the handler's stream ended without its output".into(),
                    ));
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
            log::error!(target: "livekit.agents", lane = self.lane.as_str(); "the a2a_session handler raised; the task fails: {err}");
        }
        Ok(HandlerEvent::Output(Box::new(out)))
    }

    /// Closes the handler's stream, if one is open, within the host teardown bound.
    async fn close_handler(&self, call: &mut Call) {
        let Some(stream) = call.handler.take() else { return };
        let (_, events) = stream.split();
        let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        if events.close(deadline).await == Closed::TimedOut {
            log::warn!(target: "livekit.agents", lane = self.lane.as_str(); "the handler's stream did not close within its deadline; its host task is still running");
        }
    }
}

// The declared text-turn surface over the A2A executor.

impl TurnContract for Turn {
    type Env = ProcessEnv;
    type Thread = Thread;

    fn new(
        env: &ProcessEnv,
        thread: Arc<Thread>,
        lane: String,
        host: Arc<dyn TurnHost>,
        options: Option<TurnOptions>,
    ) -> Self {
        Self::new(
            thread.as_ref().clone(),
            &lane,
            host,
            env.clock.clone(),
            options.unwrap_or_default(),
        )
    }
    fn lane(&self) -> String {
        self.lane().to_owned()
    }
    fn policy_for(&self, channel: String) -> String {
        self.policy_for(&channel).as_str().to_owned()
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
mod lease_tests {
    use super::*;
    use lk_agents_abi::errors::ProviderError;
    use lk_agents_abi::{BoxFuture, Ts};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[derive(Clone)]
    struct Probe(Arc<AtomicUsize>);

    impl StreamInput<Verdict> for Probe {
        fn push(&self, _: Verdict) -> Result<(), ProviderError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        fn flush(&self) -> Result<(), ProviderError> {
            Ok(())
        }
        fn end_input(&self) {}
    }

    impl StreamOutput<TurnEvent> for Probe {
        fn next(&mut self) -> BoxFuture<'_, Result<Option<TurnEvent>, ProviderError>> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Box::pin(async { Ok(Some(TurnEvent::Checkpoint)) })
        }
        fn close(self: Box<Self>, _: Ts) -> BoxFuture<'static, Closed> {
            Box::pin(async { Closed::Clean })
        }
    }

    impl TurnHost for Probe {
        fn handle<'a>(
            &'a self,
            _cx: &telemetry::Cx,
            _: &'a lk_agents_abi::a2a::turn::TurnContext,
        ) -> BoxFuture<'a, Result<lk_agents_abi::a2a::turn::TurnStream, ProviderError>> {
            Box::pin(async { panic!("this test supplies an already-open handler") })
        }
    }

    #[tokio::test(start_paused = true)]
    async fn expired_ownership_precedes_a_ready_handler_event_and_checkpoint_verdict() {
        let clock = Arc::new(lk_agents_net::clock::RealClock::new());
        let connection =
            Arc::new(lk_agents_storage::testing::MemoryConnection::new(clock.clone()).unwrap());
        let thread =
            Thread::new(connection, "AT_loss", Default::default(), Default::default(), "worker")
                .await
                .unwrap();
        let lane = thread.lane("desk");
        let run = lane.open(RunOptions::new("run", "task")).await.unwrap();
        run.lease().unwrap().stop();
        tokio::time::advance(Duration::from_secs(10)).await;
        let probe = Probe(Arc::default());
        let turn = Turn::new(thread, "desk", Arc::new(probe.clone()), clock, Default::default());
        let handler = lk_agents_abi::DuplexStream::new(
            Box::new(probe.clone()) as Box<dyn StreamInput<Verdict>>,
            Box::new(probe.clone()) as Box<dyn StreamOutput<TurnEvent>>,
        );
        let mut call = Call {
            lane,
            run: Some(run),
            lease: None,
            handler: Some(handler),
            caller_history: None,
        };
        let result = turn.next_event(&mut call, Some(Verdict::Continue)).await;
        assert!(matches!(result, Err(StorageError::LeaseLost(_))));
        assert_eq!(
            probe.0.load(Ordering::Relaxed),
            0,
            "neither resume nor poll the handler after loss"
        );
    }
}
