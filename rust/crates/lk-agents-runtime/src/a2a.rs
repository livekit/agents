//! The A2A text door's turns: [`Turn::run`] takes one incoming message through its channel's
//! policy against storage and drives the host's handler, applying the A2A core's decisions
//! (`lk_agents_a2a::outcome`) to the run. The turn writes the message's task row: `submitted`,
//! `working`, then its end state.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::task::JoinSet;

use lk_agents_a2a::{
    HandlerOutput, Message, Policy, Settle, Stop, TurnContext, TurnEvent, TurnHost, TurnIds,
    TurnOptions, TurnResult, Verdict, answer_pending, batch_to_chat_message, caller_history_of,
    channel_of, channel_of_record, checkpoint_stop, failed_message, host_error, python_reason,
    resolve_debounce, resolve_policy, result_for, settle, unpersisted_failed,
};
use lk_agents_abi::{Clock, Closed, HOST_TEARDOWN_TIMEOUT};
use lk_agents_storage::StorageError;
use lk_agents_storage::db::{Committed, TaskWrite};
use lk_agents_storage::lease::LanePoll;
use lk_agents_storage::run::{
    ItemRow, REASON_LEASE_TIMEOUT, REASON_STORAGE_UNAVAILABLE, REASON_WORKER_LOST, RunOutcome,
    TaskRow, TaskState,
};
use lk_agents_storage::schema::TaskData;
use serde_json::Value;

use crate::storage::{
    CommitOptions, Lane, Lease, Run, RunOptions, Thread, now_ms, now_secs, shortuuid,
};

/// How many task rows the handler's context carries.
const TASKS_IN_CONTEXT: i64 = 50;
/// How many runs `reject` reads to name the running task.
const RUNS_FOR_REJECT: usize = 50;

/// What one `run` call carries between the policy steps.
struct Call {
    lane: Lane,
    last_run_id: Option<String>,
    /// A delegating caller's conversation, when this call's message carried one.
    caller_history: Option<Value>,
}

/// One text door's turns: its lane, its options, its host.
pub struct Turn {
    thread: Thread,
    lane: String,
    host: Arc<dyn TurnHost>,
    /// The host stream's clock, for the bound on closing it.
    clock: Arc<dyn Clock>,
    opts: TurnOptions,
    /// The resolvers of commits left uncertain, owned by the door.
    resolvers: Mutex<JoinSet<()>>,
}

/// A turn's row data: the message it came from.
fn turn_data(msg: &Message) -> TaskData {
    TaskData { history: vec![msg.to_a2a()], ..TaskData::default() }
}

/// The early `submitted` row of a turn: its message as the task's history.
fn submitted_task(msg: &Message, task_id: &str) -> TaskWrite {
    TaskWrite {
        task_id: task_id.to_owned(),
        status: TaskState::Submitted,
        message_id: Some(msg.message_id.clone()),
        name: None,
        data: turn_data(msg),
        ask: None,
    }
}

/// What a turn leaves for `finish`: the task that carries the reply, and the reply.
struct Outcome {
    task_id: String,
    output: Option<Value>,
}

/// The commit of a turn that ran: its answer task read back with the writes.
fn answering(outcome: &Option<Outcome>) -> CommitOptions {
    let answer_task = outcome.as_ref().map(|o| o.task_id.clone());
    CommitOptions { answer_task, ..CommitOptions::default() }
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
            resolvers: Mutex::new(JoinSet::new()),
        }
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

    /// A record this worker has, or the failed one it could not write.
    fn record_or_failed(
        &self,
        record: Option<TaskRow>,
        task_id: &str,
        msg: Option<&Message>,
        reason: &str,
    ) -> TaskRow {
        record.unwrap_or_else(|| {
            unpersisted_failed(task_id, &self.lane, msg, reason, now_ms(), shortuuid("msg_"))
        })
    }

    /// One incoming message through its policy. `task_id` is the answer stream's id when the
    /// caller created one. [`StorageError::RunRejected`] under `reject`,
    /// [`StorageError::Cancelled`] when the host cancelled the handler.
    pub async fn run(
        &self,
        message: Value,
        task_id: Option<String>,
    ) -> Result<TurnResult, StorageError> {
        let message: a2a::Message = serde_json::from_value(message)
            .map_err(|e| StorageError::Invalid(format!("an A2A v1 message: {e}")))?;
        // the delegation's `chat_ctx` part, which `Message::from_a2a` strips
        let caller_history = caller_history_of(&message);
        let msg = Message::from_a2a(&message, now_secs(), None)?;
        let policy = self.policy_for(&msg.channel);
        let debounce = resolve_debounce(&self.opts.debounce, &msg.channel, policy);
        let task_id = task_id.unwrap_or_else(|| shortuuid("AST_"));
        let mut call =
            Call { lane: self.thread.lane(&self.lane), last_run_id: None, caller_history };
        match policy {
            Policy::Batch => self.batch(&mut call, &msg, &task_id, debounce).await,
            Policy::Concurrent => self.concurrent(&call, &msg, &task_id).await,
            Policy::Reject => self.reject(&call, &msg, &task_id).await,
            Policy::Interrupt | Policy::Queue => {
                let interrupt = policy == Policy::Interrupt;
                self.exclusive(&call, &msg, &task_id, interrupt, debounce).await
            }
        }
    }

    async fn exclusive(
        &self,
        call: &Call,
        msg: &Message,
        task_id: &str,
        interrupt: bool,
        debounce: Duration,
    ) -> Result<TurnResult, StorageError> {
        // the early `submitted` row is durable before anything waits: before the debounce when
        // there is one, else in the acquire batch with the lease and the pickup. A `tasks/get`
        // finds the task at once, and a holder that dies leaves an orphan.
        let row = submitted_task(msg, task_id);
        let mut submitted = None;
        if !debounce.is_zero() {
            submitted = Some(call.lane.db().submit(&row).await?);
            tokio::time::sleep(debounce).await;
        }
        let opened = call
            .lane
            .open(RunOptions {
                interrupt,
                channel: Some(msg.channel.clone()),
                wait_timeout: self.opts.wait_timeout(),
                submit: submitted.is_none().then_some(row),
                pickup: true,
                ..RunOptions::new(task_id, task_id)
            })
            .await;
        let mut run = match opened {
            Ok(run) => run,
            Err(StorageError::LeaseTimeout(_)) => {
                log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = task_id, threadId = self.thread.id(); "the lane was not released in time; the task fails lease-timeout");
                // the early row ends failed, so no `submitted` row outlives a message the caller
                // was told failed
                let db = call.lane.db();
                if db.pickup(task_id).await?.is_some() {
                    db.fail_orphan(task_id, &failed(task_id, REASON_LEASE_TIMEOUT)).await?;
                }
                let record = call.lane.task(task_id).await?;
                let record =
                    self.record_or_failed(record, task_id, Some(msg), REASON_LEASE_TIMEOUT);
                return Ok(TurnResult {
                    task: record,
                    output: None,
                    thread_id: self.thread.id().to_owned(),
                    answered_by: task_id.to_owned(),
                    run_id: None,
                });
            }
            Err(e) => return Err(e),
        };
        if submitted.or(run.submitted) == Some(false)
            && let Some(mut existing) = run.tasks.by_message_id(&msg.message_id).await?
        {
            // a retried messageId: the task it created, awaited when a runner still has it, or
            // failed worker-lost when its holder died and the lease lapsed
            self.close_run(&mut run, Ok(()), |_| CommitOptions::default()).await?;
            if !existing.terminal()
                && let Some(ended) =
                    self.await_task(call, &existing.task_id.clone(), false, true).await?
            {
                existing = ended;
            }
            return Ok(result_for(existing, self.thread.id(), None));
        }
        let body = async {
            if run.closed || !run.picked_up {
                return Ok(None);
            }
            self.fail_orphans(&run, &[task_id.to_owned()]).await?;
            self.turn(call, &mut run, std::slice::from_ref(msg), &[task_id.to_owned()], interrupt)
                .await
                .map(Some)
        }
        .await;
        self.end_answered(call, run, msg, task_id, body).await
    }

    /// Ends a run that may have answered `task_id`'s message: the commit and the task as
    /// committed. A lost acknowledgment nobody could resolve in the wait answers the task as it
    /// stands (still `working`) and keeps resolving in the background; a rejected save fails the
    /// task `storage-unavailable`, its staged changes absent. Never a failure it did not observe.
    async fn end_answered(
        &self,
        call: &Call,
        mut run: Run,
        msg: &Message,
        task_id: &str,
        body: Result<Option<Outcome>, StorageError>,
    ) -> Result<TurnResult, StorageError> {
        let body = match body {
            Ok(body) => body,
            Err(e) => {
                let outcome = match e {
                    StorageError::Cancelled(_) => RunOutcome::Canceled,
                    _ => RunOutcome::Failed,
                };
                run.abandon(outcome, python_reason(&e)).await?;
                return Err(e);
            }
        };
        let answer = body.as_ref().map_or(task_id, |o| o.task_id.as_str()).to_owned();
        match run.commit(answering(&body)).await {
            Ok(committed) => match body {
                Some(outcome) => Ok(self.finish(outcome, task_id, &run, committed)),
                None => {
                    let record = call.lane.task(task_id).await?;
                    let record = self.record_or_failed(record, task_id, None, "not-written");
                    Ok(result_for(record, self.thread.id(), None))
                }
            },
            Err(StorageError::Uncertain(why)) => {
                log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = answer.as_str(); "the commit is unresolved; the task stays working while it resolves: {why}");
                self.resolve_later(call, run, answer.clone());
                let current = match call.lane.task(&answer).await {
                    Ok(Some(row)) => row,
                    // unreadable: the row as this worker left it, picked up
                    _ => TaskRow {
                        status: TaskState::Working,
                        ..submitted_task(msg, &answer).row(&self.lane)
                    },
                };
                Ok(result_for(current, self.thread.id(), None))
            }
            Err(e) => {
                log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = answer.as_str(); "the commit was rejected; the task fails storage-unavailable: {e}");
                let db = call.lane.db();
                db.fail_orphan(&answer, &failed(&answer, REASON_STORAGE_UNAVAILABLE)).await?;
                let record = call.lane.task(&answer).await?;
                let record =
                    self.record_or_failed(record, &answer, Some(msg), REASON_STORAGE_UNAVAILABLE);
                Ok(result_for(record, self.thread.id(), None))
            }
        }
    }

    /// Hands a run whose commit stayed uncertain to a background resolver: an authoritative
    /// rejection fails its task `storage-unavailable`; a commit found committed needs nothing.
    fn resolve_later(&self, call: &Call, run: Run, answer: String) {
        let lane = call.lane.clone();
        let mut resolvers = self.resolvers.lock().expect("resolvers");
        while resolvers.try_join_next().is_some() {}
        resolvers.spawn(async move {
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

    /// Ends a run: the commit on success, with the options the body's value asks for, else
    /// `run_ended` with the outcome and the error returned.
    async fn close_run<T>(
        &self,
        run: &mut Run,
        body: Result<T, StorageError>,
        opts: impl FnOnce(&T) -> CommitOptions,
    ) -> Result<(T, Committed), StorageError> {
        match body {
            Ok(v) => {
                let committed = run.commit(opts(&v)).await?;
                Ok((v, committed))
            }
            Err(e) => {
                let outcome = match e {
                    StorageError::Cancelled(_) => RunOutcome::Canceled,
                    _ => RunOutcome::Failed,
                };
                run.abandon(outcome, python_reason(&e)).await?;
                Err(e)
            }
        }
    }

    async fn batch(
        &self,
        call: &mut Call,
        msg: &Message,
        task_id: &str,
        debounce: Duration,
    ) -> Result<TurnResult, StorageError> {
        let (received, lease) = call
            .lane
            .db()
            .receive(self.thread.holder(), task_id, &msg.to_a2a(), &turn_data(msg))
            .await?;
        match (received, lease) {
            (lk_agents_storage::db::Receive::Duplicate(existing), _) => {
                let existing_id = existing.unwrap_or_else(|| task_id.to_owned());
                let mut record = call.lane.task(&existing_id).await?;
                if let Some(r) = &record
                    && !r.terminal()
                {
                    // the first receive may have landed with its answer lost: nobody drains the
                    // row unless this worker does once the lease frees
                    record = self.await_task(call, &r.task_id.clone(), true, true).await?;
                }
                let record = self.record_or_failed(record, &existing_id, None, "not-written");
                Ok(result_for(record, self.thread.id(), None))
            }
            (_, Some(lease)) => {
                // this worker is the runner: let the burst land, then drain
                if !debounce.is_zero() {
                    tokio::time::sleep(debounce).await;
                }
                self.drain(call, Some(lease)).await?;
                let record = call.lane.task(task_id).await?;
                let record = self.record_or_failed(record, task_id, None, "not-written");
                Ok(result_for(record, self.thread.id(), call.last_run_id.as_deref()))
            }
            (_, None) => {
                let record = self.await_task(call, task_id, true, false).await?;
                let record = self.record_or_failed(record, task_id, None, "not-written");
                Ok(result_for(record, self.thread.id(), call.last_run_id.as_deref()))
            }
        }
    }

    /// Polls the task and the lease: the task ends, or the lease frees and this worker drains the
    /// inbox (`drain_if_free`) or fails the orphaned `working` row `worker-lost`
    /// (`fail_if_free`: a resend of the dead holder's message is told to send it again).
    async fn await_task(
        &self,
        call: &Call,
        task_id: &str,
        drain_if_free: bool,
        fail_if_free: bool,
    ) -> Result<Option<TaskRow>, StorageError> {
        let epoch = tokio::time::Instant::now();
        let mut poll = LanePoll::new(epoch.elapsed(), self.opts.wait_timeout());
        // the drain never holds a lease across the loop; a fresh call per pass
        let mut drain_call =
            Call { lane: call.lane.clone(), last_run_id: None, caller_history: None };
        loop {
            let mut record = call.lane.task(task_id).await?;
            if record.as_ref().is_some_and(TaskRow::terminal) {
                return Ok(record);
            }
            if drain_if_free && !call.lane.db().inbox(true).await?.is_empty() {
                match self.drain(&mut drain_call, None).await {
                    Err(StorageError::LeaseHeld(_)) => {}
                    Err(e) => return Err(e),
                    Ok(()) => {
                        record = call.lane.task(task_id).await?;
                        if record.as_ref().is_some_and(TaskRow::terminal) {
                            return Ok(record);
                        }
                    }
                }
            }
            if fail_if_free
                && let Some(r) = &record
                && r.status == TaskState::Working
            {
                let opened = call
                    .lane
                    .open(RunOptions {
                        wait: false,
                        channel: channel_of_record(r),
                        ..RunOptions::new(shortuuid("lost_"), task_id)
                    })
                    .await;
                match opened {
                    Err(StorageError::LeaseHeld(_)) => {}
                    Err(e) => return Err(e),
                    Ok(mut run) => {
                        let body = async {
                            if !run.closed {
                                self.fail_orphans(&run, &[]).await?;
                            }
                            Ok(())
                        }
                        .await;
                        self.close_run(&mut run, body, |_| CommitOptions::default()).await?;
                        record = call.lane.task(task_id).await?;
                        if record.as_ref().is_some_and(TaskRow::terminal) {
                            return Ok(record);
                        }
                    }
                }
            }
            let Some(at) = poll.next(epoch.elapsed()) else {
                log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = task_id, threadId = self.thread.id(); "a queued message was not answered in time");
                return Ok(record);
            };
            tokio::time::sleep_until(epoch + at).await;
        }
    }

    /// The drain loop: one turn per batch of pending rows until nothing is pending.
    /// `LeaseHeld` when another runner holds the lane.
    async fn drain(&self, call: &mut Call, lease: Option<Lease>) -> Result<(), StorageError> {
        let db = call.lane.db().clone();
        let mut lease = lease;
        loop {
            let rows = db.inbox(true).await?;
            let Some(last) = rows.last() else {
                if let Some(lease) = &lease {
                    lease.release_if_drained().await?;
                }
                return Ok(());
            };
            let task_ids: Vec<String> = rows.iter().map(|r| r.task_id.clone()).collect();
            let options = RunOptions {
                wait: false,
                channel: Some(channel_of(&last.message)),
                ..RunOptions::new(shortuuid("drain_"), last.task_id.clone())
            };
            let mut run = match lease.take() {
                Some(lease) => call.lane.resume(options, lease).await?,
                None => call.lane.open(options).await?,
            };
            let last_id = last.id;
            let body = async {
                let mut batch: Vec<Message> = Vec::new();
                let mut batch_ids: Vec<String> = Vec::new();
                for (row, tid) in rows.iter().zip(&task_ids) {
                    let Some(was) = db.pickup(tid).await? else {
                        continue; // terminal already: answered by an earlier drain
                    };
                    if was == TaskState::Working {
                        db.fail_orphan(tid, &failed(tid, REASON_WORKER_LOST)).await?;
                        log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = tid.as_str(); "a message whose drain died fails worker-lost");
                        continue;
                    }
                    batch.push(Message::from_a2a(&row.message, row.received_at as f64, Some(tid))?);
                    batch_ids.push(tid.clone());
                }
                if !batch.is_empty() {
                    self.fail_orphans(&run, &task_ids).await?;
                    self.turn(call, &mut run, &batch, &batch_ids, false).await?;
                }
                Ok(())
            }
            .await;
            let keep = |_: &()| CommitOptions {
                keep_lease: true,
                drained_id: Some(last_id),
                ..CommitOptions::default()
            };
            self.close_run(&mut run, body, keep).await?;
            call.last_run_id = Some(run.run_id.clone());
            let kept = run.lease().cloned();
            let kept =
                kept.ok_or_else(|| StorageError::Invalid("a drain run without a lease".into()))?;
            if kept.release_if_drained().await? {
                return Ok(());
            }
            lease = Some(kept);
        }
    }

    async fn concurrent(
        &self,
        call: &Call,
        msg: &Message,
        task_id: &str,
    ) -> Result<TurnResult, StorageError> {
        if !call.lane.db().submit(&submitted_task(msg, task_id)).await? {
            // a resent task or message id: the task it created, awaited, not a second run
            let mut existing = match call.lane.task(task_id).await? {
                Some(r) => Some(r),
                None => call.lane.db().task_for_message(&msg.message_id).await?,
            };
            if let Some(r) = &existing
                && !r.terminal()
            {
                existing = self.await_task(call, &r.task_id.clone(), false, false).await?;
            }
            let id = existing.as_ref().map_or(task_id, |r| r.task_id.as_str()).to_owned();
            let record = self.record_or_failed(existing, &id, None, "not-written");
            return Ok(result_for(record, self.thread.id(), None));
        }
        let mut run = call
            .lane
            .open(RunOptions {
                exclusive: false,
                channel: Some(msg.channel.clone()),
                ..RunOptions::new(task_id, task_id)
            })
            .await?;
        let body = self
            .turn(call, &mut run, std::slice::from_ref(msg), &[task_id.to_owned()], false)
            .await;
        self.end_answered(call, run, msg, task_id, body.map(Some)).await
    }

    async fn reject(
        &self,
        call: &Call,
        msg: &Message,
        task_id: &str,
    ) -> Result<TurnResult, StorageError> {
        if let Some(info) = call.lane.info().await?
            && info.lease_until > now_ms() / 1000
        {
            let running_task_id = self.running_task(call).await?;
            return Err(StorageError::RunRejected { running_task_id });
        }
        let result = async {
            let mut run = call
                .lane
                .open(RunOptions {
                    wait: false,
                    channel: Some(msg.channel.clone()),
                    ..RunOptions::new(task_id, task_id)
                })
                .await?;
            // the row is written under the lease: a refused message leaves none (docs/server.md)
            let body = async {
                if run.closed {
                    return Ok(None);
                }
                call.lane.db().submit(&submitted_task(msg, task_id)).await?;
                // the row goes `working` once the lease is held, so a holder that dies leaves an
                // orphan; a row a client canceled before the turn started runs nothing
                if call.lane.db().pickup(task_id).await?.is_none() {
                    return Ok(None);
                }
                self.fail_orphans(&run, &[task_id.to_owned()]).await?;
                self.turn(call, &mut run, std::slice::from_ref(msg), &[task_id.to_owned()], false)
                    .await
                    .map(Some)
            }
            .await;
            self.end_answered(call, run, msg, task_id, body).await
        }
        .await;
        match result {
            // the lane was taken between the check and the open: the run that took it is named
            Err(StorageError::LeaseHeld(_)) => {
                let running_task_id = self.running_task(call).await?;
                Err(StorageError::RunRejected { running_task_id })
            }
            other => other,
        }
    }

    /// The task of the run holding the lane: the newest run with no `run_ended`.
    async fn running_task(&self, call: &Call) -> Result<Option<String>, StorageError> {
        let runs = call.lane.runs(RUNS_FOR_REJECT).await?;
        Ok(runs.into_iter().find(|r| r.interrupted()).and_then(|r| r.task_id))
    }

    /// A dead holder's `working` rows, found under this run's lease: each fails `worker-lost`
    /// and nothing runs again; the client resends. `skip` is this run's own row.
    async fn fail_orphans(&self, run: &Run, skip: &[String]) -> Result<(), StorageError> {
        for row in run.tasks.working().await? {
            if skip.contains(&row.task_id) {
                continue;
            }
            run.tasks
                .lane()
                .fail_orphan(&row.task_id, &failed(&row.task_id, REASON_WORKER_LOST))
                .await?;
            log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = row.task_id.as_str(); "a task whose worker died fails worker-lost");
        }
        Ok(())
    }

    /// One turn over a batch inside an open run; writes the task rows, commits nothing itself.
    async fn turn(
        &self,
        call: &Call,
        run: &mut Run,
        batch: &[Message],
        task_ids: &[String],
        interrupt: bool,
    ) -> Result<Outcome, StorageError> {
        let last = batch.last().ok_or_else(|| StorageError::Invalid("an empty batch".into()))?;
        let received = task_ids
            .last()
            .ok_or_else(|| StorageError::Invalid("a batch without task ids".into()))?;

        // what this turn adds goes after everything loaded, whatever the messages' own times
        let after = run.history.iter().map(ItemRow::created_at).fold(0.0, f64::max);
        let mut now = now_secs();
        if after >= now {
            now = after + 2e-6;
        }

        let pending = run.tasks.pending_ask().await?;
        let pending = answer_pending(
            &mut run.history,
            pending.as_ref(),
            last,
            received,
            now,
            shortuuid("item_"),
        );
        for t in pending.transitions {
            run.tasks.write(&t.task_id, t.status, t.patch).await?;
        }
        let answer_task = pending.answer_task;

        let user_message = batch_to_chat_message(batch, now, shortuuid("item_"));
        let turn_ctx = TurnContext {
            thread_id: self.thread.id().to_owned(),
            lane: call.lane.name().to_owned(),
            message: user_message.payload.clone(),
            batch: batch.to_vec(),
            history: run.history.clone(),
            state: run.state.clone(),
            context_scope_id: format!("root:{}", call.lane.name()),
            context_scopes: run.context_scopes.clone(),
            read_only: run.lease().is_none(),
            task_id: answer_task.clone(),
            tasks: run.tasks.recent(TASKS_IN_CONTEXT).await?,
            caller_history: call.caller_history.clone(),
            answer_tool: pending.answer_tool,
        };
        let (out, stop) = self.drive(&turn_ctx, run, interrupt).await?;
        if let Some(err) = &out.error {
            log::error!(target: "livekit.agents", lane = call.lane.name(), taskId = answer_task.as_str(); "the a2a_session handler raised; the task fails: {err}");
        }
        if out.state.is_some() {
            run.state = out.state.clone();
        }
        run.scope_writes = out.scope_writes.clone();
        run.usage.add(&out.usage);
        let answer_canceled = (stop.is_some() || out.interrupted)
            && call
                .lane
                .task(&answer_task)
                .await?
                .is_some_and(|row| row.status == TaskState::Canceled);
        let settled = settle(
            &mut run.history,
            Settle {
                batch,
                task_ids,
                answer_task: &answer_task,
                user_message,
                out: &out,
                stop,
                answer_canceled,
                now,
                ids: TurnIds {
                    assistant_item: shortuuid("item_"),
                    status_message: shortuuid("msg_"),
                    artifact: shortuuid("art_"),
                },
            },
        )?;
        for (ty, payload) in settled.events {
            run.events.append(&ty, payload);
        }
        for t in settled.transitions {
            run.tasks.write(&t.task_id, t.status, t.patch).await?;
        }
        run.outcome = settled.outcome;
        run.reason = settled.reason;
        Ok(Outcome { task_id: answer_task, output: settled.output })
    }

    /// The host's stream to its output: each checkpoint answered `yield` once the turn should stop
    /// (a newer message waiting under `interrupt`, a client's `tasks/cancel`), else `continue`. A
    /// lost lease ends it with `Unavailable`; the stream is closed either way.
    async fn drive(
        &self,
        ctx: &TurnContext,
        run: &Run,
        interrupt: bool,
    ) -> Result<(HandlerOutput, Option<Stop>), StorageError> {
        let task_id = ctx.task_id.as_str();
        let (verdicts, mut events) =
            self.host.handle(ctx.clone().into()).await.map_err(host_error)?;
        let lane = self.thread.lane(&self.lane);
        let mut stop = None;
        let read = async {
            loop {
                match events.next().await.map_err(host_error)? {
                    Some(TurnEvent::Checkpoint) => {
                        // a lost lease cannot commit anyway: nothing yields to anyone
                        if stop.is_none() && !run.lease().is_some_and(Lease::is_lost) {
                            let (waiter, status) = lane.db().checkpoint(task_id).await?;
                            stop =
                                checkpoint_stop(interrupt, waiter.as_deref(), &run.holder, status);
                            match stop {
                                Some(Stop::Interrupted) => {
                                    log::info!(target: "livekit.agents", lane = self.lane.as_str(); "yielding the lane to a newer message")
                                }
                                Some(Stop::ClientCanceled) => {
                                    log::info!(target: "livekit.agents", lane = self.lane.as_str(); "the task was canceled by a client; yielding")
                                }
                                None => {}
                            }
                        }
                        let verdict =
                            if stop.is_some() { Verdict::Yield } else { Verdict::Continue };
                        verdicts.push(verdict).map_err(host_error)?;
                    }
                    Some(TurnEvent::Output { output }) => {
                        let out = HandlerOutput::try_from(output)?;
                        return Ok(out);
                    }
                    None => {
                        return Err(StorageError::Protocol(
                            "the handler's stream ended without its output".into(),
                        ));
                    }
                }
            }
        };
        let out = match run.lease() {
            Some(lease) => tokio::select! {
                out = read => out,
                e = lease.lost() => Err(e),
            },
            None => read.await,
        };
        let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        if events.close(deadline).await == Closed::TimedOut {
            log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = task_id; "the handler's stream did not close within its deadline; its host task is still running");
        }
        out.map(|out| (out, stop))
    }

    /// The task after the commit, and the reply.
    fn finish(
        &self,
        outcome: Outcome,
        task_id: &str,
        run: &Run,
        committed: Committed,
    ) -> TurnResult {
        let record = self.record_or_failed(committed.task, task_id, None, "not-written");
        TurnResult {
            task: record,
            output: outcome.output,
            thread_id: self.thread.id().to_owned(),
            answered_by: outcome.task_id,
            run_id: Some(run.run_id.clone()),
        }
    }
}
