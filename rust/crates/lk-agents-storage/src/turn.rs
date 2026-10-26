//! A message during a run: the five policies of plan 3.3.4 and 7.3, and the turn they drive. The
//! port of `storage/policies.py`'s `TurnRunner` over the [`run`](crate::run) module (plan 13.8).
//!
//! [`Turn::run`] is what the text door calls once per incoming A2A message. It resolves the policy
//! from the message's channel, takes the lane as the policy says (the inbox receive batch, the
//! `waiter` under `interrupt`, a plain wait under `queue`, no lease under `concurrent`, an error
//! under `reject`), runs the host's handler inside an open run, and writes the task row of plan
//! 5.5 for every turn: `submitted` by the receive under `batch`, `working` at pickup, then the
//! terminal state, with the Live extension's keys when the server interrupted a turn or merged a
//! message into a later one.
//!
//! The host boundary is one trait, [`TurnHost`]: the turn hands it a [`TurnContext`] and reads a
//! [`HandlerOutput`] back; the host polls the [`Boundary`] to ask whether to yield. Everything
//! crosses as JSON. What stays with the host: the developer-facing context object and the run
//! hooks.

mod host;
mod message;
mod step;

use std::sync::Arc;
use std::time::{Duration, Instant};

use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub use host::{
    Boundary, HandlerOutput, HostError, PendingCall, TurnContext, TurnHost, TurnResult,
};
pub use message::{
    A2A_CHANNEL, DEFAULT_DEBOUNCE, DEFAULT_POLICY_MAP, DELEGATION_CHANNEL, DebounceSpec,
    INTERRUPTION_NEW_MESSAGE, Message, Policy, PolicySpec, batch_to_chat_message,
    caller_history_of, channel_of, resolve_debounce, resolve_policy,
};
pub use step::{
    agent_text, artifact_text, channel_of_record, failed_message, result_for, task_by_message_id,
    unpersisted_failed,
};

use crate::StorageError;
use crate::run::{
    LEASE_POLL_MAX, LEASE_POLL_MIN, LEASE_WAIT_TIMEOUT, Lane, Run, RunOptions, RunOutcome,
    TaskRecord, TaskState, Thread, shortuuid,
};
use crate::run::{REASON_LEASE_TIMEOUT, REASON_STORAGE_UNAVAILABLE, REASON_WORKER_LOST};
use crate::schema::TaskData;
use crate::statements::{Duplicate, Receive, ReceiveRequest, Received, SubmittedTask};
use step::python_reason;

/// How many rows a lookup by id or `messageId` scans.
pub(crate) const LOOKUP_LIMIT: i64 = 500;
/// How many task rows the handler's context carries.
const TASKS_IN_CONTEXT: i64 = 50;
/// How many runs `reject` reads to name the running task.
const RUNS_FOR_REJECT: usize = 50;

/// `TurnRunner`'s arguments: the policy map, the debounce map, the wait, the state default. A
/// field left out is the crate's default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn")]
pub struct TurnOptions {
    /// `inbox`: one policy for every channel, or a map by channel over the
    /// default map; `None` is the default map.
    #[serde(default)]
    #[python(ty = "Policy | dict[str, Policy] | None", default = "None")]
    pub policy: PolicySpec,
    /// The wait before a turn starts, in seconds: one for every channel, or a map by channel;
    /// `None` is the defaults.
    #[serde(default)]
    #[python(ty = "float | dict[str, float] | None", default = "None")]
    pub debounce: DebounceSpec,
    /// How long a waiter polls for the lane before its task fails `lease-timeout`, in seconds;
    /// `None` is a minute.
    #[serde(default, with = "opt_seconds")]
    #[python(ty = "float | None", default = "None")]
    pub wait_timeout: Option<Duration>,
    /// The door's default-constructed state as JSON, for a fresh lane; `None` for a door with no
    /// state, which the `concurrent` policy requires.
    #[serde(default)]
    #[python(default = "None")]
    pub state_default: Option<Value>,
}

impl TurnOptions {
    /// `wait_timeout`, or the crate default.
    pub fn wait_timeout(&self) -> Duration {
        self.wait_timeout.unwrap_or(LEASE_WAIT_TIMEOUT)
    }
}

/// `Option<Duration>` as seconds, `null` for none.
mod opt_seconds {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(d: &Option<Duration>, s: S) -> Result<S::Ok, S::Error> {
        d.map(|d| d.as_secs_f64()).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Duration>, D::Error> {
        Option::<f64>::deserialize(d)?
            .map(|s| super::message::duration_from_secs(s).map_err(serde::de::Error::custom))
            .transpose()
    }
}

/// `_TurnContext`: what one `run` call carries between the policy steps.
pub(crate) struct Ctx {
    lane: Lane,
    #[allow(dead_code)]
    policy: Policy,
    last_run_id: Option<String>,
    /// A delegating caller's conversation, when this call's message carried one (plan 7.5).
    caller_history: Option<Value>,
}

/// One text door's turns: its lane, its policy map, its host (`TurnRunner`).
pub struct Turn {
    thread: Thread,
    lane: String,
    host: Arc<dyn TurnHost>,
    opts: TurnOptions,
}

/// A turn's row data: the message it came from.
fn turn_data(msg: &Message) -> TaskData {
    TaskData { history: vec![msg.to_a2a()], ..TaskData::default() }
}

impl Turn {
    pub fn new(thread: Thread, lane: &str, host: Arc<dyn TurnHost>, options: TurnOptions) -> Self {
        Self { thread, lane: lane.to_owned(), host, opts: options }
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

    pub fn debounce_for(&self, channel: &str) -> Duration {
        resolve_debounce(&self.opts.debounce, channel, self.policy_for(channel))
    }

    /// One incoming message through its policy (plan 3.3.4): `TurnRunner.run_turn`.
    ///
    /// `task_id` is the answer stream's id when the caller created one, minted otherwise.
    ///
    /// Errors: [`StorageError::RunRejected`] under `reject`, with the running task id when
    /// known; `VersionConflict` and `Unavailable` when the commit could not be written;
    /// [`StorageError::Cancelled`] when the host cancelled the handler.
    pub async fn run(
        &self,
        message: Value,
        task_id: Option<String>,
    ) -> Result<TurnResult, StorageError> {
        let caller_history = caller_history_of(&message);
        let msg = Message::from_a2a(&message, None, None)?;
        self.run_message(msg, task_id, caller_history).await
    }

    /// `run` over a parsed [`Message`]; `caller_history` is the `chat_ctx` part the message
    /// carried, which `Message::from_a2a` strips.
    pub async fn run_message(
        &self,
        msg: Message,
        task_id: Option<String>,
        caller_history: Option<Value>,
    ) -> Result<TurnResult, StorageError> {
        let policy = self.policy_for(&msg.channel);
        let debounce = self.debounce_for(&msg.channel);
        let task_id = task_id.unwrap_or_else(|| shortuuid("AST_"));
        let mut ctx =
            Ctx { lane: self.thread.lane(&self.lane), policy, last_run_id: None, caller_history };
        // a message that names a task waiting on a question continues that task; any other
        // message is a new task, whatever questions are open
        let named = match msg.task_id.as_deref() {
            Some(named) => ctx.lane.task(named).await?,
            None => None,
        };
        if let Some(row) = &named
            && row.data.history.iter().any(|m| {
                m.get("messageId").and_then(Value::as_str) == Some(msg.message_id.as_str())
            })
        {
            // a resend of a message that already continued this task: its answer, once
            let mut record = Some(row.clone());
            if !row.terminal() && row.state() != Some(TaskState::InputRequired) {
                record = self.await_task(&ctx, &row.id, false, true).await?;
            }
            return Ok(result_for(record, self.thread.id(), &row.id, None));
        }
        let waiting = named.filter(|r| r.state() == Some(TaskState::InputRequired)).map(|r| r.id);
        let continues = waiting.as_deref();
        match policy {
            Policy::Batch => {
                self.batch(&mut ctx, &msg, continues.unwrap_or(&task_id), debounce).await
            }
            Policy::Concurrent => self.concurrent(&ctx, &msg, &task_id).await,
            Policy::Reject => self.reject(&ctx, &msg, &task_id).await,
            Policy::Interrupt | Policy::Queue => {
                if !debounce.is_zero() {
                    tokio::time::sleep(debounce).await;
                }
                let interrupt = policy == Policy::Interrupt;
                self.exclusive(&ctx, &msg, &task_id, interrupt, continues).await
            }
        }
    }

    // -- interrupt and queue --

    /// `continues` is the task waiting on a question that this message answers: the turn runs
    /// under it, with no row of its own; `own_task_id` is the message's own row otherwise.
    async fn exclusive(
        &self,
        ctx: &Ctx,
        msg: &Message,
        own_task_id: &str,
        interrupt: bool,
        continues: Option<&str>,
    ) -> Result<TurnResult, StorageError> {
        // the early `submitted` row, the lease and the pickup, one batch (plan 3.3.4): a
        // `tasks/get` before the commit finds the task, and a holder that dies leaves an orphan.
        // A continuation picks the waiting row up instead, under a run id of its message, so a
        // resend of the message finds the run committed and runs nothing twice
        let task_id = continues.unwrap_or(own_task_id);
        let (run_id, submit) = match continues {
            Some(waiting) => (format!("{waiting}:{}", msg.message_id), None),
            None => (task_id.to_owned(), Some(submitted_task(&self.lane, msg, task_id))),
        };
        let opened = ctx
            .lane
            .open(RunOptions {
                interrupt,
                channel: Some(msg.channel.clone()),
                wait_timeout: self.opts.wait_timeout(),
                submit,
                pickup: true,
                ..RunOptions::new(run_id, task_id)
            })
            .await;
        let mut run = match opened {
            Ok(run) => run,
            Err(StorageError::LeaseTimeout(_)) if continues.is_some() => {
                // the question still waits; the message failed, on a row never written
                log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = task_id, threadId = self.thread.id(); "the lane was not released in time; the answer fails lease-timeout and the question stays open");
                return Ok(TurnResult {
                    task: unpersisted_failed(
                        own_task_id,
                        &self.lane,
                        Some(msg),
                        REASON_LEASE_TIMEOUT,
                    ),
                    output: None,
                    thread_id: self.thread.id().to_owned(),
                    answered_by: own_task_id.to_owned(),
                    run_id: None,
                });
            }
            Err(StorageError::LeaseTimeout(_)) => {
                log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = task_id, threadId = self.thread.id(); "the lane was not released in time; the task fails lease-timeout");
                let record = self.fail_unrun(ctx, task_id, REASON_LEASE_TIMEOUT).await?;
                return Ok(TurnResult {
                    task: record.unwrap_or_else(|| {
                        unpersisted_failed(task_id, &self.lane, Some(msg), REASON_LEASE_TIMEOUT)
                    }),
                    output: None,
                    thread_id: self.thread.id().to_owned(),
                    answered_by: task_id.to_owned(),
                    run_id: None,
                });
            }
            Err(e) => return Err(e),
        };
        if run.submitted == Some(false)
            && let Some(mut existing) = run.tasks.by_message_id(&msg.message_id).await?
        {
            // a retried messageId: the task it created, awaited when a runner still has it, or
            // failed worker-lost when its holder died and the lease lapsed
            self.close_run(&mut run, Ok(())).await?;
            if !existing.terminal()
                && let Some(ended) = self.await_task(ctx, &existing.id.clone(), false, true).await?
            {
                existing = ended;
            }
            return Ok(result_for(Some(existing), self.thread.id(), task_id, None));
        }
        if continues.is_some() && !run.closed && !run.picked_up {
            // another message answered the question first: this one is a new task
            self.close_run(&mut run, Ok(())).await?;
            return Box::pin(self.exclusive(ctx, msg, own_task_id, interrupt, None)).await;
        }
        let body = async {
            if run.closed || !run.picked_up {
                return Ok(None);
            }
            self.fail_orphans(&run, &[task_id.to_owned()]).await?;
            self.turn(ctx, &mut run, std::slice::from_ref(msg), &[task_id.to_owned()], interrupt)
                .await
                .map(Some)
        }
        .await;
        let outcome = match self.close_run(&mut run, body).await {
            Err(e @ (StorageError::VersionConflict { .. } | StorageError::Unavailable(_))) => {
                // the save was not written: the row the client was told about fails now, under
                // its own id, instead of staying `working` until another holder finds it
                if let Err(write) = self.fail_unrun(ctx, task_id, REASON_STORAGE_UNAVAILABLE).await
                {
                    log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = task_id; "the unsaved task could not be failed: {write}");
                }
                return Err(e);
            }
            other => other?,
        };
        match outcome {
            Some(outcome) => self.finish(ctx, outcome, task_id, &mut run).await,
            None => {
                let record = ctx.lane.task(task_id).await?;
                Ok(result_for(record, self.thread.id(), task_id, None))
            }
        }
    }

    /// The early `submitted` row (plan 3.3.4), before the lease is tried, so a `tasks/get` before
    /// the commit finds the task. Unfenced; the commit's upsert moves it to its terminal state.
    /// The batch policy writes its row in the receive batch instead.
    async fn submit(&self, msg: &Message, task_id: &str) -> Result<(), StorageError> {
        self.thread.db().insert_submitted_task(&submitted_task(&self.lane, msg, task_id)).await?;
        Ok(())
    }

    /// The `submitted` row goes `working` once the lease is held (plan 5.5), so a holder that
    /// dies leaves an orphan the next holder fails `worker-lost`. `false` when the row is
    /// already terminal: a client canceled it before the turn started, and nothing runs.
    async fn pickup(&self, task_id: &str) -> Result<bool, StorageError> {
        Ok(self.thread.db().pickup_task(&self.lane, task_id).await?.is_some())
    }

    /// The early row of a turn that never took the lease ends `failed` with `reason`, so no
    /// `submitted` row outlives a message the caller was told failed.
    async fn fail_unrun(
        &self,
        ctx: &Ctx,
        task_id: &str,
        reason: &str,
    ) -> Result<Option<TaskRecord>, StorageError> {
        let db = self.thread.db();
        if db.pickup_task(&self.lane, task_id).await?.is_some() {
            db.mark_orphan_failed(&self.lane, task_id, &failed_message(task_id, reason)).await?;
        }
        ctx.lane.task(task_id).await
    }

    /// Leaving a run the way the Python context manager does: the commit on success, `run_ended`
    /// with the outcome and nothing else on an error, which is returned.
    async fn close_run<T>(
        &self,
        run: &mut Run,
        body: Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        match body {
            Ok(v) => {
                run.commit(self.thread.now_ms()).await?;
                Ok(v)
            }
            Err(e) => {
                let outcome = match e {
                    StorageError::Cancelled(_) => RunOutcome::Canceled,
                    _ => RunOutcome::Failed,
                };
                run.abandon(self.thread.now_ms(), outcome, python_reason(&e)).await?;
                Err(e)
            }
        }
    }

    // -- inbox --

    async fn batch(
        &self,
        ctx: &mut Ctx,
        msg: &Message,
        task_id: &str,
        debounce: Duration,
    ) -> Result<TurnResult, StorageError> {
        let lease_token = shortuuid("lease_");
        let received = self
            .thread
            .db()
            .receive(&ReceiveRequest {
                lane: self.lane.clone(),
                holder: self.thread.holder().to_owned(),
                lease_token: lease_token.clone(),
                task_id: task_id.to_owned(),
                message_id: msg.message_id.clone(),
                message: msg.to_a2a(),
                name: None,
                data: turn_data(msg),
                internal: None,
            })
            .await?;
        match received {
            Receive::Duplicate(Duplicate { task_id: existing }) => {
                let existing_id = existing.unwrap_or_else(|| task_id.to_owned());
                let mut record = ctx.lane.task(&existing_id).await?;
                if let Some(r) = &record
                    && !r.terminal()
                {
                    record = self.await_task(ctx, &r.id.clone(), false, false).await?;
                }
                Ok(result_for(record, self.thread.id(), &existing_id, None))
            }
            Receive::Received(Received { acquired: Some(_), .. }) => {
                // this worker is the runner: let the burst land, then drain
                if !debounce.is_zero() {
                    tokio::time::sleep(debounce).await;
                }
                self.drain(ctx, Some(lease_token)).await?;
                let record = ctx.lane.task(task_id).await?;
                Ok(result_for(record, self.thread.id(), task_id, ctx.last_run_id.as_deref()))
            }
            Receive::Received(Received { acquired: None, .. }) => {
                let record = self.await_task(ctx, task_id, true, false).await?;
                Ok(result_for(record, self.thread.id(), task_id, ctx.last_run_id.as_deref()))
            }
        }
    }

    /// Polls the task and the lease: the task ends, or the lease frees and this worker drains the
    /// inbox (`drain_if_free`) or fails the orphaned `working` row `worker-lost`
    /// (`fail_if_free`: a resend of the dead holder's message is told to send it again).
    async fn await_task(
        &self,
        ctx: &Ctx,
        task_id: &str,
        drain_if_free: bool,
        fail_if_free: bool,
    ) -> Result<Option<TaskRecord>, StorageError> {
        let deadline = Instant::now() + self.opts.wait_timeout();
        let mut delay = LEASE_POLL_MIN;
        // the drain never holds a lease across the loop; a fresh context per pass
        let mut drain_ctx = Ctx {
            lane: ctx.lane.clone(),
            policy: ctx.policy,
            last_run_id: None,
            caller_history: None,
        };
        loop {
            let mut record = ctx.lane.task(task_id).await?;
            if record.as_ref().is_some_and(TaskRecord::terminal) {
                return Ok(record);
            }
            if drain_if_free && !ctx.lane.inbox_pending().await?.is_empty() {
                match self.drain(&mut drain_ctx, None).await {
                    Err(StorageError::LeaseHeld(_)) => {}
                    Err(e) => return Err(e),
                    Ok(()) => {
                        record = ctx.lane.task(task_id).await?;
                        if record.as_ref().is_some_and(TaskRecord::terminal) {
                            return Ok(record);
                        }
                    }
                }
            }
            if fail_if_free
                && let Some(r) = &record
                && r.status == TaskState::Working.as_str()
            {
                let opened = ctx
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
                        self.close_run(&mut run, body).await?;
                        record = ctx.lane.task(task_id).await?;
                        if record.as_ref().is_some_and(TaskRecord::terminal) {
                            return Ok(record);
                        }
                    }
                }
            }
            if Instant::now() >= deadline {
                log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = task_id, threadId = self.thread.id(); "a queued message was not answered in time");
                return Ok(record);
            }
            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(LEASE_POLL_MAX);
        }
    }

    /// The drain loop (plan 3.3.4): one turn per batch of pending rows until nothing is pending.
    /// `LeaseHeld` when another runner holds the lane.
    async fn drain(&self, ctx: &mut Ctx, lease_token: Option<String>) -> Result<(), StorageError> {
        let db = self.thread.db().clone();
        let mut lease_token = lease_token;
        loop {
            let rows = db.inbox_pending(&self.lane).await?;
            let Some(last) = rows.last() else {
                if let Some(token) = &lease_token {
                    db.drain_check(&self.lane, token).await?;
                }
                return Ok(());
            };
            let task_ids: Vec<String> = rows.iter().map(|r| r.task_id.clone()).collect();
            let mut run = ctx
                .lane
                .open(RunOptions {
                    wait: false,
                    channel: Some(channel_of(&last.message)),
                    ..RunOptions::new(shortuuid("drain_"), last.task_id.clone())
                })
                .await?;
            let last_id = last.id;
            let body = async {
                let mut batch: Vec<Message> = Vec::new();
                let mut batch_ids: Vec<String> = Vec::new();
                for (row, tid) in rows.iter().zip(&task_ids) {
                    let Some(was) = db.pickup_task(&self.lane, tid).await? else {
                        continue; // terminal already: answered by an earlier drain
                    };
                    if was == TaskState::Working.as_str() {
                        db.mark_orphan_failed(
                            &self.lane,
                            tid,
                            &failed_message(tid, REASON_WORKER_LOST),
                        )
                        .await?;
                        log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = tid.as_str(); "a message whose drain died fails worker-lost");
                        continue;
                    }
                    batch.push(Message::from_a2a(
                        &row.message,
                        Some(row.received_at as f64),
                        Some(tid),
                    )?);
                    batch_ids.push(tid.clone());
                }
                if !batch.is_empty() {
                    self.fail_orphans(&run, &task_ids).await?;
                    self.turn(ctx, &mut run, &batch, &batch_ids, false).await?;
                }
                run.drained_id = Some(last_id);
                run.keep_lease = true;
                Ok(())
            }
            .await;
            match self.close_run(&mut run, body).await {
                Err(e @ (StorageError::VersionConflict { .. } | StorageError::Unavailable(_))) => {
                    // the save was not written: the batch's rows fail under their own ids
                    for tid in &task_ids {
                        let message = failed_message(tid, REASON_STORAGE_UNAVAILABLE);
                        if let Err(write) = db.mark_orphan_failed(&self.lane, tid, &message).await {
                            log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = tid.as_str(); "the unsaved task could not be failed: {write}");
                        }
                    }
                    return Err(e);
                }
                other => other?,
            }
            ctx.last_run_id = Some(run.run_id.clone());
            let token = run
                .lease_token()
                .ok_or_else(|| StorageError::Invalid("a drain run without a lease".into()))?
                .to_owned();
            let check = db.drain_check(&self.lane, &token).await?;
            lease_token = Some(token);
            if check.released {
                return Ok(());
            }
        }
    }

    // -- concurrent and reject --

    async fn concurrent(
        &self,
        ctx: &Ctx,
        msg: &Message,
        task_id: &str,
    ) -> Result<TurnResult, StorageError> {
        if self.opts.state_default.is_some() {
            return Err(StorageError::Invalid(
                "the concurrent policy requires a door with no state".into(),
            ));
        }
        self.submit(msg, task_id).await?;
        let mut run = ctx
            .lane
            .open(RunOptions {
                exclusive: false,
                channel: Some(msg.channel.clone()),
                ..RunOptions::new(task_id, task_id)
            })
            .await?;
        let body =
            self.turn(ctx, &mut run, std::slice::from_ref(msg), &[task_id.to_owned()], false).await;
        let outcome = self.close_run(&mut run, body).await?;
        self.finish(ctx, outcome, task_id, &mut run).await
    }

    async fn reject(
        &self,
        ctx: &Ctx,
        msg: &Message,
        task_id: &str,
    ) -> Result<TurnResult, StorageError> {
        if let Some(info) = ctx.lane.info().await?
            && info.lease_until > self.thread.now_ms() / 1000
        {
            let running = ctx
                .lane
                .runs(RUNS_FOR_REJECT)
                .await?
                .into_iter()
                .find(|r| r.interrupted())
                .and_then(|r| r.task_id);
            return Err(StorageError::RunRejected { running_task_id: running });
        }
        match self.exclusive_nowait(ctx, msg, task_id).await {
            Err(StorageError::LeaseHeld(_)) => {
                Err(StorageError::RunRejected { running_task_id: None })
            }
            other => other,
        }
    }

    async fn exclusive_nowait(
        &self,
        ctx: &Ctx,
        msg: &Message,
        task_id: &str,
    ) -> Result<TurnResult, StorageError> {
        let mut run = ctx
            .lane
            .open(RunOptions {
                wait: false,
                channel: Some(msg.channel.clone()),
                ..RunOptions::new(task_id, task_id)
            })
            .await?;
        // the row is written under the lease: a refused message leaves none (docs/text-endpoints.md)
        let body = async {
            if run.closed {
                return Ok(None);
            }
            self.submit(msg, task_id).await?;
            if !self.pickup(task_id).await? {
                return Ok(None);
            }
            self.fail_orphans(&run, &[task_id.to_owned()]).await?;
            self.turn(ctx, &mut run, std::slice::from_ref(msg), &[task_id.to_owned()], false)
                .await
                .map(Some)
        }
        .await;
        match self.close_run(&mut run, body).await? {
            Some(outcome) => self.finish(ctx, outcome, task_id, &mut run).await,
            None => Ok(result_for(ctx.lane.task(task_id).await?, self.thread.id(), task_id, None)),
        }
    }

    // -- the turn --

    /// A dead holder's `working` rows, found under this run's lease: each fails `worker-lost`
    /// and nothing runs again; the client resends. `skip` is this run's own row.
    async fn fail_orphans(&self, run: &Run, skip: &[String]) -> Result<(), StorageError> {
        let db = self.thread.db();
        for row in run.tasks.working().await? {
            if skip.contains(&row.id) {
                continue;
            }

            let message = failed_message(&row.id, REASON_WORKER_LOST);
            db.mark_orphan_failed(&self.lane, &row.id, &message).await?;
            log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = row.id.as_str(); "a task whose worker died fails worker-lost");
        }
        Ok(())
    }
}

/// The early `submitted` row of a turn: its message as the task's history.
fn submitted_task(lane: &str, msg: &Message, task_id: &str) -> SubmittedTask {
    SubmittedTask {
        lane: lane.to_owned(),
        task_id: task_id.to_owned(),
        message_id: Some(msg.message_id.clone()),
        name: None,
        data: turn_data(msg),
        internal: None,
    }
}

/// One message through the lane's policy: the entry point the A2A server calls (plan 7.3), a
/// [`Turn`] built for one call.
pub async fn run_turn(
    thread: Thread,
    lane: &str,
    host: Arc<dyn TurnHost>,
    options: TurnOptions,
    message: Value,
    task_id: Option<String>,
) -> Result<TurnResult, StorageError> {
    Turn::new(thread, lane, host, options).run(message, task_id).await
}

/// `completed`, `failed`, `canceled`, `rejected`.
pub fn is_terminal(status: &str) -> bool {
    TaskState::parse(status).is_some_and(TaskState::is_terminal)
}

/// The metadata of a `canceled` row the server interrupted, as the turn writes it.
pub fn interrupted_metadata() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert(crate::run::INTERRUPTED_KEY.into(), Value::Bool(true));
    m.insert(
        crate::run::INTERRUPTION_REASON_KEY.into(),
        Value::String(INTERRUPTION_NEW_MESSAGE.into()),
    );
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_default_to_the_plan_numbers() {
        let o = TurnOptions::default();
        assert_eq!(o.wait_timeout(), LEASE_WAIT_TIMEOUT);
        assert_eq!(o.policy, PolicySpec::Default);
        assert_eq!(o.state_default, None);
        assert!(is_terminal("completed") && !is_terminal("working") && !is_terminal("x"));
        assert_eq!(
            serde_json::to_string(&interrupted_metadata()).unwrap(),
            r#"{"livekit/interrupted":true,"livekit/interruptionReason":"new-message"}"#
        );
    }

    // Catches: the options' wire shape drifting from what the host's `TurnOptions` model writes
    // (a policy as a string or a map, seconds as numbers, `null` for a default), or an unknown
    // policy accepted or reported without its name.
    #[test]
    fn options_cross_as_the_models_json() {
        let o: TurnOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(o, TurnOptions::default());
        let json = r#"{"policy": {"sms": "batch"}, "debounce": 2.5, "wait_timeout": 1, "state_default": null}"#;
        let o: TurnOptions = serde_json::from_str(json).unwrap();
        assert_eq!(
            o.policy,
            PolicySpec::ByChannel([("sms".to_owned(), Policy::Batch)].into_iter().collect())
        );
        assert_eq!(o.debounce, DebounceSpec::All(Duration::from_millis(2500)));
        assert_eq!(o.wait_timeout(), Duration::from_secs(1));
        let back: TurnOptions = serde_json::from_value(serde_json::to_value(&o).unwrap()).unwrap();
        assert_eq!(back, o);
        assert_eq!(
            serde_json::to_value(TurnOptions::default()).unwrap(),
            serde_json::json!({"policy": null, "debounce": null, "wait_timeout": null, "state_default": null})
        );
        let e = serde_json::from_str::<TurnOptions>(r#"{"policy": "later"}"#).unwrap_err();
        assert!(e.to_string().contains("later"), "{e}");
        assert!(serde_json::from_str::<TurnOptions>(r#"{"debounce": -1}"#).is_err());
    }
}
