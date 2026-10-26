//! `lk-agents-a2a`: the A2A text door's turn engine. [`Turn::run`] takes one incoming message
//! through its channel's [`Policy`], runs the host's handler in an open run, and writes the
//! message's task row: `submitted`, `working`, then its end state.
//!
//! [`TurnHost::handle`] answers a stream of [`TurnEvent`]s: each [`Checkpoint`] answered with a
//! [`Verdict`], then the [`HandlerOutput`]. A turn yields at a checkpoint once a newer message
//! waits under `interrupt` or a client canceled its task.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{Clock, HOST_TEARDOWN_TIMEOUT, StreamSink, StreamSource};
use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use livekit_protocol::agent::chat_context::{ChatItem, chat_item};
use livekit_protocol::agent::chat_message::{ChatContent, chat_content::Payload};
use livekit_protocol::agent::{ChatMessage, ChatRole, FunctionCallOutput};
use lk_agents_storage::db::{Committed, Lease, Receive, TaskWrite};
use lk_agents_storage::run::{
    CHANNEL_KEY, CommitOptions, INTERRUPTED_KEY, INTERRUPTION_REASON_KEY, ItemRow, KIND_KEY,
    LEASE_WAIT_TIMEOUT, Lane, LanePoll, MERGED_INTO_KEY, REASON_KEY, REASON_LEASE_TIMEOUT,
    REASON_WORKER_LOST, Run, RunOptions, RunOutcome, RunUsage, TaskPatch, TaskRow, TaskState,
    Thread, WaitOptions, agent_message, timestamp,
};
use lk_agents_storage::schema::TaskData;
use lk_agents_storage::{StorageError, now_ms, now_secs, shortuuid};

/// How many rows a lookup by id or `messageId` scans.
/// How many task rows the handler's context carries.
const TASKS_IN_CONTEXT: i64 = 50;
/// How many runs `reject` reads to name the running task.
const RUNS_FOR_REJECT: usize = 50;

/// A text door's turn options; a field left out is the default.
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
            .map(|s| super::duration_from_secs(s).map_err(serde::de::Error::custom))
            .transpose()
    }
}

/// What one `run` call carries between the policy steps.
pub(crate) struct Ctx {
    lane: Lane,
    #[allow(dead_code)]
    policy: Policy,
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
}

/// A turn's row data: the message it came from.
fn turn_data(msg: &Message) -> TaskData {
    TaskData { history: vec![msg.to_a2a()], ..TaskData::default() }
}

impl Turn {
    pub fn new(
        thread: Thread,
        lane: &str,
        host: Arc<dyn TurnHost>,
        clock: Arc<dyn Clock>,
        options: TurnOptions,
    ) -> Self {
        Self { thread, lane: lane.to_owned(), host, clock, opts: options }
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
        let msg = Message::from_a2a(&message, None, None)?;
        let policy = self.policy_for(&msg.channel);
        let debounce = resolve_debounce(&self.opts.debounce, &msg.channel, policy);
        let task_id = task_id.unwrap_or_else(|| shortuuid("AST_"));
        let mut ctx =
            Ctx { lane: self.thread.lane(&self.lane), policy, last_run_id: None, caller_history };
        match policy {
            Policy::Batch => self.batch(&mut ctx, &msg, &task_id, debounce).await,
            Policy::Concurrent => self.concurrent(&ctx, &msg, &task_id).await,
            Policy::Reject => self.reject(&ctx, &msg, &task_id).await,
            Policy::Interrupt | Policy::Queue => {
                if !debounce.is_zero() {
                    tokio::time::sleep(debounce).await;
                }
                self.exclusive(&ctx, &msg, &task_id, policy == Policy::Interrupt).await
            }
        }
    }

    async fn exclusive(
        &self,
        ctx: &Ctx,
        msg: &Message,
        task_id: &str,
        interrupt: bool,
    ) -> Result<TurnResult, StorageError> {
        // the early `submitted` row, the lease and the pickup, one batch: a
        // `tasks/get` before the commit finds the task, and a holder that dies leaves an orphan
        let opened = ctx
            .lane
            .open(RunOptions {
                interrupt,
                channel: Some(msg.channel.clone()),
                wait_timeout: self.opts.wait_timeout(),
                submit: Some(submitted_task(msg, task_id)),
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
                let db = ctx.lane.db();
                if db.pickup(task_id).await?.is_some() {
                    db.fail_orphan(task_id, &failed_message(task_id, REASON_LEASE_TIMEOUT)).await?;
                }
                let record = ctx.lane.task(task_id).await?;
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
            self.close_run(&mut run, Ok(()), |_| CommitOptions::default()).await?;
            if !existing.terminal()
                && let Some(ended) =
                    self.await_task(ctx, &existing.task_id.clone(), false, true).await?
            {
                existing = ended;
            }
            return Ok(result_for(Some(existing), self.thread.id(), task_id, None));
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
        match self.close_run(&mut run, body, answering).await? {
            (Some(outcome), committed) => Ok(self.finish(outcome, task_id, &run, committed)),
            (None, _) => {
                let record = ctx.lane.task(task_id).await?;
                Ok(result_for(record, self.thread.id(), task_id, None))
            }
        }
    }

    /// The early `submitted` row, before the lease is tried; `false` when it already existed.
    async fn submit(&self, ctx: &Ctx, msg: &Message, task_id: &str) -> Result<bool, StorageError> {
        ctx.lane.db().submit(&submitted_task(msg, task_id)).await
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
        ctx: &mut Ctx,
        msg: &Message,
        task_id: &str,
        debounce: Duration,
    ) -> Result<TurnResult, StorageError> {
        let received = ctx
            .lane
            .db()
            .receive(self.thread.holder(), task_id, &msg.message_id, &msg.to_a2a(), &turn_data(msg))
            .await?;
        match received {
            Receive::Duplicate(existing) => {
                let existing_id = existing.unwrap_or_else(|| task_id.to_owned());
                let mut record = ctx.lane.task(&existing_id).await?;
                if let Some(r) = &record
                    && !r.terminal()
                {
                    // the first receive may have landed with its answer lost: nobody drains the
                    // row unless this worker does once the lease frees
                    record = self.await_task(ctx, &r.task_id.clone(), true, true).await?;
                }
                Ok(result_for(record, self.thread.id(), &existing_id, None))
            }
            Receive::Received(Some(lease)) => {
                // this worker is the runner: let the burst land, then drain
                if !debounce.is_zero() {
                    tokio::time::sleep(debounce).await;
                }
                self.drain(ctx, Some(lease)).await?;
                let record = ctx.lane.task(task_id).await?;
                Ok(result_for(record, self.thread.id(), task_id, ctx.last_run_id.as_deref()))
            }
            Receive::Received(None) => {
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
    ) -> Result<Option<TaskRow>, StorageError> {
        let mut poll = LanePoll::new(self.opts.wait_timeout());
        // the drain never holds a lease across the loop; a fresh context per pass
        let mut drain_ctx = Ctx {
            lane: ctx.lane.clone(),
            policy: ctx.policy,
            last_run_id: None,
            caller_history: None,
        };
        loop {
            let mut record = ctx.lane.task(task_id).await?;
            if record.as_ref().is_some_and(TaskRow::terminal) {
                return Ok(record);
            }
            if drain_if_free && !ctx.lane.db().inbox(true).await?.is_empty() {
                match self.drain(&mut drain_ctx, None).await {
                    Err(StorageError::LeaseHeld(_)) => {}
                    Err(e) => return Err(e),
                    Ok(()) => {
                        record = ctx.lane.task(task_id).await?;
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
                        self.close_run(&mut run, body, |_| CommitOptions::default()).await?;
                        record = ctx.lane.task(task_id).await?;
                        if record.as_ref().is_some_and(TaskRow::terminal) {
                            return Ok(record);
                        }
                    }
                }
            }
            if !poll.pause().await {
                log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = task_id, threadId = self.thread.id(); "a queued message was not answered in time");
                return Ok(record);
            }
        }
    }

    /// The drain loop: one turn per batch of pending rows until nothing is pending.
    /// `LeaseHeld` when another runner holds the lane.
    async fn drain(&self, ctx: &mut Ctx, lease: Option<Lease>) -> Result<(), StorageError> {
        let db = ctx.lane.db().clone();
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
                    let Some(was) = db.pickup(tid).await? else {
                        continue; // terminal already: answered by an earlier drain
                    };
                    if was == TaskState::Working {
                        db.fail_orphan(tid, &failed_message(tid, REASON_WORKER_LOST)).await?;
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
                Ok(())
            }
            .await;
            let keep = |_: &()| CommitOptions {
                keep_lease: true,
                drained_id: Some(last_id),
                answer_task: None,
            };
            self.close_run(&mut run, body, keep).await?;
            ctx.last_run_id = Some(run.run_id.clone());
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
        ctx: &Ctx,
        msg: &Message,
        task_id: &str,
    ) -> Result<TurnResult, StorageError> {
        if !self.submit(ctx, msg, task_id).await? {
            // a resent task or message id: the task it created, awaited, not a second run
            let mut existing = match ctx.lane.task(task_id).await? {
                Some(r) => Some(r),
                None => ctx.lane.db().task_for_message(&msg.message_id).await?,
            };
            if let Some(r) = &existing
                && !r.terminal()
            {
                existing = self.await_task(ctx, &r.task_id.clone(), false, false).await?;
            }
            let id = existing.as_ref().map_or(task_id, |r| r.task_id.as_str()).to_owned();
            return Ok(result_for(existing, self.thread.id(), &id, None));
        }
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
        let (outcome, committed) = self.close_run(&mut run, body, answers).await?;
        Ok(self.finish(outcome, task_id, &run, committed))
    }

    async fn reject(
        &self,
        ctx: &Ctx,
        msg: &Message,
        task_id: &str,
    ) -> Result<TurnResult, StorageError> {
        if let Some(info) = ctx.lane.info().await?
            && info.lease_until > now_ms() / 1000
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
        let result = async {
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
                self.submit(ctx, msg, task_id).await?;
                // the row goes `working` once the lease is held, so a holder that dies leaves an
                // orphan; a row a client canceled before the turn started runs nothing
                if ctx.lane.db().pickup(task_id).await?.is_none() {
                    return Ok(None);
                }
                self.fail_orphans(&run, &[task_id.to_owned()]).await?;
                self.turn(ctx, &mut run, std::slice::from_ref(msg), &[task_id.to_owned()], false)
                    .await
                    .map(Some)
            }
            .await;
            match self.close_run(&mut run, body, answering).await? {
                (Some(outcome), committed) => Ok(self.finish(outcome, task_id, &run, committed)),
                (None, _) => {
                    Ok(result_for(ctx.lane.task(task_id).await?, self.thread.id(), task_id, None))
                }
            }
        }
        .await;
        match result {
            Err(StorageError::LeaseHeld(_)) => {
                Err(StorageError::RunRejected { running_task_id: None })
            }
            other => other,
        }
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
                .fail_orphan(&row.task_id, &failed_message(&row.task_id, REASON_WORKER_LOST))
                .await?;
            log::warn!(target: "livekit.agents", lane = self.lane.as_str(), taskId = row.task_id.as_str(); "a task whose worker died fails worker-lost");
        }
        Ok(())
    }
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

/// The commit of a turn that ran: its answer task read back with the writes.
fn answers(outcome: &Outcome) -> CommitOptions {
    CommitOptions { answer_task: Some(outcome.task_id.clone()), ..CommitOptions::default() }
}

fn answering(outcome: &Option<Outcome>) -> CommitOptions {
    outcome.as_ref().map(answers).unwrap_or_default()
}

/// The metadata of a `canceled` row the server interrupted, as the turn writes it.
pub fn interrupted_metadata() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert(INTERRUPTED_KEY.into(), Value::Bool(true));
    m.insert(INTERRUPTION_REASON_KEY.into(), Value::String(INTERRUPTION_NEW_MESSAGE.into()));
    m
}

/// A message during a run: one of the five policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Host)]
#[serde(rename_all = "lowercase")]
#[python(literal, module = "storage_turn")]
pub enum Policy {
    #[host(rename = "batch")]
    Batch,
    #[host(rename = "interrupt")]
    Interrupt,
    #[host(rename = "queue")]
    Queue,
    #[host(rename = "concurrent")]
    Concurrent,
    #[host(rename = "reject")]
    Reject,
}

impl Policy {
    /// `None` for a string no policy spells.
    pub fn parse(s: &str) -> Option<Policy> {
        Policy::from_wire(s)
    }
}

/// `inbox` as a door declares it: nothing (the default map), one policy for every
/// channel, or a map that overrides the default map channel by channel. On the wire `null`, a
/// policy string, or an object of them by channel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum PolicySpec {
    #[default]
    Default,
    All(Policy),
    ByChannel(HashMap<String, Policy>),
}

/// `PolicySpec` as JSON carries it: the policy strings, read by name so an unknown one is named
/// in the error rather than reported as "no variant matched".
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum PolicyWire {
    All(String),
    ByChannel(HashMap<String, String>),
}

impl Serialize for PolicySpec {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let wire = match self {
            PolicySpec::Default => None,
            PolicySpec::All(p) => Some(PolicyWire::All(p.as_str().to_owned())),
            PolicySpec::ByChannel(m) => Some(PolicyWire::ByChannel(
                m.iter().map(|(k, v)| (k.clone(), v.as_str().to_owned())).collect(),
            )),
        };
        wire.serialize(s)
    }
}

impl<'de> Deserialize<'de> for PolicySpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let named = |s: &str| {
            Policy::parse(s)
                .ok_or_else(|| serde::de::Error::custom(format!("unknown policy {s:?}")))
        };
        Ok(match Option::<PolicyWire>::deserialize(d)? {
            None => PolicySpec::Default,
            Some(PolicyWire::All(s)) => PolicySpec::All(named(&s)?),
            Some(PolicyWire::ByChannel(m)) => PolicySpec::ByChannel(
                m.into_iter().map(|(k, v)| named(&v).map(|p| (k, p))).collect::<Result<_, _>>()?,
            ),
        })
    }
}

/// The debounce as a door declares it: nothing (the defaults), one wait for every channel, or a
/// map per channel. On the wire `null`, seconds, or an object of seconds by channel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum DebounceSpec {
    #[default]
    Default,
    All(Duration),
    ByChannel(HashMap<String, Duration>),
}

/// `DebounceSpec` as JSON carries it: seconds as a number.
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum DebounceWire {
    All(f64),
    ByChannel(HashMap<String, f64>),
}

/// Seconds as a `Duration`; negative, infinite or NaN is the caller's error.
pub(crate) fn duration_from_secs(secs: f64) -> Result<Duration, String> {
    Duration::try_from_secs_f64(secs).map_err(|e| e.to_string())
}

impl Serialize for DebounceSpec {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let wire = match self {
            DebounceSpec::Default => None,
            DebounceSpec::All(d) => Some(DebounceWire::All(d.as_secs_f64())),
            DebounceSpec::ByChannel(m) => Some(DebounceWire::ByChannel(
                m.iter().map(|(k, v)| (k.clone(), v.as_secs_f64())).collect(),
            )),
        };
        wire.serialize(s)
    }
}

impl<'de> Deserialize<'de> for DebounceSpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let secs = |s: f64| duration_from_secs(s).map_err(serde::de::Error::custom);
        Ok(match Option::<DebounceWire>::deserialize(d)? {
            None => DebounceSpec::Default,
            Some(DebounceWire::All(s)) => DebounceSpec::All(secs(s)?),
            Some(DebounceWire::ByChannel(m)) => DebounceSpec::ByChannel(
                m.into_iter().map(|(k, v)| secs(v).map(|d| (k, d))).collect::<Result<_, _>>()?,
            ),
        })
    }
}

/// the default map, keyed by `livekit/channel`.
pub const DEFAULT_POLICY_MAP: [(&str, Policy); 5] = [
    ("sms", Policy::Batch),
    ("whatsapp", Policy::Batch),
    ("email", Policy::Batch),
    ("a2a", Policy::Interrupt),
    ("delegation", Policy::Queue),
];

/// two seconds for the inbox channels, zero for `a2a`; starting points.
pub const DEFAULT_DEBOUNCE: [(&str, Duration); 5] = [
    ("sms", Duration::from_secs(2)),
    ("whatsapp", Duration::from_secs(2)),
    ("email", Duration::from_secs(2)),
    ("a2a", Duration::ZERO),
    ("delegation", Duration::ZERO),
];

pub const INTERRUPTION_NEW_MESSAGE: &str = "new-message";
pub const A2A_CHANNEL: &str = "a2a";
pub const DELEGATION_CHANNEL: &str = "delegation";

/// `livekit/channel` of an A2A message; `livekit/kind = delegation` counts as `delegation`, and a
/// message with no channel counts as `a2a`.
pub fn channel_of(message: &a2a::Message) -> String {
    let metadata = message.metadata.as_ref();
    let field = |key: &str| metadata.and_then(|m| m.get(key)).and_then(Value::as_str);
    if field(KIND_KEY) == Some(DELEGATION_CHANNEL) {
        return DELEGATION_CHANNEL.to_owned();
    }
    match field(CHANNEL_KEY) {
        Some(c) if !c.is_empty() => c.to_owned(),
        _ => A2A_CHANNEL.to_owned(),
    }
}

/// One policy for a channel: a value applies to every channel, a map overrides the default map;
/// a channel in neither gets the `a2a` policy.
pub fn resolve_policy(policy: &PolicySpec, channel: &str) -> Policy {
    if let PolicySpec::All(p) = policy {
        return *p;
    }
    let mut table: HashMap<&str, Policy> = DEFAULT_POLICY_MAP.into_iter().collect();
    if let PolicySpec::ByChannel(map) = policy {
        for (k, v) in map {
            table.insert(k.as_str(), *v);
        }
    }
    table.get(channel).or_else(|| table.get(A2A_CHANNEL)).copied().unwrap_or(Policy::Interrupt)
}

/// The wait before a turn starts so a burst has landed; per channel, or one value for all. A
/// channel the defaults do not name waits the `sms` default under `batch` and nothing otherwise.
pub fn resolve_debounce(debounce: &DebounceSpec, channel: &str, policy: Policy) -> Duration {
    match debounce {
        DebounceSpec::All(d) => return *d,
        DebounceSpec::ByChannel(map) => {
            if let Some(d) = map.get(channel) {
                return *d;
            }
        }
        DebounceSpec::Default => {}
    }
    let defaults: HashMap<&str, Duration> = DEFAULT_DEBOUNCE.into_iter().collect();
    if let Some(d) = defaults.get(channel) {
        return *d;
    }
    if policy == Policy::Batch { defaults["sms"] } else { Duration::ZERO }
}

/// One incoming A2A message, as the handler sees it in `ctx.messages`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn", config = "frozen=True")]
pub struct Message {
    pub message_id: String,
    /// v1 parts, as protobuf JSON.
    #[python(ty = "list[dict[str, Any]]", import = "typing:Any")]
    pub parts: Vec<a2a::Part>,
    #[python(ty = "dict[str, Any]", import = "typing:Any")]
    pub metadata: Map<String, Value>,
    pub channel: String,
    /// Unix seconds.
    pub received_at: f64,
    pub task_id: Option<String>,
    pub context_id: Option<String>,
    /// The A2A message as received, less a delegation's `chat_ctx` part.
    #[python(ty = "dict[str, Any]", import = "typing:Any")]
    pub raw: a2a::Message,
}

impl Message {
    /// A message without a `messageId` is invalid.
    pub fn from_a2a(
        message: &a2a::Message,
        received_at: Option<f64>,
        task_id: Option<&str>,
    ) -> Result<Message, StorageError> {
        if message.message_id.is_empty() {
            return Err(StorageError::Invalid("an A2A message needs a messageId".into()));
        }
        // a delegation's `chat_ctx` part is the caller's conversation, read by `caller_history_of`
        // before the message is parsed; it is not part of the turn's text
        let mut raw = message.clone();
        raw.parts.retain(|p| !is_chat_ctx_part(p));
        let nonempty = |s: &Option<String>| s.clone().filter(|s| !s.is_empty());
        Ok(Message {
            message_id: message.message_id.clone(),
            parts: raw.parts.clone(),
            metadata: message.metadata.clone().unwrap_or_default().into_iter().collect(),
            channel: channel_of(message),
            received_at: received_at.unwrap_or_else(now_secs),
            task_id: task_id.map(str::to_owned).or_else(|| nonempty(&message.task_id)),
            context_id: nonempty(&message.context_id),
            raw,
        })
    }

    /// The text parts, joined by newlines.
    pub fn text(&self) -> String {
        self.parts.iter().filter_map(a2a::Part::as_text).collect::<Vec<_>>().join("\n")
    }

    /// The first data part's object.
    pub fn data(&self) -> Option<Map<String, Value>> {
        self.parts.iter().find_map(|p| match &p.content {
            a2a::PartContent::Data(Value::Object(o)) => Some(o.clone()),
            _ => None,
        })
    }

    /// The parts as chat content: text as is, data as compact JSON, a file as its part's JSON.
    pub fn content(&self) -> Vec<String> {
        self.parts
            .iter()
            .map(|p| match &p.content {
                a2a::PartContent::Text(t) => t.clone(),
                a2a::PartContent::Data(d) => d.to_string(),
                _ => serde_json::to_string(p).unwrap_or_default(),
            })
            .collect()
    }

    /// The message as received; a user's when the wire named no role.
    pub fn to_a2a(&self) -> a2a::Message {
        let mut out = self.raw.clone();
        if out.role == a2a::Role::Unspecified {
            out.role = a2a::Role::User;
        }
        out
    }
}

/// The batch as one user chat message with every part in order.
pub fn batch_to_chat_message(messages: &[Message], created_at: f64) -> ItemRow {
    let mut content: Vec<String> = messages.iter().flat_map(Message::content).collect();
    if content.is_empty() {
        content.push(String::new());
    }
    chat_message(ChatRole::User, content, created_at)
}

fn chat_message(role: ChatRole, content: Vec<String>, created_at: f64) -> ItemRow {
    ItemRow::new(chat_item::Item::Message(ChatMessage {
        id: shortuuid("item_"),
        role: role as i32,
        content: content
            .into_iter()
            .map(|t| ChatContent { payload: Some(Payload::Text(t)) })
            .collect(),
        created_at: Some(timestamp(created_at)),
        ..Default::default()
    }))
}

/// Whether a part is a delegation's `chat_ctx` data part: `livekit/kind = chat_ctx` on the part.
fn is_chat_ctx_part(part: &a2a::Part) -> bool {
    matches!(part.content, a2a::PartContent::Data(_))
        && part.metadata.as_ref().and_then(|m| m.get(KIND_KEY)).and_then(Value::as_str)
            == Some("chat_ctx")
}

/// The `chat_ctx` data part of a delegation, as the part's `data`: the caller's
/// conversation, read from the A2A message before `Message::from_a2a` strips the part.
pub fn caller_history_of(message: &a2a::Message) -> Option<Value> {
    message.parts.iter().filter(|p| is_chat_ctx_part(p)).find_map(|p| match &p.content {
        a2a::PartContent::Data(d) if d.is_object() => Some(d.clone()),
        _ => None,
    })
}

/// The host's handler, the `a2a_session` handler or `Agent.run()`.
#[lk_agents_abi::host(proxy(host = "livekit.agents.storage.context.TurnHost", clock))]
pub trait TurnHost: Send + Sync {
    /// Opens the handler on one turn (`ctx` is the [`TurnContext`] as JSON): its events, and where
    /// each checkpoint's verdict goes.
    #[allow(clippy::type_complexity)]
    fn handle(
        &self,
        ctx: Value,
    ) -> Result<(Box<dyn StreamSink<Verdict>>, Box<dyn StreamSource<TurnEvent>>), ProviderError>;
}

/// The turn's answer to a [`Checkpoint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Host)]
#[serde(rename_all = "lowercase")]
#[python(literal, module = "storage_turn")]
pub enum Verdict {
    #[host(rename = "continue")]
    Continue,
    /// Stop here with what ran so far, no reply.
    #[host(rename = "yield")]
    Yield,
}

/// A place the handler can stop safely: before each model call and before the answer streams.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "checkpoint")]
pub struct Checkpoint {}

/// The handler's end: its [`HandlerOutput`], as JSON.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "output")]
pub struct Output {
    pub output: Value,
}

/// What the handler's stream carries, told apart by `kind`.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract)]
pub enum TurnEvent {
    Checkpoint(Checkpoint),
    Output(Output),
}

/// Why a turn yielded at a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stop {
    /// A newer message waits for the lane under `interrupt`.
    Interrupted,
    /// A client canceled the task with `tasks/cancel`.
    ClientCanceled,
}

/// A failure the host's stream raised: a cancellation ends the run `canceled`, a storage error
/// the handler met fails it as that error.
fn host_error(e: ProviderError) -> StorageError {
    let ProviderError::Host { error, .. } = &e else {
        return StorageError::Unavailable(e.to_string());
    };
    let message = error.message.clone();
    match error.host_kind() {
        "CancelledError" => StorageError::Cancelled(message),
        "ThreadNotFound" => StorageError::ThreadNotFound(message),
        "StorageUnavailable" => StorageError::Unavailable(message),
        kind => StorageError::Server { code: kind.to_owned(), message },
    }
}

/// What the handler is opened on: the message, the batch, the history, the state, the task.
#[derive(Debug, Clone, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn", config = "frozen=True")]
pub struct TurnContext {
    /// The database id of the session this turn runs in.
    pub thread_id: String,
    /// The lane: the registered agent this turn belongs to.
    pub lane: String,
    /// The batch as one user chat message (`batch_to_chat_message`).
    #[python(ty = "dict[str, Any]", import = "typing:Any")]
    pub message: ChatItem,
    /// The individual messages with their channel and time; `batch.last()` is the one answered.
    pub batch: Vec<Message>,
    /// The lane's history as loaded, plus the pending input-required answer when there was one.
    pub history: Vec<ItemRow>,
    /// The state as loaded; `None` for a fresh lane.
    pub state: Option<Value>,
    /// Context field writes require the lane's exclusive lease.
    #[serde(default)]
    #[python(default = "False")]
    pub read_only: bool,
    /// The task that carries the reply.
    pub task_id: String,
    /// The lane's task rows, newest first, up to 50.
    pub tasks: Vec<TaskRow>,
    /// A delegating caller's conversation when the message carried a `chat_ctx` data part, as the
    /// part's `data`.
    pub caller_history: Option<Value>,
    /// The tool the message is handed to before the model runs: an `InputRequired(tool=)` of
    /// the last turn named it, and this message is its answer.
    pub answer_tool: Option<String>,
}

/// A conversation tool's pending call, the one an `InputRequired` came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn", config = "frozen=True")]
pub struct PendingCall {
    pub name: String,
    pub call_id: String,
    /// The tool that takes the answer, when the `InputRequired` named one.
    #[serde(default)]
    #[python(default = "None")]
    pub answer_tool: Option<String>,
}

/// A journal event the handler recorded (docs/journal.md): `tool_execution_updated` for a
/// call's start, its progress and its end.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn", config = "frozen=True")]
pub struct HostEvent {
    #[serde(rename = "type")]
    #[host(rename = "type")]
    pub r#type: String,
    pub payload: Value,
}

/// What the turn reads from the handler's results: the `RunResult` fields and the context's
/// metadata, as JSON. Every field has a default: a handler reports only what it has.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn")]
pub struct HandlerOutput {
    /// The answer: text, or a typed value as JSON.
    #[serde(default)]
    #[python(default = "None")]
    pub output: Option<Value>,
    /// The chat items the runs produced, in order, each with its `id` and `type`; empty when no
    /// run was made, in which case the turn stores the message and a text answer itself.
    #[serde(default)]
    #[python(ty = "list[dict[str, Any]]", import = "typing:Any", default_factory = "list")]
    pub new_items: Vec<ChatItem>,
    /// The state after the handler; written at commit when it differs from what was loaded.
    #[serde(default)]
    #[python(default = "None")]
    pub state: Option<Value>,
    #[serde(default)]
    #[python(default_factory = "RunUsage")]
    pub usage: RunUsage,
    /// A conversation tool's `InputRequired(question)`.
    #[serde(default)]
    #[python(default = "None")]
    pub input_required: Option<String>,
    /// The call that asked, when `input_required` is set.
    #[serde(default)]
    #[python(default = "None")]
    pub pending_call: Option<PendingCall>,
    /// The handler yielded at a checkpoint.
    #[serde(default)]
    #[python(default = "False")]
    pub interrupted: bool,
    /// What the handler put on the task's terminal row, under short keys.
    #[serde(default)]
    #[python(ty = "dict[str, Any]", import = "typing:Any", default_factory = "dict")]
    pub metadata: Map<String, Value>,
    /// The handler raised: `"<TypeName>: <text>"`; the task fails with it.
    #[serde(default)]
    #[python(default = "None")]
    pub error: Option<String>,
    /// Journal events the handler recorded, in order, written with the commit.
    #[serde(default)]
    #[python(default_factory = "list")]
    pub events: Vec<HostEvent>,
}

/// What `Turn::run` returns: the message's task after the turn, and the reply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn", config = "frozen=True")]
pub struct TurnResult {
    pub task: TaskRow,
    /// The answer, as the artifact holds it: text, or the data of a typed value.
    pub output: Option<Value>,
    /// The database id of the session the turn ran in.
    pub thread_id: String,
    /// The task that carries the reply: this one, or the later one it was merged into.
    pub answered_by: String,
    pub run_id: Option<String>,
}

/// What a turn leaves for `finish`: the task that carries the reply, and the reply.
struct Outcome {
    pub task_id: String,
    pub output: Option<Value>,
    #[allow(dead_code)]
    pub interrupted: bool,
}

impl Turn {
    /// One turn over a batch inside an open run; writes the task rows, commits nothing itself.
    async fn turn(
        &self,
        ctx: &Ctx,
        run: &mut Run,
        batch: &[Message],
        task_ids: &[String],
        interrupt: bool,
    ) -> Result<Outcome, StorageError> {
        let last = batch.last().ok_or_else(|| StorageError::Invalid("an empty batch".into()))?;
        let mut answer_task = task_ids
            .last()
            .ok_or_else(|| StorageError::Invalid("a batch without task ids".into()))?
            .clone();
        let lane = &ctx.lane;

        // what this turn adds goes after everything loaded, whatever the messages' own times
        let after = run.history.iter().map(ItemRow::created_at).fold(0.0, f64::max);
        let mut now = now_secs();
        if after >= now {
            now = after + 2e-6;
        }

        // a pending conversation tool's InputRequired: this message is the call's output, and
        // the tool the ask named, if any, takes it before the model runs
        let mut answer_tool = None;
        let pending = run.tasks.pending_ask().await?;
        let nonempty = |s: &Option<String>| s.clone().filter(|s| !s.is_empty());
        if let Some(pending) = pending
            && let Some(ask) = &pending.ask
            && let Some(call_id) = nonempty(&ask.call_id)
        {
            answer_tool = nonempty(&ask.answer_tool);
            let placeholder = run.history.iter_mut().rev().find_map(|i| match i.kind_mut() {
                Some(chat_item::Item::FunctionCallOutput(o)) if o.call_id == call_id => Some(o),
                _ => None,
            });
            match placeholder {
                Some(o) => {
                    // the run recorded a placeholder output when the tool asked; the answer
                    // replaces it
                    o.output = last.text();
                    o.created_at = Some(timestamp(now - 1e-6));
                }
                None => {
                    let tool = nonempty(&ask.tool).unwrap_or_default();
                    let item =
                        ItemRow::new(chat_item::Item::FunctionCallOutput(FunctionCallOutput {
                            id: shortuuid("item_"),
                            name: tool,
                            call_id,
                            output: last.text(),
                            is_error: false,
                            created_at: Some(timestamp(now - 1e-6)),
                        }));
                    insert_by_created_at(&mut run.history, item);
                }
            }
            if last.task_id.as_deref() == Some(pending.task_id.as_str()) {
                // the client continued the waiting task: it gets the answer, and the row this
                // message was received under is merged into it, so no working row outlives it
                answer_task = pending.task_id.clone();
                run.tasks
                    .write(
                        task_ids.last().expect("checked"),
                        TaskState::Completed,
                        TaskPatch {
                            message_id: Some(last.message_id.clone()),
                            history: Some(vec![last.to_a2a()]),
                            metadata: Some(merged_into(&pending.task_id)),
                            ..TaskPatch::default()
                        },
                    )
                    .await?;
            } else {
                run.tasks
                    .write(
                        &pending.task_id,
                        TaskState::Completed,
                        TaskPatch {
                            metadata: Some(merged_into(&answer_task)),
                            ..TaskPatch::default()
                        },
                    )
                    .await?;
            }
        }

        let user_message = batch_to_chat_message(batch, now);
        let turn_ctx = TurnContext {
            thread_id: self.thread.id().to_owned(),
            lane: lane.name().to_owned(),
            message: user_message.payload.clone(),
            batch: batch.to_vec(),
            history: run.history.clone(),
            state: run.state.clone(),
            read_only: run.lease().is_none(),
            task_id: answer_task.clone(),
            tasks: run.tasks.recent(TASKS_IN_CONTEXT).await?,
            caller_history: ctx.caller_history.clone(),
            answer_tool,
        };
        let (out, stop) = self.drive(&turn_ctx, run, interrupt).await?;
        if let Some(err) = &out.error {
            log::error!(target: "livekit.agents", lane = lane.name(), taskId = answer_task.as_str(); "the a2a_session handler raised; the task fails: {err}");
        }
        if out.state.is_some() {
            run.state = out.state.clone();
        }

        // what the runs added to the history; the message itself when no run was made
        if !out.new_items.is_empty() {
            for payload in &out.new_items {
                let kind = payload.item.clone().ok_or_else(|| {
                    StorageError::Invalid("the handler returned an empty chat item".into())
                })?;
                let user =
                    matches!(&kind, chat_item::Item::Message(m) if m.role() == ChatRole::User);
                let item = ItemRow::new(kind);
                let id = item.id.clone();
                if !run.history.iter().any(|i| i.id == id) {
                    insert_by_created_at(&mut run.history, item);
                }
                if user && let Some(stored) = run.history.iter_mut().find(|i| i.id == id) {
                    // the A2A `messageId` it came from: the idempotency key on `items`
                    stored.message_id = Some(last.message_id.clone());
                }
            }
        } else {
            let user = ItemRow { message_id: Some(last.message_id.clone()), ..user_message };
            insert_by_created_at(&mut run.history, user);
            if let Some(text) = out.output.as_ref().and_then(Value::as_str)
                && !text.is_empty()
                && out.error.is_none()
                && !out.interrupted
            {
                let created_at = now_secs().max(now + 1e-6);
                insert_by_created_at(
                    &mut run.history,
                    chat_message(ChatRole::Assistant, vec![text.to_owned()], created_at),
                );
            }
        }
        run.usage.requests += out.usage.requests;
        run.usage.input_tokens += out.usage.input_tokens;
        run.usage.output_tokens += out.usage.output_tokens;
        run.usage.tool_calls += out.usage.tool_calls;
        for event in &out.events {
            run.events.append(&event.r#type, event.payload.clone());
        }

        let interrupted = stop.is_some() || out.interrupted;
        let history: Vec<a2a::Message> = batch.iter().map(Message::to_a2a).collect();
        let outcome = |output: Option<Value>, interrupted: bool| Outcome {
            task_id: answer_task.clone(),
            output,
            interrupted,
        };
        if stop == Some(Stop::ClientCanceled)
            || (interrupted
                && lane
                    .task(&answer_task)
                    .await?
                    .is_some_and(|row| row.status == TaskState::Canceled))
        {
            // the row is the client's; the history keeps what ran, the run says why it ended
            run.outcome = RunOutcome::Canceled;
            run.reason = Some("client-cancel".into());
            return Ok(outcome(None, true));
        }
        for (tid, m) in task_ids[..task_ids.len() - 1].iter().zip(&batch[..batch.len() - 1]) {
            run.tasks
                .write(
                    tid,
                    TaskState::Completed,
                    TaskPatch {
                        message_id: Some(m.message_id.clone()),
                        history: Some(vec![m.to_a2a()]),
                        metadata: Some(merged_into(&answer_task)),
                        ..TaskPatch::default()
                    },
                )
                .await?;
        }
        if let Some(err) = &out.error {
            run.tasks
                .write(
                    &answer_task,
                    TaskState::Failed,
                    TaskPatch {
                        message_id: Some(last.message_id.clone()),
                        history: Some(history),
                        status_message: Some(agent_text(&answer_task, err)),
                        ..TaskPatch::default()
                    },
                )
                .await?;
            run.events.append("error", serde_json::json!({"error": err}));
            run.outcome = RunOutcome::Failed;
            run.reason = Some(err.clone());
            return Ok(outcome(None, false));
        }
        if interrupted {
            run.tasks
                .write(
                    &answer_task,
                    TaskState::Canceled,
                    TaskPatch {
                        message_id: Some(last.message_id.clone()),
                        history: Some(history),
                        metadata: Some(interrupted_metadata()),
                        ..TaskPatch::default()
                    },
                )
                .await?;
            run.outcome = RunOutcome::Canceled;
            run.reason = Some(INTERRUPTION_NEW_MESSAGE.into());
            return Ok(outcome(None, true));
        }
        let metadata = Some(out.metadata.clone()).filter(|m| !m.is_empty());
        if let Some(question) = &out.input_required {
            let call = out.pending_call.as_ref();
            run.tasks
                .wait(
                    &answer_task,
                    question,
                    WaitOptions {
                        tool: call.map(|c| c.name.clone()),
                        call_id: call.map(|c| c.call_id.clone()),
                        answer_tool: call.and_then(|c| c.answer_tool.clone()),
                        ..WaitOptions::default()
                    },
                )
                .await?;
            run.tasks
                .write(
                    &answer_task,
                    TaskState::InputRequired,
                    TaskPatch {
                        message_id: Some(last.message_id.clone()),
                        history: Some(history),
                        metadata,
                        ..TaskPatch::default()
                    },
                )
                .await?;
            run.outcome = RunOutcome::InputRequired;
            return Ok(outcome(out.output.clone(), false));
        }
        let output = out.output.clone().filter(|o| !o.is_null());
        let text = output.as_ref().and_then(Value::as_str).filter(|t| !t.is_empty());
        let artifacts: Vec<a2a::Artifact> = output.iter().map(artifact).collect();
        run.tasks
            .write(
                &answer_task,
                TaskState::Completed,
                TaskPatch {
                    message_id: Some(last.message_id.clone()),
                    history: Some(history),
                    artifacts: Some(artifacts),
                    status_message: text.map(|t| agent_text(&answer_task, t)),
                    metadata,
                    ..TaskPatch::default()
                },
            )
            .await?;
        Ok(outcome(output, false))
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
        let ctx = serde_json::to_value(ctx).map_err(|e| StorageError::Invalid(e.to_string()))?;
        let (verdicts, mut events) = self.host.handle(ctx).map_err(host_error)?;
        let lane = self.thread.lane(&self.lane);
        let mut stop = None;
        let read = async {
            loop {
                match events.next().await.map_err(host_error)? {
                    Some(TurnEvent::Checkpoint(_)) => {
                        // a lost lease cannot commit anyway: nothing yields to anyone
                        if stop.is_none() && !run.lease().is_some_and(Lease::is_lost) {
                            let (waiter, status) = lane.db().checkpoint(task_id).await?;
                            if interrupt && waiter.is_some_and(|w| !w.is_empty() && w != run.holder)
                            {
                                log::info!(target: "livekit.agents", lane = self.lane.as_str(); "yielding the lane to a newer message");
                                stop = Some(Stop::Interrupted);
                            } else if status == Some(TaskState::Canceled) {
                                log::info!(target: "livekit.agents", lane = self.lane.as_str(); "the task was canceled by a client; yielding");
                                stop = Some(Stop::ClientCanceled);
                            }
                        }
                        let verdict =
                            if stop.is_some() { Verdict::Yield } else { Verdict::Continue };
                        verdicts.push(verdict).map_err(host_error)?;
                    }
                    Some(TurnEvent::Output(o)) => {
                        let out = serde_json::from_value(o.output)
                            .map_err(|e| StorageError::Decode(format!("handler output: {e}")))?;
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
        events.close(self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
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
        let record = committed
            .task
            .unwrap_or_else(|| unpersisted_failed(task_id, run.lane(), None, "not-written"));
        TurnResult {
            task: record,
            output: outcome.output,
            thread_id: self.thread.id().to_owned(),
            answered_by: outcome.task_id,
            run_id: Some(run.run_id.clone()),
        }
    }
}

fn merged_into(task_id: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert(MERGED_INTO_KEY.into(), Value::String(task_id.to_owned()));
    m
}

/// An agent message of `task_id` with one text part.
pub fn agent_text(task_id: &str, text: &str) -> a2a::Message {
    agent_message(vec![a2a::Part::text(text)], task_id)
}

/// The status message of a row the worker failed without a run, with `livekit/reason`.
pub fn failed_message(task_id: &str, reason: &str) -> a2a::Message {
    let mut msg = agent_text(task_id, &format!("the task failed: {reason}"));
    msg.metadata = Some([(REASON_KEY.to_owned(), Value::String(reason.to_owned()))].into());
    msg
}

/// The answer as an artifact: a text part for a string, a data part otherwise.
fn artifact(value: &Value) -> a2a::Artifact {
    let part = match value.as_str() {
        Some(text) => a2a::Part::text(text),
        None => a2a::Part::data(value.clone()),
    };
    a2a::Artifact {
        artifact_id: shortuuid("art_"),
        name: None,
        description: None,
        parts: vec![part],
        metadata: None,
        extensions: None,
    }
}

/// The first artifact's text, or its data.
pub fn artifact_text(record: &TaskRow) -> Option<Value> {
    let first = record.data.artifacts.as_ref()?.first()?;
    first.parts.iter().find_map(|p| match &p.content {
        a2a::PartContent::Text(t) => Some(Value::String(t.clone())),
        a2a::PartContent::Data(d) => Some(d.clone()),
        _ => None,
    })
}

/// The channel of the message that started the task.
pub fn channel_of_record(record: &TaskRow) -> Option<String> {
    record.data.history.first().map(channel_of)
}

/// The result of a task that was not run here, answered by the task it was merged into when it
/// was.
pub fn result_for(
    record: Option<TaskRow>,
    thread_id: &str,
    task_id: &str,
    run_id: Option<&str>,
) -> TurnResult {
    let record = record.unwrap_or_else(|| unpersisted_failed(task_id, "", None, "not-written"));
    let answered_by = record
        .data
        .metadata
        .get(MERGED_INTO_KEY)
        .and_then(Value::as_str)
        .unwrap_or(&record.task_id)
        .to_owned();
    TurnResult {
        output: artifact_text(&record),
        answered_by,
        task: record,
        thread_id: thread_id.to_owned(),
        run_id: run_id.map(str::to_owned),
    }
}

/// A failed record the worker could not write: it held no lease (`lease-timeout`).
pub fn unpersisted_failed(
    task_id: &str,
    lane: &str,
    msg: Option<&Message>,
    reason: &str,
) -> TaskRow {
    let mut metadata = Map::new();
    metadata.insert(REASON_KEY.into(), Value::String(reason.to_owned()));
    TaskRow {
        task_id: task_id.to_owned(),
        lane: lane.to_owned(),
        seq: 0,
        name: None,
        message_id: msg.map(|m| m.message_id.clone()),
        status: TaskState::Failed,
        updated_at: now_ms(),
        data: TaskData {
            history: msg.map(|m| vec![m.to_a2a()]).unwrap_or_default(),
            artifacts: None,
            metadata,
            status_message: Some(failed_message(task_id, reason)),
        },
        ask: None,
    }
}

/// `ChatContext.insert`: after the last item whose `created_at` is not later.
pub fn insert_by_created_at(history: &mut Vec<ItemRow>, item: ItemRow) {
    let created_at = item.created_at();
    let idx = history.iter().rposition(|i| i.created_at() <= created_at).map_or(0, |i| i + 1);
    history.insert(idx, item);
}

/// A failed task's reason for a storage error: the exception name the host raises it as, and its
/// message.
pub fn python_reason(e: &StorageError) -> String {
    match e {
        StorageError::ThreadNotFound(m) => format!("ThreadNotFound: {m}"),
        StorageError::VersionConflict { .. } => format!("VersionConflict: {e}"),
        StorageError::NotSerializable(..) => format!("NotSerializable: {e}"),
        StorageError::TooLarge { .. } => format!("TooLarge: {e}"),
        StorageError::Unavailable(m) => format!("StorageUnavailable: {m}"),
        StorageError::Invalid(m) => format!("ValueError: {m}"),
        StorageError::Cancelled(m) => {
            if m.is_empty() {
                "CancelledError".into()
            } else {
                format!("CancelledError: {m}")
            }
        }
        other => format!("StorageError: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn options_default_to_the_documented_numbers() {
        let o = TurnOptions::default();
        assert_eq!(o.wait_timeout(), LEASE_WAIT_TIMEOUT);
        assert_eq!(o.policy, PolicySpec::Default);
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
        let json = r#"{"policy": {"sms": "batch"}, "debounce": 2.5, "wait_timeout": 1}"#;
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
            serde_json::json!({"policy": null, "debounce": null, "wait_timeout": null})
        );
        let e = serde_json::from_str::<TurnOptions>(r#"{"policy": "later"}"#).unwrap_err();
        assert!(e.to_string().contains("later"), "{e}");
        assert!(serde_json::from_str::<TurnOptions>(r#"{"debounce": -1}"#).is_err());
    }

    #[test]
    fn policy_and_debounce_resolution() {
        let none = PolicySpec::Default;
        assert_eq!(resolve_policy(&none, "sms"), Policy::Batch);
        assert_eq!(resolve_policy(&none, "a2a"), Policy::Interrupt);
        assert_eq!(resolve_policy(&none, "delegation"), Policy::Queue);
        assert_eq!(resolve_policy(&none, "carrier-pigeon"), Policy::Interrupt);
        let map = PolicySpec::ByChannel([("a2a".to_owned(), Policy::Queue)].into_iter().collect());
        assert_eq!(resolve_policy(&map, "a2a"), Policy::Queue);
        assert_eq!(resolve_policy(&map, "sms"), Policy::Batch);
        assert_eq!(resolve_policy(&PolicySpec::All(Policy::Reject), "sms"), Policy::Reject);
        assert_eq!(Policy::parse("later"), None);
        assert_eq!(Policy::parse("batch"), Some(Policy::Batch));
        let d = DebounceSpec::Default;
        assert_eq!(resolve_debounce(&d, "sms", Policy::Batch), Duration::from_secs(2));
        assert_eq!(resolve_debounce(&d, "a2a", Policy::Interrupt), Duration::ZERO);
        assert_eq!(resolve_debounce(&d, "pigeon", Policy::Batch), Duration::from_secs(2));
        assert_eq!(resolve_debounce(&d, "pigeon", Policy::Queue), Duration::ZERO);
        let one = DebounceSpec::All(Duration::from_secs(1));
        assert_eq!(resolve_debounce(&one, "a2a", Policy::Interrupt), Duration::from_secs(1));
        let per = DebounceSpec::ByChannel(
            [("sms".to_owned(), Duration::from_secs(5))].into_iter().collect(),
        );
        assert_eq!(resolve_debounce(&per, "sms", Policy::Batch), Duration::from_secs(5));
        assert_eq!(resolve_debounce(&per, "email", Policy::Batch), Duration::from_secs(2));
    }

    fn v1(message: Value) -> a2a::Message {
        serde_json::from_value(message).unwrap()
    }

    #[test]
    fn a_message_reads_its_channel_text_data_and_round_trips() {
        let raw = v1(json!({
            "messageId": "m1",
            "role": "ROLE_USER",
            "parts": [
                {"text": "hi"},
                {"data": {"b": 1, "a": 2}},
                {"url": "https://x/y.png", "mediaType": "image/png"},
                {"text": "there"},
            ],
            "metadata": {"livekit/channel": "sms"},
            "taskId": "AST_1",
            "contextId": "AT_1",
        }));
        let m = Message::from_a2a(&raw, Some(5.0), None).unwrap();
        assert_eq!((m.message_id.as_str(), m.channel.as_str()), ("m1", "sms"));
        assert_eq!((m.task_id.as_deref(), m.context_id.as_deref()), (Some("AST_1"), Some("AT_1")));
        assert_eq!(m.text(), "hi\nthere");
        assert_eq!(m.data(), json!({"b": 1, "a": 2}).as_object().cloned());
        assert_eq!(
            m.content(),
            vec![
                "hi",
                r#"{"b":1,"a":2}"#,
                r#"{"url":"https://x/y.png","mediaType":"image/png"}"#,
                "there"
            ]
        );
        assert_eq!(m.to_a2a(), raw, "the message comes back as it was");
        assert_eq!(
            Message::from_a2a(&raw, None, Some("AST_9")).unwrap().task_id.as_deref(),
            Some("AST_9")
        );

        let bare = v1(json!({"messageId": "m2", "role": "ROLE_UNSPECIFIED", "parts": []}));
        assert_eq!(Message::from_a2a(&bare, None, None).unwrap().to_a2a().role, a2a::Role::User);
        assert!(
            Message::from_a2a(
                &v1(json!({"messageId": "", "role": "ROLE_USER", "parts": []})),
                None,
                None
            )
            .is_err()
        );
        let delegated = v1(json!({"messageId": "x", "role": "ROLE_USER", "parts": [],
            "metadata": {"livekit/kind": "delegation"}}));
        assert_eq!(Message::from_a2a(&delegated, None, None).unwrap().channel, "delegation");
        let with_history = v1(json!({"messageId": "y", "role": "ROLE_USER", "parts": [
            {"text": "fare?"},
            {"data": {"items": []}, "metadata": {"livekit/kind": "chat_ctx"}},
        ]}));
        assert_eq!(caller_history_of(&with_history), Some(json!({"items": []})));
        let stripped = Message::from_a2a(&with_history, Some(1.0), None).unwrap();
        assert_eq!(stripped.parts, vec![a2a::Part::text("fare?")]);
        assert_eq!(stripped.to_a2a().parts, vec![a2a::Part::text("fare?")]);
    }

    #[test]
    fn the_batch_is_one_user_message_with_every_part_in_order() {
        let text = |id: &str, t: &str| {
            v1(json!({"messageId": id, "role": "ROLE_USER", "parts": [{"text": t}]}))
        };
        let a = Message::from_a2a(&text("m1", "where"), Some(1.0), None).unwrap();
        let b = Message::from_a2a(&text("m2", "are you"), Some(2.0), None).unwrap();
        let item = batch_to_chat_message(&[a, b], 2.0);
        assert!(item.id.starts_with("item_") && item.id.len() == 17);
        let Some(chat_item::Item::Message(m)) = item.kind() else { panic!("{item:?}") };
        assert_eq!(m.role(), ChatRole::User);
        let texts = |m: &ChatMessage| -> Vec<String> {
            m.content.iter().filter_map(|c| c.payload.clone().map(|Payload::Text(t)| t)).collect()
        };
        assert_eq!(texts(m), vec!["where", "are you"]);
        assert_eq!(item.created_at(), 2.0);
        let empty = batch_to_chat_message(&[], 0.0);
        let Some(chat_item::Item::Message(m)) = empty.kind() else { panic!() };
        assert_eq!(texts(m), vec![""]);
    }

    fn item(id: &str, created_at: f64) -> ItemRow {
        ItemRow { id: id.into(), ..chat_message(ChatRole::User, vec![], created_at) }
    }

    #[test]
    fn insertion_follows_created_at_after_equal_times() {
        let mut h = vec![item("a", 1.0), item("b", 2.0), item("c", 2.0)];
        insert_by_created_at(&mut h, item("d", 2.0));
        insert_by_created_at(&mut h, item("e", 0.5));
        insert_by_created_at(&mut h, item("f", 9.0));
        assert_eq!(
            h.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(),
            vec!["e", "a", "b", "c", "d", "f"]
        );
    }

    #[test]
    fn the_helpers_shape_as_the_python() {
        let f =
            a2a::Message { message_id: "msg_X".into(), ..failed_message("AST_1", "worker-lost") };
        assert_eq!(
            serde_json::to_value(&f).unwrap(),
            json!({"messageId": "msg_X", "taskId": "AST_1", "role": "ROLE_AGENT",
                "parts": [{"text": "the task failed: worker-lost"}],
                "metadata": {"livekit/reason": "worker-lost"}})
        );
        let failed = unpersisted_failed("AST_2", "front-desk", None, "lease-timeout");
        assert_eq!(failed.data.metadata.get(REASON_KEY), Some(&json!("lease-timeout")));
        assert_eq!(failed.status, TaskState::Failed);
        assert!(failed.data.history.is_empty());
        assert_eq!(artifact_text(&failed), None);
        let mut done = failed.clone();
        done.data.artifacts = Some(vec![artifact(&json!({"n": 1}))]);
        assert_eq!(artifact_text(&done), Some(json!({"n": 1})));
        done.data.history = vec![v1(json!({"messageId": "m", "role": "ROLE_USER", "parts": [],
            "metadata": {"livekit/channel": "sms"}}))];
        assert_eq!(channel_of_record(&done).as_deref(), Some("sms"));
        let r = result_for(Some(done.clone()), "DB_1", "AST_2", Some("r1"));
        assert_eq!((r.answered_by.as_str(), r.run_id.as_deref()), ("AST_2", Some("r1")));
        done.data.metadata.insert(MERGED_INTO_KEY.into(), json!("AST_9"));
        assert_eq!(result_for(Some(done), "DB_1", "AST_2", None).answered_by, "AST_9");
        assert_eq!(result_for(None, "DB_1", "AST_3", None).task.status, TaskState::Failed);
        assert_eq!(
            python_reason(&StorageError::Unavailable("lost".into())),
            "StorageUnavailable: lost"
        );
        assert_eq!(python_reason(&StorageError::Cancelled(String::new())), "CancelledError");
    }

    // Catches: a cancelled handler written as a storage failure, or a storage error the handler
    // met losing its kind.
    #[test]
    fn host_failures_map_to_storage_errors() {
        let host = |kind: &str| ProviderError::Host {
            error: lk_agents_abi::errors::HostError {
                kind: kind.into(),
                message: "m".into(),
                host_ref: None,
            },
            api_kind: None,
            retryable: false,
            status_code: None,
        };
        assert!(
            matches!(host_error(host("CancelledError")), StorageError::Cancelled(m) if m == "m")
        );
        assert!(matches!(host_error(host("StorageUnavailable")), StorageError::Unavailable(_)));
        assert!(matches!(host_error(host("ThreadNotFound")), StorageError::ThreadNotFound(_)));
        assert!(matches!(
            host_error(host("VersionConflict")),
            StorageError::Server { code, .. } if code == "VersionConflict"
        ));
    }
}
