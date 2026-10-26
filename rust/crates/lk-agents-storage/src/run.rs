//! The run: the unit of work a lane commits. [`Thread`] is one thread's database, [`Lane`] one
//! agent's part of it, and [`Lane::open`] takes it for one [`Run`], which ends with
//! [`Run::commit`] (one fenced write) or [`Run::abandon`] (`run_ended` and the release).
//! `exclusive: false` is the `concurrent` policy: no lease, append-only.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::db::{
    Acquire, Commit, Committed, EventWrite, LaneDb, Lease, Loaded, TaskPage, TaskQuery, TaskWrite,
    ThreadDb,
};
use crate::schema::{Ask, EventRow, RunEnded, RunEvent, RunStarted};
pub use crate::schema::{ItemRow, LaneRow, TaskRow};
pub use crate::timestamp;
use crate::{StorageError, shortuuid};
use livekit_protocol::agent::chat_context::ChatItem;

/// a waiter that has not acquired after 60 s fails `lease-timeout`.
pub const LEASE_WAIT_TIMEOUT: Duration = Duration::from_secs(60);

/// Waiting for a lane another holder has: a pause from 100 ms, doubling up to 1 s, until a
/// deadline.
pub struct LanePoll {
    deadline: Instant,
    pause: Duration,
}

impl LanePoll {
    pub fn new(timeout: Duration) -> Self {
        Self { deadline: Instant::now() + timeout, pause: Duration::from_millis(100) }
    }

    /// Pauses before the next attempt; `false`, at once, once the deadline passed.
    pub async fn pause(&mut self) -> bool {
        if Instant::now() >= self.deadline {
            return false;
        }
        tokio::time::sleep(self.pause).await;
        self.pause = (self.pause * 2).min(Duration::from_secs(1));
        true
    }
}

/// What this run appends to the lane's journal.
#[derive(Default)]
pub struct EventLog {
    new: Vec<EventWrite>,
}

impl EventLog {
    /// Appends an event, stamped now.
    pub fn append(&mut self, r#type: &str, payload: Value) {
        self.new.push(EventWrite::new(r#type, payload));
    }
}

/// Token usage of one run; written as a `usage` event when the run
/// made a request or a tool call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_run")]
pub struct RunUsage {
    #[python(default = "0")]
    pub requests: i64,
    #[python(default = "0")]
    pub input_tokens: i64,
    #[python(default = "0")]
    pub output_tokens: i64,
    #[python(default = "0")]
    pub tool_calls: i64,
}

/// How [`Lane::open`] takes the lane.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Idempotency key: a `run_id` that already committed yields a closed run.
    pub run_id: String,
    /// The A2A task this run serves; also its answer stream's id.
    pub task_id: String,
    /// `false` is the `concurrent` policy: no lease, append-only commit.
    pub exclusive: bool,
    /// Wait for a lease another holder has; `false` fails with [`StorageError::LeaseHeld`].
    pub wait: bool,
    /// Record this worker as the lane's `waiter` while it waits, so the holder yields.
    pub interrupt: bool,
    /// The request's channel, for the `run_started` event.
    pub channel: Option<String>,
    pub wait_timeout: Duration,
    /// The turn's `submitted` row, written in the acquire batch (`Run::submitted` tells whether
    /// it was new); `None` for a run with no row or one submitted earlier.
    pub submit: Option<TaskWrite>,
    /// Take `task_id`'s row to `working` in the acquire batch (`Run::picked_up`).
    pub pickup: bool,
}

impl RunOptions {
    pub fn new(run_id: impl Into<String>, task_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            task_id: task_id.into(),
            exclusive: true,
            wait: true,
            interrupt: false,
            channel: None,
            wait_timeout: LEASE_WAIT_TIMEOUT,
            submit: None,
            pickup: false,
        }
    }
}

/// The thread's database, shared by every lane. Cheap to clone: one
/// connection behind every clone.
#[derive(Clone)]
pub struct Thread {
    db: ThreadDb,
    holder: Arc<str>,
}

impl Thread {
    /// Over an open database. Leases carry this handle as their holder, so a second request of the
    /// same worker on a lane waits like any other worker's.
    pub fn new(db: ThreadDb, holder: &str) -> Self {
        Self { db, holder: format!("{holder}/{}", shortuuid("")).into() }
    }

    /// The database id.
    pub fn id(&self) -> &str {
        self.db.database_id()
    }

    /// `<worker holder>/<handle>`: what this handle's leases and waiter marks carry.
    pub fn holder(&self) -> &str {
        &self.holder
    }

    /// The durable commit sequence last observed on the connection.
    pub fn tip(&self) -> i64 {
        self.db.tip()
    }

    pub fn lane(&self, agent: &str) -> Lane {
        Lane { holder: self.holder.clone(), db: self.db.lane(agent) }
    }

    /// The journal, every lane, after the event `after_id`.
    pub async fn events(&self, after_id: i64, limit: i64) -> Result<Vec<EventRow>, StorageError> {
        self.db.events(after_id, limit).await
    }

    /// One task, whichever lane it is on.
    pub async fn task(&self, task_id: &str) -> Result<Option<TaskRow>, StorageError> {
        self.db.task(task_id).await
    }

    /// One page of every lane's tasks, last updated first (A2A `tasks/list`).
    pub async fn list_tasks(&self, q: &TaskQuery) -> Result<TaskPage, StorageError> {
        self.db.list_tasks(q).await
    }

    pub async fn lanes(&self) -> Result<Vec<LaneRow>, StorageError> {
        self.db.lanes().await
    }

    pub fn close(&self) {
        self.db.close();
    }
}

/// One registered agent's part of a thread.
#[derive(Clone)]
pub struct Lane {
    holder: Arc<str>,
    db: LaneDb,
}

impl Lane {
    pub fn name(&self) -> &str {
        self.db.name()
    }

    pub fn holder(&self) -> &str {
        &self.holder
    }

    pub fn db(&self) -> &LaneDb {
        &self.db
    }

    /// The whole history; how another lane reads it.
    pub async fn history(&self) -> Result<Vec<ItemRow>, StorageError> {
        self.db.history().await
    }

    pub async fn info(&self) -> Result<Option<LaneRow>, StorageError> {
        self.db.info().await
    }

    pub async fn task(&self, task_id: &str) -> Result<Option<TaskRow>, StorageError> {
        self.db.task(task_id).await
    }

    /// The newest task rows, of one state when given.
    pub async fn tasks(
        &self,
        status: Option<TaskState>,
        limit: i64,
    ) -> Result<Vec<TaskRow>, StorageError> {
        self.db.tasks(status, limit).await
    }

    /// The last `limit` runs, newest first; the `submitted` insert before a lease is
    /// bookkeeping, not a turn.
    pub async fn runs(&self, limit: usize) -> Result<Vec<RunSummary>, StorageError> {
        let rows = self.db.events(0).await?;
        let mut runs: Vec<RunSummary> = RunSummary::from_events(&rows)
            .into_iter()
            .filter(|r| r.channel.as_deref() != Some(SUBMIT_CHANNEL))
            .collect();
        let keep = runs.len().saturating_sub(limit);
        runs.drain(..keep);
        runs.reverse();
        Ok(runs)
    }

    /// `tasks/cancel`: the row `canceled` with `metadata` merged in; a running turn yields at its
    /// next checkpoint. `None` when the task is not the lane's.
    pub async fn cancel_task(
        &self,
        task_id: &str,
        metadata: Map<String, Value>,
    ) -> Result<Option<TaskRow>, StorageError> {
        let Some(row) = self.db.task(task_id).await? else { return Ok(None) };
        let mut data = row.data;
        data.metadata.extend(metadata);
        let run_id = format!("cancel_{task_id}_{}", shortuuid(""));
        let commit = Commit {
            run_id,
            tasks: vec![TaskWrite {
                task_id: row.task_id,
                status: TaskState::Canceled,
                message_id: row.message_id,
                name: row.name,
                data,
                ask: row.ask,
            }],
            events: vec![EventWrite::run_started(&RunStarted {
                task_id: Some(task_id.to_owned()),
                channel: Some("cancel".into()),
                ..RunStarted::default()
            })],
            run_ended: EventWrite::run_ended(&RunEnded {
                outcome: RunOutcome::Completed,
                reason: None,
            }),
            read_task: Some(task_id.to_owned()),
            ..Commit::default()
        };
        Ok(self.db.append(&commit).await?.task)
    }

    /// Takes the lane and loads it. The run commits with [`Run::commit`] and ends by
    /// error with [`Run::abandon`]; a run dropped without either stops renewing and lets its lease
    /// lapse, which the next acquire sees as a `run_started` without a `run_ended`.
    pub async fn open(&self, opts: RunOptions) -> Result<Run, StorageError> {
        let loaded = match opts.exclusive {
            true => Some(self.acquire(&opts).await?),
            false => None,
        };
        let (lease, closed, lane_row, items, tasks, submitted, picked_up) = match loaded {
            Some(l) => {
                if !l.already_committed {
                    l.lease.keep_alive();
                }
                let tasks = Tasks::loaded(self.db.clone(), l.tasks);
                let lane = Some(l.lane);
                (Some(l.lease), l.already_committed, lane, l.items, tasks, l.submitted, l.picked_up)
            }
            None => {
                let items = self.db.history().await?;
                (None, false, None, items, Tasks::new(self.db.clone()), None, false)
            }
        };
        let state = lane_row.as_ref().and_then(|l| l.state.clone());
        Ok(Run {
            db: self.db.clone(),
            lease,
            holder: self.holder.to_string(),
            run_id: opts.run_id,
            task_id: opts.task_id,
            channel: opts.channel,
            closed,
            loaded: items.iter().map(|i| (i.id.clone(), i.payload.clone())).collect(),
            history: items,
            loaded_state: state.clone(),
            state,
            events: EventLog::default(),
            tasks,
            submitted,
            picked_up,
            usage: RunUsage::default(),
            outcome: RunOutcome::Completed,
            reason: None,
            finished: false,
        })
    }

    /// The acquire, polled with backoff while another holder has the lane, as the lane's `waiter`
    /// under `interrupt`.
    async fn acquire(&self, opts: &RunOptions) -> Result<Loaded, StorageError> {
        let mut request = Acquire {
            holder: &self.holder,
            run_id: &opts.run_id,
            task_id: &opts.task_id,
            run_started: RunStarted {
                task_id: Some(opts.task_id.clone()),
                stream_id: Some(opts.task_id.clone()),
                channel: opts.channel.clone(),
                exclusive: None,
            },
            submit: opts.submit.as_ref(),
            pickup: opts.pickup,
            tasks_limit: LOOKUP_LIMIT,
        };
        let mut poll = LanePoll::new(opts.wait_timeout);
        let mut marked = false;
        loop {
            if let Some(loaded) = self.db.acquire(&request).await? {
                return Ok(loaded);
            }
            // the row is in; a retry of the batch must not report it as a duplicate
            request.submit = None;
            if !opts.wait {
                let name = self.name();
                return Err(StorageError::LeaseHeld(format!("lane {name} is held by another run")));
            }
            if opts.interrupt && !marked {
                marked = self.db.set_waiter(&self.holder).await?;
                if marked {
                    log::debug!(target: "livekit.agents", lane = self.name(), holder = &*self.holder; "waiting for the lane as its waiter");
                }
            }
            if !poll.pause().await {
                return Err(StorageError::LeaseTimeout(format!(
                    "lane {} was not released within {}s",
                    self.name(),
                    opts.wait_timeout.as_secs_f64()
                )));
            }
        }
    }
}

/// What [`Run::commit`] does besides writing the run.
#[derive(Debug, Clone, Default)]
pub struct CommitOptions {
    /// Extend the lease instead of releasing it: the runner continues with pending inbox rows.
    pub keep_lease: bool,
    /// The last inbox row a drain consumed.
    pub drained_id: Option<i64>,
    /// The task the turn answers, read back with the commit.
    pub answer_task: Option<String>,
}

/// Everything a run reads and writes, committed together.
pub struct Run {
    db: LaneDb,
    /// The lease the run holds; `None` under the `concurrent` policy, which takes none.
    lease: Option<Lease>,
    pub holder: String,
    pub run_id: String,
    pub task_id: String,
    pub channel: Option<String>,
    /// A `run_id` that already committed: the run writes nothing and answers nothing twice.
    pub closed: bool,
    /// The whole history, loaded; append as the model and the tools speak, delete by removing.
    pub history: Vec<ItemRow>,
    /// What was loaded, by id, in order.
    loaded: Vec<(String, ChatItem)>,
    /// The state snapshot as loaded, `None` for a fresh lane; the caller can replace it
    /// explicitly, and the commit writes it when it differs from what was loaded.
    pub state: Option<Value>,
    loaded_state: Option<Value>,
    pub events: EventLog,
    pub tasks: Tasks,
    /// Whether the acquire's `submit` inserted its row; `Some(false)` is a retried `messageId`.
    pub submitted: Option<bool>,
    /// Whether the acquire's `pickup` took the row to `working`.
    pub picked_up: bool,
    pub usage: RunUsage,
    pub outcome: RunOutcome,
    pub reason: Option<String>,
    finished: bool,
}

impl Run {
    pub fn lane(&self) -> &str {
        self.db.name()
    }

    pub fn lease(&self) -> Option<&Lease> {
        self.lease.as_ref()
    }

    /// The version loaded: the fence of the commit; `0` for a run that took no lease.
    pub fn version(&self) -> i64 {
        self.lease.as_ref().map_or(0, Lease::version)
    }

    /// The run's writes: the changed and deleted items, their journal entries, the task rows,
    /// usage, the snapshots that changed, and `run_ended`.
    pub fn writes(&self, opts: CommitOptions) -> Commit {
        let exclusive = self.lease.is_some();
        let (items, delete_items) = item_changes(&self.history, &self.loaded);
        let mut events = Vec::new();
        if !exclusive {
            events.push(EventWrite::run_started(&RunStarted {
                task_id: Some(self.task_id.clone()),
                stream_id: Some(self.task_id.clone()),
                channel: self.channel.clone(),
                exclusive: Some(false),
            }));
        }
        events.extend(
            items
                .iter()
                .filter(|i| !self.loaded.iter().any(|(id, _)| *id == i.id))
                .map(|i| EventWrite::new("conversation_item_added", json!({"item": &i.payload}))),
        );
        events.extend(self.events.new.iter().cloned());
        if self.usage.requests != 0 || self.usage.tool_calls != 0 {
            let usage = serde_json::to_value(self.usage).unwrap_or(Value::Null);
            events.push(EventWrite::new("usage", usage));
        }
        Commit {
            run_id: self.run_id.clone(),
            delete_items,
            items,
            events,
            tasks: self.tasks.writes().to_vec(),
            state: self
                .state
                .clone()
                .filter(|s| exclusive && Some(s) != self.loaded_state.as_ref()),
            drained_id: opts.drained_id.filter(|_| exclusive),
            run_ended: EventWrite::run_ended(&RunEnded {
                outcome: self.outcome,
                reason: self.reason.clone(),
            }),
            read_task: opts.answer_task,
        }
    }

    fn finish(&mut self) -> Result<(), StorageError> {
        if self.finished {
            return Err(StorageError::Invalid(format!("run {} already ended", self.run_id)));
        }
        self.finished = true;
        if let Some(lease) = &self.lease {
            lease.stop();
        }
        Ok(())
    }

    /// Leaving normally: one fenced write of everything the run holds. A closed run
    /// releases the lease and writes nothing. `VersionConflict` means the run outlived its lease.
    pub async fn commit(&mut self, opts: CommitOptions) -> Result<Committed, StorageError> {
        self.finish()?;
        let version = self.version();
        match &self.lease {
            Some(lease) if self.closed => {
                lease.release(None).await?;
                Ok(Committed { version, task: None })
            }
            Some(lease) => lease.commit(&self.writes(opts.clone()), opts.keep_lease).await,
            None => self.db.append(&self.writes(opts)).await,
        }
    }

    /// Leaving by error: `run_ended` with the outcome and the reason, and the release; nothing else.
    /// A `run_ended` that could not be written is logged.
    pub async fn abandon(
        &mut self,
        outcome: RunOutcome,
        reason: impl Into<String>,
    ) -> Result<(), StorageError> {
        self.finish()?;
        let Some(lease) = &self.lease else { return Ok(()) };
        if self.closed {
            return lease.release(None).await;
        }
        let ended = EventWrite::run_ended(&RunEnded { outcome, reason: Some(reason.into()) });
        if let Err(e) = lease.release(Some((&self.run_id, &ended))).await {
            log::warn!(target: "livekit.agents", lane = self.db.name(), runId = self.run_id.as_str(); "run_ended could not be written after a failed run: {e}");
        }
        Ok(())
    }
}

/// The stored items that differ from what was `loaded`, and the loaded ids no longer in the
/// history, in loaded order.
fn item_changes(history: &[ItemRow], loaded: &[(String, ChatItem)]) -> (Vec<ItemRow>, Vec<String>) {
    let by_id: HashMap<&str, &ChatItem> = loaded.iter().map(|(id, i)| (id.as_str(), i)).collect();
    let stored: Vec<&ItemRow> = history.iter().filter(|i| i.stored()).collect();
    let changed = stored
        .iter()
        .filter(|i| by_id.get(i.id.as_str()) != Some(&&i.payload))
        .map(|i| (*i).clone())
        .collect();
    let kept: std::collections::HashSet<&str> = stored.iter().map(|i| i.id.as_str()).collect();
    let deleted =
        loaded.iter().filter(|(id, _)| !kept.contains(id.as_str())).map(|(id, _)| id.clone());
    (changed, deleted.collect())
}

/// The Live extension's metadata keys, written `livekit/<name>`.
pub const INTERRUPTED_KEY: &str = "livekit/interrupted";
pub const INTERRUPTION_REASON_KEY: &str = "livekit/interruptionReason";
pub const MERGED_INTO_KEY: &str = "livekit/mergedInto";
pub const CHANNEL_KEY: &str = "livekit/channel";
pub const KIND_KEY: &str = "livekit/kind";
pub const KEY_KEY: &str = "livekit/key";
/// The key `Tasks::fail` writes into the status message's metadata.
pub const REASON_KEY: &str = "livekit/reason";

/// The `failed` and `rejected` reasons.
pub const REASON_WORKER_LOST: &str = "worker-lost";
pub const REASON_LEASE_TIMEOUT: &str = "lease-timeout";
pub const REASON_REJECTED: &str = "rejected";
pub const REASON_THREAD_NOT_FOUND: &str = "thread-not-found";
pub const REASON_STORAGE_UNAVAILABLE: &str = "storage-unavailable";
pub const REASON_TIMEOUT: &str = "timeout";
pub const REASON_UNKNOWN_CLASS: &str = "unknown-class";
pub const REASON_INVALID_ARGUMENTS: &str = "invalid-arguments";

/// The channel of the append-only run that writes a task's `submitted` row before the lease.
pub const SUBMIT_CHANNEL: &str = "submit";

/// The framework's task states; the terminal ones are immutable. The A2A edge maps them to its
/// own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Host)]
#[serde(rename_all = "kebab-case")]
#[python(literal, module = "storage_run")]
pub enum TaskState {
    #[host(rename = "submitted")]
    Submitted,
    #[host(rename = "working")]
    Working,
    #[host(rename = "input-required")]
    InputRequired,
    #[host(rename = "completed")]
    Completed,
    #[host(rename = "failed")]
    Failed,
    #[host(rename = "canceled")]
    Canceled,
    #[host(rename = "rejected")]
    Rejected,
}

impl TaskState {
    /// `None` for a status no state spells (the derive's `from_wire`).
    pub fn parse(s: &str) -> Option<TaskState> {
        TaskState::from_wire(s)
    }

    /// Completed, failed, canceled or rejected: immutable once written.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TaskState::Completed | TaskState::Failed | TaskState::Canceled | TaskState::Rejected
        )
    }
}

/// What `run_ended` records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Host)]
#[serde(rename_all = "kebab-case")]
#[python(literal, module = "storage_run")]
pub enum RunOutcome {
    #[host(rename = "completed")]
    Completed,
    #[host(rename = "input-required")]
    InputRequired,
    #[host(rename = "failed")]
    Failed,
    #[host(rename = "canceled")]
    Canceled,
    #[host(rename = "rejected")]
    Rejected,
}

/// One run as the journal tells it: `run_started` and, when it ended, `run_ended`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_run")]
pub struct RunSummary {
    pub run_id: String,
    pub lane: String,
    pub task_id: Option<String>,
    pub channel: Option<String>,
    pub started_at_ms: i64,
    pub ended_at_ms: Option<i64>,
    pub outcome: Option<RunOutcome>,
    pub reason: Option<String>,
}

impl RunSummary {
    /// A `run_started` without a `run_ended`: the worker died mid-run.
    pub fn interrupted(&self) -> bool {
        self.ended_at_ms.is_none()
    }

    /// In the order the runs started; a `run_ended` whose run was not started is ignored, and a
    /// second `run_started` of a run (a holder re-entering its lease) replaces the first in place.
    pub fn from_events(events: &[EventRow]) -> Vec<RunSummary> {
        let mut runs: Vec<RunSummary> = Vec::new();
        for ev in events {
            match ev.run_event() {
                Some(RunEvent::Started(started)) => {
                    let summary = RunSummary {
                        run_id: ev.run_id.clone(),
                        lane: ev.lane.clone(),
                        task_id: started.task_id,
                        channel: started.channel,
                        started_at_ms: ev.ts_ms,
                        ended_at_ms: None,
                        outcome: None,
                        reason: None,
                    };
                    match runs.iter_mut().find(|r| r.run_id == ev.run_id) {
                        Some(existing) => *existing = summary,
                        None => runs.push(summary),
                    }
                }
                Some(RunEvent::Ended(ended)) => {
                    if let Some(run) = runs.iter_mut().find(|r| r.run_id == ev.run_id) {
                        run.ended_at_ms = Some(ev.ts_ms);
                        run.outcome = Some(ended.outcome);
                        run.reason = ended.reason;
                    }
                }
                None => {}
            }
        }
        runs
    }
}

/// How many rows a lookup by id or `messageId` scans.
pub(crate) const LOOKUP_LIMIT: i64 = 500;
/// How many `input-required` rows `pending_ask` scans.
const PENDING_ASK_LIMIT: i64 = 50;

/// What a write changes; everything left `None` keeps the existing row's value.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TaskPatch {
    pub message_id: Option<String>,
    /// An optional application task label, the `name` column.
    pub name: Option<String>,
    /// Merged over the row's `data.metadata`; an empty map changes nothing.
    pub metadata: Option<Map<String, Value>>,
    pub ask: Option<Ask>,
    pub status_message: Option<a2a::Message>,
    pub history: Option<Vec<a2a::Message>>,
    pub artifacts: Option<Vec<a2a::Artifact>>,
}

/// The pending transition a task waits on.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WaitOptions {
    /// For a conversation tool's `InputRequired`, the pending call's tool.
    pub tool: Option<String>,
    /// For a conversation tool's `InputRequired`, the pending call.
    pub call_id: Option<String>,
    /// The tool the answer is handed to, when the `InputRequired` named one.
    pub answer_tool: Option<String>,
    /// The tool's parameter schema, added to the status message as a data part.
    pub schema: Option<Value>,
}

/// The lane's task rows as this run writes them.
pub struct Tasks {
    lane: LaneDb,
    /// In write order.
    writes: Vec<TaskWrite>,
    /// The lane's newest rows as the acquire loaded them, newest first: what the run's lookups
    /// read instead of the database. Complete when shorter than [`LOOKUP_LIMIT`].
    loaded: Option<Vec<TaskRow>>,
}

impl Tasks {
    pub fn new(lane: LaneDb) -> Self {
        Self { lane, writes: Vec::new(), loaded: None }
    }

    /// Over the rows an acquire loaded, newest first.
    pub fn loaded(lane: LaneDb, loaded: Vec<TaskRow>) -> Self {
        Self { lane, writes: Vec::new(), loaded: Some(loaded) }
    }

    /// The loaded rows, when they are the whole lane.
    fn snapshot(&self) -> Option<&[TaskRow]> {
        self.loaded.as_deref().filter(|rows| rows.len() < LOOKUP_LIMIT as usize)
    }

    /// The lane's newest rows, `limit` of them, as of the acquire.
    pub async fn recent(&self, limit: i64) -> Result<Vec<TaskRow>, StorageError> {
        if let Some(rows) = self.loaded.as_deref() {
            return Ok(rows.iter().take(limit as usize).cloned().collect());
        }
        self.lane.tasks(None, limit).await
    }

    /// The `working` rows, oldest first: a dead holder's, as of the acquire.
    pub async fn working(&self) -> Result<Vec<TaskRow>, StorageError> {
        if let Some(rows) = self.snapshot() {
            return Ok(rows
                .iter()
                .rev()
                .filter(|r| r.status == TaskState::Working)
                .cloned()
                .collect());
        }
        let working = self.lane.tasks(Some(TaskState::Working), LOOKUP_LIMIT).await?;
        Ok(working.into_iter().rev().collect())
    }

    pub fn lane(&self) -> &LaneDb {
        &self.lane
    }

    /// The pending writes, in the order they were first made.
    pub fn writes(&self) -> &[TaskWrite] {
        &self.writes
    }

    /// The first loaded row that matches, else, when the loaded rows are not the whole lane, the
    /// stored one `read` finds.
    async fn find(
        &self,
        matches: impl Fn(&TaskRow) -> bool,
        read: impl std::future::Future<Output = Result<Option<TaskRow>, StorageError>>,
    ) -> Result<Option<TaskRow>, StorageError> {
        match self.loaded.as_deref().and_then(|rows| rows.iter().find(|r| matches(r))) {
            Some(r) => Ok(Some(r.clone())),
            None if self.snapshot().is_some() => Ok(None),
            None => read.await,
        }
    }

    /// The task of an A2A `messageId`: a pending write first, then the lane's rows.
    pub async fn by_message_id(&self, message_id: &str) -> Result<Option<TaskRow>, StorageError> {
        if let Some(w) = self.writes.iter().find(|w| w.message_id.as_deref() == Some(message_id)) {
            // not stored yet: no `seq` and no `updated_at`
            return Ok(Some(TaskRow {
                task_id: w.task_id.clone(),
                lane: self.lane.name().to_owned(),
                seq: 0,
                name: w.name.clone(),
                message_id: w.message_id.clone(),
                status: w.status,
                updated_at: 0,
                data: w.data.clone(),
                ask: w.ask.clone(),
            }));
        }
        let read = self.lane.task_for_message(message_id);
        self.find(|r| r.message_id.as_deref() == Some(message_id), read).await
    }

    /// The newest `input-required` row holding a conversation tool's pending call.
    pub async fn pending_ask(&self) -> Result<Option<TaskRow>, StorageError> {
        if let Some(rows) = self.snapshot() {
            return Ok(rows
                .iter()
                .find(|r| r.status == TaskState::InputRequired && r.ask.is_some())
                .cloned());
        }
        Ok(self
            .lane
            .tasks(Some(TaskState::InputRequired), PENDING_ASK_LIMIT)
            .await?
            .into_iter()
            .find(|r| r.ask.is_some()))
    }

    /// Insert or update one row; what the patch leaves `None` keeps the existing row's value.
    pub async fn write(
        &mut self,
        task_id: &str,
        status: TaskState,
        patch: TaskPatch,
    ) -> Result<&TaskWrite, StorageError> {
        let base = match self.writes.iter().find(|w| w.task_id == task_id) {
            Some(w) => Some(w.clone()),
            None => {
                let row = self.find(|r| r.task_id == task_id, self.lane.task(task_id)).await?;
                row.map(|r| TaskWrite {
                    task_id: r.task_id,
                    status: r.status,
                    message_id: r.message_id,
                    name: r.name,
                    data: r.data,
                    ask: r.ask,
                })
            }
        };
        let write = merge_write(task_id, base.as_ref(), status, patch)?;
        match self.writes.iter().position(|w| w.task_id == task_id) {
            Some(i) => {
                self.writes[i] = write;
                Ok(&self.writes[i])
            }
            None => {
                self.writes.push(write);
                Ok(self.writes.last().expect("just pushed"))
            }
        }
    }

    /// `input-required`: the question is the status message, `ask` the pending transition.
    pub async fn wait(
        &mut self,
        task_id: &str,
        question: &str,
        opts: WaitOptions,
    ) -> Result<(), StorageError> {
        let ask = Ask {
            question: question.to_owned(),
            tool: opts.tool,
            call_id: opts.call_id,
            answer_tool: opts.answer_tool,
        };
        let mut parts = vec![a2a::Part::text(question)];
        if let Some(schema) = opts.schema {
            parts.push(a2a::Part::data(schema));
        }
        self.write(
            task_id,
            TaskState::InputRequired,
            TaskPatch {
                ask: Some(ask),
                status_message: Some(agent_message(parts, task_id)),
                ..TaskPatch::default()
            },
        )
        .await?;
        Ok(())
    }
}

/// `Tasks.write`'s row: `patch` over `base`, `data` and `internal` rebuilt from both.
fn merge_write(
    task_id: &str,
    base: Option<&TaskWrite>,
    status: TaskState,
    patch: TaskPatch,
) -> Result<TaskWrite, StorageError> {
    if let Some(b) = base
        && b.status.is_terminal()
    {
        return Err(StorageError::Invalid(format!(
            "task {task_id} is {}: terminal rows are immutable",
            b.status.as_str()
        )));
    }
    let mut data = base.map(|b| b.data.clone()).unwrap_or_default();
    if let Some(metadata) = patch.metadata {
        data.metadata.extend(metadata);
    }
    if let Some(history) = patch.history {
        data.history = history;
    }
    if let Some(artifacts) = patch.artifacts {
        data.artifacts = Some(artifacts);
    }
    if let Some(message) = patch.status_message {
        data.status_message = Some(message);
    }

    let ask = if patch.ask.is_none() && (status.is_terminal() || status == TaskState::Working) {
        // a terminal or running row holds no pending question
        None
    } else {
        patch.ask.or_else(|| base.and_then(|b| b.ask.clone()))
    };

    Ok(TaskWrite {
        task_id: task_id.to_owned(),
        status,
        message_id: patch.message_id.or_else(|| base.and_then(|b| b.message_id.clone())),
        name: patch.name.or_else(|| base.and_then(|b| b.name.clone())),
        data,
        ask,
    })
}

/// An agent message of `task_id`.
pub fn agent_message(parts: Vec<a2a::Part>, task_id: &str) -> a2a::Message {
    a2a::Message {
        message_id: shortuuid("msg_"),
        task_id: Some(task_id.to_owned()),
        ..a2a::Message::new(a2a::Role::Agent, parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use livekit_protocol::agent::chat_context::chat_item;
    use serde_json::json;

    #[test]
    fn item_changes_skip_instructions_and_unchanged_items_and_list_deletions() {
        use livekit_protocol::agent::chat_message::{ChatContent, chat_content::Payload};
        use livekit_protocol::agent::{AgentConfigUpdate, ChatMessage, ChatRole, FunctionCall};
        let msg = |id: &str, role: ChatRole, text: &str| {
            ItemRow::new(chat_item::Item::Message(ChatMessage {
                id: id.into(),
                role: role as i32,
                content: vec![ChatContent { payload: Some(Payload::Text(text.into())) }],
                ..Default::default()
            }))
        };
        let call = ItemRow::new(chat_item::Item::FunctionCall(FunctionCall {
            id: "i3".into(),
            name: "f".into(),
            ..Default::default()
        }));
        let loaded: Vec<(String, ChatItem)> =
            [msg("i1", ChatRole::User, "hi"), msg("i2", ChatRole::Assistant, "hello"), call]
                .into_iter()
                .map(|i| (i.id, i.payload))
                .collect();
        let mut history = vec![
            msg("sys", ChatRole::System, "be nice"),
            msg("dev", ChatRole::Developer, "x"),
            ItemRow::new(chat_item::Item::AgentConfigUpdate(AgentConfigUpdate {
                id: "cfg".into(),
                ..Default::default()
            })),
            msg("i1", ChatRole::User, "hi"),
            // edited
            msg("i2", ChatRole::Assistant, "hey"),
            // i3 removed; i4 new
            ItemRow { message_id: Some("m4".into()), ..msg("i4", ChatRole::User, "bye") },
        ];
        let (changed, deleted) = item_changes(&history, &loaded);
        assert_eq!(
            changed.iter().map(|w| (w.id.as_str(), w.message_id.as_deref())).collect::<Vec<_>>(),
            vec![("i2", None), ("i4", Some("m4"))]
        );
        assert_eq!(deleted, vec!["i3".to_owned()]);
        history.clear();
        let (changed, deleted) = item_changes(&history, &loaded);
        assert!(changed.is_empty());
        assert_eq!(deleted, vec!["i1", "i2", "i3"]);
    }

    #[test]
    fn run_summaries_follow_run_started_and_run_ended() {
        let ev = |id: i64, run_id: &str, ty: &str, payload: Value| EventRow {
            id,
            ts_ms: id * 10,
            lane: "front-desk".into(),
            run_id: run_id.into(),
            r#type: ty.into(),
            payload,
        };
        let events = vec![
            ev(1, "r1", "run_started", json!({"task_id": "AST_1", "channel": "a2a"})),
            ev(2, "r1", "run_ended", json!({"outcome": "completed", "reason": null})),
            ev(3, "r2", "run_started", json!({"task_id": "AST_2", "channel": null})),
            ev(4, "r9", "run_ended", json!({"outcome": "failed"})),
        ];
        let runs = RunSummary::from_events(&events);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].outcome, Some(RunOutcome::Completed));
        assert_eq!(runs[0].ended_at_ms, Some(20));
        assert!(!runs[0].interrupted());
        assert_eq!(runs[1].channel, None);
        assert!(runs[1].interrupted());
    }

    fn message(id: &str) -> a2a::Message {
        a2a::Message { message_id: id.into(), ..a2a::Message::new(a2a::Role::User, vec![]) }
    }

    #[test]
    fn a_later_write_keeps_the_row_and_clears_the_ask_when_terminal_or_working() {
        let base = merge_write(
            "AST_1",
            None,
            TaskState::InputRequired,
            TaskPatch {
                message_id: Some("m1".into()),
                ask: Some(Ask { question: "when?".into(), ..Ask::default() }),
                history: Some(vec![message("m1")]),
                ..TaskPatch::default()
            },
        )
        .unwrap();
        let ask = |w: &TaskWrite| w.ask.clone();
        assert_eq!(ask(&base).map(|a| a.question), Some("when?".into()));
        let msg = merge_write(
            "AST_1",
            Some(&base),
            TaskState::InputRequired,
            TaskPatch { history: Some(vec![message("m2"), message("m3")]), ..TaskPatch::default() },
        )
        .unwrap();
        assert_eq!(ask(&msg), ask(&base), "input-required keeps the pending transition");
        assert_eq!(msg.message_id.as_deref(), Some("m1"));
        let done = merge_write(
            "AST_1",
            Some(&base),
            TaskState::Completed,
            TaskPatch { artifacts: Some(vec![]), ..TaskPatch::default() },
        )
        .unwrap();
        assert_eq!(ask(&done), None);
        assert_eq!(done.data.artifacts, Some(vec![]));
        assert_eq!(done.data.history, vec![message("m1")]);
        let working =
            merge_write("AST_1", Some(&base), TaskState::Working, TaskPatch::default()).unwrap();
        assert_eq!(ask(&working), None);
        let again = merge_write("AST_1", Some(&done), TaskState::Working, TaskPatch::default());
        assert!(matches!(again, Err(StorageError::Invalid(m)) if m.contains("is completed")));
    }

    #[test]
    fn metadata_merges_over_the_stored_metadata() {
        let base = merge_write(
            "AST_1",
            None,
            TaskState::Submitted,
            TaskPatch {
                name: Some("T".into()),
                metadata: Some(json!({"a": 1, "b": 1}).as_object().cloned().unwrap()),
                ..TaskPatch::default()
            },
        )
        .unwrap();
        let next = merge_write(
            "AST_1",
            Some(&base),
            TaskState::Canceled,
            TaskPatch {
                metadata: Some(json!({"b": 2, "c": 3}).as_object().cloned().unwrap()),
                ..TaskPatch::default()
            },
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&next.data.metadata).unwrap(),
            json!({"a": 1, "b": 2, "c": 3})
        );
        assert_eq!(next.name.as_deref(), Some("T"));
    }
}
