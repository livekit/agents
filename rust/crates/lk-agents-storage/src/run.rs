//! The run: the unit of work a lane commits (plan 5.2, 13.8), over a [`ThreadDb`]. The port of
//! `storage/run.py`; Python and Node keep the developer-facing surface and call this through
//! `_core`.
//!
//! [`Thread`] is one thread's database, shared by every lane; [`Lane`] one registered agent's
//! part of it; [`Lane::open`] is the `lane.run(...)` of 5.2:
//!
//! * it takes the lease and loads the lane in one acquire batch (3.3.2), waiting for a held lease
//!   with backoff from 100 ms to 1 s, as the lane's `waiter` under `interrupt`; a `run_id` that
//!   already committed yields a closed [`Run`] that writes nothing;
//! * [`Run::commit`] writes history, state, agent stack, events, task rows and usage in one fenced
//!   write, releasing the lease or renewing it when the runner continues;
//! * [`Run::abandon`] writes `run_ended` with the outcome and releases, nothing else;
//! * `exclusive: false` is the `concurrent` policy: no lease, append-only, no fence.
//!
//! History items and state cross as JSON: an [`Item`] is the stored payload of one chat item, in
//! the extension's canonical form, with its id; the state is the host's snapshot.

mod records;
mod tasks;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use records::*;
pub use tasks::{SubmitOptions, TaskPatch, Tasks, WaitOptions, agent_message, artifact};

use crate::StorageError;
use crate::schema::{ChatItemRow, EventRow, InboxRow};
use crate::statements::{
    AcquireRequest, CommitRequest, EventWrite, Fence, ItemWrite, Lease, LeaseAfter, Loaded,
    RunEndedRequest, SubmittedTask, ThreadDb,
};

/// Plan 3.3.2: a waiter that has not acquired after 60 s fails `lease-timeout`.
pub const LEASE_WAIT_TIMEOUT: Duration = Duration::from_secs(60);
pub const LEASE_POLL_MIN: Duration = Duration::from_millis(100);
pub const LEASE_POLL_MAX: Duration = Duration::from_secs(1);

/// `utils.shortuuid`: `prefix + uuid4().hex[:12]`, six random bytes as hex.
pub fn shortuuid(prefix: &str) -> String {
    use rand::RngCore;
    use std::fmt::Write as _;
    let mut bytes = [0u8; 6];
    rand::rng().fill_bytes(&mut bytes);
    let mut out = String::with_capacity(prefix.len() + 12);
    out.push_str(prefix);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// The clock a session stamps events with, in Unix milliseconds; tests inject one.
pub type Clock = Arc<dyn Fn() -> i64 + Send + Sync>;

fn wall_clock() -> Clock {
    Arc::new(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as i64)
    })
}

/// A JSON value in one canonical text: object keys sorted, no spaces. Only equality matters
/// (`json.dumps(payload, sort_keys=True, separators=(",", ":"))` in the Python).
pub fn canonical(v: &Value) -> String {
    fn sorted(v: &Value) -> Value {
        match v {
            Value::Object(o) => {
                let mut keys: Vec<&String> = o.keys().collect();
                keys.sort();
                Value::Object(keys.into_iter().map(|k| (k.clone(), sorted(&o[k]))).collect())
            }
            Value::Array(a) => Value::Array(a.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    sorted(v).to_string()
}

/// One history item in memory: the chat item's JSON (`item_to_payload`, the form `chat_item`
/// converts to and from the stored protobuf JSON), with the `id` the row carries. The payload's `type` is the item
/// kind: `message`, `function_call`, `function_call_output`, `agent_handoff`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_run", config = "frozen=True")]
pub struct Item {
    pub id: String,
    pub payload: Value,
}

impl Item {
    /// From a payload that carries its own `id` and `type`, as every chat item does.
    pub fn new(payload: Value) -> Result<Self, StorageError> {
        let text =
            |key: &str| {
                payload.get(key).and_then(Value::as_str).map(str::to_owned).ok_or_else(|| {
                    StorageError::Invalid(format!("chat item without a string {key}"))
                })
            };
        text("type")?;
        Ok(Self { id: text("id")?, payload })
    }

    pub fn from_row(row: ChatItemRow) -> Self {
        Self { id: row.id, payload: row.payload }
    }

    /// The payload's `type`.
    pub fn kind(&self) -> &str {
        self.payload.get("type").and_then(Value::as_str).unwrap_or_default()
    }

    /// Instructions are rendered per run, never stored (plan 5.1), and `agent_config_update` is a
    /// runtime note to the model, not a chat item of the extension.
    fn stored(&self) -> bool {
        if self.kind() == "agent_config_update" {
            return false;
        }
        if self.kind() == "message"
            && matches!(
                self.payload.get("role").and_then(Value::as_str),
                Some("system" | "developer")
            )
        {
            return false;
        }
        true
    }
}

/// Who is speaking in the lane: `Ref` entries, pushed on a handoff, popped on `Completed`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentStack {
    refs: Vec<Ref>,
    dirty: bool,
}

impl AgentStack {
    pub fn new(refs: Vec<Ref>) -> Self {
        Self { refs, dirty: false }
    }

    pub fn push(&mut self, r: Ref) {
        self.refs.push(r);
        self.dirty = true;
    }

    pub fn pop(&mut self) -> Option<Ref> {
        let r = self.refs.pop()?;
        self.dirty = true;
        Some(r)
    }

    pub fn replace(&mut self, refs: Vec<Ref>) {
        self.refs = refs;
        self.dirty = true;
    }

    pub fn clear(&mut self) {
        if !self.refs.is_empty() {
            self.refs.clear();
            self.dirty = true;
        }
    }

    pub fn current(&self) -> Option<&Ref> {
        self.refs.last()
    }

    pub fn refs(&self) -> &[Ref] {
        &self.refs
    }

    pub fn len(&self) -> usize {
        self.refs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.refs.is_empty()
    }

    /// Changed since it was loaded: the commit writes the column only then.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn to_json(&self) -> Value {
        Value::Array(self.refs.iter().map(Ref::to_json).collect())
    }
}

/// The lane's journal as this run sees it: what was loaded, and what it appends.
pub struct EventLog {
    pub loaded: Vec<EventRow>,
    clock: Clock,
    new: Vec<EventWrite>,
}

impl EventLog {
    pub fn new(loaded: Vec<EventRow>, clock: Clock) -> Self {
        Self { loaded, clock, new: Vec::new() }
    }

    /// Appends an event, stamped now.
    pub fn append(&mut self, r#type: &str, payload: Value) {
        self.new.push(EventWrite { ts_ms: (self.clock)(), r#type: r#type.to_owned(), payload });
    }

    pub fn new_events(&self) -> &[EventWrite] {
        &self.new
    }
}

/// Token usage of one run (`voice.run_result.RunUsage`); written as a `usage` event when the run
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

impl RunUsage {
    pub fn total_tokens(&self) -> i64 {
        self.input_tokens + self.output_tokens
    }
}

/// `lane.run(...)`'s arguments.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Idempotency key: a `run_id` that already committed yields a closed run.
    pub run_id: String,
    /// The A2A task this run serves; also its answer stream's id.
    pub task_id: String,
    /// `false` is the `concurrent` policy: no lease, append-only commit.
    pub exclusive: bool,
    /// Poll for a held lease (100 ms to 1 s backoff, `wait_timeout`); `false` fails with
    /// [`StorageError::LeaseHeld`] at once.
    pub wait: bool,
    /// Record this worker as the lane's `waiter` while it polls, so the holder yields at its next
    /// boundary (plan 3.3.4).
    pub interrupt: bool,
    /// The request's channel, for the `run_started` event.
    pub channel: Option<String>,
    pub wait_timeout: Duration,
    /// The turn's `submitted` row, written in the acquire batch (`Run::submitted` tells whether
    /// it was new); `None` for a run with no row or one submitted earlier.
    pub submit: Option<SubmittedTask>,
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

// ---------------------------------------------------------------------------------------------
// the session
// ---------------------------------------------------------------------------------------------

struct SessionInner {
    db: ThreadDb,
    id: String,
    holder: String,
    clock: Clock,
    after_commit: Option<Arc<dyn Fn() + Send + Sync>>,
}

/// The session's database, shared by every lane (plan 5.2 `SessionHandle`). Cheap to clone: one
/// connection behind every clone.
#[derive(Clone)]
pub struct Thread {
    inner: Arc<SessionInner>,
}

impl Thread {
    /// Over an open database. The lease's holder is this handle, not the process: the acquire lets a
    /// holder re-take its own lease (plan 3.3.2), which is for the inbox receive and the drain that
    /// follows it, never for a second request of the same worker on the same lane, which must wait
    /// like a request from any other worker.
    pub fn new(db: ThreadDb, holder: &str) -> Self {
        let id = db.conn().database_id().to_owned();
        Self {
            inner: Arc::new(SessionInner {
                db,
                id,
                holder: format!("{holder}/{}", shortuuid("")),
                clock: wall_clock(),
                after_commit: None,
            }),
        }
    }

    /// The clock events are stamped with; tests set one.
    pub fn with_clock(mut self, clock: Clock) -> Self {
        Arc::get_mut(&mut self.inner).expect("a fresh thread").clock = clock;
        self
    }

    /// The backend's hook after every commit: the touch of the thread (plan 3.3.8).
    pub fn with_after_commit(mut self, hook: Arc<dyn Fn() + Send + Sync>) -> Self {
        Arc::get_mut(&mut self.inner).expect("a fresh thread").after_commit = Some(hook);
        self
    }

    /// The database id.
    pub fn id(&self) -> &str {
        &self.inner.id
    }

    /// `<worker holder>/<handle>`: what this handle's leases and waiter marks carry.
    pub fn holder(&self) -> &str {
        &self.inner.holder
    }

    pub fn now_ms(&self) -> i64 {
        (self.inner.clock)()
    }

    pub fn db(&self) -> &ThreadDb {
        &self.inner.db
    }

    fn after_commit(&self) {
        if let Some(hook) = &self.inner.after_commit {
            hook();
        }
    }

    pub fn lane(&self, agent: &str) -> Lane {
        Lane { thread: self.clone(), name: agent.to_owned() }
    }

    /// The journal, every lane, after the event `after_id`.
    pub async fn events(&self, after_id: i64, limit: i64) -> Result<Vec<EventRow>, StorageError> {
        self.inner.db.events(after_id, limit).await
    }

    pub async fn lanes(&self) -> Result<Vec<LaneInfo>, StorageError> {
        self.inner.db.lanes().await?.into_iter().map(LaneInfo::from_row).collect()
    }

    pub fn close(&self) {
        self.inner.db.close();
    }
}

// ---------------------------------------------------------------------------------------------
// the lane
// ---------------------------------------------------------------------------------------------

/// One registered agent's part of a session (plan 5.2 `LaneHandle`).
#[derive(Clone)]
pub struct Lane {
    thread: Thread,
    name: String,
}

impl Lane {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn thread(&self) -> &Thread {
        &self.thread
    }

    fn db(&self) -> &ThreadDb {
        self.thread.db()
    }

    /// The whole history; how another lane reads it.
    pub async fn history(&self) -> Result<Vec<Item>, StorageError> {
        Ok(self.db().history(&self.name).await?.into_iter().map(Item::from_row).collect())
    }

    /// The last `limit` runs, newest first; the `submitted` insert before a lease is
    /// bookkeeping, not a turn.
    pub async fn runs(&self, limit: usize) -> Result<Vec<RunSummary>, StorageError> {
        let rows = self.db().lane_events(&self.name, 0).await?;
        let mut runs: Vec<RunSummary> = RunSummary::from_events(&rows)
            .into_iter()
            .filter(|r| r.channel.as_deref() != Some(SUBMIT_CHANNEL))
            .collect();
        let keep = runs.len().saturating_sub(limit);
        runs.drain(..keep);
        runs.reverse();
        Ok(runs)
    }

    pub async fn tasks(
        &self,
        status: Option<TaskState>,
        limit: i64,
    ) -> Result<Vec<TaskRecord>, StorageError> {
        let rows = self.db().tasks(&self.name, status.map(TaskState::as_str), limit).await?;
        Ok(rows.into_iter().map(TaskRecord::from_row).collect())
    }

    pub async fn task(&self, task_id: &str) -> Result<Option<TaskRecord>, StorageError> {
        Ok(self.tasks(None, 500).await?.into_iter().find(|r| r.id == task_id))
    }

    pub async fn inbox_pending(&self) -> Result<Vec<InboxRow>, StorageError> {
        self.db().inbox_pending(&self.name).await
    }

    pub async fn info(&self) -> Result<Option<LaneInfo>, StorageError> {
        self.db().lane(&self.name).await?.map(LaneInfo::from_row).transpose()
    }

    /// Takes the lane (plan 3.3) and loads it. The run commits with [`Run::commit`] and ends by
    /// error with [`Run::abandon`]; a run dropped without either stops renewing and lets its lease
    /// lapse, which the next acquire sees as a `run_started` without a `run_ended`.
    pub async fn open(&self, opts: RunOptions) -> Result<Run, StorageError> {
        if opts.exclusive { self.acquire(opts).await } else { self.open_concurrent(opts).await }
    }

    async fn acquire(&self, opts: RunOptions) -> Result<Run, StorageError> {
        let thread = &self.thread;
        let holder = thread.holder().to_owned();
        let token = shortuuid("lease_");
        let mut request = AcquireRequest {
            lane: self.name.clone(),
            holder: holder.clone(),
            lease_token: token.clone(),
            run_id: opts.run_id.clone(),
            ts_ms: thread.now_ms(),
            task_id: opts.task_id.clone(),
            run_started: serde_json::json!({
                "task_id": opts.task_id,
                "stream_id": opts.task_id,
                "channel": opts.channel,
            }),
            events_after: 0,
            submit: opts.submit.clone(),
            pickup: opts.pickup,
            tasks_limit: tasks::LOOKUP_LIMIT,
        };
        let deadline = Instant::now() + opts.wait_timeout;
        let mut delay = LEASE_POLL_MIN;
        let mut marked = false;
        let loaded: Loaded = loop {
            request.ts_ms = thread.now_ms();
            if let Some(loaded) = self.db().acquire(&request).await? {
                break loaded;
            }
            // the row is in; a retry of the batch must not report it as a duplicate
            request.submit = None;
            if !opts.wait {
                return Err(StorageError::LeaseHeld(format!(
                    "lane {} is held by another run",
                    self.name
                )));
            }
            if opts.interrupt && !marked {
                marked = self.db().set_waiter(&self.name, &holder).await?;
                if marked {
                    log::debug!(target: "livekit.agents", lane = self.name.as_str(), holder = holder.as_str(); "waiting for the lane as its waiter");
                }
            }
            if Instant::now() >= deadline {
                return Err(StorageError::LeaseTimeout(format!(
                    "lane {} was not released within {}s",
                    self.name,
                    opts.wait_timeout.as_secs_f64()
                )));
            }
            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(LEASE_POLL_MAX);
        };
        let mut history = Vec::with_capacity(loaded.items.len());
        let mut loaded_order = Vec::with_capacity(loaded.items.len());
        let mut loaded_payloads = HashMap::new();
        let mut message_ids = HashMap::new();
        for row in loaded.items {
            if let Some(m) = &row.message_id {
                message_ids.insert(row.id.clone(), m.clone());
            }
            loaded_order.push(row.id.clone());
            loaded_payloads.insert(row.id.clone(), canonical(&row.payload));
            history.push(Item::from_row(row));
        }
        let stack = Ref::stack_from_json(loaded.lane.agent_stack.as_ref())?;
        let closed = loaded.already_committed;
        let lease =
            if closed { None } else { Some(self.db().start_renewal(&self.name, &holder, &token)) };
        Ok(Run {
            thread: thread.clone(),
            lane: self.name.clone(),
            db: self.db().clone(),
            run_id: opts.run_id,
            task_id: opts.task_id,
            channel: opts.channel,
            version: loaded.version,
            closed,
            history,
            loaded_order,
            loaded_payloads,
            message_ids,
            state: loaded.lane.state.clone(),
            state_json: loaded.lane.state,
            agents: AgentStack::new(stack),
            events: EventLog::new(loaded.events, thread.inner.clock.clone()),
            tasks: Tasks::loaded(self.db().clone(), &self.name, loaded.tasks),
            submitted: loaded.submitted,
            picked_up: loaded.picked_up,
            answer_task: None,
            committed_task: None,
            usage: RunUsage::default(),
            lease,
            lease_token: Some(token),
            exclusive: true,
            holder,
            outcome: RunOutcome::Completed,
            reason: None,
            keep_lease: false,
            drained_id: None,
            committed_version: None,
            finished: false,
        })
    }

    async fn open_concurrent(&self, opts: RunOptions) -> Result<Run, StorageError> {
        let thread = &self.thread;
        let rows = self.db().history(&self.name).await?;
        let loaded_order: Vec<String> = rows.iter().map(|r| r.id.clone()).collect();
        let loaded_payloads = rows.iter().map(|r| (r.id.clone(), canonical(&r.payload))).collect();
        Ok(Run {
            thread: thread.clone(),
            lane: self.name.clone(),
            db: self.db().clone(),
            run_id: opts.run_id,
            task_id: opts.task_id,
            channel: opts.channel,
            version: 0,
            closed: false,
            history: rows.into_iter().map(Item::from_row).collect(),
            loaded_order,
            loaded_payloads,
            message_ids: HashMap::new(),
            state: None,
            state_json: None,
            agents: AgentStack::default(),
            events: EventLog::new(Vec::new(), thread.inner.clock.clone()),
            tasks: Tasks::new(self.db().clone(), &self.name),
            submitted: None,
            picked_up: false,
            answer_task: None,
            committed_task: None,
            usage: RunUsage::default(),
            lease: None,
            lease_token: None,
            exclusive: false,
            holder: thread.holder().to_owned(),
            outcome: RunOutcome::Completed,
            reason: None,
            keep_lease: false,
            drained_id: None,
            committed_version: None,
            finished: false,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// the run
// ---------------------------------------------------------------------------------------------

/// Everything a run reads and writes, committed together (plan 5.2).
pub struct Run {
    thread: Thread,
    lane: String,
    db: ThreadDb,
    pub run_id: String,
    pub task_id: String,
    pub channel: Option<String>,
    /// The version loaded: the fence of the commit.
    pub version: i64,
    /// A `run_id` that already committed: the run writes nothing and answers nothing twice.
    pub closed: bool,
    /// The whole history, loaded; append as the model and the tools speak, delete by removing.
    pub history: Vec<Item>,
    loaded_order: Vec<String>,
    loaded_payloads: HashMap<String, String>,
    message_ids: HashMap<String, String>,
    /// The state snapshot as loaded, `None` for a fresh lane; the host replaces it with its
    /// default-constructed state, and the commit writes it when it differs from what was loaded.
    pub state: Option<Value>,
    state_json: Option<Value>,
    pub agents: AgentStack,
    pub events: EventLog,
    pub tasks: Tasks,
    /// Whether the acquire's `submit` inserted its row; `Some(false)` is a retried `messageId`.
    pub submitted: Option<bool>,
    /// Whether the acquire's `pickup` took the row to `working`.
    pub picked_up: bool,
    /// The task the turn answers: its row comes back with the commit (`committed_task`).
    pub answer_task: Option<String>,
    /// `answer_task`'s row as the commit left it.
    pub committed_task: Option<TaskRecord>,
    pub usage: RunUsage,
    lease: Option<Lease>,
    lease_token: Option<String>,
    pub exclusive: bool,
    pub holder: String,
    pub outcome: RunOutcome,
    pub reason: Option<String>,
    /// Renew the lease at commit instead of releasing it: the runner continues with pending inbox
    /// rows or due tasks (3.3.4).
    pub keep_lease: bool,
    /// The last inbox row a drain consumed, written at commit.
    pub drained_id: Option<i64>,
    committed_version: Option<i64>,
    finished: bool,
}

impl Run {
    pub fn lane(&self) -> &str {
        &self.lane
    }

    pub fn lease_token(&self) -> Option<&str> {
        self.lease_token.as_deref()
    }

    pub fn lease(&self) -> Option<&Lease> {
        self.lease.as_ref()
    }

    /// A renew returned no row or failed: the run can no longer commit.
    pub fn lease_lost(&self) -> bool {
        self.lease.as_ref().is_some_and(Lease::is_lost)
    }

    /// The version after the commit; `None` before it, and for a closed run.
    pub fn committed_version(&self) -> Option<i64> {
        self.committed_version
    }

    /// Records the A2A `messageId` an item came from: the idempotency key on `items`.
    pub fn mark_message(&mut self, item_id: &str, message_id: &str) {
        self.message_ids.insert(item_id.to_owned(), message_id.to_owned());
    }

    /// Awaits `fut` while the lease holds; a lost lease cancels it with `Unavailable`.
    pub async fn guard<F: Future>(&self, fut: F) -> Result<F::Output, StorageError> {
        let Some(lease) = &self.lease else {
            return Ok(fut.await);
        };
        tokio::pin!(fut);
        tokio::select! {
            out = &mut fut => Ok(out),
            _ = lease.lost() => Err(StorageError::Unavailable(format!("lease on lane {} was lost", self.lane))),
        }
    }

    /// The boundary check under `interrupt`: a worker waiting for the lane, or `None`.
    pub async fn waiter(&self) -> Result<Option<String>, StorageError> {
        if !self.exclusive {
            return Ok(None);
        }
        Ok(self.db.waiter(&self.lane).await?.filter(|w| !w.is_empty() && *w != self.holder))
    }

    /// The items that changed since the load, and the loaded ids no longer in the history.
    pub fn item_writes(&self) -> (Vec<ItemWrite>, Vec<String>) {
        item_writes(&self.history, &self.loaded_order, &self.loaded_payloads, &self.message_ids)
    }

    /// `Run._commit_request`: what [`Run::commit`] sends, for inspection.
    pub fn commit_request(&self, now_ms: i64) -> CommitRequest {
        let (items, delete_items) = self.item_writes();
        let mut events: Vec<EventWrite> = Vec::new();
        if !self.exclusive {
            events.push(EventWrite {
                ts_ms: now_ms,
                r#type: "run_started".into(),
                payload: serde_json::json!({
                    "task_id": self.task_id,
                    "stream_id": self.task_id,
                    "channel": self.channel,
                    "exclusive": false,
                }),
            });
        }
        // docs/journal.md: every item that entered the history, before the run's own events
        events.extend(items.iter().filter(|w| !self.loaded_payloads.contains_key(&w.id)).map(
            |w| EventWrite {
                ts_ms: now_ms,
                r#type: "conversation_item_added".into(),
                payload: serde_json::json!({"item": w.payload}),
            },
        ));
        events.extend(self.events.new_events().iter().cloned());
        if self.usage.requests != 0 || self.usage.tool_calls != 0 {
            events.push(EventWrite {
                ts_ms: now_ms,
                r#type: "usage".into(),
                payload: serde_json::to_value(self.usage).unwrap_or(Value::Null),
            });
        }
        let state = if self.exclusive {
            self.state.clone().filter(|s| Some(s) != self.state_json.as_ref())
        } else {
            None
        };
        CommitRequest {
            lane: self.lane.clone(),
            run_id: self.run_id.clone(),
            fence: self.exclusive.then(|| Fence {
                version: self.version,
                lease_token: self.lease_token.clone().unwrap_or_default(),
            }),
            items,
            delete_items,
            events,
            tasks: self.tasks.writes().to_vec(),
            state,
            agent_stack: (self.exclusive && self.agents.is_dirty()).then(|| self.agents.to_json()),
            drained_id: if self.exclusive { self.drained_id } else { None },
            run_ended: EventWrite {
                ts_ms: now_ms,
                r#type: "run_ended".into(),
                payload: serde_json::json!({"outcome": self.outcome, "reason": self.reason}),
            },
            after: if self.keep_lease { LeaseAfter::Renew } else { LeaseAfter::Release },
            read_task: self.answer_task.clone(),
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

    /// Leaving normally: one fenced write of everything the run holds (plan 5.2). A closed run
    /// releases the lease and writes nothing. `VersionConflict` means the run outlived its lease.
    pub async fn commit(&mut self, now_ms: i64) -> Result<(), StorageError> {
        self.finish()?;
        if self.closed {
            if let Some(token) = &self.lease_token {
                self.db.release(&self.lane, token).await?;
            }
            return Ok(());
        }
        let request = self.commit_request(now_ms);
        let result = self.db.commit(&request).await?;
        self.committed_version = Some(result.version);
        self.committed_task = result.task.map(TaskRecord::from_row);
        self.thread.after_commit();
        Ok(())
    }

    /// Leaving by error: `run_ended` with the outcome and the reason, and the release; nothing else.
    /// A closed run releases; a concurrent run has nothing to write. The Python passes `Canceled`
    /// for a `CancelledError` and `Failed` otherwise, with `reason` as `<type>: <message>`. A
    /// `run_ended` that could not be written is logged, as a crash would leave the same journal.
    pub async fn abandon(
        &mut self,
        now_ms: i64,
        outcome: RunOutcome,
        reason: impl Into<String>,
    ) -> Result<(), StorageError> {
        self.finish()?;
        if self.closed || !self.exclusive {
            if let (true, Some(token)) = (self.closed, &self.lease_token) {
                self.db.release(&self.lane, token).await?;
            }
            return Ok(());
        }
        let reason = reason.into();
        let request = RunEndedRequest {
            lane: self.lane.clone(),
            run_id: self.run_id.clone(),
            lease_token: self.lease_token.clone().unwrap_or_default(),
            ts_ms: now_ms,
            payload: serde_json::json!({"outcome": outcome, "reason": reason}),
        };
        if let Err(e) = self.db.run_ended(&request).await {
            log::warn!(target: "livekit.agents", lane = self.lane.as_str(), runId = self.run_id.as_str(); "run_ended could not be written after a failed run: {e}");
        }
        Ok(())
    }
}

/// `Run._item_writes`: every stored item whose canonical JSON differs from what was loaded, with
/// the A2A `messageId` it came from; the loaded ids no longer in the history, in loaded order.
fn item_writes(
    history: &[Item],
    loaded_order: &[String],
    loaded_payloads: &HashMap<String, String>,
    message_ids: &HashMap<String, String>,
) -> (Vec<ItemWrite>, Vec<String>) {
    let mut writes = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for item in history.iter().filter(|i| i.stored()) {
        seen.insert(item.id.as_str());
        if loaded_payloads.get(&item.id).is_some_and(|l| *l == canonical(&item.payload)) {
            continue;
        }
        writes.push(ItemWrite {
            id: item.id.clone(),
            message_id: message_ids.get(&item.id).cloned(),
            payload: item.payload.clone(),
        });
    }
    let deleted = loaded_order.iter().filter(|id| !seen.contains(id.as_str())).cloned().collect();
    (writes, deleted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn shortuuid_is_twelve_hex_after_the_prefix() {
        let id = shortuuid("AST_");
        assert_eq!(id.len(), 16);
        assert!(id[4..].chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(shortuuid(""), shortuuid(""));
    }

    #[test]
    fn canonical_sorts_keys_at_every_depth() {
        let a = json!({"b": [{"y": 1, "x": 2}], "a": null});
        let b = json!({"a": null, "b": [{"x": 2, "y": 1}]});
        assert_eq!(canonical(&a), canonical(&b));
        assert_eq!(canonical(&a), r#"{"a":null,"b":[{"x":2,"y":1}]}"#);
    }

    #[test]
    fn item_writes_skip_instructions_and_unchanged_items_and_list_deletions() {
        let loaded = [
            Item::new(json!({"id": "i1", "type": "message", "role": "user", "content": ["hi"]}))
                .unwrap(),
            Item::new(
                json!({"id": "i2", "type": "message", "role": "assistant", "content": ["hello"]}),
            )
            .unwrap(),
            Item::new(json!({"id": "i3", "type": "function_call", "name": "f"})).unwrap(),
        ];
        let order: Vec<String> = loaded.iter().map(|i| i.id.clone()).collect();
        let payloads: HashMap<String, String> =
            loaded.iter().map(|i| (i.id.clone(), canonical(&i.payload))).collect();
        let mut history = vec![
            Item::new(
                json!({"id": "sys", "type": "message", "role": "system", "content": ["be nice"]}),
            )
            .unwrap(),
            Item::new(
                json!({"id": "dev", "type": "message", "role": "developer", "content": ["x"]}),
            )
            .unwrap(),
            Item::new(json!({"id": "cfg", "type": "agent_config_update", "tools": []})).unwrap(),
            // the same item with its keys in another order: unchanged
            Item::new(json!({"content": ["hi"], "role": "user", "type": "message", "id": "i1"}))
                .unwrap(),
            // edited
            Item::new(
                json!({"id": "i2", "type": "message", "role": "assistant", "content": ["hey"]}),
            )
            .unwrap(),
            // i3 removed; i4 new
            Item::new(json!({"id": "i4", "type": "message", "role": "user", "content": ["bye"]}))
                .unwrap(),
        ];
        let mut message_ids = HashMap::new();
        message_ids.insert("i4".to_owned(), "m4".to_owned());
        let (writes, deleted) = item_writes(&history, &order, &payloads, &message_ids);
        assert_eq!(
            writes.iter().map(|w| (w.id.as_str(), w.message_id.as_deref())).collect::<Vec<_>>(),
            vec![("i2", None), ("i4", Some("m4"))]
        );
        assert_eq!(writes[1].payload["type"], "message");
        assert_eq!(deleted, vec!["i3".to_owned()]);
        history.clear();
        let (writes, deleted) = item_writes(&history, &order, &payloads, &message_ids);
        assert!(writes.is_empty());
        assert_eq!(deleted, order);
        assert!(Item::new(json!({"type": "message"})).is_err());
    }

    #[test]
    fn the_agent_stack_tracks_dirtiness() {
        let mut s = AgentStack::new(vec![Ref::new("front-desk")]);
        assert!(!s.is_dirty());
        assert!(s.current().is_some_and(|r| r.name == "front-desk"));
        s.clear();
        assert!(s.is_dirty() && s.is_empty());
        let mut s = AgentStack::default();
        s.clear();
        assert!(!s.is_dirty(), "clearing an empty stack changes nothing");
        assert_eq!(s.pop(), None);
        assert!(!s.is_dirty());
        s.push(Ref::with_args("Followup", json!({"n": 1})));
        assert_eq!(s.to_json().to_string(), r#"[{"name":"Followup","args":{"n":1}}]"#);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn the_event_log_stamps_with_the_clock() {
        let clock: Clock = Arc::new(|| 42);
        let mut log = EventLog::new(Vec::new(), clock);
        log.append("agent_state_changed", json!({"state": "thinking"}));
        log.append("tool_called", json!({}));
        let new = log.new_events();
        assert_eq!((new[0].ts_ms, new[1].r#type.as_str()), (42, "tool_called"));
    }

    #[test]
    fn run_usage_serializes_as_the_dataclass() {
        let u = RunUsage { requests: 1, input_tokens: 2, output_tokens: 3, tool_calls: 0 };
        assert_eq!(
            serde_json::to_string(&u).unwrap(),
            r#"{"requests":1,"input_tokens":2,"output_tokens":3,"tool_calls":0}"#
        );
        assert_eq!(u.total_tokens(), 5);
        let o = RunOptions::new("r1", "AST_1");
        assert!(o.exclusive && o.wait && !o.interrupt);
        assert_eq!(o.wait_timeout, LEASE_WAIT_TIMEOUT);
    }
}
