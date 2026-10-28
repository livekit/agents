//! The thread database's operations as plans: each operation is the statements of one atomic
//! batch (or one read) plus the typed decoding of its answer. The runtime executes a [`Plan`] on
//! a connection and hands the [`BatchResult`] back to its [`Decode`]; nothing here performs I/O,
//! reads a clock or mints an id: lease tokens, timestamps and run ids are supplied.

use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::StorageError;
use crate::run::TaskState;
use crate::schema::*;
use crate::sql::{BatchResult, Bind, FromCell, Row, SqlValue, Statement, first_cell};

/// The largest `tasks/list` page A2A allows.
const MAX_TASK_PAGE: i64 = 100;
/// The largest state, in bytes.
pub const STATE_LIMIT: usize = 64 * 1024;

/// A2A `tasks/list`'s filters and position.
#[derive(Debug, Clone)]
pub struct TaskQuery {
    pub status: Option<TaskState>,
    /// Unix milliseconds: only tasks updated after it.
    pub updated_after: i64,
    pub page_size: i64,
    /// The previous page's `next_page_token`; none for the first page.
    pub page_token: Option<String>,
}

/// One page of A2A `tasks/list`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_run", config = "frozen=True")]
pub struct TaskPage {
    pub tasks: Vec<TaskRow>,
    /// Empty on the last page.
    pub next_page_token: String,
    pub page_size: i64,
    /// Every page's tasks, in all.
    pub total_size: i64,
}

/// One journal entry a commit appends, stamped by the caller.
#[derive(Debug, Clone, Default)]
pub struct EventWrite {
    pub ts_ms: i64,
    pub r#type: String,
    pub payload: Value,
}

impl EventWrite {
    pub fn run_started(ts_ms: i64, p: &RunStarted) -> Self {
        let payload = serde_json::to_value(p).unwrap_or_default();
        Self { ts_ms, r#type: RUN_STARTED_EVENT.into(), payload }
    }

    pub fn run_ended(ts_ms: i64, p: &RunEnded) -> Self {
        let payload = serde_json::to_value(p).unwrap_or_default();
        Self { ts_ms, r#type: RUN_ENDED_EVENT.into(), payload }
    }
}

/// One task row a write upserts: every column but `lane`, `seq` and `updated_at`, which the
/// statement fills.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TaskWrite {
    pub task_id: String,
    pub status: TaskState,
    pub message_id: Option<String>,
    /// An optional application task label.
    pub name: Option<String>,
    pub data: TaskData,
    pub ask: Option<Ask>,
}

impl TaskWrite {
    /// The row as stored, `seq` and `updated_at` unknown until written.
    pub fn row(&self, lane: &str) -> TaskRow {
        TaskRow {
            task_id: self.task_id.clone(),
            lane: lane.to_owned(),
            seq: 0,
            name: self.name.clone(),
            message_id: self.message_id.clone(),
            status: self.status,
            updated_at: 0,
            data: self.data.clone(),
            internal: self.ask.clone(),
        }
    }
}

impl From<TaskRow> for TaskWrite {
    fn from(r: TaskRow) -> Self {
        Self {
            task_id: r.task_id,
            status: r.status,
            message_id: r.message_id,
            name: r.name,
            data: r.data,
            ask: r.internal,
        }
    }
}

/// The lane ownership a lease grants: who holds it, under which token, at which version. Every
/// managed write is fenced on `(version, token)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fence {
    pub lane: String,
    pub holder: String,
    pub token: String,
    pub version: i64,
}

/// Every write of a run, as one batch.
#[derive(Debug, Default)]
pub struct Commit {
    pub run_id: String,
    pub items: Vec<ItemRow>,
    pub delete_items: Vec<String>,
    pub events: Vec<EventWrite>,
    pub tasks: Vec<TaskWrite>,
    /// `None` keeps the stored snapshot.
    pub state: Option<Value>,
    pub drained_id: Option<i64>,
    pub run_ended: EventWrite,
    /// A task row to read back after the writes, in the same batch: the answer's record.
    pub read_task: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Committed {
    /// The version after the bump; the current one for an unfenced write.
    pub version: i64,
    /// `read_task`'s row after the commit.
    pub task: Option<TaskRow>,
}

/// Statements sent as one atomic batch, and how to read its answer.
pub struct Plan<D> {
    pub statements: Vec<Statement>,
    pub decoder: D,
}

/// The typed reading of one plan's batch answer.
pub trait Decode {
    type Output;
    fn decode(self, out: &BatchResult) -> Result<Self::Output, StorageError>;
}

/// A batch under construction; `add` answers the index of the statement's result.
#[derive(Default)]
struct Batch(Vec<Statement>);

impl Batch {
    fn add<S: Bind>(&mut self, s: S) -> usize {
        self.0.push(s.bind());
        self.0.len() - 1
    }

    fn plan<D>(self, decoder: D) -> Plan<D> {
        Plan { statements: self.0, decoder }
    }
}

/// One read statement and the decoding of each of its rows.
pub struct Read<T> {
    pub statement: Statement,
    decode_row: fn(&Row) -> Result<T, StorageError>,
}

impl<T> Read<T> {
    fn new<S: Bind>(s: S, decode_row: fn(&Row) -> Result<T, StorageError>) -> Self {
        Self { statement: s.bind(), decode_row }
    }

    pub fn decode(&self, rows: &[Row]) -> Result<Vec<T>, StorageError> {
        rows.iter().map(self.decode_row).collect()
    }
}

/// The DDL and the `schema_version` row of a database no SDK wrote yet.
pub fn bootstrap() -> Vec<Statement> {
    let mut out: Vec<Statement> =
        crate::sql::schema_statements(THREAD_SCHEMA).map(Statement::raw).collect();
    out.push(InsertSchemaVersion { version: THREAD_SCHEMA_VERSION }.bind());
    out
}

/// The schema probe.
pub fn schema_version() -> Statement {
    SelectSchemaVersion {}.bind()
}

/// The schema version a probe read: `None` before any SDK wrote it; a version this client does
/// not write is refused, behind or ahead.
pub fn decode_schema_version(rows: &[Row]) -> Result<Option<i64>, StorageError> {
    match first_cell(rows).and_then(SqlValue::as_i64) {
        Some(v) if v == THREAD_SCHEMA_VERSION => Ok(Some(v)),
        Some(v) => Err(StorageError::Schema(format!(
            "database schema is version {v}, this client writes {THREAD_SCHEMA_VERSION}"
        ))),
        None => Ok(None),
    }
}

/// What an acquire writes and loads besides the lease.
pub struct Acquire<'a> {
    pub lane: &'a str,
    pub holder: &'a str,
    /// The fresh lease token this attempt writes.
    pub token: &'a str,
    pub lease_secs: i64,
    /// How long a waiting `submitted` row keeps its place in the arrival order.
    pub wait_ms: i64,
    pub run_id: &'a str,
    /// The run's task: a `submitted` row of it orders the acquire behind older waiters; empty
    /// for a run with no row (a voice turn).
    pub task_id: &'a str,
    pub run_started: EventWrite,
    /// The turn's `submitted` row, written first in the same batch.
    pub submit: Option<&'a TaskWrite>,
    /// Take `task_id`'s row to `working` under the lease, in the same batch.
    pub pickup: bool,
    /// Load the lane's newest task rows, this many.
    pub tasks_limit: i64,
}

/// What an acquire loaded, under the lease it took.
#[derive(Debug)]
pub struct Acquired {
    pub fence: Fence,
    /// This `run_id` already committed; the caller releases without running.
    pub already_committed: bool,
    pub lane: LaneRow,
    /// The whole history, in the same snapshot as the lease, the state and the tasks.
    pub items: Vec<ItemRow>,
    /// The newest task rows, newest first.
    pub tasks: Vec<TaskRow>,
    /// Whether `submit` inserted its row; `Some(false)` is a `task_id` or `messageId` that
    /// already had one.
    pub submitted: Option<bool>,
    /// Whether `pickup` moved the row to `working`; `false` when it is terminal or missing.
    pub picked_up: bool,
}

/// The lane's outcome of an acquire attempt: taken, or held by another holder.
#[derive(Debug)]
pub enum AcquireOutcome {
    Acquired(Box<Acquired>),
    /// Not taken; `submitted` says whether this attempt's `submit` inserted its row.
    Held {
        submitted: Option<bool>,
    },
}

pub struct AcquireDecode {
    fence: Fence,
    submitted: Option<usize>,
    take: usize,
    committed: usize,
    pickup: Option<usize>,
    items: usize,
    lane: usize,
    tasks: usize,
}

/// Takes the lease and loads the lane, one atomic batch, `schema` prepended for a fresh
/// database. A retried `run_id` loads as `already_committed` and writes nothing.
pub fn acquire(req: &Acquire<'_>, schema: Vec<Statement>) -> Plan<AcquireDecode> {
    let (lane, token) = (req.lane, req.token);
    let mut b = Batch(schema);
    let submitted = req.submit.map(|task| b.add(submitted_task(lane, task)));
    b.add(InsertLane { lane });
    let take = b.add(TakeLease {
        lane,
        holder: req.holder,
        token,
        lease_secs: req.lease_secs,
        task_id: req.task_id,
        wait_ms: req.wait_ms,
    });
    b.add(ClearWaiter { lane, holder: req.holder, token });
    b.add(InsertRunStarted {
        lane,
        token,
        ts_ms: req.run_started.ts_ms,
        run_id: req.run_id,
        payload: SqlValue::json(&req.run_started.payload),
    });
    let committed = b.add(SelectCommittedRun { lane, run_id: req.run_id });
    let pickup = req.pickup.then(|| b.add(PickupTaskHeld { lane, token, task_id: req.task_id }));
    let items = b.add(SelectItems { lane });
    let lane_row = b.add(SelectLane { lane });
    let tasks = b.add(SelectTasks { lane, limit: req.tasks_limit });
    let fence = Fence {
        lane: lane.to_owned(),
        holder: req.holder.to_owned(),
        token: token.to_owned(),
        version: 0,
    };
    b.plan(AcquireDecode {
        fence,
        submitted,
        take,
        committed,
        pickup,
        items,
        lane: lane_row,
        tasks,
    })
}

impl Decode for AcquireDecode {
    type Output = AcquireOutcome;

    fn decode(self, out: &BatchResult) -> Result<AcquireOutcome, StorageError> {
        let submitted = match self.submitted {
            Some(i) => Some(out.at(i)?.rows_affected() > 0),
            None => None,
        };
        let Some(version) = out.at(self.take)?.first_int() else {
            return Ok(AcquireOutcome::Held { submitted }); // the reads are discarded
        };
        let lane_row = out
            .at(self.lane)?
            .rows()
            .first()
            .ok_or_else(|| StorageError::Decode("acquire: no lane row".into()))?;
        let picked_up = match self.pickup {
            Some(i) => out.at(i)?.rows_affected() > 0,
            None => false,
        };
        Ok(AcquireOutcome::Acquired(Box::new(Acquired {
            fence: Fence { version, ..self.fence },
            already_committed: !out.at(self.committed)?.rows().is_empty(),
            lane: LaneRow::from_row(lane_row)?,
            items: rows(out, self.items, ItemRow::from_row)?,
            tasks: rows(out, self.tasks, TaskRow::from_row)?,
            submitted,
            picked_up,
        })))
    }
}

fn rows<T>(
    out: &BatchResult,
    i: usize,
    decode: fn(&Row) -> Result<T, StorageError>,
) -> Result<Vec<T>, StorageError> {
    out.at(i)?.rows().iter().map(decode).collect()
}

/// What the inbox receive answers.
#[derive(Debug, Clone, PartialEq)]
pub enum Receive {
    /// The row went in; the lease's version when this worker took it and is the runner.
    Received(Option<i64>),
    /// The `messageId` was in the inbox already: nothing inserted; the task it belongs to.
    Duplicate(Option<String>),
}

pub struct ReceiveDecode {
    inbox: usize,
    take: usize,
    existing: usize,
}

/// The inbox receive, one atomic batch: the inbox row, the `submitted` task row and the lease
/// attempt under `token`.
pub fn receive(
    lane: &str,
    holder: &str,
    token: &str,
    lease_secs: i64,
    task_id: &str,
    message: &a2a::Message,
    data: &TaskData,
) -> Plan<ReceiveDecode> {
    let message_id = message.message_id.as_str();
    let mut b = Batch::default();
    b.add(InsertLane { lane });
    let inbox = b.add(InsertInbox { lane, task_id, message_id, message: SqlValue::json(message) });
    b.add(InsertSubmittedTaskForMessage { lane, task_id, message_id, data: SqlValue::json(data) });
    let take = b.add(TakeLeaseForMessage { lane, holder, token, lease_secs, message_id, task_id });
    let existing = b.add(SelectInboxTask { lane, message_id });
    b.plan(ReceiveDecode { inbox, take, existing })
}

impl Decode for ReceiveDecode {
    type Output = Receive;

    fn decode(self, out: &BatchResult) -> Result<Receive, StorageError> {
        let crate::sql::StatementResult::Exec { rows_affected, .. } = out.at(self.inbox)? else {
            return Err(StorageError::Protocol("receive: the inbox insert answered rows".into()));
        };
        if *rows_affected == 0 {
            let task_id = first_cell(out.at(self.existing)?.rows())
                .and_then(SqlValue::as_str)
                .map(str::to_owned);
            return Ok(Receive::Duplicate(task_id));
        }
        Ok(Receive::Received(out.at(self.take)?.first_int()))
    }
}

/// A turn's `submitted` row, before the lease is tried; changes nothing when its task or message
/// id already had one.
pub fn submit(lane: &str, task: &TaskWrite) -> Statement {
    submitted_task(lane, task).bind()
}

/// Records a worker waiting under `interrupt`, only while the lease is held.
pub fn set_waiter(lane: &str, holder: &str) -> Statement {
    SetWaiter { lane, holder }.bind()
}

/// Pushes the lease out while its token is still this one.
pub fn extend(fence: &Fence, lease_secs: i64) -> Statement {
    ExtendLease { lane: &fence.lane, holder: &fence.holder, token: &fence.token, lease_secs }.bind()
}

/// Releases without writing anything else but, when given, the `run_ended` event of `run_id`:
/// the path of a run that failed or was already committed.
pub fn release(fence: &Fence, ended: Option<(&str, &EventWrite)>) -> Vec<Statement> {
    let (lane, token) = (fence.lane.as_str(), fence.token.as_str());
    let mut b = Batch::default();
    if let Some((run_id, e)) = ended {
        let payload = SqlValue::json(&e.payload);
        b.add(InsertRunEnded { lane, token, ts_ms: e.ts_ms, run_id, payload });
    }
    b.add(ReleaseLease { lane, token });
    b.0
}

/// Releases when no inbox row is pending past `drained_id`; one row changed means released.
pub fn release_if_drained(fence: &Fence) -> Statement {
    ReleaseLeaseIfDrained { lane: &fence.lane, token: &fence.token }.bind()
}

pub struct CheckpointDecode {
    waiter: usize,
    status: usize,
}

/// A turn's checkpoint, one batch: the lane's waiter and the task's status.
pub fn checkpoint(lane: &str, task_id: &str) -> Plan<CheckpointDecode> {
    let mut b = Batch::default();
    let waiter = b.add(SelectWaiter { lane });
    let status = b.add(SelectTaskStatus { lane, task_id });
    b.plan(CheckpointDecode { waiter, status })
}

impl Decode for CheckpointDecode {
    type Output = (Option<String>, Option<TaskState>);

    fn decode(self, out: &BatchResult) -> Result<Self::Output, StorageError> {
        let waiter =
            first_cell(out.at(self.waiter)?.rows()).and_then(SqlValue::as_str).map(str::to_owned);
        let status = first_cell(out.at(self.status)?.rows())
            .map(TaskState::from_cell)
            .transpose()
            .map_err(StorageError::Decode)?;
        Ok((waiter, status))
    }
}

pub struct PickupDecode {
    before: usize,
    moved: usize,
}

/// Takes a `submitted` or `working` row to `working`, one atomic batch.
pub fn pickup(lane: &str, task_id: &str) -> Plan<PickupDecode> {
    let mut b = Batch::default();
    let before = b.add(SelectPickupStatus { lane, task_id });
    let moved = b.add(PickupTask { lane, task_id });
    b.plan(PickupDecode { before, moved })
}

impl Decode for PickupDecode {
    /// The status the row had (`working` means a dead holder had it); `None` when it is
    /// terminal or missing.
    type Output = Option<TaskState>;

    fn decode(self, out: &BatchResult) -> Result<Option<TaskState>, StorageError> {
        if out.at(self.moved)?.rows_affected() == 0 {
            return Ok(None);
        }
        first_cell(out.at(self.before)?.rows())
            .map(TaskState::from_cell)
            .transpose()
            .map_err(StorageError::Decode)
    }
}

pub struct CancelDecode {
    row: usize,
}

/// `tasks/cancel`, one batch: the conditional transition, then the row as it is after.
pub fn cancel(lane: &str, task_id: &str, metadata: &Map<String, Value>) -> Plan<CancelDecode> {
    let mut b = Batch::default();
    b.add(CancelTask { lane, task_id, metadata: SqlValue::json(metadata) });
    let row = b.add(SelectTask { lane, task_id });
    b.plan(CancelDecode { row })
}

impl Decode for CancelDecode {
    /// The row after: `canceled`, or unchanged when it was already terminal; `None` when the
    /// lane has no such task.
    type Output = Option<TaskRow>;

    fn decode(self, out: &BatchResult) -> Result<Option<TaskRow>, StorageError> {
        out.at(self.row)?.rows().first().map(TaskRow::from_row).transpose()
    }
}

/// A `working` row whose holder died ends `failed` with the given status message.
pub fn fail_orphan(lane: &str, task_id: &str, status_message: &a2a::Message) -> Statement {
    let status_message = SqlValue::json(status_message);
    FailOrphan { lane, task_id, status_message }.bind()
}

/// A lease and how long it lasts, for a fenced commit.
pub struct Fenced<'a> {
    pub fence: &'a Fence,
    /// Extend the lease by `lease_secs` instead of releasing it.
    pub keep: Option<i64>,
}

pub struct CommitDecode {
    fence: Option<Fence>,
    lane: String,
    lane_row: usize,
    read_task: Option<usize>,
}

/// A run's writes, one batch: fenced on the lease (and renewing it when kept), else
/// append-only. `schema` is prepended for a fresh database. A state over [`STATE_LIMIT`] is
/// refused before anything is sent.
pub fn commit(
    lane: &str,
    c: &Commit,
    fenced: Option<Fenced<'_>>,
    schema: Vec<Statement>,
) -> Result<Plan<CommitDecode>, StorageError> {
    if let Some(state) = &c.state {
        let size = serde_json::to_vec(state)
            .map_err(|e| StorageError::NotSerializable("state".into(), e.to_string()))?
            .len();
        if size > STATE_LIMIT {
            return Err(StorageError::TooLarge { what: "state".into(), limit: STATE_LIMIT });
        }
    }
    let mut b = Batch(schema);
    match &fenced {
        Some(Fenced { fence, .. }) => {
            let (version, token) = (fence.version, fence.token.as_str());
            for item in &c.items {
                b.add(InsertItem {
                    lane,
                    version,
                    token,
                    id: &item.id,
                    message_id: item.message_id.as_deref(),
                    payload: SqlValue::json(&item.payload),
                });
            }
            for id in &c.delete_items {
                b.add(DeleteItem { lane, version, token, id });
            }
            for e in c.events.iter().chain([&c.run_ended]) {
                b.add(InsertEvent {
                    lane,
                    version,
                    token,
                    ts_ms: e.ts_ms,
                    run_id: &c.run_id,
                    event_type: &e.r#type,
                    payload: SqlValue::json(&e.payload),
                });
            }
            for t in &c.tasks {
                b.add(UpsertTask {
                    lane,
                    version,
                    token,
                    task_id: &t.task_id,
                    name: t.name.as_deref(),
                    message_id: t.message_id.as_deref(),
                    status: t.status.as_str(),
                    data: SqlValue::json(&t.data),
                    internal: SqlValue::opt_json(t.ask.as_ref()),
                });
            }
        }
        None => {
            // no lease was taken, so nothing created the lane row yet
            b.add(InsertLane { lane });
            for item in &c.items {
                b.add(InsertItemUnfenced {
                    lane,
                    id: &item.id,
                    message_id: item.message_id.as_deref(),
                    payload: SqlValue::json(&item.payload),
                });
            }
            for id in &c.delete_items {
                b.add(DeleteItemUnfenced { lane, id });
            }
            for e in c.events.iter().chain([&c.run_ended]) {
                b.add(InsertEventUnfenced {
                    lane,
                    ts_ms: e.ts_ms,
                    run_id: &c.run_id,
                    event_type: &e.r#type,
                    payload: SqlValue::json(&e.payload),
                });
            }
            for t in &c.tasks {
                b.add(UpsertTaskUnfenced {
                    lane,
                    task_id: &t.task_id,
                    name: t.name.as_deref(),
                    message_id: t.message_id.as_deref(),
                    status: t.status.as_str(),
                    data: SqlValue::json(&t.data),
                    internal: SqlValue::opt_json(t.ask.as_ref()),
                });
            }
        }
    }
    let lane_row = match &fenced {
        Some(Fenced { fence, keep }) => {
            let (version, token) = (fence.version, fence.token.as_str());
            let state = SqlValue::opt_json(c.state.as_ref());
            let drained_id = c.drained_id;
            match *keep {
                Some(lease_secs) => {
                    b.add(CommitAndExtend { lane, version, token, state, drained_id, lease_secs })
                }
                None => b.add(CommitAndRelease { lane, version, token, state, drained_id }),
            }
        }
        None => b.add(SelectLane { lane }),
    };
    let read_task = c.read_task.as_ref().map(|id| b.add(SelectTask { lane, task_id: id }));
    let fence = fenced.map(|f| f.fence.clone());
    Ok(b.plan(CommitDecode { fence, lane: lane.to_owned(), lane_row, read_task }))
}

impl Decode for CommitDecode {
    type Output = Committed;

    /// `VersionConflict` when the fenced run outlived its lease: nothing was written.
    fn decode(self, out: &BatchResult) -> Result<Committed, StorageError> {
        let version = match self.fence {
            Some(_) => out.at(self.lane_row)?.first_int(),
            None => out
                .at(self.lane_row)?
                .rows()
                .first()
                .map(LaneRow::from_row)
                .transpose()?
                .map(|l| l.version),
        };
        let task = match self.read_task {
            Some(i) => out.at(i)?.rows().first().map(TaskRow::from_row).transpose()?,
            None => None,
        };
        match (self.fence, version) {
            (_, Some(version)) => Ok(Committed { version, task }),
            (Some(fence), None) => {
                Err(StorageError::VersionConflict { lane: self.lane, expected: fence.version })
            }
            (None, None) => Err(StorageError::Decode("commit: no lane row".into())),
        }
    }
}

/// What a commit whose acknowledgment was lost turned out to be.
#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    /// The run's `run_ended` is there: it committed, making `version`, with `task` read back.
    Committed(Committed),
    /// Not committed, and the fence still holds: sending the same commit again is safe.
    NotApplied,
    /// Not committed, and the lane moved on without it: its staged writes are absent.
    Rejected,
}

pub struct ResolveDecode {
    fence: Option<Fence>,
    committed: usize,
    lane: usize,
    task: Option<usize>,
}

/// Looks up the authoritative record of `run_id` after a lost acknowledgment, one batch: its
/// `run_ended`, the lane's fence, and the answer task.
pub fn resolve_commit(
    lane: &str,
    run_id: &str,
    fence: Option<&Fence>,
    read_task: Option<&str>,
) -> Plan<ResolveDecode> {
    let mut b = Batch::default();
    let committed = b.add(SelectCommittedRun { lane, run_id });
    let lane_fence = b.add(SelectFence { lane });
    let task = read_task.map(|task_id| b.add(SelectTask { lane, task_id }));
    b.plan(ResolveDecode { fence: fence.cloned(), committed, lane: lane_fence, task })
}

impl Decode for ResolveDecode {
    type Output = Resolution;

    fn decode(self, out: &BatchResult) -> Result<Resolution, StorageError> {
        let row = out.at(self.lane)?.rows().first();
        let version = row.and_then(|r| r.first()).and_then(SqlValue::as_i64).unwrap_or(0);
        let token = row.and_then(|r| r.get(1)).and_then(SqlValue::as_str);
        if !out.at(self.committed)?.rows().is_empty() {
            let task = match self.task {
                Some(i) => out.at(i)?.rows().first().map(TaskRow::from_row).transpose()?,
                None => None,
            };
            // the version this run's commit made, not the lane's now: others may have
            // committed since
            let version = self.fence.as_ref().map_or(version, |f| f.version + 1);
            return Ok(Resolution::Committed(Committed { version, task }));
        }
        Ok(match self.fence {
            // an append-only batch that did not land can be sent again
            None => Resolution::NotApplied,
            Some(f) if f.version == version && token == Some(f.token.as_str()) => {
                Resolution::NotApplied
            }
            Some(_) => Resolution::Rejected,
        })
    }
}

pub struct ListTasksDecode {
    page: usize,
    count: usize,
    page_size: i64,
}

/// One page of the thread's tasks, last updated first, as A2A `tasks/list` answers it.
pub fn list_tasks(q: &TaskQuery) -> Result<Plan<ListTasksDecode>, StorageError> {
    let (before_at, before_id) = match q.page_token.as_deref().filter(|t| !t.is_empty()) {
        None => (i64::MAX, ""),
        Some(token) => token
            .split_once(':')
            .and_then(|(at, id)| Some((at.parse().ok()?, id)))
            .ok_or_else(|| StorageError::Invalid(format!("page token {token:?}")))?,
    };
    let (status, updated_after) = (q.status.map(TaskState::as_str), q.updated_after);
    let page_size = q.page_size.clamp(1, MAX_TASK_PAGE);
    let mut b = Batch::default();
    let page = b.add(SelectThreadTasksPage {
        status,
        updated_after,
        before_at,
        before_id,
        limit: page_size + 1,
    });
    let count = b.add(CountThreadTasks { status, updated_after });
    Ok(b.plan(ListTasksDecode { page, count, page_size }))
}

impl Decode for ListTasksDecode {
    type Output = TaskPage;

    fn decode(self, out: &BatchResult) -> Result<TaskPage, StorageError> {
        let mut tasks = rows(out, self.page, TaskRow::from_row)?;
        let more = tasks.len() as i64 > self.page_size;
        tasks.truncate(self.page_size as usize);
        let next_page_token = match tasks.last() {
            Some(last) if more => format!("{}:{}", last.updated_at, last.task_id),
            _ => String::new(),
        };
        let total_size = out.at(self.count)?.first_int().unwrap_or(0);
        Ok(TaskPage { tasks, next_page_token, page_size: self.page_size, total_size })
    }
}

/// One page of the thread's items: in insertion order with their lane, and
/// the cursor of the next page when there is one.
#[derive(Debug, Clone, PartialEq)]
pub struct ItemPage {
    pub items: Vec<(String, ItemRow)>,
    pub next: Option<String>,
}

/// The largest item page.
const MAX_ITEM_PAGE: i64 = 500;

/// A page cursor: opaque to callers, bound to the lane filter it was made under.
fn item_cursor(lane: Option<&str>, pos: i64) -> String {
    format!("{}:{pos}", lane.unwrap_or("*"))
}

pub struct ItemPageDecode {
    lane: Option<String>,
    limit: i64,
}

/// One page of the thread's items after `cursor`, of `lane` when given. A cursor made under
/// another filter is refused.
pub fn list_items(
    lane: Option<&str>,
    cursor: Option<&str>,
    limit: i64,
) -> Result<Plan<ItemPageDecode>, StorageError> {
    let after = match cursor {
        None => 0,
        Some(c) => c
            .rsplit_once(':')
            .filter(|(scope, _)| *scope == lane.unwrap_or("*"))
            .and_then(|(_, pos)| pos.parse().ok())
            .ok_or_else(|| StorageError::Invalid(format!("item cursor {c:?} for this listing")))?,
    };
    let limit = limit.clamp(1, MAX_ITEM_PAGE);
    let mut b = Batch::default();
    b.add(SelectItemsAfter { lane, after, limit: limit + 1 });
    Ok(b.plan(ItemPageDecode { lane: lane.map(str::to_owned), limit }))
}

impl Decode for ItemPageDecode {
    type Output = ItemPage;

    fn decode(self, out: &BatchResult) -> Result<ItemPage, StorageError> {
        let mut items = Vec::new();
        let mut last = 0;
        for row in out.at(0)?.rows() {
            let pos = row.first().and_then(SqlValue::as_i64);
            let lane = row.get(1).and_then(SqlValue::as_str);
            let (Some(pos), Some(lane)) = (pos, lane) else {
                return Err(StorageError::Decode("an item page row".into()));
            };
            if items.len() as i64 == self.limit {
                return Ok(ItemPage { items, next: Some(item_cursor(self.lane.as_deref(), last)) });
            }
            items.push((lane.to_owned(), ItemRow::from_row(&row[2..].to_vec())?));
            last = pos;
        }
        Ok(ItemPage { items, next: None })
    }
}

/// The journal, every lane, after the event `after_id`.
pub fn events(after_id: i64, limit: i64) -> Read<EventRow> {
    Read::new(SelectEvents { after_id, limit }, EventRow::from_row)
}

/// One task, whichever lane it is on.
pub fn thread_task(task_id: &str) -> Read<TaskRow> {
    Read::new(SelectThreadTask { task_id }, TaskRow::from_row)
}

pub fn lanes() -> Read<LaneRow> {
    Read::new(SelectLanes {}, LaneRow::from_row)
}

/// The lane's whole history.
pub fn history(lane: &str) -> Read<ItemRow> {
    Read::new(SelectItems { lane }, ItemRow::from_row)
}

pub fn lane_info(lane: &str) -> Read<LaneRow> {
    Read::new(SelectLane { lane }, LaneRow::from_row)
}

/// The newest task rows, of one status when given.
pub fn tasks(lane: &str, status: Option<TaskState>, limit: i64) -> Read<TaskRow> {
    match status {
        Some(s) => {
            Read::new(SelectTasksByStatus { lane, status: s.as_str(), limit }, TaskRow::from_row)
        }
        None => Read::new(SelectTasks { lane, limit }, TaskRow::from_row),
    }
}

pub fn task(lane: &str, task_id: &str) -> Read<TaskRow> {
    Read::new(SelectTask { lane, task_id }, TaskRow::from_row)
}

/// The task an A2A `messageId` created.
pub fn task_for_message(lane: &str, message_id: &str) -> Read<TaskRow> {
    Read::new(SelectTaskForMessage { lane, message_id }, TaskRow::from_row)
}

/// The inbox rows in `id` order: every one, or only those above `drained_id`.
pub fn inbox(lane: &str, pending: bool) -> Read<InboxRow> {
    match pending {
        true => Read::new(SelectInboxPending { lane }, InboxRow::from_row),
        false => Read::new(SelectInbox { lane }, InboxRow::from_row),
    }
}

/// The lane's journal after the event `after_id`.
pub fn lane_events(lane: &str, after_id: i64) -> Read<EventRow> {
    Read::new(SelectLaneEvents { lane, after_id }, EventRow::from_row)
}

fn submitted_task<'a>(lane: &'a str, task: &'a TaskWrite) -> InsertSubmittedTask<'a> {
    InsertSubmittedTask {
        lane,
        task_id: &task.task_id,
        name: task.name.as_deref(),
        message_id: task.message_id.as_deref(),
        data: SqlValue::json(&task.data),
        internal: SqlValue::opt_json(task.ask.as_ref()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::StatementResult;

    fn fence() -> Fence {
        Fence {
            lane: "front-desk".into(),
            holder: "w1".into(),
            token: "lease_a".into(),
            version: 3,
        }
    }

    /// The lookup's answer: whether `run_ended` is there, and the lane's version and token.
    fn answer(committed: bool, version: i64, token: Option<&str>) -> BatchResult {
        let ended = if committed { vec![vec![SqlValue::Int(1)]] } else { vec![] };
        let lane = vec![vec![SqlValue::Int(version), token.map_or(SqlValue::Null, Into::into)]];
        BatchResult { results: vec![StatementResult::Rows(ended), StatementResult::Rows(lane)] }
    }

    fn resolve(fence: Option<&Fence>, out: BatchResult) -> Resolution {
        resolve_commit("front-desk", "r1", fence, None).decoder.decode(&out).unwrap()
    }

    // Decision 39: a lost acknowledgment is decided by the run's own record, never guessed.
    #[test]
    fn a_lost_commit_resolves_by_its_run_ended_and_the_fence() {
        let f = fence();
        let committed = Resolution::Committed(Committed { version: 4, task: None });
        assert_eq!(resolve(Some(&f), answer(true, 4, None)), committed);
        // found by the storage DST: the lane moved on after the lost commit; the run still
        // reports the version its own commit made
        assert_eq!(resolve(Some(&f), answer(true, 9, Some("lease_c"))), committed);
        assert_eq!(resolve(Some(&f), answer(false, 3, Some("lease_a"))), Resolution::NotApplied);
        assert_eq!(resolve(Some(&f), answer(false, 3, Some("lease_b"))), Resolution::Rejected);
        assert_eq!(resolve(Some(&f), answer(false, 4, None)), Resolution::Rejected);
        assert_eq!(resolve(None, answer(false, 0, None)), Resolution::NotApplied);
        // an answer missing a statement's result is malformed, not an empty success
        let short = BatchResult { results: vec![StatementResult::Rows(vec![])] };
        assert!(resolve_commit("front-desk", "r1", Some(&f), None).decoder.decode(&short).is_err());
    }
}
