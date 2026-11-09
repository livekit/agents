use lk_agents_abi::storage::context::{AppStatement, ContextScope, ContextScopeWrite};
use lk_agents_abi::storage::journal::JournalEvent;
pub use lk_agents_abi::storage::thread::TaskOrder;
use serde_json::{Map, Value};

use crate::cursor::Cursor;
use crate::sql::{BatchResult, Bind, Cells, FromCell, Row, SqlValue, Statement, Table, first_cell};
use crate::thread::run::TaskState;
use crate::thread::schema::*;
use crate::{StorageError, TaskPickup, TaskWrite, ThreadDbConnection};

/// The largest `tasks/list` page A2A allows.
const MAX_TASK_PAGE: i64 = 100;
/// The largest state, in bytes.
pub const STATE_LIMIT: usize = 64 * 1024;

/// A task listing's filters, order and position: A2A `tasks/list`, or browsing a thread.
#[derive(Debug, Clone, Default)]
pub struct TaskQuery {
    /// Only this lane's tasks when given.
    pub lane: Option<String>,
    pub status: Option<TaskState>,
    pub order: TaskOrder,
    /// Unix milliseconds: only tasks updated after it.
    pub updated_after: i64,
    pub page_size: i64,
    /// The previous page's `next_page_token`; none for the first page.
    pub page_token: Option<String>,
}

/// One page of a task listing.
#[derive(Debug, Clone, Default, PartialEq)]
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
    pub fn new(ts_ms: i64, event: JournalEvent) -> Self {
        let (r#type, payload) = event.into_row();
        Self { ts_ms, r#type, payload }
    }
}

/// The lane ownership a lease grants: who holds it, under which token, at which version. Every
/// managed write is fenced on `(version, token)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fence {
    pub lane: String,
    pub holder: crate::Holder,
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
    /// The unchanged routing snapshot still counts toward the aggregate state budget.
    pub loaded_state: Option<Value>,
    /// The lane's loaded scopes; staged replacements are substituted before size accounting.
    pub context_scopes: Vec<ContextScope>,
    pub scope_writes: Vec<ContextScopeWrite>,
    /// Application statements a managed turn staged: applied first, and only while the fence
    /// holds.
    pub app_writes: Vec<AppStatement>,
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

/// A batch under construction; `add` answers the index of the statement's result.
#[derive(Default)]
struct Batch(Vec<Statement>);

impl Batch {
    fn add<S: Bind>(&mut self, s: S) -> usize {
        self.0.push(s.bind());
        self.0.len() - 1
    }
}

/// The DDL and the `schema_version` row of a database no SDK wrote yet.
pub fn bootstrap() -> Vec<Statement> {
    let mut out: Vec<Statement> =
        crate::sql::schema_statements(THREAD_SCHEMA).map(Statement::ddl).collect();
    out.push(InsertSchemaVersion { version: THREAD_SCHEMA_VERSION }.bind());
    out.extend(framework_schema());
    out
}

/// [`FRAMEWORK_SCHEMA`]'s statements.
pub(crate) fn framework_schema() -> impl Iterator<Item = Statement> {
    crate::sql::schema_statements(FRAMEWORK_SCHEMA).map(Statement::ddl)
}

/// The schema version the database holds, `None` before any SDK wrote it; the catalog is probed
/// first, so a missing table is never provoked. A version this client does not write is
/// refused, behind or ahead.
pub(crate) async fn schema_version(
    conn: &dyn ThreadDbConnection,
) -> Result<Option<i64>, StorageError> {
    let catalog = conn.query(SelectSchemaExists {}.bind()).await?;
    match catalog.as_slice() {
        [row] if row.as_slice() == [SqlValue::Int(0)] => return Ok(None),
        [row] if row.as_slice() == [SqlValue::Int(1)] => {}
        _ => {
            return Err(StorageError::Invalid(
                "schema catalog probe did not return one boolean".into(),
            ));
        }
    }
    let rows = conn.query(SelectSchemaVersion {}.bind()).await?;
    match first_cell(&rows).and_then(SqlValue::as_i64) {
        Some(v) if v == THREAD_SCHEMA_VERSION => Ok(Some(v)),
        Some(v) => Err(StorageError::Invalid(format!(
            "database schema is version {v}, this client writes {THREAD_SCHEMA_VERSION}"
        ))),
        None => Err(StorageError::Invalid("schema_version has no valid version".into())),
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
    /// What the batch does with `task_id`'s row under the lease.
    pub pickup: TaskPickup,
    /// Load the lane's newest task rows, this many.
    pub tasks_limit: i64,
}

/// What an acquire loaded, under the lease it took.
#[derive(Debug)]
pub struct Acquired {
    /// The lane's version under the lease taken.
    pub version: i64,
    /// This `run_id` already committed; the caller releases without running.
    pub already_committed: bool,
    pub lane: LaneRow,
    /// The whole history, in the same snapshot as the lease, the state and the tasks.
    pub items: Vec<ItemRow>,
    /// The newest task rows, newest first.
    pub tasks: Vec<TaskRow>,
    pub context_scopes: Vec<ContextScope>,
    /// Whether `pickup` moved the row to `working`; `false` when it is terminal or missing.
    pub picked_up: bool,
}

/// The lease a lane had when an acquire tried it, on the database's clock (unix seconds).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LeaseBefore {
    pub holder: Option<String>,
    pub token: Option<String>,
    pub until: i64,
    pub now: i64,
}

impl LeaseBefore {
    /// Another holder's lease that lapsed without a release.
    pub(crate) fn lapsed(&self, holder: &str) -> bool {
        self.token.is_some() && self.holder.as_deref() != Some(holder) && self.until < self.now
    }

    /// Another holder's live lease, not renewed for longer than `stale` (`ttl` the lease length).
    pub(crate) fn ownerless(&self, holder: &str, ttl: i64, stale: std::time::Duration) -> bool {
        let renewed_ago = (self.now - (self.until - ttl)) as f64;
        self.token.is_some()
            && self.holder.as_deref() != Some(holder)
            && self.until >= self.now
            && renewed_ago > stale.as_secs_f64()
    }
}

/// Takes the lease and loads the lane, one atomic batch, `schema` prepended for a fresh
/// database; `None` while another holder has the lane. A retried `run_id` loads as
/// `already_committed` and writes nothing.
pub(crate) async fn acquire(
    conn: &dyn ThreadDbConnection,
    req: &Acquire<'_>,
    schema: Vec<Statement>,
) -> Result<(Option<Acquired>, LeaseBefore), StorageError> {
    let (lane, token) = (req.lane, req.token);
    let mut b = Batch(schema);
    b.0.extend(framework_schema());
    b.add(PruneAppliedWrites {});
    b.add(InsertLane { lane });
    let before = b.add(SelectLeaseBefore { lane });
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
    let pickup = (req.pickup == TaskPickup::Claim)
        .then(|| b.add(PickupTaskHeld { lane, token, task_id: req.task_id }));
    let items = b.add(SelectItems { lane });
    let lane_row = b.add(SelectLane { lane });
    let tasks = b.add(SelectTasks { lane, limit: req.tasks_limit });
    b.0.push(crate::thread::schema::SelectScopes { lane }.bind());
    let context_scopes = b.0.len() - 1;
    let out = conn.batch(b.0).await?;
    let before = match out.at(before)?.rows().first() {
        Some(row) => LeaseBefore {
            holder: row[0].as_str().map(str::to_owned),
            token: row[1].as_str().map(str::to_owned),
            until: row[2].as_i64().unwrap_or(0),
            now: row[3].as_i64().unwrap_or(0),
        },
        None => LeaseBefore::default(),
    };

    let Some(version) = out.at(take)?.first_int() else {
        return Ok((None, before)); // the reads are discarded
    };
    let lane_row = out
        .at(lane_row)?
        .rows()
        .first()
        .ok_or_else(|| StorageError::Invalid("acquire: no lane row".into()))?;
    let picked_up = match pickup {
        Some(i) => out.at(i)?.rows_affected() > 0,
        None => false,
    };
    let acquired = Acquired {
        version,
        already_committed: !out.at(committed)?.rows().is_empty(),
        lane: LaneRow::from_row(lane_row)?,
        items: rows(&out, items, ItemRow::from_row)?,
        tasks: rows(&out, tasks, TaskRow::from_row)?,
        context_scopes: crate::context::scopes_from_rows(out.at(context_scopes)?.rows())?,
        picked_up,
    };
    Ok((Some(acquired), before))
}

fn rows<T>(
    out: &BatchResult,
    i: usize,
    decode: fn(&[SqlValue]) -> Result<T, StorageError>,
) -> Result<Vec<T>, StorageError> {
    out.at(i)?.rows().iter().map(|row| decode(row)).collect()
}

/// What the inbox receive answers.
#[derive(Debug, Clone, PartialEq)]
pub enum Receive {
    /// The row went in; the lease's version when this worker took it and is the runner.
    Received(Option<i64>),
    /// The `messageId` was in the inbox already: nothing inserted; the task it belongs to.
    Duplicate(Option<String>),
}

/// The message a receive inserts, with its `submitted` task row.
#[derive(Clone, Copy)]
pub struct ReceiveInput<'a> {
    pub task_id: &'a str,
    pub message: &'a a2a::Message,
    pub data: &'a TaskData,
    pub internal: Option<&'a TaskInternal>,
}

/// The inbox receive, one atomic batch: the inbox row, the `submitted` task row and the lease
/// attempt under `token`.
pub(crate) async fn receive(
    conn: &dyn ThreadDbConnection,
    lane: &str,
    holder: &str,
    token: &str,
    lease_secs: i64,
    input: ReceiveInput<'_>,
) -> Result<Receive, StorageError> {
    let ReceiveInput { task_id, message, data, internal } = input;
    let message_id = message.message_id.as_str();
    let mut b = Batch::default();
    b.add(InsertLane { lane });
    let inbox = b.add(InsertInbox { lane, task_id, message_id, message: SqlValue::json(message) });
    b.add(InsertSubmittedTaskForMessage {
        lane,
        task_id,
        message_id,
        data: SqlValue::json(data),
        internal: internal.map(SqlValue::json).unwrap_or(SqlValue::Null),
    });
    let take = b.add(TakeLeaseForMessage { lane, holder, token, lease_secs, message_id, task_id });
    let existing = b.add(SelectInboxTask { lane, message_id });
    let out = conn.batch(b.0).await?;

    let crate::sql::StatementResult::Exec { rows_affected, .. } = out.at(inbox)? else {
        return Err(StorageError::Invalid("receive: the inbox insert answered rows".into()));
    };
    if *rows_affected == 0 {
        let task_id =
            first_cell(out.at(existing)?.rows()).and_then(SqlValue::as_str).map(str::to_owned);
        return Ok(Receive::Duplicate(task_id));
    }
    Ok(Receive::Received(out.at(take)?.first_int()))
}

/// A turn's checkpoint, one batch: the lane's waiter and the task's status.
pub(crate) async fn checkpoint(
    conn: &dyn ThreadDbConnection,
    lane: &str,
    task_id: &str,
) -> Result<(Option<String>, Option<TaskState>), StorageError> {
    let mut b = Batch::default();
    let waiter = b.add(SelectWaiter { lane });
    let status = b.add(SelectTaskStatus { lane, task_id });
    let out = super::read_batch(conn, b.0).await?;
    let waiter = first_cell(out.at(waiter)?.rows()).and_then(SqlValue::as_str).map(str::to_owned);
    let status = first_cell(out.at(status)?.rows()).map(TaskState::from_cell).transpose()?;
    Ok((waiter, status))
}

/// Takes a `submitted` or `working` row to `working`, one atomic batch. The status the row had
/// (`working` means a dead holder had it); `None` when it is terminal or missing.
pub(crate) async fn pickup(
    conn: &dyn ThreadDbConnection,
    lane: &str,
    task_id: &str,
) -> Result<Option<TaskState>, StorageError> {
    let mut b = Batch::default();
    let before = b.add(SelectPickupStatus { lane, task_id });
    let moved = b.add(PickupTask { lane, task_id });
    let out = conn.batch(b.0).await?;
    if out.at(moved)?.rows_affected() == 0 {
        return Ok(None);
    }
    first_cell(out.at(before)?.rows()).map(TaskState::from_cell).transpose()
}

/// `tasks/cancel`, one batch: the conditional transition, then the row as it is after:
/// `canceled`, or unchanged when it was already terminal; `None` when the lane has no such task.
pub(crate) async fn cancel(
    conn: &dyn ThreadDbConnection,
    lane: &str,
    task_id: &str,
    metadata: &Map<String, Value>,
) -> Result<Option<TaskRow>, StorageError> {
    let mut b = Batch::default();
    b.add(CancelTask { lane, task_id, metadata: SqlValue::json(metadata) });
    let row = b.add(SelectTask { lane, task_id });
    let out = conn.batch(b.0).await?;
    out.at(row)?.rows().first().map(|row| TaskRow::from_row(row)).transpose()
}

/// A lease and how long it lasts, for a fenced commit.
pub struct Fenced<'a> {
    pub fence: &'a Fence,
    /// Extend the lease by `lease_secs` instead of releasing it.
    pub keep: Option<i64>,
}

/// A run's writes, one batch: fenced on the lease (and renewing it when kept), else
/// append-only. A state over [`STATE_LIMIT`] is refused before anything is sent.
/// `VersionConflict` when the fenced run outlived its lease: nothing was written.
pub(crate) async fn commit(
    conn: &dyn ThreadDbConnection,
    lane: &str,
    c: &Commit,
    fenced: Option<Fenced<'_>>,
) -> Result<Committed, StorageError> {
    let mut scopes: std::collections::BTreeMap<_, _> =
        c.context_scopes.iter().map(|scope| (&scope.scope_id, scope)).collect();
    for write in &c.scope_writes {
        scopes.insert(&write.scope.scope_id, &write.scope);
    }
    crate::context::check_state_size(
        c.state.as_ref().or(c.loaded_state.as_ref()),
        scopes.into_values(),
    )?;
    if fenced.is_none() && !c.scope_writes.is_empty() {
        return Err(StorageError::Invalid(
            "context writes require an exclusive managed turn".into(),
        ));
    }
    let fence = fenced.as_ref().map(|fenced| fenced.fence);
    let mut b = Batch::default();
    add_app_writes(&mut b, lane, c, fence)?;
    match fence {
        Some(fence) => add_fenced_rows(&mut b, lane, fence, c)?,
        None => add_unfenced_rows(&mut b, lane, c),
    }
    let reads = add_lane_commit(&mut b, lane, c, fenced.as_ref());
    let out = conn.batch(b.0).await?;
    reads.committed(&out, lane, fence)
}

/// The application's own statements, first, behind a fence check when the run holds one.
fn add_app_writes(
    b: &mut Batch,
    lane: &str,
    c: &Commit,
    fence: Option<&Fence>,
) -> Result<(), StorageError> {
    if c.app_writes.is_empty() {
        return Ok(());
    }
    if let Some(fence) = fence {
        // arbitrary SQL cannot be made conditional on the fence: abort the batch instead
        b.add(AssertFence { lane, version: fence.version, token: fence.token.as_str() });
    }
    for write in &c.app_writes {
        b.0.push(crate::context::transaction::statement(write)?);
    }
    Ok(())
}

/// Context writes, items, deletions, events (`run_ended` last) and tasks, each on the fence.
fn add_fenced_rows(
    b: &mut Batch,
    lane: &str,
    fence: &Fence,
    c: &Commit,
) -> Result<(), StorageError> {
    let (version, token) = (fence.version, fence.token.as_str());
    for write in &c.scope_writes {
        b.0.push(crate::context::scope_write(lane, write, Some(fence))?);
    }
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
            internal: SqlValue::opt_json(t.internal.as_ref()),
        });
    }
    Ok(())
}

/// The append-only rows of a run without a lease, after creating the lane row. Each is guarded on
/// the run's `run_ended`, written last: a commit re-sent after a lost acknowledgment, or applied
/// twice, writes nothing the second time.
fn add_unfenced_rows(b: &mut Batch, lane: &str, c: &Commit) {
    // no lease was taken, so nothing created the lane row yet
    b.add(InsertLane { lane });
    let run_id = c.run_id.as_str();
    for item in &c.items {
        b.add(InsertItemUnfenced {
            lane,
            run_id,
            id: &item.id,
            message_id: item.message_id.as_deref(),
            payload: SqlValue::json(&item.payload),
        });
    }
    for id in &c.delete_items {
        b.add(DeleteItemUnfenced { lane, run_id, id });
    }
    for t in &c.tasks {
        b.add(UpsertTaskUnfenced {
            lane,
            run_id,
            task_id: &t.task_id,
            name: t.name.as_deref(),
            message_id: t.message_id.as_deref(),
            status: t.status.as_str(),
            data: SqlValue::json(&t.data),
            internal: SqlValue::opt_json(t.internal.as_ref()),
        });
    }
    for e in c.events.iter().chain([&c.run_ended]) {
        b.add(InsertEventUnfenced {
            lane,
            ts_ms: e.ts_ms,
            run_id,
            event_type: &e.r#type,
            payload: SqlValue::json(&e.payload),
        });
    }
}

/// The lane row's update (fenced: commit, then extend or release the lease) or read, then
/// the answer task's read.
fn add_lane_commit(
    b: &mut Batch,
    lane: &str,
    c: &Commit,
    fenced: Option<&Fenced<'_>>,
) -> CommitReads {
    let state = || SqlValue::opt_json(c.state.as_ref());
    let lane_row = match fenced {
        Some(Fenced { fence, keep: Some(lease_secs) }) => b.add(CommitAndExtend {
            lane,
            version: fence.version,
            token: fence.token.as_str(),
            state: state(),
            drained_id: c.drained_id,
            lease_secs: *lease_secs,
        }),
        Some(Fenced { fence, keep: None }) => b.add(CommitAndRelease {
            lane,
            version: fence.version,
            token: fence.token.as_str(),
            state: state(),
            drained_id: c.drained_id,
        }),
        None => b.add(SelectLane { lane }),
    };
    let task = c.read_task.as_ref().map(|id| b.add(SelectTask { lane, task_id: id }));
    CommitReads { lane_row, task }
}

/// Where a commit's batch answers with the lane's new version and the answer task.
struct CommitReads {
    lane_row: usize,
    task: Option<usize>,
}

impl CommitReads {
    fn committed(
        self,
        out: &BatchResult,
        lane: &str,
        fence: Option<&Fence>,
    ) -> Result<Committed, StorageError> {
        let version = match fence {
            Some(_) => out.at(self.lane_row)?.first_int(),
            None => {
                let row = out.at(self.lane_row)?.rows().first();
                row.map(|row| LaneRow::from_row(row)).transpose()?.map(|l| l.version)
            }
        };
        let task = match self.task {
            Some(i) => out.at(i)?.rows().first().map(|row| TaskRow::from_row(row)).transpose()?,
            None => None,
        };
        match (fence, version) {
            (_, Some(version)) => Ok(Committed { version, task }),
            (Some(fence), None) => Err(StorageError::VersionConflict {
                lane: lane.to_owned(),
                expected: fence.version,
            }),
            (None, None) => Err(StorageError::Invalid("commit: no lane row".into())),
        }
    }
}

/// What a commit whose acknowledgment was lost turned out to be.
#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    /// The run's `run_ended` is there: it committed, making `version`, with `task` read back.
    Committed(Box<Committed>),
    /// Not committed, and the fence still holds: sending the same commit again is safe.
    NotApplied,
    /// Not committed, and the lane moved on without it: its staged writes are absent.
    Rejected,
}

/// Looks up the authoritative record of `run_id` after a lost acknowledgment, one batch: its
/// `run_ended`, the lane's fence, and the answer task.
pub(crate) async fn resolve_commit(
    conn: &dyn ThreadDbConnection,
    lane: &str,
    run_id: &str,
    fence: Option<&Fence>,
    read_task: Option<&str>,
) -> Result<Resolution, StorageError> {
    let mut b = Batch::default();
    let committed = b.add(SelectCommittedRun { lane, run_id });
    let lane_fence = b.add(SelectFence { lane });
    let task = read_task.map(|task_id| b.add(SelectTask { lane, task_id }));
    let out = super::read_batch(conn, b.0).await?;
    let committed = !out.at(committed)?.rows().is_empty();
    let task = match task {
        Some(i) if committed => {
            out.at(i)?.rows().first().map(|row| TaskRow::from_row(row)).transpose()?
        }
        _ => None,
    };
    resolution(fence, committed, out.at(lane_fence)?.rows().first(), task)
}

/// The resolution of a run whose `run_ended` is there (`committed`) or not, given the lane's
/// fence row as it stands now.
fn resolution(
    fence: Option<&Fence>,
    committed: bool,
    lane: Option<&Row>,
    task: Option<TaskRow>,
) -> Result<Resolution, StorageError> {
    let (version, token) = match lane {
        Some(row) => {
            let cells = Cells::new(row, &["version", "lease_token", "lease_until"])?;
            (cells.get::<i64>(0)?, cells.get::<Option<String>>(1)?)
        }
        None => (0, None),
    };
    if committed {
        // the version this run's commit made, not the lane's now: others may have committed
        // since
        let version = fence.map_or(version, |f| f.version + 1);
        return Ok(Resolution::Committed(Box::new(Committed { version, task })));
    }
    Ok(match fence {
        // an append-only batch that did not land can be sent again
        None => Resolution::NotApplied,
        Some(f) if f.version == version && token.as_deref() == Some(f.token.as_str()) => {
            Resolution::NotApplied
        }
        Some(_) => Resolution::Rejected,
    })
}

/// One page of the thread's tasks in `q.order`. A page token is opaque, bound to the filters
/// and order it was made under: another listing's token is refused.
pub(crate) async fn list_tasks(
    conn: &dyn ThreadDbConnection,
    q: &TaskQuery,
) -> Result<TaskPage, StorageError> {
    let order = match q.order {
        TaskOrder::Updated => "u",
        TaskOrder::Inserted => "i",
    };
    let status = q.status.map_or("*", TaskState::as_str);
    let filters = ("tasks", order, status, q.updated_after, q.lane.as_deref());
    let (lane, status, updated_after) =
        (q.lane.as_deref(), q.status.map(TaskState::as_str), q.updated_after);
    let page_size = q.page_size.clamp(1, MAX_TASK_PAGE);
    let mut b = Batch::default();
    let page = match q.order {
        TaskOrder::Updated => {
            let (before_at, before_id) =
                Cursor::position(q.page_token.as_deref(), filters, (i64::MAX, String::new()))?;
            b.add(SelectThreadTasksPage {
                lane,
                status,
                updated_after,
                before_at,
                before_id: &before_id,
                limit: page_size + 1,
            })
        }
        TaskOrder::Inserted => {
            let after = Cursor::position(q.page_token.as_deref(), filters, 0_i64)?;
            b.add(SelectThreadTasksInserted {
                lane,
                status,
                updated_after,
                after,
                limit: page_size + 1,
            })
        }
    };
    let count = b.add(CountThreadTasks { lane, status, updated_after });
    let out = super::read_batch(conn, b.0).await?;

    let (mut tasks, mut positions) = (Vec::new(), Vec::new());
    for row in out.at(page)?.rows() {
        match q.order {
            TaskOrder::Updated => tasks.push(TaskRow::from_row(row)?),
            TaskOrder::Inserted => {
                let (cells, task) = Cells::followed_by::<TaskRow>(row, &["rowid"])?;
                positions.push(cells.get::<i64>(0)?);
                tasks.push(task);
            }
        }
    }
    let more = tasks.len() as i64 > page_size;
    tasks.truncate(page_size as usize);
    let next_page_token = match (tasks.last(), q.order) {
        (Some(last), TaskOrder::Updated) if more => {
            Cursor::encode(filters, (last.updated_at, &last.task_id))
        }
        (Some(_), TaskOrder::Inserted) if more => {
            Cursor::encode(filters, positions[tasks.len() - 1])
        }
        _ => String::new(),
    };
    let total_size = out.at(count)?.first_int().unwrap_or(0);
    Ok(TaskPage { tasks, next_page_token, page_size, total_size })
}

/// One page of the thread's items in insertion order, and the cursor of the next page when
/// there is one.
#[derive(Debug, Clone, PartialEq)]
pub struct ItemPage {
    pub items: Vec<PagedItem>,
    pub next: Option<String>,
}

/// An item of a page: its position in the thread's history and its lane.
#[derive(Debug, Clone, PartialEq)]
pub struct PagedItem {
    pub seq: i64,
    pub lane: String,
    pub item: ItemRow,
}

/// The largest item page.
const MAX_ITEM_PAGE: i64 = 500;

/// One page of the thread's items after `cursor`, of `lane` when given. A cursor, opaque to
/// callers, is bound to the lane filter it was made under: another filter's is refused.
pub(crate) async fn list_items(
    conn: &dyn ThreadDbConnection,
    lane: Option<&str>,
    cursor: Option<&str>,
    limit: i64,
) -> Result<ItemPage, StorageError> {
    let scope = ("items", lane);
    let after = Cursor::position(cursor, scope, 0_i64)?;
    let limit = limit.clamp(1, MAX_ITEM_PAGE);
    let rows = conn.query(SelectItemsAfter { lane, after, limit: limit + 1 }.bind()).await?;
    let mut items = Vec::new();
    let mut last = 0;
    for row in &rows {
        let (cells, item) = Cells::followed_by::<ItemRow>(row, &["rowid", "lane", "seq"])?;
        let pos: i64 = cells.get(0)?;
        if items.len() as i64 == limit {
            return Ok(ItemPage { items, next: Some(Cursor::encode(scope, last)) });
        }
        items.push(PagedItem { seq: cells.get(2)?, lane: cells.get(1)?, item });
        last = pos;
    }
    Ok(ItemPage { items, next: None })
}

/// A turn's `submitted` row, before the lease is tried; changes nothing when its task or message
/// id already had one.
pub(crate) fn submitted_task<'a>(lane: &'a str, task: &'a TaskWrite) -> InsertSubmittedTask<'a> {
    InsertSubmittedTask {
        lane,
        task_id: &task.task_id,
        name: task.name.as_deref(),
        message_id: task.message_id.as_deref(),
        data: SqlValue::json(&task.data),
        internal: SqlValue::opt_json(task.internal.as_ref()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fence() -> Fence {
        Fence {
            lane: "front-desk".into(),
            holder: crate::Holder::new("w1"),
            token: "lease_a".into(),
            version: 3,
        }
    }

    fn lane(version: i64, token: Option<&str>) -> Row {
        vec![SqlValue::Int(version), token.map_or(SqlValue::Null, Into::into), SqlValue::Int(0)]
    }

    fn resolve(fence: Option<&Fence>, committed: bool, lane: Row) -> Resolution {
        resolution(fence, committed, Some(&lane), None).unwrap()
    }

    // A lost acknowledgment is decided by the run's own record, never guessed.
    #[test]
    fn a_lost_commit_resolves_by_its_run_ended_and_the_fence() {
        let f = fence();
        let committed = Resolution::Committed(Box::new(Committed { version: 4, task: None }));
        assert_eq!(resolve(Some(&f), true, lane(4, None)), committed);
        // The lane moved on after the lost commit; the run still reports its own commit's version.
        assert_eq!(resolve(Some(&f), true, lane(9, Some("lease_c"))), committed);
        assert_eq!(resolve(Some(&f), false, lane(3, Some("lease_a"))), Resolution::NotApplied);
        assert_eq!(resolve(Some(&f), false, lane(3, Some("lease_b"))), Resolution::Rejected);
        assert_eq!(resolve(Some(&f), false, lane(4, None)), Resolution::Rejected);
        assert_eq!(resolve(None, false, lane(0, None)), Resolution::NotApplied);
    }

    /// Answers every batch with no results.
    struct Short;

    #[async_trait::async_trait]
    impl ThreadDbConnection for Short {
        async fn query(&self, _: Statement) -> Result<Vec<Row>, StorageError> {
            Ok(Vec::new())
        }
        async fn batch(&self, _: Vec<Statement>) -> Result<BatchResult, StorageError> {
            Ok(BatchResult { results: Vec::new() })
        }
        async fn close(&self) {}
    }

    /// An answer missing a statement's result is malformed, never an empty success: a lost
    /// commit is not resolved as not applied, nor a commit as made.
    #[tokio::test]
    async fn a_batch_answer_missing_a_statement_is_invalid() {
        let f = fence();
        let resolved = resolve_commit(&Short, "front-desk", "r1", Some(&f), None).await;
        assert!(matches!(resolved, Err(StorageError::Invalid(_))), "{resolved:?}");
        let fenced = Some(Fenced { fence: &f, keep: None });
        let committed = commit(&Short, "front-desk", &Commit::default(), fenced).await;
        assert!(matches!(committed, Err(StorageError::Invalid(_))), "{committed:?}");
    }
}
