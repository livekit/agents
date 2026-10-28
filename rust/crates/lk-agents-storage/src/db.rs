//! The thread database's operations: [`ThreadDb`] for the database, [`LaneDb`] for one lane, and
//! the [`Lease`] an acquire takes, which keeps itself alive, commits and releases.

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::collections::HashMap;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::atomic::{AtomicBool, Ordering};
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::conn::{
    BatchResult, Conn, ExecResult, Row, SqlValue, StatementResult, WsdbError, first_cell,
    statement, wire as pb,
};
use crate::run::{LEASE_WAIT_TIMEOUT, TaskState};
use crate::schema::*;
use crate::sql::{FromCell, Statement, wire};
use crate::{StorageError, now_ms};

/// The default lease TTL, measured on agent-db's clock.
pub const DEFAULT_LEASE_TTL: Duration = Duration::from_secs(10);
/// The default interval at which an open run renews.
pub const DEFAULT_RENEW_INTERVAL: Duration = Duration::from_secs(3);
/// How many history items one read loads.
const HISTORY_PAGE: i64 = 500;
/// The largest `tasks/list` page A2A allows.
const MAX_TASK_PAGE: i64 = 100;

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

/// The largest state, in bytes.
pub const STATE_LIMIT: usize = 64 * 1024;

/// What an acquire writes and loads besides the lease.
pub struct Acquire<'a> {
    pub holder: &'a str,
    pub run_id: &'a str,
    /// The run's task: a `submitted` row of it orders the acquire behind older waiters; empty
    /// for a run with no row (a voice turn).
    pub task_id: &'a str,
    pub run_started: RunStarted,
    /// The turn's `submitted` row, written first in the same batch.
    pub submit: Option<&'a TaskWrite>,
    /// Take `task_id`'s row to `working` under the lease, in the same batch.
    pub pickup: bool,
    /// Load the lane's newest task rows, this many.
    pub tasks_limit: i64,
}

/// What an acquire loaded, under the lease it took.
pub struct Loaded {
    pub lease: Lease,
    /// this `run_id` already committed; the caller releases without running.
    pub already_committed: bool,
    pub lane: LaneRow,
    pub items: Vec<ItemRow>,
    /// The newest task rows, newest first.
    pub tasks: Vec<TaskRow>,
    /// Whether `submit` inserted its row; `Some(false)` is a `task_id` or `messageId` that
    /// already had one.
    pub submitted: Option<bool>,
    /// Whether `pickup` moved the row to `working`; `false` when it is terminal or missing.
    pub picked_up: bool,
}

/// One journal entry a commit appends.
#[derive(Debug, Clone, Default)]
pub struct EventWrite {
    pub ts_ms: i64,
    pub r#type: String,
    pub payload: Value,
}

impl EventWrite {
    /// Stamped now.
    pub fn new(r#type: &str, payload: Value) -> Self {
        Self { ts_ms: now_ms(), r#type: r#type.to_owned(), payload }
    }

    pub fn run_started(p: &RunStarted) -> Self {
        Self::new(RUN_STARTED_EVENT, serde_json::to_value(p).unwrap_or_default())
    }

    pub fn run_ended(p: &RunEnded) -> Self {
        Self::new(RUN_ENDED_EVENT, serde_json::to_value(p).unwrap_or_default())
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

pub struct Committed {
    /// The version after the bump; the current one for an unfenced write.
    pub version: i64,
    /// `read_task`'s row after the commit.
    pub task: Option<TaskRow>,
}

/// What the inbox receive answers.
pub enum Receive {
    /// The row went in; the lease when this worker took it and is the runner.
    Received(Option<Lease>),
    /// The `messageId` was in the inbox already: nothing inserted; the task it belongs to.
    Duplicate(Option<String>),
}

/// How long a parked connection waits for the thread's next turn before it is closed.
const IDLE_TTL: Duration = Duration::from_secs(60);
/// Parked connections per database, and in all.
const IDLE_PER_DATABASE: usize = 2;
const IDLE_TOTAL: usize = 64;

/// The connections closed threads left behind, by database id: a thread's next turn takes one
/// back, with the schema version it knew, instead of dialing and probing again.
#[derive(Default)]
pub(crate) struct DbPool {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    idle: Mutex<HashMap<String, Vec<Idle>>>,
}

struct Idle {
    conn: Conn,
    schema: Option<i64>,
    since: Instant,
}

impl DbPool {
    #[allow(clippy::disallowed_methods, reason = "S0: runtime migration")]
    fn take(&self, database_id: &str) -> Option<Idle> {
        let mut idle = self.idle.lock().expect("pool");
        let parked = idle.get_mut(database_id)?;
        while let Some(entry) = parked.pop() {
            if entry.since.elapsed() < IDLE_TTL && entry.conn.is_idle() {
                return Some(entry);
            }
            entry.conn.close();
        }
        None
    }

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    fn put(&self, conn: Conn, schema: Option<i64>) {
        let mut idle = self.idle.lock().expect("pool");
        idle.retain(|_, parked| {
            parked.retain(|entry| {
                let keep = entry.since.elapsed() < IDLE_TTL && entry.conn.is_idle();
                if !keep {
                    entry.conn.close();
                }
                keep
            });
            !parked.is_empty()
        });
        let total: usize = idle.values().map(Vec::len).sum();
        let parked = idle.entry(conn.database_id().to_owned()).or_default();
        if !conn.is_idle() || parked.len() >= IDLE_PER_DATABASE || total >= IDLE_TOTAL {
            conn.close();
            return;
        }
        parked.push(Idle { conn, schema, since: Instant::now() });
    }
}

/// One thread's database over one connection, closed at run end.
#[derive(Clone)]
pub struct ThreadDb {
    conn: Conn,
    lease_ttl_secs: i64,
    renew_interval: Duration,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// The schema version found at open, `None` when the database has no schema yet.
    schema: Arc<Mutex<Option<i64>>>,
    /// Where `close` parks the connection.
    pool: Arc<DbPool>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// Set by `close`, shared by the clones: a parked connection is no longer this database's.
    closed: Arc<AtomicBool>,
}

impl ThreadDb {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// The parked connection of `database_id` when there is one, else `dial`'s with its schema
    /// version read: missing means the first write applies the DDL, ahead means a newer SDK wrote
    /// this database. `close` parks it for the next open.
    pub(crate) async fn pooled<F, Fut>(
        pool: Arc<DbPool>,
        database_id: &str,
        lease_ttl: Duration,
        renew_interval: Duration,
        dial: F,
    ) -> Result<Self, StorageError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<Conn, StorageError>>,
    {
        let (conn, schema) = match pool.take(database_id) {
            Some(idle) => (idle.conn, idle.schema),
            None => {
                let conn = dial().await?;
                let schema = schema_version(&conn).await?;
                (conn, schema)
            }
        };
        Ok(Self {
            conn,
            lease_ttl_secs: (lease_ttl.as_secs() + u64::from(lease_ttl.subsec_nanos() > 0)) as i64,
            renew_interval,
            schema: Arc::new(Mutex::new(schema)),
            pool,
            closed: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn database_id(&self) -> &str {
        self.conn.database_id()
    }

    /// The durable commit sequence last observed on the connection.
    pub fn tip(&self) -> i64 {
        self.conn.tip()
    }

    /// Done with the database: the connection is parked for the thread's next turn. Every clone
    /// is closed with it.
    pub fn close(&self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            self.pool.put(self.conn.clone(), *self.schema.lock().expect("schema"));
        }
    }

    pub fn lane(&self, name: &str) -> LaneDb {
        LaneDb { db: self.clone(), name: name.to_owned() }
    }

    /// The journal, every lane, after the event `after_id`.
    pub async fn events(&self, after_id: i64, limit: i64) -> Result<Vec<EventRow>, StorageError> {
        self.read(SelectEvents { after_id, limit }, EventRow::from_row).await
    }

    /// One task, whichever lane it is on.
    pub async fn task(&self, task_id: &str) -> Result<Option<TaskRow>, StorageError> {
        Ok(self.read(SelectThreadTask { task_id }, TaskRow::from_row).await?.into_iter().next())
    }

    /// One page of the thread's tasks, last updated first, as A2A `tasks/list` answers it.
    pub async fn list_tasks(&self, q: &TaskQuery) -> Result<TaskPage, StorageError> {
        if !self.refresh_schema().await? {
            return Ok(TaskPage::default());
        }
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
        let out = self.batch(b).await?;
        let mut tasks: Vec<TaskRow> =
            out.results[page].rows().iter().map(TaskRow::from_row).collect::<Result<_, _>>()?;
        let more = tasks.len() as i64 > page_size;
        tasks.truncate(page_size as usize);
        let next_page_token = match tasks.last() {
            Some(last) if more => format!("{}:{}", last.updated_at, last.task_id),
            _ => String::new(),
        };
        let total_size = first_int(&out.results[count]).unwrap_or(0);
        Ok(TaskPage { tasks, next_page_token, page_size, total_size })
    }

    pub async fn lanes(&self) -> Result<Vec<LaneRow>, StorageError> {
        self.read(SelectLanes {}, LaneRow::from_row).await
    }

    /// The connection, for a statement; `Unavailable` once closed.
    fn live(&self) -> Result<&Conn, StorageError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(StorageError::Unavailable("the database is closed".into()));
        }
        Ok(&self.conn)
    }

    fn schema_missing(&self) -> bool {
        self.schema.lock().expect("schema").is_none()
    }

    fn schema_applied(&self) {
        *self.schema.lock().expect("schema") = Some(THREAD_SCHEMA_VERSION);
    }

    /// Asks the database again whether the schema is there. A connection that found no schema
    /// on open must not remember that: another worker may have created it since (two workers
    /// opening a brand-new thread at once). True when the schema is present now.
    async fn refresh_schema(&self) -> Result<bool, StorageError> {
        if !self.schema_missing() {
            return Ok(true);
        }
        match schema_version(self.live()?).await? {
            Some(_) => {
                self.schema_applied();
                Ok(true)
            }
            // no schema, or the tables exist but the version row is not written yet: the
            // creator's batch is in flight; the caller retries
            None => Ok(false),
        }
    }

    /// The DDL and the `schema_version` row, when the database has no schema yet.
    fn bootstrap(&self) -> Vec<pb::Statement> {
        if !self.schema_missing() {
            return Vec::new();
        }
        let ddl = crate::sql::schema_statements(THREAD_SCHEMA);
        let mut out: Vec<pb::Statement> = ddl.map(|s| statement(s, vec![])).collect();
        out.push(wire(InsertSchemaVersion { version: THREAD_SCHEMA_VERSION }));
        out
    }

    /// Applies the DDL when the schema is missing.
    async fn ensure_schema(&self) -> Result<(), StorageError> {
        if self.refresh_schema().await? {
            return Ok(());
        }
        match self.live()?.batch(self.bootstrap()).await {
            Ok(_) => {
                self.schema_applied();
                Ok(())
            }
            Err(e) if lost_schema_race(&e) => {
                self.schema_applied();
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn read<S: Statement, T>(
        &self,
        s: S,
        decode: fn(&Row) -> Result<T, StorageError>,
    ) -> Result<Vec<T>, StorageError> {
        if !self.refresh_schema().await? {
            return Ok(Vec::new()); // a database with no schema holds nothing
        }
        self.live()?.query_all(S::sql(), s.params()).await?.iter().map(decode).collect()
    }

    async fn exec<S: Statement>(&self, s: S) -> Result<ExecResult, StorageError> {
        Ok(self.live()?.exec(S::sql(), s.params()).await?)
    }

    async fn batch(&self, b: Batch) -> Result<BatchResult, StorageError> {
        Ok(self.live()?.batch(b.0).await?)
    }
}

/// A batch that lost the race against another creator's DDL: the schema exists now.
fn lost_schema_race(e: &dyn std::fmt::Display) -> bool {
    e.to_string().contains("already exists")
}

/// A thread's database scoped to one lane: every statement of the lane.
#[derive(Clone)]
pub struct LaneDb {
    db: ThreadDb,
    name: String,
}

impl LaneDb {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Takes the lease and loads the lane, one atomic `Batch`; `None` when another holder has the
    /// lease. A retried `run_id` loads as `already_committed` and writes nothing.
    pub async fn acquire(&self, req: &Acquire<'_>) -> Result<Option<Loaded>, StorageError> {
        // the DDL rides in the acquire batch of a fresh database; a batch that loses the race to
        // another creator's DDL is retried without it
        match self.acquire_batch(req).await {
            Err(e) if lost_schema_race(&e) => {
                self.db.schema_applied();
                self.acquire_batch(req).await
            }
            out => out,
        }
    }

    async fn acquire_batch(&self, req: &Acquire<'_>) -> Result<Option<Loaded>, StorageError> {
        let lane = self.name.as_str();
        let token = crate::shortuuid("lease_");
        let (holder, token_ref) = (req.holder, token.as_str());
        let mut b = Batch(self.db.bootstrap());
        let submitted = req.submit.map(|task| b.add(submitted_task(lane, task)));
        b.add(InsertLane { lane });
        let take = b.add(TakeLease {
            lane,
            holder,
            token: token_ref,
            lease_secs: self.db.lease_ttl_secs,
            task_id: req.task_id,
            wait_ms: LEASE_WAIT_TIMEOUT.as_millis() as i64,
        });
        b.add(ClearWaiter { lane, holder, token: token_ref });
        b.add(InsertRunStarted {
            lane,
            token: token_ref,
            ts_ms: now_ms(),
            run_id: req.run_id,
            payload: SqlValue::json(&req.run_started),
        });
        let committed = b.add(SelectCommittedRun { lane, run_id: req.run_id });
        let pickup = req
            .pickup
            .then(|| b.add(PickupTaskHeld { lane, token: token_ref, task_id: req.task_id }));
        let items = b.add(SelectItemsPage { lane, limit: HISTORY_PAGE, offset: 0 });
        let lane_row = b.add(SelectLane { lane });
        let tasks = b.add(SelectTasks { lane, limit: req.tasks_limit });
        let out = self.db.batch(b).await?;
        if self.db.schema_missing() {
            self.db.schema_applied();
        }
        let r = &out.results;
        let Some(version) = first_int(&r[take]) else {
            return Ok(None); // not acquired: the reads are discarded
        };
        let lane_row = r[lane_row]
            .rows()
            .first()
            .ok_or_else(|| StorageError::Decode("acquire: no lane row".into()))?;
        let mut items: Vec<ItemRow> =
            r[items].rows().iter().map(ItemRow::from_row).collect::<Result<_, _>>()?;
        // a long history past the first page is read after the batch, so the batch's answer
        // stays bounded; under the lease nothing but this run deletes or reorders it
        let mut page = items.len();
        while page as i64 == HISTORY_PAGE {
            let next = SelectItemsPage { lane, limit: HISTORY_PAGE, offset: items.len() as i64 };
            let more = self.db.read(next, ItemRow::from_row).await?;
            page = more.len();
            items.extend(more);
        }
        Ok(Some(Loaded {
            lease: Lease::new(self.clone(), req.holder, token, version),
            already_committed: !r[committed].rows().is_empty(),
            lane: LaneRow::from_row(lane_row)?,
            items,
            tasks: r[tasks].rows().iter().map(TaskRow::from_row).collect::<Result<_, _>>()?,
            submitted: submitted.map(|i| r[i].rows_affected() > 0),
            picked_up: pickup.is_some_and(|i| r[i].rows_affected() > 0),
        }))
    }

    /// The inbox receive, one atomic `Batch`: the inbox row, the `submitted` task row and the
    /// lease attempt.
    pub async fn receive(
        &self,
        holder: &str,
        task_id: &str,
        message_id: &str,
        message: &a2a::Message,
        data: &TaskData,
    ) -> Result<Receive, StorageError> {
        self.db.ensure_schema().await?;
        let lane = self.name.as_str();
        let token = crate::shortuuid("lease_");
        let mut b = Batch::default();
        b.add(InsertLane { lane });
        let inbox =
            b.add(InsertInbox { lane, task_id, message_id, message: SqlValue::json(message) });
        b.add(InsertSubmittedTaskForMessage {
            lane,
            task_id,
            message_id,
            data: SqlValue::json(data),
        });
        let take = b.add(TakeLeaseForMessage {
            lane,
            holder,
            token: &token,
            lease_secs: self.db.lease_ttl_secs,
            message_id,
            task_id,
        });
        let existing = b.add(SelectInboxTask { lane, message_id });
        let out = self.db.batch(b).await?;
        let r = &out.results;
        let StatementResult::Exec { rows_affected, .. } = r[inbox] else {
            return Err(StorageError::Protocol("receive: the inbox insert answered rows".into()));
        };
        if rows_affected == 0 {
            let task_id =
                first_cell(r[existing].rows()).and_then(SqlValue::as_str).map(str::to_owned);
            return Ok(Receive::Duplicate(task_id));
        }
        let lease = first_int(&r[take]).map(|v| Lease::new(self.clone(), holder, token, v));
        Ok(Receive::Received(lease))
    }

    /// A turn's `submitted` row, before the lease is tried; `false` when its task or message id
    /// already had one.
    pub async fn submit(&self, task: &TaskWrite) -> Result<bool, StorageError> {
        self.db.ensure_schema().await?;
        Ok(self.db.exec(submitted_task(&self.name, task)).await?.rows_affected > 0)
    }

    /// The `concurrent` policy's append-only write: no lease, no fence, no version bump.
    pub async fn append(&self, commit: &Commit) -> Result<Committed, StorageError> {
        write(self, commit, None).await
    }

    /// Records a worker waiting under `interrupt`; `false` when the lease is not held, in which
    /// case the caller acquires instead.
    pub async fn set_waiter(&self, holder: &str) -> Result<bool, StorageError> {
        self.db.ensure_schema().await?;
        let r = self.db.exec(SetWaiter { lane: &self.name, holder }).await?;
        Ok(r.rows_affected > 0)
    }

    /// A turn's checkpoint, one batch: the lane's waiter and the task's status.
    pub async fn checkpoint(
        &self,
        task_id: &str,
    ) -> Result<(Option<String>, Option<TaskState>), StorageError> {
        let lane = self.name.as_str();
        let mut b = Batch::default();
        let waiter = b.add(SelectWaiter { lane });
        let status = b.add(SelectTaskStatus { lane, task_id });
        let out = self.db.batch(b).await?;
        let r = &out.results;
        let waiter = first_cell(r[waiter].rows()).and_then(SqlValue::as_str).map(str::to_owned);
        Ok((
            waiter,
            first_cell(r[status].rows())
                .map(TaskState::from_cell)
                .transpose()
                .map_err(StorageError::Decode)?,
        ))
    }

    /// Takes a `submitted` or `working` row to `working`, one atomic `Batch`. The status it had
    /// (`working` means a dead holder had it), `None` when the row is terminal or missing.
    pub async fn pickup(&self, task_id: &str) -> Result<Option<TaskState>, StorageError> {
        let lane = self.name.as_str();
        let mut b = Batch::default();
        let before = b.add(SelectPickupStatus { lane, task_id });
        let moved = b.add(PickupTask { lane, task_id });
        let out = self.db.batch(b).await?;
        if out.results[moved].rows_affected() == 0 {
            return Ok(None);
        }
        first_cell(out.results[before].rows())
            .map(TaskState::from_cell)
            .transpose()
            .map_err(StorageError::Decode)
    }

    /// A `working` row whose holder died ends `failed` with the given status message (reason
    /// `worker-lost`).
    pub async fn fail_orphan(
        &self,
        task_id: &str,
        status_message: &a2a::Message,
    ) -> Result<bool, StorageError> {
        let status_message = SqlValue::json(status_message);
        let r = self.db.exec(FailOrphan { lane: &self.name, task_id, status_message }).await?;
        Ok(r.rows_affected > 0)
    }

    /// The whole history; how another lane reads it.
    pub async fn history(&self) -> Result<Vec<ItemRow>, StorageError> {
        self.db.read(SelectItems { lane: &self.name }, ItemRow::from_row).await
    }

    pub async fn info(&self) -> Result<Option<LaneRow>, StorageError> {
        let rows = self.db.read(SelectLane { lane: &self.name }, LaneRow::from_row).await?;
        Ok(rows.into_iter().next())
    }

    /// The newest task rows, of one status when given.
    pub async fn tasks(
        &self,
        status: Option<TaskState>,
        limit: i64,
    ) -> Result<Vec<TaskRow>, StorageError> {
        let lane = self.name.as_str();
        match status {
            Some(s) => {
                let read = SelectTasksByStatus { lane, status: s.as_str(), limit };
                self.db.read(read, TaskRow::from_row).await
            }
            None => self.db.read(SelectTasks { lane, limit }, TaskRow::from_row).await,
        }
    }

    pub async fn task(&self, task_id: &str) -> Result<Option<TaskRow>, StorageError> {
        let rows = self.db.read(SelectTask { lane: &self.name, task_id }, TaskRow::from_row);
        Ok(rows.await?.into_iter().next())
    }

    /// The task an A2A `messageId` created.
    pub async fn task_for_message(
        &self,
        message_id: &str,
    ) -> Result<Option<TaskRow>, StorageError> {
        let read = SelectTaskForMessage { lane: &self.name, message_id };
        Ok(self.db.read(read, TaskRow::from_row).await?.into_iter().next())
    }

    /// The inbox rows in `id` order: every one, or only those above `drained_id`.
    pub async fn inbox(&self, pending: bool) -> Result<Vec<InboxRow>, StorageError> {
        let lane = self.name.as_str();
        match pending {
            true => self.db.read(SelectInboxPending { lane }, InboxRow::from_row).await,
            false => self.db.read(SelectInbox { lane }, InboxRow::from_row).await,
        }
    }

    /// The lane's journal after the event `after_id`.
    pub async fn events(&self, after_id: i64) -> Result<Vec<EventRow>, StorageError> {
        self.db.read(SelectLaneEvents { lane: &self.name, after_id }, EventRow::from_row).await
    }
}

/// A lease an acquire or a receive took: the lane at `version` under `token`. Cloned by the run and
/// its checkpoint; [`Lease::keep_alive`] extends it while the run works.
#[derive(Clone)]
pub struct Lease {
    inner: Arc<LeaseInner>,
}

struct LeaseInner {
    lane: LaneDb,
    holder: String,
    token: String,
    version: i64,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    lost: AtomicBool,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    reason: Mutex<Option<String>>,
    notify: tokio::sync::Notify,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    renewal: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl Drop for LeaseInner {
    fn drop(&mut self) {
        if let Some(t) = self.renewal.get_mut().ok().and_then(Option::take) {
            t.abort();
        }
    }
}

impl Lease {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    fn new(lane: LaneDb, holder: &str, token: String, version: i64) -> Self {
        Self {
            inner: Arc::new(LeaseInner {
                lane,
                holder: holder.to_owned(),
                token,
                version,
                lost: AtomicBool::new(false),
                reason: Mutex::new(None),
                notify: tokio::sync::Notify::new(),
                renewal: Mutex::new(None),
            }),
        }
    }

    /// The version the lease loaded: the commit's fence.
    pub fn version(&self) -> i64 {
        self.inner.version
    }

    pub fn holder(&self) -> &str {
        &self.inner.holder
    }

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    /// Extends the lease every interval until [`Lease::stop`]. An extend that changes no row, or
    /// one that keeps failing until the lease could have lapsed, marks it lost.
    pub fn keep_alive(&self) {
        let inner = self.inner.clone();
        let task = tokio::spawn(async move {
            let db = &inner.lane.db;
            let lane = inner.lane.name.as_str();
            let alive_for =
                Duration::from_secs(db.lease_ttl_secs as u64).saturating_sub(db.renew_interval);
            let mut extended_at = Instant::now();
            let mut tick = tokio::time::interval(db.renew_interval);
            tick.tick().await;
            loop {
                tick.tick().await;
                let extend = ExtendLease {
                    lane,
                    holder: &inner.holder,
                    token: &inner.token,
                    lease_secs: db.lease_ttl_secs,
                };
                let sent = db.conn.exec(ExtendLease::sql(), extend.params());
                let attempt = tokio::time::timeout(db.renew_interval, sent);
                let failure = match attempt.await {
                    Ok(Ok(r)) if r.rows_affected > 0 => {
                        extended_at = Instant::now();
                        continue;
                    }
                    Ok(Ok(_)) => None,
                    Ok(Err(e)) => Some(StorageError::from(e)),
                    Err(_) => Some(StorageError::Unavailable("the extend timed out".into())),
                };
                let reason = match failure {
                    None => format!("lease on lane {lane} was lost"),
                    Some(e) if e.retryable() && extended_at.elapsed() < alive_for => {
                        log::debug!(target: "livekit.agents", lane = lane; "lease extend failed, retrying: {e}");
                        continue;
                    }
                    Some(e) => format!("lease renewal on lane {lane} failed: {e}"),
                };
                log::warn!(target: "livekit.agents", lane = inner.lane.name.as_str(); "{reason}");
                *inner.reason.lock().expect("reason") = Some(reason);
                inner.lost.store(true, Ordering::Release);
                inner.notify.notify_waiters();
                return;
            }
        });
        *self.inner.renewal.lock().expect("renewal") = Some(task);
    }

    /// Ends the renewal; the lease itself is released by the commit or a release.
    pub fn stop(&self) {
        if let Some(t) = self.inner.renewal.lock().expect("renewal").take() {
            t.abort();
        }
    }

    pub fn is_lost(&self) -> bool {
        self.inner.lost.load(Ordering::Acquire)
    }

    /// Resolves with the error that cancels the run once the lease is lost; pending forever while
    /// it is held, so race it against the run.
    pub async fn lost(&self) -> StorageError {
        loop {
            if self.is_lost() {
                let reason = self.inner.reason.lock().expect("reason").clone();
                return StorageError::Unavailable(reason.unwrap_or_else(|| "lease lost".into()));
            }
            self.inner.notify.notified().await;
        }
    }

    /// Writes the run fenced on the version loaded, then bumps `version` and releases, or extends
    /// when `keep`. `VersionConflict` when the run outlived its lease.
    pub async fn commit(&self, commit: &Commit, keep: bool) -> Result<Committed, StorageError> {
        write(&self.inner.lane, commit, Some((self, keep))).await
    }

    /// Releases without writing anything else but, when given, the `run_ended` event of `run_id`:
    /// the path of a run that failed or was already committed.
    pub async fn release(&self, ended: Option<(&str, &EventWrite)>) -> Result<(), StorageError> {
        let (lane, token) = (self.inner.lane.name.as_str(), self.inner.token.as_str());
        let mut b = Batch::default();
        if let Some((run_id, e)) = ended {
            let payload = SqlValue::json(&e.payload);
            b.add(InsertRunEnded { lane, token, ts_ms: e.ts_ms, run_id, payload });
        }
        b.add(ReleaseLease { lane, token });
        self.inner.lane.db.batch(b).await?;
        Ok(())
    }

    /// Releases when no inbox row is pending past `drained_id`; `true` when released.
    pub async fn release_if_drained(&self) -> Result<bool, StorageError> {
        let (lane, token) = (self.inner.lane.name.as_str(), self.inner.token.as_str());
        let r = self.inner.lane.db.exec(ReleaseLeaseIfDrained { lane, token }).await?;
        Ok(r.rows_affected > 0)
    }
}

/// A run's writes, one batch: fenced on `lease` (and renewing when `keep`), else append-only.
async fn write(
    lane: &LaneDb,
    c: &Commit,
    fence: Option<(&Lease, bool)>,
) -> Result<Committed, StorageError> {
    if let Some(state) = &c.state {
        let size = serde_json::to_vec(state)
            .map_err(|e| StorageError::NotSerializable("state".into(), e.to_string()))?
            .len();
        if size > STATE_LIMIT {
            return Err(StorageError::TooLarge { what: "state".into(), limit: STATE_LIMIT });
        }
    }
    let db = &lane.db;
    db.ensure_schema().await?;
    let lane_name = lane.name.as_str();
    let mut b = Batch::default();
    match fence {
        Some((lease, _)) => {
            let (lane, version, token) =
                (lane_name, lease.inner.version, lease.inner.token.as_str());
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
            let lane = lane_name;
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
    let lane_row = match fence {
        Some((lease, keep)) => {
            let (lane, version, token) =
                (lane_name, lease.inner.version, lease.inner.token.as_str());
            let state = SqlValue::opt_json(c.state.as_ref());
            let drained_id = c.drained_id;
            match keep {
                true => b.add(CommitAndExtend {
                    lane,
                    version,
                    token,
                    state,
                    drained_id,
                    lease_secs: db.lease_ttl_secs,
                }),
                false => b.add(CommitAndRelease { lane, version, token, state, drained_id }),
            }
        }
        None => b.add(SelectLane { lane: lane_name }),
    };
    let read_task =
        c.read_task.as_ref().map(|id| b.add(SelectTask { lane: lane_name, task_id: id }));
    let out = db.batch(b).await?;
    let r = &out.results;
    let version = match fence {
        Some(_) => first_int(&r[lane_row]),
        None => r[lane_row].rows().first().map(LaneRow::from_row).transpose()?.map(|l| l.version),
    };
    let task = match read_task {
        Some(i) => r[i].rows().first().map(TaskRow::from_row).transpose()?,
        None => None,
    };
    match (fence, version) {
        (_, Some(version)) => Ok(Committed { version, task }),
        (Some((lease, _)), None) => Err(StorageError::VersionConflict {
            lane: lane.name.clone(),
            expected: lease.inner.version,
        }),
        (None, None) => Err(StorageError::Decode("commit: no lane row".into())),
    }
}

/// Statements sent as one atomic `Batch`; `add` answers the index of the statement's result.
#[derive(Default)]
struct Batch(Vec<pb::Statement>);

impl Batch {
    fn add<S: Statement>(&mut self, s: S) -> usize {
        self.0.push(wire(s));
        self.0.len() - 1
    }
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

/// The schema version the database holds: `None` before any SDK wrote it. Behind means a
/// database an older SDK wrote, ahead a newer one's; neither is read.
async fn schema_version(conn: &Conn) -> Result<Option<i64>, StorageError> {
    let set = match conn
        .query_all(SelectSchemaVersion::sql(), SelectSchemaVersion {}.params())
        .await
    {
        Ok(set) => set,
        Err(WsdbError::Server { message, .. }) if message.contains(NO_SCHEMA) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    match first_cell(&set).and_then(SqlValue::as_i64) {
        Some(v) if v == THREAD_SCHEMA_VERSION => Ok(Some(v)),
        Some(v) => Err(StorageError::Schema(format!(
            "database schema is version {v}, this client writes {THREAD_SCHEMA_VERSION}"
        ))),
        None => Ok(None),
    }
}

fn first_int(r: &StatementResult) -> Option<i64> {
    first_cell(r.rows()).and_then(SqlValue::as_i64)
}

#[cfg(test)]
mod tests {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    use std::sync::atomic::AtomicU32;

    use super::*;
    use crate::conn::tests::{dial, reply, script, serve};
    use crate::conn::wire as pb;

    /// A lane on the fake `script`, its leases `lease_ttl` long and extended every 100 ms.
    async fn lane_whose_extends_fail(
        script: fn(&pb::ClientMessage) -> Vec<pb::ServerMessage>,
        lease_ttl: Duration,
    ) -> LaneDb {
        let (url, _) = serve(script, false).await;
        let db = ThreadDb::pooled(
            Arc::new(DbPool::default()),
            "DB_1",
            lease_ttl,
            Duration::from_millis(100),
            || async { dial(&url, "DB_1").await.map_err(StorageError::from) },
        )
        .await
        .unwrap();
        db.lane("front-desk")
    }

    /// The schema probe answers this client's version.
    fn schema_answer(m: &pb::ClientMessage) -> Option<Vec<pb::ServerMessage>> {
        use pb::server_message::Message as S;
        let Some(pb::client_message::Message::Query(q)) = &m.message else { return None };
        (q.sql == SelectSchemaVersion::sql()).then(|| {
            let rows = crate::conn::encode_batch(&[vec![THREAD_SCHEMA_VERSION.into()]], 1);
            vec![
                reply(S::Columns(pb::Columns { names: vec!["version".into()], statement: 0 })),
                reply(S::ColumnBatch(rows)),
                reply(S::Done(pb::Done { tip: 7, total_rows: 1 })),
            ]
        })
    }

    fn extend_answer(m: &pb::ClientMessage, landed: bool) -> Option<Vec<pb::ServerMessage>> {
        use pb::server_message::Message as S;
        let Some(pb::client_message::Message::Exec(s)) = &m.message else { return None };
        (s.sql == ExtendLease::sql()).then(|| match landed {
            true => vec![reply(S::ExecResult(pb::ExecResult {
                rows_affected: 1,
                ..Default::default()
            }))],
            false => vec![reply(S::Error(pb::Error {
                code: "unavailable".into(),
                message: "the node is draining".into(),
            }))],
        })
    }

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[tokio::test]
    async fn an_extend_that_fails_for_a_while_is_retried_and_the_lease_kept() {
        fn flaky(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
            #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
            static EXTENDS: AtomicU32 = AtomicU32::new(0);
            let landed = || EXTENDS.fetch_add(1, Ordering::SeqCst) >= 2;
            extend_answer(m, landed()).or_else(|| schema_answer(m)).unwrap_or_else(|| script(m))
        }
        let lane = lane_whose_extends_fail(flaky, Duration::from_secs(10)).await;
        let lease = Lease::new(lane, "worker-a", "lease_t".into(), 1);
        lease.keep_alive();
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(!lease.is_lost());
        lease.stop();
    }

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[tokio::test]
    async fn an_extend_that_keeps_failing_loses_the_lease_before_it_could_lapse() {
        fn down(m: &pb::ClientMessage) -> Vec<pb::ServerMessage> {
            extend_answer(m, false).or_else(|| schema_answer(m)).unwrap_or_else(|| script(m))
        }
        let lane = lane_whose_extends_fail(down, Duration::from_secs(1)).await;
        let lease = Lease::new(lane, "worker-a", "lease_t".into(), 1);
        lease.keep_alive();
        let lost = tokio::time::timeout(Duration::from_secs(2), lease.lost()).await;
        assert!(matches!(lost, Ok(StorageError::Unavailable(m)) if m.contains("draining")));
    }
}
