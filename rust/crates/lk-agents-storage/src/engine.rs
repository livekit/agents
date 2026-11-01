//! The shared engine over one thread database, whichever backend: [`Thread`] is the handle a
//! worker opens, executing the core's plans on a [`ThreadDbConnection`], [`Lane`] one agent's part
//! of it, and [`Lane::open`] takes it for one [`Run`], which ends with
//! [`Run::commit`] (one fenced write) or [`Run::abandon`] (`run_ended` and the release).
//! `exclusive: false` is the `concurrent` policy: no lease, append-only.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use crate::db::{
    self, Acquire, Acquired, Commit, Committed, EventWrite, Fenced, ItemPage, Receive, Resolution,
    TaskPage, TaskQuery,
};
use crate::lease::LanePoll;
use crate::run::{
    self, LEASE_WAIT_TIMEOUT, LOOKUP_LIMIT, PENDING_ASK_LIMIT, RunOutcome, RunSummary, RunWrites,
    SUBMIT_CHANNEL, TaskPatch, TaskState, WaitOptions,
};
use crate::schema::*;
use crate::sql::{Bind, Row, SqlValue, Statement, Table};
use crate::{Fence, StorageError, TaskWrite};
use livekit_protocol::agent::chat_context::ChatItem;
use lk_agents_abi::storage::context::{AppStatement, ContextScope, ContextScopeWrite};
use lk_agents_abi::storage::journal::{JournalEvent, RunEnded, RunStarted};
use lk_agents_abi::usage::Usage;
use serde_json::{Map, Value};

use crate::lease::Lease;
use crate::{ThreadDbConnection, shortuuid};
use lk_agents_net::clock::unix_now_ms;

/// How long a lease lasts and how often its holder renews it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaseTiming {
    pub ttl: Duration,
    pub renew_interval: Duration,
}

impl LeaseTiming {
    pub fn new(ttl: Duration, renew_interval: Duration) -> Result<Self, StorageError> {
        if ttl.is_zero() || ttl.as_secs() >= i64::MAX as u64 {
            return Err(StorageError::Invalid(
                "lease_ttl must be positive and fit in signed seconds".into(),
            ));
        }
        if renew_interval.is_zero() {
            return Err(StorageError::Invalid("renew_interval must be positive".into()));
        }
        Ok(Self { ttl, renew_interval })
    }

    /// The TTL as the whole seconds the database counts, rounded up.
    pub(crate) fn secs(&self) -> i64 {
        (self.ttl.as_secs() + u64::from(self.ttl.subsec_nanos() > 0)) as i64
    }
}

impl Default for LeaseTiming {
    /// A 10 s lease, measured on the database's clock, renewed every 3 s.
    fn default() -> Self {
        Self { ttl: Duration::from_secs(10), renew_interval: Duration::from_secs(3) }
    }
}

/// What the engine knows of a database's schema: `None` until a probe or a write found it.
pub type SchemaCell = Arc<Mutex<Option<i64>>>;

/// The thread a worker opened: its database over one connection, shared by every lane. Cheap to
/// clone.
#[derive(Clone)]
pub struct Thread {
    conn: Arc<dyn ThreadDbConnection>,
    thread_id: Arc<str>,
    pub(crate) timing: LeaseTiming,
    schema: SchemaCell,
    /// `<worker holder>/<handle>`: what this handle's leases and waiter marks carry.
    holder: Arc<str>,
}

impl Thread {
    /// Over an open connection; `schema` is what is known of its schema, probed when unknown.
    /// Leases carry this handle as their holder, so a second request of the same worker on a lane
    /// waits like any other worker's.
    pub async fn new(
        conn: Arc<dyn ThreadDbConnection>,
        thread_id: &str,
        timing: LeaseTiming,
        schema: SchemaCell,
        holder: &str,
    ) -> Result<Self, StorageError> {
        let holder = format!("{holder}/{}", shortuuid("")).into();
        let thread = Self { conn, thread_id: thread_id.into(), timing, schema, holder };
        if thread.schema_missing() {
            let found = db::schema_version(&*thread.conn).await?;
            *thread.schema.lock() = found;
        }
        Ok(thread)
    }

    pub fn id(&self) -> &str {
        &self.thread_id
    }

    pub fn holder(&self) -> &str {
        &self.holder
    }

    /// Done with the database: the backend releases or parks the connection.
    pub async fn close(&self) {
        self.conn.close().await;
    }

    pub fn lane(&self, name: &str) -> Lane {
        Lane { thread: self.clone(), name: name.to_owned() }
    }

    pub(crate) fn conn(&self) -> &Arc<dyn ThreadDbConnection> {
        &self.conn
    }

    fn schema_missing(&self) -> bool {
        self.schema.lock().is_none()
    }

    fn schema_applied(&self) {
        *self.schema.lock() = Some(THREAD_SCHEMA_VERSION);
    }

    /// A competing initializer may have committed while this batch failed. Only an
    /// authoritative supported schema permits retry. A failed probe explains the failure better
    /// than the batch's error, except an `Uncertain` batch, which may still have applied.
    async fn reconcile_schema(&self, original: StorageError) -> Result<(), StorageError> {
        match db::schema_version(&*self.conn).await {
            Ok(Some(_)) => Ok(()),
            Ok(None) => Err(original),
            Err(_) if matches!(original, StorageError::Uncertain(_)) => Err(original),
            Err(error) => Err(error),
        }
    }

    /// Asks the database again whether the schema is there: another worker may have created it
    /// since this connection found none. True when present now.
    async fn refresh_schema(&self) -> Result<bool, StorageError> {
        if !self.schema_missing() {
            return Ok(true);
        }
        match db::schema_version(&*self.conn).await? {
            Some(_) => {
                self.schema_applied();
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// The DDL to prepend to a write when the database has none yet.
    fn bootstrap(&self) -> Vec<Statement> {
        if self.schema_missing() { db::bootstrap() } else { Vec::new() }
    }

    /// Applies the DDL when the schema is missing.
    pub(crate) async fn ensure_schema(&self) -> Result<(), StorageError> {
        if self.refresh_schema().await? {
            return Ok(());
        }
        match self.conn.batch(db::bootstrap()).await {
            Ok(_) => {}
            Err(error) => self.reconcile_schema(error).await?,
        }
        self.schema_applied();
        Ok(())
    }

    /// One read's rows, each decoded; a database with no schema holds nothing.
    async fn read<S: Bind, T>(
        &self,
        statement: S,
        decode: fn(&Row) -> Result<T, StorageError>,
    ) -> Result<Vec<T>, StorageError> {
        if !self.refresh_schema().await? {
            return Ok(Vec::new());
        }
        self.conn.query(statement.bind()).await?.iter().map(decode).collect()
    }

    /// One write statement; how many rows it changed.
    pub(crate) async fn exec<S: Bind>(&self, statement: S) -> Result<i64, StorageError> {
        let out = self.conn.batch(vec![statement.bind()]).await?;
        Ok(out.at(0)?.rows_affected())
    }

    /// The journal, every lane, after the event `after_id`, at most `limit` events.
    pub async fn journal(&self, after_id: i64, limit: i64) -> Result<Vec<EventRow>, StorageError> {
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
        db::list_tasks(&*self.conn, q).await
    }

    pub async fn lanes(&self) -> Result<Vec<LaneRow>, StorageError> {
        self.read(SelectLanes {}, LaneRow::from_row).await
    }

    /// One page of the thread's committed items in insertion order, of `lane` when given,
    /// after the opaque `cursor` of the previous page. No lease needed.
    pub async fn list_items(
        &self,
        lane: Option<&str>,
        cursor: Option<&str>,
        limit: i64,
    ) -> Result<ItemPage, StorageError> {
        if !self.refresh_schema().await? {
            return Ok(ItemPage { items: Vec::new(), next: None });
        }
        db::list_items(&*self.conn, lane, cursor, limit).await
    }

    /// A direct field store's committed scope snapshot. Initialization remains runtime-owned.
    pub async fn context_scopes(&self, lane: &str) -> Result<Vec<ContextScope>, StorageError> {
        self.ensure_schema().await?;
        let result = self
            .conn
            .batch(vec![
                Statement::raw(crate::context::db::SCHEMA),
                crate::context::db::select(lane),
            ])
            .await?;
        crate::context::db::from_rows(result.at(1)?.rows())
    }
}

/// One registered agent's part of a thread: every statement of the lane.
#[derive(Clone)]
pub struct Lane {
    thread: Thread,
    name: String,
}

impl Lane {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The inbox receive, one atomic batch: the inbox row, the `submitted` task row and the
    /// lease attempt.
    pub async fn receive(
        &self,
        holder: &str,
        input: db::ReceiveInput<'_>,
    ) -> Result<(Receive, Option<Lease>), StorageError> {
        self.thread.ensure_schema().await?;
        let token = shortuuid("lease_");
        let holder = format!("{holder}/{token}");
        let lease_secs = self.thread.timing.secs();
        let conn = &*self.thread.conn;
        let received = db::receive(conn, &self.name, &holder, &token, lease_secs, input).await?;
        let lease = match received {
            Receive::Received(Some(version)) => Some(Lease::new(
                self.thread.clone(),
                Fence { lane: self.name.clone(), holder, token, version },
            )),
            _ => None,
        };
        Ok((received, lease))
    }

    /// A turn's `submitted` row, before the lease is tried; `false` when its task or message id
    /// already had one.
    pub async fn submit(&self, task: &TaskWrite) -> Result<bool, StorageError> {
        self.thread.ensure_schema().await?;
        Ok(self.thread.exec(db::submitted_task(&self.name, task)).await? > 0)
    }

    /// The `concurrent` policy's append-only write: no lease, no fence, no version bump.
    pub async fn append(&self, commit: &Commit) -> Result<Committed, StorageError> {
        self.thread.ensure_schema().await?;
        db::commit(&*self.thread.conn, &self.name, commit, None).await
    }

    /// Records a worker waiting under `interrupt`; `false` when the lease is not held, in which
    /// case the caller acquires instead.
    pub async fn set_waiter(&self, holder: &str) -> Result<bool, StorageError> {
        self.thread.ensure_schema().await?;
        Ok(self.thread.exec(SetWaiter { lane: &self.name, holder }).await? > 0)
    }

    /// A turn's checkpoint, one batch: the lane's waiter and the task's status.
    pub async fn checkpoint(
        &self,
        task_id: &str,
    ) -> Result<(Option<String>, Option<TaskState>), StorageError> {
        db::checkpoint(&*self.thread.conn, &self.name, task_id).await
    }

    /// Takes a `submitted` or `working` row to `working`. The status it had (`working` means a
    /// dead holder had it), `None` when the row is terminal or missing.
    pub async fn pickup(&self, task_id: &str) -> Result<Option<TaskState>, StorageError> {
        db::pickup(&*self.thread.conn, &self.name, task_id).await
    }

    /// A `working` row whose holder died ends `failed` with the given status message.
    pub async fn fail_orphan(
        &self,
        task_id: &str,
        status_message: &a2a::Message,
    ) -> Result<bool, StorageError> {
        let status_message = SqlValue::json(status_message);
        Ok(self.thread.exec(FailOrphan { lane: &self.name, task_id, status_message }).await? > 0)
    }

    /// The whole history; how another lane reads it.
    pub async fn history(&self) -> Result<Vec<ItemRow>, StorageError> {
        self.thread.read(SelectItems { lane: &self.name }, ItemRow::from_row).await
    }

    pub async fn info(&self) -> Result<Option<LaneRow>, StorageError> {
        let lane = SelectLane { lane: &self.name };
        Ok(self.thread.read(lane, LaneRow::from_row).await?.into_iter().next())
    }

    /// The newest task rows, of one status when given.
    pub async fn tasks(
        &self,
        status: Option<TaskState>,
        limit: i64,
    ) -> Result<Vec<TaskRow>, StorageError> {
        let lane = &self.name;
        match status {
            Some(status) => {
                let by_status = SelectTasksByStatus { lane, status: status.as_str(), limit };
                self.thread.read(by_status, TaskRow::from_row).await
            }
            None => self.thread.read(SelectTasks { lane, limit }, TaskRow::from_row).await,
        }
    }

    pub async fn task(&self, task_id: &str) -> Result<Option<TaskRow>, StorageError> {
        let task = SelectTask { lane: &self.name, task_id };
        Ok(self.thread.read(task, TaskRow::from_row).await?.into_iter().next())
    }

    /// The task an A2A `messageId` created.
    pub async fn task_for_message(
        &self,
        message_id: &str,
    ) -> Result<Option<TaskRow>, StorageError> {
        let task = SelectTaskForMessage { lane: &self.name, message_id };
        Ok(self.thread.read(task, TaskRow::from_row).await?.into_iter().next())
    }

    /// The inbox rows in `id` order: every one, or only those above `drained_id`.
    pub async fn inbox(&self, pending: bool) -> Result<Vec<InboxRow>, StorageError> {
        let lane = &self.name;
        match pending {
            true => self.thread.read(SelectInboxPending { lane }, InboxRow::from_row).await,
            false => self.thread.read(SelectInbox { lane }, InboxRow::from_row).await,
        }
    }

    /// The lane's journal after the event `after_id`.
    pub async fn journal(&self, after_id: i64) -> Result<Vec<EventRow>, StorageError> {
        let events = SelectLaneEvents { lane: &self.name, after_id };
        self.thread.read(events, EventRow::from_row).await
    }

    /// The last `limit` runs, newest first; the `submitted` insert before a lease is
    /// bookkeeping, not a turn.
    pub async fn runs(&self, limit: usize) -> Result<Vec<RunSummary>, StorageError> {
        let rows = self.journal(0).await?;
        let mut runs: Vec<RunSummary> = RunSummary::from_events(&rows)?
            .into_iter()
            .filter(|r| r.channel.as_deref() != Some(SUBMIT_CHANNEL))
            .collect();
        let keep = runs.len().saturating_sub(limit);
        runs.drain(..keep);
        runs.reverse();
        Ok(runs)
    }

    /// `tasks/cancel`: a row not yet terminal goes `canceled` with `metadata` merged in; a
    /// running turn yields at its next checkpoint. The row after, unchanged when it was already
    /// terminal; `None` when the task is not the lane's.
    pub async fn cancel_task(
        &self,
        task_id: &str,
        metadata: Map<String, Value>,
    ) -> Result<Option<TaskRow>, StorageError> {
        self.thread.ensure_schema().await?;
        db::cancel(&*self.thread.conn, self.name(), task_id, &metadata).await
    }

    /// Takes the lane and loads it. The run commits with [`Run::commit`] and ends by error with
    /// [`Run::abandon`]; a run dropped without either stops renewing and lets its lease lapse,
    /// which the next acquire sees as a `run_started` without a `run_ended`.
    pub async fn open(&self, opts: RunOptions) -> Result<Run, StorageError> {
        let holder = format!("{}/{}", self.thread.holder, shortuuid("run_"));
        self.open_with_holder(opts, &holder).await
    }

    /// Transfer a batch drain's existing lease into its next run deliberately.
    pub async fn resume(&self, opts: RunOptions, lease: Lease) -> Result<Run, StorageError> {
        let result = self.open_with_holder(opts, lease.holder()).await;
        lease.stop();
        result
    }

    async fn open_with_holder(&self, opts: RunOptions, holder: &str) -> Result<Run, StorageError> {
        let (lease, closed, state, items, tasks, picked_up, context_scopes) = match opts.exclusive {
            true => {
                let a = self.acquire(&opts, holder).await?;
                let lease = Lease::new(self.thread.clone(), a.fence.clone());
                if !a.already_committed {
                    lease.keep_alive();
                }
                let tasks = Tasks::loaded(self.clone(), a.tasks);
                (
                    Some(lease),
                    a.already_committed,
                    a.lane.state,
                    a.items,
                    tasks,
                    a.picked_up,
                    a.context_scopes,
                )
            }
            false => {
                // without a lease nothing is claimed: the run goes only while its row is still
                // `submitted` (a client did not cancel it), and the row stays so, never an
                // orphan for a lease holder to fail
                let picked_up = opts.pickup
                    && self.task(&opts.task_id).await?.map(|r| r.status)
                        == Some(TaskState::Submitted);
                let items = self.history().await?;
                (None, false, None, items, Tasks::new(self.clone()), picked_up, Vec::new())
            }
        };
        Ok(Run {
            lane: self.clone(),
            lease,
            holder: holder.to_owned(),
            run_id: opts.run_id,
            task_id: opts.task_id,
            channel: opts.channel,
            closed,
            loaded: items.iter().map(|i| (i.id.clone(), i.payload.clone())).collect(),
            history: items,
            loaded_state: state.clone(),
            state,
            context_scopes,
            scope_writes: Vec::new(),
            app_writes: Vec::new(),
            events: EventLog::default(),
            tasks,
            picked_up,
            usage: Usage::default(),
            outcome: RunOutcome::Completed,
            reason: None,
            finished: false,
            unresolved: None,
        })
    }

    /// The acquire, polled with backoff while another holder has the lane, as the lane's `waiter`
    /// under `interrupt`.
    async fn acquire(&self, opts: &RunOptions, holder: &str) -> Result<Acquired, StorageError> {
        let epoch = tokio::time::Instant::now();
        let mut poll = LanePoll::new(epoch.elapsed(), opts.wait_timeout);
        let mut marked = false;
        loop {
            let token = shortuuid("lease_");
            let req = Acquire {
                lane: self.name(),
                holder,
                token: &token,
                lease_secs: self.thread.timing.secs(),
                wait_ms: LEASE_WAIT_TIMEOUT.as_millis() as i64,
                run_id: &opts.run_id,
                task_id: &opts.task_id,
                run_started: EventWrite::new(
                    unix_now_ms(),
                    JournalEvent::RunStarted(RunStarted {
                        task_id: Some(opts.task_id.clone()),
                        stream_id: Some(opts.task_id.clone()),
                        channel: opts.channel.clone(),
                        exclusive: None,
                    }),
                ),
                pickup: opts.pickup,
                tasks_limit: LOOKUP_LIMIT,
            };
            // the DDL rides along on a fresh database; a batch that loses the race to another
            // creator's DDL is retried without it
            let bootstrap = self.thread.bootstrap();
            let initializing = !bootstrap.is_empty();
            let conn = &*self.thread.conn;
            let acquired = match db::acquire(conn, &req, bootstrap).await {
                Err(error) if initializing => {
                    self.thread.reconcile_schema(error).await?;
                    self.thread.schema_applied();
                    db::acquire(conn, &req, Vec::new()).await
                }
                out => out,
            }?;
            self.thread.schema_applied();
            if let Some(acquired) = acquired {
                return Ok(acquired);
            }
            if !opts.wait {
                return Err(StorageError::Rejected { running_task_id: None });
            }
            if opts.interrupt && !marked {
                marked = self.set_waiter(holder).await?;
                if marked {
                    log::debug!(target: "livekit.agents", lane = self.name(), holder = holder; "waiting for the lane as its waiter");
                }
            }
            let Some(at) = poll.next(epoch.elapsed()) else {
                log::debug!(target: "livekit.agents", lane = self.name(); "the lane was not released within {}s", opts.wait_timeout.as_secs_f64());
                return Err(StorageError::Rejected { running_task_id: None });
            };
            tokio::time::sleep_until(epoch + at).await;
        }
    }
}

/// How [`Lane::open`] takes the lane.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Idempotency key: a `run_id` that already committed yields a closed run.
    pub run_id: String,
    /// The A2A task this run serves.
    pub task_id: String,
    /// `false` is the `concurrent` policy: no lease, append-only commit.
    pub exclusive: bool,
    /// Wait for a lease another holder has; `false` fails with [`StorageError::Rejected`].
    pub wait: bool,
    /// Record this worker as the lane's `waiter` while it waits, so the holder yields.
    pub interrupt: bool,
    /// The request's channel, for the `run_started` event.
    pub channel: Option<String>,
    pub wait_timeout: Duration,
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
            pickup: false,
        }
    }
}

/// What this run appends to the lane's journal, stamped as recorded.
#[derive(Default)]
pub struct EventLog {
    new: Vec<EventWrite>,
}

impl EventLog {
    pub fn append(&mut self, event: JournalEvent) {
        self.new.push(EventWrite::new(unix_now_ms(), event));
    }
}

/// How long a commit whose acknowledgment was lost is resolved before the caller is answered
/// [`StorageError::Uncertain`]; the run then still holds the commit for [`Run::resolve_later`].
pub const RESOLVE_WAIT: Duration = Duration::from_secs(10);
/// How long a resolver keeps checking a commit nobody could resolve in the caller's wait.
pub const RESOLVE_LATER: Duration = Duration::from_secs(3600);

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
    lane: Lane,
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
    pub context_scopes: Vec<ContextScope>,
    pub scope_writes: Vec<ContextScopeWrite>,
    /// Application statements managed transactions staged, applied first in the commit.
    pub app_writes: Vec<AppStatement>,
    loaded_state: Option<Value>,
    pub events: EventLog,
    pub tasks: Tasks,
    /// Whether the acquire's `pickup` took the row to `working`.
    pub picked_up: bool,
    pub usage: Usage,
    pub outcome: RunOutcome,
    pub reason: Option<String>,
    finished: bool,
    /// The commit [`Run::commit`] could not resolve in its wait, and whether it kept the lease.
    unresolved: Option<(Commit, bool)>,
}

impl Run {
    pub fn lease(&self) -> Option<&Lease> {
        self.lease.as_ref()
    }

    /// The version loaded: the commit's fence; `0` for a run that took no lease.
    pub fn version(&self) -> i64 {
        self.lease.as_ref().map_or(0, |l| l.fence().version)
    }

    /// The run's writes, as the core composes them.
    pub fn writes(&self, opts: CommitOptions) -> Result<Commit, StorageError> {
        let mut commit = run::run_commit(RunWrites {
            run_id: &self.run_id,
            task_id: &self.task_id,
            channel: self.channel.as_deref(),
            exclusive: self.lease.is_some(),
            history: &self.history,
            loaded: &self.loaded,
            state: self.state.as_ref(),
            loaded_state: self.loaded_state.as_ref(),
            events: &self.events.new,
            tasks: self.tasks.writes(),
            usage: &self.usage,
            outcome: self.outcome,
            reason: self.reason.as_deref(),
            now_ms: unix_now_ms(),
            drained_id: opts.drained_id,
            answer_task: opts.answer_task,
        })?;
        commit.context_scopes = self.context_scopes.clone();
        commit.scope_writes = self.scope_writes.clone();
        commit.app_writes = self.app_writes.clone();
        Ok(commit)
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

    /// Leaving normally: one fenced write of everything the run holds. A closed run releases the
    /// lease and writes nothing. `VersionConflict` means the run outlived its lease.
    pub async fn commit(&mut self, opts: CommitOptions) -> Result<Committed, StorageError> {
        self.finish()?;
        let version = self.version();
        match &self.lease {
            Some(lease) if self.closed => {
                lease.release(None).await?;
                Ok(Committed { version, task: None })
            }
            _ => {
                if self.lease.is_none() {
                    self.lane.thread.ensure_schema().await?;
                }
                let keep = opts.keep_lease;
                let commit = self.writes(opts)?;
                match self.send(&commit, keep).await {
                    Err(StorageError::Uncertain(why)) => {
                        log::warn!(target: "livekit.agents", lane = self.lane.name(), runId = self.run_id.as_str(); "commit acknowledgment lost, resolving: {why}");
                        let resolved = self.resolve(&commit, keep, RESOLVE_WAIT).await;
                        if matches!(resolved, Err(StorageError::Uncertain(_))) {
                            self.unresolved = Some((commit, keep));
                        }
                        resolved
                    }
                    Err(StorageError::Invalid(why)) if !commit.app_writes.is_empty() => {
                        Err(self.refused_commit(&commit, why).await)
                    }
                    out => out,
                }
            }
        }
    }

    /// A commit carrying application SQL the database refused: a lane that moved on aborted it
    /// at its fence (`VersionConflict`); otherwise an application statement was refused.
    async fn refused_commit(&self, commit: &Commit, why: String) -> StorageError {
        let fence = self.lease.as_ref().map(Lease::fence);
        let (conn, read_task) = (&*self.lane.thread.conn, commit.read_task.as_deref());
        match db::resolve_commit(conn, self.lane.name(), &self.run_id, fence, read_task).await {
            Ok(Resolution::Rejected) => StorageError::VersionConflict {
                lane: self.lane.name().to_owned(),
                expected: fence.map_or(0, |f| f.version),
            },
            _ => StorageError::Invalid(why),
        }
    }

    /// Sends the run's one write, fenced on its lease when it holds one.
    async fn send(&self, commit: &Commit, keep: bool) -> Result<Committed, StorageError> {
        let fenced = self.lease.as_ref().map(|lease| Fenced {
            fence: lease.fence(),
            keep: keep.then(|| self.lane.thread.timing.secs()),
        });
        db::commit(&*self.lane.thread.conn, self.lane.name(), commit, fenced).await
    }

    /// A commit whose acknowledgment was lost, resolved by the run's identity: committed (its
    /// record), not applied while the fence holds (sent again: the fence and the run id make it
    /// safe), or rejected (`VersionConflict`: the lane moved on without it). `Uncertain` once
    /// `wait` passes with no authoritative answer.
    async fn resolve(
        &self,
        commit: &Commit,
        keep: bool,
        wait: Duration,
    ) -> Result<Committed, StorageError> {
        let fence = self.lease.as_ref().map(Lease::fence);
        let epoch = tokio::time::Instant::now();
        let mut poll = LanePoll::new(epoch.elapsed(), wait);
        let mut last: String;
        loop {
            let (conn, read_task) = (&*self.lane.thread.conn, commit.read_task.as_deref());
            match db::resolve_commit(conn, self.lane.name(), &self.run_id, fence, read_task).await {
                Ok(Resolution::Committed(committed)) => return Ok(*committed),
                Ok(Resolution::Rejected) => {
                    return Err(StorageError::VersionConflict {
                        lane: self.lane.name().to_owned(),
                        expected: fence.map_or(0, |f| f.version),
                    });
                }
                Ok(Resolution::NotApplied) => match self.send(commit, keep).await {
                    Err(StorageError::Uncertain(why) | StorageError::Unavailable(why)) => {
                        last = why
                    }
                    out => return out,
                },
                Err(e) => last = e.to_string(),
            }
            let Some(at) = poll.next(epoch.elapsed()) else {
                return Err(StorageError::Uncertain(format!(
                    "run {} unresolved after {}s: {last}",
                    self.run_id,
                    wait.as_secs_f64()
                )));
            };
            tokio::time::sleep_until(epoch + at).await;
        }
    }

    /// Keeps resolving a commit [`Run::commit`] left uncertain, up to [`RESOLVE_LATER`]: the
    /// authoritative outcome, never a guess. `None` when nothing was left unresolved.
    pub async fn resolve_later(mut self) -> Option<Result<Committed, StorageError>> {
        let (commit, keep) = self.unresolved.take()?;
        Some(self.resolve(&commit, keep, RESOLVE_LATER).await)
    }

    /// Leaving by error: `run_ended` with the outcome and the reason, and the release; nothing
    /// else. A `run_ended` that could not be written is logged.
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
        let ended = EventWrite::new(
            unix_now_ms(),
            JournalEvent::RunEnded(RunEnded { outcome, reason: Some(reason.into()) }),
        );
        if let Err(e) = lease.release(Some((&self.run_id, &ended))).await {
            log::warn!(target: "livekit.agents", lane = self.lane.name(), runId = self.run_id.as_str(); "run_ended could not be written after a failed run: {e}");
        }
        Ok(())
    }
}

/// The lane's task rows as this run writes them.
pub struct Tasks {
    lane: Lane,
    /// In write order.
    writes: Vec<TaskWrite>,
    /// The lane's newest rows as the acquire loaded them, newest first: what the run's lookups
    /// read instead of the database. Complete when shorter than [`LOOKUP_LIMIT`].
    loaded: Option<Vec<TaskRow>>,
}

impl Tasks {
    pub fn new(lane: Lane) -> Self {
        Self { lane, writes: Vec::new(), loaded: None }
    }

    /// Over the rows an acquire loaded, newest first.
    pub fn loaded(lane: Lane, loaded: Vec<TaskRow>) -> Self {
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

    /// The requested pending question, or the newest one when no task was named.
    pub async fn pending_ask(
        &self,
        task_id: Option<&str>,
    ) -> Result<Option<TaskRow>, StorageError> {
        if let Some(task_id) = task_id {
            return Ok(self
                .find(|row| row.task_id == task_id, self.lane.task(task_id))
                .await?
                .filter(|row| {
                    row.status == TaskState::InputRequired
                        && row.internal.as_ref().is_some_and(|internal| internal.ask.is_some())
                }));
        }

        if let Some(rows) = self.snapshot() {
            return Ok(rows
                .iter()
                .find(|r| {
                    r.status == TaskState::InputRequired
                        && r.internal.as_ref().is_some_and(|i| i.ask.is_some())
                })
                .cloned());
        }
        Ok(self
            .lane
            .tasks(Some(TaskState::InputRequired), PENDING_ASK_LIMIT)
            .await?
            .into_iter()
            .find(|r| r.internal.as_ref().is_some_and(|i| i.ask.is_some())))
    }

    /// Insert or update one row; what the patch leaves `None` keeps the existing row's value. A
    /// stored row that is already terminal (a client canceled it mid-turn) stays as it is, as the
    /// commit would leave it; rewriting this run's own terminal write is refused.
    pub async fn write(
        &mut self,
        task_id: &str,
        status: TaskState,
        patch: TaskPatch,
    ) -> Result<(), StorageError> {
        let base = match self.writes.iter().find(|w| w.task_id == task_id) {
            Some(w) => Some(w.clone()),
            None => match self.find(|r| r.task_id == task_id, self.lane.task(task_id)).await? {
                Some(row) if row.terminal() => return Ok(()),
                row => row.map(TaskWrite::from),
            },
        };
        let write = run::merge_write(task_id, base.as_ref(), status, patch)?;
        match self.writes.iter().position(|w| w.task_id == task_id) {
            Some(i) => self.writes[i] = write,
            None => self.writes.push(write),
        }
        Ok(())
    }

    /// `input-required`: the question is the status message, `ask` the pending transition.
    pub async fn wait(
        &mut self,
        task_id: &str,
        question: &str,
        opts: WaitOptions,
    ) -> Result<(), StorageError> {
        let patch = run::wait_patch(task_id, question, opts, shortuuid("msg_"));
        self.write(task_id, TaskState::InputRequired, patch).await?;
        Ok(())
    }
}
