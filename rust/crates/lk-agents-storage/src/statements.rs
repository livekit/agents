//! Every statement a thread database sees (plan 13.5), each one method of [`ThreadDb`], each the
//! SQL of plan 3.3.2, 3.3.4 and 5.2. Nothing else issues SQL against a thread database; Python
//! never sees SQL.
//!
//! * [`ThreadDb::acquire`]: the acquire and the load as one atomic `Batch`: the DDL when
//!   `schema_version` is missing, the lane bootstrap, the lease `UPDATE .. RETURNING version`, the
//!   `run_started` event as a conditional insert so a failed acquire writes nothing, then the reads
//!   of the items, the events and the lane row, discarded when the acquire returned no row. Two
//!   statements the plan's list leaves implicit: the holder that acquired clears its own `waiter`
//!   mark (a waiter that won would otherwise yield to itself at its first boundary), and a read of
//!   whether this `run_id` already committed, so a retried run yields a closed `Run` and writes
//!   nothing (plan 5.2, rule 4 of 5.3); `run_started` is not written again for it.
//! * [`ThreadDb::renew`], and [`ThreadDb::start_renewal`]: the timer that issues it at the
//!   configured interval while a run is open and marks the lease lost when a renew returns no
//!   row or the connection fails, which cancels the run with `StorageUnavailable`.
//! * [`ThreadDb::commit`]: one `Batch`, fenced on the version loaded and on the lease token: every
//!   insert is conditional on the fence, the lane update bumps `version` and releases or renews;
//!   zero rows from the fence is `VersionConflict`. Without a fence it is the `concurrent` policy's
//!   append-only commit.
//! * [`ThreadDb::run_ended`]: the exception path, `run_ended` plus release, nothing else.
//! * [`ThreadDb::receive`]: the inbox transaction, [`ThreadDb::drain_check`], [`ThreadDb::set_waiter`],
//!   [`ThreadDb::pickup_task`] and [`ThreadDb::mark_orphan_failed`], and the typed reads.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::StorageError;
use crate::run::LEASE_WAIT_TIMEOUT;
use crate::schema::{
    self, ChatItemRow, EventRow, InboxRow, LaneRow, SCHEMA_VERSION, TaskData, TaskInternal, TaskRow,
};
use crate::wire::{SqlValue, statement, wire as pb};
use crate::wsdb::{Conn, StatementResult, WsdbError};

mod sql;
use sql::*;

/// The default lease TTL, measured on agent-db's clock.
pub const DEFAULT_LEASE_TTL: Duration = Duration::from_secs(10);
/// The default interval at which an open run renews.
pub const DEFAULT_RENEW_INTERVAL: Duration = Duration::from_secs(3);
/// Plan 5.3 rule 6: state under 64 KiB.
pub const STATE_LIMIT: usize = 64 * 1024;

// ---------------------------------------------------------------------------------------------
// requests and results
// ---------------------------------------------------------------------------------------------

/// The acquire and the load (plan 3.3.2), one batch: the lane under `holder`'s lease, with the
/// `run_started` event written only when the lease was taken.
#[derive(Debug, Clone, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage")]
pub struct AcquireRequest {
    pub lane: String,
    /// The worker, for re-entry (`lease_holder = ?`).
    pub holder: String,
    pub lease_token: String,
    pub run_id: String,
    pub ts_ms: i64,
    /// The run's task: a `submitted` row of it orders the acquire behind older waiters; empty
    /// for a run with no row (a voice turn).
    #[serde(default)]
    #[python(default = "\"\"")]
    pub task_id: String,
    /// The `run_started` event's payload: the task id, the stream id, the channel.
    pub run_started: Value,
    /// Load the lane's events after this id (0: all).
    #[serde(default)]
    #[python(default = "0")]
    pub events_after: i64,
}

/// What an acquire loaded: the lane at `version`.
#[derive(Debug, Clone, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage", config = "frozen=True")]
pub struct Loaded {
    pub version: i64,
    /// Plan 5.2: this `run_id` already committed; the caller releases without running.
    pub already_committed: bool,
    pub lane: LaneRow,
    pub items: Vec<ChatItemRow>,
    pub events: Vec<EventRow>,
    pub tip: i64,
}

/// One history item a commit writes: an upsert on `id`.
#[derive(Debug, Clone, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage")]
pub struct ItemWrite {
    pub id: String,
    #[serde(default)]
    #[python(default = "None")]
    pub message_id: Option<String>,
    pub payload: Value,
}

/// One journal entry a commit appends; `run_ended` is one of them, written last.
#[derive(Debug, Clone, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage")]
pub struct EventWrite {
    pub ts_ms: i64,
    #[serde(rename = "type")]
    #[host(rename = "type")]
    pub r#type: String,
    pub payload: Value,
}

/// One task row a commit upserts: every column but `lane`, `seq` and `updated_at`, which the
/// statement fills.
#[derive(Debug, Clone, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage")]
pub struct TaskWrite {
    pub task_id: String,
    pub status: String,
    #[serde(default)]
    #[python(default = "None")]
    pub message_id: Option<String>,
    /// The `AgentTask` class; `None` for a turn.
    #[serde(default)]
    #[python(default = "None")]
    pub name: Option<String>,
    #[serde(default)]
    #[python(default = "None")]
    pub status_message: Option<Value>,
    pub data: TaskData,
    #[serde(default)]
    #[python(default = "None")]
    pub internal: Option<TaskInternal>,
}

/// Release (the default) or renew when the runner continues with pending inbox rows or due tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Host)]
#[serde(rename_all = "lowercase")]
#[python(literal, module = "storage")]
pub enum LeaseAfter {
    #[host(rename = "release")]
    Release,
    #[host(rename = "renew")]
    Renew,
}

/// What a fenced commit is conditioned on: the lane at the version the run loaded, under the
/// lease token it holds.
#[derive(Debug, Clone, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage")]
pub struct Fence {
    pub version: i64,
    pub lease_token: String,
}

/// The commit (plan 5.2): every write of a run as one batch, fenced on the version loaded.
#[derive(Debug, Clone, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage")]
pub struct CommitRequest {
    pub lane: String,
    pub run_id: String,
    /// `None` is the `concurrent` policy: no lease, no version bump, appends only.
    #[serde(default)]
    #[python(default = "None")]
    pub fence: Option<Fence>,
    #[serde(default)]
    #[python(default_factory = "list")]
    pub items: Vec<ItemWrite>,
    #[serde(default)]
    #[python(default_factory = "list")]
    pub delete_items: Vec<String>,
    #[serde(default)]
    #[python(default_factory = "list")]
    pub events: Vec<EventWrite>,
    #[serde(default)]
    #[python(default_factory = "list")]
    pub tasks: Vec<TaskWrite>,
    /// `None` keeps the stored snapshot.
    #[serde(default)]
    #[python(default = "None")]
    pub state: Option<Value>,
    #[serde(default)]
    #[python(default = "None")]
    pub agent_stack: Option<Value>,
    #[serde(default)]
    #[python(default = "None")]
    pub drained_id: Option<i64>,
    /// The `run_ended` event, written last.
    pub run_ended: EventWrite,
    #[serde(default = "release")]
    #[python(default = "\"release\"")]
    pub after: LeaseAfter,
}

fn release() -> LeaseAfter {
    LeaseAfter::Release
}

/// What a commit wrote: the lane's version after it and the connection's durable tip.
#[derive(Debug, Clone, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage", config = "frozen=True")]
pub struct CommitResult {
    /// The version after the bump; the loaded one under `concurrent`.
    pub version: i64,
    pub tip: i64,
}

/// Leaving by exception: the `run_ended` event with the outcome, and the release.
#[derive(Debug, Clone, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage")]
pub struct RunEndedRequest {
    pub lane: String,
    pub run_id: String,
    pub lease_token: String,
    pub ts_ms: i64,
    pub payload: Value,
}

/// The inbox receive (plan 3.3.4): the row, the task it creates, and the lease attempt, one
/// transaction.
#[derive(Debug, Clone, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage")]
pub struct ReceiveRequest {
    pub lane: String,
    pub holder: String,
    pub lease_token: String,
    pub task_id: String,
    pub message_id: String,
    pub message: Value,
    /// The `AgentTask` class; `None` for a turn.
    #[serde(default)]
    #[python(default = "None")]
    pub name: Option<String>,
    #[serde(default)]
    #[python(default_factory = "TaskData")]
    pub data: TaskData,
    #[serde(default)]
    #[python(default = "None")]
    pub internal: Option<TaskInternal>,
}

/// The early `submitted` row of [`ThreadDb::insert_submitted_task`]: the columns the inbox
/// receive fills, minus the inbox row and the lease.
#[derive(Debug, Clone, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage")]
pub struct SubmittedTask {
    pub lane: String,
    pub task_id: String,
    #[serde(default)]
    #[python(default = "None")]
    pub message_id: Option<String>,
    /// The `AgentTask` class; `None` for a turn.
    #[serde(default)]
    #[python(default = "None")]
    pub name: Option<String>,
    #[serde(default)]
    #[python(default_factory = "TaskData")]
    pub data: TaskData,
    #[serde(default)]
    #[python(default = "None")]
    pub internal: Option<TaskInternal>,
}

/// The receive inserted the row: its id, and the version when this worker took the lease and
/// is the runner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(
    pydantic,
    module = "storage",
    tag = "outcome",
    tag_value = "received",
    config = "frozen=True"
)]
pub struct Received {
    pub id: i64,
    pub acquired: Option<i64>,
}

/// The `messageId` was in the inbox already: nothing inserted; the task it belongs to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(
    pydantic,
    module = "storage",
    tag = "outcome",
    tag_value = "duplicate",
    config = "frozen=True"
)]
pub struct Duplicate {
    pub task_id: Option<String>,
}

/// What [`ThreadDb::receive`] answers, told apart by `outcome`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[serde(tag = "outcome", rename_all = "lowercase")]
#[python(union, module = "storage", discriminator = "outcome")]
pub enum Receive {
    Received(Received),
    Duplicate(Duplicate),
}

/// The drain check: how many inbox rows are pending, and whether the lease was released.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage", config = "frozen=True")]
pub struct DrainCheck {
    pub pending: i64,
    pub released: bool,
}

// ---------------------------------------------------------------------------------------------
// the thread database
// ---------------------------------------------------------------------------------------------

/// One session's database over one connection (plan 13.4), closed at run end.
#[derive(Clone)]
pub struct ThreadDb {
    conn: Conn,
    lease_ttl_secs: i64,
    renew_interval: Duration,
    /// The schema version found at open, `None` when the database has no schema yet.
    schema: Arc<Mutex<Option<i64>>>,
}

impl ThreadDb {
    /// Reads the schema version: missing means the first write applies the DDL, behind means a
    /// migration (none in v1), ahead means a newer SDK wrote this database.
    pub async fn open(conn: Conn) -> Result<Self, StorageError> {
        Self::open_with_timing(conn, DEFAULT_LEASE_TTL, DEFAULT_RENEW_INTERVAL).await
    }

    pub(crate) async fn open_with_timing(
        conn: Conn,
        lease_ttl: Duration,
        renew_interval: Duration,
    ) -> Result<Self, StorageError> {
        let exists = !conn.query_all(TABLE_EXISTS, vec![]).await?.rows.is_empty();
        let version = if exists {
            let set = conn.query_all(SCHEMA_VERSION_READ, vec![]).await?;
            set.rows.first().and_then(|r| r.first()).and_then(SqlValue::as_i64)
        } else {
            None
        };
        if let Some(v) = version
            && v != SCHEMA_VERSION
        {
            return Err(StorageError::Schema(format!(
                "database schema is version {v}, this client writes {SCHEMA_VERSION}"
            )));
        }
        Ok(Self {
            conn,
            lease_ttl_secs: (lease_ttl.as_secs() + u64::from(lease_ttl.subsec_nanos() > 0)) as i64,
            renew_interval,
            schema: Arc::new(Mutex::new(version)),
        })
    }

    pub fn conn(&self) -> &Conn {
        &self.conn
    }

    pub fn close(&self) {
        self.conn.close();
    }

    fn schema_missing(&self) -> bool {
        self.schema.lock().expect("schema").is_none()
    }

    fn schema_applied(&self) {
        *self.schema.lock().expect("schema") = Some(SCHEMA_VERSION);
    }

    /// Asks the database again whether the schema is there. A connection that found no schema
    /// on open must not remember that: another worker may have created it since (two workers
    /// opening a brand-new session at once). True when the schema is present now.
    async fn refresh_schema(&self) -> Result<bool, StorageError> {
        if !self.schema_missing() {
            return Ok(true);
        }
        let exists = !self.conn.query_all(TABLE_EXISTS, vec![]).await?.rows.is_empty();
        if !exists {
            return Ok(false);
        }
        let set = self.conn.query_all(SCHEMA_VERSION_READ, vec![]).await?;
        match set.rows.first().and_then(|r| r.first()).and_then(SqlValue::as_i64) {
            Some(v) if v == SCHEMA_VERSION => {
                self.schema_applied();
                Ok(true)
            }
            Some(v) => Err(StorageError::Schema(format!(
                "database schema is version {v}, this client writes {SCHEMA_VERSION}"
            ))),
            // the tables exist but the version row is not written yet: the creator's batch is
            // in flight; the caller retries
            None => Ok(false),
        }
    }

    /// A DDL batch lost the race against another creator: the schema exists now.
    fn lost_schema_race(e: &dyn std::fmt::Display) -> bool {
        e.to_string().contains("already exists")
    }

    /// The DDL and the `schema_version` row, when the database has no schema yet.
    fn bootstrap(&self) -> Vec<pb::Statement> {
        if !self.schema_missing() {
            return Vec::new();
        }
        let mut out: Vec<pb::Statement> =
            schema::ddl().into_iter().map(|s| statement(s, vec![])).collect();
        out.push(statement(schema::INSERT_SCHEMA_VERSION, vec![SCHEMA_VERSION.into()]));
        out
    }

    /// Applies the DDL when the schema is missing; the paths that are not the acquire batch call
    /// it first.
    pub async fn ensure_schema(&self) -> Result<(), StorageError> {
        if self.refresh_schema().await? {
            return Ok(());
        }
        let stmts = self.bootstrap();
        match self.conn.batch(stmts).await {
            Ok(_) => {
                self.schema_applied();
                Ok(())
            }
            Err(e) if Self::lost_schema_race(&e) => {
                self.schema_applied();
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    /// The acquire and the load, one `Batch`; `None` when the lease is held by another holder.
    pub async fn acquire(&self, req: &AcquireRequest) -> Result<Option<Loaded>, StorageError> {
        // the DDL rides in the acquire batch of a fresh database; a connection that opened before
        // another worker created the schema learns so here, and a batch that loses the race to
        // the DDL is retried without it
        self.refresh_schema().await?;
        match self.acquire_batch(req).await {
            Err(e) if Self::lost_schema_race(&e) => {
                self.schema_applied();
                self.acquire_batch(req).await
            }
            out => out,
        }
    }

    async fn acquire_batch(&self, req: &AcquireRequest) -> Result<Option<Loaded>, StorageError> {
        let lane = req.lane.as_str();
        let token = req.lease_token.as_str();
        let mut stmts = self.bootstrap();
        let ddl = stmts.len();
        stmts.push(statement(INSERT_LANE, vec![lane.into()]));
        stmts.push(statement(
            ACQUIRE,
            vec![
                req.holder.as_str().into(),
                token.into(),
                self.lease_ttl_secs.into(),
                lane.into(),
                req.holder.as_str().into(),
                (LEASE_WAIT_TIMEOUT.as_secs() as i64).into(),
                req.task_id.as_str().into(),
            ],
        ));
        stmts.push(statement(
            CLEAR_WAITER,
            vec![lane.into(), token.into(), req.holder.as_str().into()],
        ));
        stmts.push(statement(
            RUN_STARTED,
            vec![
                req.ts_ms.into(),
                lane.into(),
                req.run_id.as_str().into(),
                SqlValue::json(&req.run_started),
                lane.into(),
                token.into(),
                req.run_id.as_str().into(),
            ],
        ));
        stmts.push(statement(COMMITTED_RUN, vec![lane.into(), req.run_id.as_str().into()]));
        stmts.push(statement(&select_items(), vec![lane.into()]));
        stmts.push(statement(&select_lane_events(), vec![lane.into(), req.events_after.into()]));
        stmts.push(statement(&select_lane(), vec![lane.into()]));
        let out = self.conn.batch(stmts).await?;
        if ddl > 0 {
            self.schema_applied();
        }
        let r = &out.results;
        let Some(version) = first_int(&r[ddl + 1]) else {
            return Ok(None); // not acquired: the reads are discarded
        };
        let already_committed = !r[ddl + 4].rows().is_empty();
        let items =
            r[ddl + 5].rows().iter().map(ChatItemRow::from_row).collect::<Result<_, _>>()?;
        let events = r[ddl + 6].rows().iter().map(EventRow::from_row).collect::<Result<_, _>>()?;
        let lane = r[ddl + 7]
            .rows()
            .first()
            .ok_or_else(|| StorageError::Decode("acquire: no lane row".into()))?;
        Ok(Some(Loaded {
            version,
            already_committed,
            lane: LaneRow::from_row(lane)?,
            items,
            events,
            tip: out.tip,
        }))
    }

    /// Releases the lease without writing anything else: the path of a retried `run_id`
    /// (`Loaded::already_committed`). `true` when this token still held the lane.
    pub async fn release(&self, lane: &str, lease_token: &str) -> Result<bool, StorageError> {
        Ok(self.conn.exec(RELEASE, vec![lane.into(), lease_token.into()]).await?.rows_affected > 0)
    }

    /// `true` while this holder still has the lease.
    pub async fn renew(
        &self,
        lane: &str,
        holder: &str,
        lease_token: &str,
    ) -> Result<bool, StorageError> {
        let set = self
            .conn
            .query_all(
                RENEW,
                vec![self.lease_ttl_secs.into(), lane.into(), holder.into(), lease_token.into()],
            )
            .await?;
        Ok(!set.rows.is_empty())
    }

    /// The renewal timer: [`RENEW`] at the configured interval until [`Lease::stop`], or until a
    /// renew returns no row or fails, which marks the lease lost.
    pub fn start_renewal(&self, lane: &str, holder: &str, lease_token: &str) -> Lease {
        let inner = Arc::new(LeaseInner {
            lane: lane.to_owned(),
            holder: holder.to_owned(),
            token: lease_token.to_owned(),
            lost: AtomicBool::new(false),
            reason: Mutex::new(None),
            notify: tokio::sync::Notify::new(),
            task: Mutex::new(None),
        });
        let db = self.clone();
        let lease = Lease { inner: inner.clone() };
        let task = tokio::spawn(async move {
            let mut tick = tokio::time::interval(db.renew_interval);
            tick.tick().await;
            loop {
                tick.tick().await;
                let outcome = db.renew(&inner.lane, &inner.holder, &inner.token).await;
                let reason = match outcome {
                    Ok(true) => continue,
                    Ok(false) => format!("lease on lane {} was lost", inner.lane),
                    Err(e) => format!("lease renewal on lane {} failed: {e}", inner.lane),
                };
                log::warn!(target: "livekit.agents", lane = inner.lane.as_str(); "{reason}");
                *inner.reason.lock().expect("reason") = Some(reason);
                inner.lost.store(true, Ordering::Release);
                inner.notify.notify_waiters();
                return;
            }
        });
        *lease.inner.task.lock().expect("task") = Some(task);
        lease
    }

    /// The fenced commit (plan 5.2): items, events, task rows, the snapshots and `drained_id`,
    /// `run_ended`, then the lane update that bumps `version` and releases or renews.
    pub async fn commit(&self, req: &CommitRequest) -> Result<CommitResult, StorageError> {
        if let Some(state) = &req.state {
            let size = serde_json::to_vec(state)
                .map_err(|e| StorageError::NotSerializable("state".into(), e.to_string()))?
                .len();
            if size > STATE_LIMIT {
                return Err(StorageError::TooLarge { what: "state".into(), limit: STATE_LIMIT });
            }
        }
        self.ensure_schema().await?;
        let lane = req.lane.as_str();
        let fenced = req.fence.is_some();
        let cond = |mut params: Vec<SqlValue>| -> Vec<SqlValue> {
            params.push(lane.into());
            if let Some(f) = &req.fence {
                params.push(f.version.into());
                params.push(f.lease_token.as_str().into());
            }
            params
        };
        let mut stmts = Vec::new();
        if !fenced {
            // The `concurrent` policy took no lease, so nothing created the lane row yet.
            stmts.push(statement(INSERT_LANE, vec![lane.into()]));
        }
        for item in &req.items {
            stmts.push(statement(
                &insert_item(fenced),
                cond(vec![
                    item.id.as_str().into(),
                    lane.into(),
                    lane.into(),
                    item.message_id.clone().into(),
                    SqlValue::json(&crate::chat_item::encode(&item.payload)?),
                ]),
            ));
        }
        for id in &req.delete_items {
            stmts
                .push(statement(&delete_item(fenced), cond(vec![lane.into(), id.as_str().into()])));
        }
        let event = |e: &EventWrite| -> pb::Statement {
            statement(
                &insert_event(fenced),
                cond(vec![
                    e.ts_ms.into(),
                    lane.into(),
                    req.run_id.as_str().into(),
                    e.r#type.as_str().into(),
                    SqlValue::json(&e.payload),
                ]),
            )
        };
        stmts.extend(req.events.iter().map(event));
        for t in &req.tasks {
            stmts.push(statement(
                &upsert_task(fenced),
                cond(vec![
                    t.task_id.as_str().into(),
                    lane.into(),
                    lane.into(),
                    t.name.clone().into(),
                    t.message_id.clone().into(),
                    t.status.as_str().into(),
                    SqlValue::opt_json(t.status_message.as_ref()),
                    SqlValue::json(&t.data),
                    SqlValue::opt_json(t.internal.as_ref()),
                ]),
            ));
        }
        stmts.push(event(&req.run_ended));
        let last = stmts.len();
        match &req.fence {
            Some(f) => {
                let mut params = vec![
                    SqlValue::opt_json(req.state.as_ref()),
                    SqlValue::opt_json(req.agent_stack.as_ref()),
                    req.drained_id.into(),
                ];
                if req.after == LeaseAfter::Renew {
                    params.push(self.lease_ttl_secs.into());
                }
                params.extend([lane.into(), f.version.into(), f.lease_token.as_str().into()]);
                stmts.push(statement(&update_lane(req.after), params));
            }
            // Appends only, no bump: the version reported is the lane's current one.
            None => stmts.push(statement(&select_lane(), vec![lane.into()])),
        }
        let out = self.conn.batch(stmts).await?;
        let version = match &req.fence {
            Some(_) => first_int(&out.results[last]),
            // The lane row, whose first column is its name: the version is the second.
            None => out.results[last]
                .rows()
                .first()
                .map(LaneRow::from_row)
                .transpose()?
                .map(|l| l.version),
        };
        match (&req.fence, version) {
            (_, Some(version)) => Ok(CommitResult { version, tip: out.tip }),
            (Some(f), None) => {
                Err(StorageError::VersionConflict { lane: req.lane.clone(), expected: f.version })
            }
            (None, None) => Err(StorageError::Decode("commit: no lane row".into())),
        }
    }

    /// Leaving by exception: `run_ended` with the outcome, and the release; nothing else.
    pub async fn run_ended(&self, req: &RunEndedRequest) -> Result<(), StorageError> {
        let lane = req.lane.as_str();
        self.conn
            .batch(vec![
                statement(
                    RUN_ENDED,
                    vec![
                        req.ts_ms.into(),
                        lane.into(),
                        req.run_id.as_str().into(),
                        SqlValue::json(&req.payload),
                        lane.into(),
                        req.lease_token.as_str().into(),
                    ],
                ),
                statement(RELEASE, vec![lane.into(), req.lease_token.as_str().into()]),
            ])
            .await?;
        Ok(())
    }

    /// The inbox receive (plan 3.3.4), one atomic `Batch`: the inbox row, the task row as
    /// `submitted`, and the lease attempt. A `messageId` already in the inbox inserts nothing and
    /// reports the task it belongs to.
    pub async fn receive(&self, req: &ReceiveRequest) -> Result<Receive, StorageError> {
        self.ensure_schema().await?;
        let lane = req.lane.as_str();
        let (task_id, message_id) = (req.task_id.as_str(), req.message_id.as_str());
        let out = self
            .conn
            .batch(vec![
                statement(INSERT_LANE, vec![lane.into()]),
                statement(
                    INSERT_INBOX,
                    vec![
                        lane.into(),
                        task_id.into(),
                        message_id.into(),
                        SqlValue::json(&req.message),
                    ],
                ),
                statement(
                    INSERT_TASK_SUBMITTED,
                    vec![
                        task_id.into(),
                        lane.into(),
                        lane.into(),
                        req.name.clone().into(),
                        message_id.into(),
                        SqlValue::json(&req.data),
                        SqlValue::opt_json(req.internal.as_ref()),
                        lane.into(),
                        message_id.into(),
                        task_id.into(),
                    ],
                ),
                statement(
                    ACQUIRE_FOR_MESSAGE,
                    vec![
                        req.holder.as_str().into(),
                        req.lease_token.as_str().into(),
                        self.lease_ttl_secs.into(),
                        lane.into(),
                        req.holder.as_str().into(),
                        message_id.into(),
                        task_id.into(),
                    ],
                ),
                statement(INBOX_TASK, vec![lane.into(), message_id.into()]),
            ])
            .await?;
        let r = &out.results;
        let StatementResult::Exec { rows_affected, last_insert_id } = r[1] else {
            return Err(StorageError::Protocol("receive: the inbox insert answered rows".into()));
        };
        if rows_affected == 0 {
            let task_id = r[4]
                .rows()
                .first()
                .and_then(|row| row.first())
                .and_then(SqlValue::as_str)
                .map(str::to_owned);
            return Ok(Receive::Duplicate(Duplicate { task_id }));
        }
        Ok(Receive::Received(Received { id: last_insert_id, acquired: first_int(&r[3]) }))
    }

    /// The early `submitted` row under interrupt, queue, concurrent and reject (plan 3.3.4), written
    /// before the lease is tried so a `tasks/get` before the commit finds it. Unfenced; the
    /// commit's upsert moves it to its terminal state. `true` when the row was inserted, `false`
    /// when the `task_id` or the `messageId` already had one.
    pub async fn insert_submitted_task(&self, task: &SubmittedTask) -> Result<bool, StorageError> {
        self.ensure_schema().await?;
        let lane = task.lane.as_str();
        let r = self
            .conn
            .exec(
                INSERT_SUBMITTED_TASK,
                vec![
                    task.task_id.as_str().into(),
                    lane.into(),
                    lane.into(),
                    task.name.clone().into(),
                    task.message_id.clone().into(),
                    SqlValue::json(&task.data),
                    SqlValue::opt_json(task.internal.as_ref()),
                ],
            )
            .await?;
        Ok(r.rows_affected > 0)
    }

    /// The drain check (plan 3.3.4), one atomic `Batch`: count the rows above `drained_id`, and
    /// release the lease when there are none, else keep it. The release is conditional in its own
    /// statement, so a message inserted after the count leaves the lease held and the runner loops.
    pub async fn drain_check(
        &self,
        lane: &str,
        lease_token: &str,
    ) -> Result<DrainCheck, StorageError> {
        let out = self
            .conn
            .batch(vec![
                statement(COUNT_PENDING, vec![lane.into(), lane.into()]),
                statement(RELEASE_IF_DRAINED, vec![lane.into(), lease_token.into()]),
            ])
            .await?;
        let pending = first_int(&out.results[0]).unwrap_or(0);
        Ok(DrainCheck { pending, released: out.results[1].rows_affected() > 0 })
    }

    /// Records a worker waiting under `interrupt`; `false` when the lease is not held, in which case
    /// the caller acquires instead.
    pub async fn set_waiter(&self, lane: &str, waiter: &str) -> Result<bool, StorageError> {
        self.ensure_schema().await?;
        Ok(self.conn.exec(SET_WAITER, vec![waiter.into(), lane.into()]).await?.rows_affected > 0)
    }

    /// The holder's boundary check.
    pub async fn waiter(&self, lane: &str) -> Result<Option<String>, StorageError> {
        let set = self.conn.query_all(WAITER, vec![lane.into()]).await?;
        Ok(set.rows.first().and_then(|r| r.first()).and_then(SqlValue::as_str).map(str::to_owned))
    }

    /// Takes a `submitted` or `working` row to `working`, one atomic `Batch`. The status it had
    /// (`working` means a dead holder had it), `None` when the row is terminal or missing.
    pub async fn pickup_task(
        &self,
        lane: &str,
        task_id: &str,
    ) -> Result<Option<String>, StorageError> {
        let params = || vec![lane.into(), task_id.into()];
        let out = self
            .conn
            .batch(vec![statement(PICKUP_STATUS, params()), statement(PICKUP_TASK, params())])
            .await?;
        if out.results[1].rows_affected() == 0 {
            return Ok(None);
        }
        let status = out.results[0].rows().first().and_then(|r| r.first());
        Ok(status.and_then(SqlValue::as_str).map(str::to_owned))
    }

    /// A `working` row whose holder died ends `failed` with the given status message (reason
    /// `worker-lost`, plan 5.12).
    pub async fn mark_orphan_failed(
        &self,
        lane: &str,
        task_id: &str,
        status_message: &Value,
    ) -> Result<bool, StorageError> {
        let r = self
            .conn
            .exec(
                MARK_ORPHAN_FAILED,
                vec![SqlValue::json(status_message), lane.into(), task_id.into()],
            )
            .await?;
        Ok(r.rows_affected > 0)
    }

    // ----------------------------------------------------------------------------------- reads

    /// The journal, every lane, after the event `after_id`.
    pub async fn events(&self, after_id: i64, limit: i64) -> Result<Vec<EventRow>, StorageError> {
        self.read(&select_events(), vec![after_id.into(), limit.into()], EventRow::from_row).await
    }

    pub async fn lane_events(
        &self,
        lane: &str,
        after_id: i64,
    ) -> Result<Vec<EventRow>, StorageError> {
        self.read(&select_lane_events(), vec![lane.into(), after_id.into()], EventRow::from_row)
            .await
    }

    pub async fn lanes(&self) -> Result<Vec<LaneRow>, StorageError> {
        self.read(&select_lanes(), vec![], LaneRow::from_row).await
    }

    pub async fn lane(&self, lane: &str) -> Result<Option<LaneRow>, StorageError> {
        Ok(self
            .read(&select_lane(), vec![lane.into()], LaneRow::from_row)
            .await?
            .into_iter()
            .next())
    }

    /// The whole history of a lane; how another lane reads it.
    pub async fn history(&self, lane: &str) -> Result<Vec<ChatItemRow>, StorageError> {
        self.read(&select_items(), vec![lane.into()], ChatItemRow::from_row).await
    }

    pub async fn tasks(
        &self,
        lane: &str,
        status: Option<&str>,
        limit: i64,
    ) -> Result<Vec<TaskRow>, StorageError> {
        match status {
            Some(s) => {
                self.read(
                    &select_tasks_by_status(),
                    vec![lane.into(), s.into(), limit.into()],
                    TaskRow::from_row,
                )
                .await
            }
            None => {
                self.read(&select_tasks(), vec![lane.into(), limit.into()], TaskRow::from_row).await
            }
        }
    }

    pub async fn orphans(&self, lane: &str) -> Result<Vec<TaskRow>, StorageError> {
        self.read(&select_orphans(), vec![lane.into()], TaskRow::from_row).await
    }

    /// Every inbox row of the lane, drained ones included, in `id` order.
    pub async fn inbox(&self, lane: &str) -> Result<Vec<InboxRow>, StorageError> {
        self.read(&select_inbox(), vec![lane.into()], InboxRow::from_row).await
    }

    /// The inbox rows above `drained_id`.
    pub async fn inbox_pending(&self, lane: &str) -> Result<Vec<InboxRow>, StorageError> {
        self.read(&select_inbox_pending(), vec![lane.into(), lane.into()], InboxRow::from_row).await
    }

    async fn read<T>(
        &self,
        sql: &str,
        params: Vec<SqlValue>,
        decode: fn(&crate::wire::Row) -> Result<T, StorageError>,
    ) -> Result<Vec<T>, StorageError> {
        if !self.refresh_schema().await? {
            return Ok(Vec::new()); // a database with no schema holds nothing
        }
        self.conn.query_all(sql, params).await?.rows.iter().map(decode).collect()
    }
}

/// A held lease with its renewal timer. [`Lease::lost`] resolves when a renew failed, which cancels
/// the run with `StorageUnavailable`; [`Lease::stop`] ends the timer when the run closes.
#[derive(Clone)]
pub struct Lease {
    inner: Arc<LeaseInner>,
}

struct LeaseInner {
    lane: String,
    holder: String,
    token: String,
    lost: AtomicBool,
    reason: Mutex<Option<String>>,
    notify: tokio::sync::Notify,
    task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl Drop for LeaseInner {
    fn drop(&mut self) {
        if let Some(t) = self.task.get_mut().ok().and_then(Option::take) {
            t.abort();
        }
    }
}

impl Lease {
    pub fn lane(&self) -> &str {
        &self.inner.lane
    }

    pub fn holder(&self) -> &str {
        &self.inner.holder
    }

    pub fn token(&self) -> &str {
        &self.inner.token
    }

    pub fn is_lost(&self) -> bool {
        self.inner.lost.load(Ordering::Acquire)
    }

    /// Resolves with the error that cancels the run once the lease is lost; pending forever while
    /// it is held, so race it against the run.
    pub async fn lost(&self) -> StorageError {
        loop {
            if self.is_lost() {
                let reason = self
                    .inner
                    .reason
                    .lock()
                    .expect("reason")
                    .clone()
                    .unwrap_or_else(|| "lease lost".into());
                return StorageError::Unavailable(reason);
            }
            self.inner.notify.notified().await;
        }
    }

    /// Ends the timer; the lease itself is released by the commit or `run_ended`.
    pub fn stop(&self) {
        if let Some(t) = self.inner.task.lock().expect("task").take() {
            t.abort();
        }
    }
}

/// The error mapping of plan 13.5: a lost transaction, a drop mid-request, a severed database and
/// a closed connection are `StorageUnavailable`, retryable; `not_found` at `Hello` is
/// `ThreadNotFound`; a refused token is `Unauthorized`; the rest keep agent-db's code.
impl From<WsdbError> for StorageError {
    fn from(e: WsdbError) -> Self {
        match e {
            WsdbError::Closed
            | WsdbError::Disconnected
            | WsdbError::TxnAborted
            | WsdbError::Timeout
            | WsdbError::Io(_) => StorageError::Unavailable(e.to_string()),
            WsdbError::Server { code, message } => match code.as_str() {
                "unavailable" | "resource_exhausted" => StorageError::Unavailable(message),
                "not_found" => StorageError::ThreadNotFound(message),
                "unauthenticated" | "permission_denied" => StorageError::Unauthorized(message),
                _ => StorageError::Server { code, message },
            },
            WsdbError::Protocol(m) => StorageError::Protocol(m),
        }
    }
}

/// The first cell of the first row, as an integer: `RETURNING version`, `COUNT(*)`.
fn first_int(r: &StatementResult) -> Option<i64> {
    r.rows().first().and_then(|row| row.first()).and_then(SqlValue::as_i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_acquire_is_the_plan_statement() {
        assert_eq!(
            ACQUIRE,
            "UPDATE lanes SET lease_holder = ?, lease_token = ?, lease_until = unixepoch() + ? \
             WHERE lane = ? AND (lease_until < unixepoch() OR lease_holder = ?) \
             AND NOT EXISTS (SELECT 1 FROM tasks older WHERE older.lane = lanes.lane AND older.status = 'submitted' \
             AND older.message_id IS NOT NULL AND older.updated_at > unixepoch() - ? \
             AND NOT EXISTS (SELECT 1 FROM inbox WHERE inbox.lane = older.lane AND inbox.task_id = older.task_id) \
             AND older.seq < (SELECT seq FROM tasks WHERE lane = lanes.lane AND task_id = ?)) RETURNING version"
        );
        assert!(RUN_STARTED.contains("SELECT ?, ?, ?, 'run_started', jsonb(?) FROM lanes WHERE lane = ? AND lease_token = ?"));
        assert!(RUN_STARTED.ends_with("AND NOT EXISTS (SELECT 1 FROM events WHERE lane = lanes.lane AND run_id = ? AND type = 'run_ended')"));
        assert!(RELEASE_IF_DRAINED.contains("NOT EXISTS (SELECT 1 FROM inbox WHERE inbox.lane = lanes.lane AND inbox.id > lanes.drained_id)"));
        assert_eq!(
            SET_WAITER,
            "UPDATE lanes SET waiter = ? WHERE lane = ? AND lease_until > unixepoch()"
        );
    }

    #[test]
    fn every_lease_extension_binds_the_ttl() {
        for (sql, params) in [
            (ACQUIRE.to_owned(), 7),
            (RENEW.to_owned(), 4),
            (ACQUIRE_FOR_MESSAGE.to_owned(), 7),
            (update_lane(LeaseAfter::Renew), 7),
        ] {
            assert!(sql.contains("lease_until = unixepoch() + ?"));
            assert_eq!(sql.matches('?').count(), params);
        }
        assert_eq!(update_lane(LeaseAfter::Release).matches('?').count(), 6);
    }

    #[test]
    fn commit_writes_are_fenced_on_version_and_token() {
        let fence = "FROM lanes WHERE lane = ? AND version = ? AND lease_token = ?";
        assert!(insert_item(true).contains(&format!("jsonb(?) {fence} ON CONFLICT(id)")));
        assert!(insert_event(true).ends_with(fence));
        assert!(upsert_task(true).contains(&format!("unixepoch() {fence} ON CONFLICT(task_id)")));
        assert!(delete_item(true).ends_with(&format!("EXISTS (SELECT 1 {fence})")));
        assert!(insert_item(false).contains("jsonb(?) FROM lanes WHERE lane = ? ON CONFLICT"));
        assert!(update_lane(LeaseAfter::Release).contains(
            "lease_until = 0, lease_token = NULL WHERE lane = ? AND version = ? AND lease_token = ? RETURNING version"
        ));
        assert!(update_lane(LeaseAfter::Renew).contains("lease_until = unixepoch() + ? WHERE"));
        assert!(
            update_lane(LeaseAfter::Release).starts_with("UPDATE lanes SET version = version + 1,")
        );
    }

    #[test]
    fn seq_is_assigned_per_lane_inside_the_statement() {
        assert!(
            insert_item(true)
                .contains("COALESCE((SELECT MAX(seq) FROM items WHERE lane = ?), 0) + 1")
        );
        assert!(
            upsert_task(true)
                .contains("COALESCE((SELECT MAX(seq) FROM tasks WHERE lane = ?), 0) + 1")
        );
        assert!(INSERT_TASK_SUBMITTED.contains("'submitted'"));
        assert!(
            INSERT_TASK_SUBMITTED
                .ends_with("FROM inbox WHERE lane = ? AND message_id = ? AND task_id = ?")
        );
        assert!(ACQUIRE_FOR_MESSAGE.starts_with(ACQUIRE.split(" AND NOT EXISTS").next().unwrap()));
    }

    /// The early row and the commit agree on the columns, so the commit's upsert lands on the row
    /// the early insert wrote and moves it to its terminal state without touching its `seq`.
    #[test]
    fn the_early_submitted_row_is_an_unfenced_insert_or_ignore_the_commit_upserts_over() {
        assert!(INSERT_SUBMITTED_TASK.starts_with("INSERT OR IGNORE INTO tasks (task_id, lane, seq, name, message_id, status, status_message, data, internal, updated_at)"));
        assert!(INSERT_SUBMITTED_TASK.contains(
            "SELECT ?, ?, COALESCE((SELECT MAX(seq) FROM tasks WHERE lane = ?), 0) + 1, ?, ?, 'submitted', NULL, jsonb(?), jsonb(?), unixepoch()"
        ));
        assert!(!INSERT_SUBMITTED_TASK.contains("FROM lanes"), "the early row is unfenced");
        assert!(!INSERT_SUBMITTED_TASK.contains("FROM inbox"), "no inbox row under these policies");
        assert_eq!(INSERT_SUBMITTED_TASK.matches('?').count(), 7);
        assert_eq!(INSERT_TASK_SUBMITTED.matches('?').count(), 10);
        for fenced in [true, false] {
            let upsert = upsert_task(fenced);
            assert!(
                upsert.contains("ON CONFLICT(task_id) DO UPDATE SET status = excluded.status,")
            );
            assert!(!upsert.contains("seq = excluded.seq"), "the upsert keeps the early row's seq");
            assert!(upsert.contains("internal = excluded.internal, updated_at = excluded.updated_at"));
        }
        let t: SubmittedTask = serde_json::from_value(serde_json::json!({
            "lane": "front-desk", "task_id": "AST_1", "message_id": "m1",
        }))
        .unwrap();
        assert_eq!(t.data, TaskData::default());
        assert_eq!(serde_json::to_string(&t.data).unwrap(), "{}");
        assert_eq!(t.name, None);
    }

    #[test]
    fn reads_select_json_columns_through_json() {
        assert_eq!(
            select_items(),
            "SELECT id, lane, seq, message_id, json(payload) FROM items WHERE lane = ? ORDER BY seq"
        );
        assert!(select_lane().starts_with("SELECT lane, version, json(state), json(agent_stack),"));
        assert!(
            select_inbox_pending().contains("id > (SELECT drained_id FROM lanes WHERE lane = ?)")
        );
    }

    #[test]
    fn requests_deserialize_from_the_host_dicts() {
        let req: CommitRequest = serde_json::from_value(serde_json::json!({
            "lane": "front-desk", "run_id": "r1",
            "fence": {"version": 3, "lease_token": "t"},
            "items": [{"id": "i1", "payload": {"type": "message", "role": "user"}}],
            "run_ended": {"ts_ms": 1, "type": "run_ended", "payload": {}},
            "after": "renew"
        }))
        .unwrap();
        assert_eq!(req.after, LeaseAfter::Renew);
        assert_eq!(req.items[0].message_id, None);
        assert!(req.tasks.is_empty());
        let r: Receive = serde_json::from_value(
            serde_json::json!({"outcome": "received", "id": 4, "acquired": null}),
        )
        .unwrap();
        assert_eq!(r, Receive::Received(Received { id: 4, acquired: None }));
    }

    #[test]
    fn wsdb_errors_map_as_the_plan_says() {
        assert!(matches!(StorageError::from(WsdbError::TxnAborted), StorageError::Unavailable(_)));
        assert!(matches!(
            StorageError::from(WsdbError::Disconnected),
            StorageError::Unavailable(_)
        ));
        let nf =
            WsdbError::Server { code: "not_found".into(), message: "database not found".into() };
        assert!(matches!(StorageError::from(nf), StorageError::ThreadNotFound(_)));
        let other = WsdbError::Server { code: "exec".into(), message: "constraint".into() };
        assert!(matches!(StorageError::from(other), StorageError::Server { .. }));
    }
}
