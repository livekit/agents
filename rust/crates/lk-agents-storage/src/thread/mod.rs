//! The shared engine over one thread database, whichever backend: [`Thread`] is the handle a
//! worker opens, executing the core's plans on a [`ThreadDbConnection`], [`Lane`] one agent's part
//! of it, and [`Lane::open`] takes it for one [`Run`], which ends with
//! [`Run::commit`] (one fenced write) or [`Run::abandon`] (`run_ended` and the release).
//! [`LaneAccess::Concurrent`] is the `concurrent` policy: no lease, append-only.

pub mod db;
mod lane;
pub mod lease;
pub mod run;
pub mod schema;

pub use lane::Lane;
pub use run::{CommitOptions, EventLog, RESOLVE_LATER, RESOLVE_WAIT, Run, RunOptions, Tasks};

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use crate::StorageError;
use crate::sql::{Bind, Row, Statement, Table};
use crate::thread::db::{ItemPage, TaskPage, TaskQuery};
use crate::thread::schema::*;
use lk_agents_abi::storage::context::ContextScope;

use crate::{ThreadDbConnection, shortuuid};

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
        if renew_interval.is_zero() || renew_interval >= ttl.saturating_sub(renew_interval) {
            return Err(StorageError::Invalid(
                "renew_interval must be positive and less than half of lease_ttl".into(),
            ));
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
                Statement::ddl(crate::thread::schema::CONTEXT_SCHEMA),
                crate::thread::schema::SelectScopes { lane }.bind(),
            ])
            .await?;
        crate::context::scopes_from_rows(result.at(1)?.rows())
    }
}
