pub mod db;
pub mod lease;
mod partition;
pub mod run;
pub mod schema;

#[cfg(test)]
#[path = "schema_routing_tests.rs"]
mod schema_routing_tests;

pub use partition::Partition;
pub use run::{CommitOptions, EventLog, RESOLVE_LATER, RESOLVE_WAIT, Run, RunOptions, Tasks};

use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::{Clock, Ts};
use lk_agents_net::owned::Background;
use lk_agents_net::task::{Leave, TaskSet};
use parking_lot::Mutex;

use crate::StorageError;
use crate::sql::{Bind, SqlValue, Statement, Table};
use crate::telemetry::{self, Cx, StorageWrite};
use crate::thread::db::{ItemPage, TaskPage, TaskQuery};
use crate::thread::schema::*;
use lk_agents_abi::storage::Unacknowledged;
use lk_agents_abi::storage::context::ContextScope;
use lk_agents_abi::storage::thread::Backoff;

use crate::ThreadDbConnection;

/// How long a lease lasts and how often its holder renews it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaseTiming {
    pub ttl: Duration,
    pub renew_interval: Duration,
}

impl LeaseTiming {
    pub fn new(ttl: Duration, renew_interval: Duration) -> Result<Self, StorageError> {
        if ttl.is_zero() || ttl.as_nanos().div_ceil(1_000_000) > i64::MAX as u128 {
            return Err(StorageError::Invalid(
                "lease_ttl must be positive and fit in signed milliseconds".into(),
            ));
        }
        if renew_interval.is_zero() || renew_interval >= ttl.saturating_sub(renew_interval) {
            return Err(StorageError::Invalid(
                "renew_interval must be positive and less than half of lease_ttl".into(),
            ));
        }
        Ok(Self { ttl, renew_interval })
    }

    /// The TTL in database milliseconds, rounded up.
    pub(crate) fn millis(&self) -> i64 {
        self.ttl.as_nanos().div_ceil(1_000_000) as i64
    }
}

impl Default for LeaseTiming {
    /// A 10 s lease, measured on the database's clock, renewed every 3 s.
    fn default() -> Self {
        Self { ttl: Duration::from_secs(10), renew_interval: Duration::from_secs(3) }
    }
}

/// One write's resends after a lost acknowledgment: the resolver every write shares. The write
/// is idempotent by its key (a lease token, a run, task or message id): sent again it cannot
/// apply twice. `Uncertain` reaches the caller only once `wait` ran out with the outcome still
/// unknown.
pub(crate) struct Resend {
    clock: Arc<dyn Clock>,
    epoch: Ts,
    backoff: Backoff,
    lost: Option<Unacknowledged>,
    /// Where a write left unresolved is counted, under its kind.
    cx: Cx,
    write: StorageWrite,
}

impl Resend {
    /// Its backoff waits on `clock`.
    pub(crate) fn new(clock: Arc<dyn Clock>, wait: Duration, cx: &Cx, write: StorageWrite) -> Self {
        let epoch = clock.now();
        let backoff = Backoff::new(Duration::ZERO, wait);
        Self { clock, epoch, backoff, lost: None, cx: cx.clone(), write }
    }

    /// Resolving a write whose acknowledgment `lost` already reported lost.
    pub(crate) fn after_lost(
        clock: Arc<dyn Clock>,
        wait: Duration,
        cx: &Cx,
        write: StorageWrite,
        lost: Unacknowledged,
    ) -> Self {
        Self { lost: Some(lost), ..Self::new(clock, wait, cx, write) }
    }

    fn elapsed(&self) -> Duration {
        self.clock.now().saturating_duration_since(self.epoch)
    }

    /// Whether an earlier send may have applied.
    pub(crate) fn lost(&self) -> bool {
        self.lost.is_some()
    }

    /// `Break` with the outcome once it is known, else waits out the next backoff step. After a
    /// lost acknowledgment a refused connection is still no answer.
    pub(crate) async fn after<T>(
        &mut self,
        out: Result<T, StorageError>,
    ) -> ControlFlow<Result<T, StorageError>> {
        let lost = match (out, self.lost.take()) {
            (Err(StorageError::Uncertain(why)), _)
            | (Err(StorageError::Unavailable(_)), Some(why)) => why,
            (out, lost) => {
                self.lost = lost;
                return ControlFlow::Break(out);
            }
        };
        let Some(at) = self.backoff.next(self.elapsed()) else {
            telemetry::write_uncertain(&self.cx, &*self.clock, self.write);
            return ControlFlow::Break(Err(StorageError::Uncertain(lost)));
        };
        log::debug!(target: lk_agents_abi::LOG_TARGET, "write acknowledgment lost, sending it again: {lost}");
        self.lost = Some(lost);
        self.clock.wait_until(self.epoch.saturating_add(at)).await;
        ControlFlow::Continue(())
    }

    /// `write` until its outcome is known: one that already applied answers as applied.
    pub(crate) async fn send<T, W>(self, write: impl FnMut() -> W) -> Result<T, StorageError>
    where
        W: Future<Output = Result<T, StorageError>>,
    {
        self.probed(write, || std::future::ready(Ok(None))).await
    }

    /// [`Resend::send`] for a write whose second apply answers differently: after each lost
    /// acknowledgment `probe` asks once whether it applied, `Some` with its outcome, `None` to
    /// send it again.
    pub(crate) async fn probed<T, W, P>(
        mut self,
        mut write: impl FnMut() -> W,
        mut probe: impl FnMut() -> P,
    ) -> Result<T, StorageError>
    where
        W: Future<Output = Result<T, StorageError>>,
        P: Future<Output = Result<Option<T>, StorageError>>,
    {
        loop {
            let out = match self.lost() {
                true => match probe().await {
                    Ok(Some(applied)) => Ok(applied),
                    Ok(None) => write().await,
                    Err(error) => Err(error),
                },
                false => write().await,
            };
            if let ControlFlow::Break(out) = self.after(out).await {
                return out;
            }
        }
    }
}

/// A read sent as a batch changed nothing, so its lost answer is only retryable.
pub(crate) async fn read_batch(
    conn: &dyn ThreadDbConnection,
    statements: Vec<Statement>,
) -> Result<crate::sql::BatchResult, StorageError> {
    conn.batch(statements).await.map_err(|error| match error {
        StorageError::Uncertain(why) => StorageError::Unavailable(why.to_string()),
        error => error,
    })
}

/// What the engine knows of a database's schema: `None` until a probe or a write found it.
pub type SchemaCell = Arc<Mutex<Option<i64>>>;

/// The thread a worker opened: its database over one connection, shared by every partition. Cheap to
/// clone.
#[derive(Clone)]
pub struct Thread {
    conn: Arc<dyn ThreadDbConnection>,
    thread_id: Arc<str>,
    pub(crate) timing: LeaseTiming,
    schema: SchemaCell,
    /// Where its health metrics go.
    pub(crate) cx: Cx,
    /// The background writes its callers left (a dropped lease's release, a dropped turn's
    /// cleanup): the connection stays open past [`Thread::close`] until they end.
    left: Arc<TaskSet>,
    /// The storage's background: where those writes and a deferred close run, and the clock
    /// every wait on this thread reads.
    pub(crate) background: Background,
    /// The lease holder id of the storage that opened it, labelling its turns' leases.
    holder: Option<Arc<str>>,
}

impl Thread {
    /// Over an open connection; `schema` is what is known of its schema, probed when unknown.
    /// Its background work runs on, and its waits read, the storage's `background`.
    pub async fn new(
        conn: Arc<dyn ThreadDbConnection>,
        thread_id: &str,
        timing: LeaseTiming,
        schema: SchemaCell,
        background: Background,
    ) -> Result<Self, StorageError> {
        let (left, thread_id, cx, holder) = (Arc::default(), thread_id.into(), Cx::NONE, None);
        let thread = Self { conn, thread_id, timing, schema, cx, left, background, holder };
        if thread.schema_missing() {
            let found = db::schema_version(&*thread.conn).await?;
            *thread.schema.lock() = found;
        }
        Ok(thread)
    }

    pub fn id(&self) -> &str {
        &self.thread_id
    }

    /// Labels its turns' leases with the storage's lease holder id.
    pub fn with_holder(mut self, holder: &str) -> Self {
        self.holder = Some(holder.into());
        self
    }

    pub fn holder(&self) -> Option<&str> {
        self.holder.as_deref()
    }

    /// Records its health metrics under `cx`.
    pub fn with_telemetry(mut self, cx: Cx) -> Self {
        self.cx = cx;
        self
    }

    /// The resends of one `write`, for up to [`RESOLVE_WAIT`].
    pub(crate) fn resend(&self, write: StorageWrite) -> Resend {
        self.resend_within(RESOLVE_WAIT, write)
    }

    /// The resends of one `write`, for up to `wait`.
    pub(crate) fn resend_within(&self, wait: Duration, write: StorageWrite) -> Resend {
        Resend::new(self.background.clock.clone(), wait, &self.cx, write)
    }

    pub(crate) fn clock(&self) -> &dyn Clock {
        &*self.background.clock
    }

    /// Done with the database: the backend releases or parks the connection, once the
    /// background writes its callers left ended. Deferred, the close is the storage's task, which
    /// its close joins; refused or cancelled there, the dropped connection closes itself.
    pub async fn close(&self) {
        if self.left.is_empty() {
            return self.conn.close().await;
        }
        let (conn, left, clock) =
            (self.conn.clone(), self.left.clone(), self.background.clock.clone());
        self.background.spawn("thread close", async move {
            left.wait_empty(&*clock, None).await;
            conn.close().await;
        });
    }

    /// Keeps the connection open past [`Thread::close`] until the guard drops: what a background
    /// write on this thread holds.
    pub fn keep_open(&self, name: &str) -> Leave {
        self.left.track(name)
    }

    pub fn partition(&self, name: &str) -> Partition {
        Partition { thread: self.clone(), name: name.to_owned() }
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
        // a lost acknowledgment is sent again; a resend refused by its own schema reconciles
        match self.resend(StorageWrite::Schema).send(|| self.conn.batch(db::bootstrap())).await {
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
        decode: fn(&[SqlValue]) -> Result<T, StorageError>,
    ) -> Result<Vec<T>, StorageError> {
        if !self.refresh_schema().await? {
            return Ok(Vec::new());
        }
        self.conn.query(statement.bind()).await?.iter().map(|row| decode(row)).collect()
    }

    /// One write statement; how many rows it changed.
    pub(crate) async fn exec<S: Bind>(&self, statement: S) -> Result<i64, StorageError> {
        let out = self.conn.batch(vec![statement.bind()]).await?;
        Ok(out.at(0)?.rows_affected())
    }

    /// One idempotent write statement, sent again after a lost acknowledgment until its
    /// outcome is known: the rows it matched.
    pub(crate) async fn exec_idempotent<S: Bind>(
        &self,
        write: StorageWrite,
        statement: impl Fn() -> S,
    ) -> Result<i64, StorageError> {
        self.resend(write).send(|| self.exec(statement())).await
    }

    /// The journal, every partition, after the event `after_id`, at most `limit` events.
    pub async fn journal(&self, after_id: i64, limit: i64) -> Result<Vec<EventRow>, StorageError> {
        self.read(SelectEvents { after_id, limit }, EventRow::from_row).await
    }

    /// One task, whichever partition it is on.
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

    pub async fn partitions(&self) -> Result<Vec<PartitionRow>, StorageError> {
        self.read(SelectPartitions {}, PartitionRow::from_row).await
    }

    /// One page of the thread's committed items in insertion order, of `partition_key` when given,
    /// after the opaque `cursor` of the previous page. No lease needed.
    pub async fn list_items(
        &self,
        partition_key: Option<&str>,
        cursor: Option<&str>,
        limit: i64,
    ) -> Result<ItemPage, StorageError> {
        if !self.refresh_schema().await? {
            return Ok(ItemPage { items: Vec::new(), next: None });
        }
        db::list_items(&*self.conn, partition_key, cursor, limit).await
    }

    /// A direct field store's committed scope snapshot. Initialization remains runtime-owned.
    pub async fn context_scopes(
        &self,
        partition_key: &str,
    ) -> Result<Vec<ContextScope>, StorageError> {
        self.ensure_schema().await?;
        let mut statements: Vec<Statement> = db::framework_schema().collect();
        statements.push(SelectScopes { partition_key }.bind());
        let scopes = statements.len() - 1;
        let result = read_batch(&*self.conn, statements).await?;
        crate::context::scopes_from_rows(result.at(scopes)?.rows())
    }
}

#[cfg(test)]
#[path = "thread_tests.rs"]
pub(crate) mod tests;
