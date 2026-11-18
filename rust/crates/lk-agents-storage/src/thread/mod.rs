pub mod db;
pub mod lease;
mod partition;
pub mod run;
mod session_checkpoint;
pub use session_checkpoint::SessionCheckpointData;
pub mod schema;

#[cfg(test)]
#[path = "schema_routing_tests.rs"]
mod schema_routing_tests;

pub use partition::Partition;
pub use run::{CommitOptions, EventLog, Run, RunOptions, Tasks};

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

/// Sends again a request refused before it applied: what a write's [`Resend`] and a read share.
/// Every attempt runs under the caller's deadline, which its requests' waits inherit; another
/// starts only while it can before that deadline.
pub(crate) struct Retry {
    clock: Arc<dyn Clock>,
    epoch: Ts,
    deadline: Ts,
    backoff: Backoff,
}

impl Retry {
    /// Until the caller's deadline, else [`crate::deadline::REQUEST_WAIT`] from now; its backoff
    /// waits on `clock`.
    pub(crate) fn new(clock: Arc<dyn Clock>) -> Self {
        let deadline = crate::deadline::or_default(&*clock);
        Self::until(clock, deadline)
    }

    /// Until `deadline`, never past the caller's.
    pub(crate) fn until(clock: Arc<dyn Clock>, deadline: Ts) -> Self {
        let deadline = crate::deadline::current().map_or(deadline, |outer| outer.min(deadline));
        let epoch = clock.now();
        let backoff = Backoff::new(Duration::ZERO, deadline.saturating_duration_since(epoch));
        Self { clock, epoch, deadline, backoff }
    }

    /// `attempt` under the deadline: what its requests' waits inherit.
    pub(crate) async fn attempt<F: Future>(&self, attempt: F) -> F::Output {
        crate::deadline::within(self.deadline, attempt).await
    }

    /// Waits out the next backoff step; `false`, at once, when no attempt could start after it
    /// before the deadline.
    pub(crate) async fn wait(&mut self, why: &str) -> bool {
        let elapsed = self.clock.now().saturating_duration_since(self.epoch);
        let Some(at) = self.backoff.next(elapsed).map(|after| self.epoch.saturating_add(after))
        else {
            return false;
        };
        if at >= self.deadline {
            return false;
        }
        log::debug!(target: lk_agents_abi::LOG_TARGET, "storage request refused unapplied, sending it again: {why}");
        self.clock.wait_until(at).await;
        true
    }

    /// `request` again while it is refused unapplied (`Unavailable`), until the deadline: a read,
    /// or a request with no effect to apply twice.
    pub(crate) async fn send<T, R>(
        mut self,
        mut request: impl FnMut() -> R,
    ) -> Result<T, StorageError>
    where
        R: Future<Output = Result<T, StorageError>>,
    {
        loop {
            match self.attempt(request()).await {
                Err(StorageError::Unavailable(why)) if self.wait(&why).await => {}
                out => return out,
            }
        }
    }
}

/// One write's resends: the one retry owner every write shares, as transports never resend. A
/// write refused before it applied (`Unavailable`) is sent again while another attempt can start
/// before the caller's deadline, then reaches the caller as it is. A write whose outcome is
/// unknown (`Uncertain`) is never sent again: its probe reads back once whether it applied, else
/// it reaches the caller `Uncertain`.
pub(crate) struct Resend {
    retry: Retry,
    /// Where a write left unresolved is counted, under its kind.
    cx: Cx,
    write: StorageWrite,
}

impl Resend {
    /// Until the caller's deadline, else [`crate::deadline::REQUEST_WAIT`] from now; its backoff
    /// waits on `clock`.
    pub(crate) fn new(clock: Arc<dyn Clock>, cx: &Cx, write: StorageWrite) -> Self {
        Self { retry: Retry::new(clock), cx: cx.clone(), write }
    }

    /// Until `deadline`, never past the caller's.
    pub(crate) fn until(clock: Arc<dyn Clock>, deadline: Ts, cx: &Cx, write: StorageWrite) -> Self {
        Self { retry: Retry::until(clock, deadline), cx: cx.clone(), write }
    }

    /// `attempt` under this write's deadline: what its requests' waits inherit.
    pub(crate) async fn attempt<F: Future>(&self, attempt: F) -> F::Output {
        self.retry.attempt(attempt).await
    }

    /// `Break` with the outcome once it is known, else waits out the next backoff step after a
    /// refusal that applied nothing. An unknown outcome is final, and counted.
    pub(crate) async fn after<T>(
        &mut self,
        out: Result<T, StorageError>,
    ) -> ControlFlow<Result<T, StorageError>> {
        match out {
            Err(StorageError::Unavailable(why)) if self.retry.wait(&why).await => {
                ControlFlow::Continue(())
            }
            Err(StorageError::Uncertain(lost)) => ControlFlow::Break(Err(self.uncertain(lost))),
            out => ControlFlow::Break(out),
        }
    }

    fn uncertain(&self, lost: Unacknowledged) -> StorageError {
        telemetry::write_uncertain(&self.cx, &*self.retry.clock, self.write);
        StorageError::Uncertain(lost)
    }

    /// `write` until its outcome is known or unknowable.
    pub(crate) async fn send<T, W>(self, write: impl FnMut() -> W) -> Result<T, StorageError>
    where
        W: Future<Output = Result<T, StorageError>>,
    {
        self.probed(write, || std::future::ready(Ok(None))).await
    }

    /// [`Resend::send`], and after a lost acknowledgment `probe` reads back once whether the
    /// write applied: `Some` with its outcome; `None`, or a probe refused unapplied, leaves it
    /// `Uncertain`, as the write may still apply; any other error is the probe's answer.
    pub(crate) async fn probed<T, W, P>(
        mut self,
        mut write: impl FnMut() -> W,
        probe: impl FnOnce() -> P,
    ) -> Result<T, StorageError>
    where
        W: Future<Output = Result<T, StorageError>>,
        P: Future<Output = Result<Option<T>, StorageError>>,
    {
        let lost = loop {
            match self.attempt(write()).await {
                Err(StorageError::Uncertain(lost)) => break lost,
                out => {
                    if let ControlFlow::Break(out) = self.after(out).await {
                        return out;
                    }
                }
            }
        };
        log::debug!(target: lk_agents_abi::LOG_TARGET, "write acknowledgment lost, reading back whether it applied: {lost}");
        match self.attempt(probe()).await {
            Ok(Some(applied)) => Ok(applied),
            Ok(None) | Err(StorageError::Unavailable(_) | StorageError::Uncertain(_)) => {
                Err(self.uncertain(lost))
            }
            Err(error) => Err(error),
        }
    }
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

    /// The resends of one `write`, until the caller's deadline.
    pub(crate) fn resend(&self, write: StorageWrite) -> Resend {
        Resend::new(self.background.clock.clone(), &self.cx, write)
    }

    /// The resends of one `write`, until `deadline`.
    pub(crate) fn resend_until(&self, deadline: Ts, write: StorageWrite) -> Resend {
        Resend::until(self.background.clock.clone(), deadline, &self.cx, write)
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
        let result = self.conn.read_batch(statements).await?;
        crate::context::scopes_from_rows(result.at(scopes)?.rows())
    }
}

#[cfg(test)]
#[path = "thread_tests.rs"]
pub(crate) mod tests;
