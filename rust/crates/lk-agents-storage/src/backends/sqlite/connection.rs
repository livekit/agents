use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use rusqlite::types::ValueRef;
use tokio::sync::{Mutex, mpsc, oneshot};

use crate::sql::{BatchResult, Row, SqlValue, Statement, StatementResult};
use crate::{StorageError, ThreadDbConnection};
use lk_agents_abi::storage::Unacknowledged;
use lk_agents_abi::{Clock, OwnedTask, Spawner, SpawnerExt as _};
use lk_agents_net::owned::Background;

// Each database is served by its own worker, so no query blocks the executor.

/// How long a statement waits for another process's write lock before failing retryably.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);
/// Requests queued to one worker before senders wait.
const QUEUE: usize = 32;

pub(super) enum Request {
    Query(Statement, oneshot::Sender<Result<Vec<Row>, StorageError>>),
    Batch(Vec<Statement>, oneshot::Sender<Result<BatchResult, StorageError>>),
    DeleteThread(PathBuf, String, oneshot::Sender<Result<(), StorageError>>),
    /// Resolved, or ended with the worker, once the database is closed.
    Close(lk_agents_net::signal::Producer<Option<()>>),
}

/// Where a local storage's work runs: each database's worker on `spawner` (a dedicated thread
/// per worker in production, the scheduler in a simulation), everything else in `background`,
/// whose clock `unixepoch()` and every wait read.
#[derive(Clone)]
pub struct Workers {
    pub spawner: Arc<dyn Spawner>,
    pub background: Background,
}

impl Workers {
    /// Production: each worker on a thread of its own.
    pub fn threads(background: Background) -> Self {
        Self { spawner: Arc::new(lk_agents_net::task::DedicatedThreads), background }
    }

    pub fn clock(&self) -> &Arc<dyn Clock> {
        &self.background.clock
    }
}

type Open = Arc<dyn Fn() -> Result<SqliteThread, StorageError> + Send + Sync>;

/// Opens the database, then runs its requests in order, each inline. Closes the connection once,
/// at a close request, when every sender is gone or when its owner drops it. Dropped before it
/// answers, the worker dies as a crashed one does: what it still held has an unknown outcome.
async fn serve(
    open: Open,
    ready: oneshot::Sender<Result<(), StorageError>>,
    mut rx: mpsc::Receiver<Request>,
) {
    let mut sqlite = match open() {
        Ok(sqlite) => {
            let _ = ready.send(Ok(()));
            sqlite
        }
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let done = loop {
        let Some(request) = rx.recv().await else { break None };
        let SqliteThread { conn, access } = &mut sqlite;
        let (conn, access) = (conn.as_mut().expect("open until the worker ends"), access.as_ref());
        match request {
            Request::Query(s, reply) => {
                let _ = reply.send(super::lifecycle::with_access(access, || query(conn, &s)));
            }
            Request::Batch(statements, reply) => {
                let run = || batch(conn, &statements);
                let _ = reply.send(super::lifecycle::with_access(access, run));
            }
            Request::DeleteThread(directory, id, reply) => {
                let _ = reply.send(super::lifecycle::delete(conn, &directory, &id));
            }
            Request::Close(done) => break Some(done),
        }
    };
    drop(sqlite);
    if let Some(done) = done {
        done.resolve(());
    }
}

/// A database's worker and its queue: dropping it cancels the worker, which closes the database.
struct Worker {
    tx: mpsc::Sender<Request>,
    _task: OwnedTask,
}

/// One SQLite database, served by its worker in order, which it owns. A worker found gone is
/// replaced by the next request.
pub struct FileConnection {
    worker: Mutex<Worker>,
    workers: Workers,
    open: Open,
    closed: AtomicBool,
}

impl FileConnection {
    /// Opens (creating when `create`) the file and starts its worker.
    pub async fn open(path: PathBuf, create: bool, workers: Workers) -> Result<Self, StorageError> {
        Self::open_with_access(path, create, None, workers).await
    }

    pub(super) async fn open_with_access(
        path: PathBuf,
        create: bool,
        thread: Option<(PathBuf, String)>,
        workers: Workers,
    ) -> Result<Self, StorageError> {
        let clock = workers.clock().clone();
        let open = move || SqliteThread::open(&path, create, thread.clone(), &clock);
        Self::start(workers, Arc::new(open)).await
    }

    /// Over the connection `open` makes, run first on the worker.
    pub(super) async fn start(workers: Workers, open: Open) -> Result<Self, StorageError> {
        let worker = spawn(&workers, &open).await?;
        Ok(Self { worker: Mutex::new(worker), workers, open, closed: AtomicBool::new(false) })
    }

    /// A request that changes nothing, or whose loss before an answer is harmless.
    pub(super) async fn send<T>(
        &self,
        request: impl FnOnce(oneshot::Sender<Result<T, StorageError>>) -> Request,
    ) -> Result<T, StorageError> {
        self.enqueue(request).await?.await.map_err(|_| closed())?
    }

    /// Queues `request` for the worker; refused, retryably, before the worker has it.
    async fn enqueue<T>(
        &self,
        request: impl FnOnce(oneshot::Sender<Result<T, StorageError>>) -> Request,
    ) -> Result<oneshot::Receiver<Result<T, StorageError>>, StorageError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(closed());
        }
        let mut worker = self.worker.lock().await;
        if worker.tx.is_closed() {
            *worker = spawn(&self.workers, &self.open).await?;
        }
        let (reply, answer) = oneshot::channel();
        worker.tx.send(request(reply)).await.map_err(|_| closed())?;
        Ok(answer)
    }
}

/// Starts a worker over `open` and waits for it to open the database.
async fn spawn(workers: &Workers, open: &Open) -> Result<Worker, StorageError> {
    let (tx, rx) = mpsc::channel(QUEUE);
    let (ready, opened) = oneshot::channel();
    let task = workers.spawner.spawn(serve(open.clone(), ready, rx));
    opened.await.map_err(|_| closed())??;
    Ok(Worker { tx, _task: task })
}

/// The connection a worker owns, and the thread's lifecycle access when it serves one. Dropped,
/// however its worker ended, it closes the connection under the lifecycle lock.
pub(super) struct SqliteThread {
    /// `None` only while dropped.
    conn: Option<rusqlite::Connection>,
    access: Option<super::lifecycle::Access>,
}

impl Drop for SqliteThread {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            super::lifecycle::close(conn, self.access.as_ref());
        }
    }
}

impl SqliteThread {
    pub(super) fn open(
        path: &Path,
        create: bool,
        thread: Option<(PathBuf, String)>,
        clock: &Arc<dyn Clock>,
    ) -> Result<Self, StorageError> {
        let access = thread
            .map(|(directory, id)| super::lifecycle::Access::open(&directory, id, clock))
            .transpose()?;
        let conn = super::lifecycle::with_access(access.as_ref(), || connect(path, create, clock))?;
        Ok(Self { conn: Some(conn), access })
    }
}

fn closed() -> StorageError {
    StorageError::Unavailable("the database is closed".into())
}

#[async_trait]
impl ThreadDbConnection for FileConnection {
    async fn query(&self, statement: Statement) -> Result<Vec<Row>, StorageError> {
        self.send(|reply| Request::Query(statement, reply)).await
    }

    /// A worker that dies with the batch queued (a panic mid-batch, after its commit or not)
    /// leaves its outcome unknown.
    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError> {
        let answer = self.enqueue(|reply| Request::Batch(statements, reply)).await?;
        answer.await.map_err(|_| {
            StorageError::Uncertain(Unacknowledged::sent("the sqlite worker stopped mid-batch"))
        })?
    }

    async fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let tx = self.worker.lock().await.tx.clone();
        let (done, closed) = lk_agents_net::signal::once();
        if tx.send(Request::Close(done)).await.is_ok() {
            closed.until(Option::is_some).await;
        }
    }
}

pub(super) fn connect(
    path: &Path,
    create: bool,
    clock: &Arc<dyn Clock>,
) -> Result<rusqlite::Connection, StorageError> {
    use rusqlite::OpenFlags;
    let mut flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    if create {
        flags |= OpenFlags::SQLITE_OPEN_CREATE;
    }
    let conn = rusqlite::Connection::open_with_flags(path, flags).map_err(|e| match e {
        rusqlite::Error::SqliteFailure(f, _) if f.code == rusqlite::ErrorCode::CannotOpen => {
            StorageError::NotFound(path.display().to_string())
        }
        e => sqlite_error(e),
    })?;
    conn.busy_timeout(BUSY_TIMEOUT).map_err(sqlite_error)?;
    conn.pragma_update(None, "journal_mode", "WAL").map_err(sqlite_error)?;
    conn.pragma_update(None, "synchronous", "NORMAL").map_err(sqlite_error)?;
    unixepoch(&conn, clock.clone())?;
    Ok(conn)
}

/// `unixepoch()` and `unixepoch('subsec')`, the forms the schema uses, read from `clock`.
fn unixepoch(conn: &rusqlite::Connection, clock: Arc<dyn Clock>) -> Result<(), StorageError> {
    use rusqlite::functions::FunctionFlags;
    let secs = move || {
        let wall = clock.wall().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        wall.as_secs_f64()
    };
    let flags = FunctionFlags::SQLITE_UTF8;
    let whole = secs.clone();
    conn.create_scalar_function("unixepoch", 0, flags, move |_| Ok(whole() as i64))
        .map_err(sqlite_error)?;
    conn.create_scalar_function("unixepoch", 1, flags, move |cx| match cx.get_raw(0) {
        ValueRef::Text(b"subsec") => Ok(secs()),
        _ => Err(rusqlite::Error::UserFunctionError("unixepoch takes only 'subsec'".into())),
    })
    .map_err(sqlite_error)?;
    Ok(())
}

/// Transient database failures (a busy lock, I/O, a full disk or memory) are retryable.
/// Corruption, permissions and what the caller asked wrongly (a
/// refused constraint, which is how a fence aborts a batch, a type mismatch, SQL it rejects)
/// stays `Invalid`.
pub fn sqlite_error(e: rusqlite::Error) -> StorageError {
    use rusqlite::ErrorCode as Code;
    match &e {
        rusqlite::Error::SqliteFailure(f, _)
            if matches!(
                f.code,
                Code::DatabaseBusy
                    | Code::DatabaseLocked
                    | Code::SystemIoFailure
                    | Code::DiskFull
                    | Code::CannotOpen
                    | Code::FileLockingProtocolFailed
                    | Code::OutOfMemory
            ) =>
        {
            StorageError::Unavailable(format!("sqlite: {e}"))
        }
        _ => StorageError::Invalid(format!("sqlite: {e}")),
    }
}

fn bind(stmt: &mut rusqlite::Statement<'_>, params: &[SqlValue]) -> Result<(), StorageError> {
    for (i, v) in params.iter().enumerate() {
        let i = i + 1;
        let bound = match v {
            SqlValue::Null => stmt.raw_bind_parameter(i, rusqlite::types::Null),
            SqlValue::Int(n) => stmt.raw_bind_parameter(i, n),
            SqlValue::Double(d) => stmt.raw_bind_parameter(i, d),
            SqlValue::Text(s) => stmt.raw_bind_parameter(i, s),
            SqlValue::Blob(b) => stmt.raw_bind_parameter(i, b),
        };
        bound.map_err(sqlite_error)?;
    }
    Ok(())
}

fn cell(v: ValueRef<'_>) -> Result<SqlValue, StorageError> {
    Ok(match v {
        ValueRef::Null => SqlValue::Null,
        ValueRef::Integer(n) => SqlValue::Int(n),
        ValueRef::Real(d) => SqlValue::Double(d),
        ValueRef::Text(t) => SqlValue::Text(
            std::str::from_utf8(t)
                .map_err(|e| StorageError::Invalid(format!("a text cell: {e}")))?
                .to_owned(),
        ),
        ValueRef::Blob(b) => SqlValue::Blob(b.to_vec()),
    })
}

/// One statement: its rows when it has columns (`SELECT`, `RETURNING`), else what it changed.
fn run(
    conn: &rusqlite::Connection,
    s: &Statement,
    read_only: bool,
) -> Result<StatementResult, StorageError> {
    // The batch owns BEGIN/COMMIT/ROLLBACK. Let SQLite recognize transaction control,
    // including comments and alternate spellings, before it can end our transaction.
    struct Admission<'a>(&'a rusqlite::Connection);
    impl Drop for Admission<'_> {
        fn drop(&mut self) {
            self.0.authorizer(None::<fn(AuthContext<'_>) -> Authorization>);
        }
    }
    conn.authorizer(Some(|context: AuthContext<'_>| match context.action {
        AuthAction::Transaction { .. } => Authorization::Deny,
        _ => Authorization::Allow,
    }));
    let _admission = Admission(conn);
    // SQLite parses trigger bodies and quoted/comment semicolons itself. Preparing the
    // complete tail before stepping prevents silently accepting only the first statement.
    let mut statements = rusqlite::Batch::new(conn, s.sql());
    let mut stmt = statements
        .next()
        .map_err(sqlite_error)?
        .ok_or_else(|| StorageError::Invalid("expected one SQL statement".into()))?;
    if statements.next().map_err(sqlite_error)?.is_some() {
        return Err(StorageError::Invalid("expected one SQL statement".into()));
    }
    if read_only && (!stmt.readonly() || stmt.column_count() == 0) {
        return Err(StorageError::Invalid("query requires a read-only SQL statement".into()));
    }
    bind(&mut stmt, s.params())?;
    let width = stmt.column_count();
    if width == 0 {
        let changed = stmt.raw_execute().map_err(sqlite_error)?;
        return Ok(StatementResult::Exec {
            rows_affected: changed as i64,
            last_insert_id: conn.last_insert_rowid(),
        });
    }
    let mut rows = stmt.raw_query();
    let mut out = Vec::new();
    while let Some(row) = rows.next().map_err(sqlite_error)? {
        let row = (0..width)
            .map(|i| cell(row.get_ref(i).map_err(sqlite_error)?))
            .collect::<Result<Row, _>>()?;
        out.push(row);
    }
    Ok(StatementResult::Rows(out))
}

pub fn query(conn: &rusqlite::Connection, s: &Statement) -> Result<Vec<Row>, StorageError> {
    match run(conn, s, true)? {
        StatementResult::Rows(rows) => Ok(rows),
        StatementResult::Exec { .. } => Ok(Vec::new()),
    }
}

/// The statements in one immediate transaction: all committed, or none.
pub fn batch(
    conn: &mut rusqlite::Connection,
    statements: &[Statement],
) -> Result<BatchResult, StorageError> {
    let tx = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(sqlite_error)?;
    let mut results = Vec::with_capacity(statements.len());
    for statement in statements {
        results.push(run(&tx, statement, false)?);
        if tx.is_autocommit() {
            return Err(StorageError::Invalid("the batch's transaction ended unexpectedly".into()));
        }
    }
    tx.commit().map_err(sqlite_error)?;
    Ok(BatchResult { results })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A worker keeps the queue's order. One that dies with a batch queued leaves it `Uncertain`;
    /// the next request starts a fresh worker over the same database.
    #[tokio::test(start_paused = true)]
    async fn a_worker_keeps_order_and_dying_with_a_batch_queued_leaves_it_uncertain() {
        let dir = crate::testing::TempDir::new("worker_");
        let dying = Arc::new(crate::testing::Dying::new(Arc::new(
            lk_agents_net::task::TokioSpawner::current(),
        )));
        let workers = Workers { spawner: dying.clone(), background: crate::testing::background() };
        let conn = FileConnection::open(dir.0.join("db.sqlite"), true, workers).await.unwrap();
        conn.batch(vec![Statement::raw("CREATE TABLE notes (n INTEGER)")]).await.unwrap();
        let insert = |n: i64| {
            let insert =
                Statement::raw("INSERT INTO notes VALUES (?1)").with(vec![SqlValue::Int(n)]);
            conn.batch(vec![insert])
        };
        let (a, b, c) = tokio::join!(insert(1), insert(2), insert(3));
        for applied in [a, b, c] {
            applied.unwrap();
        }
        dying.die_next_poll();
        let lost = insert(4).await;
        assert!(matches!(lost, Err(StorageError::Uncertain(_))), "{lost:?}");
        insert(5).await.unwrap();
        let rows = conn.query(Statement::raw("SELECT n FROM notes ORDER BY rowid")).await.unwrap();
        let order: Vec<Row> = [1, 2, 3, 5].map(|n| vec![SqlValue::Int(n)]).to_vec();
        assert_eq!(rows, order, "applied in queue order; the lost batch never ran");
        conn.close().await;
        let closed = conn.query(Statement::raw("SELECT 1")).await;
        assert!(matches!(closed, Err(StorageError::Unavailable(_))), "{closed:?}");
    }
    #[test]
    fn corrupt_or_forbidden_databases_are_not_retryable() {
        for code in [
            rusqlite::ffi::SQLITE_CORRUPT,
            rusqlite::ffi::SQLITE_NOTADB,
            rusqlite::ffi::SQLITE_PERM,
            rusqlite::ffi::SQLITE_READONLY,
        ] {
            let error =
                sqlite_error(rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(code), None));
            assert!(!error.retryable(), "{error}");
        }
        let busy = sqlite_error(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
            None,
        ));
        assert!(busy.retryable());
    }
    #[test]
    fn transaction_control_cannot_partially_commit_a_batch() {
        let mut conn = rusqlite::Connection::open_in_memory().unwrap();
        batch(&mut conn, &[Statement::raw("CREATE TABLE notes (body TEXT)")]).unwrap();
        for control in ["COMMIT", "END TRANSACTION", "/* comment */ cOmMiT", "ROLLBACK", "BEGIN"] {
            let result = batch(
                &mut conn,
                &[
                    Statement::raw("INSERT INTO notes VALUES ('must roll back')"),
                    Statement::raw(control),
                    Statement::raw("INSERT INTO nonexistent VALUES (1)"),
                ],
            );
            assert!(matches!(result, Err(StorageError::Invalid(_))), "{control}: {result:?}");
            assert!(query(&conn, &Statement::raw("SELECT * FROM notes")).unwrap().is_empty());
            assert!(conn.is_autocommit());
        }
        // Nested application savepoints are safe inside the framework's transaction.
        batch(
            &mut conn,
            &[
                Statement::raw("SAVEPOINT app"),
                Statement::raw("INSERT INTO notes VALUES ('discard')"),
                Statement::raw("ROLLBACK TO app"),
                Statement::raw("RELEASE app"),
                Statement::raw("INSERT INTO notes VALUES ('saved')"),
            ],
        )
        .unwrap();
        assert_eq!(
            query(&conn, &Statement::raw("SELECT * FROM notes")).unwrap(),
            vec![vec![SqlValue::Text("saved".into())]]
        );
    }

    #[test]
    fn queries_cannot_change_data_or_schema() {
        let mut conn = rusqlite::Connection::open_in_memory().unwrap();
        batch(
            &mut conn,
            &[
                Statement::raw("CREATE TABLE notes (body TEXT)"),
                Statement::raw("INSERT INTO notes VALUES ('saved')"),
            ],
        )
        .unwrap();
        for sql in [
            "UPDATE notes SET body = 'changed' RETURNING body",
            "DELETE FROM notes RETURNING body",
            "INSERT INTO notes VALUES ('new')",
            "CREATE TABLE unwanted (body TEXT)",
            "DROP TABLE notes",
        ] {
            assert!(query(&conn, &Statement::raw(sql)).is_err(), "accepted {sql}");
            assert_eq!(
                query(&conn, &Statement::raw("SELECT body FROM notes")).unwrap(),
                vec![vec![SqlValue::Text("saved".into())]]
            );
        }
        assert_eq!(
            query(
                &conn,
                &Statement::raw("SELECT name FROM sqlite_master WHERE name = 'unwanted'"),
            )
            .unwrap(),
            Vec::<Row>::new(),
        );
    }

    #[test]
    fn a_statement_tail_is_refused_before_either_write() {
        let mut conn = rusqlite::Connection::open_in_memory().unwrap();
        batch(&mut conn, &[Statement::raw("CREATE TABLE notes (body TEXT)")]).unwrap();
        let sql = "INSERT INTO notes VALUES ('first'); INSERT INTO notes VALUES ('second')";
        assert!(batch(&mut conn, &[Statement::raw(sql)]).is_err());
        assert!(query(&conn, &Statement::raw(sql)).is_err());
        assert!(query(&conn, &Statement::raw("SELECT body FROM notes")).unwrap().is_empty());

        // SQLite owns parsing: semicolons in identifiers, strings, comments and trigger
        // bodies remain part of the single statement.
        batch(
            &mut conn,
            &[
                Statement::raw("CREATE TABLE audit (body TEXT)"),
                Statement::raw(
                    "CREATE TRIGGER \"changed;\" AFTER INSERT ON notes BEGIN \
                     INSERT INTO audit VALUES ('first;'); \
                     INSERT INTO audit VALUES ('second;'); END; -- tail;",
                ),
                Statement::raw("INSERT INTO notes VALUES ('one;two'); /* comment; */ ;"),
            ],
        )
        .unwrap();
        assert_eq!(
            query(&conn, &Statement::raw("SELECT body FROM audit ORDER BY rowid")).unwrap(),
            vec![vec![SqlValue::Text("first;".into())], vec![SqlValue::Text("second;".into())]]
        );
    }
}
