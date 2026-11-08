use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use rusqlite::types::ValueRef;
use tokio::sync::{mpsc, oneshot};

use crate::sql::{BatchResult, Row, SqlValue, Statement, StatementResult};
use crate::{StorageError, ThreadDbConnection};

// Each database file is served by its own worker thread, so no query blocks the executor.

/// How long a statement waits for another process's write lock before failing retryably.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);
/// Requests queued to one connection thread before senders wait.
const QUEUE: usize = 32;

pub(super) enum Request {
    Query(Statement, oneshot::Sender<Result<Vec<Row>, StorageError>>),
    Batch(Vec<Statement>, oneshot::Sender<Result<BatchResult, StorageError>>),
    DeleteThread(PathBuf, String, oneshot::Sender<Result<(), StorageError>>),
    Close(oneshot::Sender<()>),
}

/// One SQLite database, executed on a thread of its own; requests are served in order.
pub struct FileConnection {
    tx: mpsc::Sender<Request>,
    closed: AtomicBool,
}

impl FileConnection {
    /// Opens (creating when `create`) the file and starts its thread.
    pub async fn open(path: PathBuf, create: bool) -> Result<Self, StorageError> {
        Self::open_with_access(path, create, None).await
    }

    pub(super) async fn open_with_access(
        path: PathBuf,
        create: bool,
        thread: Option<(PathBuf, String)>,
    ) -> Result<Self, StorageError> {
        let (tx, rx) = mpsc::channel::<Request>(QUEUE);
        let (ready, opened) = oneshot::channel();
        std::thread::Builder::new()
            .name("lk-agents-sqlite".into())
            .spawn(move || match SqliteThread::open(&path, create, thread) {
                Ok(sqlite) => {
                    let _ = ready.send(Ok(()));
                    sqlite.serve(rx);
                }
                Err(error) => {
                    let _ = ready.send(Err(error));
                }
            })
            .map_err(|e| StorageError::Unavailable(format!("sqlite thread: {e}")))?;
        opened.await.map_err(|_| closed())??;
        Ok(Self { tx, closed: AtomicBool::new(false) })
    }

    pub(super) async fn send<T>(
        &self,
        request: impl FnOnce(oneshot::Sender<Result<T, StorageError>>) -> Request,
    ) -> Result<T, StorageError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(closed());
        }
        let (reply, answer) = oneshot::channel();
        self.tx.send(request(reply)).await.map_err(|_| closed())?;
        answer.await.map_err(|_| closed())?
    }
}

/// The thread that owns one SQLite connection and serves its requests in order.
struct SqliteThread {
    conn: rusqlite::Connection,
    access: Option<super::lifecycle::Access>,
}

impl SqliteThread {
    fn open(
        path: &Path,
        create: bool,
        thread: Option<(PathBuf, String)>,
    ) -> Result<Self, StorageError> {
        let access = thread
            .map(|(directory, id)| super::lifecycle::Access::open(&directory, id))
            .transpose()?;
        let conn = super::lifecycle::with_access(access.as_ref(), || connect(path, create))?;
        Ok(Self { conn, access })
    }

    /// Serves until a close request or until every sender is gone; closes the connection once.
    fn serve(mut self, mut rx: mpsc::Receiver<Request>) {
        while let Some(request) = rx.blocking_recv() {
            if let ControlFlow::Break(done) = self.handle(request) {
                super::lifecycle::close(self.conn, self.access.as_ref());
                let _ = done.send(());
                return;
            }
        }
        super::lifecycle::close(self.conn, self.access.as_ref());
    }

    /// Breaks with the close request's reply, sent once the connection has closed.
    fn handle(&mut self, request: Request) -> ControlFlow<oneshot::Sender<()>> {
        let access = self.access.as_ref();
        match request {
            Request::Query(s, reply) => {
                let _ = reply.send(super::lifecycle::with_access(access, || query(&self.conn, &s)));
            }
            Request::Batch(statements, reply) => {
                let run = || batch(&mut self.conn, &statements);
                let _ = reply.send(super::lifecycle::with_access(access, run));
            }
            Request::DeleteThread(directory, id, reply) => {
                let _ = reply.send(super::lifecycle::delete(&mut self.conn, &directory, &id));
            }
            Request::Close(done) => return ControlFlow::Break(done),
        }
        ControlFlow::Continue(())
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

    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError> {
        self.send(|reply| Request::Batch(statements, reply)).await
    }

    async fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let (done, closed) = oneshot::channel();
        if self.tx.send(Request::Close(done)).await.is_ok() {
            let _ = closed.await;
        }
    }
}

pub(super) fn connect(path: &Path, create: bool) -> Result<rusqlite::Connection, StorageError> {
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
    Ok(conn)
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
