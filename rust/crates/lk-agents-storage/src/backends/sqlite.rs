//! SQLite threads in an explicit local directory. `registry.sqlite` holds metadata,
//! `threads/<AT_id>.sqlite` each thread. Processes sharing
//! a directory coordinate through database leases and fences. Each connection has its own
//! worker ([`FileConnection`]), so no query blocks the executor.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use livekit_protocol::agent::agent_thread::{
    CreateRequest, ListRequest, ListResponse, Thread as ThreadRecord, UpdateRequest,
};
use lk_agents_net::clock::{unix_now, unix_now_ms};
use tokio::sync::{mpsc, oneshot};

use crate::StorageError;
use crate::sql::{BatchResult, Row, SqlValue, Statement};
use crate::thread::db;
use crate::thread::{LeaseTiming, SchemaCell, Thread};
use crate::{Storage, ThreadDbConnection, shortuuid};

pub use exec::{batch, query, sqlite_error};

/// How long a statement waits for another process's write lock before failing retryably.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);
/// Requests queued to one connection thread before senders wait.
const QUEUE: usize = 32;
/// The largest `ListThreads` page.
const MAX_THREAD_PAGE: i32 = 200;
const DEFAULT_THREAD_PAGE: i32 = 100;

enum Request {
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

    async fn open_with_access(
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

    async fn send<T>(
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
    access: Option<lifecycle::Access>,
}

impl SqliteThread {
    fn open(
        path: &Path,
        create: bool,
        thread: Option<(PathBuf, String)>,
    ) -> Result<Self, StorageError> {
        let access =
            thread.map(|(directory, id)| lifecycle::Access::open(&directory, id)).transpose()?;
        let conn = lifecycle::with_access(access.as_ref(), || connect(path, create))?;
        Ok(Self { conn, access })
    }

    /// Serves until a close request or until every sender is gone; closes the connection once.
    fn serve(mut self, mut rx: mpsc::Receiver<Request>) {
        while let Some(request) = rx.blocking_recv() {
            if let ControlFlow::Break(done) = self.handle(request) {
                lifecycle::close(self.conn, self.access.as_ref());
                let _ = done.send(());
                return;
            }
        }
        lifecycle::close(self.conn, self.access.as_ref());
    }

    /// Breaks with the close request's reply, sent once the connection has closed.
    fn handle(&mut self, request: Request) -> ControlFlow<oneshot::Sender<()>> {
        let access = self.access.as_ref();
        match request {
            Request::Query(s, reply) => {
                let _ = reply.send(lifecycle::with_access(access, || query(&self.conn, &s)));
            }
            Request::Batch(statements, reply) => {
                let run = || batch(&mut self.conn, &statements);
                let _ = reply.send(lifecycle::with_access(access, run));
            }
            Request::DeleteThread(directory, id, reply) => {
                let _ = reply.send(lifecycle::delete(&mut self.conn, &directory, &id));
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

fn connect(path: &Path, create: bool) -> Result<rusqlite::Connection, StorageError> {
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

/// A local thread id: `AT_` then letters, digits, `-` or `_`; anything else is refused before a
/// path is built from it.
fn valid_thread_id(id: &str) -> Result<&str, StorageError> {
    let rest = id.strip_prefix("AT_").unwrap_or("");
    let ok = !rest.is_empty()
        && rest.len() <= 64
        && rest.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    if ok { Ok(id) } else { Err(StorageError::NotFound(id.to_owned())) }
}

const REGISTRY_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS threads (
  thread_id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  attributes TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  last_active INTEGER NOT NULL,
  idle_ttl INTEGER NOT NULL,
  expires_at INTEGER NOT NULL
)";
const THREAD_COLUMNS: &str =
    "thread_id, scope, attributes, created_at, last_active, idle_ttl, expires_at";

/// SQLite threads in an explicit local directory.
pub struct SqliteStorage {
    directory: PathBuf,
    registry: FileConnection,
    lease: LeaseTiming,
}

impl SqliteStorage {
    /// Opens the directory, creating it and its registry when missing. A relative path resolves
    /// against the working directory, once.
    pub async fn open(
        directory: impl AsRef<Path>,
        lease: LeaseTiming,
    ) -> Result<Self, StorageError> {
        let directory = std::path::absolute(directory.as_ref())
            .map_err(|e| StorageError::Invalid(format!("storage directory: {e}")))?;
        let threads = directory.join("threads");
        let created = threads.clone();
        lk_agents_net::blocking_io::blocking_io()
            .run((), move || std::fs::create_dir_all(created))
            .await
            .map_err(|e| StorageError::Invalid(format!("{}: {e}", threads.display())))?
            .map_err(|e| StorageError::Invalid(format!("{}: {e}", threads.display())))?;
        let registry = FileConnection::open(directory.join("registry.sqlite"), true).await?;
        registry.batch(vec![
            Statement::raw(REGISTRY_SCHEMA),
            // Older local registries recorded activity in seconds. Normalize once so updates
            // within a second have the same ordering as the Cloud threads service.
            Statement::raw("UPDATE threads SET \
                created_at = CASE WHEN created_at < 100000000000 THEN created_at * 1000 ELSE created_at END, \
                last_active = CASE WHEN last_active < 100000000000 THEN last_active * 1000 ELSE last_active END \
                WHERE created_at < 100000000000 OR last_active < 100000000000"),
        ]).await?;
        Ok(Self { directory, registry, lease })
    }

    fn file(&self, thread_id: &str) -> Result<PathBuf, StorageError> {
        let id = valid_thread_id(thread_id)?;
        Ok(self.directory.join("threads").join(format!("{id}.sqlite")))
    }

    /// The registry row of a live thread; an expired one is deleted on the way.
    async fn live(&self, thread_id: &str) -> Result<ThreadRecord, StorageError> {
        valid_thread_id(thread_id)?;
        let select = Statement {
            sql: format!("SELECT {THREAD_COLUMNS} FROM threads WHERE thread_id = ?1").into(),
            params: vec![thread_id.into()],
        };
        let rows = self.registry.query(select).await?;
        let row = rows.first().ok_or_else(|| StorageError::NotFound(thread_id.into()))?;
        let (record, expires_at) = thread_record(row)?;
        if expires_at != 0 && expires_at <= unix_now() as i64 {
            self.delete_thread(thread_id).await?;
            return Err(StorageError::NotFound(thread_id.into()));
        }
        Ok(record)
    }
}

fn timestamp(value: i64) -> Option<pbjson_types::Timestamp> {
    (value != 0).then_some(if value >= 100000000000 {
        pbjson_types::Timestamp {
            seconds: value / 1000,
            nanos: ((value % 1000) * 1_000_000) as i32,
        }
    } else {
        pbjson_types::Timestamp { seconds: value, nanos: 0 }
    })
}

/// A registry row as the threads service's record, and its expiry (`0` never).
fn thread_record(row: &Row) -> Result<(ThreadRecord, i64), StorageError> {
    let int = |i: usize| row.get(i).and_then(SqlValue::as_i64).unwrap_or(0);
    let text = |i: usize| row.get(i).and_then(SqlValue::as_str).unwrap_or("").to_owned();
    let attributes: HashMap<String, String> = serde_json::from_str(&text(2))
        .map_err(|e| StorageError::Unavailable(format!("thread attributes: {e}")))?;
    let expires_at = int(6);
    let record = ThreadRecord {
        thread_id: text(0),
        scope: text(1),
        attributes,
        created_at: timestamp(int(3)),
        last_active: timestamp(int(4)),
        expires_at: timestamp(expires_at),
        idle_ttl: Some(pbjson_types::Duration { seconds: int(5), nanos: 0 }),
        database_id: text(0),
    };
    Ok((record, expires_at))
}

fn ttl_secs(d: Option<&pbjson_types::Duration>) -> i64 {
    d.map_or(0, |d| d.seconds + i64::from(d.nanos > 0))
}

fn expiry(last_active_ms: i64, idle_ttl: i64) -> i64 {
    if idle_ttl == 0 { 0 } else { last_active_ms / 1000 + idle_ttl }
}

#[async_trait]
impl Storage for SqliteStorage {
    /// The database file with its schema first, then the registry row: a crash between the two
    /// leaves a file no lookup finds.
    async fn create_thread(&self, request: CreateRequest) -> Result<ThreadRecord, StorageError> {
        let thread_id = shortuuid("AT_");
        let connection = FileConnection::open(self.file(&thread_id)?, true).await?;
        let now = unix_now_ms();
        let idle_ttl = ttl_secs(request.idle_ttl.as_ref());
        let attributes: HashMap<&String, &String> =
            request.attributes.iter().filter(|(_, v)| !v.is_empty()).collect();
        let insert = Statement {
            sql: format!(
                "INSERT INTO threads ({THREAD_COLUMNS}) VALUES (?1, ?2, ?3, ?4, ?4, ?5, ?6)"
            )
            .into(),
            params: vec![
                thread_id.as_str().into(),
                request.scope.as_str().into(),
                SqlValue::json(&attributes),
                now.into(),
                idle_ttl.into(),
                expiry(now, idle_ttl).into(),
            ],
        };
        let created = async {
            connection.batch(db::bootstrap()).await?;
            self.registry.batch(vec![insert]).await?;
            Ok::<(), StorageError>(())
        }
        .await;
        connection.close().await;
        created?;
        self.live(&thread_id).await
    }

    async fn get_thread(&self, thread_id: &str) -> Result<ThreadRecord, StorageError> {
        self.live(thread_id).await
    }

    /// Last active first, then by id; the cursor is bound to its scope.
    async fn list_threads(&self, request: ListRequest) -> Result<ListResponse, StorageError> {
        let limit = match request.limit {
            0 => DEFAULT_THREAD_PAGE,
            n => n.clamp(1, MAX_THREAD_PAGE),
        };
        let (before_at, before_id) = match request.page_token.as_str() {
            "" => (i64::MAX, String::new()),
            token => {
                let (scope, at, id): (String, i64, String) = serde_json::from_str(token)
                    .map_err(|_| StorageError::Invalid("invalid thread cursor".into()))?;
                if scope != request.scope {
                    return Err(StorageError::Invalid("cursor belongs to another scope".into()));
                }
                (at, id)
            }
        };
        let select = Statement {
            sql: format!(
                "SELECT {THREAD_COLUMNS} FROM threads WHERE scope = ?1 \
                 AND (expires_at = 0 OR expires_at > ?2) \
                 AND (last_active < ?3 OR (last_active = ?3 AND thread_id < ?4)) \
                 ORDER BY last_active DESC, thread_id DESC LIMIT ?5"
            )
            .into(),
            params: vec![
                request.scope.as_str().into(),
                (unix_now() as i64).into(),
                before_at.into(),
                before_id.into(),
                i64::from(limit + 1).into(),
            ],
        };
        let rows = self.registry.query(select).await?;
        let mut threads =
            rows.iter().map(|r| thread_record(r).map(|(t, _)| t)).collect::<Result<Vec<_>, _>>()?;
        let more = threads.len() > limit as usize;
        threads.truncate(limit as usize);
        let next_page_token = match threads.last() {
            Some(t) if more => {
                let at = t
                    .last_active
                    .as_ref()
                    .map_or(0, |c| c.seconds * 1000 + i64::from(c.nanos) / 1_000_000);
                serde_json::to_string(&(&request.scope, at, &t.thread_id))
                    .map_err(|error| StorageError::Invalid(error.to_string()))?
            }
            _ => String::new(),
        };
        Ok(ListResponse { threads, next_page_token })
    }

    async fn update_thread(&self, request: UpdateRequest) -> Result<ThreadRecord, StorageError> {
        valid_thread_id(&request.thread_id)?;
        let now = unix_now_ms();
        let update = Statement {
            sql: format!(
                "UPDATE threads SET \
                 attributes = json_patch(attributes, \
                   (SELECT json_group_object(key, NULLIF(value, '')) FROM json_each(?2))), \
                 last_active = ?3, idle_ttl = COALESCE(?4, idle_ttl), \
                 expires_at = CASE WHEN COALESCE(?4, idle_ttl) = 0 THEN 0 \
                                   ELSE ?3 / 1000 + COALESCE(?4, idle_ttl) END \
                 WHERE thread_id = ?1 AND (expires_at = 0 OR expires_at > ?3 / 1000) \
                 RETURNING {THREAD_COLUMNS}"
            )
            .into(),
            params: vec![
                request.thread_id.as_str().into(),
                SqlValue::json(&request.attributes),
                now.into(),
                request.idle_ttl.as_ref().map(|d| ttl_secs(Some(d))).into(),
            ],
        };
        let result = self.registry.batch(vec![update]).await?;
        let row = result
            .at(0)?
            .rows()
            .first()
            .ok_or_else(|| StorageError::NotFound(request.thread_id.clone()))?;
        thread_record(row).map(|(record, _)| record)
    }

    /// The registry row first, which makes the thread unavailable, then its files.
    async fn delete_thread(&self, thread_id: &str) -> Result<(), StorageError> {
        valid_thread_id(thread_id)?;
        self.registry
            .send(|reply| {
                Request::DeleteThread(self.directory.clone(), thread_id.to_owned(), reply)
            })
            .await
    }

    async fn open(&self, thread_id: &str, holder: &str) -> Result<Thread, StorageError> {
        self.live(thread_id).await?;
        let connection = FileConnection::open_with_access(
            self.file(thread_id)?,
            false,
            Some((self.directory.clone(), thread_id.to_owned())),
        )
        .await?;
        Thread::new(Arc::new(connection), thread_id, self.lease, SchemaCell::default(), holder)
            .await
    }

    async fn close(&self) {
        self.registry.close().await;
    }
}

mod exec {
    //! Executes the backend-neutral statements on a SQLite connection: what the local backend runs
    //! on its files and the storage simulation on an in-memory database. Synchronous; the caller
    //! owns the connection and where it runs.

    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
    use rusqlite::types::ValueRef;

    use crate::StorageError;
    use crate::sql::{BatchResult, Row, SqlValue, Statement, StatementResult};

    /// The database's own failure (a lock held past the busy timeout, I/O, a full disk, corruption,
    /// memory) applied nothing and is retryable: `Unavailable`. What the caller asked wrongly (a
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
                        | Code::DatabaseCorrupt
                        | Code::NotADatabase
                        | Code::CannotOpen
                        | Code::ReadOnly
                        | Code::PermissionDenied
                        | Code::FileLockingProtocolFailed
                        | Code::NoLargeFileSupport
                        | Code::OutOfMemory
                        | Code::InternalMalfunction
                        | Code::SchemaChanged
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
                    .map_err(|e| StorageError::Unavailable(format!("a text cell: {e}")))?
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
        let mut statements = rusqlite::Batch::new(conn, &s.sql);
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
        bind(&mut stmt, &s.params)?;
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
                return Err(StorageError::Invalid(
                    "the batch's transaction ended unexpectedly".into(),
                ));
            }
        }
        tx.commit().map_err(sqlite_error)?;
        Ok(BatchResult { results })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn transaction_control_cannot_partially_commit_a_batch() {
            let mut conn = rusqlite::Connection::open_in_memory().unwrap();
            batch(&mut conn, &[Statement::raw("CREATE TABLE notes (body TEXT)")]).unwrap();
            for control in
                ["COMMIT", "END TRANSACTION", "/* comment */ cOmMiT", "ROLLBACK", "BEGIN"]
            {
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
}

mod lifecycle {
    //! Coordinate local thread operations and deletion across processes. The lock file is
    //! permanent: replacing its inode would let two processes lock different files.

    use std::fs::{File, OpenOptions};
    use std::path::Path;

    use crate::StorageError;

    use super::{connect, sqlite_error};

    pub(super) struct Lock<'a>(&'a File);

    impl Drop for Lock<'_> {
        fn drop(&mut self) {
            let _ = self.0.unlock();
        }
    }

    fn lock_file(directory: &Path) -> Result<File, StorageError> {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(directory.join("lifecycle.lock"))
            .map_err(io_error)
    }

    fn io_error(error: std::io::Error) -> StorageError {
        StorageError::Unavailable(format!("storage lifecycle: {error}"))
    }

    /// Each thread worker checks current metadata while holding a shared lifecycle lock.
    pub(super) struct Access {
        lock: File,
        registry: rusqlite::Connection,
        thread_id: String,
    }

    impl Access {
        pub(super) fn open(directory: &Path, thread_id: String) -> Result<Self, StorageError> {
            let lock = lock_file(directory)?;
            let registry = connect(&directory.join("registry.sqlite"), false)?;
            Ok(Self { lock, registry, thread_id })
        }

        pub(super) fn lock(&self) -> Result<Lock<'_>, StorageError> {
            self.lock.lock_shared().map_err(io_error)?;
            Ok(Lock(&self.lock))
        }

        fn enter(&self) -> Result<Lock<'_>, StorageError> {
            let lock = self.lock()?;
            let live: bool = self
                .registry
                .query_row(
                    "SELECT EXISTS (SELECT 1 FROM threads WHERE thread_id = ?1 \
                 AND (expires_at = 0 OR expires_at > unixepoch()))",
                    [&self.thread_id],
                    |row| row.get(0),
                )
                .map_err(sqlite_error)?;
            if !live {
                return Err(StorageError::NotFound(self.thread_id.clone()));
            }
            Ok(lock)
        }
    }

    pub(super) fn with_access<T>(
        access: Option<&Access>,
        run: impl FnOnce() -> Result<T, StorageError>,
    ) -> Result<T, StorageError> {
        let _lock = access.map(Access::enter).transpose()?;
        run()
    }

    pub(super) fn close(connection: rusqlite::Connection, access: Option<&Access>) {
        let _lock = access.and_then(|access| access.lock().ok());
        drop(connection);
    }

    /// Wait for in-flight operations, hide the thread, then reclaim files. Existing workers
    /// cannot access their open SQLite connection again after the registry row is gone.
    pub(super) fn delete(
        registry: &mut rusqlite::Connection,
        directory: &Path,
        thread_id: &str,
    ) -> Result<(), StorageError> {
        let file = lock_file(directory)?;
        file.lock().map_err(io_error)?;
        let _lock = Lock(&file);
        registry
            .execute("DELETE FROM threads WHERE thread_id = ?1", [thread_id])
            .map_err(sqlite_error)?;
        let path = directory.join("threads").join(format!("{thread_id}.sqlite"));
        for suffix in ["", "-wal", "-shm"] {
            let mut path = path.clone().into_os_string();
            path.push(suffix);
            match std::fs::remove_file(path) {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                    return Err(io_error(error));
                }
                _ => {}
            }
        }
        Ok(())
    }
}
