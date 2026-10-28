//! [`SqliteStorage`]: threads in a local directory. `registry.sqlite` holds thread metadata,
//! `threads/<AT_id>.sqlite` each thread's database, every lane in it, and `streams.sqlite` the
//! response streams. Separate processes sharing the directory coordinate through the engine's
//! database leases and fences. Each open file is executed by its own connection thread
//! ([`FileConnection`]), so no query blocks the executor.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use livekit_protocol::agent::agent_thread::{
    CreateRequest, ListRequest, ListResponse, Thread as ThreadRecord, UpdateRequest,
};
use lk_agents_storage::StorageError;
use lk_agents_storage::db;
use lk_agents_storage::sql::{BatchResult, Row, SqlValue, Statement};
use lk_agents_storage::sqlite::{batch, query, sqlite_error};
use tokio::sync::{Mutex, mpsc, oneshot};

use super::engine::{LeaseTiming, SchemaCell, Thread, ThreadDb};
use super::streams::{LocalStreams, ResponseStreams};
use super::{SqliteConnection, Storage, shortuuid};

/// How long a statement waits for another process's write lock before failing retryably.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);
/// Requests queued to one connection thread before senders wait.
const QUEUE: usize = 32;
/// The largest `ListThreads` page.
const MAX_THREAD_PAGE: i32 = 100;
const DEFAULT_THREAD_PAGE: i32 = 50;

enum Request {
    Query(Statement, oneshot::Sender<Result<Vec<Row>, StorageError>>),
    Batch(Vec<Statement>, oneshot::Sender<Result<BatchResult, StorageError>>),
    Close(oneshot::Sender<()>),
}

/// One SQLite file, executed on a thread of its own; requests are served in order.
pub struct FileConnection {
    tx: mpsc::Sender<Request>,
    closed: AtomicBool,
}

impl FileConnection {
    /// Opens (creating when `create`) the file and starts its thread.
    pub async fn open(path: PathBuf, create: bool) -> Result<Self, StorageError> {
        let (tx, mut rx) = mpsc::channel::<Request>(QUEUE);
        let (ready, opened) = oneshot::channel();
        std::thread::Builder::new()
            .name("lk-agents-sqlite".into())
            .spawn(move || {
                let mut conn = match connect(&path, create) {
                    Ok(conn) => {
                        let _ = ready.send(Ok(()));
                        conn
                    }
                    Err(e) => {
                        let _ = ready.send(Err(e));
                        return;
                    }
                };
                while let Some(request) = rx.blocking_recv() {
                    match request {
                        Request::Query(s, reply) => {
                            let _ = reply.send(query(&conn, &s));
                        }
                        Request::Batch(statements, reply) => {
                            let _ = reply.send(batch(&mut conn, &statements));
                        }
                        Request::Close(done) => {
                            drop(conn);
                            let _ = done.send(());
                            return;
                        }
                    }
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

fn closed() -> StorageError {
    StorageError::Unavailable("the database is closed".into())
}

#[async_trait]
impl SqliteConnection for FileConnection {
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
    let mut flags = OpenFlags::SQLITE_OPEN_READ_WRITE
        | OpenFlags::SQLITE_OPEN_NO_MUTEX
        | OpenFlags::SQLITE_OPEN_URI;
    if create {
        flags |= OpenFlags::SQLITE_OPEN_CREATE;
    }
    let conn = rusqlite::Connection::open_with_flags(path, flags).map_err(|e| match e {
        rusqlite::Error::SqliteFailure(f, _) if f.code == rusqlite::ErrorCode::CannotOpen => {
            StorageError::ThreadNotFound(path.display().to_string())
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
    if ok { Ok(id) } else { Err(StorageError::ThreadNotFound(id.to_owned())) }
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

enum Databases {
    Directory(PathBuf),
    Memory { namespace: String, keepers: Mutex<HashMap<String, FileConnection>> },
}

/// SQLite threads, either in an explicit local directory or owned entirely in memory.
pub struct SqliteStorage {
    databases: Databases,
    registry: FileConnection,
    streams: Arc<LocalStreams>,
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
        tokio::fs::create_dir_all(&threads)
            .await
            .map_err(|e| StorageError::Invalid(format!("{}: {e}", threads.display())))?;
        let registry = FileConnection::open(directory.join("registry.sqlite"), true).await?;
        registry.batch(vec![Statement::raw(REGISTRY_SCHEMA)]).await?;
        let streams = Arc::new(LocalStreams::open(directory.join("streams.sqlite")).await?);
        Ok(Self { databases: Databases::Directory(directory), registry, streams, lease })
    }

    /// Uses the same thread engine and statements without creating any files.
    pub async fn in_memory(lease: LeaseTiming) -> Result<Self, StorageError> {
        let registry = FileConnection::open(":memory:".into(), true).await?;
        registry.batch(vec![Statement::raw(REGISTRY_SCHEMA)]).await?;
        let streams = Arc::new(LocalStreams::open(":memory:".into()).await?);
        Ok(Self {
            databases: Databases::Memory {
                namespace: shortuuid("memory_"),
                keepers: Mutex::new(HashMap::new()),
            },
            registry,
            streams,
            lease,
        })
    }

    fn file(&self, thread_id: &str) -> Result<PathBuf, StorageError> {
        let id = valid_thread_id(thread_id)?;
        Ok(match &self.databases {
            Databases::Directory(directory) => {
                directory.join("threads").join(format!("{id}.sqlite"))
            }
            Databases::Memory { namespace, .. } => {
                format!("file:{namespace}_{id}?mode=memory&cache=shared").into()
            }
        })
    }

    /// The registry row of a live thread; an expired one is deleted on the way.
    async fn live(&self, thread_id: &str) -> Result<ThreadRecord, StorageError> {
        valid_thread_id(thread_id)?;
        let select = Statement {
            sql: format!("SELECT {THREAD_COLUMNS} FROM threads WHERE thread_id = ?1").into(),
            params: vec![thread_id.into()],
        };
        let rows = self.registry.query(select).await?;
        let row = rows.first().ok_or_else(|| StorageError::ThreadNotFound(thread_id.into()))?;
        let (record, expires_at) = thread_record(row)?;
        if expires_at != 0 && expires_at <= unix_now() {
            self.delete_thread(thread_id).await?;
            return Err(StorageError::ThreadNotFound(thread_id.into()));
        }
        Ok(record)
    }
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

fn timestamp(secs: i64) -> Option<pbjson_types::Timestamp> {
    (secs != 0).then_some(pbjson_types::Timestamp { seconds: secs, nanos: 0 })
}

/// A registry row as the threads service's record, and its expiry (`0` never).
fn thread_record(row: &Row) -> Result<(ThreadRecord, i64), StorageError> {
    let int = |i: usize| row.get(i).and_then(SqlValue::as_i64).unwrap_or(0);
    let text = |i: usize| row.get(i).and_then(SqlValue::as_str).unwrap_or("").to_owned();
    let attributes: HashMap<String, String> = serde_json::from_str(&text(2))
        .map_err(|e| StorageError::Decode(format!("thread attributes: {e}")))?;
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

fn expiry(last_active: i64, idle_ttl: i64) -> i64 {
    if idle_ttl == 0 { 0 } else { last_active + idle_ttl }
}

#[async_trait]
impl Storage for SqliteStorage {
    /// The database file with its schema first, then the registry row: a crash between the two
    /// leaves a file no lookup finds.
    async fn create_thread(&self, request: CreateRequest) -> Result<ThreadRecord, StorageError> {
        let thread_id = shortuuid("AT_");
        let file = FileConnection::open(self.file(&thread_id)?, true).await?;
        file.batch(db::bootstrap()).await?;
        match &self.databases {
            Databases::Directory(_) => file.close().await,
            Databases::Memory { keepers, .. } => {
                // A keeper keeps the shared memory database alive between ordinary handles.
                keepers.lock().await.insert(thread_id.clone(), file);
            }
        }
        let now = unix_now();
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
        self.registry.batch(vec![insert]).await?;
        self.live(&thread_id).await
    }

    async fn get_thread(&self, thread_id: &str) -> Result<ThreadRecord, StorageError> {
        self.live(thread_id).await
    }

    /// Newest first, then by id; the token is the last row's `created_at:thread_id`.
    async fn list_threads(&self, request: ListRequest) -> Result<ListResponse, StorageError> {
        let limit = match request.limit {
            0 => DEFAULT_THREAD_PAGE,
            n => n.clamp(1, MAX_THREAD_PAGE),
        };
        let (before_at, before_id) = match request.page_token.as_str() {
            "" => (i64::MAX, String::new()),
            token => token
                .split_once(':')
                .and_then(|(at, id)| Some((at.parse().ok()?, id.to_owned())))
                .ok_or_else(|| StorageError::Invalid(format!("page token {token:?}")))?,
        };
        let select = Statement {
            sql: format!(
                "SELECT {THREAD_COLUMNS} FROM threads WHERE scope = ?1 \
                 AND (expires_at = 0 OR expires_at > ?2) \
                 AND (created_at < ?3 OR (created_at = ?3 AND thread_id < ?4)) \
                 ORDER BY created_at DESC, thread_id DESC LIMIT ?5"
            )
            .into(),
            params: vec![
                request.scope.as_str().into(),
                unix_now().into(),
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
                format!("{}:{}", t.created_at.as_ref().map_or(0, |c| c.seconds), t.thread_id)
            }
            _ => String::new(),
        };
        Ok(ListResponse { threads, next_page_token })
    }

    async fn update_thread(&self, request: UpdateRequest) -> Result<ThreadRecord, StorageError> {
        let current = self.live(&request.thread_id).await?;
        let mut attributes = current.attributes;
        for (k, v) in request.attributes {
            match v.is_empty() {
                true => attributes.remove(&k),
                false => attributes.insert(k, v),
            };
        }
        let idle_ttl = match &request.idle_ttl {
            Some(d) => ttl_secs(Some(d)),
            None => ttl_secs(current.idle_ttl.as_ref()),
        };
        let now = unix_now();
        let update = Statement {
            sql: "UPDATE threads SET attributes = ?2, last_active = ?3, idle_ttl = ?4, \
                  expires_at = ?5 WHERE thread_id = ?1"
                .into(),
            params: vec![
                request.thread_id.as_str().into(),
                SqlValue::json(&attributes),
                now.into(),
                idle_ttl.into(),
                expiry(now, idle_ttl).into(),
            ],
        };
        self.registry.batch(vec![update]).await?;
        self.live(&request.thread_id).await
    }

    /// The registry row first, which makes the thread unavailable, then its files.
    async fn delete_thread(&self, thread_id: &str) -> Result<(), StorageError> {
        let file = self.file(thread_id)?;
        let delete = Statement {
            sql: "DELETE FROM threads WHERE thread_id = ?1".into(),
            params: vec![thread_id.into()],
        };
        self.registry.batch(vec![delete]).await?;
        if let Databases::Memory { keepers, .. } = &self.databases {
            let keeper = keepers.lock().await.remove(thread_id);
            if let Some(keeper) = keeper {
                keeper.close().await;
            }
            return Ok(());
        }
        for suffix in ["", "-wal", "-shm"] {
            let mut path = file.clone().into_os_string();
            path.push(suffix);
            match tokio::fs::remove_file(&path).await {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                    return Err(StorageError::Unavailable(format!("{}: {e}", file.display())));
                }
                _ => {}
            }
        }
        Ok(())
    }

    async fn open(&self, thread_id: &str, holder: &str) -> Result<Thread, StorageError> {
        self.live(thread_id).await?;
        let conn = FileConnection::open(self.file(thread_id)?, false).await?;
        let db =
            ThreadDb::new(Arc::new(conn), thread_id, self.lease, SchemaCell::default()).await?;
        Ok(Thread::new(db, holder))
    }

    fn response_streams(&self) -> Arc<dyn ResponseStreams> {
        self.streams.clone()
    }

    async fn close(&self) {
        self.registry.close().await;
        self.streams.shutdown().await;
        if let Databases::Memory { keepers, .. } = &self.databases {
            let connections = std::mem::take(&mut *keepers.lock().await);
            for connection in connections.into_values() {
                connection.close().await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_plain_at_ids_name_a_file() {
        assert!(valid_thread_id("AT_abc-1_2").is_ok());
        for bad in ["", "AT_", "AT_../x", "AT_a/b", "DB_abc", "abc", "AT_a.sqlite"] {
            assert!(matches!(valid_thread_id(bad), Err(StorageError::ThreadNotFound(_))), "{bad}");
        }
    }
}
