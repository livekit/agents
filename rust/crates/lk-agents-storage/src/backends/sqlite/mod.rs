use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use livekit_protocol::agent::agent_thread::{
    CreateRequest, ListRequest, ListResponse, Thread as ThreadRecord, UpdateRequest,
};

use crate::StorageError;
use crate::backends::thread_page_limit;
use crate::cursor::Cursor;
use crate::sql::{Bind, Cells, Row, SqlValue, Statement, statement};
use crate::thread::db;
use crate::thread::{LeaseTiming, SchemaCell, Thread};
use crate::{Storage, ThreadDbConnection, shortuuid};
use lk_agents_abi::telemetry::Cx;
use lk_agents_net::owned::Background;

mod connection;
mod lifecycle;

use connection::Request;
pub use connection::{FileConnection, Workers, batch, query, sqlite_error};

// `registry.sqlite` holds metadata and `threads/<AT_id>.sqlite` each thread; processes sharing a
// directory coordinate through database leases and fences.

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

// A scope's listing seeks this index; the project's seeks the primary key.
const REGISTRY_SCOPE_INDEX: &str =
    "CREATE INDEX IF NOT EXISTS threads_scope ON threads (scope, thread_id)";

// One current registry format; unreleased layouts are not migrated.
const REGISTRY_METADATA: &str = "CREATE TABLE IF NOT EXISTS registry_schema (
  version INTEGER NOT NULL
)";

statement! {
    SetRegistryVersion {} => r#"
INSERT INTO registry_schema (version) SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM registry_schema)
"#
}

statement! {
    SelectRegistryVersion {} => "SELECT version FROM registry_schema"
}

statement! {
    /// One thread's registry row.
    SelectRegistryThread<'a> { thread_id: &'a str } => r#"
SELECT thread_id, scope, attributes, created_at, last_active, idle_ttl, expires_at
FROM threads WHERE thread_id = :thread_id
"#
}

statement! {
    /// A new thread's registry row, last active when created.
    InsertRegistryThread<'a> {
        thread_id: &'a str,
        scope: &'a str,
        attributes: SqlValue,
        now: i64,
        idle_ttl: i64,
        expires_at: i64,
    } => r#"
INSERT INTO threads (thread_id, scope, attributes, created_at, last_active, idle_ttl, expires_at)
VALUES (:thread_id, :scope, :attributes, :now, :now, :idle_ttl, :expires_at)
"#
}

// One page of live threads by id, past the previous page's last id: a scope's or the project's.
statement! {
    SelectScopeThreads<'a> { scope: &'a str, now: i64, after: &'a str, limit: i64 } => r#"
SELECT thread_id, scope, attributes, created_at, last_active, idle_ttl, expires_at
FROM threads WHERE scope = :scope AND thread_id > :after
  AND (expires_at = 0 OR expires_at > :now)
ORDER BY thread_id LIMIT :limit
"#
}

statement! {
    SelectAllThreads<'a> { now: i64, after: &'a str, limit: i64 } => r#"
SELECT thread_id, scope, attributes, created_at, last_active, idle_ttl, expires_at
FROM threads WHERE thread_id > :after AND (expires_at = 0 OR expires_at > :now)
ORDER BY thread_id LIMIT :limit
"#
}

statement! {
    /// Merges `attributes` into a live thread's (an empty value removes one), touches it and
    /// replaces its idle TTL when given, returning its row.
    UpdateRegistryThread<'a> {
        thread_id: &'a str,
        attributes: SqlValue,
        now: i64,
        idle_ttl: Option<i64>,
    } => r#"
UPDATE threads SET
  attributes = json_patch(attributes,
    (SELECT json_group_object(key, NULLIF(value, '')) FROM json_each(:attributes))),
  last_active = :now, idle_ttl = COALESCE(:idle_ttl, idle_ttl),
  expires_at = CASE WHEN COALESCE(:idle_ttl, idle_ttl) = 0 THEN 0
                    ELSE :now + COALESCE(:idle_ttl, idle_ttl) END
WHERE thread_id = :thread_id AND (expires_at = 0 OR expires_at > :now)
RETURNING thread_id, scope, attributes, created_at, last_active, idle_ttl, expires_at
"#
}

statement! {
    /// Whether the thread's registry row is live.
    RegistryThreadIsLive<'a> { thread_id: &'a str } => r#"
SELECT EXISTS (SELECT 1 FROM threads WHERE thread_id = :thread_id
  AND (expires_at = 0 OR expires_at > CAST(unixepoch('subsec') * 1000 AS INTEGER)))
"#
}

statement! {
    /// Removes the thread's registry row.
    DeleteRegistryThread<'a> { thread_id: &'a str } => r#"
DELETE FROM threads WHERE thread_id = :thread_id
"#
}

/// SQLite threads in an explicit local directory.
pub struct SqliteStorage {
    directory: PathBuf,
    registry: FileConnection,
    pub(crate) lease: LeaseTiming,
    workers: Workers,
    cx: Cx,
}

impl SqliteStorage {
    /// Opens the directory, creating it and its registry when missing. A relative path resolves
    /// against the working directory, once.
    pub async fn open(
        directory: impl AsRef<Path>,
        lease: LeaseTiming,
        workers: Workers,
    ) -> Result<Self, StorageError> {
        let directory = std::path::absolute(directory.as_ref())
            .map_err(|e| StorageError::Invalid(format!("storage directory: {e}")))?;
        let (threads, path, clock) =
            (directory.join("threads"), directory.join("registry.sqlite"), workers.clock().clone());
        let open = move || {
            std::fs::create_dir_all(&threads)
                .map_err(|e| StorageError::Invalid(format!("{}: {e}", threads.display())))?;
            connection::SqliteThread::open(&path, true, None, &clock)
        };
        let registry = FileConnection::start(workers.clone(), Arc::new(open)).await?;
        registry
            .batch(vec![
                Statement::ddl(REGISTRY_SCHEMA),
                Statement::ddl(REGISTRY_SCOPE_INDEX),
                Statement::ddl(REGISTRY_METADATA),
                SetRegistryVersion {}.bind(),
            ])
            .await?;
        let versions = registry.query(SelectRegistryVersion {}.bind()).await?;
        if versions.as_slice() != [vec![SqlValue::Int(1)]] {
            registry.close().await;
            return Err(StorageError::Invalid("unsupported registry schema version".into()));
        }
        Ok(Self { directory, registry, lease, workers, cx: Cx::NONE })
    }

    /// Its threads record their health metrics under `cx`.
    pub fn with_telemetry(mut self, cx: Cx) -> Self {
        self.cx = cx;
        self
    }

    fn now_ms(&self) -> i64 {
        (lk_agents_abi::unix_secs(self.workers.clock().wall()) * 1000.0) as i64
    }

    /// A live thread's own connection, under its lifecycle access.
    pub async fn connect(&self, thread_id: &str) -> Result<FileConnection, StorageError> {
        self.live(thread_id).await?;
        let access = Some((self.directory.clone(), thread_id.to_owned()));
        FileConnection::open_with_access(self.file(thread_id)?, false, access, self.workers.clone())
            .await
    }

    fn file(&self, thread_id: &str) -> Result<PathBuf, StorageError> {
        let id = valid_thread_id(thread_id)?;
        Ok(self.directory.join("threads").join(format!("{id}.sqlite")))
    }

    /// The registry row of a live thread; an expired one is deleted on the way.
    async fn live(&self, thread_id: &str) -> Result<ThreadRecord, StorageError> {
        valid_thread_id(thread_id)?;
        let rows = self.registry.query(SelectRegistryThread { thread_id }.bind()).await?;
        let row = rows.first().ok_or_else(|| StorageError::NotFound(thread_id.into()))?;
        let (record, expires_at) = thread_record(row)?;
        if expires_at != 0 && expires_at <= self.now_ms() {
            self.delete_thread(thread_id).await?;
            return Err(StorageError::NotFound(thread_id.into()));
        }
        Ok(record)
    }
}

fn timestamp(value: i64) -> Option<pbjson_types::Timestamp> {
    (value != 0).then_some(pbjson_types::Timestamp {
        seconds: value / 1000,
        nanos: ((value % 1000) * 1_000_000) as i32,
    })
}

/// A registry row as the threads service's record, and its expiry (`0` never).
fn thread_record(row: &Row) -> Result<(ThreadRecord, i64), StorageError> {
    let cells = Cells::new(
        row,
        &[
            "thread_id",
            "scope",
            "attributes",
            "created_at",
            "last_active",
            "idle_ttl",
            "expires_at",
        ],
    )?;
    let attributes: HashMap<String, String> = serde_json::from_value(cells.get(2)?)
        .map_err(|e| StorageError::Invalid(format!("thread attributes: {e}")))?;
    let expires_at = cells.get(6)?;
    let record = ThreadRecord {
        thread_id: cells.get(0)?,
        scope: cells.get(1)?,
        attributes,
        created_at: timestamp(cells.get(3)?),
        last_active: timestamp(cells.get(4)?),
        expires_at: timestamp(expires_at),
        idle_ttl: Some(pbjson_types::Duration { seconds: cells.get::<i64>(5)? / 1000, nanos: 0 }),
    };
    Ok((record, expires_at))
}

fn ttl_millis(d: Option<&pbjson_types::Duration>) -> i64 {
    d.map_or(0, |d| (d.seconds + i64::from(d.nanos > 0)).saturating_mul(1000))
}

fn expiry(last_active_ms: i64, idle_ttl: i64) -> i64 {
    if idle_ttl == 0 { 0 } else { last_active_ms.saturating_add(idle_ttl) }
}

#[async_trait]
impl Storage for SqliteStorage {
    /// The database file with its schema first, then the registry row: a crash between the two
    /// leaves a file no lookup finds.
    async fn create_thread(&self, request: CreateRequest) -> Result<ThreadRecord, StorageError> {
        let thread_id = shortuuid("AT_");
        let workers = self.workers.clone();
        let connection = FileConnection::open(self.file(&thread_id)?, true, workers).await?;
        let now = self.now_ms();
        let idle_ttl = ttl_millis(request.idle_ttl.as_ref());
        let attributes: HashMap<&String, &String> =
            request.attributes.iter().filter(|(_, v)| !v.is_empty()).collect();
        let insert = InsertRegistryThread {
            thread_id: &thread_id,
            scope: &request.scope,
            attributes: SqlValue::json(&attributes),
            now,
            idle_ttl,
            expires_at: expiry(now, idle_ttl),
        }
        .bind();
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

    async fn list_threads(&self, request: ListRequest) -> Result<ListResponse, StorageError> {
        let limit = thread_page_limit(request.limit);
        let scope = request.scope.as_str();
        let listing = ("threads", scope);
        let after = Cursor::position(Some(&request.page_token), listing, String::new())?;
        let (now, after, limit_plus_one) = (self.now_ms(), after.as_str(), i64::from(limit) + 1);
        let select = if scope.is_empty() {
            SelectAllThreads { now, after, limit: limit_plus_one }.bind()
        } else {
            SelectScopeThreads { scope, now, after, limit: limit_plus_one }.bind()
        };
        let rows = self.registry.query(select).await?;
        let mut threads = Vec::with_capacity(rows.len());
        for row in rows.iter().take(limit as usize) {
            let (mut thread, _) = thread_record(row)?;
            (thread.last_active, thread.expires_at) = (None, None);
            threads.push(thread);
        }
        let next_page_token = match threads.last() {
            Some(last) if rows.len() > limit as usize => Cursor::encode(listing, &last.thread_id),
            _ => String::new(),
        };
        Ok(ListResponse { threads, next_page_token })
    }

    async fn update_thread(&self, request: UpdateRequest) -> Result<ThreadRecord, StorageError> {
        valid_thread_id(&request.thread_id)?;
        let update = UpdateRegistryThread {
            thread_id: &request.thread_id,
            attributes: SqlValue::json(&request.attributes),
            now: self.now_ms(),
            idle_ttl: request.idle_ttl.as_ref().map(|d| ttl_millis(Some(d))),
        }
        .bind();
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

    async fn open(&self, thread_id: &str) -> Result<Thread, StorageError> {
        let connection = self.connect(thread_id).await?;
        let background = self.workers.background.clone();
        let thread = Thread::new(
            Arc::new(connection),
            thread_id,
            self.lease,
            SchemaCell::default(),
            background,
        );
        Ok(thread.await?.with_telemetry(self.cx.clone()))
    }

    fn background(&self) -> &Background {
        &self.workers.background
    }

    async fn close(&self) {
        crate::backends::settle(&self.workers.background).await;
        self.registry.close().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each listing seeks an index: a page reads its rows, never sorts the registry, whichever
    /// scope it lists.
    #[test]
    fn registry_pages_seek_without_sorting() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute(REGISTRY_SCHEMA, []).unwrap();
        conn.execute(REGISTRY_SCOPE_INDEX, []).unwrap();
        conn.execute_batch(
            "WITH RECURSIVE numbers(n) AS (
               SELECT 1 UNION ALL SELECT n + 1 FROM numbers WHERE n < 10000
             ) INSERT INTO threads
             SELECT 'AT_' || n, 'other', '{}', n, n, 0, 0 FROM numbers",
        )
        .unwrap();
        conn.execute("INSERT INTO threads VALUES ('AT_wanted', 'wanted', '{}', 1, 1, 0, 0)", [])
            .unwrap();
        type Params<'a> = &'a [(&'a str, &'a dyn rusqlite::ToSql)];
        let cases: [(&str, Params); 2] =
            [(SelectScopeThreads::SQL, &[(":scope", &"wanted")]), (SelectAllThreads::SQL, &[])];
        for (sql, scope) in cases {
            let mut statement = conn.prepare(sql).unwrap();
            let mut params: Vec<(&str, &dyn rusqlite::ToSql)> =
                vec![(":now", &0), (":after", &""), (":limit", &10)];
            params.extend_from_slice(scope);
            let ids = statement
                .query_map(params.as_slice(), |row| row.get::<_, String>(0))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(ids.len(), if scope.is_empty() { 10 } else { 1 }, "{sql}");
            assert_eq!(statement.get_status(rusqlite::StatementStatus::Sort), 0, "{sql}");
            assert!(statement.get_status(rusqlite::StatementStatus::VmStep) < 500, "{sql}");
        }
    }

    /// Each registry read returns the columns `thread_record` reads, in its order.
    #[test]
    fn registry_reads_return_thread_record_columns() {
        let columns = "thread_id, scope, attributes, created_at, last_active, idle_ttl, expires_at";
        let one_line = |sql: &str| sql.split_whitespace().collect::<Vec<_>>().join(" ");
        for sql in [SelectRegistryThread::SQL, SelectScopeThreads::SQL, SelectAllThreads::SQL] {
            assert!(one_line(sql).starts_with(&format!("SELECT {columns} FROM")), "{sql}");
        }
        let update = one_line(UpdateRegistryThread::SQL);
        assert!(update.ends_with(&format!("RETURNING {columns}")), "{update}");
        let insert = one_line(InsertRegistryThread::SQL);
        assert!(insert.contains(&format!("({columns})")), "{insert}");
    }
}
