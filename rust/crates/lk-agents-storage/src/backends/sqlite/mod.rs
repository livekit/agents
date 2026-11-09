use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use livekit_protocol::agent::agent_thread::{
    CreateRequest, ListRequest, ListResponse, Thread as ThreadRecord, UpdateRequest,
};

use crate::StorageError;
use crate::cursor::Cursor;
use crate::sql::{Bind, Cells, Row, SqlValue, Statement, statement};
use crate::thread::db;
use crate::thread::{LeaseTiming, SchemaCell, Thread};
use crate::{Storage, ThreadDbConnection, shortuuid};

mod connection;
mod lifecycle;

use connection::Request;
pub use connection::{FileConnection, Workers, batch, query, sqlite_error};

// `registry.sqlite` holds metadata and `threads/<AT_id>.sqlite` each thread; processes sharing a
// directory coordinate through database leases and fences.

/// The largest `ListThreads` page.
const MAX_THREAD_PAGE: i32 = 200;
const DEFAULT_THREAD_PAGE: i32 = 100;

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

// Version 0 used milliseconds for creation/activity, seconds for TTL/expiry. The version
// update and conversion are one transaction, including when several processes open at once.
const REGISTRY_MIGRATION: &str = "CREATE TABLE IF NOT EXISTS registry_schema (
  version INTEGER NOT NULL
)";

statement! {
    MigrateRegistryToMillis {} => r#"
UPDATE threads SET idle_ttl = idle_ttl * 1000, expires_at = expires_at * 1000
WHERE NOT EXISTS (SELECT 1 FROM registry_schema)
"#
}

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

statement! {
    /// One page of a scope's live threads, last active first, then by id, past the previous
    /// page's last row, `(before_at, before_id)`.
    SelectRegistryPage<'a> {
        scope: &'a str,
        now: i64,
        before_at: i64,
        before_id: &'a str,
        limit: i64,
    } => r#"
SELECT thread_id, scope, attributes, created_at, last_active, idle_ttl, expires_at
FROM threads WHERE scope = :scope
  AND (expires_at = 0 OR expires_at > :now)
  AND (last_active < :before_at OR (last_active = :before_at AND thread_id < :before_id))
ORDER BY last_active DESC, thread_id DESC LIMIT :limit
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
            (directory.join("threads"), directory.join("registry.sqlite"), workers.clock.clone());
        let open = move || {
            std::fs::create_dir_all(&threads)
                .map_err(|e| StorageError::Invalid(format!("{}: {e}", threads.display())))?;
            connection::SqliteThread::open(&path, true, None, &clock)
        };
        let registry = FileConnection::start(workers.clone(), Arc::new(open)).await?;
        registry
            .batch(vec![
                Statement::ddl(REGISTRY_SCHEMA),
                Statement::ddl(REGISTRY_MIGRATION),
                MigrateRegistryToMillis {}.bind(),
                SetRegistryVersion {}.bind(),
            ])
            .await?;
        let versions = registry.query(SelectRegistryVersion {}.bind()).await?;
        if versions.as_slice() != [vec![SqlValue::Int(1)]] {
            registry.close().await;
            return Err(StorageError::Invalid("unsupported registry schema version".into()));
        }
        Ok(Self { directory, registry, lease, workers })
    }

    fn now_ms(&self) -> i64 {
        (lk_agents_abi::unix_secs(self.workers.clock.wall()) * 1000.0) as i64
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
        database_id: cells.get(0)?,
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

    /// Last active first, then by id; the cursor is bound to its scope.
    async fn list_threads(&self, request: ListRequest) -> Result<ListResponse, StorageError> {
        let limit = match request.limit {
            0 => DEFAULT_THREAD_PAGE,
            n => n.clamp(1, MAX_THREAD_PAGE),
        };
        let listing = ("threads", &request.scope);
        let (before_at, before_id) =
            Cursor::position(Some(&request.page_token), listing, (i64::MAX, String::new()))?;
        let select = SelectRegistryPage {
            scope: &request.scope,
            now: self.now_ms(),
            before_at,
            before_id: &before_id,
            limit: i64::from(limit + 1),
        }
        .bind();
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
                Cursor::encode(listing, (at, &t.thread_id))
            }
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
        Thread::new(Arc::new(connection), thread_id, self.lease, SchemaCell::default()).await
    }

    async fn close(&self) {
        self.registry.close().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn registry_migration_is_versioned_and_preserves_millisecond_timestamps() {
        let directory = std::env::temp_dir().join(shortuuid("registry_migration_"));
        std::fs::create_dir_all(&directory).unwrap();
        {
            let conn = rusqlite::Connection::open(directory.join("registry.sqlite")).unwrap();
            conn.execute(REGISTRY_SCHEMA, []).unwrap();
            conn.execute(
                "INSERT INTO threads VALUES ('AT_legacy', 'scope', '{}', 42123, 44123, 2, 46)",
                [],
            )
            .unwrap();
        }
        for _ in 0..2 {
            let workers = Workers::threads(Arc::new(lk_agents_net::clock::RealClock::new()));
            let storage = SqliteStorage::open(&directory, LeaseTiming::default(), workers);
            let storage = storage.await.unwrap();
            let rows = storage
                .registry
                .query(SelectRegistryThread { thread_id: "AT_legacy" }.bind())
                .await
                .unwrap();
            let (record, expires) = thread_record(&rows[0]).unwrap();
            assert_eq!(
                record.created_at,
                Some(pbjson_types::Timestamp { seconds: 42, nanos: 123_000_000 })
            );
            assert_eq!(
                record.last_active,
                Some(pbjson_types::Timestamp { seconds: 44, nanos: 123_000_000 })
            );
            assert_eq!(expires, 46000);
            assert_eq!(record.idle_ttl.unwrap().seconds, 2);
            storage.close().await;
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
    /// Each registry read returns the columns `thread_record` reads, in its order.
    #[test]
    fn registry_reads_return_thread_record_columns() {
        let columns = "thread_id, scope, attributes, created_at, last_active, idle_ttl, expires_at";
        let one_line = |sql: &str| sql.split_whitespace().collect::<Vec<_>>().join(" ");
        for sql in [SelectRegistryThread::SQL, SelectRegistryPage::SQL] {
            assert!(one_line(sql).starts_with(&format!("SELECT {columns} FROM")), "{sql}");
        }
        let update = one_line(UpdateRegistryThread::SQL);
        assert!(update.ends_with(&format!("RETURNING {columns}")), "{update}");
        let insert = one_line(InsertRegistryThread::SQL);
        assert!(insert.contains(&format!("({columns})")), "{insert}");
    }
}
