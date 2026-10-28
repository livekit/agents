//! Response streams: the A2A event frames a task sent, kept so a subscriber on any worker reads
//! them from an offset and follows them live until the stream closes. [`ResponseStreams`] is the
//! contract both backends meet: AgentDB's streams service, and [`LocalStreams`], a
//! `streams.sqlite` beside a local storage directory's threads.
//!
//! An append is durable when it returns: a frame is relayed only after it is kept. A stream that
//! is gone, or past its expiry, reads [`StorageError::NotFound`], distinct from a transient
//! failure.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::{Notify, mpsc, oneshot};

use crate::StorageError;
use crate::agentdb::{StreamReader, Streams};
use crate::sql::{SqlValue, Statement, StatementResult};
use crate::sqlite::FileConnection;
use crate::{SqliteConnection, now_ms, shortuuid};

pub use lk_agents_abi::storage::StreamInfo;

pub const DEFAULT_TTL: Duration = Duration::from_secs(86_400);
pub const CONTENT_TYPE: &str = "application/json";
/// The beginning, as the server spells it.
pub const START: &str = "-1";

/// How often a live reader looks for frames another process appended.
const POLL: Duration = Duration::from_millis(100);

/// The streams of one storage backend.
#[async_trait]
pub trait ResponseStreams: Send + Sync {
    /// A new open stream; `None` is the default day.
    async fn create(&self, ttl: Option<Duration>) -> Result<StreamInfo, StorageError>;
    /// Keeps one frame and returns its offset; a closed or missing stream refuses it.
    async fn append(&self, stream_id: &str, record: &[u8]) -> Result<String, StorageError>;
    /// Closes the stream: live readers end after its last frame.
    async fn close(&self, stream_id: &str) -> Result<(), StorageError>;
    async fn delete(&self, stream_id: &str) -> Result<(), StorageError>;
    /// A reader after `after` (`None`: the beginning). With `live`, it follows the tail until the
    /// stream closes; without, it ends at the tail.
    fn read(&self, stream_id: &str, after: Option<&str>, live: bool) -> Box<dyn StreamRead>;
}

/// One read position over a stream.
#[async_trait]
pub trait StreamRead: Send {
    /// The next frame, or `None` at the end.
    async fn next(&mut self) -> Result<Option<Vec<u8>>, StorageError>;
    /// The offset to resume from after the frames returned so far.
    fn next_offset(&self) -> &str;
}

const SCHEMA: [&str; 2] = [
    "CREATE TABLE IF NOT EXISTS streams (
  stream_id TEXT PRIMARY KEY,
  expires_at INTEGER NOT NULL,
  closed INTEGER NOT NULL DEFAULT 0
)",
    "CREATE TABLE IF NOT EXISTS records (
  stream_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  body BLOB NOT NULL,
  PRIMARY KEY (stream_id, seq)
)",
];

/// The streams of a local storage directory, in its `streams.sqlite`. Processes sharing the
/// directory see each other's frames; a live reader in this process wakes on an append here and
/// polls for the others'.
pub struct LocalStreams {
    conn: Arc<FileConnection>,
    appended: Arc<Notify>,
}

impl LocalStreams {
    pub async fn open(file: PathBuf) -> Result<Self, StorageError> {
        let conn = FileConnection::open(file, true).await?;
        conn.batch(SCHEMA.into_iter().map(Statement::raw).collect()).await?;
        Ok(Self { conn: Arc::new(conn), appended: Arc::new(Notify::new()) })
    }

    pub async fn shutdown(&self) {
        self.conn.close().await;
    }

    /// Whether the stream is live (`Some(closed)`), or gone or expired (`None`).
    async fn state(conn: &FileConnection, stream_id: &str) -> Result<Option<bool>, StorageError> {
        let rows = conn
            .query(statement(
                "SELECT closed FROM streams WHERE stream_id = ?1 AND expires_at > ?2",
                vec![stream_id.into(), now_ms().into()],
            ))
            .await?;
        Ok(rows.first().map(|r| r.first().and_then(SqlValue::as_i64) == Some(1)))
    }
}

fn statement(sql: &'static str, params: Vec<SqlValue>) -> Statement {
    Statement { sql: sql.into(), params }
}

fn not_found(stream_id: &str) -> StorageError {
    StorageError::NotFound(format!("stream {stream_id} does not exist or expired"))
}

#[async_trait]
impl ResponseStreams for LocalStreams {
    async fn create(&self, ttl: Option<Duration>) -> Result<StreamInfo, StorageError> {
        let stream_id = shortuuid("AST_");
        let ttl_ms = i64::try_from(ttl.unwrap_or(DEFAULT_TTL).as_millis()).unwrap_or(i64::MAX);
        let expires_at = now_ms().saturating_add(ttl_ms);
        self.conn
            .batch(vec![statement(
                "INSERT INTO streams (stream_id, expires_at) VALUES (?1, ?2)",
                vec![stream_id.as_str().into(), expires_at.into()],
            )])
            .await?;
        Ok(StreamInfo { stream_id, expires_at: expires_at / 1000 })
    }

    async fn append(&self, stream_id: &str, record: &[u8]) -> Result<String, StorageError> {
        let out = self
            .conn
            .batch(vec![statement(
                "INSERT INTO records (stream_id, seq, body)
SELECT ?1, COALESCE((SELECT MAX(seq) FROM records WHERE stream_id = ?1), 0) + 1, ?2
WHERE EXISTS (SELECT 1 FROM streams WHERE stream_id = ?1 AND closed = 0 AND expires_at > ?3)
RETURNING seq",
                vec![stream_id.into(), record.to_vec().into(), now_ms().into()],
            )])
            .await?;
        let seq = match out.results.first() {
            Some(StatementResult::Rows(rows)) => {
                rows.first().and_then(|r| r.first()).and_then(SqlValue::as_i64)
            }
            _ => return Err(StorageError::Unavailable("a stream append's answer".into())),
        };
        match seq {
            Some(seq) => {
                self.appended.notify_waiters();
                Ok(seq.to_string())
            }
            None => match Self::state(&self.conn, stream_id).await? {
                Some(true) => Err(StorageError::Invalid(format!("stream {stream_id} is closed"))),
                _ => Err(not_found(stream_id)),
            },
        }
    }

    async fn close(&self, stream_id: &str) -> Result<(), StorageError> {
        let out = self
            .conn
            .batch(vec![statement(
                "UPDATE streams SET closed = 1 WHERE stream_id = ?1 AND expires_at > ?2",
                vec![stream_id.into(), now_ms().into()],
            )])
            .await?;
        self.appended.notify_waiters();
        match out.results.first() {
            Some(StatementResult::Exec { rows_affected: 1, .. }) => Ok(()),
            _ => Err(not_found(stream_id)),
        }
    }

    async fn delete(&self, stream_id: &str) -> Result<(), StorageError> {
        self.conn
            .batch(vec![
                statement("DELETE FROM records WHERE stream_id = ?1", vec![stream_id.into()]),
                statement("DELETE FROM streams WHERE stream_id = ?1", vec![stream_id.into()]),
            ])
            .await?;
        self.appended.notify_waiters();
        Ok(())
    }

    fn read(&self, stream_id: &str, after: Option<&str>, live: bool) -> Box<dyn StreamRead> {
        Box::new(LocalReader {
            conn: self.conn.clone(),
            appended: self.appended.clone(),
            stream_id: stream_id.to_owned(),
            offset: after.unwrap_or(START).to_owned(),
            live,
            buffered: VecDeque::new(),
            ended: false,
        })
    }
}

#[async_trait]
impl ResponseStreams for Streams {
    async fn create(&self, ttl: Option<Duration>) -> Result<StreamInfo, StorageError> {
        Streams::create(self, ttl, None).await
    }

    async fn append(&self, stream_id: &str, record: &[u8]) -> Result<String, StorageError> {
        self.append_now(stream_id, record).await
    }

    async fn close(&self, stream_id: &str) -> Result<(), StorageError> {
        Streams::close(self, stream_id).await
    }

    async fn delete(&self, stream_id: &str) -> Result<(), StorageError> {
        Streams::delete(self, stream_id).await
    }

    fn read(&self, stream_id: &str, after: Option<&str>, live: bool) -> Box<dyn StreamRead> {
        Box::new(Streams::read(self, stream_id, after, live))
    }
}

#[async_trait]
impl StreamRead for StreamReader {
    async fn next(&mut self) -> Result<Option<Vec<u8>>, StorageError> {
        StreamReader::next(self).await
    }

    fn next_offset(&self) -> &str {
        StreamReader::next_offset(self)
    }
}

struct LocalReader {
    conn: Arc<FileConnection>,
    appended: Arc<Notify>,
    stream_id: String,
    offset: String,
    live: bool,
    buffered: VecDeque<(String, Vec<u8>)>,
    ended: bool,
}

impl LocalReader {
    fn after(&self) -> Result<i64, StorageError> {
        if self.offset == START {
            return Ok(0);
        }
        self.offset
            .parse()
            .map_err(|_| StorageError::Invalid(format!("stream offset {:?}", self.offset)))
    }

    /// The frames after the offset, and whether the stream was closed before they were read.
    async fn fetch(&mut self) -> Result<bool, StorageError> {
        let closed = LocalStreams::state(&self.conn, &self.stream_id)
            .await?
            .ok_or_else(|| not_found(&self.stream_id))?;
        let rows = self
            .conn
            .query(statement(
                "SELECT seq, body FROM records WHERE stream_id = ?1 AND seq > ?2 ORDER BY seq",
                vec![self.stream_id.as_str().into(), self.after()?.into()],
            ))
            .await?;
        for row in rows {
            match (row.first().and_then(SqlValue::as_i64), row.get(1)) {
                (Some(seq), Some(SqlValue::Blob(body))) => {
                    self.buffered.push_back((seq.to_string(), body.clone()))
                }
                _ => return Err(StorageError::Unavailable("a stream record".into())),
            }
        }
        Ok(closed)
    }
}

#[async_trait]
impl StreamRead for LocalReader {
    async fn next(&mut self) -> Result<Option<Vec<u8>>, StorageError> {
        loop {
            if let Some((offset, body)) = self.buffered.pop_front() {
                self.offset = offset;
                return Ok(Some(body));
            }
            if self.ended {
                return Ok(None);
            }
            // registered before the read, so an append between the read and the wait wakes it
            let appended = self.appended.clone();
            let woken = appended.notified();
            let closed = self.fetch().await?;
            if !self.buffered.is_empty() {
                continue;
            }
            if closed || !self.live {
                self.ended = true;
                continue;
            }
            let _ = tokio::time::timeout(POLL, woken).await;
        }
    }

    fn next_offset(&self) -> &str {
        &self.offset
    }
}

/// A turn's appends in order without awaiting each: one task sends them, and `close` waits for
/// the stream's queued appends before closing it. A failed append is logged, never the turn's.
pub struct QueuedStreams {
    streams: Arc<dyn ResponseStreams>,
    queue: mpsc::UnboundedSender<Queued>,
}

enum Queued {
    Append(String, Vec<u8>),
    Drained(oneshot::Sender<()>),
}

impl QueuedStreams {
    /// Spawns the sender on the current runtime.
    pub fn new(streams: Arc<dyn ResponseStreams>) -> Self {
        let (queue, mut rx) = mpsc::unbounded_channel();
        let sender = streams.clone();
        tokio::spawn(async move {
            while let Some(q) = rx.recv().await {
                match q {
                    Queued::Append(stream_id, record) => {
                        if let Err(e) = sender.append(&stream_id, &record).await {
                            log::warn!("an answer frame was not kept (stream {stream_id}): {e}");
                        }
                    }
                    Queued::Drained(done) => {
                        let _ = done.send(());
                    }
                }
            }
        });
        Self { streams, queue }
    }

    pub async fn create(&self, ttl: Option<Duration>) -> Result<StreamInfo, StorageError> {
        self.streams.create(ttl).await
    }

    pub fn append(&self, stream_id: &str, record: Vec<u8>) {
        let _ = self.queue.send(Queued::Append(stream_id.to_owned(), record));
    }

    /// Once every append queued before it was sent.
    pub async fn close(&self, stream_id: &str) -> Result<(), StorageError> {
        let (done, drained) = oneshot::channel();
        if self.queue.send(Queued::Drained(done)).is_ok() {
            let _ = drained.await;
        }
        self.streams.close(stream_id).await
    }

    pub async fn delete(&self, stream_id: &str) -> Result<(), StorageError> {
        self.streams.delete(stream_id).await
    }

    pub fn read(&self, stream_id: &str, after: Option<&str>, live: bool) -> Box<dyn StreamRead> {
        self.streams.read(stream_id, after, live)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh directory, removed when dropped.
    struct Dir(PathBuf);

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tempdir() -> Dir {
        let dir = std::env::temp_dir().join(shortuuid("lk-agents-streams-"));
        std::fs::create_dir_all(&dir).unwrap();
        Dir(dir)
    }

    async fn streams(dir: &Dir) -> LocalStreams {
        LocalStreams::open(dir.0.join("streams.sqlite")).await.unwrap()
    }

    async fn all(reader: &mut Box<dyn StreamRead>) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Some(r) = reader.next().await.unwrap() {
            out.push(r);
        }
        out
    }

    // Catches: a live subscriber missing frames appended while it waits, or never ending after
    // the close; a resumed reader repeating or skipping frames.
    #[tokio::test]
    async fn a_live_reader_follows_until_close_and_resumes_after_its_offset() {
        let dir = tempdir();
        let s = streams(&dir).await;
        let id = s.create(None).await.unwrap().stream_id;
        s.append(&id, b"a").await.unwrap();
        let mut live = s.read(&id, None, true);
        assert_eq!(live.next().await.unwrap(), Some(b"a".to_vec()));
        let resume = live.next_offset().to_owned();
        let waiting = tokio::spawn(async move { all(&mut live).await });
        tokio::task::yield_now().await;
        s.append(&id, b"b").await.unwrap();
        s.append(&id, b"c").await.unwrap();
        s.close(&id).await.unwrap();
        assert_eq!(waiting.await.unwrap(), vec![b"b".to_vec(), b"c".to_vec()]);
        assert_eq!(
            all(&mut s.read(&id, Some(&resume), false)).await,
            vec![b"b".to_vec(), b"c".to_vec()]
        );
        assert!(matches!(s.append(&id, b"d").await, Err(StorageError::Invalid(_))));
    }

    // Catches: an expired or deleted stream read as an empty one, indistinguishable from a
    // stream with nothing yet (decision 61: expiry is explicit).
    #[tokio::test]
    async fn an_expired_or_deleted_stream_is_not_found() {
        let dir = tempdir();
        let s = streams(&dir).await;
        let expired = s.create(Some(Duration::ZERO)).await.unwrap().stream_id;
        assert!(matches!(s.append(&expired, b"x").await, Err(StorageError::NotFound(_))));
        assert!(matches!(
            s.read(&expired, None, false).next().await,
            Err(StorageError::NotFound(_))
        ));
        let gone = s.create(None).await.unwrap().stream_id;
        s.delete(&gone).await.unwrap();
        assert!(matches!(s.read(&gone, None, true).next().await, Err(StorageError::NotFound(_))));
    }

    // Catches: a second process sharing the directory never seeing frames (no notification
    // crosses processes; the reader must poll).
    #[tokio::test]
    async fn a_reader_sees_frames_another_connection_appends() {
        let dir = tempdir();
        let (a, b) = (streams(&dir).await, streams(&dir).await);
        let id = a.create(None).await.unwrap().stream_id;
        let mut live = b.read(&id, None, true);
        let waiting = tokio::spawn(async move { all(&mut live).await });
        a.append(&id, b"x").await.unwrap();
        a.close(&id).await.unwrap();
        let read = tokio::time::timeout(Duration::from_secs(2), waiting).await.unwrap().unwrap();
        assert_eq!(read, vec![b"x".to_vec()]);
    }
}
