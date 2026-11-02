//! Storage and response-stream contracts, implemented by Cloud AgentDB and local SQLite.

pub mod agentdb;
pub mod sqlite;

use std::sync::Arc;
use std::time::Duration;

use crate::sql::{BatchResult, Row, Statement};
use crate::{StorageError, Thread};
use async_trait::async_trait;
use livekit_protocol::agent::agent_thread::{
    CreateRequest, ListRequest, ListResponse, Thread as ThreadRecord, UpdateRequest,
};

/// One thread database's execution: typed reads and atomic ordered batches. Statements and
/// results are backend-neutral; an `Err` from `batch` never claims a rollback it did not see.
#[async_trait]
pub trait ThreadDbConnection: Send + Sync {
    /// A read's rows; changes nothing.
    async fn query(&self, statement: Statement) -> Result<Vec<Row>, StorageError>;
    /// The statements in one transaction, in order, each answered; success means committed.
    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError>;
    /// Releases the connection; repeat-safe.
    async fn close(&self);
}

/// Thread lifecycle and opening a thread, the same contract on every backend.
#[async_trait]
pub trait Storage: Send + Sync {
    /// A new `AT_` thread, returned only once its metadata and database exist.
    async fn create_thread(&self, request: CreateRequest) -> Result<ThreadRecord, StorageError>;
    /// Unknown, deleted or expired threads fail [`StorageError::NotFound`].
    async fn get_thread(&self, thread_id: &str) -> Result<ThreadRecord, StorageError>;
    async fn list_threads(&self, request: ListRequest) -> Result<ListResponse, StorageError>;
    /// Merges attributes (an empty value deletes the key) and moves expiry: the activity
    /// boundary.
    async fn update_thread(&self, request: UpdateRequest) -> Result<ThreadRecord, StorageError>;
    /// Makes the thread unavailable and reclaims it; repeat-safe.
    async fn delete_thread(&self, thread_id: &str) -> Result<(), StorageError>;
    /// The shared engine's handle on an existing thread, its writes under `holder`. Opening
    /// takes no lease.
    async fn open(&self, thread_id: &str, holder: &str) -> Result<Thread, StorageError>;
    /// Where a task's response frames are kept for its subscribers.
    fn response_streams(&self) -> Arc<dyn ResponseStreams>;
    /// Releases backend resources; repeat-safe.
    async fn close(&self);
}

pub use lk_agents_abi::storage::StreamInfo;

pub const DEFAULT_TTL: Duration = Duration::from_secs(86_400);
pub const CONTENT_TYPE: &str = "application/json";
/// The beginning, as the server spells it.
pub const START: &str = "-1";

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
