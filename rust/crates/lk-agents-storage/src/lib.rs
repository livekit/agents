//! `lk-agents-storage`: all database interaction of Agents 2.0, an ordinary async crate
//! (docs/internals/storage.md#rust-interface).
//!
//! * [`Storage`]: thread lifecycle and opening a thread; [`agentdb::AgentDbStorage`] (Cloud)
//!   and [`sqlite::SqliteStorage`] (a local directory) implement it.
//! * [`SqliteConnection`]: one thread database's reads and atomic batches, whichever backend.
//! * [`engine`]: the shared [`Thread`], [`Lane`] and [`Run`] handles over a connection.
//! * [`lease`]: a held lease, its renewal on the core's schedule, and its loss.
//! * [`streams`]: the response streams a task's subscribers read, on either backend.
//! * [`sql`], [`schema`], [`db`], [`run`]: the SQL vocabulary, the thread tables, each lane
//!   operation's statements and decoding, and the run's records and commit.

pub mod agentdb;
pub mod context;
pub mod db;
pub mod engine;
pub mod host;
pub mod lease;
pub mod run;
pub mod schema;
pub mod sql;
pub mod sqlite;
pub mod streams;
#[cfg(feature = "testing")]
pub mod testing;

use std::sync::Arc;

use async_trait::async_trait;
use livekit_protocol::agent::agent_thread::{
    CreateRequest, ListRequest, ListResponse, Thread as ThreadRecord, UpdateRequest,
};

pub use db::{Commit, Committed, EventWrite, Fence, TaskQuery};
pub use engine::{CommitOptions, Lane, LaneDb, Run, RunOptions, Tasks, Thread, ThreadDb};
pub use lease::Lease;
pub use run::{ItemRow, RunOutcome, TaskState};
pub use streams::{LocalStreams, QueuedStreams, ResponseStreams, StreamRead};

pub use lk_agents_abi::storage::StorageError;
pub use lk_agents_abi::storage::thread::{TaskWrite, timestamp};

use sql::{BatchResult, Row, Statement};

/// One thread database's execution: typed reads and atomic ordered batches. Statements and
/// results are backend-neutral; an `Err` from `batch` never claims a rollback it did not see.
#[async_trait]
pub trait SqliteConnection: Send + Sync {
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
    /// Unknown, deleted or expired threads fail [`StorageError::ThreadNotFound`].
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

/// `prefix` and twelve random hex digits, as the Python `utils.shortuuid`.
pub fn shortuuid(prefix: &str) -> String {
    use rand::RngCore;
    use std::fmt::Write as _;
    let mut bytes = [0u8; 6];
    rand::rng().fill_bytes(&mut bytes);
    let mut out = String::with_capacity(prefix.len() + 12);
    out.push_str(prefix);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Wall time, Unix seconds.
pub fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// Wall time, Unix milliseconds: what journal events are stamped with.
pub fn now_ms() -> i64 {
    (now_secs() * 1000.0) as i64
}
