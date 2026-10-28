//! Thread storage as the host sees it: the `Storage` class `AgentDBStorage` and `SQLiteStorage`
//! configure, the thread records it answers with, and the error every operation fails with.

use serde_json::{Map, Value};

/// A storage failure. `Unavailable` is the one retryable error; a task that ends on it fails
/// with reason `storage-unavailable`. The host raises the classes of `livekit.agents.storage`.
#[crate::host(error(with = crate::storage::storage_error_to_py))]
#[derive(Debug, Clone, thiserror::Error)]
pub enum StorageError {
    /// An unknown, deleted or expired thread, or a database agent-db does not have.
    #[error("thread not found: {0}")]
    ThreadNotFound(String),
    /// A fenced write that matched no row: the lane moved past the version the run loaded, or
    /// the lease token is no longer the lane's.
    #[error("version conflict on lane {lane}: expected version {expected}")]
    VersionConflict { lane: String, expected: i64 },
    /// A field that could not be serialized; the second string says why.
    #[error("field {0} is not serializable: {1}")]
    NotSerializable(String, String),
    #[error("{what} exceeds the limit of {limit} bytes")]
    TooLarge { what: String, limit: usize },
    /// Retryable: a connection lost mid-request, a severed database, a lease that lapsed.
    #[error("agent-store unavailable: {0}")]
    Unavailable(String),
    /// A write sent and never answered: it may or may not have committed. Resolved by the run's
    /// identity before anything is reported or retried; never proof of a rollback.
    #[error("write outcome unknown: {0}")]
    Uncertain(String),
    /// `Lane::open` with `wait: false` found the lease held.
    #[error("lease held: {0}")]
    LeaseHeld(String),
    /// A waiter that did not acquire the lane within the wait (reason `lease-timeout`).
    #[error("lease timeout: {0}")]
    LeaseTimeout(String),
    /// The `reject` policy refused the message while a turn runs; the running task, if known.
    #[error("the lane is running a turn; the message was rejected")]
    RunRejected { running_task_id: Option<String> },
    /// The host cancelled the handler mid-turn: the run ended `canceled` and nothing was written.
    #[error("the turn was cancelled: {0}")]
    Cancelled(String),
    /// A refused token on either plane.
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    /// An HTTP resource that is not there: a stream, a database on the management API.
    #[error("not found: {0}")]
    NotFound(String),
    /// A server-reported error with its stable code, from the data plane or Twirp.
    #[error("{code}: {message}")]
    Server { code: String, message: String },
    /// A non-2xx the streams API answered that the mapping above does not name.
    #[error("http {status}: {message}")]
    Http { status: u16, message: String },
    /// A frame or a row that did not decode.
    #[error("decode: {0}")]
    Decode(String),
    /// An unexpected frame.
    #[error("protocol: {0}")]
    Protocol(String),
    /// A database whose `schema_version` this client does not write.
    #[error("schema: {0}")]
    Schema(String),
    #[error("invalid argument: {0}")]
    Invalid(String),
}

impl StorageError {
    /// `true` for the errors a caller may retry: [`StorageError::Unavailable`] only.
    pub fn retryable(&self) -> bool {
        matches!(self, StorageError::Unavailable(_))
    }
}

/// The `livekit.agents.storage` class of `error`: `RunRejected` carries `running_task_id`, a
/// cancelled turn is the `asyncio.CancelledError` the host raised, an invalid argument a
/// `ValueError`, and every other failure a plain `StorageError`.
#[cfg(feature = "python")]
pub fn storage_error_to_py(error: StorageError) -> pyo3::PyErr {
    use pyo3::prelude::*;

    let message = error.to_string();
    let class = match &error {
        StorageError::ThreadNotFound(_) => "ThreadNotFound",
        StorageError::VersionConflict { .. } => "VersionConflict",
        StorageError::NotSerializable(..) => "NotSerializable",
        StorageError::TooLarge { .. } => "TooLarge",
        StorageError::Unavailable(_) => "StorageUnavailable",
        StorageError::RunRejected { .. } => "RunRejected",
        StorageError::Invalid(_) => return pyo3::exceptions::PyValueError::new_err(message),
        StorageError::Cancelled(_) => {
            return pyo3::exceptions::asyncio::CancelledError::new_err(message);
        }
        _ => "StorageError",
    };
    let err = crate::python::host_exception("livekit.agents.storage.errors", class, message);
    if let StorageError::RunRejected { running_task_id } = error {
        // an attribute of the instance, so the exception's text stays the plain sentence
        Python::attach(|py| {
            let _ = err.value(py).setattr("running_task_id", running_task_id);
        });
    }
    err
}

/// Threads in one storage namespace (docs/internals/storage.md#thread-lifecycle): a local
/// directory, or the LiveKit Cloud project of a key pair. The same contract on both backends.
#[crate::host(class(name = "Storage"))]
#[async_trait::async_trait]
pub trait Storage: Send + Sync + Sized + 'static {
    /// An open thread: the shared engine's handle, bound to a holder.
    #[host(handle(class = "Thread"))]
    type Thread: Send + Sync + 'static;
    /// The response streams a task's subscribers read.
    #[host(handle(class = "Streams"))]
    type Streams: Send + Sync + 'static;

    /// `directory`: threads in that local directory, resolved to an absolute path once.
    /// Otherwise agentstore through the public API derived from `url` (the project's
    /// `LIVEKIT_URL`), with tokens minted from the key pair. Opens nothing until first use.
    #[host(new)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        #[host(kw, default = None)] directory: Option<String>,
        #[host(default = None)] url: Option<String>,
        #[host(default = None)] api_key: Option<String>,
        #[host(default = None)] api_secret: Option<String>,
        #[host(default = None)] lease_ttl: Option<f64>,
        #[host(default = None)] renew_interval: Option<f64>,
    ) -> Result<Self, StorageError>;

    /// The absolute local directory, or the public API agentstore is reached through.
    #[host(getter)]
    fn location(&self) -> String;

    /// A new `AT_` thread, returned once its metadata and database exist. `attributes` maps
    /// strings to strings.
    #[host(await)]
    async fn create_thread(
        &self,
        #[host(kw, default = "")] scope: String,
        #[host(default = None)] attributes: Option<Map<String, Value>>,
        #[host(default = 0)] idle_ttl_seconds: i64,
    ) -> Result<ThreadInfo, StorageError>;

    /// Unknown, deleted or expired threads raise `ThreadNotFound`.
    #[host(await)]
    async fn get_thread(&self, thread_id: String) -> Result<ThreadInfo, StorageError>;

    /// The threads of one scope, newest first; `before` is the previous page's cursor.
    #[host(await)]
    async fn list_threads(
        &self,
        scope: String,
        #[host(kw, default = "")] before: String,
        #[host(default = 0)] limit: i64,
    ) -> Result<ThreadPage, StorageError>;

    /// Merges `attributes` (an empty value deletes the key) and sets the idle TTL when given:
    /// the activity boundary that moves expiry.
    #[host(await)]
    async fn update_thread(
        &self,
        thread_id: String,
        #[host(kw, default = None)] attributes: Option<Map<String, Value>>,
        #[host(default = None)] idle_ttl_seconds: Option<i64>,
    ) -> Result<ThreadInfo, StorageError>;

    /// Makes the thread unavailable and reclaims its content; repeat-safe.
    #[host(await)]
    async fn delete_thread(&self, thread_id: String) -> Result<(), StorageError>;

    /// The thread's chat items, oldest first, of one agent's lane when `agent` is given.
    #[host(await)]
    async fn get_thread_items(
        &self,
        thread_id: String,
        #[host(kw, default = "")] agent: String,
        #[host(default = "")] after: String,
        #[host(default = 0)] limit: i64,
    ) -> Result<ThreadItemPage, StorageError>;

    /// The thread's A2A tasks, of one `status` when given.
    #[host(await)]
    async fn get_thread_tasks(
        &self,
        thread_id: String,
        #[host(kw, default = None)] status: Option<String>,
    ) -> Result<Vec<ThreadTask>, StorageError>;

    /// The existing thread's handle, its writes under `holder`. Takes no lease.
    #[host(await)]
    async fn open(&self, thread_id: String, holder: String) -> Result<Self::Thread, StorageError>;

    /// The storage's response streams, shared by its threads.
    #[host(await)]
    async fn streams(&self) -> Result<Self::Streams, StorageError>;

    /// Releases backend resources once open handles settled; repeat-safe.
    #[host(await)]
    async fn aclose(&self);
}

/// A thread: times are unix seconds, `0` for never, TTLs whole seconds.
#[crate::host(record(module = "storage_class", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadInfo {
    pub thread_id: String,
    /// The subject the thread is about, the one threads are listed and granted by.
    pub scope: String,
    pub attributes: Map<String, Value>,
    pub created_at: i64,
    pub last_active: i64,
    pub expires_at: i64,
    pub idle_ttl_seconds: i64,
    /// The thread's database: `DB_..` on agent-db, the thread id locally.
    pub database_id: String,
}

/// One page of `list_threads`, newest first.
#[crate::host(record(module = "storage_class", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadPage {
    pub threads: Vec<ThreadInfo>,
    /// The next page's cursor; empty at the end.
    pub before: String,
}

/// One chat item of a thread; `item` is the `ChatContext.ChatItem` as protobuf JSON.
#[crate::host(record(module = "storage_class", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadItem {
    pub agent: String,
    /// Position in the thread's history.
    pub seq: i64,
    pub message_id: String,
    pub item: Value,
}

/// One page of `get_thread_items`, oldest first.
#[crate::host(record(module = "storage_class", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadItemPage {
    pub items: Vec<ThreadItem>,
    /// The next page's cursor; empty at the end.
    pub after: String,
}

/// One A2A task of a thread.
#[crate::host(record(module = "storage_class", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadTask {
    pub task_id: String,
    pub thread_id: String,
    /// The application's task label; empty when none.
    pub name: String,
    /// The A2A state: `submitted`, `working`, `input-required`, `completed`, ...
    pub status: String,
    pub updated_at: i64,
}
