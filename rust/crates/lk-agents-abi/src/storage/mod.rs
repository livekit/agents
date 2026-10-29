//! Thread storage as the host sees it: the `Storage` class `AgentDBStorage` and `SQLiteStorage`
//! configure, the thread records it answers with, and the error every operation fails with.

pub mod context;
mod rows;
pub mod thread;
pub mod voice;

use serde_json::{Map, Value};

pub use rows::*;

/// A storage failure (decision 82). Every database failure is one of six: `Uncertain`,
/// `Rejected`, `LeaseLost`, `Unavailable`, `NotFound`, `VersionConflict`; the rest are the
/// caller's (`Invalid`, `NotSerializable`, `TooLarge`, `Unauthorized`) or a cancelled turn.
/// `Unavailable` is the one retryable error. The host raises the classes of
/// `livekit.agents.storage`: a cancelled turn is the `asyncio.CancelledError` the host raised, an
/// invalid argument a `ValueError`, and a failure with no class of its own a plain `StorageError`.
#[crate::host(error(kind = "livekit.agents.storage.errors.StorageError"))]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StorageError {
    /// A write sent and never answered: it may or may not have committed. Resolved by the run's
    /// identity before anything is reported or retried; never proof of a rollback.
    #[error("write outcome unknown: {0}")]
    Uncertain(String),
    /// The lane refused the run: another holder has it (`reject`, or a waiter whose wait ran
    /// out). The running task, when the journal names it.
    #[error("the lane is held by another run")]
    #[host(kind = "livekit.agents.storage.errors.RunRejected", kwargs)]
    Rejected { running_task_id: Option<String> },
    /// The run's lease was taken, released or could not be renewed before it could lapse.
    #[error("lease lost: {0}")]
    #[host(kind = "livekit.agents.storage.errors.LeaseLost")]
    LeaseLost(String),
    /// Retryable: a connection lost mid-request, a severed or unreachable database, a busy lock,
    /// an answer that did not decode.
    #[error("storage unavailable: {0}")]
    #[host(kind = "livekit.agents.storage.errors.StorageUnavailable")]
    Unavailable(String),
    /// An unknown, deleted or expired thread, or a stream, task or database that is not there.
    #[error("not found: {0}")]
    #[host(kind = "livekit.agents.storage.errors.ThreadNotFound")]
    NotFound(String),
    /// A fenced write that matched no row: the lane moved past the version the run loaded, or
    /// the lease token is no longer the lane's.
    #[error("version conflict on lane {lane}: expected version {expected}")]
    #[host(kind = "livekit.agents.storage.errors.VersionConflict")]
    VersionConflict { lane: String, expected: i64 },
    /// A field that could not be serialized; the second string says why.
    #[error("field {0} is not serializable: {1}")]
    #[host(kind = "livekit.agents.storage.errors.NotSerializable")]
    NotSerializable(String, String),
    #[error("{what} exceeds the limit of {limit} bytes")]
    #[host(kind = "livekit.agents.storage.errors.TooLarge")]
    TooLarge { what: String, limit: usize },
    /// A refused token on either plane.
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    /// The host cancelled the handler mid-turn: the run ended `canceled` and nothing was written.
    #[error("the turn was cancelled: {0}")]
    #[host(kind = "asyncio.CancelledError")]
    Cancelled(String),
    /// A request the database or this client refuses as made: a bad argument, a statement the
    /// database rejected, a schema this client does not write.
    #[error("invalid: {0}")]
    #[host(kind = "ValueError")]
    Invalid(String),
}

impl StorageError {
    /// `true` for the errors a caller may retry: [`StorageError::Unavailable`] only.
    pub fn retryable(&self) -> bool {
        matches!(self, StorageError::Unavailable(_))
    }
}

/// A native `Storage`, as an A2A door is opened over it.
#[crate::host(native(refused = "expected a native Storage"))]
pub trait StorageNamespace: std::any::Any + Send + Sync {}

/// Threads in one storage namespace (docs/internals/storage.md#thread-lifecycle): a local
/// directory, or the LiveKit Cloud project of a key pair. The same contract on both backends.
#[crate::host(class(name = "Storage", native = StorageNamespace))]
pub trait Storage: Send + Sync + Sized + 'static {
    /// An open thread: the shared engine's handle, bound to a holder.
    #[host(handle(class = "Thread"))]
    type Thread: Send + Sync + 'static;
    /// The response streams a task's subscribers read.
    #[host(handle(class = "Streams"))]
    type Streams: Send + Sync + 'static;
    #[host(class(name = "_core_ContextActivation"))]
    type ContextActivation: Send + Sync + 'static;

    #[host(class(name = "_core_VoiceStore"))]
    type VoiceStore: Send + Sync + 'static;

    /// What the hosting process lends the constructor.
    #[host(env)]
    type Env;

    #[host(await)]
    async fn _core_voice_store(
        &self,
        thread_id: String,
        lane: String,
        #[host(kw, default = None)] initial_history: Option<Vec<crate::ChatItem>>,
    ) -> Result<Self::VoiceStore, StorageError>;

    /// `directory`: threads in that local directory, resolved to an absolute path once.
    /// Otherwise agentstore through the public API derived from `url` (the project's
    /// `LIVEKIT_URL`), with tokens minted from the key pair. Opens nothing until first use.
    #[host(new)]
    fn new(
        env: &Self::Env,
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
        #[host(kw, default = None)] scope: Option<String>,
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
        #[host(kw, default = None)] before: Option<String>,
        #[host(default = None)] limit: Option<u32>,
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
        #[host(kw, default = None)] agent: Option<String>,
        #[host(default = None)] after: Option<String>,
        #[host(default = None)] limit: Option<u32>,
    ) -> Result<ThreadItemPage, StorageError>;

    /// The thread's A2A tasks, of one `status` when given.
    #[host(await)]
    async fn get_thread_tasks(
        &self,
        thread_id: String,
        #[host(kw, default = None)] status: Option<TaskState>,
    ) -> Result<Vec<ThreadTask>, StorageError>;

    /// The existing thread's handle, its writes under `holder`. Takes no lease.
    #[host(await)]
    async fn open(&self, thread_id: String, holder: String) -> Result<Self::Thread, StorageError>;

    /// A runtime-supplied field store whose explicit saves commit outside a managed turn.
    #[host(await)]
    async fn _core_context_store(
        &self,
        thread_id: String,
        lane: String,
    ) -> Result<Self::ContextActivation, StorageError>;

    /// A managed turn's field store over the scopes and routing state its run loaded: saves
    /// and `transaction()` writes stage for the turn's commit.
    #[host(await)]
    async fn _core_turn_store(
        &self,
        thread_id: String,
        lane: String,
        scopes: Vec<context::ContextScope>,
        #[host(kw, default = None)] routing_state: Option<Value>,
        #[host(default = false)] read_only: bool,
    ) -> Result<Self::ContextActivation, StorageError>;

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
    /// The application's task label.
    pub name: Option<String>,
    pub status: TaskState,
    pub updated_at: i64,
}
