pub mod connection;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use parking_lot::Mutex;

use crate::StorageError;
use crate::sql::{BatchResult, Row, Statement};
use async_trait::async_trait;
use livekit_api::services::threads::ThreadClient;
use livekit_api::services::{
    LiveKitApi, ServerError, ServerErrorCode, ServiceError, public_api_url,
};
use livekit_protocol::agent::agent_task::{GetTasksRequest, Status as TaskStatus};
use livekit_protocol::agent::agent_thread::{
    CreateRequest, ListRequest, ListResponse, Thread as ThreadRecord, UpdateRequest,
};
use lk_agents_abi::storage::Unacknowledged;
use lk_agents_abi::storage::agentdb_error::ErrorCode;
use lk_agents_abi::{Clock, Ts};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::owned::Background;

use crate::telemetry::{Cx, StorageWrite};
use crate::thread::{LeaseTiming, Resend, SchemaCell, Thread};
use crate::{Storage, ThreadDbConnection};
pub use connection::{Conn, Options as ConnOptions, TokenMinter, WriteError, WsdbError};

// The threads service owns thread metadata; agent-db holds each thread's database, over `wsdb`.

fn task_status(state: crate::TaskState) -> TaskStatus {
    match state {
        crate::TaskState::Submitted => TaskStatus::Submitted,
        crate::TaskState::Working => TaskStatus::Working,
        crate::TaskState::InputRequired => TaskStatus::InputRequired,
        crate::TaskState::Completed => TaskStatus::Completed,
        crate::TaskState::Failed => TaskStatus::Failed,
        crate::TaskState::Canceled => TaskStatus::Canceled,
        crate::TaskState::Rejected => TaskStatus::Rejected,
    }
}

fn task_state(status: i32) -> Result<crate::TaskState, StorageError> {
    match TaskStatus::try_from(status) {
        Ok(TaskStatus::Submitted) => Ok(crate::TaskState::Submitted),
        Ok(TaskStatus::Working) => Ok(crate::TaskState::Working),
        Ok(TaskStatus::InputRequired) => Ok(crate::TaskState::InputRequired),
        Ok(TaskStatus::Completed) => Ok(crate::TaskState::Completed),
        Ok(TaskStatus::Failed) => Ok(crate::TaskState::Failed),
        Ok(TaskStatus::Canceled) => Ok(crate::TaskState::Canceled),
        Ok(TaskStatus::Rejected) => Ok(crate::TaskState::Rejected),
        Ok(TaskStatus::Unspecified) | Err(_) => {
            Err(StorageError::Invalid(format!("unknown task state {status}")))
        }
    }
}

/// The agent-db code a management call's REST error body `code` answers, as agent-store's
/// `errcode.OfStatus` maps its gRPC status; a code that says nothing of its request's effect is
/// unknown: maybe applied, final.
fn api_code(code: &str) -> ErrorCode {
    type T = ServerErrorCode;
    match code {
        T::NOT_FOUND => ErrorCode::NOT_FOUND,
        T::PERMISSION_DENIED => ErrorCode::PERMISSION_DENIED,
        T::UNAUTHENTICATED => ErrorCode::UNAUTHENTICATED,
        T::INVALID_ARGUMENT
        | T::MALFORMED
        | T::BAD_ROUTE
        | T::FAILED_PRECONDITION
        | T::OUT_OF_RANGE
        | T::ALREADY_EXISTS
        | T::UNIMPLEMENTED => ErrorCode::INVALID_ARGUMENT,
        // a conflict with a concurrent change, or pressure that passes: nothing applied
        T::ABORTED | T::RESOURCE_EXHAUSTED => ErrorCode::RETRY,
        T::UNAVAILABLE | T::DEADLINE_EXCEEDED | T::CANCELED => ErrorCode::COMMIT_UNKNOWN_RESULT,
        T::INTERNAL | T::DATA_LOSS => ErrorCode::INTERNAL,
        _ => ErrorCode(0),
    }
}

/// A server's refusal of a request that changed nothing, by its code: the codes naming a cause of
/// their own as it, one that may succeed if sent again `Unavailable`, any other final (`too_large`
/// included: no caller branches on it, its description names the limit).
fn refused(code: ErrorCode, message: String) -> StorageError {
    match code {
        ErrorCode::NOT_FOUND => StorageError::NotFound(message),
        ErrorCode::UNAUTHENTICATED | ErrorCode::PERMISSION_DENIED => {
            StorageError::Unauthorized(message)
        }
        _ if code.retryable() => StorageError::Unavailable(format!("{code}: {message}")),
        _ => StorageError::Invalid(format!("{code}: {message}")),
    }
}

/// A failure of a request that changed nothing: a read, or a write never sent. A lost
/// connection, or a code that may succeed if sent again, may be retried.
impl From<WsdbError> for StorageError {
    fn from(e: WsdbError) -> Self {
        match e {
            WsdbError::Closed | WsdbError::Disconnected | WsdbError::Timeout | WsdbError::Io(_) => {
                StorageError::Unavailable(e.to_string())
            }
            WsdbError::Server { code, message } => refused(code, message),
            WsdbError::Protocol(m) => StorageError::Invalid(m),
            WsdbError::TooLarge(_) => {
                StorageError::TooLarge { what: "request".into(), limit: connection::MAX_FRAME }
            }
        }
    }
}

/// A sent write answered with a code that applied nothing is refused, as an unsent one is; one
/// answered with a code that may have applied, or never answered (a dropped or closed socket, a
/// timeout, a malformed answer, a server going away), may have committed: `Uncertain`. Only a
/// sent write can be.
impl From<WriteError> for StorageError {
    fn from(e: WriteError) -> Self {
        match e {
            WriteError::NotSent(e) => e.into(),
            WriteError::Sent(e @ WsdbError::Server { code, .. }) if !code.maybe_applied() => {
                e.into()
            }
            WriteError::Sent(e) => StorageError::Uncertain(Unacknowledged::sent(e)),
        }
    }
}

/// A `livekit-api` failure as a storage error: a lost request is `Unavailable`, an answered one
/// is refused by the agent-db code its REST error body's `code` answers.
pub fn service_error(e: ServiceError) -> StorageError {
    match e {
        ServiceError::Twirp(ServerError::Request(e)) => StorageError::Unavailable(e.to_string()),
        ServiceError::Twirp(ServerError::Twirp(c)) => refused(api_code(&c.code), c.msg),
        ServiceError::Twirp(e) => StorageError::Invalid(e.to_string()),
        other => StorageError::Invalid(other.to_string()),
    }
}

/// A thread write's failure: one sent and never answered, or answered with a code that may have
/// applied, is `Uncertain`, never retryable.
fn service_write_error(e: ServiceError) -> StorageError {
    match e {
        ServiceError::Twirp(ServerError::Request(e)) if !e.is_connect() && !e.is_builder() => {
            StorageError::Uncertain(Unacknowledged::sent(e))
        }
        ServiceError::Twirp(ServerError::Twirp(c)) if api_code(&c.code).maybe_applied() => {
            StorageError::Uncertain(Unacknowledged::sent(format!("{}: {}", c.code, c.msg)))
        }
        e => service_error(e),
    }
}

/// Where agentstore is and how leases behave. The data plane is
/// `wss://<public API>/v0/agentstore/db`.
#[derive(Debug, Clone)]
pub struct AgentDbOptions {
    /// The project's public API, `https://host`.
    pub url: String,
    pub reconnect: connection::Reconnect,
    pub lease: LeaseTiming,
}

impl AgentDbOptions {
    /// `project_url` is the project's `LIVEKIT_URL`; its public API is derived from it.
    pub fn new(project_url: &str) -> Self {
        Self {
            url: public_api_url(project_url),
            reconnect: connection::Reconnect::default(),
            lease: LeaseTiming::default(),
        }
    }
}

/// The Cloud backend a worker holds: credentials and URLs, one `livekit-api`
/// client for threads and databases, and the connections closed threads parked.
#[derive(Clone)]
pub struct AgentDbStorage {
    opts: AgentDbOptions,
    data_plane: String,
    minter: TokenMinter,
    api: Arc<LiveKitApi>,
    pool: Arc<DbPool>,
    /// Its own tasks, on the process's executor and clock: its connections run there.
    background: Background,
    /// Where its health metrics go.
    cx: Cx,
}

impl AgentDbStorage {
    /// Its connections run on the process's executor and clock, in a task set of its own.
    pub fn new(
        opts: AgentDbOptions,
        minter: TokenMinter,
        env: &ProcessEnv,
    ) -> Result<Self, StorageError> {
        let unavailable = |what: &str, e: lk_agents_net::env::ProcessError| {
            StorageError::Unavailable(format!("{what}: {e}"))
        };
        let spawner = (env.spawner)().map_err(|e| unavailable("storage runtime", e))?;
        let background = Background::new(Arc::default(), spawner, env.clock.clone());
        let url = &opts.url;
        let data_plane = match (url.strip_prefix("https://"), url.strip_prefix("http://")) {
            (Some(host), _) => format!("wss://{host}/v0/agentstore/db"),
            (_, Some(host)) => format!("ws://{host}/v0/agentstore/db"),
            _ => return Err(StorageError::Invalid(format!("agentstore url {url} is not http(s)"))),
        };
        let http = (env.http)().map_err(|e| unavailable("storage HTTP client", e))?;
        let api = Arc::new(LiveKitApi::with_api_key_and_client(
            &opts.url,
            minter.api_key(),
            minter.api_secret(),
            http.client(),
        ));
        let (cx, pool) = (env.tracer.root(), Arc::new(DbPool::new(env.clock.clone())));
        Ok(Self { data_plane, opts, minter, api, pool, background, cx })
    }

    pub fn options(&self) -> &AgentDbOptions {
        &self.opts
    }

    /// The threads service: map its errors with [`service_error`].
    pub fn threads(&self) -> &ThreadClient {
        self.api.threads()
    }

    /// The resends of one thread-service write.
    fn resend(&self) -> Resend {
        let clock = self.background.clock.clone();
        Resend::new(clock, &self.cx, StorageWrite::Thread)
    }
}

#[async_trait]
impl Storage for AgentDbStorage {
    /// The create carries no idempotency key yet (the threads service takes none), so a lost
    /// response is `Uncertain` and is not sent again.
    async fn create_thread(&self, request: CreateRequest) -> Result<ThreadRecord, StorageError> {
        self.threads().create(request).await.map_err(service_write_error)
    }

    async fn get_thread(&self, thread_id: &str) -> Result<ThreadRecord, StorageError> {
        self.threads().get(thread_id).await.map_err(service_error)
    }

    async fn list_threads(&self, request: ListRequest) -> Result<ListResponse, StorageError> {
        self.threads().list(request).await.map_err(service_error)
    }

    /// The same merge applied twice is one: a lost response is sent again.
    async fn update_thread(&self, request: UpdateRequest) -> Result<ThreadRecord, StorageError> {
        let request = &request;
        self.resend()
            .send(|| async move {
                self.threads().update(request.clone()).await.map_err(service_write_error)
            })
            .await
    }

    /// Repeat-safe, so a lost response is sent again.
    async fn delete_thread(&self, thread_id: &str) -> Result<(), StorageError> {
        self.resend()
            .send(|| async move {
                match self.threads().delete(thread_id).await.map_err(service_write_error) {
                    Err(StorageError::NotFound(_)) => Ok(()),
                    out => out,
                }
            })
            .await
    }

    /// One connection to the thread's database, which has the thread's id, parked for the next
    /// open when the thread closes. An unknown thread is the `Hello`'s `not_found`.
    async fn open(&self, thread_id: &str) -> Result<Thread, StorageError> {
        let (conn, schema) = match self.pool.take(thread_id) {
            Some(idle) => (idle.conn, idle.schema),
            None => {
                let mut opts = connection::Options::new(self.data_plane.clone(), thread_id);
                opts.reconnect = self.opts.reconnect.clone();
                let (spawner, clock) =
                    (self.background.spawner.clone(), self.background.clock.clone());
                let conn = Conn::dial(opts, self.minter.clone(), spawner, clock).await?;
                (conn, SchemaCell::default())
            }
        };
        let pooled = Pooled {
            conn,
            schema: schema.clone(),
            pool: self.pool.clone(),
            closed: AtomicBool::new(false),
        };
        let background = self.background.clone();
        let thread =
            Thread::new(Arc::new(pooled), thread_id, self.opts.lease, schema, background).await?;
        Ok(thread.with_telemetry(self.cx.clone()))
    }

    async fn get_thread_items(
        &self,
        id: &str,
        partition_key: Option<&str>,
        after: Option<&str>,
        limit: i64,
    ) -> Result<lk_agents_abi::storage::ThreadItemPage, StorageError> {
        use lk_agents_abi::storage::{ThreadItem, ThreadItemPage};
        let request = livekit_protocol::agent::agent_thread::GetItemsRequest {
            thread_id: id.into(),
            agent: partition_key.unwrap_or_default().into(),
            page_token: after.unwrap_or_default().into(),
            limit: limit.clamp(1, 500) as i32,
            ..Default::default()
        };
        let page = self.threads().items(request).await.map_err(service_error)?;
        let items = page
            .items
            .into_iter()
            .map(|item| {
                Ok(ThreadItem {
                    agent: item.agent,
                    seq: item.seq,
                    message_id: item.message_id,
                    item: lk_agents_abi::llm::chat_context::protocol::decode_chat_item(
                        item.item.unwrap_or_default(),
                    )?,
                })
            })
            .collect::<Result<_, StorageError>>()?;
        Ok(ThreadItemPage { items, after: page.next_page_token })
    }

    async fn get_thread_tasks(
        &self,
        id: &str,
        status: Option<crate::TaskState>,
    ) -> Result<Vec<lk_agents_abi::storage::ThreadTask>, StorageError> {
        use lk_agents_abi::storage::ThreadTask;
        let mut tasks = Vec::new();
        let mut page_token = String::new();
        loop {
            let request = GetTasksRequest {
                thread_id: id.into(),
                status: status.map(|state| task_status(state) as i32),
                page_token,
                ..Default::default()
            };
            let page = self.threads().tasks(request).await.map_err(service_error)?;
            for task in page.tasks {
                let state = task_state(task.status)?;
                tasks.push(ThreadTask {
                    task_id: task.task_id,
                    thread_id: task.thread_id,
                    name: task.name,
                    status: state,
                    updated_at: task.updated_at.map_or(0, |t| t.seconds),
                });
            }
            if page.next_page_token.is_empty() {
                return Ok(tasks);
            }
            page_token = page.next_page_token;
        }
    }

    fn background(&self) -> &Background {
        &self.background
    }

    async fn close(&self) {
        crate::backends::settle(&self.background).await;
        self.pool.clear().await;
    }
}

/// A connection on loan from the pool: closing parks it, with what is known of its schema.
struct Pooled {
    conn: Conn,
    schema: SchemaCell,
    pool: Arc<DbPool>,
    closed: AtomicBool,
}

#[async_trait]
impl ThreadDbConnection for Pooled {
    /// Waits for the socket and the answer until the caller's deadline.
    async fn query(&self, statement: Statement) -> Result<Vec<Row>, StorageError> {
        self.live()?;
        Ok(self.conn.query_all(&statement, self.conn.deadline()).await?)
    }

    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError> {
        self.live()?;
        Ok(self.conn.batch(&statements, self.conn.deadline()).await?)
    }

    async fn close(&self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            self.pool.put(self.conn.clone(), self.schema.clone());
        }
    }
}

impl Pooled {
    /// A parked connection is no longer this handle's.
    fn live(&self) -> Result<(), StorageError> {
        match self.closed.load(Ordering::Acquire) {
            true => Err(StorageError::Unavailable("the database is closed".into())),
            false => Ok(()),
        }
    }
}

/// How long a parked connection waits for the thread's next turn before it is closed, on the
/// storage's clock.
const IDLE_TTL: Duration = Duration::from_secs(60);
/// Parked connections per database, and in all.
const IDLE_PER_DATABASE: usize = 2;
const IDLE_TOTAL: usize = 64;

/// The connections closed threads left behind, by database id: a thread's next turn takes one
/// back, with the schema version it knew, instead of dialing and probing again.
struct DbPool {
    idle: Mutex<Vec<(String, Idle)>>,
    clock: Arc<dyn Clock>,
}

struct Idle {
    conn: Conn,
    schema: SchemaCell,
    since: Ts,
}

impl Idle {
    fn usable(&self, now: Ts) -> bool {
        now.saturating_duration_since(self.since) < IDLE_TTL && self.conn.is_idle()
    }
}

impl DbPool {
    fn new(clock: Arc<dyn Clock>) -> Self {
        Self { idle: Mutex::default(), clock }
    }

    fn take(&self, database_id: &str) -> Option<Idle> {
        let now = self.clock.now();
        let mut idle = self.idle.lock();
        while let Some(i) = idle.iter().rposition(|(id, _)| id == database_id) {
            let (_, entry) = idle.remove(i);
            if entry.usable(now) {
                return Some(entry);
            }
            entry.conn.close();
        }
        None
    }

    fn put(&self, conn: Conn, schema: SchemaCell) {
        let now = self.clock.now();
        let mut idle = self.idle.lock();
        idle.retain(|(_, entry)| {
            let keep = entry.usable(now);
            if !keep {
                entry.conn.close();
            }
            keep
        });
        let database_id = conn.database_id().to_owned();
        let parked = idle.iter().filter(|(id, _)| *id == database_id).count();
        if !conn.is_idle() || parked >= IDLE_PER_DATABASE || idle.len() >= IDLE_TOTAL {
            conn.close();
            return;
        }
        idle.push((database_id, Idle { conn, schema, since: now }));
    }

    async fn clear(&self) {
        let idle = std::mem::take(&mut *self.idle.lock());
        for (_, entry) in idle {
            entry.conn.aclose().await;
        }
    }
}

#[cfg(test)]
#[path = "agentdb_tests.rs"]
mod tests;
