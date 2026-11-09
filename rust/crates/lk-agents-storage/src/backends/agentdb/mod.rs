pub mod connection;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::StorageError;
use crate::sql::{BatchResult, Row, Statement};
use async_trait::async_trait;
use livekit_api::services::threads::ThreadClient;
use livekit_api::services::{
    LiveKitApi, ServerError, ServerErrorCode, ServiceError, public_api_url,
};
use livekit_protocol::agent::agent_thread::{
    CreateRequest, ListRequest, ListResponse, Thread as ThreadRecord, UpdateRequest,
};
use lk_agents_abi::storage::Unacknowledged;
use lk_agents_abi::storage::wsdb::WsdbCode;
use lk_agents_net::env::ProcessEnv;

use crate::telemetry::{Cx, StorageWrite};
use crate::thread::{LeaseTiming, RESOLVE_WAIT, Resend, SchemaCell, Thread};
use crate::{Storage, ThreadDbConnection};
pub use connection::{Conn, Options as ConnOptions, TokenMinter, WriteError, WsdbError};

// The threads service owns thread metadata; agent-db holds each thread's database, over `wsdb`.

/// A failure of a request that changed nothing: a read, or a write never sent. A lost
/// connection or a backpressured database may be retried.
impl From<WsdbError> for StorageError {
    fn from(e: WsdbError) -> Self {
        match e {
            WsdbError::Closed | WsdbError::Disconnected | WsdbError::Timeout | WsdbError::Io(_) => {
                StorageError::Unavailable(e.to_string())
            }
            WsdbError::Server { code, message } => match code {
                WsdbCode::Unavailable | WsdbCode::ResourceExhausted => {
                    StorageError::Unavailable(message)
                }
                WsdbCode::NotFound => StorageError::NotFound(message),
                WsdbCode::Unauthenticated => StorageError::Unauthorized(message),
                WsdbCode::Protocol
                | WsdbCode::Internal
                | WsdbCode::Txn
                | WsdbCode::Exec
                | WsdbCode::Query => StorageError::Invalid(format!("{}: {message}", code.as_str())),
            },
            WsdbError::Protocol(m) => StorageError::Invalid(m),
            WsdbError::TooLarge(_) => {
                StorageError::TooLarge { what: "request".into(), limit: connection::MAX_FRAME }
            }
        }
    }
}

/// A write the server answered was refused and rolled back; one sent and never answered (a
/// dropped or closed socket, a timeout, a malformed answer, a server going away) may have
/// committed: `Uncertain`. Only a sent write can be.
impl From<WriteError> for StorageError {
    fn from(e: WriteError) -> Self {
        match e {
            WriteError::NotSent(e) => e.into(),
            WriteError::Sent(e @ WsdbError::Server { code, .. })
                if code != WsdbCode::Unavailable =>
            {
                e.into()
            }
            WriteError::Sent(e) => StorageError::Uncertain(Unacknowledged::sent(e)),
        }
    }
}

/// A `livekit-api` failure as a storage error: `not_found` is `NotFound`, `unauthenticated` and
/// `permission_denied` `Unauthorized`, a lost request and the retryable codes `Unavailable`,
/// anything else `Server` with its code.
pub fn service_error(e: ServiceError) -> StorageError {
    match e {
        ServiceError::Twirp(ServerError::Request(e)) => StorageError::Unavailable(e.to_string()),
        ServiceError::Twirp(ServerError::Twirp(c)) => match c.code.as_str() {
            ServerErrorCode::NOT_FOUND => StorageError::NotFound(c.msg),
            ServerErrorCode::UNAUTHENTICATED | ServerErrorCode::PERMISSION_DENIED => {
                StorageError::Unauthorized(c.msg)
            }
            // the server's own cancellation, never this caller's: nothing to retry over
            ServerErrorCode::CANCELED
            | ServerErrorCode::UNAVAILABLE
            | ServerErrorCode::DEADLINE_EXCEEDED
            | ServerErrorCode::RESOURCE_EXHAUSTED => StorageError::Unavailable(c.msg),
            _ => StorageError::Invalid(format!("{}: {}", c.code, c.msg)),
        },
        ServiceError::Twirp(e) => StorageError::Invalid(e.to_string()),
        other => StorageError::Invalid(other.to_string()),
    }
}

/// A thread write's failure: one sent and never answered (a lost response, a deadline, a
/// server-side cancellation) may have applied, so it is `Uncertain`, never retryable.
fn service_write_error(e: ServiceError) -> StorageError {
    match e {
        ServiceError::Twirp(ServerError::Request(e)) if !e.is_connect() && !e.is_builder() => {
            StorageError::Uncertain(Unacknowledged::sent(e))
        }
        ServiceError::Twirp(ServerError::Twirp(c))
            if matches!(
                c.code.as_str(),
                ServerErrorCode::CANCELED | ServerErrorCode::DEADLINE_EXCEEDED
            ) =>
        {
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
    /// The runtime the process lends: its connections run there.
    runtime: tokio::runtime::Handle,
    /// Where its health metrics go.
    cx: Cx,
}

impl AgentDbStorage {
    /// Its connections run on the process runtime.
    pub fn new(
        opts: AgentDbOptions,
        minter: TokenMinter,
        env: &ProcessEnv,
    ) -> Result<Self, StorageError> {
        let unavailable = |what: &str, e: lk_agents_net::env::ProcessError| {
            StorageError::Unavailable(format!("{what}: {e}"))
        };
        let runtime = (env.runtime)().map_err(|e| unavailable("storage runtime", e))?;
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
        let cx = env.tracer.root();
        Ok(Self { data_plane, opts, minter, api, pool: Arc::default(), runtime, cx })
    }

    pub fn options(&self) -> &AgentDbOptions {
        &self.opts
    }

    /// The threads service: map its errors with [`service_error`].
    pub fn threads(&self) -> &ThreadClient {
        self.api.threads()
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
        Resend::new(RESOLVE_WAIT, &self.cx, StorageWrite::Thread)
            .send(|| async move {
                self.threads().update(request.clone()).await.map_err(service_write_error)
            })
            .await
    }

    /// Repeat-safe, so a lost response is sent again.
    async fn delete_thread(&self, thread_id: &str) -> Result<(), StorageError> {
        Resend::new(RESOLVE_WAIT, &self.cx, StorageWrite::Thread)
            .send(|| async move {
                match self.threads().delete(thread_id).await.map_err(service_write_error) {
                    Err(StorageError::NotFound(_)) => Ok(()),
                    out => out,
                }
            })
            .await
    }

    /// One connection to the thread's database, parked for the next open when the thread
    /// closes.
    async fn open(&self, thread_id: &str) -> Result<Thread, StorageError> {
        let database_id = self.get_thread(thread_id).await?.database_id;
        let (conn, schema) = match self.pool.take(&database_id) {
            Some(idle) => (idle.conn, idle.schema),
            None => {
                let mut opts = connection::Options::new(self.data_plane.clone(), &database_id);
                opts.reconnect = self.opts.reconnect.clone();
                let conn = Conn::dial(opts, self.minter.clone(), self.runtime.clone()).await?;
                (conn, SchemaCell::default())
            }
        };
        let pooled = Pooled {
            conn,
            schema: schema.clone(),
            pool: self.pool.clone(),
            closed: AtomicBool::new(false),
        };
        let thread = Thread::new(Arc::new(pooled), thread_id, self.opts.lease, schema).await?;
        Ok(thread.with_telemetry(self.cx.clone()))
    }

    async fn get_thread_items(
        &self,
        id: &str,
        lane: Option<&str>,
        after: Option<&str>,
        limit: i64,
    ) -> Result<lk_agents_abi::storage::ThreadItemPage, StorageError> {
        use lk_agents_abi::storage::{ThreadItem, ThreadItemPage};
        let request = livekit_protocol::agent::agent_thread::GetItemsRequest {
            thread_id: id.into(),
            agent: lane.unwrap_or_default().into(),
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
            let request = livekit_protocol::agent::agent_task::GetTasksRequest {
                thread_id: id.into(),
                status: status.map(|s| crate::http::task_status(s).as_str().to_owned()),
                page_token,
                ..Default::default()
            };
            let page = self.threads().tasks(request).await.map_err(service_error)?;
            for task in page.tasks {
                let state = serde_json::from_value::<crate::http::generated::TaskStatus>(
                    serde_json::Value::String(task.status.clone()),
                )
                .ok()
                .and_then(crate::http::task_filter)
                .ok_or_else(|| {
                    StorageError::Invalid(format!("unknown task state {}", task.status))
                })?;
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

    async fn close(&self) {
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
    async fn query(&self, statement: Statement) -> Result<Vec<Row>, StorageError> {
        self.live()?;
        Ok(self.conn.query_all(&statement).await?)
    }

    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError> {
        self.live()?;
        Ok(self.conn.batch(&statements).await?)
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

/// How long a parked connection waits for the thread's next turn before it is closed.
const IDLE_TTL: Duration = Duration::from_secs(60);
/// Parked connections per database, and in all.
const IDLE_PER_DATABASE: usize = 2;
const IDLE_TOTAL: usize = 64;

/// The connections closed threads left behind, by database id: a thread's next turn takes one
/// back, with the schema version it knew, instead of dialing and probing again.
#[derive(Default)]
struct DbPool {
    idle: Mutex<Vec<(String, Idle)>>,
}

struct Idle {
    conn: Conn,
    schema: SchemaCell,
    since: Instant,
}

impl Idle {
    fn usable(&self) -> bool {
        self.since.elapsed() < IDLE_TTL && self.conn.is_idle()
    }
}

impl DbPool {
    fn take(&self, database_id: &str) -> Option<Idle> {
        let mut idle = self.idle.lock();
        while let Some(i) = idle.iter().rposition(|(id, _)| id == database_id) {
            let (_, entry) = idle.remove(i);
            if entry.usable() {
                return Some(entry);
            }
            entry.conn.close();
        }
        None
    }

    fn put(&self, conn: Conn, schema: SchemaCell) {
        let mut idle = self.idle.lock();
        idle.retain(|(_, entry)| {
            let keep = entry.usable();
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
        idle.push((database_id, Idle { conn, schema, since: Instant::now() }));
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
