//! [`AgentDbStorage`]: threads on LiveKit Cloud's agentstore. The threads service owns thread
//! metadata, agent-db holds each thread's database, reached over the `wsdb` WebSocket
//! ([`conn`]); answer streams live at `/v0/agentstore/streams` ([`streams`]).

pub mod conn;
pub mod streams;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use livekit_api::services::databases::DatabaseClient;
use livekit_api::services::threads::ThreadClient;
use livekit_api::services::{LiveKitApi, ServerError, ServiceError, public_api_url};
use livekit_protocol::agent::agent_thread::{
    CreateRequest, ListRequest, ListResponse, Thread as ThreadRecord, UpdateRequest,
};
use lk_agents_storage::StorageError;
use lk_agents_storage::sql::{BatchResult, Row, Statement};

use super::engine::{LeaseTiming, SchemaCell, Thread, ThreadDb};
use super::{SqliteConnection, Storage};
pub use conn::{Conn, Options as ConnOptions, TokenMinter, WsdbError};
pub use streams::{StreamReader, Streams};

/// The data plane's errors: a drop mid-request, a severed database, a request still refused after
/// its resends (`retry`) and a closed connection are `Unavailable`, retryable (a batch's lost
/// answer is `Uncertain`, see [`Pooled`]); `not_found` at `Hello` is `ThreadNotFound`; a refused
/// token is `Unauthorized`; the rest keep agent-db's code.
impl From<WsdbError> for StorageError {
    fn from(e: WsdbError) -> Self {
        match e {
            WsdbError::Closed | WsdbError::Disconnected | WsdbError::Timeout | WsdbError::Io(_) => {
                StorageError::Unavailable(e.to_string())
            }
            WsdbError::Server { code, message } => match code.as_str() {
                "unavailable" | "resource_exhausted" | "retry" => {
                    StorageError::Unavailable(message)
                }
                "outcome_unknown" => StorageError::Uncertain(message),
                "not_found" => StorageError::ThreadNotFound(message),
                "unauthenticated" | "permission_denied" => StorageError::Unauthorized(message),
                _ => StorageError::Server { code, message },
            },
            WsdbError::Protocol(m) => StorageError::Protocol(m),
            WsdbError::TooLarge(_) => {
                StorageError::TooLarge { what: "request".into(), limit: conn::MAX_FRAME }
            }
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
            "not_found" => StorageError::NotFound(c.msg),
            "unauthenticated" | "permission_denied" => StorageError::Unauthorized(c.msg),
            "unavailable" | "deadline_exceeded" | "resource_exhausted" => {
                StorageError::Unavailable(c.msg)
            }
            _ => StorageError::Server { code: c.code, message: c.msg },
        },
        ServiceError::Twirp(e) => StorageError::Decode(e.to_string()),
        other => StorageError::Invalid(other.to_string()),
    }
}

/// A threads service failure, with a missing thread as [`StorageError::ThreadNotFound`].
pub fn thread_error(e: ServiceError) -> StorageError {
    match service_error(e) {
        StorageError::NotFound(m) => StorageError::ThreadNotFound(m),
        other => other,
    }
}

/// A non-2xx with the body read: 503 is retryable (`Unavailable`), 404 is `NotFound`.
pub(crate) async fn http_ok(resp: reqwest::Response) -> Result<reqwest::Response, StorageError> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let message = resp.text().await.unwrap_or_default();
    let message = serde_json::from_str::<serde_json::Value>(&message)
        .ok()
        .and_then(|v| {
            v.get("message").or_else(|| v.get("msg")).and_then(|m| m.as_str()).map(str::to_owned)
        })
        .unwrap_or(message);
    Err(match status.as_u16() {
        404 => StorageError::NotFound(message),
        502..=504 => StorageError::Unavailable(message),
        401 | 403 => StorageError::Unauthorized(message),
        s => StorageError::Http { status: s, message },
    })
}

/// Make `ring` the process TLS provider, once: `reqwest` and `tokio-tungstenite` use
/// `rustls-no-provider` here, as in `lk-agents-inference`, and panic without one.
pub fn install_tls_provider() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _already_installed = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Where agentstore is and how leases behave. The data plane is
/// `wss://<public API>/v0/agentstore/db`, the streams `https://<public API>/v0/agentstore/streams`.
#[derive(Debug, Clone)]
pub struct AgentDbOptions {
    /// The project's public API, `https://host`.
    pub url: String,
    pub reconnect: conn::Reconnect,
    pub lease: LeaseTiming,
}

impl AgentDbOptions {
    /// `project_url` is the project's `LIVEKIT_URL`; its public API is derived from it.
    pub fn new(project_url: &str) -> Self {
        Self {
            url: public_api_url(project_url),
            reconnect: conn::Reconnect::default(),
            lease: LeaseTiming::default(),
        }
    }
}

/// The Cloud backend a worker holds: credentials and URLs, one [`Streams`] and one `livekit-api`
/// client for threads and databases, and the connections closed threads parked.
#[derive(Clone)]
pub struct AgentDbStorage {
    opts: AgentDbOptions,
    data_plane: String,
    minter: TokenMinter,
    streams: Streams,
    api: Arc<LiveKitApi>,
    pool: Arc<DbPool>,
}

impl AgentDbStorage {
    pub fn new(opts: AgentDbOptions, minter: TokenMinter) -> Result<Self, StorageError> {
        let url = &opts.url;
        let data_plane = match (url.strip_prefix("https://"), url.strip_prefix("http://")) {
            (Some(host), _) => format!("wss://{host}/v0/agentstore/db"),
            (_, Some(host)) => format!("ws://{host}/v0/agentstore/db"),
            _ => return Err(StorageError::Invalid(format!("agentstore url {url} is not http(s)"))),
        };
        install_tls_provider();
        let http = reqwest::Client::builder()
            .use_rustls_tls()
            .build()
            .map_err(|e| StorageError::Invalid(format!("http client: {e}")))?;
        let streams = Streams::new(http, format!("{url}/v0/agentstore/streams"), minter.clone());
        let api =
            Arc::new(LiveKitApi::with_api_key(&opts.url, minter.api_key(), minter.api_secret()));
        Ok(Self { data_plane, opts, minter, streams, api, pool: Arc::default() })
    }

    pub fn options(&self) -> &AgentDbOptions {
        &self.opts
    }

    pub fn minter(&self) -> &TokenMinter {
        &self.minter
    }

    pub fn streams(&self) -> &Streams {
        &self.streams
    }

    /// The threads service: map its errors with [`thread_error`].
    pub fn threads(&self) -> &ThreadClient {
        self.api.threads()
    }

    /// The database lifecycle, for development and tests: map its errors with
    /// [`service_error`].
    pub fn databases(&self) -> &DatabaseClient {
        self.api.databases()
    }

    /// One connection to `database_id`, parked for the next open when the thread closes; the
    /// thread's id is `thread_id`.
    pub async fn open_database(
        &self,
        thread_id: &str,
        database_id: &str,
        holder: &str,
    ) -> Result<Thread, StorageError> {
        let (conn, schema) = match self.pool.take(database_id) {
            Some(idle) => (idle.conn, idle.schema),
            None => {
                let mut opts = conn::Options::new(self.data_plane.clone(), database_id);
                opts.reconnect = self.opts.reconnect.clone();
                (Conn::dial(opts, self.minter.clone()).await?, SchemaCell::default())
            }
        };
        let pooled = Pooled {
            conn,
            schema: schema.clone(),
            pool: self.pool.clone(),
            closed: AtomicBool::new(false),
        };
        let db = ThreadDb::new(Arc::new(pooled), thread_id, self.opts.lease, schema).await?;
        Ok(Thread::new(db, holder))
    }
}

#[async_trait]
impl Storage for AgentDbStorage {
    async fn create_thread(&self, request: CreateRequest) -> Result<ThreadRecord, StorageError> {
        self.threads().create(request).await.map_err(thread_error)
    }

    async fn get_thread(&self, thread_id: &str) -> Result<ThreadRecord, StorageError> {
        self.threads().get(thread_id).await.map_err(thread_error)
    }

    async fn list_threads(&self, request: ListRequest) -> Result<ListResponse, StorageError> {
        self.threads().list(request).await.map_err(thread_error)
    }

    async fn update_thread(&self, request: UpdateRequest) -> Result<ThreadRecord, StorageError> {
        self.threads().update(request).await.map_err(thread_error)
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<(), StorageError> {
        match self.threads().delete(thread_id).await.map_err(thread_error) {
            Err(StorageError::ThreadNotFound(_)) => Ok(()),
            out => out,
        }
    }

    async fn open(&self, thread_id: &str, holder: &str) -> Result<Thread, StorageError> {
        let record = self.get_thread(thread_id).await?;
        self.open_database(thread_id, &record.database_id, holder).await
    }

    async fn close(&self) {
        self.pool.clear();
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
impl SqliteConnection for Pooled {
    async fn query(&self, statement: Statement) -> Result<Vec<Row>, StorageError> {
        self.live()?;
        Ok(self.conn.query_all(&statement).await?)
    }

    /// A batch whose socket dropped or timed out after it was sent may have committed:
    /// `Uncertain`, never `Unavailable`.
    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError> {
        self.live()?;
        self.conn.batch(&statements).await.map_err(|e| match e {
            WsdbError::Disconnected | WsdbError::Timeout => StorageError::Uncertain(e.to_string()),
            e => e.into(),
        })
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
        let mut idle = self.idle.lock().expect("pool");
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
        let mut idle = self.idle.lock().expect("pool");
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

    fn clear(&self) {
        for (_, entry) in self.idle.lock().expect("pool").drain(..) {
            entry.conn.close();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_data_plane_sits_under_the_project_public_api() {
        let minter = || TokenMinter::new("devkey", "secret").unwrap();
        let storage = |url: &str| AgentDbStorage::new(AgentDbOptions::new(url), minter());
        let staging = storage("wss://my-app.staging.livekit.cloud").unwrap();
        assert_eq!(staging.opts.url, "https://beta-api.staging.livekit.cloud");
        assert_eq!(staging.data_plane, "wss://beta-api.staging.livekit.cloud/v0/agentstore/db");
        let local = storage("ws://localhost:9000/").unwrap();
        assert_eq!(local.data_plane, "ws://localhost:9000/v0/agentstore/db");
        assert!(storage("ftp://x").is_err());
    }
}
