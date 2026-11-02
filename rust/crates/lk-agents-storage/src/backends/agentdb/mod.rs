//! [`AgentDbStorage`]: threads on LiveKit Cloud's agentstore. The threads service owns thread
//! metadata, agent-db holds each thread's database, reached over the `wsdb` WebSocket
//! ([`connection`]).

pub mod connection;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::StorageError;
use crate::sql::{BatchResult, Row, Statement};
use async_trait::async_trait;
use livekit_api::services::threads::ThreadClient;
use livekit_api::services::{LiveKitApi, ServerError, ServiceError, public_api_url};
use livekit_protocol::agent::agent_thread::{
    CreateRequest, ListRequest, ListResponse, Thread as ThreadRecord, UpdateRequest,
};
use lk_agents_net::env::ProcessEnv;

use crate::thread::{LeaseTiming, SchemaCell, Thread};
use crate::{Storage, ThreadDbConnection};
pub use connection::{Conn, Options as ConnOptions, TokenMinter, WsdbError};

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
                "not_found" => StorageError::NotFound(message),
                "unauthenticated" | "permission_denied" => StorageError::Unauthorized(message),
                _ => StorageError::Invalid(format!("{code}: {message}")),
            },
            WsdbError::Protocol(m) => StorageError::Unavailable(m),
            WsdbError::TooLarge(_) => {
                StorageError::TooLarge { what: "request".into(), limit: connection::MAX_FRAME }
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
            _ => StorageError::Invalid(format!("{}: {}", c.code, c.msg)),
        },
        ServiceError::Twirp(e) => StorageError::Unavailable(e.to_string()),
        other => StorageError::Invalid(other.to_string()),
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
}

impl AgentDbStorage {
    /// Its connections run on the process runtime.
    pub fn new(
        opts: AgentDbOptions,
        minter: TokenMinter,
        env: &ProcessEnv,
    ) -> Result<Self, StorageError> {
        let unavailable = |what: &str, e: String| StorageError::Unavailable(format!("{what}: {e}"));
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
        Ok(Self { data_plane, opts, minter, api, pool: Arc::default(), runtime })
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
    async fn create_thread(&self, request: CreateRequest) -> Result<ThreadRecord, StorageError> {
        self.threads().create(request).await.map_err(service_error)
    }

    async fn get_thread(&self, thread_id: &str) -> Result<ThreadRecord, StorageError> {
        self.threads().get(thread_id).await.map_err(service_error)
    }

    async fn list_threads(&self, request: ListRequest) -> Result<ListResponse, StorageError> {
        self.threads().list(request).await.map_err(service_error)
    }

    async fn update_thread(&self, request: UpdateRequest) -> Result<ThreadRecord, StorageError> {
        self.threads().update(request).await.map_err(service_error)
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<(), StorageError> {
        match self.threads().delete(thread_id).await.map_err(service_error) {
            Err(StorageError::NotFound(_)) => Ok(()),
            out => out,
        }
    }

    /// One connection to the thread's database, parked for the next open when the thread
    /// closes.
    async fn open(&self, thread_id: &str, holder: &str) -> Result<Thread, StorageError> {
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
        Thread::new(Arc::new(pooled), thread_id, self.opts.lease, schema, holder).await
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

    /// A batch whose socket dropped or timed out after it was sent, or whose answer was
    /// malformed, may have committed: `Uncertain`, never `Unavailable`.
    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError> {
        self.live()?;
        self.conn.batch(&statements).await.map_err(|e| match e {
            WsdbError::Disconnected | WsdbError::Timeout | WsdbError::Protocol(_) => {
                StorageError::Uncertain(e.to_string())
            }
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
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_data_plane_sits_under_the_project_public_api() {
        let minter = || TokenMinter::new("devkey", "secret").unwrap();
        let storage = |url: &str| {
            AgentDbStorage::new(AgentDbOptions::new(url), minter(), &ProcessEnv::current_runtime())
        };
        let staging = storage("wss://my-app.staging.livekit.cloud").unwrap();
        assert_eq!(staging.opts.url, "https://beta-api.staging.livekit.cloud");
        assert_eq!(staging.data_plane, "wss://beta-api.staging.livekit.cloud/v0/agentstore/db");
        let local = storage("ws://localhost:9000/").unwrap();
        assert_eq!(local.data_plane, "ws://localhost:9000/v0/agentstore/db");
        assert!(storage("ftp://x").is_err());
    }
}
