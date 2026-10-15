//! `lk-agents-storage`: the session storage client of the Agents 2.0 plan (agent-db/docs/
//! thread-registry-plan.md, 13.2 and 13.5). Rust owns everything that talks to a socket on the
//! worker's behalf and every byte of SQL; Python (`livekit.agents.storage`) is a thin layer over
//! the typed methods here and never sees a statement.
//!
//! * [`wsdb`]: the data-plane client, one authenticated WebSocket per open thread database:
//!   `Hello`, `Exec`, streaming `Query` paced by credits, `Batch` with one result per statement,
//!   `Begin`/`Commit`/`Rollback`, `Cancel`, and a reconnect that maps a lost transaction to
//!   [`StorageError::Unavailable`].
//! * [`schema`]: the session tables as `#[derive(Host)]` records with `#[table]`, whose pydantic
//!   twins `lk-agents-codegen` writes; the DDL is livekit/protocol's `thread_schema_v1.sql`.
//! * [`statements`]: every statement a thread database sees, each one method of
//!   [`ThreadDb`]: acquire and load, renew (and the renewal timer), the fenced commit,
//!   `run_ended`, the inbox receive transaction, the drain check, `waiter`, the orphaned task.
//! * [`run`]: the run unit of work over those statements (plan 5.2, 13.8): [`run::Thread`],
//!   [`run::Lane::open`] with the lease wait, [`run::Run::commit`] and [`run::Run::abandon`],
//!   [`run::Tasks`]; the host supplies history items and state as JSON.
//! * [`turn`]: the turn state machine over a run (plan 3.3.4, 13.8): [`turn::Turn::run`] takes one
//!   A2A message through its policy, and [`turn::TurnHost`] is the one trait the host implements
//!   to run the handler and report progress.
//! * [`streams`]: the answer streams over `/v0/agentstore/streams` (plan 6).
//! * [`threads`]: the Public API's `ThreadService` (plan 4.3), and [`agentstore`] the base URL,
//!   its error mapping and the databases client.
//! * [`token`]: the worker's data plane credentials, `agent.databaseAdmin`, minted per dial
//!   (plan 13.6).
//!
//! Every request goes to agentstore through public-api-server, under one base URL
//! ([`agentstore::agentstore_url_from_env`]). [`Storage`] is the root: that URL and the project's
//! key pair, from which a thread database is opened and the streams and threads clients are made. Host-neutral by design: no pyo3 here;
//! `lk-agents-py/src/storage.rs` is the Python face.
//!
//! Errors are [`StorageError`], the list of plan 5.2 and 13.5: `ThreadNotFound`,
//! `VersionConflict`, `NotSerializable(field)`, `TooLarge(what, limit)` and `Unavailable`
//! (retryable), plus the transport and decoding failures a client has to name.

use std::sync::Arc;
use std::time::Duration;

pub mod agentstore;
pub mod chat_item;
pub mod run;
pub mod schema;
pub mod statements;
pub mod streams;
pub mod threads;
pub mod token;
pub mod turn;
pub mod wire;
pub mod wsdb;

pub use agentstore::Databases;
pub use run::{Lane, Run, RunOptions, Tasks, Thread};
use statements::DbPool;
pub use statements::{Lease, ThreadDb};
pub use streams::{StreamReader, Streams};
pub use threads::Threads;
pub use token::TokenMinter;
pub use turn::{Turn, TurnHost, TurnOptions, TurnResult};
pub use wsdb::{Conn, Options as ConnOptions};

/// Plan 5.2 and 13.5. `Unavailable` is the one retryable error; a task that ends on it fails
/// with reason `storage-unavailable` (5.12).
#[derive(Debug, Clone, thiserror::Error)]
pub enum StorageError {
    /// An id the threads service does not know, or a database agent-db does not have.
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
    /// Retryable: a connection lost mid-request, a transaction a reconnect discarded, a severed
    /// database, a closed connection, a lease that lapsed.
    #[error("agent-store unavailable: {0}")]
    Unavailable(String),
    /// `Lane::open` with `wait: false` found the lease held.
    #[error("lease held: {0}")]
    LeaseHeld(String),
    /// A waiter that did not acquire the lane within the wait (plan 3.3.2, reason `lease-timeout`).
    #[error("lease timeout: {0}")]
    LeaseTimeout(String),
    /// The `reject` policy refused the message while a turn runs (plan 3.3.4); the running task
    /// when the journal names it. The Python `RunRejected`.
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

/// Where agentstore is: the public API's base URL, `https://host`. The data plane is
/// `wss://host/v0/agentstore/db`, the streams `https://host/v0/agentstore/streams`.
#[derive(Debug, Clone)]
pub struct StorageOptions {
    /// The public API, derived from `LIVEKIT_URL`.
    pub url: String,
    pub reconnect: wsdb::Reconnect,
    /// Lease lifetime on the database clock, rounded up to whole seconds.
    pub lease_ttl: Duration,
    /// How often an open run renews its lease; must be nonzero.
    pub renew_interval: Duration,
}

impl StorageOptions {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into().trim_end_matches('/').to_owned(),
            reconnect: wsdb::Reconnect::default(),
            lease_ttl: statements::DEFAULT_LEASE_TTL,
            renew_interval: statements::DEFAULT_RENEW_INTERVAL,
        }
    }

    /// The public API of `LIVEKIT_URL`'s project.
    pub fn from_env() -> Result<Self, StorageError> {
        Ok(Self::new(agentstore::agentstore_url_from_env()?))
    }

    fn base(&self) -> Result<url::Url, StorageError> {
        let u = url::Url::parse(&self.url)
            .map_err(|e| StorageError::Invalid(format!("agentstore url {}: {e}", self.url)))?;
        if !matches!(u.scheme(), "http" | "https") || u.host_str().is_none() {
            return Err(StorageError::Invalid(format!(
                "agentstore url {} is not http(s)",
                self.url
            )));
        }
        Ok(u)
    }

    /// The data plane, the base with `ws`/`wss` for `http`/`https`.
    pub fn data_plane(&self) -> Result<String, StorageError> {
        let u = self.base()?;
        let scheme = if u.scheme() == "https" { "wss" } else { "ws" };
        let rest = &self.url[u.scheme().len()..];
        Ok(format!("{scheme}{rest}{}", agentstore::DB_PATH))
    }

    /// The streams root.
    pub fn streams(&self) -> Result<String, StorageError> {
        self.base()?;
        Ok(format!("{}{}", self.url, agentstore::STREAMS_PATH))
    }
}

/// The root a worker holds: credentials and URLs. One [`ThreadDb`] per open thread, one
/// [`Streams`], one [`Threads`] and one [`Databases`] shared by all of them.
#[derive(Clone)]
pub struct Storage {
    opts: StorageOptions,
    data_plane: String,
    minter: TokenMinter,
    streams: Streams,
    threads: Threads,
    databases: Databases,
    /// The connections closed threads parked for their next turn.
    pool: Arc<DbPool>,
}

/// Make `ring` the process TLS provider, once: `reqwest` and `tokio-tungstenite` use
/// `rustls-no-provider` here, as in `lk-agents-inference`, and panic without one.
pub fn install_tls_provider() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _already_installed = rustls::crypto::ring::default_provider().install_default();
    });
}

impl Storage {
    pub fn new(opts: StorageOptions, minter: TokenMinter) -> Result<Self, StorageError> {
        if opts.lease_ttl.is_zero() || opts.lease_ttl.as_secs() >= i64::MAX as u64 {
            return Err(StorageError::Invalid(
                "lease_ttl must be positive and fit in signed seconds".into(),
            ));
        }
        if opts.renew_interval.is_zero() {
            return Err(StorageError::Invalid("renew_interval must be positive".into()));
        }
        install_tls_provider();
        let http = reqwest::Client::builder()
            .use_rustls_tls()
            .build()
            .map_err(|e| StorageError::Invalid(format!("http client: {e}")))?;
        let streams = Streams::new(http, opts.streams()?, minter.clone());
        let threads = Threads::new(minter.threads(&opts.url));
        let databases = Databases::new(minter.databases(&opts.url));
        Ok(Self {
            data_plane: opts.data_plane()?,
            opts,
            minter,
            streams,
            threads,
            databases,
            pool: Arc::default(),
        })
    }

    /// [`StorageOptions::from_env`], `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`.
    pub fn from_env() -> Result<Self, StorageError> {
        Self::new(StorageOptions::from_env()?, TokenMinter::from_env()?)
    }

    pub fn options(&self) -> &StorageOptions {
        &self.opts
    }

    pub fn minter(&self) -> &TokenMinter {
        &self.minter
    }

    /// Dials one connection to `database_id` and reads its schema version (plan 13.4: one
    /// connection per open thread, closed at run end with [`ThreadDb::close`]).
    pub async fn open(&self, database_id: &str) -> Result<ThreadDb, StorageError> {
        let mut opts = ConnOptions::new(self.data_plane.clone(), database_id);
        opts.reconnect = self.opts.reconnect.clone();
        let minter = self.minter.clone();
        ThreadDb::pooled(
            self.pool.clone(),
            database_id,
            self.opts.lease_ttl,
            self.opts.renew_interval,
            || async move { Ok(Conn::dial(opts, minter).await?) },
        )
        .await
    }

    pub fn streams(&self) -> &Streams {
        &self.streams
    }

    pub fn threads(&self) -> &Threads {
        &self.threads
    }

    /// The database lifecycle, for development and tests.
    pub fn databases(&self) -> &Databases {
        &self.databases
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_data_plane_and_streams_sit_under_the_public_api() {
        let o = StorageOptions::new("https://beta-api.staging.livekit.cloud/");
        assert_eq!(
            o.data_plane().unwrap(),
            "wss://beta-api.staging.livekit.cloud/v0/agentstore/db"
        );
        assert_eq!(
            o.streams().unwrap(),
            "https://beta-api.staging.livekit.cloud/v0/agentstore/streams"
        );
        let o = StorageOptions::new("http://localhost:9000");
        assert_eq!(o.data_plane().unwrap(), "ws://localhost:9000/v0/agentstore/db");
        assert!(StorageOptions::new("ws://x").data_plane().is_err());
        assert!(StorageOptions::new("ftp://x").streams().is_err());
    }

    #[test]
    fn lease_timing_defaults_and_invalid_durations() {
        let defaults = StorageOptions::new("https://beta-api.staging.livekit.cloud");
        assert_eq!(defaults.lease_ttl, statements::DEFAULT_LEASE_TTL);
        assert_eq!(defaults.renew_interval, statements::DEFAULT_RENEW_INTERVAL);
        for (ttl, interval) in [
            (Duration::ZERO, defaults.renew_interval),
            (Duration::MAX, defaults.renew_interval),
            (defaults.lease_ttl, Duration::ZERO),
        ] {
            let mut opts = defaults.clone();
            opts.lease_ttl = ttl;
            opts.renew_interval = interval;
            assert!(matches!(
                Storage::new(opts, TokenMinter::new("devkey", "secret").unwrap()),
                Err(StorageError::Invalid(_))
            ));
        }
    }

    #[test]
    fn only_unavailable_is_retryable() {
        assert!(StorageError::Unavailable("x".into()).retryable());
        assert!(!StorageError::VersionConflict { lane: "l".into(), expected: 1 }.retryable());
        assert!(!StorageError::ThreadNotFound("s".into()).retryable());
    }
}
