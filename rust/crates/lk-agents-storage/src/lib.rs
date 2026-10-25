//! `lk-agents-storage`: the thread storage client of Agents 2.0. Python never sees SQL.
//!
//! * [`conn`]: one authenticated WebSocket per open thread database.
//! * [`schema`]: the thread tables as records, and every statement.
//! * `sql`: typed row reads, the `statement!` macro, and the checks of both.
//! * [`db`]: [`ThreadDb`] and the lane's operations: acquire, lease, commit, inbox, reads.
//! * [`run`]: [`Thread`], [`Lane::open`], [`Run`] and [`Tasks`].
//! * [`streams`]: the answer streams over `/v0/agentstore/streams`.
//!
//! [`Storage`] is the root: the project's URL and key pair. No pyo3 here;
//! `lk-agents-py/src/storage.rs` is the Python face.

use std::sync::Arc;
use std::time::Duration;

use livekit_api::services::databases::DatabaseClient;
use livekit_api::services::threads::ThreadClient;
use livekit_api::services::{LiveKitApi, ServerError, ServiceError, public_api_url};

use conn::WsdbError;

pub mod conn;
pub mod db;
pub mod run;
pub mod schema;
pub(crate) mod sql;
pub mod streams;

pub use conn::{Conn, Options as ConnOptions, TokenMinter};
use db::DbPool;
pub use db::{LaneDb, Lease, ThreadDb};
pub use run::{Lane, Run, RunOptions, Tasks, Thread};
pub use streams::{StreamReader, Streams};

/// A storage failure. `Unavailable` is the one retryable error; a task that ends on it fails
/// with reason `storage-unavailable`.
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
    /// Retryable: a connection lost mid-request, a severed database, a lease that lapsed.
    #[error("agent-store unavailable: {0}")]
    Unavailable(String),
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

/// The data plane's errors: a drop mid-request, a severed database, a write of unknown outcome
/// (`outcome_unknown`), a request still refused after its resends (`retry`) and a closed
/// connection are `StorageUnavailable`, retryable; `not_found` at `Hello` is
/// `ThreadNotFound`; a refused token is `Unauthorized`; the rest keep agent-db's code.
impl From<WsdbError> for StorageError {
    fn from(e: WsdbError) -> Self {
        match e {
            WsdbError::Closed | WsdbError::Disconnected | WsdbError::Timeout | WsdbError::Io(_) => {
                StorageError::Unavailable(e.to_string())
            }
            WsdbError::Server { code, message } => match code.as_str() {
                "unavailable" | "resource_exhausted" | "retry" | "outcome_unknown" => {
                    StorageError::Unavailable(message)
                }
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

/// Where agentstore is and how leases behave. The data plane is `wss://<public API>/v0/agentstore/db`,
/// the streams `https://<public API>/v0/agentstore/streams`.
#[derive(Debug, Clone)]
pub struct StorageOptions {
    /// The project's public API, `https://host`.
    pub url: String,
    pub reconnect: conn::Reconnect,
    /// Lease lifetime on the database clock, rounded up to whole seconds.
    pub lease_ttl: Duration,
    /// How often an open run renews its lease; must be nonzero.
    pub renew_interval: Duration,
}

impl StorageOptions {
    /// `project_url` is the project's `LIVEKIT_URL`; its public API is derived from it.
    pub fn new(project_url: &str) -> Self {
        Self {
            url: public_api_url(project_url),
            reconnect: conn::Reconnect::default(),
            lease_ttl: db::DEFAULT_LEASE_TTL,
            renew_interval: db::DEFAULT_RENEW_INTERVAL,
        }
    }
}

/// The root a worker holds: credentials and URLs. One [`Thread`] per open thread, one
/// [`Streams`] and one `livekit-api` client for threads and databases, shared by all of them.
#[derive(Clone)]
pub struct Storage {
    opts: StorageOptions,
    data_plane: String,
    minter: TokenMinter,
    streams: Streams,
    api: Arc<LiveKitApi>,
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
            Arc::new(LiveKitApi::with_api_key(&opts.url, minter.api_key(), &minter.api_secret));
        Ok(Self { data_plane, opts, minter, streams, api, pool: Arc::default() })
    }

    pub fn options(&self) -> &StorageOptions {
        &self.opts
    }

    pub fn minter(&self) -> &TokenMinter {
        &self.minter
    }

    /// One connection to `database_id` (one per open thread, closed at run end with
    /// [`Thread::close`]), its leases and waiter marks under `holder`.
    pub async fn open(&self, database_id: &str, holder: &str) -> Result<Thread, StorageError> {
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
        .map(|db| Thread::new(db, holder))
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

pub fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// Unix seconds as a `Timestamp`, to the nearest nanosecond, so the float reads back unchanged.
pub fn timestamp(t: f64) -> pbjson_types::Timestamp {
    let seconds = t.floor();
    let nanos = ((t - seconds) * 1e9).round() as i32;
    if nanos >= 1_000_000_000 {
        pbjson_types::Timestamp { seconds: seconds as i64 + 1, nanos: 0 }
    } else {
        pbjson_types::Timestamp { seconds: seconds as i64, nanos }
    }
}

/// Unix milliseconds: what events are stamped with.
pub fn now_ms() -> i64 {
    (now_secs() * 1000.0) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_data_plane_sits_under_the_project_public_api() {
        let minter = || TokenMinter::new("devkey", "secret").unwrap();
        let storage = |url: &str| Storage::new(StorageOptions::new(url), minter());
        let staging = storage("wss://my-app.staging.livekit.cloud").unwrap();
        assert_eq!(staging.opts.url, "https://beta-api.staging.livekit.cloud");
        assert_eq!(staging.data_plane, "wss://beta-api.staging.livekit.cloud/v0/agentstore/db");
        let local = storage("ws://localhost:9000/").unwrap();
        assert_eq!(local.data_plane, "ws://localhost:9000/v0/agentstore/db");
        assert!(storage("ftp://x").is_err());
    }

    /// The float reads back unchanged, and a rounding to a whole second carries.
    #[test]
    fn a_timestamp_keeps_the_seconds_float() {
        let t = timestamp(1791760302.4035509);
        assert_eq!(t.seconds as f64 + f64::from(t.nanos) / 1e9, 1791760302.4035509);
        assert_eq!(timestamp(0.9999999999), pbjson_types::Timestamp { seconds: 1, nanos: 0 });
    }

    #[test]
    fn wsdb_errors_map_to_storage_errors() {
        assert!(matches!(StorageError::from(WsdbError::Timeout), StorageError::Unavailable(_)));
        assert!(matches!(
            StorageError::from(WsdbError::Disconnected),
            StorageError::Unavailable(_)
        ));
        let nf =
            WsdbError::Server { code: "not_found".into(), message: "database not found".into() };
        assert!(matches!(StorageError::from(nf), StorageError::ThreadNotFound(_)));
        let other = WsdbError::Server { code: "exec".into(), message: "constraint".into() };
        assert!(matches!(StorageError::from(other), StorageError::Server { .. }));
        for code in ["retry", "outcome_unknown"] {
            let e =
                StorageError::from(WsdbError::Server { code: code.into(), message: "m".into() });
            assert!(e.retryable(), "{code}");
        }
    }
}
