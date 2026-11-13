//! A plugin's provider socket, driven end to end: the plugin's protocol encodes its input and
//! reads each frame; the executor owns the connects, the send, receive and keepalive loops, the
//! rotations, the bounded closes and the retries. A retry is decided by the core's request
//! machine and a close code's meaning by `ErrorKind::for_close`.

mod connection;
mod executor;
mod mux;

use std::any::Any;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ErrorKind, ProviderError};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::utils::socket::{self as abi, MuxHost, SocketConnection, SocketHost, WsFrame};
use lk_agents_abi::{ApiConnectOptions, ApiConnectOptionsError, Clock, Closed, Headers, Ts};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::transport::{
    TransportError, WsMessage, WsReceiver, WsSender, WsTransport, within,
};
use tokio_util::sync::CancellationToken;

pub use mux::Mux;

/// How long a drain waits for the provider when `conn_options` has no read limit.
const DRAIN_LIMIT: Duration = Duration::from_secs(10);

/// Opens a direct socket through the job's proxy (a test hands its fake).
pub type Transports = Arc<dyn Fn(Option<String>) -> Arc<dyn WsTransport> + Send + Sync>;

/// The real transport, through the proxy each connection names.
pub fn proxied() -> Transports {
    Arc::new(|proxy| Arc::new(lk_agents_net::websocket::ProxiedTransport { proxy }))
}

/// What a connection reads: a frame, or the peer's close.
#[derive(Debug, Clone, PartialEq)]
enum Inbound {
    Frame(WsFrame),
    Closed { code: Option<u16>, reason: Option<String> },
}

fn lost(error: TransportError) -> ApiError {
    ApiError::connection(Some(format!("the connection was lost: {error}")))
}

/// Opens one direct socket for `connection`, within the connect limit.
async fn open_direct(
    transports: &Transports,
    clock: &dyn Clock,
    conn_options: &ApiConnectOptions,
    connection: &SocketConnection,
) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), ApiError> {
    let target = &connection.target;
    let mut url = url::Url::parse(&target.url).map_err(|e| {
        ApiError::new(ErrorKind::InvalidRequest, format!("invalid socket URL: {e}"))
    })?;
    if !target.params.is_empty() {
        url.query_pairs_mut().extend_pairs(&target.params);
    }
    let mut headers: Headers =
        target.headers.iter().map(|(name, value)| (name.clone(), value.clone())).collect();
    if !target.protocols.is_empty() {
        headers.insert("Sec-WebSocket-Protocol", target.protocols.join(", "));
    }
    let transport = transports(connection.proxy.clone());
    let deadline = conn_options.connect_deadline(clock.now());
    within(clock, deadline, transport.connect(url.as_str(), headers)).await.map_err(|error| {
        match error {
            TransportError::Status { status, message, headers, .. } => {
                let request_id = target
                    .request_id_header
                    .as_deref()
                    .and_then(|name| headers.get(name))
                    .map(str::to_owned);
                ApiError::from_http(&message, status, request_id)
            }
            other => lk_agents_net::transport::connect_error("the provider", other),
        }
    })
}

async fn recv_direct(receiver: &mut dyn WsReceiver) -> Result<Inbound, ApiError> {
    loop {
        return Ok(match receiver.recv().await.map_err(lost)? {
            WsMessage::Text(text) => Inbound::Frame(WsFrame::Text(text)),
            WsMessage::Binary(data) => Inbound::Frame(WsFrame::Binary(data)),
            WsMessage::Closed => {
                Inbound::Closed { code: receiver.close_code(), reason: receiver.close_reason() }
            }
            WsMessage::Other => continue,
        });
    }
}

async fn send(sender: &dyn WsSender, frame: WsFrame) -> Result<(), ApiError> {
    match frame {
        WsFrame::Text(text) => sender.send_text(text).await,
        WsFrame::Binary(data) => sender.send_binary(data).await,
    }
    .map_err(lost)
}

/// Closes `sender` by `deadline`.
async fn close_by(clock: &dyn Clock, deadline: Ts, sender: &dyn WsSender) -> Closed {
    let closed = Closed::bounded(clock, deadline, sender.close()).await;
    if closed == Closed::TimedOut {
        log::warn!(target: lk_agents_abi::LOG_TARGET, "a provider socket did not close in time");
    }
    closed
}

/// `_core.SocketSession`.
pub struct Session {
    host: Arc<dyn SocketHost>,
    conn_options: ApiConnectOptions,
    label: String,
    env: ProcessEnv,
    links: Links,
    /// Bumped by each `reconnect()`.
    rotations: lk_agents_net::signal::Shared<u64>,
    stop: CancellationToken,
    /// Held while `run()` runs, so `aclose()` returns once it ended.
    running: tokio::sync::Mutex<()>,
}

enum Links {
    Direct(Transports),
    Shared { mux: Arc<Mux>, context_id: String },
}

impl Session {
    /// A stream with its own socket, opened through `transports`.
    pub fn direct(
        host: Arc<dyn SocketHost>,
        conn_options: ApiConnectOptions,
        label: String,
        env: ProcessEnv,
        transports: Transports,
    ) -> Self {
        Session {
            host,
            conn_options,
            label,
            env,
            links: Links::Direct(transports),
            rotations: lk_agents_net::signal::Shared::new(0),
            stop: CancellationToken::new(),
            running: tokio::sync::Mutex::new(()),
        }
    }

    /// A stream as the context `context_id` on `mux`'s shared sockets.
    pub fn shared(
        host: Arc<dyn SocketHost>,
        conn_options: ApiConnectOptions,
        label: String,
        env: ProcessEnv,
        mux: Arc<Mux>,
        context_id: String,
    ) -> Self {
        Session {
            links: Links::Shared { mux, context_id },
            ..Session::direct(host, conn_options, label, env, proxied())
        }
    }

    pub fn reconnect(&self, reason: &str) {
        log::debug!(target: lk_agents_abi::LOG_TARGET, reason; "{}: reconnecting", self.label);
        self.rotations.modify(|n| *n += 1);
    }

    pub async fn run(&self, cx: &Cx) -> Result<(), ProviderError> {
        let _running = self.running.lock().await;
        executor::run(self, cx).await
    }

    pub async fn aclose(&self) {
        self.stop.cancel();
        drop(self.running.lock().await);
    }
}

impl abi::SocketSession for Session {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        host: Arc<dyn SocketHost>,
        conn_options: ApiConnectOptions,
        label: String,
        mux: Option<Arc<dyn abi::SharedSockets>>,
        context_id: Option<String>,
    ) -> Result<Self, ApiConnectOptionsError> {
        conn_options.validate()?;
        let env = env.clone();
        let shared = mux.and_then(|mux| (mux as Arc<dyn Any + Send + Sync>).downcast::<Mux>().ok());
        Ok(match shared {
            Some(mux) => {
                let context_id = context_id.unwrap_or_else(|| crate::shortuuid(""));
                Session::shared(host, conn_options, label, env, mux, context_id)
            }
            None => Session::direct(host, conn_options, label, env, proxied()),
        })
    }

    fn reconnect(&self, reason: String) {
        Session::reconnect(self, &reason);
    }

    async fn run(&self, cx: Cx) -> Result<(), ProviderError> {
        Session::run(self, &cx).await
    }

    async fn aclose(&self) {
        Session::aclose(self).await;
    }
}

impl abi::SharedSockets for Mux {}

impl abi::SocketMux for Mux {
    type Env = ProcessEnv;

    fn new(env: &ProcessEnv, host: Arc<dyn MuxHost>, max_contexts: Option<u32>) -> Self {
        Mux::new(host, max_contexts, env.clone(), proxied())
    }

    fn invalidate(&self) {
        Mux::invalidate(self);
    }

    async fn aclose(&self) {
        Mux::aclose(self).await;
    }
}

#[cfg(test)]
#[path = "socket_tests.rs"]
mod tests;
