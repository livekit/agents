//! A stream's provider socket, driven end to end: its protocol encodes its input and reads each
//! frame; the executor owns the connects, the send, receive and keepalive loops, the rotations,
//! the bounded closes and the retries. A retry is decided by the core's request
//! machine and a close code's meaning by `ErrorKind::for_close`.

mod connection;
mod executor;
mod liveness;
mod mux;
mod pacing;

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ErrorKind, Invalid, ProviderError, ProviderSaid};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::utils::websocket::{
    self as abi, WebSocketConnection, WebSocketHost, WebSocketMuxHost, WebSocketMuxOptions, WsFrame,
};
use lk_agents_abi::{
    ApiConnectOptions, ApiConnectOptionsError, Clock, Closed, Headers, HostRef, Ts,
};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::transport::{
    TransportError, WsMessage, WsReceiver, WsSender, WsTransport, within,
};
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

pub use mux::Mux;
use pacing::Pacer;

/// How long a wait on the provider lasts when `conn_options` sets no limit for it: a drain, a
/// hold for its acknowledgement, its readiness.
const DEFAULT_WAIT: Duration = Duration::from_secs(10);

/// Opens a direct socket through the job's proxy (a test hands its fake).
pub type Transports = Arc<dyn Fn(Option<String>) -> Arc<dyn WsTransport> + Send + Sync>;

/// The real transport, through the proxy it is given.
pub fn proxied() -> Transports {
    Arc::new(|proxy| Arc::new(lk_agents_net::websocket::ProxiedTransport { proxy }))
}

/// What a connection reads: a frame (with the message a multiplexer decoded from it), or the
/// peer's close.
#[derive(Debug, Clone, PartialEq)]
enum Inbound {
    Frame { frame: WsFrame, message: Option<HostRef> },
    Closed { code: Option<u16>, reason: Option<String> },
}

fn lost(error: TransportError) -> ApiError {
    ApiError::connection(Some(format!("the connection was lost: {error}")))
}

/// Opens one direct socket for `connection`, through the process's HTTP proxy, within the
/// connect limit.
async fn open_direct(
    transports: &Transports,
    env: &ProcessEnv,
    conn_options: &ApiConnectOptions,
    connection: &WebSocketConnection,
) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), ApiError> {
    let clock = env.clock.as_ref();
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
    let transport = transports(env.http_proxy());
    let deadline = conn_options.connect_deadline(clock.now());
    within(clock, deadline, transport.connect(url.as_str(), headers)).await.map_err(|error| {
        match error {
            TransportError::Status { status, message, headers, .. } => {
                let request_id = target
                    .request_id_header
                    .as_deref()
                    .and_then(|name| headers.get(name))
                    .map(str::to_owned);
                // the provider's reason and body are its own words: detail, never the message
                let mut error = ApiError::from_http("", status, request_id);
                error.said = ProviderSaid::boxed(Some(message).filter(|m| !m.is_empty()), None);
                error
            }
            other => lk_agents_net::transport::connect_error("the provider", other),
        }
    })
}

/// A direct socket's next frame or close, calling `heard` for everything the peer sends, a pong
/// included.
async fn recv_heard(
    receiver: &mut dyn WsReceiver,
    heard: &(dyn Fn() + Sync),
) -> Result<Inbound, ApiError> {
    loop {
        let message = receiver.recv().await.map_err(lost)?;
        heard();
        return Ok(match message {
            WsMessage::Text(text) => Inbound::Frame { frame: WsFrame::Text(text), message: None },
            WsMessage::Binary(data) => {
                Inbound::Frame { frame: WsFrame::Binary(data), message: None }
            }
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

/// `_core.WebSocketSession`.
pub struct Session {
    host: Arc<dyn WebSocketHost>,
    conn_options: ApiConnectOptions,
    label: String,
    env: ProcessEnv,
    links: Links,
    /// Shared with the streams that pace their frames together.
    pacing: Option<Arc<Pacing>>,
    /// Bumped by each `reconnect()`.
    rotations: lk_agents_net::signal::Shared<u64>,
    /// `end_session()` asked to end the connection before the next input item.
    session_ended: AtomicBool,
    /// The provider's id for what the current connection carries: its handshake's, or the last
    /// one a message named. Failures carry it.
    request_id: Mutex<Option<String>>,
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
        host: Arc<dyn WebSocketHost>,
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
            pacing: None,
            rotations: lk_agents_net::signal::Shared::new(0),
            session_ended: AtomicBool::new(false),
            request_id: Mutex::new(None),
            stop: CancellationToken::new(),
            running: tokio::sync::Mutex::new(()),
        }
    }

    /// A stream as the context `context_id` on `mux`'s shared sockets.
    pub fn shared(
        host: Arc<dyn WebSocketHost>,
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

    pub fn end_session(&self) {
        self.session_ended.store(true, Ordering::SeqCst);
    }

    /// Paces the stream's frames by `pacing`.
    pub fn paced(self, pacing: Arc<Pacing>) -> Self {
        Session { pacing: Some(pacing), ..self }
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

impl abi::WebSocketSession for Session {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        host: Arc<dyn WebSocketHost>,
        conn_options: ApiConnectOptions,
        label: String,
        mux: Option<Arc<dyn abi::SharedWebSockets>>,
        context_id: Option<String>,
        pacing: Option<Arc<dyn abi::SharedWebSocketPacing>>,
    ) -> Result<Self, ApiConnectOptionsError> {
        conn_options.validate()?;
        let env = env.clone();
        let shared = mux.and_then(|mux| (mux as Arc<dyn Any + Send + Sync>).downcast::<Mux>().ok());
        let session = match shared {
            Some(mux) => {
                let context_id = context_id.unwrap_or_else(|| crate::shortuuid(""));
                Session::shared(host, conn_options, label, env, mux, context_id)
            }
            None => Session::direct(host, conn_options, label, env, proxied()),
        };
        let pacing =
            pacing.and_then(|p| (p as Arc<dyn Any + Send + Sync>).downcast::<Pacing>().ok());
        Ok(match pacing {
            Some(pacing) => session.paced(pacing),
            None => session,
        })
    }

    fn reconnect(&self, reason: String) {
        Session::reconnect(self, &reason);
    }

    fn end_session(&self) {
        Session::end_session(self);
    }

    async fn run(&self, cx: Cx) -> Result<(), ProviderError> {
        Session::run(self, &cx).await
    }

    async fn aclose(&self) {
        Session::aclose(self).await;
    }
}

impl abi::SharedWebSockets for Mux {}

impl abi::WebSocketMux for Mux {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        host: Arc<dyn WebSocketMuxHost>,
        options: WebSocketMuxOptions,
    ) -> Self {
        Mux::new(host, options, env.clone(), proxied())
    }

    fn prewarm(&self) {
        Mux::prewarm(self);
    }

    fn invalidate(&self) {
        Mux::invalidate(self);
    }

    async fn aclose(&self) {
        Mux::aclose(self).await;
    }
}

/// `_core.WebSocketPacing`: the pace the streams sharing it send at.
pub struct Pacing(Mutex<Pacer>);

impl Pacing {
    /// When a frame lasting `duration` may go, its slot now reserved.
    fn reserve(&self, now: Ts, duration: Duration) -> Ts {
        self.0.lock().reserve(now, duration)
    }
}

impl abi::SharedWebSocketPacing for Pacing {}

impl abi::WebSocketPacing for Pacing {
    type Env = ProcessEnv;

    fn new(_env: &ProcessEnv, min_interval: Option<f64>, real_time: bool) -> Result<Self, Invalid> {
        let min_interval = match min_interval {
            Some(seconds) => Some(
                Duration::try_from_secs_f64(seconds)
                    .map_err(|_| Invalid("min_interval must be a duration in seconds".into()))?,
            ),
            None => None,
        };
        Ok(Pacing(Mutex::new(Pacer::new(min_interval, real_time))))
    }
}

#[cfg(test)]
#[path = "websocket_tests.rs"]
mod tests;
