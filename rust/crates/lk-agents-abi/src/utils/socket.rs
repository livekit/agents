use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::errors::ProviderError;
use crate::{ApiConnectOptions, ApiConnectOptionsError};

/// One websocket frame: text, or a binary payload.
#[crate::host(record(module = "socket", extract, emit, transparent))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsFrame {
    Text(String),
    Binary(Vec<u8>),
}

/// Where the framework opens a provider socket: through the job's HTTP proxy, pinging so a
/// half-open socket is noticed. A rejected handshake's request id is read from
/// `request_id_header`.
#[crate::host(record(module = "socket", path = "livekit.agents.http", dataclass, extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SocketTarget {
    pub url: String,
    #[host(default_factory = "dict")]
    pub headers: BTreeMap<String, String>,
    #[host(default_factory = "dict")]
    pub params: BTreeMap<String, String>,
    #[host(default_factory = "list")]
    pub protocols: Vec<String>,
    #[host(default = None)]
    pub request_id_header: Option<String>,
}

/// A frame sent once nothing else was sent for `interval` seconds.
#[crate::host(record(module = "socket", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct KeepaliveFrame {
    pub interval: f64,
    pub frame: WsFrame,
}

/// One connection as the protocol asks for it, read at each connection: options changed since
/// the last one apply to the next.
#[crate::host(record(module = "socket", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct SocketConnection {
    pub target: SocketTarget,
    pub keepalive: Option<KeepaliveFrame>,
    /// Sent once the input ended and before a rotation; the provider then drains and closes.
    pub goodbye: Option<WsFrame>,
    /// Seconds after which the connection is rotated, at the next input boundary.
    pub max_session_duration: Option<f64>,
    /// The job's HTTP proxy, as the host's HTTP session uses it.
    pub proxy: Option<String>,
}

/// The frames of the next input item, encoded. `boundary`: it ended a segment (a flush), where a
/// due rotation may run. `end`: it was the last item.
#[crate::host(record(module = "socket", extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SocketSend {
    pub frames: Vec<WsFrame>,
    pub boundary: bool,
    pub end: bool,
}

/// A plugin's socket protocol (`http.SocketProtocol`), as the executor drives one connection
/// after another. Its hooks run on the host loop.
#[crate::host(proxy(host = "livekit.agents.http.SocketProtocol"))]
pub trait SocketHost: Send + Sync {
    /// The next connection, built from the current options.
    #[host(await, name = "_core_connect")]
    fn connect(
        &self,
        cx: &crate::telemetry::Cx,
        conn_options: ApiConnectOptions,
    ) -> crate::BoxFuture<'_, Result<SocketConnection, ProviderError>>;

    /// Takes the next input item and encodes it; waits while there is none.
    #[host(await, name = "_core_next")]
    fn next(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<SocketSend, ProviderError>>;

    /// One received frame, decoded and handled. True when the provider finished.
    #[host(await, name = "_core_on_message")]
    fn on_message(
        &self,
        cx: &crate::telemetry::Cx,
        frame: WsFrame,
    ) -> crate::BoxFuture<'_, Result<bool, ProviderError>>;

    /// A failed attempt the session retries, as the provider's recoverable `error` event.
    #[host(notify, name = "_core_retrying")]
    fn retrying(&self, error: ProviderError);
}

/// A plugin's `http.Multiplexer`: one provider socket carrying many streams, each a context.
#[crate::host(proxy(host = "livekit.agents.http.Multiplexer"))]
pub trait MuxHost: Send + Sync {
    #[host(await, name = "_core_connect")]
    fn connect(
        &self,
        cx: &crate::telemetry::Cx,
        conn_options: ApiConnectOptions,
    ) -> crate::BoxFuture<'_, Result<SocketConnection, ProviderError>>;

    /// The context a received frame belongs to; `None` for a socket-level frame. Raising fails
    /// the socket and every context on it.
    #[host(await, name = "_core_route")]
    fn route(
        &self,
        cx: &crate::telemetry::Cx,
        frame: WsFrame,
    ) -> crate::BoxFuture<'_, Result<Option<String>, ProviderError>>;

    #[host(await, name = "_core_open_context")]
    fn open_context(
        &self,
        cx: &crate::telemetry::Cx,
        context_id: String,
    ) -> crate::BoxFuture<'_, Result<Vec<WsFrame>, ProviderError>>;

    /// The frames that end a context the stream left before the provider finished it.
    #[host(await, name = "_core_close_context")]
    fn close_context(
        &self,
        cx: &crate::telemetry::Cx,
        context_id: String,
    ) -> crate::BoxFuture<'_, Result<Vec<WsFrame>, ProviderError>>;
}

/// A native `SocketMux`, as a session takes it.
#[crate::host(native(refused = "expected a native SocketMux"))]
pub trait SharedSockets: Any + Send + Sync {}

/// `_core.SocketSession`: one provider socket stream. `run()` connects through the protocol,
/// sends its input, hands it each received frame, keeps the connection alive and rotates it;
/// failures retry under `conn_options`.
#[crate::host(class(name = "SocketSession"))]
pub trait SocketSession: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// `label` names the provider in retries. With `mux`, the stream is the context
    /// `context_id` on one of its shared sockets.
    #[host(new)]
    fn new(
        env: &Self::Env,
        host: Arc<dyn SocketHost>,
        conn_options: ApiConnectOptions,
        label: String,
        #[host(kw, default = None)] mux: Option<Arc<dyn SharedSockets>>,
        #[host(default = None)] context_id: Option<String>,
    ) -> Result<Self, ApiConnectOptionsError>;

    /// Rotate the connection now: goodbye, drain, close, connect again. No retry is spent.
    #[host(call)]
    fn reconnect(&self, reason: String);

    /// Run until the input ended and the provider finished, or the failure that ended it.
    #[host(await)]
    async fn run(&self, cx: crate::telemetry::Cx) -> Result<(), ProviderError>;

    /// Stop: the socket is closed, bounded, and `run()` returns.
    #[host(await)]
    async fn aclose(&self);
}

/// `_core.SocketMux`: the shared sockets of a multiplexing provider, opened on demand, each
/// carrying up to `max_contexts` streams and replaced after `max_session_duration`.
#[crate::host(class(name = "SocketMux", native = SharedSockets))]
pub trait SocketMux: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        host: Arc<dyn MuxHost>,
        #[host(kw, default = None)] max_contexts: Option<u32>,
    ) -> Self;

    /// Options changed: idle sockets close, busy ones close once their streams end.
    #[host(call)]
    fn invalidate(&self);

    /// Close every socket, bounded.
    #[host(await)]
    async fn aclose(&self);
}
