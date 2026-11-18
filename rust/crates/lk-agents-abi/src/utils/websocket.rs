use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::errors::{HostError, ProviderError};
use crate::{ApiConnectOptions, ApiConnectOptionsError};

/// One websocket frame: text, or a binary payload.
#[crate::host(record(module = "websocket", extract, emit, transparent))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsFrame {
    Text(String),
    Binary(Vec<u8>),
}

/// Where the framework opens a provider WebSocket: through the job's HTTP proxy, pinging so a
/// half-open socket is noticed. A rejected handshake's request id is read from
/// `request_id_header`. `max_message_size`: the largest message (and frame) the provider may
/// send, in bytes; `None` keeps the transport's limit.
#[crate::host(record(
    module = "websocket",
    path = "livekit.agents._websocket",
    dataclass,
    extract
))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebSocketTarget {
    pub url: String,
    #[host(default_factory = "dict")]
    pub headers: BTreeMap<String, String>,
    #[host(default_factory = "dict")]
    pub params: BTreeMap<String, String>,
    #[host(default_factory = "list")]
    pub protocols: Vec<String>,
    #[host(default = None)]
    pub request_id_header: Option<String>,
    #[host(default = None)]
    pub max_message_size: Option<u64>,
}

/// A frame sent once nothing else was sent for `interval` seconds. `only_while_unused`: on a
/// multiplexer's socket, sent only while no stream holds it.
#[crate::host(record(module = "websocket", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct WebSocketKeepaliveFrame {
    pub interval: f64,
    pub frame: WsFrame,
    pub only_while_unused: bool,
}

/// One connection as the protocol asks for it, read at each connection: options changed since
/// the last one apply to the next.
#[crate::host(record(module = "websocket", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct WebSocketConnection {
    pub target: WebSocketTarget,
    pub keepalive: Option<WebSocketKeepaliveFrame>,
    /// Seconds after which the connection is replaced, at the stream's first moment it can be.
    pub max_session_duration: Option<f64>,
    /// Seconds between pings; a ping unanswered for half of it fails the connection. `None`
    /// sends none.
    pub ping_interval: Option<f64>,
    /// Sent first on the connection, before any input or control message: the provider
    /// session's setup.
    pub setup: Vec<WsFrame>,
    /// The provider must signal `ready` before anything but the setup is sent; it has the
    /// connect limit to say so, else the attempt fails with a timeout.
    pub awaits_ready: bool,
    /// The frames that ask the provider to end its session, sent when a replacement is due
    /// (`max_session_duration`) and when the input ends: the connection then takes no more input
    /// and ends, or is replaced, once the provider finished its session or closed cleanly, or the
    /// read limit passed. Empty: it ends or is replaced at once. A shared socket's own goodbye is
    /// its multiplexer's.
    pub goodbye: Vec<WsFrame>,
}

/// One input item's frames, encoded, and the seconds they play for, which real-time pacing
/// spaces them by.
#[crate::host(record(module = "websocket", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct WebSocketInput {
    pub frames: Vec<WsFrame>,
    pub pace: f64,
}

/// The protocol's evidence that all input has been processed, independent of turn finals.
#[crate::host(string)]
#[crate::host(wire_strs(msg = format!("unknown WebSocket completion {s:?}")))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum WebSocketCompletion {
    /// No terminal guarantee: settled transcripts may finish at the bounded read limit.
    #[default]
    Settled,
    /// A promised terminal message must signal `SessionFinished` before success.
    SessionFinished,
    /// The provider promises to flush results and close; keep reading until its clean close.
    PeerClosed,
}

/// What a received message tells the connection.
#[crate::host(string)]
#[crate::host(wire_strs(msg = format!("unknown WebSocket signal {s:?}")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebSocketSignal {
    /// The provider is ready for input, after the connection's setup.
    Ready,
    /// The provider finished its session: what was sent on the connection is answered. The next
    /// input goes on a new connection; after the input's end the stream ends.
    SessionFinished,
}

/// A received message, handled: what it tells the connection, and the provider's id for the
/// request or response it belongs to, when it names one.
#[crate::host(record(module = "websocket", extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebSocketRead {
    pub signal: Option<WebSocketSignal>,
    pub request_id: Option<String>,
    /// Response data arrived; control frames and unknown messages are not progress.
    pub response: bool,
}

/// Where a shared socket's received frame goes: to `context` as `message` (decoded once, by the
/// multiplexer), or to the socket itself (`None`); `ready`: the socket's provider session is ready
/// for streams; `ended_context`: the context the message ends, the provider's last word on it
/// (what a cancel waits for).
#[crate::host(record(module = "websocket", extract))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WebSocketMuxRoute {
    pub context: Option<String>,
    #[host(annotation = "object")]
    pub message: Option<crate::HostRef>,
    pub ready: bool,
    pub ended_context: Option<String>,
}

/// The frames that cancel the contexts a stream left before the provider finished them, and
/// those contexts.
#[crate::host(record(module = "websocket", extract))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WebSocketContextsCancel {
    pub contexts: Vec<String>,
    pub frames: Vec<WsFrame>,
}

/// How a multiplexer shares its sockets. `max_contexts`: streams one socket carries at once.
/// `max_sockets`: sockets open at once; a stream finding none with room waits for one within
/// the pool limit. `idle_timeout`: seconds a socket with no stream stays open. `keep_warm`: a
/// socket with room is kept open as a standby: one opens when a stream takes the last, and an
/// idle one the provider closed is opened again. `routed`: each frame is routed to its context;
/// an unrouted socket carries one stream, gets every frame, and is reused only after a stream
/// that ended cleanly. `reuses_sockets`: `false` closes a socket once its stream left it, so
/// each stream (each session, with a connection per segment) has a fresh one.
/// `reads_cancel_end`: an unrouted socket a stream left mid-answer is cancelled with the stream's
/// frames and reused once the route reads the end of every context cancelled, within
/// [`CANCELLED_ANSWER_WAIT`]; otherwise it is closed.
#[crate::host(record(module = "websocket", extract))]
#[derive(Debug, Clone, PartialEq)]
pub struct WebSocketMuxOptions {
    pub max_contexts: Option<u32>,
    pub max_sockets: Option<u32>,
    pub idle_timeout: Option<f64>,
    pub keep_warm: bool,
    pub routed: bool,
    pub reuses_sockets: bool,
    pub reads_cancel_end: bool,
}

/// How long a pooled socket a stream left mid-answer waits for the provider to end the cancelled
/// contexts before it is closed instead of reused: a cancel is answered in a round trip.
pub const CANCELLED_ANSWER_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

/// One stream's WebSocket protocol, as the executor drives one connection after another: the
/// plugin's hooks and its kind's terms, behind an object of its own. Its hooks run on the host
/// loop.
#[crate::host(proxy(host = "livekit.agents._websocket.WebSocketHost", owns))]
pub trait WebSocketHost: Send + Sync {
    /// The next connection, built from the current options.
    #[host(await, name = "_core_connect")]
    fn connect(
        &self,
        cx: &crate::telemetry::Cx,
        conn_options: ApiConnectOptions,
    ) -> crate::BoxFuture<'_, Result<WebSocketConnection, ProviderError>>;

    /// The next input item, encoded; waits while there is none. `None`: the input ended, its
    /// end item already sent.
    #[host(await, name = "_core_next_input")]
    fn next_input(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Option<WebSocketInput>, ProviderError>>;

    /// The next control message the stream sends now, ahead of its input; waits while there is
    /// none.
    #[host(await, name = "_core_next_control")]
    fn next_control(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Vec<WsFrame>, ProviderError>>;

    /// One received frame, handled; `message`: the frame as the multiplexer decoded it.
    #[host(await, name = "_core_on_message")]
    fn on_message(
        &self,
        cx: &crate::telemetry::Cx,
        frame: WsFrame,
        message: Option<crate::HostRef>,
    ) -> crate::BoxFuture<'_, Result<WebSocketRead, ProviderError>>;

    /// Seconds without input after which the stream's `input_idle` frames go, once per silence;
    /// read at each connection.
    #[host(call, name = "_core_input_idle_after")]
    fn input_idle_after(&self) -> Result<Option<f64>, HostError>;

    /// No input went out for `input_idle_after`: the frames the stream sends then, which are not
    /// input.
    #[host(await, name = "_core_input_idle")]
    fn input_idle(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Vec<WsFrame>, ProviderError>>;

    /// Seconds of silence after response data and the latest input, once per silence.
    #[host(call, name = "_core_response_idle_after")]
    fn response_idle_after(&self) -> Result<Option<f64>, HostError>;

    /// Response data went silent; no callback runs before the first response.
    #[host(await, name = "_core_response_idle")]
    fn response_idle(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Vec<WsFrame>, ProviderError>>;

    /// The frames that end the contexts a shared stream left before the provider finished them,
    /// and those contexts.
    #[host(await, name = "_core_close_context")]
    fn close_context(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<WebSocketContextsCancel, ProviderError>>;

    /// A connection opened; `request_id` is the provider's id for it, from the handshake.
    #[host(notify, name = "_core_opened")]
    fn opened(&self, request_id: Option<String>);

    /// The provider finished a session; the next connection opens for the next input.
    #[host(notify, name = "_core_session_finished")]
    fn session_finished(&self);

    /// A failed attempt the session retries, as the provider's recoverable `error` event.
    #[host(notify, name = "_core_retrying")]
    fn retrying(&self, error: ProviderError);

    /// The provider's completion contract, read when each stream connection opens.
    #[host(call, name = "_core_completion")]
    fn completion(&self) -> Result<WebSocketCompletion, HostError>;

    /// The input ended: whether the provider owes a reply to what this connection sent, so the
    /// connection waits for it.
    #[host(call, name = "_core_owes_reply")]
    fn owes_reply(&self) -> Result<bool, HostError>;

    /// The wait for that reply reached the read limit with the provider silent: whether nothing
    /// it was sent is left unanswered, so the stream ends cleanly instead of with a timeout.
    #[host(call, name = "_core_nothing_pending")]
    fn nothing_pending(&self) -> Result<bool, HostError>;

    /// The provider closed the connection cleanly: whether the stream asked it to end its
    /// session, so the close finishes the session instead of failing the attempt.
    #[host(call, name = "_core_expects_close")]
    fn expects_close(&self) -> Result<bool, HostError>;

    /// A replacement is due (`max_session_duration`): whether the stream has nothing in progress
    /// a new connection would cut.
    #[host(call, name = "_core_can_reconnect")]
    fn can_reconnect(&self) -> Result<bool, HostError>;
}

/// A plugin's `WebSocketMultiplexer`: one provider WebSocket carrying many streams, each a context.
#[crate::host(proxy(host = "livekit.agents._websocket.WebSocketMultiplexer", generic = 2, owns))]
pub trait WebSocketMuxHost: Send + Sync {
    #[host(await, name = "_core_connect")]
    fn connect(
        &self,
        cx: &crate::telemetry::Cx,
        conn_options: ApiConnectOptions,
    ) -> crate::BoxFuture<'_, Result<WebSocketConnection, ProviderError>>;

    /// Where a received frame goes. Raising fails the socket and every context on it.
    #[host(await, name = "_core_route")]
    fn route(
        &self,
        cx: &crate::telemetry::Cx,
        frame: WsFrame,
    ) -> crate::BoxFuture<'_, Result<WebSocketMuxRoute, ProviderError>>;

    #[host(await, name = "_core_open_context")]
    fn open_context(
        &self,
        cx: &crate::telemetry::Cx,
        context_id: String,
    ) -> crate::BoxFuture<'_, Result<Vec<WsFrame>, ProviderError>>;

    /// The frames sent on a healthy socket the multiplexer closes (idle, stale, spent or at its
    /// close), ahead of the close: the provider session's own end.
    #[host(await, name = "_core_goodbye")]
    fn goodbye(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Vec<WsFrame>, ProviderError>>;
}

/// A native `WebSocketMux`, as a session takes it.
#[crate::host(native(refused = "expected a native WebSocketMux"))]
pub trait SharedWebSockets: Any + Send + Sync {}

/// A native `WebSocketPacing`, as a session takes it.
#[crate::host(native(refused = "expected a native WebSocketPacing"))]
pub trait SharedWebSocketPacing: Any + Send + Sync {}

/// `_core.WebSocketSession`: one provider socket stream. `run()` connects through the protocol,
/// sends its input, hands it each received frame, keeps the connection alive and replaces it;
/// failures retry under `conn_options`.
#[crate::host(class(name = "WebSocketSession"))]
pub trait WebSocketSession: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// `label` names the provider in retries. With `mux`, the stream is the context
    /// `context_id` on one of its shared sockets. With `pacing`, its frames go no faster than
    /// the pacing allows.
    #[host(new)]
    fn new(
        env: &Self::Env,
        host: Arc<dyn WebSocketHost>,
        conn_options: ApiConnectOptions,
        label: String,
        #[host(kw, default = None)] mux: Option<Arc<dyn SharedWebSockets>>,
        #[host(default = None)] context_id: Option<String>,
        #[host(default = None)] pacing: Option<Arc<dyn SharedWebSocketPacing>>,
    ) -> Result<Self, ApiConnectOptionsError>;

    /// Replace the connection now, once the item being written went: close it, connect again.
    /// No retry is spent.
    #[host(call)]
    fn reconnect(&self, reason: String);

    /// The provider's session is finished, as the stream read it: the connection closes before
    /// the next input item, which goes on a new connection.
    #[host(call)]
    fn end_session(&self);

    /// Run until the input ended and the provider finished, or the failure that ended it.
    #[host(await)]
    async fn run(&self, cx: crate::telemetry::Cx) -> Result<(), ProviderError>;

    /// Stop: the socket is closed, bounded, and `run()` returns.
    #[host(await)]
    async fn aclose(&self);
}

/// `_core.WebSocketMux`: the shared sockets of a multiplexing provider, opened on demand, each
/// carrying up to `max_contexts` streams and replaced after `max_session_duration`.
#[crate::host(class(name = "WebSocketMux", native = SharedWebSockets))]
pub trait WebSocketMux: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(env: &Self::Env, host: Arc<dyn WebSocketMuxHost>, options: WebSocketMuxOptions) -> Self;

    /// Open a shared socket now, unless one has room: the first stream then finds it open.
    #[host(call)]
    fn prewarm(&self);

    /// Options changed: idle sockets close, busy ones close once their streams end.
    #[host(call)]
    fn invalidate(&self);

    /// Close every socket, bounded.
    #[host(await)]
    async fn aclose(&self);
}

/// `_core.WebSocketPacing`: how fast the streams that share it send. `min_interval`: seconds between
/// two frames. `real_time`: frames go no faster than the duration their stream gives them.
#[crate::host(class(name = "WebSocketPacing", native = SharedWebSocketPacing))]
pub trait WebSocketPacing: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        #[host(kw, default = None)] min_interval: Option<f64>,
        #[host(default = false)] real_time: bool,
    ) -> Result<Self, crate::errors::Invalid>;
}
