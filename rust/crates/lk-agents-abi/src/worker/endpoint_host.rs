//! The worker endpoint's host-facing stream, its ASGI server and the ASGI messages they exchange.

use std::any::Any;
use std::sync::Arc;

use crate::errors::HostError;
use crate::worker::endpoint::EndpointRequest;
use crate::worker::ipc_host::WorkerError;

pub type Headers = Vec<(Vec<u8>, Vec<u8>)>;

/// The ASGI scope of one exchange, without `state` (the host adds `lk_request`).
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "worker_endpoint", emit))]
pub struct AsgiScope {
    /// `"http"` or `"websocket"`.
    pub kind: String,
    /// HTTP only.
    pub method: String,
    pub scheme: String,
    pub path: String,
    pub raw_path: Vec<u8>,
    pub query_string: Vec<u8>,
    pub headers: Headers,
    pub client: Option<(String, u16)>,
    /// WebSocket only.
    pub subprotocols: Vec<String>,
}

/// What the application's `receive()` returns.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "worker_endpoint", emit))]
pub enum AsgiReceived {
    Request {
        body: Vec<u8>,
        more_body: bool,
    },
    Disconnect,
    WsConnect,
    #[host(field = "text")]
    WsText(String),
    #[host(field = "data")]
    WsBytes(Vec<u8>),
    WsDisconnect {
        code: u16,
    },
}

/// What the application's `send()` takes.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "worker_endpoint", extract))]
pub enum AsgiSent {
    ResponseStart {
        status: u16,
        headers: Headers,
    },
    ResponseBody {
        body: Vec<u8>,
        more_body: bool,
    },
    ResponseTrailers,
    WsAccept {
        subprotocol: Option<String>,
        headers: Headers,
    },
    #[host(field = "text")]
    WsText(String),
    #[host(field = "data")]
    WsBytes(Vec<u8>),
    WsClose {
        code: u16,
        reason: String,
    },
    WsHttpResponseStart {
        status: u16,
        headers: Headers,
    },
    WsHttpResponseBody {
        body: Vec<u8>,
        more_body: bool,
    },
    /// Any other `type`: a `RuntimeError` for the application.
    #[host(field = "kind")]
    Unknown(String),
}

/// The application raised: `"{Type}: {message}"` and its traceback, for the worker's log.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "worker_endpoint", extract))]
pub struct AsgiFailure {
    pub summary: String,
    pub traceback: String,
}

/// An exchange after its preamble, with independent reads and writes: the native
/// `EndpointStream`, or any host object with its five methods (a test's node).
#[crate::host(proxy(host = "livekit.agents.http.ExchangeStream"))]
pub trait ExchangeStream: Send + Sync {
    /// Up to `max_bytes` of the peer's bytes; empty once the peer half-closed (FIN).
    #[host(await, name = "read")]
    fn read(&self, max_bytes: usize) -> crate::BoxFuture<'_, Result<Vec<u8>, HostError>>;
    /// All of `data`, waiting on flow control.
    #[host(await, name = "write")]
    fn write(&self, data: Vec<u8>) -> crate::BoxFuture<'_, Result<(), HostError>>;
    /// FIN: the response is complete.
    #[host(notify, name = "finish")]
    fn finish(&self);
    /// RESET_STREAM and STOP_SENDING.
    #[host(notify, name = "reset")]
    fn reset(&self, code: i32);
    /// STOP_SENDING only.
    #[host(notify, name = "stop")]
    fn stop(&self, code: i32);
}

#[crate::host(class(name = "EndpointStream", native = ExchangeStream))]
pub trait EndpointStream: Send + Sync + Sized + 'static {
    #[host(await)]
    async fn read(&self, #[host(default = 65536)] max_bytes: usize)
    -> Result<Vec<u8>, WorkerError>;

    #[host(await)]
    async fn write(&self, data: Vec<u8>) -> Result<(), WorkerError>;

    /// Idempotent; a stream already reset raises.
    #[host(call)]
    fn finish(&self) -> Result<(), WorkerError>;

    #[host(call)]
    fn reset(&self, code: i32) -> Result<(), WorkerError>;

    #[host(call)]
    fn stop(&self, code: i32) -> Result<(), WorkerError>;
}

/// The host's ASGI application, called with the scope, the request's preamble and the native
/// receive and send callables; it answers with the failure it raised, if any.
#[crate::host(proxy(host = "livekit.agents.http.AsgiApplication"))]
pub trait AsgiApplication: Send + Sync {
    #[host(await, name = "_core_call")]
    fn call(
        &self,
        scope: AsgiScope,
        request: EndpointRequest,
        receive: Arc<dyn AsgiReceiver>,
        send: Arc<dyn AsgiSender>,
    ) -> crate::BoxFuture<'_, Result<Option<AsgiFailure>, HostError>>;
}

/// One exchange's `receive()`.
#[crate::host(native(refused = "expected a native AsgiReceive"))]
pub trait AsgiReceiver: Any + Send + Sync {}

/// One exchange's `send()`.
#[crate::host(native(refused = "expected a native AsgiSend"))]
pub trait AsgiSender: Any + Send + Sync {}

/// A native `EndpointServer`, as the server runtime serves its routes with it.
#[crate::host(native(refused = "expected a native EndpointServer"))]
pub trait EndpointService: Any + Send + Sync {}

/// The host's ASGI application bridge and its active WebSockets.
#[crate::host(class(name = "EndpointServer", native = EndpointService))]
pub trait EndpointServer: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new() -> Self;

    /// Serve `stream` with `app` to completion; `request` is its preamble.
    #[host(await)]
    async fn serve(
        &self,
        stream: Arc<dyn ExchangeStream>,
        app: Arc<dyn AsgiApplication>,
        request: EndpointRequest,
    );

    /// A close frame to every open WebSocket, so clients reconnect to another worker.
    #[host(await)]
    async fn close_websockets(
        &self,
        #[host(default = 1001)] code: u16,
        #[host(default = "worker draining")] reason: String,
    );
}

#[crate::host(class(name = "_core_AsgiReceive", native = AsgiReceiver))]
pub trait AsgiReceive: Send + Sync + Sized + 'static {
    #[host(await)]
    async fn __call__(&self) -> AsgiReceived;
}

#[crate::host(class(name = "_core_AsgiSend", native = AsgiSender))]
pub trait AsgiSend: Send + Sync + Sized + 'static {
    #[host(await)]
    async fn __call__(&self, message: AsgiSent) -> Result<(), WorkerError>;
}
