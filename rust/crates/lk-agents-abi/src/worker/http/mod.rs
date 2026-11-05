//! Endpoint exchanges: the worker serves HTTP/1.1 and WebSocket on each endpoint stream and hands
//! the binding one parsed exchange, which it adapts to its own application framework.

use std::any::Any;
use std::sync::Arc;

use crate::errors::HostError;

/// One parsed exchange, as the worker passes it to the binding's handler.
#[crate::host(native(refused = "expected a native HttpExchange"))]
pub trait EndpointExchange: Any + Send + Sync {}

/// One request and its response, or one WebSocket connection. The worker owns parsing, framing,
/// limits, deadlines and closing; the binding reads events and sends messages.
#[crate::host(class(name = "HttpExchange", native = EndpointExchange))]
pub trait HttpExchange: Send + Sync + Sized + 'static {
    #[host(getter)]
    fn head(&self) -> RequestHead;

    /// The next request event. After the request body ends it waits for the response to
    /// complete, then reports `Disconnected`.
    #[host(await)]
    async fn receive(&self) -> Result<ExchangeEvent, ExchangeError>;

    #[host(await)]
    async fn send(&self, message: ExchangeMessage) -> Result<(), ExchangeError>;
}

/// Whether the exchange is an HTTP request or a WebSocket upgrade.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[crate::host(record(module = "worker_endpoint", emit))]
pub enum ExchangeKind {
    #[default]
    Http,
    Websocket,
}

/// The parsed request head. `path` is percent-decoded; `raw_path` and `query_string` are the
/// bytes on the wire.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(module = "worker_endpoint", emit))]
pub struct RequestHead {
    pub kind: ExchangeKind,
    pub method: String,
    pub http_version: String,
    pub path: String,
    pub raw_path: Vec<u8>,
    pub query_string: Vec<u8>,
    pub headers: Vec<(Vec<u8>, Vec<u8>)>,
    /// The WebSocket subprotocols the client offered.
    pub subprotocols: Vec<String>,
}

/// What `HttpExchange.receive` reports.
#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::host(record(module = "worker_endpoint", emit))]
pub enum ExchangeEvent {
    /// A piece of the request body; `more` is false on the last one.
    Body {
        data: Vec<u8>,
        more: bool,
    },
    /// The HTTP exchange ended: the response completed, or the client is gone.
    Disconnected,
    Text {
        text: String,
    },
    Binary {
        data: Vec<u8>,
    },
    /// The WebSocket connection closed, with its close code.
    Closed {
        code: u16,
    },
}

/// What `HttpExchange.send` accepts. On a WebSocket exchange, `Start` and `Body` before `Accept`
/// answer the upgrade with an HTTP response instead.
#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::host(record(module = "worker_endpoint", extract))]
pub enum ExchangeMessage {
    Start {
        status: u16,
        headers: Vec<(Vec<u8>, Vec<u8>)>,
    },
    /// Response body bytes; `more: false` completes the response.
    Body {
        data: Vec<u8>,
        more: bool,
    },
    Accept {
        subprotocol: Option<String>,
        headers: Vec<(Vec<u8>, Vec<u8>)>,
    },
    Text {
        text: String,
    },
    Binary {
        data: Vec<u8>,
    },
    Close {
        code: u16,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
pub enum ExchangeError {
    /// The client is gone or the exchange already ended.
    #[error("{0}")]
    #[host(kind = "ConnectionError")]
    Closed(String),
    /// The application broke the exchange's protocol.
    #[error("{0}")]
    Protocol(String),
}

/// The binding's application server: it adapts each exchange to its framework.
#[crate::host(proxy(host = "livekit.agents.http._AsgiServer"))]
pub trait EndpointHandler: Send + Sync {
    #[host(await, name = "_core_handle")]
    fn handle(
        &self,
        cx: &crate::telemetry::Cx,
        exchange: Arc<dyn EndpointExchange>,
        request: EndpointRequest,
    ) -> crate::BoxFuture<'_, Result<(), HostError>>;
}

/// The payload semantics declared in the Cloud route manifest and node preamble.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[crate::host(record(module = "worker_endpoint", emit, extract))]
pub enum EndpointKind {
    #[default]
    Http,
    Text,
}

/// The end client's scheme, asserted by the node rather than inferred from worker transport.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[crate::host(record(module = "worker_endpoint", emit, extract))]
pub enum HttpScheme {
    #[default]
    Http,
    Https,
}

/// The authenticated node preamble attached to an endpoint byte stream.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(module = "worker_endpoint", emit, extract))]
pub struct EndpointRequest {
    pub request_id: String,
    pub authorized: bool,
    pub route: String,
    pub kind: EndpointKind,
    pub client_addr: String,
    pub scheme: HttpScheme,
    pub timeout_ms: u64,
    /// The console's listener: nothing authenticates the caller, who holds every permission.
    #[host(default = false)]
    pub unauthenticated: bool,
}

/// Application HTTP route visibility; A2A routes always require authentication.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(module = "worker_endpoint", emit, extract))]
pub struct EndpointOptions {
    #[host(default = false)]
    pub public: bool,
}
