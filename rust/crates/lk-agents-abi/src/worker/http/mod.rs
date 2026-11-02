//! Language-neutral endpoint byte streams and their host-supplied consumer.

use std::any::Any;
use std::sync::Arc;

use crate::errors::HostError;

/// An exchange after its preamble, with independent reads and writes: the native
/// `ByteStream`. No application protocol is interpreted by the worker.
#[crate::host(native(refused = "expected a native ByteStream"))]
pub trait ByteTransport: Any + Send + Sync {
    /// Up to `max_bytes` of the peer's bytes; empty once the peer half-closed (FIN).
    fn read(&self, max_bytes: usize) -> crate::BoxFuture<'_, Result<Vec<u8>, HostError>>;
    /// All of `data`, waiting on flow control.
    fn write(&self, data: Vec<u8>) -> crate::BoxFuture<'_, Result<(), HostError>>;
    /// FIN: the response is complete.
    fn finish(&self);
    /// RESET_STREAM and STOP_SENDING.
    fn reset(&self, code: livekit_protocol::agent_http::HttpStreamResetCode);
    /// STOP_SENDING only.
    fn stop(&self, code: livekit_protocol::agent_http::HttpStreamResetCode);
}

#[crate::host(class(name = "ByteStream", native = ByteTransport))]
pub trait ByteStream: Send + Sync + Sized + 'static {
    #[host(await)]
    async fn read(
        &self,
        #[host(default = 65536)] max_bytes: usize,
    ) -> Result<Vec<u8>, crate::worker::ipc_host::WorkerError>;

    #[host(await)]
    async fn write(&self, data: Vec<u8>) -> Result<(), crate::worker::ipc_host::WorkerError>;

    /// Idempotent; a stream already reset raises.
    #[host(call)]
    fn finish(&self) -> Result<(), crate::worker::ipc_host::WorkerError>;

    #[host(call)]
    fn reset(&self, code: i32) -> Result<(), crate::worker::ipc_host::WorkerError>;

    #[host(call)]
    fn stop(&self, code: i32) -> Result<(), crate::worker::ipc_host::WorkerError>;
}

/// Shared admission and payload reservations for a host's HTTP application server.
#[crate::host(class(name = "_core_HttpServerCapacity", pid = false))]
pub trait HttpServerCapacity: Send + Sized + 'static {
    #[host(new)]
    fn new(max_handlers: usize, max_bytes: usize) -> Self;
    #[host(getter)]
    fn handlers(&self) -> usize;
    #[host(getter)]
    fn payload_bytes(&self) -> usize;
    /// Atomically acquire a handler slot and its initial read/head reservation.
    #[host(call)]
    fn admit(&mut self, bytes: usize) -> bool;
    #[host(call)]
    fn reserve(&mut self, bytes: usize) -> bool;
    #[host(call)]
    fn release(&mut self, bytes: usize) -> Result<(), CapacityError>;
    /// Release the handler and its initial reservation after all connection cleanup settles.
    #[host(call)]
    fn complete(&mut self, bytes: usize) -> Result<(), CapacityError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
pub enum CapacityError {
    #[error("unbalanced HTTP payload reservation")]
    UnbalancedBytes,
    #[error("unbalanced HTTP handler reservation")]
    UnbalancedHandlers,
}

/// Resource limits of the host HTTP/ASGI adapter.
#[crate::host(record(
    module = "http",
    name = "_HttpServerLimits",
    path = "livekit.agents.http",
    dataclass,
    frozen
))]
pub struct HttpServerLimits {
    #[host(default = 1048576)]
    pub request_head_bytes: usize,
    #[host(default = 65536)]
    pub body_metadata_bytes: usize,
    #[host(default = 1.0)]
    pub close_flush_timeout: f64,
    #[host(default = 1048576)]
    pub websocket_frame_bytes: usize,
    #[host(default = 4194304)]
    pub websocket_message_bytes: usize,
    #[host(default = 16)]
    pub websocket_inbox: usize,
    #[host(default = 8388608)]
    pub websocket_inbox_bytes: usize,
    #[host(default = 8388608)]
    pub websocket_outbox_bytes: usize,
    #[host(default = 256)]
    pub pending_handlers: usize,
    #[host(default = 536870912)]
    pub payload_bytes: usize,
}

/// The binding supplies the application server. The worker only dispatches streams and asks
/// the host to drain its endpoint resources; it does not parse application protocol bytes.
#[crate::host(proxy(host = "livekit.agents.http._AsgiServer"))]
pub trait EndpointHandler: Send + Sync {
    #[host(await, name = "_core_handle")]
    fn handle(
        &self,
        stream: Arc<dyn ByteTransport>,
        request: EndpointRequest,
    ) -> crate::BoxFuture<'_, Result<(), HostError>>;

    #[host(await, name = "_core_drain")]
    fn drain(&self) -> crate::BoxFuture<'_, Result<(), HostError>>;
}

mod reset_codes;
pub use reset_codes::*;

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
}

/// Application HTTP route visibility; A2A routes always require authentication.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(module = "worker_endpoint", emit, extract))]
pub struct EndpointOptions {
    #[host(default = false)]
    pub public: bool,
}
