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
