//! The worker endpoint's host-facing stream and ASGI server handles.

use crate::HostRef;
use crate::errors::HostError;

/// An exchange after its preamble, with independent reads and writes.
#[crate::host(class(name = "EndpointStream"))]
pub trait EndpointStream: Send + Sync + Sized + 'static {
    #[host(call, future = Vec<u8>)]
    fn read(&self, #[host(default = 65536)] max_bytes: usize) -> Result<HostRef, HostError>;

    #[host(call, future = ())]
    fn write(&self, data: Vec<u8>) -> Result<HostRef, HostError>;

    #[host(call)]
    fn finish(&self) -> Result<(), HostError>;

    #[host(call)]
    fn reset(&self, code: i32) -> Result<(), HostError>;

    #[host(call)]
    fn stop(&self, code: i32) -> Result<(), HostError>;
}

/// The host's ASGI application bridge and its active WebSockets.
#[crate::host(class(name = "EndpointServer"))]
pub trait EndpointServer: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new() -> Self;

    #[host(call, future = ())]
    fn serve(&self, stream: HostRef, app: HostRef, request: HostRef) -> Result<HostRef, HostError>;

    #[host(call, future = ())]
    fn close_websockets(
        &self,
        #[host(default = 1001)] code: u16,
        #[host(default = "worker draining")] reason: String,
    ) -> Result<HostRef, HostError>;
}

/// The native ASGI receive callback, bound to one endpoint exchange.
#[crate::host(class(name = "_core_AsgiReceive"))]
pub trait AsgiReceive: Send + Sync + Sized + 'static {
    #[host(call, future = HostRef)]
    fn __call__(&self) -> Result<HostRef, HostError>;
}

/// The native ASGI send callback, bound to one endpoint exchange.
#[crate::host(class(name = "_core_AsgiSend"))]
pub trait AsgiSend: Send + Sync + Sized + 'static {
    #[host(call, future = ())]
    fn __call__(&self, message: HostRef) -> Result<HostRef, HostError>;
}
