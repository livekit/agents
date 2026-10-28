//! Endpoint request and route policy, shared by the worker and host authoring surface.

mod reset_codes;
pub use reset_codes::*;

/// The authenticated node preamble attached to an ASGI request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(module = "worker_endpoint", emit, extract))]
pub struct EndpointRequest {
    pub request_id: String,
    pub authorized: bool,
    pub route: String,
    pub kind: String,
    pub client_addr: String,
    pub scheme: String,
    pub timeout_ms: u64,
}

/// Application HTTP route visibility; A2A routes always require authentication.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(module = "worker_endpoint", emit, extract))]
pub struct EndpointOptions {
    #[host(default = false)]
    pub public: bool,
}

/// Bounds on one exchange and on the worker's admitted endpoint requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndpointLimits {
    pub request_head_bytes: usize,
    pub chunk_line_bytes: usize,
    pub trailers_bytes: usize,
    pub websocket_frame_bytes: usize,
    pub websocket_message_bytes: usize,
    pub websocket_inbox: usize,
    pub websocket_inbox_bytes: usize,
    pub websocket_outbox_bytes: usize,
    pub pending_handlers: usize,
    /// Logical endpoint payload reservations, excluding application-owned results and transport TLS.
    pub worker_payload_bytes: usize,
}

impl Default for EndpointLimits {
    fn default() -> Self {
        Self {
            request_head_bytes: 1 << 20,
            chunk_line_bytes: 8 << 10,
            trailers_bytes: 64 << 10,
            websocket_frame_bytes: 1 << 20,
            websocket_message_bytes: 4 << 20,
            websocket_inbox: 16,
            websocket_inbox_bytes: 8 << 20,
            websocket_outbox_bytes: 8 << 20,
            pending_handlers: 256,
            worker_payload_bytes: 512 << 20,
        }
    }
}
