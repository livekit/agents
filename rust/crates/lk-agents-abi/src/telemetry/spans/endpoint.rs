//! The worker's endpoint server.

use crate::telemetry::decl::{Attrs, External, SpanValue, span};
use crate::telemetry::semconv::otel;

/// One exchange the worker serves on an agent endpoint: an HTTP request and its response, or a
/// WebSocket connection. Continues the caller's `traceparent`.
#[span(name = "http_request", parent = External, kind = Server, end = HttpRequestEnd)]
#[derive(Attrs)]
pub struct HttpRequest {
    #[attr(otel::HTTP_REQUEST_METHOD)]
    pub method: String,
    /// The manifest route the request matched.
    #[attr(otel::HTTP_ROUTE)]
    pub route: String,
    /// The node's id for the request, as its logs name it.
    #[attr("lk.endpoint.request_id")]
    pub request_id: String,
}

#[derive(Attrs)]
pub struct HttpRequestEnd {
    /// The response status, `101` for an accepted WebSocket; none when nothing was answered.
    #[attr(otel::HTTP_RESPONSE_STATUS_CODE)]
    pub status: Option<u32>,
    /// Why the exchange did not complete, as its `x-lk-completion` names it.
    #[attr("lk.endpoint.completion")]
    pub completion: Option<HttpCompletion>,
}

#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpCompletion {
    /// The application failed.
    Internal,
    /// The worker stopped while the exchange ran.
    Canceled,
    /// No response head, or no answer to an upgrade, before the deadline.
    Timeout,
    /// The client broke the request's framing.
    Protocol,
}
