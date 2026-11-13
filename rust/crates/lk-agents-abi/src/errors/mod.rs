mod api;
mod host_capture;
mod provider;
pub(crate) mod side_table;

pub use api::{
    ApiError, ApiErrorFunctions, ApiErrorKind, DEFAULT_CONNECTION_MESSAGE, DEFAULT_TIMEOUT_MESSAGE,
    ErrorKind, ProviderSaid, Timeout, WebSocketFunctions, http_error_message, http_reason,
};
mod host;
pub use host::{CapturedError, HostError, HostErrorKind, HostResult, label_or};
pub use host_capture::{
    ProviderErrorMetadata, RealtimeErrorMetadata, TaxonomyMatches, host_retryable,
};
pub use provider::{ProviderError, PullError};
pub use side_table::RefRegistry;

/// A blocking job ended without its result: it unwound, or was dropped unrun (a forked child
/// never runs its parent's queue).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("worker job ended without a result")]
pub struct JobLost;

impl From<JobLost> for std::io::Error {
    fn from(error: JobLost) -> Self {
        std::io::Error::other(error)
    }
}

// Constructors are part of the boundary contract, including plugin-authored exceptions. Only
// cause traversal and status display/normalization require host behavior.
#[crate::host(exceptions(module = "api_errors", path = "livekit.agents._bindings", classes(
    UnexpectedModelBehavior(base = "RuntimeError", doc = "The model exhausted its recovery attempts."),
    AssignmentTimeoutError(base = "Exception", doc = "The job was accepted but no assignment arrived in time."),
    APIError(base = "Exception", doc = "An API request failed: `kind` says what failed and decides `retryable`. `detail` holds the provider's own message, logged once under an `lk.pii.*` key and never shown in the text.",
        fields(
            message(ty = "str", positional),
            kind(ty = "livekit.agents._generated.types.ErrorKind"),
            body(ty = "object | None", default = None),
            detail(ty = "str | None", default = None),
            retryable(ty = "bool", hidden, default = false, class_default = false)
        ), base_args(message), normalize(retryable(call = "_core.error_kind_retryable", args(kind))),
        str = "message", repr(message, kind, body)),
    APIStatusError(base = "APIError", doc = "An API response has an error status code; its kind is the status's unless the provider's meaning says otherwise. `retry_after` is the seconds the provider asked to wait.",
        fields(
            message(ty = "str", positional, inherited),
            status_code(ty = "int"),
            kind(ty = "livekit.agents._generated.types.ErrorKind | None", default = None, inherited),
            request_id(ty = "str | None", default = None),
            body(ty = "object | None", default = None, inherited),
            detail(ty = "str | None", default = None, inherited),
            retry_after(ty = "float | None", default = None)
        ), base_args(message, kind, body, detail), normalize(kind(call = "_core.error_kind_for_status", args(status_code, kind))),
        str = "_status_error_str", repr(message, status_code, kind, request_id, body)),
    APIConnectionError(base = "APIError", doc = "An API request failed due to a connection error, or the connection was closed for the reason `kind` names.",
        fields(
            message(ty = "str", positional, default = "Connection error.", inherited),
            kind(ty = "livekit.agents._generated.types.ErrorKind", default = "connection", inherited),
            body(ty = "object | None", default = None, hidden, inherited),
            detail(ty = "str | None", default = None, inherited)
        ), base_args(message, kind, body, detail), str = "_connection_error_str"),
    APITimeoutError(base = "APIConnectionError", doc = "An API request timed out; timeout names the expired connection limit.",
        fields(
            message(ty = "str", positional, default = "Request timed out.", inherited),
            kind(ty = "livekit.agents._generated.types.ErrorKind", default = "timeout", hidden, inherited),
            timeout(ty = "livekit.agents._generated.types.Timeout | None", default = None),
            detail(ty = "str | None", default = None, inherited)
        ), base_args(message, kind, detail)),
    StreamStateError(base = "RuntimeError", doc = "A stream was pulled while another pull was still in flight."),
    CLIError(base = "Exception"),
    RedactedCause(base = "Exception", doc = "A provider cause with its type and traceback, without sensitive message text.")
)))]
const _: () = ();

/// An input, option, payload or definition the operation cannot use: what is wrong with it.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct Invalid(pub String);
