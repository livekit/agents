//! The exception taxonomy and its ordinary host exception classes. The generated Python classes
//! retain the public inheritance hierarchy and constructor contracts.

mod api;
mod host_capture;
mod provider;
pub(crate) mod side_table;

pub use api::{
    ApiError, ApiErrorFunctions, ApiErrorKind, DEFAULT_CONNECTION_MESSAGE, DEFAULT_TIMEOUT_MESSAGE,
    TRANSIENT_CLIENT_STATUSES, TimeoutPhase, http_error_message, http_reason, retryable_for_status,
};
mod host;
pub use host::{CapturedError, HostError, HostResult};
pub use host_capture::{
    ProviderErrorMetadata, RealtimeErrorMetadata, TaxonomyMatches, host_retryable,
};
pub use provider::{ProviderError, PullError};
pub use side_table::RefRegistry;

// Constructors are part of the boundary contract, including plugin-authored exceptions. Only
// cause traversal and status display/normalization require host behavior.
#[crate::host(exceptions(module = "api_errors", path = "livekit.agents._bindings", classes(
    UnexpectedModelBehavior(base = "RuntimeError", doc = "The model exhausted its recovery attempts."),
    AssignmentTimeoutError(base = "Exception", doc = "The job was accepted but no assignment arrived in time."),
    APIError(base = "Exception", doc = "An API request failed.",
        fields(
            message(ty = "str", positional),
            body(ty = "object | None", default = None),
            retryable(ty = "bool", default = true, class_default = false)
        ), base_args(message), str = "message", repr(message, body, retryable)),
    APIStatusError(base = "APIError", doc = "An API response has an error status code.",
        fields(
            message(ty = "str", positional, inherited),
            status_code(ty = "int", default = -1),
            request_id(ty = "str | None", default = None),
            body(ty = "object | None", default = None, inherited),
            retryable(ty = "bool | None", default = None, inherited)
        ), base_args(message, body, retryable), normalize(retryable(call = "_status_retryable", args(status_code, retryable))),
        str = "_status_error_str", repr(message, status_code, request_id, body, retryable)),
    APIConnectionError(base = "APIError", doc = "An API request failed due to a connection error.",
        fields(
            message(ty = "str", positional, default = "Connection error.", inherited),
            retryable(ty = "bool", default = true, inherited),
            body(ty = "object | None", default = None, hidden, inherited)
        ), base_args(message, body, retryable), str = "_connection_error_str"),
    APITimeoutError(base = "APIConnectionError", doc = "An API request timed out; phase names the expired connection limit.",
        fields(
            message(ty = "str", positional, default = "Request timed out.", inherited),
            retryable(ty = "bool", default = true, inherited),
            phase(ty = "Literal['pool', 'connect', 'read', 'write', 'total'] | None", default = None)
        ), base_args(message, retryable)),
    StreamStateError(base = "RuntimeError", doc = "A stream was pulled while another pull was still in flight."),
    CLIError(base = "Exception"),
    RedactedCause(base = "Exception", doc = "A provider cause with its type and traceback, without sensitive message text.")
)))]
const _: () = ();
