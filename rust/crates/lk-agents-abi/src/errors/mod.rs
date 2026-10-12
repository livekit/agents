//! The exception taxonomy's Rust half. The Python classes stay Python: users catch the bases, and a
//! pyclass cannot sit in that MRO.

mod api;
mod host_capture;
mod payload_table;
mod provider;
pub(crate) mod side_table;

pub use api::{
    ApiError, ApiErrorKind, DEFAULT_CONNECTION_MESSAGE, DEFAULT_TIMEOUT_MESSAGE,
    TRANSIENT_CLIENT_STATUSES, http_error_message, http_reason, retryable_for_status,
};
mod host;
pub use host::{CapturedError, HostError, HostResult};
pub use host_capture::{TaxonomyMatches, host_retryable};
pub use payload_table::PayloadTable;
pub use provider::ProviderError;
pub use side_table::RefRegistry;
