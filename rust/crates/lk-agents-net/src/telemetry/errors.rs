//! What a span records of a failure (docs/internals/telemetry.md "Errors").

use lk_agents_abi::errors::{ApiError, ApiErrorKind, HostError, ProviderError};

/// A typed failure a span records: `error.type`, the status message and the exception event.
pub trait SpanError {
    /// `error.type` and `exception.type`.
    fn error_type(&self) -> String;
    fn message(&self) -> String;
    fn stacktrace(&self) -> Option<String> {
        None
    }
}

/// An application exception: the host's type, message and traceback.
impl SpanError for HostError {
    fn error_type(&self) -> String {
        if self.kind.is_empty() { "HostError".to_owned() } else { self.kind.clone() }
    }
    fn message(&self) -> String {
        self.message.clone()
    }
    fn stacktrace(&self) -> Option<String> {
        self.traceback.clone()
    }
}

impl SpanError for ApiError {
    fn error_type(&self) -> String {
        match self.kind {
            ApiErrorKind::Api => "api_error",
            ApiErrorKind::Status => "api_status_error",
            ApiErrorKind::Connection => "api_connection_error",
            ApiErrorKind::Timeout => "timeout",
            ApiErrorKind::Cancelled => "cancelled",
        }
        .to_owned()
    }
    fn message(&self) -> String {
        self.message.clone()
    }
}

impl SpanError for ProviderError {
    fn error_type(&self) -> String {
        match self {
            ProviderError::Host { error, .. } => error.error_type(),
            ProviderError::Local(error) => error.error_type(),
        }
    }
    fn message(&self) -> String {
        match self {
            ProviderError::Host { error, .. } => error.message(),
            ProviderError::Local(error) => error.message(),
        }
    }
    fn stacktrace(&self) -> Option<String> {
        match self {
            ProviderError::Host { error, .. } => error.stacktrace(),
            ProviderError::Local(_) => None,
        }
    }
}

/// A deadline that passed.
impl SpanError for lk_agents_abi::Elapsed {
    fn error_type(&self) -> String {
        "timeout".to_owned()
    }
    fn message(&self) -> String {
        "the deadline passed".to_owned()
    }
}

/// A storage failure: its kind and message.
impl SpanError for lk_agents_abi::storage::StorageError {
    fn error_type(&self) -> String {
        "storage_error".to_owned()
    }
    fn message(&self) -> String {
        self.to_string()
    }
}

/// An A2A request the door refused or failed.
impl SpanError for lk_agents_abi::a2a::door::DoorError {
    fn error_type(&self) -> String {
        match self {
            Self::Auth(_) => "auth_error".to_owned(),
            Self::Storage(error) => error.error_type(),
        }
    }
    fn message(&self) -> String {
        self.to_string()
    }
}

/// A room I/O failure.
impl SpanError for lk_agents_abi::voice::io::IoError {
    fn error_type(&self) -> String {
        "io_error".to_owned()
    }
    fn message(&self) -> String {
        self.to_string()
    }
}

/// A failure known only by its kind and message (a run's error, an exporter-side refusal).
pub struct Failure {
    pub kind: &'static str,
    pub message: String,
}

impl SpanError for Failure {
    fn error_type(&self) -> String {
        self.kind.to_owned()
    }
    fn message(&self) -> String {
        self.message.clone()
    }
}
