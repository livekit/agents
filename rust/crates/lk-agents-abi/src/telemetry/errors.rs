//! What a span records of a failure (docs/internals/telemetry.md "Errors"), for both span
//! handles.

use std::borrow::Cow;

use super::decl::SpanValue;
use crate::errors::{ApiError, ApiErrorKind, HostError, ProviderError};

/// `error.type`: the class of a recorded failure.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorType {
    /// An application exception; `exception.type` names its class.
    Exception,
    Timeout,
    Cancelled,
    ApiError,
    ApiStatusError,
    ApiConnectionError,
    StorageError,
    AuthError,
    IoError,
    /// A tool reported an error to the model.
    ToolError,
    /// A run ended with an error.
    RunError,
    JudgeError,
    /// A callback that could not run (its agent was gone, its call undeliverable).
    Unanswered,
    EventLoopBlocked,
}

/// A typed failure a span records: `error.type`, the status message and the exception event.
pub trait SpanError {
    fn error_type(&self) -> ErrorType;
    fn message(&self) -> String;
    /// `exception.type`: the application exception's class, else `error.type`.
    fn exception_type(&self) -> Cow<'static, str> {
        self.error_type().as_str().into()
    }
    fn stacktrace(&self) -> Option<String> {
        None
    }
}

/// An application exception: the host's type, message and traceback.
impl SpanError for HostError {
    fn error_type(&self) -> ErrorType {
        ErrorType::Exception
    }
    fn message(&self) -> String {
        self.message.clone()
    }
    fn exception_type(&self) -> Cow<'static, str> {
        if self.kind.is_empty() { "HostError".into() } else { self.kind.clone().into() }
    }
    fn stacktrace(&self) -> Option<String> {
        self.traceback.clone()
    }
}

impl SpanError for ApiError {
    fn error_type(&self) -> ErrorType {
        match self.kind {
            ApiErrorKind::Api => ErrorType::ApiError,
            ApiErrorKind::Status => ErrorType::ApiStatusError,
            ApiErrorKind::Connection => ErrorType::ApiConnectionError,
            ApiErrorKind::Timeout => ErrorType::Timeout,
            ApiErrorKind::Cancelled => ErrorType::Cancelled,
        }
    }
    fn message(&self) -> String {
        self.message.clone()
    }
}

impl SpanError for ProviderError {
    fn error_type(&self) -> ErrorType {
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
    fn exception_type(&self) -> Cow<'static, str> {
        match self {
            ProviderError::Host { error, .. } => error.exception_type(),
            ProviderError::Local(error) => error.exception_type(),
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
impl SpanError for crate::Elapsed {
    fn error_type(&self) -> ErrorType {
        ErrorType::Timeout
    }
    fn message(&self) -> String {
        "the deadline passed".to_owned()
    }
}

impl SpanError for crate::storage::StorageError {
    fn error_type(&self) -> ErrorType {
        ErrorType::StorageError
    }
    fn message(&self) -> String {
        self.to_string()
    }
}

/// An A2A request the door refused or failed.
impl SpanError for crate::a2a::door::DoorError {
    fn error_type(&self) -> ErrorType {
        match self {
            Self::Auth(_) => ErrorType::AuthError,
            Self::Storage(error) => error.error_type(),
        }
    }
    fn message(&self) -> String {
        self.to_string()
    }
}

impl SpanError for crate::voice::io::IoError {
    fn error_type(&self) -> ErrorType {
        ErrorType::IoError
    }
    fn message(&self) -> String {
        self.to_string()
    }
}

/// A failure with no error value of its own (a hook that timed out, a run's error, a stalled
/// event loop).
pub struct Failure {
    pub kind: ErrorType,
    pub message: String,
}

impl SpanError for Failure {
    fn error_type(&self) -> ErrorType {
        self.kind
    }
    fn message(&self) -> String {
        self.message.clone()
    }
}
