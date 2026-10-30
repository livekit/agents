//! A host exception travels through Rust behind a [`CapturedError`] and is re-raised as the same
//! object; `retryable` and `status_code` are copied out so `FallbackChain` decides with no GIL.

use super::{CapturedError, HostError};
use crate::errors::api::{ApiError, ApiErrorKind};

/// A provider failure, from either side of the seam.
// Clones share the side-table entry. Crossing to the host re-raises the original exception.
#[crate::host(error)]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProviderError {
    /// A host failure plus the provider-specific metadata used by retry policy.
    #[error("{error}")]
    #[host(delegate = error)]
    Host {
        error: HostError,
        api_kind: Option<ApiErrorKind>,
        retryable: bool,
        status_code: Option<i64>,
    },
    /// A native provider's failure, already classified by [`crate::errors::retryable_for_status`].
    #[error("{0}")]
    #[host(delegate)]
    Local(#[from] ApiError),
}


impl ProviderError {
    /// The same failure, no longer to be retried: whoever spent the budget already gave up.
    pub fn into_final(self) -> ProviderError {
        match self {
            ProviderError::Host { error, api_kind, status_code, .. } => {
                ProviderError::Host { error, api_kind, retryable: false, status_code }
            }
            ProviderError::Local(error) => {
                ProviderError::Local(ApiError { retryable: false, ..error })
            }
        }
    }

    /// Scalar policy input. A host error outside the API taxonomy never grants a retry,
    /// even if foreign metadata includes a truthy retryable flag or a status code.
    pub fn attempt_error(&self) -> ApiError {
        match self {
            ProviderError::Local(error) => error.clone(),
            ProviderError::Host { error, api_kind, retryable, status_code } => ApiError {
                kind: api_kind.unwrap_or(ApiErrorKind::Api),
                message: error.message.clone(),
                retryable: api_kind.is_some() && *retryable,
                status_code: *status_code,
                request_id: None,
                phase: None,
                cause: None,
            },
        }
    }

    /// Answerable with no GIL held.
    pub fn retryable(&self) -> bool {
        match self {
            ProviderError::Host { retryable, .. } => *retryable,
            ProviderError::Local(err) => err.retryable,
        }
    }

    /// The host cancelled the call itself (`asyncio.CancelledError`): its provider stopped, nothing
    /// failed.
    pub fn cancelled(&self) -> bool {
        matches!(self, ProviderError::Host { error, .. } if error.kind == "CancelledError")
    }

    /// `Some` only for a host exception.
    pub fn host_ref(&self) -> Option<CapturedError> {
        match self {
            ProviderError::Host { error, .. } => error.host_ref.clone(),
            ProviderError::Local(_) => None,
        }
    }

    /// `None` (outside the taxonomy) also means never retried.
    pub fn api_kind(&self) -> Option<ApiErrorKind> {
        match self {
            ProviderError::Host { api_kind, .. } => *api_kind,
            ProviderError::Local(err) => Some(err.kind),
        }
    }

    pub fn status_code(&self) -> Option<i64> {
        match self {
            ProviderError::Host { status_code, .. } => *status_code,
            ProviderError::Local(err) => err.status_code,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            ProviderError::Host { error, .. } => &error.message,
            ProviderError::Local(err) => &err.message,
        }
    }
}
