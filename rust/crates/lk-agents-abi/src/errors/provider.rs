//! A host exception travels through Rust behind a [`CapturedError`] and is re-raised as the same
//! object; `retryable` and `status_code` are copied out so `FallbackChain` decides with no GIL.

use crate::errors::api::{ApiError, ApiErrorKind};
use super::{CapturedError, HostError};

/// A provider failure, from either side of the seam.
// Clones share the side-table entry. Crossing to the host re-raises the original exception.
#[crate::host(error(with = crate::python::provider_error_to_py))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProviderError {
    /// A host failure plus the provider-specific metadata used by retry policy.
    #[error("{error}")]
    Host {
        error: HostError,
        api_kind: Option<ApiErrorKind>,
        retryable: bool,
        status_code: Option<i64>,
    },
    /// A native provider's failure, already classified by [`crate::errors::retryable_for_status`].
    #[error("{0}")]
    Local(#[from] ApiError),
}

/// The host's own exception (resolved, not released), else an `APIError` from the engine's message.
#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for ProviderError {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        Ok(crate::python::to_py_exception(py, &self)?.into_bound(py))
    }
}

impl ProviderError {
    /// Answerable with no GIL held.
    pub fn retryable(&self) -> bool {
        match self {
            ProviderError::Host { retryable, .. } => *retryable,
            ProviderError::Local(err) => err.retryable,
        }
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

#[cfg(test)]
mod tests;
