//! Language-neutral host failures. The adapter owns the original exception object.

use super::side_table::side_table_ref;

side_table_ref!(CapturedError, "a captured host exception");

impl CapturedError {
    /// Take the original exception exactly once across all clones.
    pub fn take(&self) -> Option<u64> {
        self.0.take()
    }
}

/// `None` when another clone already took it.
#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for CapturedError {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        Ok(match crate::python::release(self) {
            Some(obj) => obj.into_bound(py),
            None => py.None().into_bound(py),
        })
    }
}

/// An error crossing a host boundary. Recovery policy belongs to the operation receiving it.
/// Clones share ownership of the original exception; the last drop releases its adapter entry.
/// Raised back to the host as the exception it was.
#[crate::host(error(original))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostError {
    pub kind: String,
    pub message: String,
    pub host_ref: Option<CapturedError>,
}

impl HostError {
    /// The captured exception's own class name, what a wrapping error's `host_kind` delegates to.
    pub fn host_kind(&self) -> &str {
        &self.kind
    }

    /// A diagnostic without an original language exception (for example, an invalid host result).
    pub fn new(message: impl Into<String>) -> Self {
        Self { kind: String::new(), message: message.into(), host_ref: None }
    }

    pub fn captured(
        kind: impl Into<String>,
        message: impl Into<String>,
        host_ref: CapturedError,
    ) -> Self {
        Self { kind: kind.into(), message: message.into(), host_ref: Some(host_ref) }
    }
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.kind.is_empty() {
            write!(f, "{}", self.message)
        } else {
            write!(f, "{}: {}", self.kind, self.message)
        }
    }
}
impl std::error::Error for HostError {}
impl From<String> for HostError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}
impl From<&str> for HostError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

pub type HostResult<T = ()> = Result<T, HostError>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{RefRegistry, side_table::testkit::Table};

    #[test]
    fn shared_host_errors_preserve_diagnostics_and_release_on_last_drop() {
        let table = Table::holding([8]);
        let error = HostError::captured(
            "PluginError",
            "failed",
            CapturedError::new(8, &table.as_registry()),
        );
        let clone = error.clone();
        assert_eq!(error.to_string(), "PluginError: failed");
        drop(error);
        assert_eq!(table.outstanding(), 1);
        drop(clone);
        assert_eq!(table.outstanding(), 0);
        assert_eq!(HostError::new("host closed").to_string(), "host closed");
    }
}
