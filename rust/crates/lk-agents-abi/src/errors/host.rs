use super::side_table::side_table_ref;

/// A key into the binding's side table for a captured host exception.
#[crate::host(codec(emit, host = "BaseException", py = "py_objects::captured_error"))]
#[derive(Clone)]
pub struct CapturedError(std::sync::Arc<super::side_table::Entry>);
side_table_ref!(CapturedError);

impl CapturedError {
    /// Take the original exception exactly once across all clones.
    pub fn take(&self) -> Option<u64> {
        self.0.take()
    }
}

/// What a caller branches on in a host failure. The adapter classifies the exception once, by
/// class: cancellation, the storage failures a handler raises, one per `StorageError` variant,
/// and an RPC error with its code (`rtc.RpcError`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HostErrorKind {
    #[default]
    Other,
    Cancelled,
    NotFound,
    Unavailable,
    LeaseLost,
    VersionConflict,
    NotSerializable,
    TooLarge,
    Rpc {
        code: u32,
    },
}

/// An error crossing a host boundary. Recovery policy belongs to the operation receiving it.
/// Clones share ownership of the original exception; the last drop releases its adapter entry.
/// Raised back to the host as the exception it was.
#[crate::host(error(original))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostError {
    pub kind: HostErrorKind,
    /// The exception's class name: diagnostics only, never a decision.
    pub class: String,
    pub message: String,
    pub host_ref: Option<CapturedError>,
    /// The host's formatted traceback, for the span that records the failure.
    pub traceback: Option<String>,
}

impl HostError {
    /// The captured exception's class name, what a wrapping error's `host_kind` delegates to.
    pub fn host_kind(&self) -> &str {
        &self.class
    }

    /// A diagnostic without an original language exception (for example, an invalid host result).
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            kind: HostErrorKind::Other,
            class: String::new(),
            message: message.into(),
            host_ref: None,
            traceback: None,
        }
    }

    pub fn captured(
        kind: HostErrorKind,
        class: impl Into<String>,
        message: impl Into<String>,
        host_ref: CapturedError,
    ) -> Self {
        Self {
            kind,
            class: class.into(),
            message: message.into(),
            host_ref: Some(host_ref),
            traceback: None,
        }
    }

    pub fn with_traceback(self, traceback: Option<String>) -> Self {
        Self { traceback, ..self }
    }

    /// The failure as Python's `logging` prints an exception (a record's `exc_text`): its
    /// traceback, else `Kind: message`.
    pub fn exc_text(&self) -> String {
        self.traceback.as_deref().map_or_else(|| self.to_string(), |t| t.trim_end().to_owned())
    }
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.class.is_empty() {
            write!(f, "{}", self.message)
        } else {
            write!(f, "{}: {}", self.class, self.message)
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

/// A label read from the host each time it is used (a provider's model, its markup dialect): a
/// failed read is logged and answered by `fallback`, since a label never fails what it names.
pub fn label_or<T>(what: &str, read: HostResult<T>, fallback: impl FnOnce() -> T) -> T {
    read.unwrap_or_else(|e| {
        log::warn!(target: crate::LOG_TARGET, "reading {what} failed: {e}");
        fallback()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{RefRegistry, side_table::testkit::Table};

    #[test]
    fn shared_host_errors_preserve_diagnostics_and_release_on_last_drop() {
        let table = Table::holding([8]);
        let error = HostError::captured(
            HostErrorKind::Other,
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
