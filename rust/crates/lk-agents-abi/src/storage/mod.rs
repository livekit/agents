//! Thread storage as the host sees it: the error every storage operation fails with.

/// A storage failure. `Unavailable` is the one retryable error; a task that ends on it fails
/// with reason `storage-unavailable`. The host raises the classes of `livekit.agents.storage`.
#[crate::host(error(with = crate::storage::storage_error_to_py))]
#[derive(Debug, Clone, thiserror::Error)]
pub enum StorageError {
    /// An unknown, deleted or expired thread, or a database agent-db does not have.
    #[error("thread not found: {0}")]
    ThreadNotFound(String),
    /// A fenced write that matched no row: the lane moved past the version the run loaded, or
    /// the lease token is no longer the lane's.
    #[error("version conflict on lane {lane}: expected version {expected}")]
    VersionConflict { lane: String, expected: i64 },
    /// A field that could not be serialized; the second string says why.
    #[error("field {0} is not serializable: {1}")]
    NotSerializable(String, String),
    #[error("{what} exceeds the limit of {limit} bytes")]
    TooLarge { what: String, limit: usize },
    /// Retryable: a connection lost mid-request, a severed database, a lease that lapsed.
    #[error("agent-store unavailable: {0}")]
    Unavailable(String),
    /// A write sent and never answered: it may or may not have committed. Resolved by the run's
    /// identity before anything is reported or retried; never proof of a rollback.
    #[error("write outcome unknown: {0}")]
    Uncertain(String),
    /// `Lane::open` with `wait: false` found the lease held.
    #[error("lease held: {0}")]
    LeaseHeld(String),
    /// A waiter that did not acquire the lane within the wait (reason `lease-timeout`).
    #[error("lease timeout: {0}")]
    LeaseTimeout(String),
    /// The `reject` policy refused the message while a turn runs; the running task, if known.
    #[error("the lane is running a turn; the message was rejected")]
    RunRejected { running_task_id: Option<String> },
    /// The host cancelled the handler mid-turn: the run ended `canceled` and nothing was written.
    #[error("the turn was cancelled: {0}")]
    Cancelled(String),
    /// A refused token on either plane.
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    /// An HTTP resource that is not there: a stream, a database on the management API.
    #[error("not found: {0}")]
    NotFound(String),
    /// A server-reported error with its stable code, from the data plane or Twirp.
    #[error("{code}: {message}")]
    Server { code: String, message: String },
    /// A non-2xx the streams API answered that the mapping above does not name.
    #[error("http {status}: {message}")]
    Http { status: u16, message: String },
    /// A frame or a row that did not decode.
    #[error("decode: {0}")]
    Decode(String),
    /// An unexpected frame.
    #[error("protocol: {0}")]
    Protocol(String),
    /// A database whose `schema_version` this client does not write.
    #[error("schema: {0}")]
    Schema(String),
    #[error("invalid argument: {0}")]
    Invalid(String),
}

impl StorageError {
    /// `true` for the errors a caller may retry: [`StorageError::Unavailable`] only.
    pub fn retryable(&self) -> bool {
        matches!(self, StorageError::Unavailable(_))
    }
}

/// The `livekit.agents.storage` class of `error`: `RunRejected` carries `running_task_id`, a
/// cancelled turn is the `asyncio.CancelledError` the host raised, an invalid argument a
/// `ValueError`, and every other failure a plain `StorageError`.
#[cfg(feature = "python")]
pub fn storage_error_to_py(error: StorageError) -> pyo3::PyErr {
    use pyo3::prelude::*;

    let message = error.to_string();
    let class = match &error {
        StorageError::ThreadNotFound(_) => "ThreadNotFound",
        StorageError::VersionConflict { .. } => "VersionConflict",
        StorageError::NotSerializable(..) => "NotSerializable",
        StorageError::TooLarge { .. } => "TooLarge",
        StorageError::Unavailable(_) => "StorageUnavailable",
        StorageError::RunRejected { .. } => "RunRejected",
        StorageError::Invalid(_) => return pyo3::exceptions::PyValueError::new_err(message),
        StorageError::Cancelled(_) => {
            return pyo3::exceptions::asyncio::CancelledError::new_err(message);
        }
        _ => "StorageError",
    };
    let err = crate::python::host_exception("livekit.agents.storage.errors", class, message);
    if let StorageError::RunRejected { running_task_id } = error {
        // an attribute of the instance, so the exception's text stays the plain sentence
        Python::attach(|py| {
            let _ = err.value(py).setattr("running_task_id", running_task_id);
        });
    }
    err
}
