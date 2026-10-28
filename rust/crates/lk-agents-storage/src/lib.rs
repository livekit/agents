#![allow(
    clippy::disallowed_macros,
    reason = "Clippy checks macro bans at crate scope; exact legacy macro sites are ratcheted."
)]

//! `lk-agents-storage`: the synchronous storage core of Agents 2.0. It decides; the runtime
//! (`lk_agents_runtime::storage`) executes. No I/O, clock, entropy or task lives here.
//!
//! * [`sql`]: the backend-neutral SQL vocabulary: values, statements, results, typed row reads.
//! * [`schema`]: the thread tables as records, and every statement.
//! * [`db`]: each lane operation as a plan (one batch or read) and its typed decoding.
//! * [`run`]: task states and transitions, the run's outcome and the commit it writes.
//! * [`lease`]: lease renewal and lane polling decisions on supplied time.
//! * [`streams`]: the answer streams' records.
//! * `sqlite` (feature): the statements executed on a SQLite connection the caller owns.

pub mod db;
pub mod lease;
pub mod run;
pub mod schema;
pub mod sql;
#[cfg(feature = "sqlite")]
pub mod sqlite;
pub mod streams;

pub use db::{Commit, Committed, EventWrite, Fence, TaskWrite};
pub use run::{RunOutcome, TaskState};

/// A storage failure. `Unavailable` is the one retryable error; a task that ends on it fails
/// with reason `storage-unavailable`.
#[derive(Debug, Clone, thiserror::Error)]
pub enum StorageError {
    /// An id the threads service does not know, or a database agent-db does not have.
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

/// Unix seconds as a `Timestamp`, to the nearest nanosecond, so the float reads back unchanged.
pub fn timestamp(t: f64) -> pbjson_types::Timestamp {
    let seconds = t.floor();
    let nanos = ((t - seconds) * 1e9).round() as i32;
    if nanos >= 1_000_000_000 {
        pbjson_types::Timestamp { seconds: seconds as i64 + 1, nanos: 0 }
    } else {
        pbjson_types::Timestamp { seconds: seconds as i64, nanos }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The float reads back unchanged, and a rounding to a whole second carries.
    #[test]
    fn a_timestamp_keeps_the_seconds_float() {
        let t = timestamp(1791760302.4035509);
        assert_eq!(t.seconds as f64 + f64::from(t.nanos) / 1e9, 1791760302.4035509);
        assert_eq!(timestamp(0.9999999999), pbjson_types::Timestamp { seconds: 1, nanos: 0 });
    }
}
