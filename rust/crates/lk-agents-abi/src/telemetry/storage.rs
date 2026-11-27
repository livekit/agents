//! agent-db as its client sees it: each request's time and outcome, each refusal sent again and
//! each redial, beside the server's own `db_requests_total{op,class}`.

use super::decl::{Attrs, SpanValue, metric};

/// A storage request's bucket bounds, in seconds: from 1 ms, doubling, so a tail of tens of
/// milliseconds shows.
pub const STORAGE_DURATION_BOUNDS: &[f64] = &[
    0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024, 2.048, 4.096,
    8.192, 16.384,
];

/// A request to agent-db or to the threads service.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageOp {
    /// One write statement.
    Exec,
    /// Statements applied atomically.
    Batch,
    /// One query's rows.
    Query,
    /// A socket's dial and `Hello`.
    Hello,
    /// The threads service's create.
    ThreadCreate,
    /// The threads service's get.
    ThreadGet,
    /// The threads service's list.
    ThreadList,
    /// The threads service's update.
    ThreadUpdate,
    /// The threads service's delete.
    ThreadDelete,
    /// The threads service's item page.
    ThreadItems,
    /// The threads service's task page.
    ThreadTasks,
}

/// Who owns a request's outcome: agent-db's classes, and the client's own `unknown`.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageOutcome {
    /// It succeeded.
    Ok,
    /// The caller's request, credentials or own cancel.
    Client,
    /// The database is served elsewhere: the client redials and lands there, nothing lost.
    Redirect,
    /// Refused for pressure that passes, sent again after a backoff.
    Capacity,
    /// A failure the server owns, its timeouts included.
    Server,
    /// No answer said: the connection was lost or the wait ran out, or a code this build does not
    /// know.
    Unknown,
}

/// Why a request refused before it applied is sent again, or a socket is dialed again.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageRetryReason {
    /// The database moved, drains or was evicted from its server (`wrong_shard_server`,
    /// `idle_evicted`).
    Moved,
    /// Its owner cannot serve the database now (`unavailable`), or a read's answer said nothing of
    /// its effect.
    Unavailable,
    /// A read held its snapshot past the log's limit (`snapshot_expired`).
    SnapshotExpired,
    /// Refused for pressure that passes (`retry`).
    Capacity,
    /// Its socket drained before answering it, again after it was sent once more.
    Drained,
    /// The connection was lost, unreachable or unanswered in time.
    Transport,
    /// This process refused it: a local database busy or closed, a task that ended.
    Local,
}

/// Time one request took as its client saw it, from its call to its last answer: its wait for a
/// socket and the connection's own resend to a moved database included, a retry owner's later
/// attempt not.
#[metric(
    name = "lk.agents.storage.request_duration",
    kind = Histogram,
    unit = "s",
    bounds = STORAGE_DURATION_BOUNDS
)]
#[derive(Attrs)]
pub struct StorageRequestDuration {
    /// The request.
    #[attr("op")]
    pub op: StorageOp,
    /// Who owns its outcome.
    #[attr("class")]
    pub class: StorageOutcome,
}

/// Requests sent again: by the connection to a database that moved, or by the retry owner after a
/// refusal that applied nothing.
#[metric(name = "lk.agents.storage.retries", kind = Counter, unit = "{retry}")]
#[derive(Attrs)]
pub struct StorageRetries {
    /// Why it was sent again.
    #[attr("reason")]
    pub reason: StorageRetryReason,
}

/// Sockets dialed again for a connection whose socket ended or whose last dial failed.
#[metric(name = "lk.agents.storage.redials", kind = Counter, unit = "{redial}")]
#[derive(Attrs)]
pub struct StorageRedials {
    /// Why the last socket ended, or why the last dial failed.
    #[attr("reason")]
    pub reason: StorageRetryReason,
}
