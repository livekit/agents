//! agent-db's error codes. The wire carries a code's number, its name and a description; a
//! client branches on the number alone, through [`ErrorCode::retry`] and
//! [`ErrorCode::maybe_applied`], never on the text. A code this build does not know may have
//! applied and is never retried. The numbers group by kind: 1xxx the service could not serve
//! the request now, 2xxx the request itself is refused, 4xxx a broken invariant, 6xxx the
//! caller's credentials.

/// An error's number on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorCode(pub u32);

use crate::telemetry::storage::{StorageOutcome, StorageRetryReason};

/// A known code: its name, why the same request sent again may succeed (`None`: it may not),
/// whether a request answered with it may have taken effect, whole, and who owns that outcome,
/// as agent-db classes it.
struct Declared {
    code: ErrorCode,
    name: &'static str,
    retry: Option<StorageRetryReason>,
    maybe_applied: bool,
    class: StorageOutcome,
}

const fn declared(
    code: ErrorCode,
    name: &'static str,
    retry: Option<StorageRetryReason>,
    maybe_applied: bool,
    class: StorageOutcome,
) -> Declared {
    Declared { code, name, retry, maybe_applied, class }
}

impl ErrorCode {
    /// This server does not serve the database (evicted, draining or moved): redial. On request
    /// id 0, the socket ends now.
    pub const WRONG_SHARD_SERVER: Self = Self(1001);
    /// On request id 0: the database was evicted while the socket idled, and the socket ends.
    /// Redial when next needed.
    pub const IDLE_EVICTED: Self = Self(1002);
    /// Refused for pressure that passes, or cut while it waited its turn: send it again after a
    /// backoff.
    pub const RETRY: Self = Self(1003);
    /// The owner cannot serve the database now (its store failing or not confirming its
    /// ownership), and nothing applied: send it again after a backoff, as `RETRY`.
    pub const UNAVAILABLE: Self = Self(1004);
    /// A read held its snapshot past the log's limit and it was let go; nothing applied: send it
    /// again after a backoff, as `RETRY`, and it reads a fresh snapshot.
    pub const SNAPSHOT_EXPIRED: Self = Self(1007);
    /// The request may have committed, whole, or not at all.
    pub const COMMIT_UNKNOWN_RESULT: Self = Self(1021);
    /// Its client's Cancel stopped the request before it applied: only the canceller sees it.
    pub const CANCELLED: Self = Self(1025);
    /// The request ran out of time while it ran, cut by the server's request timeout before it
    /// applied: final, as FoundationDB's `transaction_timed_out`.
    pub const TIMED_OUT: Self = Self(1031);
    pub const INVALID_ARGUMENT: Self = Self(2001);
    pub const SQL_ERROR: Self = Self(2002);
    pub const TOO_LARGE: Self = Self(2003);
    pub const NOT_FOUND: Self = Self(2004);
    /// An invariant broke: the request may have applied.
    pub const INTERNAL: Self = Self(4100);
    pub const PERMISSION_DENIED: Self = Self(6000);
    pub const UNAUTHENTICATED: Self = Self(6001);

    const TABLE: [Declared; 15] = {
        use StorageOutcome::{Client, Redirect, Server};
        use StorageRetryReason::{Capacity, Moved, SnapshotExpired, Unavailable};
        [
            declared(Self::WRONG_SHARD_SERVER, "wrong_shard_server", Some(Moved), false, Redirect),
            declared(Self::IDLE_EVICTED, "idle_evicted", Some(Moved), false, Redirect),
            declared(Self::RETRY, "retry", Some(Capacity), false, StorageOutcome::Capacity),
            declared(Self::UNAVAILABLE, "unavailable", Some(Unavailable), false, Server),
            declared(
                Self::SNAPSHOT_EXPIRED,
                "snapshot_expired",
                Some(SnapshotExpired),
                false,
                Server,
            ),
            declared(
                Self::COMMIT_UNKNOWN_RESULT,
                "commit_unknown_result",
                Some(Unavailable),
                true,
                Server,
            ),
            declared(Self::CANCELLED, "cancelled", None, false, Client),
            declared(Self::TIMED_OUT, "timed_out", None, false, Server),
            declared(Self::INVALID_ARGUMENT, "invalid_argument", None, false, Client),
            declared(Self::SQL_ERROR, "sql_error", None, false, Client),
            declared(Self::TOO_LARGE, "too_large", None, false, Client),
            declared(Self::NOT_FOUND, "not_found", None, false, Client),
            declared(Self::INTERNAL, "internal", None, true, Server),
            declared(Self::PERMISSION_DENIED, "permission_denied", None, false, Client),
            declared(Self::UNAUTHENTICATED, "unauthenticated", None, false, Client),
        ]
    };

    fn declared(self) -> Option<&'static Declared> {
        Self::TABLE.iter().find(|d| d.code == self)
    }

    /// Whether this build knows the code.
    pub fn known(self) -> bool {
        self.declared().is_some()
    }

    /// Whether the same request, sent again, may succeed; false for a code this build does not
    /// know.
    pub fn retryable(self) -> bool {
        self.retry().is_some()
    }

    /// Why the same request, sent again, may succeed; `None` when it may not, or for a code this
    /// build does not know.
    pub fn retry(self) -> Option<StorageRetryReason> {
        self.declared().and_then(|d| d.retry)
    }

    /// Who owns a request's outcome answered with this code; `Unknown` for a code this build does
    /// not know.
    pub fn class(self) -> StorageOutcome {
        self.declared().map_or(StorageOutcome::Unknown, |d| d.class)
    }

    /// Whether a request answered with this code may have taken effect; true for a code this
    /// build does not know.
    pub fn maybe_applied(self) -> bool {
        self.declared().is_none_or(|d| d.maybe_applied)
    }

    /// The code's name, `None` for a code this build does not know.
    pub fn name(self) -> Option<&'static str> {
        self.declared().map(|d| d.name)
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.name() {
            Some(name) => f.write_str(name),
            None => write!(f, "unknown_{}", self.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_code_is_declared_once_by_number_and_by_name() {
        for (i, d) in ErrorCode::TABLE.iter().enumerate() {
            for other in &ErrorCode::TABLE[i + 1..] {
                assert_ne!(d.code, other.code, "{} twice", d.code.0);
                assert_ne!(d.name, other.name, "{} twice", d.name);
            }
        }
    }

    /// Catches: `unavailable` or `snapshot_expired` read as unknown (maybe applied, final), so a
    /// store outage or a long read failed its write `Uncertain` instead of being sent again.
    #[test]
    fn unavailable_and_snapshot_expired_applied_nothing_and_are_sent_again() {
        for code in [ErrorCode::UNAVAILABLE, ErrorCode::SNAPSHOT_EXPIRED] {
            assert!(code.known() && code.retryable() && !code.maybe_applied(), "{code}");
        }
        assert_eq!(ErrorCode(1004).to_string(), "unavailable");
        assert_eq!(ErrorCode(1007).to_string(), "snapshot_expired");
    }

    /// Catches: an unknown code read as nothing applied or as worth sending again.
    #[test]
    fn an_unknown_code_may_have_applied_and_is_final() {
        let unknown = ErrorCode(9999);
        assert!(!unknown.known() && !unknown.retryable() && unknown.maybe_applied());
        assert_eq!(unknown.to_string(), "unknown_9999");
    }
}
