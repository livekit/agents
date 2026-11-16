//! agent-db's error codes. The wire carries a code's number, its name and a description; a
//! client branches on the number alone, through [`ErrorCode::retryable`] and
//! [`ErrorCode::maybe_applied`], never on the text. A code this build does not know may have
//! applied and is never retried. The numbers group by kind: 1xxx the service could not serve
//! the request now, 2xxx the request itself is refused, 4xxx a broken invariant, 6xxx the
//! caller's credentials.

/// An error's number on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorCode(pub u32);

/// A known code: its name, whether the same request sent again may succeed, and whether a
/// request answered with it may have taken effect, whole.
struct Declared {
    code: ErrorCode,
    name: &'static str,
    retryable: bool,
    maybe_applied: bool,
}

const fn declared(
    code: ErrorCode,
    name: &'static str,
    retryable: bool,
    maybe_applied: bool,
) -> Declared {
    Declared { code, name, retryable, maybe_applied }
}

impl ErrorCode {
    /// This server does not serve the database (evicted, draining or moved): redial. On request
    /// id 0, the socket ends now.
    pub const WRONG_SHARD_SERVER: Self = Self(1001);
    /// On request id 0: the database was evicted while the socket idled, and the socket ends.
    /// Redial when next needed.
    pub const IDLE_EVICTED: Self = Self(1002);
    /// Refused for pressure that passes: send it again after a backoff.
    pub const RETRY: Self = Self(1003);
    /// The request may have committed, whole, or not at all.
    pub const COMMIT_UNKNOWN_RESULT: Self = Self(1021);
    pub const INVALID_ARGUMENT: Self = Self(2001);
    pub const SQL_ERROR: Self = Self(2002);
    pub const TOO_LARGE: Self = Self(2003);
    pub const NOT_FOUND: Self = Self(2004);
    /// An invariant broke: the request may have applied.
    pub const INTERNAL: Self = Self(4100);
    pub const PERMISSION_DENIED: Self = Self(6000);
    pub const UNAUTHENTICATED: Self = Self(6001);

    const TABLE: [Declared; 11] = [
        declared(Self::WRONG_SHARD_SERVER, "wrong_shard_server", true, false),
        declared(Self::IDLE_EVICTED, "idle_evicted", true, false),
        declared(Self::RETRY, "retry", true, false),
        declared(Self::COMMIT_UNKNOWN_RESULT, "commit_unknown_result", true, true),
        declared(Self::INVALID_ARGUMENT, "invalid_argument", false, false),
        declared(Self::SQL_ERROR, "sql_error", false, false),
        declared(Self::TOO_LARGE, "too_large", false, false),
        declared(Self::NOT_FOUND, "not_found", false, false),
        declared(Self::INTERNAL, "internal", false, true),
        declared(Self::PERMISSION_DENIED, "permission_denied", false, false),
        declared(Self::UNAUTHENTICATED, "unauthenticated", false, false),
    ];

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
        self.declared().is_some_and(|d| d.retryable)
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

    /// Catches: an unknown code read as nothing applied or as worth sending again.
    #[test]
    fn an_unknown_code_may_have_applied_and_is_final() {
        let unknown = ErrorCode(9999);
        assert!(!unknown.known() && !unknown.retryable() && unknown.maybe_applied());
        assert_eq!(unknown.to_string(), "unknown_9999");
    }
}
