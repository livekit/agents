//! GENERATED FILE -- DO NOT EDIT.
//! `livekit.AgentDB.ErrorCode` (`livekit_agentdb.proto`), written by `lk-agents-codegen`: every agent-db error code, on the data plane and the management services, with what a client may conclude from it.

/// Whether a request answered with an error may have taken effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applied {
    /// It may have committed, whole: resend only under its idempotency key.
    Maybe,
    /// Nothing took effect.
    No,
}

/// Whether the same request can succeed if sent again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retry {
    /// The same request fails the same way.
    Never,
    /// On the same connection, after a backoff.
    Here,
    /// On a new connection, which may reach another node.
    Elsewhere,
}

/// An agent-db error code. It travels as its name in lowercase; one this build does not know is `Unknown`, which may have applied and is never retried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    Unknown,
    Canceled,
    InvalidArgument,
    DeadlineExceeded,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated,
    Protocol,
    Exec,
    Query,
    Txn,
    Retry,
    OutcomeUnknown,
    IdleEvicted,
    TxnAborted,
}

impl ErrorCode {
    /// The code as it travels.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Canceled => "canceled",
            Self::InvalidArgument => "invalid_argument",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::NotFound => "not_found",
            Self::AlreadyExists => "already_exists",
            Self::PermissionDenied => "permission_denied",
            Self::ResourceExhausted => "resource_exhausted",
            Self::FailedPrecondition => "failed_precondition",
            Self::Aborted => "aborted",
            Self::OutOfRange => "out_of_range",
            Self::Unimplemented => "unimplemented",
            Self::Internal => "internal",
            Self::Unavailable => "unavailable",
            Self::DataLoss => "data_loss",
            Self::Unauthenticated => "unauthenticated",
            Self::Protocol => "protocol",
            Self::Exec => "exec",
            Self::Query => "query",
            Self::Txn => "txn",
            Self::Retry => "retry",
            Self::OutcomeUnknown => "outcome_unknown",
            Self::IdleEvicted => "idle_evicted",
            Self::TxnAborted => "txn_aborted",
        }
    }

    /// A code as it travels; anything else is `Unknown`.
    pub fn from_wire(code: &str) -> Self {
        match code {
            "canceled" => Self::Canceled,
            "invalid_argument" => Self::InvalidArgument,
            "deadline_exceeded" => Self::DeadlineExceeded,
            "not_found" => Self::NotFound,
            "already_exists" => Self::AlreadyExists,
            "permission_denied" => Self::PermissionDenied,
            "resource_exhausted" => Self::ResourceExhausted,
            "failed_precondition" => Self::FailedPrecondition,
            "aborted" => Self::Aborted,
            "out_of_range" => Self::OutOfRange,
            "unimplemented" => Self::Unimplemented,
            "internal" => Self::Internal,
            "unavailable" => Self::Unavailable,
            "data_loss" => Self::DataLoss,
            "unauthenticated" => Self::Unauthenticated,
            "protocol" => Self::Protocol,
            "exec" => Self::Exec,
            "query" => Self::Query,
            "txn" => Self::Txn,
            "retry" => Self::Retry,
            "outcome_unknown" => Self::OutcomeUnknown,
            "idle_evicted" => Self::IdleEvicted,
            "txn_aborted" => Self::TxnAborted,
            _ => Self::Unknown,
        }
    }

    /// Whether a request answered with this code may have taken effect.
    pub const fn applied(self) -> Applied {
        match self {
            Self::Unknown => Applied::Maybe,
            Self::Canceled => Applied::Maybe,
            Self::InvalidArgument => Applied::No,
            Self::DeadlineExceeded => Applied::Maybe,
            Self::NotFound => Applied::No,
            Self::AlreadyExists => Applied::No,
            Self::PermissionDenied => Applied::No,
            Self::ResourceExhausted => Applied::No,
            Self::FailedPrecondition => Applied::No,
            Self::Aborted => Applied::No,
            Self::OutOfRange => Applied::No,
            Self::Unimplemented => Applied::No,
            Self::Internal => Applied::Maybe,
            Self::Unavailable => Applied::Maybe,
            Self::DataLoss => Applied::Maybe,
            Self::Unauthenticated => Applied::No,
            Self::Protocol => Applied::No,
            Self::Exec => Applied::No,
            Self::Query => Applied::No,
            Self::Txn => Applied::No,
            Self::Retry => Applied::No,
            Self::OutcomeUnknown => Applied::Maybe,
            Self::IdleEvicted => Applied::No,
            Self::TxnAborted => Applied::No,
        }
    }

    /// Whether the same request may succeed if sent again, and where.
    pub const fn retry(self) -> Retry {
        match self {
            Self::Unknown => Retry::Never,
            Self::Canceled => Retry::Here,
            Self::InvalidArgument => Retry::Never,
            Self::DeadlineExceeded => Retry::Here,
            Self::NotFound => Retry::Never,
            Self::AlreadyExists => Retry::Never,
            Self::PermissionDenied => Retry::Never,
            Self::ResourceExhausted => Retry::Here,
            Self::FailedPrecondition => Retry::Never,
            Self::Aborted => Retry::Here,
            Self::OutOfRange => Retry::Never,
            Self::Unimplemented => Retry::Never,
            Self::Internal => Retry::Never,
            Self::Unavailable => Retry::Elsewhere,
            Self::DataLoss => Retry::Never,
            Self::Unauthenticated => Retry::Never,
            Self::Protocol => Retry::Never,
            Self::Exec => Retry::Never,
            Self::Query => Retry::Never,
            Self::Txn => Retry::Never,
            Self::Retry => Retry::Here,
            Self::OutcomeUnknown => Retry::Here,
            Self::IdleEvicted => Retry::Elsewhere,
            Self::TxnAborted => Retry::Here,
        }
    }
}
