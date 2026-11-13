#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsdbCode {
    Protocol,
    Unauthenticated,
    Unavailable,
    NotFound,
    Internal,
    ResourceExhausted,
    Txn,
    Exec,
    Query,
    /// Nothing was applied; the same request may be sent again.
    Retry,
    /// A write that may or may not have applied, and takes effect whole if it did.
    OutcomeUnknown,
}
