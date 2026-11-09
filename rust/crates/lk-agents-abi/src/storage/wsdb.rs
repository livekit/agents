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
}
