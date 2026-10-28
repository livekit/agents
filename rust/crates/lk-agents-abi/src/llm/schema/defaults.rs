//! Schema default validation errors.
use crate::llm::provider_format::StrictError;
pub const MAX_DEPTH: usize = 100;

/// Everything the defaults walk can refuse. A wrapped strict fault keeps its own kind.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SchemaError {
    /// A strict-schema normalization fault.
    #[error(transparent)]
    #[host(delegate)]
    Strict(#[from] StrictError),
    #[error("expected a JSON object for {what}")]
    #[host(kind = "TypeError")]
    NotAnObject { what: &'static str },
    /// The reference's `resolve_ref` raises `ValueError` here.
    #[error("unexpected $ref format {reference:?}; does not start with #/")]
    RefNotRooted { reference: String },
    /// The reference's `KeyError` (missing segment) and `AssertionError` (not a dict).
    #[error("$ref {reference:?} does not resolve: no object at segment {segment:?}")]
    #[host(kind = "KeyError")]
    RefUnresolvable { reference: String, segment: String },
    #[error("value or schema nests deeper than {MAX_DEPTH} levels")]
    TooDeep,
}
