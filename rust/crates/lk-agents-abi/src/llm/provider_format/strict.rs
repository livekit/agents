//! Strict schema error contract.
pub const MAX_DEPTH: usize = 100;

/// A strict-schema fault; the kind table picks the host exception. Plugins catch the reference's
/// exceptions per site; `AssertionError` (a non-dict `resolve_ref` segment) folds into `KeyError`.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StrictError {
    #[error("expected a JSON object at {path}")]
    #[host(kind = "TypeError")]
    NotAnObject { path: String },
    #[error("non-string $ref at {path}")]
    NonStringRef { path: String },
    #[error("unexpected $ref format {reference:?}; does not start with #/")]
    RefNotRooted { reference: String },
    #[error("$ref {reference:?} does not resolve: no object at segment {segment:?}")]
    #[host(kind = "KeyError")]
    RefUnresolvable { reference: String, segment: String },
    #[error("nullable union at {path} has no non-null object branch")]
    NullableUnionBranch { path: String },
    #[error("schema nests deeper than {MAX_DEPTH} levels at {path}; is a $ref self-referential?")]
    TooDeep { path: String },
}
