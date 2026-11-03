//! Telemetry limits and the host value a session report's event retention charges; the decisions
//! are `lk_agents_core::telemetry::policy`'s.

/// Nested host values deeper than this are refused.
#[crate::host(export(name = "TELEMETRY_MAX_DEPTH"))]
pub const MAX_DEPTH: usize = 4;

/// Host mappings and sequences longer than this are refused.
pub const MAX_COLLECTION: usize = 128;

/// The bound on a job's final telemetry flush.
pub const SHUTDOWN_TIMEOUT_SECONDS: f64 = 10.0;

pub const UPLOAD_TIMEOUT_SECONDS: u64 = 900;

/// What a redacted destination sees in place of an exception message or error status.
pub const REDACTED_EXCEPTION: &str = "exception details redacted";

pub const REPORT_EVENT_ITEMS: usize = 1024;
pub const REPORT_EVENT_BYTES: usize = 1024 * 1024;
/// Retained bytes one event may charge.
pub const REPORT_EVENT_RECORD_BYTES: usize = 64 * 1024;

/// A host value as OpenTelemetry's `AnyValue`, read once from the host object: a mapping's
/// non-string keys are skipped, a host record (a pydantic model, a dataclass) reads as the
/// mapping of its fields.
#[crate::host(codec(extract, emit, host = "object", py = "py_values::telemetry_value"))]
#[derive(Debug, Clone, PartialEq)]
pub enum TelemetryValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    Array(Vec<TelemetryValue>),
    Map(Vec<(String, TelemetryValue)>),
    /// What OTel cannot carry: another host object, an integer beyond 64 bits, or a collection
    /// nested deeper than [`MAX_DEPTH`] or longer than [`MAX_COLLECTION`], which is not walked.
    /// Retention refuses it.
    Unsupported,
}
