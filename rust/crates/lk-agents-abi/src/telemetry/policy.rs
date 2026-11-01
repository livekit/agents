//! Destination privacy and admission limits and records; the decisions are
//! `lk_agents_core::telemetry::policy`'s. The limits reach the host as generated constants.

/// Nested SDK values deeper than this are refused.
#[crate::host(export(name = "TELEMETRY_MAX_DEPTH"))]
pub const MAX_DEPTH: usize = 4;

/// SDK mappings and sequences longer than this are refused.
pub const MAX_COLLECTION: usize = 128;

/// Spans with more events or links than this are refused.
#[crate::host(export(name = "TELEMETRY_MAX_EVENTS"))]
pub const MAX_EVENTS: usize = 32;

/// Retained bytes one SDK record may charge.
#[crate::host(export(name = "SDK_RECORD_BYTES"))]
pub const SDK_RECORD_BYTES: usize = 64 * 1024;

/// Records one framework export queue holds.
#[crate::host(export(name = "SDK_PENDING_RECORDS"))]
pub const SDK_PENDING_RECORDS: usize = 64;

#[crate::host(export(name = "SDK_BATCH_ITEMS"))]
pub const SDK_BATCH_ITEMS: usize = 16;

#[crate::host(export(name = "SDK_SCHEDULE_DELAY_MILLIS"))]
pub const SDK_SCHEDULE_DELAY_MILLIS: usize = 1000;

#[crate::host(export(name = "SDK_EXPORT_TIMEOUT_MILLIS"))]
pub const SDK_EXPORT_TIMEOUT_MILLIS: usize = 5000;

/// Framework Cloud span pipelines one process keeps.
#[crate::host(export(name = "SDK_MAX_PROVIDERS"))]
pub const SDK_MAX_PROVIDERS: usize = 4;

#[crate::host(export(name = "SDK_SHUTDOWN_TIMEOUT_SECONDS"))]
pub const SDK_SHUTDOWN_TIMEOUT_SECONDS: f64 = 10.0;

#[crate::host(export(name = "UPLOAD_TIMEOUT_SECONDS"))]
pub const UPLOAD_TIMEOUT_SECONDS: u64 = 900;

/// What a redacted destination sees in place of an exception message or error status.
#[crate::host(export(name = "REDACTED_EXCEPTION"))]
pub const REDACTED_EXCEPTION: &str = "exception details redacted";

/// What a redacted destination sees in place of a log body that may carry content.
#[crate::host(export(name = "REDACTED_LOG_BODY"))]
pub const REDACTED_LOG_BODY: &str = "log body redacted";

pub const REPORT_EVENT_ITEMS: usize = 1024;
pub const REPORT_EVENT_BYTES: usize = 1024 * 1024;
pub const REPORT_EVENT_RECORD_BYTES: usize = 64 * 1024;
pub const REPORT_RETAINED_BYTES: usize = 4 * 1024 * 1024;

/// An OpenTelemetry `AnyValue` as the host SDK holds it, read once from the host object: a
/// mapping's non-string keys are skipped, a host record (a pydantic model, a dataclass) reads as
/// the mapping of its fields.
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
    /// Admission refuses it; a snapshot drops it.
    Unsupported,
}

#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryRecordKind {
    Span,
    Log,
    Event,
    Link,
    Scope,
    Value,
}

#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryExportAction {
    Export,
    Hold,
    Drop,
    Capacity,
    Disabled,
}

/// One destination's content policy: what its snapshot keeps, drops or replaces.
#[crate::host(class(name = "_core_TelemetryPolicy"))]
pub trait TelemetryPolicy: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(allow_pii: bool, capture_content: bool, project_redaction: bool) -> Self;

    #[host(getter)]
    fn redacted(&self) -> bool;

    /// This destination's copy of `value`: fields it may not see dropped or replaced, at every
    /// depth. `destination=False` copies a value into the framework's own queue, keeping the
    /// destination-private markers.
    #[host(call)]
    fn snapshot(&self, value: TelemetryValue, destination: bool) -> TelemetryValue;

    #[host(call)]
    fn keep_event(&self, name: String) -> bool;

    /// A replacement body, or None to snapshot the original SDK value.
    #[host(call)]
    fn log_body(&self, safe_body: bool, body: Option<String>) -> Option<String>;

    #[host(call)]
    fn status_description(&self, is_error: bool, description: Option<String>) -> Option<String>;
}

/// SDK retention charges and Cloud export admission, which no destination's consent changes.
#[crate::host(functions)]
pub trait TelemetryAdmission {
    /// The destination may export, hold, drop, refuse capacity or stop after recording is disabled.
    #[host(function)]
    fn _core_export_action(
        registered: bool,
        signal_enabled: bool,
        job_open: bool,
        held_count: usize,
        recording_disabled: bool,
    ) -> TelemetryExportAction;

    /// Conservative retained bytes of an SDK value; more than `SDK_RECORD_BYTES` refuses it.
    #[host(function)]
    fn _core_value_bytes(value: TelemetryValue) -> usize;

    /// SDK record overhead plus field charges. Counts are supplied before traversing events/links.
    #[host(function)]
    fn _core_record_bytes(
        kind: TelemetryRecordKind,
        fields: Vec<usize>,
        event_count: usize,
        link_count: usize,
    ) -> usize;
}

/// The framework's metric series. Subject folds decide measurements; this refuses unbounded labels
/// and series before the host SDK creates a new one.
#[crate::host(class(name = "_core_MetricSeries"))]
pub trait MetricSeries: Send + Sized + 'static {
    #[host(new)]
    fn new() -> Self;

    #[host(call)]
    fn admit(&mut self, name: String, attributes: Vec<(String, String)>) -> bool;
}
