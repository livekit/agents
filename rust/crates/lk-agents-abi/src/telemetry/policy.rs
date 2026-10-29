//! Destination privacy and admission limits and records; the decisions are
//! `lk_agents_core::telemetry::policy`'s. The limits reach the host as generated constants.

/// Nested SDK values deeper than this are refused.
#[crate::host(export(name = "TELEMETRY_MAX_DEPTH"))]
pub const MAX_DEPTH: usize = 4;

/// SDK mappings and sequences longer than this are refused.
#[crate::host(export(name = "TELEMETRY_MAX_COLLECTION"))]
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

#[crate::host(record(module = "telemetry", extract, emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryValueKind {
    Scalar,
    Integer,
    String,
    Bytes,
    Mapping,
    Sequence,
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

/// The host supplies SDK fields and lengths; shared policy decides what may be retained/exported.
#[crate::host(class(name = "_core_TelemetryPolicy"))]
pub trait TelemetryPolicy: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(allow_pii: bool, capture_content: bool, project_redaction: bool) -> Self;

    #[host(getter)]
    fn redacted(&self) -> bool;

    /// The destination may export, hold, drop, refuse capacity or stop after recording is disabled.
    #[host(call)]
    fn export_action(
        &self,
        registered: bool,
        signal_enabled: bool,
        job_open: bool,
        held_count: usize,
        recording_disabled: bool,
    ) -> TelemetryExportAction;

    /// Whether to keep the field and, if present, the replacement for its value.
    #[host(call)]
    fn attribute_action(
        &self,
        parent: Option<String>,
        key: String,
        destination: bool,
    ) -> (bool, Option<String>);

    #[host(call)]
    fn keep_event(&self, name: String) -> bool;

    /// A replacement body, or None to snapshot the original SDK value.
    #[host(call)]
    fn log_body(&self, safe_body: bool, body: Option<String>) -> Option<String>;

    #[host(call)]
    fn status_description(&self, is_error: bool, description: Option<String>) -> Option<String>;

    /// Conservative retained bytes for an SDK scalar or collection. Child charges are already
    /// bounded verdicts; unsupported kinds, depth and collection overflow refuse the record.
    #[host(call)]
    fn value_bytes(
        &self,
        kind: TelemetryValueKind,
        length: usize,
        depth: usize,
        children: Vec<usize>,
    ) -> usize;

    /// SDK record overhead plus field charges. Counts are supplied before traversing events/links.
    #[host(call)]
    fn record_bytes(
        &self,
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
