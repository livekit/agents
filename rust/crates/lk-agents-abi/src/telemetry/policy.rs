//! Destination privacy and admission decisions over host SDK field facts.

pub const SDK_RECORD_BYTES: usize = 64 * 1024;
pub const REPORT_EVENT_ITEMS: usize = 1024;
pub const REPORT_EVENT_BYTES: usize = 1024 * 1024;
pub const REPORT_EVENT_RECORD_BYTES: usize = 64 * 1024;
pub const SDK_PENDING_RECORDS: usize = 64;
pub const SDK_BATCH_ITEMS: usize = 16;
pub const SDK_SCHEDULE_DELAY_MILLIS: usize = 1000;
pub const SDK_EXPORT_TIMEOUT_MILLIS: usize = 5000;
pub const SDK_MAX_PROVIDERS: usize = 4;
pub const SDK_SHUTDOWN_TIMEOUT_SECONDS: f64 = 10.0;
pub const REPORT_RETAINED_BYTES: usize = 4 * 1024 * 1024;
pub const UPLOAD_TIMEOUT_SECONDS: u64 = 900;

pub fn report_body_is_safe(body: &str) -> bool {
    matches!(
        body,
        "session report" | "session event" | "chat item" | "evaluation" | "tag" | "outcome"
    )
}

pub fn content_allowed(capture_content: bool, redaction: bool) -> bool {
    capture_content && !redaction
}

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

    #[host(getter)]
    fn max_depth(&self) -> usize;

    #[host(getter)]
    fn max_collection(&self) -> usize;

    #[host(getter)]
    fn max_events(&self) -> usize;

    #[host(getter)]
    fn max_record_bytes(&self) -> usize;

    #[host(getter)]
    fn max_pending(&self) -> usize;

    #[host(getter)]
    fn max_records(&self) -> usize;

    #[host(getter)]
    fn batch_items(&self) -> usize;

    #[host(getter)]
    fn schedule_delay_millis(&self) -> usize;

    #[host(getter)]
    fn export_timeout_millis(&self) -> usize;

    #[host(getter)]
    fn max_providers(&self) -> usize;

    #[host(getter)]
    fn shutdown_timeout_seconds(&self) -> f64;

    #[host(getter)]
    fn report_retained_bytes(&self) -> usize;

    #[host(getter)]
    fn upload_timeout_seconds(&self) -> f64;

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

    #[host(call)]
    fn is_pii_attribute(&self, key: String) -> bool;

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
