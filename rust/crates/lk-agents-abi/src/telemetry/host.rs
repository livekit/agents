//! The host's OpenTelemetry API, recorded in Rust (docs/internals/telemetry.md "What Python
//! keeps"): the span the host's tracer starts, and the process's telemetry configuration.

use std::collections::BTreeMap;

use serde_json::Value;

use super::record::{SpanContext, SpanKind};

/// A span started through the host's OpenTelemetry API (an application span, or a framework span
/// around host-only code). Ended once; recorded by the process's exporter.
#[crate::host(class(name = "TelemetrySpan"))]
pub trait HostSpan: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class: its tracer.
    #[host(env)]
    type Env;

    /// `parent`: the current span, `None` for a new trace. `ids`: a span the host already started (an SDK span the bridge forwards). `job`: the
    /// job it belongs to.
    #[host(new)]
    #[allow(clippy::too_many_arguments, reason = "Preserve the public Python signature")]
    fn new(
        env: &Self::Env,
        name: String,
        scope: String,
        #[host(kw, default = None)] parent: Option<SpanContext>,
        #[host(kw, default = None)] kind: Option<SpanKind>,
        #[host(kw, default = None)] start_time: Option<u64>,
        #[host(kw, default = None)] ids: Option<SpanContext>,
        #[host(kw, default = None)] job: Option<String>,
    ) -> Self;

    #[host(getter)]
    fn context(&self) -> SpanContext;

    /// Whether the span has not ended.
    #[host(call)]
    fn is_recording(&self) -> bool;

    /// A JSON scalar or array; `None` removes nothing and is ignored.
    #[host(call)]
    fn set_attribute(&self, key: String, value: Value);

    #[host(call)]
    fn add_event(
        &self,
        name: String,
        #[host(default = None)] attributes: Option<BTreeMap<String, Value>>,
        #[host(default = None)] timestamp: Option<u64>,
    );

    /// `description` is kept only with `Error`; `Ok` is final.
    #[host(call)]
    fn set_status(&self, code: StatusCode, #[host(default = None)] description: Option<String>);

    #[host(call)]
    fn update_name(&self, name: String);

    /// Ends the span; later calls do nothing.
    #[host(call)]
    fn end(&self, #[host(default = None)] end_time: Option<u64>);
}

/// The process's telemetry: the application endpoint, message capture and the export counters.
#[crate::host(functions(pid = false))]
pub trait Telemetry {
    /// Starts the process's exporter, stamping `sdk_version`, with the application endpoint from
    /// the `OTEL_EXPORTER_OTLP_*` environment. Later calls do nothing.
    #[host(function)]
    fn _core_telemetry_install(sdk_version: String);

    /// Replaces the application endpoint; arguments override the environment.
    #[host(function)]
    fn _core_telemetry_configure(
        #[host(kw, default = None)] endpoint: Option<String>,
        #[host(kw, default = None)] headers: Option<BTreeMap<String, String>>,
        #[host(kw, default = None)] metadata: Option<BTreeMap<String, Value>>,
        #[host(kw, default = None)] allow_pii: Option<bool>,
    ) -> Result<(), TelemetryConfigError>;

    #[host(function)]
    fn _core_telemetry_diagnostics() -> TelemetryDiagnostics;

    /// Counts a configuration conflict the host found (a tracer provider it cannot bridge).
    #[host(function)]
    fn _core_telemetry_conflict();

    /// GenAI message capture for spans recorded afterwards.
    #[host(function)]
    fn _core_set_capture_content(enabled: bool);

    /// The process-exit flush: every destination's queue, bounded.
    #[host(function)]
    async fn _core_telemetry_shutdown();

    /// A stall of the host's event loop, `duration` seconds long, reported once it ended: its
    /// measurement, and an `event_loop_blocked` span under `parent`, at most 30 a minute. `task`:
    /// the task that was running.
    #[host(function)]
    fn _core_event_loop_blocked(
        duration: f64,
        #[host(kw, default = None)] parent: Option<SpanContext>,
        #[host(kw, default = None)] job: Option<String>,
        #[host(kw, default = None)] task: Option<String>,
    );
}

/// OpenTelemetry's `StatusCode`.
#[crate::host(native_enum(name = "StatusCode"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusCode {
    Unset,
    Ok,
    Error,
}

/// The process's export counters.
#[crate::host(record(module = "telemetry", emit))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TelemetryDiagnostics {
    /// Records refused by a full queue or a metric cardinality limit, or abandoned at a deadline.
    pub dropped: u64,
    /// Records in requests the endpoint refused or that failed in transport.
    pub failed: u64,
    /// Configuration that could not apply.
    pub conflicts: u64,
    /// Job closes and shutdowns that reached their deadline before every record was sent.
    pub incomplete_cleanup: u64,
    /// Spans a voice session's close ended because their owner never did (a framework bug).
    pub swept_spans: u64,
    /// Voice sessions dropped without their close; their open spans were ended and exported.
    pub abandoned_sessions: u64,
}

/// A `configure` argument out of its domain.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct TelemetryConfigError(pub String);
