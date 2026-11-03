//! The host's OpenTelemetry API, recorded in Rust (docs/internals/telemetry.md "What Python
//! keeps"): the span the host's tracer starts, and the process's telemetry configuration.

use std::collections::BTreeMap;

use serde_json::Value;

/// A span started through the host's OpenTelemetry API (an application span, or a framework span
/// around host-only code). Ended once; recorded by the process's exporter.
#[crate::host(class(name = "TelemetrySpan"))]
pub trait HostSpan: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class: its tracer.
    #[host(env)]
    type Env;

    /// `parent`: `(trace_id, span_id, sampled)` of the current span, `None` for a new trace.
    /// `kind`: the OpenTelemetry `SpanKind` value. `ids`: `(trace_id, span_id)` of a span the
    /// host already started (an SDK span the bridge forwards). `job`: the job it belongs to.
    #[host(new)]
    #[allow(clippy::too_many_arguments, reason = "Preserve the public Python signature")]
    fn new(
        env: &Self::Env,
        name: String,
        scope: String,
        #[host(kw, default = None)] parent: Option<(u128, u64, bool)>,
        #[host(kw, default = 0)] kind: i64,
        #[host(kw, default = None)] start_time: Option<u64>,
        #[host(kw, default = None)] ids: Option<(u128, u64)>,
        #[host(kw, default = None)] job: Option<String>,
    ) -> Self;

    /// `(trace_id, span_id, sampled)`.
    #[host(getter)]
    fn context(&self) -> (u128, u64, bool);

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

    /// `code`: the OpenTelemetry `StatusCode` value.
    #[host(call)]
    fn set_status(&self, code: i64, #[host(default = None)] description: Option<String>);

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
        #[host(kw, default = None)] protocol: Option<String>,
        #[host(kw, default = None)] metadata: Option<BTreeMap<String, Value>>,
        #[host(kw, default = None)] allow_pii: Option<bool>,
    ) -> Result<(), TelemetryConfigError>;

    /// The export counters: `dropped`, `failed`, `conflicts`, `incomplete_cleanup`.
    #[host(function)]
    fn _core_telemetry_diagnostics() -> BTreeMap<String, u64>;

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
    /// measurement, and an `event_loop_blocked` span under `parent` (`(trace_id, span_id,
    /// sampled)`), at most 30 a minute. `task`: the task that was running.
    #[host(function)]
    fn _core_event_loop_blocked(
        duration: f64,
        #[host(kw, default = None)] parent: Option<(u128, u64, bool)>,
        #[host(kw, default = None)] job: Option<String>,
        #[host(kw, default = None)] task: Option<String>,
    );
}

/// A `configure` argument out of its domain.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct TelemetryConfigError(pub String);
