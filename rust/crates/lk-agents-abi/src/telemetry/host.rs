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

    /// `None` removes nothing and is ignored.
    #[host(call)]
    fn set_attribute(&self, key: String, value: AttributeValue);

    /// Each attribute, as `set_attribute` sets it.
    #[host(call)]
    fn set_attributes(&self, attributes: Attributes);

    #[host(call)]
    fn add_event(
        &self,
        name: String,
        #[host(default = None)] attributes: Option<Attributes>,
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

/// `telemetry._Span`: the host's OpenTelemetry span over a native one.
#[crate::host(facade(
    path = "livekit.agents.telemetry",
    name = "_Span",
    base = "opentelemetry.trace.Span",
    foreign,
    native = HostSpan,
    from_native,
    extended
))]
pub trait SpanFacade {
    #[host(native = context)]
    fn get_span_context(&self) -> SpanContext;
    fn end(&self, #[host(default = None)] end_time: Option<u64>);
    fn set_attributes(&self, attributes: Attributes);
    fn set_attribute(&self, key: String, value: AttributeValue);
    fn add_event(
        &self,
        name: String,
        #[host(default = None)] attributes: Option<Attributes>,
        #[host(default = None)] timestamp: Option<u64>,
    );
    fn update_name(&self, name: String);
    fn is_recording(&self) -> bool;
}

/// The process's telemetry: the application endpoint, message capture and the export counters.
#[crate::host(functions(pid = false))]
pub trait Telemetry {
    /// Starts the process's exporter, stamping `sdk_version`, with the application endpoint from
    /// the `OTEL_EXPORTER_OTLP_*` environment. Later calls do nothing.
    #[host(function)]
    fn _core_telemetry_install(sdk_version: String);

    /// Send the process's spans and metrics to an OTLP/HTTP protobuf endpoint; arguments override
    /// the `OTEL_EXPORTER_OTLP_*` environment.
    #[host(function, name = "configure", path = "livekit.agents.telemetry")]
    fn _core_telemetry_configure(
        #[host(kw, default = None)] endpoint: Option<String>,
        #[host(kw, default = None)] headers: Option<BTreeMap<String, String>>,
        #[host(kw, default = None)] metadata: Option<BTreeMap<String, Value>>,
        #[host(kw, default = None)] allow_pii: Option<bool>,
    ) -> Result<(), TelemetryConfigError>;

    /// The process's export counters.
    #[host(function, name = "diagnostics", path = "livekit.agents.telemetry")]
    fn _core_telemetry_diagnostics() -> TelemetryDiagnostics;

    /// Counts a configuration conflict the host found (a tracer provider it cannot bridge).
    #[host(function)]
    fn _core_telemetry_conflict();

    /// Records a span the application's OpenTelemetry SDK ended (the bridge of an SDK provider),
    /// counted on `job`'s sessions.
    #[host(function)]
    fn _core_telemetry_export(span: EndedSpan, #[host(kw, default = None)] job: Option<String>);

    /// Whether model request and response messages and judge explanations are recorded.
    #[host(function, name = "set_capture_content", path = "livekit.agents.telemetry")]
    fn _core_set_capture_content(enabled: bool);

    /// The process-exit flush: every destination's queue, bounded.
    #[host(function)]
    async fn _core_telemetry_shutdown();
}

/// One asyncio loop's stall watch: the loop's
/// own timer beats it every 20 ms; the process's detector records a beat 100 ms old as a stall,
/// with the loop thread's stack where it can be read. Beats take no lock.
#[crate::host(class(name = "_core_LoopWatch"))]
pub trait LoopWatch: Send + Sync + Sized + 'static {
    /// Watches the loop running on OS thread `os_thread` (`threading.get_native_id()`); its
    /// stalls go under `parent` and count on `job`'s sessions. `whole_process`: the process runs
    /// this job alone (the process executor), so its other running Python threads join the job's
    /// profile; otherwise the profile holds the loop thread alone.
    #[host(new)]
    fn new(
        os_thread: u64,
        #[host(kw, default = None)] parent: Option<SpanContext>,
        #[host(kw, default = None)] job: Option<String>,
        #[host(kw, default = false)] whole_process: bool,
    ) -> Self;

    /// The loop ran.
    #[host(call)]
    fn beat(&self);

    /// The loop is done; a stall in progress is recorded.
    #[host(call)]
    fn close(&self);
}

/// An OpenTelemetry attribute value: a scalar, or a sequence of scalars (a tuple too).
#[crate::host(codec(
    extract,
    host = "opentelemetry.util.types.AttributeValue",
    py = "py_values::attribute_value"
))]
pub type AttributeValue = Value;

/// OpenTelemetry attributes: any mapping of keys to attribute values.
#[crate::host(codec(
    extract,
    host = "collections.abc.Mapping[str, opentelemetry.util.types.AttributeValue]",
    py = "py_values::attributes"
))]
pub type Attributes = BTreeMap<String, Value>;

/// A span an application's OpenTelemetry SDK ended.
#[crate::host(record(
    module = "telemetry",
    host_type = "opentelemetry.sdk.trace.ReadableSpan",
    boundary,
    extract
))]
#[derive(Debug, Clone)]
pub struct EndedSpan {
    pub name: String,
    #[host(path = "instrumentation_scope.name", missing_default)]
    pub scope: String,
    #[host(call = "get_span_context")]
    pub context: Option<SpanContext>,
    pub parent: Option<SpanContext>,
    pub kind: SpanKind,
    pub start_time: Option<u64>,
    pub end_time: Option<u64>,
    #[host(missing_default)]
    pub attributes: Attributes,
    pub events: Vec<EndedSpanEvent>,
    #[host(path = "status.status_code")]
    pub status: StatusCode,
    #[host(path = "status.description")]
    pub description: Option<String>,
}

/// An event of an [`EndedSpan`].
#[crate::host(record(
    module = "telemetry",
    host_type = "opentelemetry.sdk.trace.Event",
    boundary,
    extract
))]
#[derive(Debug, Clone)]
pub struct EndedSpanEvent {
    pub name: String,
    #[host(missing_default)]
    pub attributes: Attributes,
    pub timestamp: u64,
}

/// OpenTelemetry's `StatusCode`.
#[crate::host(native_enum(name = "StatusCode", host_type = "opentelemetry.trace.StatusCode"))]
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
