//! Event-loop watchdog observations. Time and samples are supplied by the runtime adapters.

/// A heartbeat that arrived after its configured warning threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopBlock {
    pub tick_sequence: u64,
    pub duration: f64,
    pub started_at: f64,
    pub warn_threshold: f64,
    pub severity: LoopBlockSeverity,
    pub gc_time: f64,
    pub cpu_time: f64,
    pub watchdog_gap: f64,
    pub process_descheduled: bool,
    pub emit_span: bool,
    pub emit_log: bool,
    pub reported: bool,
    pub suppressed: u64,
}

#[crate::host(record(module = "telemetry", emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopBlockSeverity {
    Warning,
    Error,
}

impl LoopBlockSeverity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

/// A reserved stack sample. Only samples for the same heartbeat sequence may be attached.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopSampleRequest {
    pub tick_sequence: u64,
    pub lag: f64,
}

/// A host monitor's rendered report. Its parent is an opaque SDK context used only by the host
/// adapter; the policy's LoopBlock above contains no live objects.
#[crate::host(record(module = "telemetry", emit))]
#[derive(Clone)]
pub struct BlockedReport {
    pub duration: f64,
    pub started_at: f64,
    pub warn_threshold: f64,
    pub severity: LoopBlockSeverity,
    pub gc_time: f64,
    pub cpu_time: f64,
    pub watchdog_gap: f64,
    pub process_descheduled: bool,
    pub task_name: Option<String>,
    pub stacks: Vec<String>,
    #[host(default = None)]
    pub importing: Option<String>,
    #[host(default = None)]
    pub parent_span_context: Option<crate::HostRef>,
    #[host(default = None)]
    pub location: Option<String>,
}

#[crate::host(record(module = "telemetry", emit))]
pub struct LoopMonitorSpan {
    pub attributes: serde_json::Map<String, serde_json::Value>,
    pub error: Option<String>,
}

#[crate::host(record(module = "telemetry", emit))]
pub struct LoopMonitorLog {
    pub message: String,
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[crate::host(record(module = "telemetry", emit))]
pub struct LoopMonitorEmission {
    pub report: BlockedReport,
    pub span: Option<LoopMonitorSpan>,
    pub log: Option<LoopMonitorLog>,
    pub reported: bool,
}

/// A loop's watchdog: runtime owns thresholds, timing, admission and cleanup, while the host
/// supplies asyncio scheduling, stack samples and OpenTelemetry emission.
#[crate::host(class(name = "LoopMonitor"))]
pub trait LoopMonitor: Send + Sync + Sized + 'static {
    #[host(new)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        event_loop: crate::HostRef,
        #[host(kw)] warn_threshold: f64,
        error_threshold: f64,
        tick_interval: Option<f64>,
        name: String,
        emit_spans: bool,
        report: crate::HostRef,
        span_of: crate::HostRef,
    ) -> Result<Self, crate::errors::HostError>;

    #[host(getter)]
    fn warn_threshold(&self) -> f64;

    #[host(getter)]
    fn error_threshold(&self) -> f64;

    #[host(call)]
    fn start(&self) -> Result<(), crate::errors::HostError>;

    #[host(call)]
    fn stop(&self) -> Result<(), crate::errors::HostError>;
}

/// Thresholds from the host process environment's `LIVEKIT_AGENTS_LOOP_BLOCK_WARN_MS` and
/// `LIVEKIT_AGENTS_LOOP_BLOCK_ERROR_MS`, as the host read them.
#[crate::host(functions)]
pub trait LoopMonitorThresholds {
    #[host(function)]
    fn _core_loop_monitor_thresholds(
        warn_ms: Option<String>,
        error_ms: Option<String>,
    ) -> Option<(f64, f64)>;
}
