//! `metrics/`: the performance records (`base`) and the provider folds that make them
//! (`provider`). Usage is `crate::usage`'s.

pub mod base;
pub mod provider;

#[cfg(any(test, feature = "testkit"))]
pub mod testkit;

pub use base::{
    AgentMetrics, AvatarMetrics, EotInferenceMetrics, EouMetrics, InterruptionMetrics, LlmMetrics,
    RealtimeModelMetrics, SttMetrics, TtsMetrics, VadMetrics,
};

/// The process runtime's load, as the latency benchmark reads it.
#[crate::host(functions(pid = false))]
pub trait RuntimeLoad {
    /// Each async worker's total busy seconds so far; empty before the runtime started.
    #[host(function)]
    fn runtime_busy_seconds() -> Vec<f64>;
}

/// One metrics event's log line: its message and the fields logged as `extra`.
#[crate::host(record(module = "metrics", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct MetricsLog {
    pub message: String,
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// What the host's logger records for a metrics event; the decisions are
/// `lk_agents_core::metrics::records`'.
#[crate::host(functions)]
pub trait MetricsRecords {
    /// `log_metrics(metrics)` minus the logging, or `None` for do not log (`VADMetrics` and
    /// `EOTInferenceMetrics` have no branch in the reference).
    #[host(function)]
    fn metrics_log_fields(metrics: AgentMetrics) -> Option<MetricsLog>;
}

/// Not in `__all__`, but imported directly by `inference/interruption.py`.
pub use base::Metadata;
