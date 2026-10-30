//! `metrics/`: the performance records (`base`). Usage is `crate::usage`'s.

use crate::MetricsReport;

pub mod base;

#[cfg(any(test, feature = "testkit"))]
pub mod testkit;

pub use base::{
    AgentMetrics, AvatarMetrics, CachedTokenDetails, EotInferenceMetrics, EouMetrics,
    InputTokenDetails, InterruptionMetrics, LlmMetrics, OutputTokenDetails, RealtimeModelMetrics,
    SttMetrics, TtsMetrics, VadMetrics,
};
/// Token counts stay integers: the exported data point type follows the first value seen.
#[crate::host(record(module = "telemetry", emit, transparent))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OtelValue {
    Int(u64),
    Float(f64),
}

/// One OTel observation: `(instrument, kind, value, attributes)`, attributes in insertion order.
pub type OtelWire = (String, String, OtelValue, indexmap::IndexMap<String, String>);

/// What the host's logger and meters record for a metrics event; the decisions are
/// `lk_agents_core::telemetry::records`'.
#[crate::host(functions)]
pub trait MetricsRecords {
    /// `log_metrics(metrics)` minus the logging: `(message, extra)`, or `None` for do not log
    /// (`VADMetrics` and `EOTInferenceMetrics` have no branch in the reference).
    #[host(function)]
    fn metrics_log_fields(
        metrics: AgentMetrics,
    ) -> Option<(String, serde_json::Map<String, serde_json::Value>)>;

    /// `_record_turn_metrics(report)` minus the recording. `None` and `{}` yield nothing; a
    /// present `0.0` yields a record.
    #[host(function)]
    fn otel_turn_records(report: Option<MetricsReport>) -> Vec<OtelWire>;

    /// `collect_usage(ev)` minus the recording: the counter increments it would make. Empty for
    /// unhandled types, unknown types and honest zeros alike (the reference gates on truthiness).
    #[host(function)]
    fn otel_usage_records(metrics: AgentMetrics) -> Vec<OtelWire>;
}

/// Not in `__all__`, but imported directly by `inference/interruption.py`.
pub use base::Metadata;
