//! `metrics/`: the records (`base`, `usage::records`) and pure folds, records in,
//! [`AgentSessionUsage`] out. Not ported: the `__repr__`s, `__call__`.

use crate::MetricsReport;

pub mod base;
pub mod log;
pub mod otel;
pub mod usage;

#[cfg(test)]
pub(crate) mod testkit;

pub use base::{
    AgentMetrics, AvatarMetrics, CachedTokenDetails, EotInferenceMetrics, EouMetrics,
    InputTokenDetails, InterruptionMetrics, LlmMetrics, OutputTokenDetails,
    RealtimeModelMetrics, SttMetrics, TtsMetrics, VadMetrics,
};
pub use log::{LogValue, MetricsLog, log_fields};
pub use otel::{
    INSTRUMENTS, InstrumentKind, OtelRecord, OtelValue, turn_records, usage_records,
};
/// One OTel observation: `(instrument, kind, value, attributes)`, attributes in insertion order.
pub type OtelWire = (String, String, OtelValue, indexmap::IndexMap<String, String>);

/// What the host's logger and meters record for a metrics event; the decisions are `log` and
/// `otel`'s, the implementation `lk-agents-core`'s.
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

pub use usage::{
    AgentSessionUsage, EotModelUsage, InterruptionModelUsage, LlmModelUsage, ModelUsage,
    ModelUsageCollector, SttModelUsage, TtsModelUsage,
};

/// Not in `__all__`, but imported directly by `inference/interruption.py`.
pub use base::Metadata;
