//! `metrics/`: the records (`base`, `usage::records`, `usage_collector::records`) and pure folds,
//! records in, [`AgentSessionUsage`] out. Not ported: the `__repr__`s, the `DeprecationWarning`s
//! (the binding raises them), `__call__`.

pub mod base;
pub mod log;
pub mod otel;
pub mod usage;
pub mod usage_collector;

#[cfg(test)]
pub(crate) mod testkit;

pub use base::{
    AgentMetrics, AvatarMetrics, CachedTokenDetails, EotInferenceMetrics, EouMetrics,
    InputTokenDetails, InterruptionMetrics, LlmMetrics, OutputTokenDetails,
    RealtimeModelMetrics, SttMetrics, TtsMetrics, VadMetrics,
};
pub use log::{LogValue, MetricsLog, log_fields};
pub use otel::{
    INSTRUMENTS, InstrumentKind, OtelRecord, OtelValue, turn_records, turn_records_wire,
    usage_records,
};
pub use usage::{
    AgentSessionUsage, EotModelUsage, InterruptionModelUsage, LlmModelUsage, ModelUsage,
    ModelUsageCollector, SttModelUsage, TtsModelUsage,
};
pub use usage_collector::{UsageCollector, UsageSummary};

/// Not in `__all__`, but imported directly by `inference/interruption.py`.
pub use base::Metadata;
