//! `metrics/`: the records (`base`, `usage::records`) and pure folds, records in,
//! [`AgentSessionUsage`] out. Not ported: the `__repr__`s, `__call__`.

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
    INSTRUMENTS, InstrumentKind, OtelRecord, OtelValue, turn_records, turn_records_wire,
    usage_records,
};
pub use usage::{
    AgentSessionUsage, EotModelUsage, InterruptionModelUsage, LlmModelUsage, ModelUsage,
    ModelUsageCollector, SttModelUsage, TtsModelUsage,
};

/// Not in `__all__`, but imported directly by `inference/interruption.py`.
pub use base::Metadata;
