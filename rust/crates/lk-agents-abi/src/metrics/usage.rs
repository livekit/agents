//! `metrics/usage.py`'s rollup.

pub mod records;

pub use records::{
    AgentSessionUsage, BaseModelUsage, EotModelUsage, InterruptionModelUsage, LlmModelUsage,
    ModelUsage, SttModelUsage, TtsModelUsage,
};

use crate::metrics::base::AgentMetrics;

/// `metrics.ModelUsageCollector` -- the live per-model rollup.
#[crate::host(class(name = "ModelUsageCollector", pid = false))]
pub trait UsageCollector: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new() -> Self;

    #[host(call)]
    fn collect(&mut self, metrics: &AgentMetrics);

    #[host(call)]
    fn flatten(&self) -> Vec<ModelUsage>;
}
