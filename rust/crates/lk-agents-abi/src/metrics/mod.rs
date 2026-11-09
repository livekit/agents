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

/// Not in `__all__`, but imported directly by `inference/interruption.py`.
pub use base::Metadata;
