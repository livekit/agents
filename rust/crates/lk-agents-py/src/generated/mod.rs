//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The declared classes' pyo3 adapters, one module per ABI subject.

use pyo3::prelude::*;

pub mod agent;
pub mod inference;
pub mod llm;
pub mod storage;
pub mod telemetry;

/// The implementation of each declared class. The registration shell implements it.
pub trait Implementations: 'static {
    type AgentRunner: ::lk_agents_abi::agent::run::AgentRunner;
    type InferenceAvatar: ::lk_agents_abi::inference::avatar_session::InferenceAvatar;
    type InferenceLlm: ::lk_agents_abi::llm::inference::InferenceLlm;
    type ChoiceParser: ::lk_agents_abi::llm::inference::ChoiceParser;
    type CompletionWire: ::lk_agents_abi::llm::inference::CompletionWire;
    type Storage: ::lk_agents_abi::storage::Storage;
    type LoopMonitor: ::lk_agents_abi::telemetry::loop_monitor::LoopMonitor;
    type LoopMonitorThresholds: ::lk_agents_abi::telemetry::loop_monitor::LoopMonitorThresholds;
    type SessionReportUpload: ::lk_agents_abi::telemetry::upload::SessionReportUpload;
    type TelemetryWorker: ::lk_agents_abi::telemetry::work::TelemetryWorker;
}

/// Adds every declared class to `_core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<agent::AgentRunner>()?;
    m.add_class::<agent::AgentRunUpdates>()?;
    m.add_class::<inference::InferenceAvatar>()?;
    m.add_class::<llm::InferenceLlm>()?;
    m.add_class::<llm::InferenceLlmChunks>()?;
    m.add_class::<llm::LlmChoiceParser>()?;
    m.add_function(wrap_pyfunction!(llm::llm_usage_chunk, m)?)?;
    m.add_class::<storage::Storage>()?;
    m.add_class::<telemetry::LoopMonitor>()?;
    m.add_function(wrap_pyfunction!(telemetry::_core_loop_monitor_thresholds_from_env, m)?)?;
    m.add_class::<telemetry::_core_SessionReportUpload>()?;
    m.add_class::<telemetry::_core_TelemetryWorker>()?;
    Ok(())
}
