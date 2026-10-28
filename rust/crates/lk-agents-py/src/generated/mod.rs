//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The declared classes' pyo3 adapters, one module per ABI subject.

use pyo3::prelude::*;

pub mod a2a;
pub mod agent;
pub mod evals;
pub mod inference;
pub mod llm;
pub mod recorder;
pub mod storage;
pub mod stt;
pub mod telemetry;
pub mod text_session;
pub mod tts;
pub mod turn;
pub mod vad;
pub mod voice;
pub mod worker;

/// The implementation of each declared class. The registration shell implements it.
pub trait Implementations: 'static {
    type Credentials: ::lk_agents_abi::a2a::auth::Credentials;
    type AuthPermissions: ::lk_agents_abi::a2a::auth::AuthPermissions;
    type EndpointCard: ::lk_agents_abi::a2a::card::EndpointCard;
    type Reply: ::lk_agents_abi::a2a::client::Reply;
    type AgentClient: ::lk_agents_abi::a2a::client::AgentClient<Reply = Self::Reply>;
    type A2ADoor: ::lk_agents_abi::a2a::door::A2ADoor<Response = Self::A2AResponse>;
    type A2AResponse: ::lk_agents_abi::a2a::door::A2AResponse;
    type RemoteInteraction: ::lk_agents_abi::a2a::remote::RemoteInteraction;
    type RemoteConfiguration: ::lk_agents_abi::a2a::remote::RemoteConfiguration;
    type RunEvents: ::lk_agents_abi::agent::assertions::RunEvents;
    type AgentRunner: ::lk_agents_abi::agent::run::AgentRunner;
    type JudgeRunner: ::lk_agents_abi::evals::JudgeRunner;
    type Evaluations: ::lk_agents_abi::evals::Evaluations;
    type InferenceAvatar: ::lk_agents_abi::inference::avatar_session::InferenceAvatar;
    type InferenceLlm: ::lk_agents_abi::llm::inference::InferenceLlm;
    type ChoiceParser: ::lk_agents_abi::llm::inference::ChoiceParser;
    type CompletionWire: ::lk_agents_abi::llm::inference::CompletionWire;
    type McpSession: ::lk_agents_abi::llm::mcp::McpSession;
    type McpTools: ::lk_agents_abi::llm::mcp::McpTools;
    type NativeRealtimeModel: ::lk_agents_abi::llm::realtime_native::NativeRealtimeModel;
    type RealtimeFallbackModel: ::lk_agents_abi::llm::realtime_native::RealtimeFallbackModel;
    type SessionRecorder: ::lk_agents_abi::recorder::SessionRecorder;
    type Storage: ::lk_agents_abi::storage::Storage<ContextActivation = Self::ContextActivation, VoiceStore = Self::VoiceStore, A2ADoor = Self::A2ADoor>;
    type ContextActivation: ::lk_agents_abi::storage::context::ContextActivation;
    type VoiceStore: ::lk_agents_abi::storage::voice::VoiceStore<ContextActivation = Self::ContextActivation>;
    type TelemetryPolicy: ::lk_agents_abi::telemetry::policy::TelemetryPolicy;
    type LoopMonitor: ::lk_agents_abi::telemetry::loop_monitor::LoopMonitor;
    type LoopMonitorThresholds: ::lk_agents_abi::telemetry::loop_monitor::LoopMonitorThresholds;
    type EventRetention: ::lk_agents_abi::telemetry::observation::EventRetention;
    type SessionReportUpload: ::lk_agents_abi::telemetry::upload::SessionReportUpload;
    type PreparedTranscript: ::lk_agents_abi::telemetry::upload::PreparedTranscript;
    type TelemetryWorker: ::lk_agents_abi::telemetry::work::TelemetryWorker;
    type TextSessionRunner: ::lk_agents_abi::text_session::TextSessionRunner;
    type WorkerCli: ::lk_agents_abi::worker::cli::WorkerCli;
    type CliSignals: ::lk_agents_abi::worker::cli::CliSignals;
}

/// Adds every declared class to `_core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<a2a::_core_Credentials>()?;
    m.add_function(wrap_pyfunction!(a2a::_core_auth_grant_allows, m)?)?;
    m.add_function(wrap_pyfunction!(a2a::_core_a2a_extensions_header, m)?)?;
    m.add_function(wrap_pyfunction!(a2a::_core_a2a_agent_card, m)?)?;
    m.add_class::<a2a::Reply>()?;
    m.add_class::<a2a::AgentClient>()?;
    m.add_class::<a2a::_core_A2ADoor>()?;
    m.add_class::<a2a::_core_A2AResponse>()?;
    m.add_class::<a2a::_core_RemoteInteraction>()?;
    m.add_function(wrap_pyfunction!(a2a::_core_remote_token, m)?)?;
    m.add_class::<agent::RunEvents>()?;
    m.add_class::<agent::AgentRunner>()?;
    m.add_class::<agent::AgentRunUpdates>()?;
    m.add_class::<evals::_core_JudgeRunner>()?;
    m.add_function(wrap_pyfunction!(evals::_core_evaluation_score, m)?)?;
    m.add_function(wrap_pyfunction!(evals::_core_judge_tool_schemas, m)?)?;
    m.add_class::<inference::InferenceAvatar>()?;
    m.add_class::<llm::InferenceLlm>()?;
    m.add_class::<llm::InferenceLlmChunks>()?;
    m.add_class::<llm::LlmChoiceParser>()?;
    m.add_function(wrap_pyfunction!(llm::llm_usage_chunk, m)?)?;
    m.add_class::<llm::McpSession>()?;
    m.add_function(wrap_pyfunction!(llm::resolve_mcp_tool_options, m)?)?;
    m.add_function(wrap_pyfunction!(llm::mcp_raw_schema, m)?)?;
    m.add_function(wrap_pyfunction!(llm::mcp_plan_setup, m)?)?;
    m.add_function(wrap_pyfunction!(llm::mcp_http_transport, m)?)?;
    m.add_function(wrap_pyfunction!(llm::mcp_render_result, m)?)?;
    m.add_function(wrap_pyfunction!(llm::mcp_check_result, m)?)?;
    m.add_class::<llm::NativeRealtimeModel>()?;
    m.add_class::<llm::RealtimeFallbackModel>()?;
    m.add_class::<recorder::SessionRecorder>()?;
    m.add_class::<storage::Storage>()?;
    m.add_class::<storage::_core_ContextActivation>()?;
    m.add_class::<storage::_core_VoiceStore>()?;
    m.add_class::<telemetry::_core_TelemetryPolicy>()?;
    m.add_class::<telemetry::LoopMonitor>()?;
    m.add_function(wrap_pyfunction!(telemetry::_core_loop_monitor_thresholds_from_env, m)?)?;
    m.add_class::<telemetry::_core_EventRetention>()?;
    m.add_class::<telemetry::_core_SessionReportUpload>()?;
    m.add_class::<telemetry::_core_PreparedTranscript>()?;
    m.add_class::<telemetry::_core_TelemetryWorker>()?;
    m.add_class::<text_session::TextSessionRunner>()?;
    m.add_class::<text_session::TextSessionUpdates>()?;
    m.add_function(wrap_pyfunction!(worker::parse_cli, m)?)?;
    m.add_function(wrap_pyfunction!(worker::entrypoint_module, m)?)?;
    m.add_class::<worker::ExitSignals>()?;
    m.add_function(wrap_pyfunction!(text_session::_core_persist_text_session_state, m)?)?;
    m.add_function(wrap_pyfunction!(text_session::_core_restore_text_session_state, m)?)?;
    Ok(())
}
