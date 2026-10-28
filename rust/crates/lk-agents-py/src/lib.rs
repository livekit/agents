//! `livekit.agents._core` -- the PyO3 binding.
//!
//! Pure glue: every class, function and exception `_core` holds is declared beside its Rust item
//! in the core crate that owns it (`#[host(..)]`, expanded in place behind that crate's `python`
//! feature, which only this crate turns on) and registers itself; `#[pymodule]` performs the
//! registrations.
//!
//! Importing this module must start nothing (§1.13): everything that starts lives behind
//! `init(role)`, and every entry point calls `check_pid()` first, so a fork-inherited core raises
//! instead of deadlocking.

use pyo3::prelude::*;

// Linked for their registrations alone: nothing here names their items, and a crate the binary
// never mentions is not linked (its `inventory` entries with it).
extern crate lk_agents_inference as _;
extern crate lk_agents_media as _;
extern crate lk_agents_rtc as _;

extern crate lk_agents_worker as _;

/// The declared classes' pyo3 adapters, written by `lk-agents-codegen`.
pub mod generated;
/// `AgentNodes`, `tts_node` and `stt_node`: an agent's nodes over `lk-agents-voice`'s `NodeFactory`.
pub mod nodes;
/// `init(role)` and the registration table.
pub mod runtime;
/// `Streams` and `StreamReader`, and the record helpers of `turn.rs`.
pub mod storage;
/// `Thread`, `Turn` and `Boundary`: the run and the turn of `lk-agents-storage` over a Python host.
pub mod turn;

/// The classes the generated adapters hand an implementation's own objects out as.
pub mod handles {
    pub use crate::storage::PyStreams as Streams;
    pub use crate::turn::PyThread as Thread;
}

/// The implementation of each declared class: a class without one, or an implementation
/// missing a declared method, does not compile.
pub struct Selected;

impl generated::Implementations for Selected {
    type AgentClient = lk_agents_runtime::a2a_client::AgentClient;
    type Reply = lk_agents_runtime::a2a_client::Reply;
    type RemoteInteraction = lk_agents_runtime::a2a_client::remote::RemoteInteraction;
    type RemoteConfiguration = lk_agents_runtime::a2a_client::remote::RemoteInteraction;
    type ContextActivation = lk_agents_runtime::storage::context::ContextActivation;
    type SessionRecorder = lk_agents_runtime::recording::SessionRecorder;
    type Credentials = lk_agents_runtime::a2a_auth::Credentials;
    type AuthPermissions = lk_agents_runtime::a2a_auth::Credentials;
    type EndpointCard = lk_agents_runtime::a2a_card::CardBuilder;
    type McpSession = lk_agents_runtime::mcp::McpSession;
    type McpTools = lk_agents_runtime::mcp::McpTools;
    type WorkerCli = lk_agents_worker::cli::WorkerCli;
    type CliSignals = lk_agents_runtime::worker_cli::CliSignals;
    type TelemetryPolicy = lk_agents_runtime::telemetry::policy::DestinationPolicy;
    type EventRetention = lk_agents_runtime::telemetry::observation::EventRetention;
    type PreparedTranscript = lk_agents_runtime::telemetry::upload::TranscriptBuilder;
    type TelemetryWorker = lk_agents_runtime::telemetry::work::SdkWorker;
    type SessionReportUpload = lk_agents_runtime::telemetry::upload::SessionReportUploader;
    type LoopMonitor = lk_agents_runtime::telemetry::loop_monitor::python::LoopMonitor;
    type LoopMonitorThresholds = lk_agents_runtime::telemetry::loop_monitor::LoopMonitorOptions;
    type TextSessionRunner = lk_agents_runtime::text_session::Runner;
    type AgentRunner = lk_agents_runtime::agent_run::Runner;
    type RunEvents = lk_agents_runtime::agent_run::RunEvents;
    type InferenceAvatar = lk_agents_runtime::avatar::GatewayAvatar;
    type InferenceLlm = lk_agents_runtime::llm::GatewayLlm;
    type ChoiceParser = lk_agents_inference::llm::parse::ChoiceParser;
    type CompletionWire = lk_agents_inference::llm::parse::CompletionWire;
    type Storage = lk_agents_runtime::storage::host::HostStorage;
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Nothing here may spawn, connect, or allocate a runtime. The core's `log` lines reach the
    // Python logger of the same name from import on (installing a logger starts nothing).
    let _ = pyo3_log::try_init();
    runtime::register_all(m)?;
    generated::register(m)
}
