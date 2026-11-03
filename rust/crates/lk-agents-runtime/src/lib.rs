//! `lk-agents-runtime`: I/O, Tokio execution, real clocks, task ownership and effect execution for
//! the synchronous cores (docs/internals/rust-core.md "Cores and the runtime").
//!
//! - [`process`]: the process-wide Tokio runtime and its fork guard.
//! - [`host`]: `HostInvoker`, the lifecycle of host callbacks under their captured invocation.
//! - [`agent_run`]: one agent invocation, the core's `AgentRun` executed.
//! - [`a2a`]: the A2A text door's turns, driving the A2A core with the host's handler.

/// What the extension names of the subject cores (it reaches them through the
/// runtime): the declared classes a core implements outright.
pub mod cores {
    pub use lk_agents_core::a2a::card::CardBuilder;
    pub use lk_agents_core::a2a::handler::HandlerOutcomes;
    pub use lk_agents_core::agent_authoring::AgentConfig;
    pub use lk_agents_core::attempts::ConnectOptionsRules;
    pub use lk_agents_core::llm::chat_dict::ChatDict;
    pub use lk_agents_core::llm::ctx_diff::ChatDiffs;
    pub use lk_agents_core::llm::history::ChatPlans;
    pub use lk_agents_core::telemetry::metrics::MetricSeries;
    pub use lk_agents_core::telemetry::policy::{Admission, DestinationPolicy};
    pub use lk_agents_core::telemetry::provider::{
        LlmMetricState, ProviderMetrics, TtsMetricState,
    };
    pub use lk_agents_core::telemetry::records::MetricsRecords;
    pub use lk_agents_core::telemetry::report::UploadGate;
    pub use lk_agents_core::telemetry::transport::TraceParents;
    pub use lk_agents_core::text::Languages;
    pub use lk_agents_core::tokenize::{
        BasicSentenceTokenizer, BasicWordTokenizer, BufferedTokenStream, HostTokenizer,
    };
    pub use lk_agents_core::tool_search::{Bm25, Document as SearchDocument, ToolDiscovery};
    pub use lk_agents_core::tts::markup::{ProviderMarkup, SpeechMarkup, TranscriptMarkupStripper};
    pub use lk_agents_voice::run_result::VoiceRun;
    pub use lk_agents_voice::tokenize::SpeechTokenizer;
}

// The cores' in-place `_core` registrations are linked with the extension only from a crate it
// links that names them.
extern crate lk_agents_media as _;
extern crate lk_agents_voice as _;

/// The provider clients, judges and detectors, wired to this process by [`process::env`]; the
/// extension names their declared classes through this path.
pub use lk_agents_inference as inference;
/// The shared transport, A2A credentials and the delegation client, wired the same way.
pub use lk_agents_net as net;
/// Thread storage over AgentDB and local SQLite, wired the same way.
pub use lk_agents_storage as storage;

pub mod a2a;
pub mod agent_run;
pub mod host;
pub mod logging;
pub mod mcp;
pub mod monotone;
mod output_validation;
pub mod process;
pub mod recording;
pub mod telemetry;

pub mod text_session;

pub use lk_agents_worker as worker;

/// Process launcher and its host adapters.
pub use lk_agents_cli as cli;

/// Synchronous media algorithms used by the generated host adapters.
pub use lk_agents_media as media;

/// The voice session queue owner and its I/O effect execution.
pub mod voice;

pub mod avatar;
/// Runtime room execution over the ABI-declared SDK capabilities.
pub mod room_io;
pub mod rtc;
pub mod tcp;

pub mod toolsets;

pub mod run_stream;
