//! `lk-agents-runtime`: I/O, Tokio execution, real clocks, task ownership and effect execution for
//! the synchronous cores (docs/internals/rust-core.md "Cores and the runtime").
//!
//! - [`process`]: the process-wide Tokio runtime and its fork guard.
//! - [`host`]: `HostInvoker`, the lifecycle of host callbacks under their captured invocation.
//! - [`agent_run`]: one agent invocation, the core's `AgentRun` executed.
//! - [`a2a`]: the A2A text door's turns, driving the A2A core with the host's handler.

/// What the extension names of the subject cores (decision 67: it reaches them through the
/// runtime): the declared classes a core implements outright.
pub mod cores {
    pub use lk_agents_core::a2a::card::CardBuilder;
    pub use lk_agents_core::text::TextAlgorithms;
    pub use lk_agents_core::telemetry::loop_monitor::{LoopMonitorOptions, LoopMonitorPolicy};
    pub use lk_agents_core::telemetry::metrics::MetricSeries;
    pub use lk_agents_core::telemetry::policy::DestinationPolicy;
    pub use lk_agents_core::tool_search::{Bm25, Document as SearchDocument, ToolDiscovery};
    pub use lk_agents_voice::tokenize::{
        BasicSentenceTokenizer, BasicWordTokenizer, SpeechTokenizer,
    };
    pub use lk_agents_voice::tts::markup::{
        ProviderMarkup, SpeechMarkup, TranscriptMarkupStripper,
    };
}

// The cores' in-place `_core` registrations are linked with the extension only from a crate it
// links that names them.
#[cfg(feature = "python")]
extern crate lk_agents_media as _;
#[cfg(feature = "python")]
extern crate lk_agents_voice as _;

/// The provider clients, judges and detectors, wired to this process by [`process::Process`]
/// (decision 88); the extension names their declared classes through this path.
pub use lk_agents_inference as inference;
/// The shared transport, A2A credentials and the delegation client, wired the same way.
pub use lk_agents_net as net;
/// Thread storage over AgentDB and local SQLite; the extension selects its `HostStorage` with
/// this process.
pub use lk_agents_storage as storage;

pub mod a2a;
pub mod agent_run;
pub mod audio_emitter;
pub mod codecs;
pub mod host;
#[cfg(feature = "media-images")]
pub mod images;
pub mod loop_watchdog;
pub mod mcp;
pub mod process;
pub mod recording;

pub mod text_session;

pub use lk_agents_worker as worker;

/// Synchronous media algorithms used by the generated host adapters.
pub use lk_agents_media as media;

/// The voice session queue owner and its I/O effect execution.
pub mod voice;

pub mod avatar;
pub mod avatar_receiver;
pub mod avatar_runner;
mod avatar_runner_host;
/// Runtime room execution over the ABI-declared SDK capabilities.
pub mod room_io;
pub mod rtc;
/// Tokenization through the application's tokenizer callbacks.
pub mod tokenize;
#[cfg(test)]
pub(crate) use lk_agents_test_support::rtc as test_support_rtc;
