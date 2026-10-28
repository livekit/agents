//! `lk-agents-runtime`: I/O, Tokio execution, real clocks, task ownership and effect execution for
//! the synchronous cores (docs/internals/rust-core.md "Cores and the runtime").
//!
//! - [`process`]: the process-wide Tokio runtime and its fork guard.
//! - [`http`]: the gateway's HTTPS exchange, reporting transport observations.
//! - [`llm`]: the gateway LLM, executing the LLM request core's effects.
//! - [`avatar`]: the gateway avatar, executing the avatar session core's effects.
//! - [`source`]: a native source as one host consumer pulls and closes it.
//! - [`host`]: `HostInvoker`, the lifecycle of host callbacks under their captured invocation.
//! - [`agent_run`]: one agent invocation, the core's `AgentRun` executed.
//! - [`evals`]: conversation judges, run concurrently on one conversation.
//! - [`storage`]: thread storage I/O over AgentDB and local SQLite.
//! - [`a2a`]: the A2A text door's turns, driving the A2A core with the host's handler.

/// What the extension names of the subject cores (decision 67: it reaches them through the
/// runtime): the declared classes a core implements outright, and the sentence splitter its
/// default TTS node composes.
pub mod cores {
    pub use lk_agents_inference::creds::Credentials as InferenceCredentials;
    pub use lk_agents_inference::llm::parse::{ChoiceParser, CompletionWire};
    pub use lk_agents_inference::provider::ProviderAttempts;
    pub use lk_agents_media::blingfire::SentenceBuffer;
}

// The cores' in-place `_core` registrations are linked with the extension only from a crate it
// links that names them.
#[cfg(feature = "python")]
extern crate lk_agents_inference as _;
#[cfg(feature = "python")]
extern crate lk_agents_media as _;

pub mod a2a;
pub mod a2a_auth;
pub mod a2a_card;
pub mod a2a_client;
pub mod agent_run;
pub mod avatar;
pub mod avatar_output;
pub mod codecs;
pub mod duplex;
pub mod evals;
pub mod fallback;
pub mod host;
pub mod http;
#[cfg(feature = "media-images")]
pub mod images;
pub mod llm;
pub mod mcp;
pub mod media_workers;
pub mod process;
pub mod realtime;
pub mod recording;
pub mod source;
pub mod storage;
pub mod stt;
pub mod telemetry;

pub mod worker_cli;

pub mod text_session;

pub mod worker;

pub mod vad;

pub mod llm_algorithms;
pub use lk_agents_media::filters;
