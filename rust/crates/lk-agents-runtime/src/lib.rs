//! `lk-agents-runtime`: I/O, Tokio execution, real clocks, task ownership and effect execution for
//! the synchronous cores (docs/internals/rust-core.md "Cores and the runtime").
//!
//! - [`process`]: the process-wide Tokio runtime and its fork guard.
//! - [`task`]: owned tasks with correlated cancellation and one observed outcome.
//! - [`sse`]: event-stream framing over response bytes.
//! - [`token`]: framework credentials signed with LiveKit's token crate.
//! - [`http`]: the gateway's HTTPS exchange, reporting transport observations.
//! - [`llm`]: the gateway LLM, executing the LLM request core's effects.
//! - [`avatar`]: the gateway avatar, executing the avatar session core's effects.
//! - [`source`]: a native source as one host consumer pulls and closes it.
//! - [`host`]: `HostInvoker`, the lifecycle of host callbacks under their captured invocation.
//! - [`agent_run`]: one agent invocation, the core's `AgentRun` executed.
//! - [`storage`]: thread storage I/O over AgentDB and local SQLite.
//! - [`a2a`]: the A2A text door's turns, driving the A2A core with the host's handler.

pub mod agent_run;
pub mod codecs;
pub mod media_workers;
pub mod recording;
#[cfg(feature = "media-images")]
pub mod images;
pub mod avatar;
pub mod host;
pub mod a2a;
pub mod a2a_client;
pub mod http;
pub mod llm;
pub mod process;
pub mod source;
pub mod storage;
pub mod sse;
pub mod task;
pub mod telemetry;
pub mod token;
