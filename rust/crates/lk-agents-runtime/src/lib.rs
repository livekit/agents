//! `lk-agents-runtime`: I/O, Tokio execution, real clocks, task ownership and effect execution for
//! the synchronous cores (docs/internals/rust-core.md "Cores and the runtime").
//!
//! - [`process`]: the process-wide Tokio runtime and its fork guard.
//! - [`task`]: owned tasks with correlated cancellation and one observed outcome.
//! - [`sse`]: event-stream framing over response bytes.
//! - [`token`]: framework credentials signed with LiveKit's token crate.
//! - [`http`]: the gateway's HTTPS exchange, reporting transport observations.
//! - [`llm`]: the gateway LLM, executing the LLM request core's effects.
//! - [`source`]: a native source as one host consumer pulls and closes it.

pub mod http;
pub mod llm;
pub mod process;
pub mod source;
pub mod sse;
pub mod task;
pub mod token;
