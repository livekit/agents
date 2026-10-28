//! The shared agent execution core (docs/internals/rust-core.md "Agent execution"): the decisions
//! direct `Agent.run()`, TextSession and VoiceSession share, as synchronous functions and state
//! machines over supplied data. No I/O, executor, clock or host object.
//!
//! - [`agent`]: the shared agent definition every invocation reads.
//! - [`assertions`]: the run event assertions `result.expect` walks.
//! - [`agent_run`]: `AgentRun`, one invocation's model/tool/output machine.
//! - [`request`]: request planning over an explicit snapshot.
//! - [`tool_outcome`]: the outcome table, what a tool's classified outcome records and asks for.
//! - [`tool_calls`]: one step's tool calls, from arguments to recorded outcomes.
//! - [`approval`]: the approval gate's identity, marks and read-back.
//! - [`text_session`]: request-scoped backend frames and child continuation.

pub mod agent;
pub mod agent_run;
pub mod approval;
pub mod assertions;
pub mod request;
pub mod text_session;
pub mod tool_calls;
pub mod tool_outcome;

pub use agent::AgentError;
pub use agent_run::{AgentRun, Effect, Event, RunInput};
