//! The shared agent execution core (docs/internals/rust-core.md "Agent execution"): the decisions
//! direct `Agent.run()`, TextSession and VoiceSession share, as synchronous functions and state
//! machines over supplied data. No I/O, executor, clock or host object.
//!
//! - [`agent`]: the shared agent definition every invocation reads.
//! - [`agent_run`]: `AgentRun`, one invocation's model/tool/output machine.
//! - [`request`]: request planning over an explicit snapshot.
//! - [`tool_outcome`]: the outcome table, what a tool's classified outcome records and asks for.
//! - [`approval`]: the approval gate's identity, marks and read-back.

pub mod agent;
pub mod agent_run;
pub mod approval;
pub mod request;
pub mod tool_outcome;

pub use agent::{Agent, AgentError, FunctionToolInfo, OutputSpec, ReadBack, Retries};
pub use agent_run::{AgentRun, Effect, Event, RunInput};
