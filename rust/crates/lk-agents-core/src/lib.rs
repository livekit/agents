//! The agent execution decisions that direct runs, TextSession and VoiceSession share, as
//! synchronous state machines over supplied data. No I/O, executor, clock or host object.

pub mod a2a;
pub mod agent;
pub mod agent_run;
pub mod agent_tool;
pub mod approval;
pub mod assertions;
pub mod effects;
pub mod lexer;
pub mod llm;
pub mod mcp;
pub mod metrics;
pub mod request;
pub mod telemetry;
pub mod text_session;
pub mod tokenize;
pub mod tool_calls;
pub mod tool_outcome;
pub mod tool_search;
pub mod tts;
pub mod usage;

pub use agent::AgentError;
pub use agent_run::{AgentRun, AgentRunEffect, AgentRunEvent, Opening, RunInput};

pub mod text;

pub mod toolsets;

pub mod agent_authoring;

pub mod resource_lifecycle;

pub mod run_stream;
