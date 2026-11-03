//! Every framework span, its start and end attributes and its events (`telemetry.md`, "Span
//! API"). The registry, the consumer schema and the trace rules are generated from these.

pub mod agent_turn;
pub mod common;
pub mod inference;
pub mod job;
pub mod session;
pub mod text;
pub mod user_turn;
