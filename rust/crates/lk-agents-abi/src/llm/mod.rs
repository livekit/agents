//! `llm/`: the chat context and its dumps, the LLM and realtime seams, the provider formats and
//! schemas, tool arguments, and the MCP, tool-executor and fallback records.

pub mod chat_context;
pub mod ctx_diff;
pub mod duplex;
pub mod events;
pub mod fallback;
pub mod inference;
pub mod llm;
pub mod mcp;
#[cfg(feature = "serde")]
mod persistence;
pub mod provider_format;
pub mod realtime;
pub mod realtime_native;
pub mod schema;
pub mod to_dict;
pub mod tool_executor;
