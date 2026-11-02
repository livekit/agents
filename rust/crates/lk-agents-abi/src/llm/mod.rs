//! `llm/`: the chat context and its dumps, the LLM and realtime seams, the provider formats and
//! schemas, tool arguments, and the MCP, tool-executor and fallback records.

pub mod chat_context;
pub mod duplex;
pub mod events;
pub mod fallback;
pub mod inference;
pub mod mcp;
pub mod provider_format;
pub mod realtime;
pub mod schema;
pub mod to_dict;
pub mod tool_executor;
#[path = "llm.rs"]
pub mod types;

pub mod toolsets;
