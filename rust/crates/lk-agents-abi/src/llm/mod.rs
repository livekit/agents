//! `llm/`: the chat context and its dumps, the LLM and realtime seams, the provider formats and
//! schemas, tool arguments, and the MCP, tool-executor and fallback records.

pub mod chat_context;
pub mod ctx_diff;
pub mod events;
pub mod fallback;
pub mod json_repair;
pub mod llm;
pub mod mcp;
pub mod provider_format;
pub mod realtime;
pub mod schema;
pub mod to_dict;
pub mod tool_arguments;
pub mod tool_executor;
