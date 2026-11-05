//! The MCP session and its transports: thin I/O over the upstream SDK, and an `MCPToolset`'s
//! lifecycle (its recovery and idle close). The tool logic is `lk_agents_core::mcp`.

mod http;
mod session;
mod toolset;
mod transport;

pub use lk_agents_abi::llm::mcp::{McpError, McpProgress};
pub use lk_agents_core::mcp::McpTools;
pub use session::McpSession;
pub use toolset::McpToolsetLifecycle;

pub(crate) fn conn(e: impl std::error::Error + Send + Sync + 'static) -> McpError {
    McpError::Transport(std::sync::Arc::new(e))
}

#[cfg(test)]
#[path = "../mcp_tests.rs"]
mod tests;
