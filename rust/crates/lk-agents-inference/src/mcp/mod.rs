//! `llm/mcp.py` over the official Rust SDK (`rmcp`): the session and its transports are
//! [`session`], result rendering is [`render`], and the small decisions the shells ask for are here.
//! The Python classes forward to these; wrapping tools in `function_tool` stays host code.

pub mod render;
pub mod session;

// The generated readers resolve `McpToolOptions`' field types through this root glob.
pub use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope, ToolFlag};
pub use lk_agents_abi::llm::mcp::{McpToolOptions, McpToolOptionsPatch};
pub use render::{check_result, render_result};
use serde_json::{Map, Value, json};
pub use session::{McpClient, McpProgress, McpSession, McpTransport};

/// Every failure the adapter raises. The first three are `ToolError`s whose text the model reads.
/// `ToolError` for what the model reads, `ValueError` for a bad `transport_type`, else
/// `RuntimeError` (the reference raises the SDK's error).
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum McpError {
    #[error(
        "Tool invocation failed: internal service is unavailable. Please check that the MCPServer is still running."
    )]
    #[host(kind = "livekit.agents.llm.tool_context.ToolError")]
    Unavailable,
    #[error(
        "Tool '{0}' completed without producing a result. This might indicate an issue with internal processing."
    )]
    #[host(kind = "livekit.agents.llm.tool_context.ToolError")]
    EmptyResult(String),
    #[error("{0}")]
    #[host(kind = "livekit.agents.llm.tool_context.ToolError")]
    ToolFailed(String),
    #[error("MCPServer isn't initialized")]
    NotInitialized,
    #[error("transport_type must be 'sse' or 'streamable_http', got '{0}'")]
    #[host(kind = "ValueError")]
    InvalidTransport(String),
    /// A failed connection or request; the reference raises the SDK's own error.
    #[error("{0}")]
    Connection(String),
}

pub(crate) fn conn(e: impl std::fmt::Display) -> McpError {
    McpError::Connection(e.to_string())
}

/// `_resolve_tool_options`: `{**defaults, **config}`, key by key.
#[lk_agents_abi::host(export(
    name = "resolve_mcp_tool_options",
    pid = false
))]
pub fn resolve_tool_options(config: Option<&McpToolOptionsPatch>) -> McpToolOptions {
    let (d, p) = (McpToolOptions::default(), config.copied().unwrap_or_default());
    McpToolOptions {
        flags: p.flags.unwrap_or(d.flags),
        on_duplicate: p.on_duplicate.unwrap_or(d.on_duplicate),
        duplicate_scope: p.duplicate_scope.unwrap_or(d.duplicate_scope),
        report_progress: p.report_progress.unwrap_or(d.report_progress),
    }
}

/// `_make_function_tool`'s `raw_schema`: `description` always present (null when absent), `meta`
/// only when truthy.
#[lk_agents_abi::host(export(name = "mcp_raw_schema", pid = false))]
pub fn raw_schema(
    name: &str,
    description: Option<&str>,
    input_schema: &Map<String, Value>,
    #[host(required)] meta: Option<&Map<String, Value>>,
) -> Value {
    let mut schema = json!({"name": name, "description": description, "parameters": input_schema});
    if let Some(meta) = meta.filter(|m| !m.is_empty()) {
        schema["meta"] = json!(meta);
    }
    schema
}

/// `MCPToolset.setup(reload=)`'s branch.
#[lk_agents_abi::host(wire_enum)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupPlan {
    Skip,
    ConnectThenList,
    InvalidateThenList,
    ListOnly,
}

#[lk_agents_abi::host(export(name = "mcp_plan_setup", pid = false))]
pub fn plan_setup(reload: bool, toolset_initialized: bool, server_connected: bool) -> SetupPlan {
    match (reload, toolset_initialized, server_connected) {
        (false, true, _) => SetupPlan::Skip,
        (_, _, false) => SetupPlan::ConnectThenList,
        (true, _, true) => SetupPlan::InvalidateThenList,
        (false, false, true) => SetupPlan::ListOnly,
    }
}

/// `Literal["sse", "streamable_http"]`.
#[lk_agents_abi::host(wire_enum)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpTransport {
    Sse,
    StreamableHttp,
}

/// `transport_type` when given, else streamable HTTP only for a `urlparse` path ending in `/mcp`
/// (case- and trailing-slash-insensitive); everything else is SSE.
#[lk_agents_abi::host(export(
    name = "mcp_http_transport",
    pid = false
))]
pub fn http_transport(url: &str, transport_type: Option<&str>) -> Result<HttpTransport, McpError> {
    match transport_type {
        Some("sse") => Ok(HttpTransport::Sse),
        Some("streamable_http") => Ok(HttpTransport::StreamableHttp),
        Some(other) => Err(McpError::InvalidTransport(other.to_owned())),
        None if url_path(url).to_lowercase().trim_end_matches('/').ends_with("/mcp") => {
            Ok(HttpTransport::StreamableHttp)
        }
        None => Ok(HttpTransport::Sse),
    }
}

/// CPython's `urlparse(url).path` (the `url` crate normalises differently).
fn url_path(url: &str) -> String {
    const USES_PARAMS: &[&str] = &[
        "", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp", "rtsp", "rtsps", "rtspu",
        "sip", "sips", "mms", "sftp", "tel",
    ];
    let url: String = url
        .trim_start_matches(|c| c <= ' ')
        .chars()
        .filter(|c| !matches!(c, '\t' | '\r' | '\n'))
        .collect();
    let (mut scheme, mut rest) = ("", url.as_str());
    if let Some(i) = rest.find(':')
        && i > 0
        && rest.as_bytes()[0].is_ascii_alphabetic()
        && rest[..i].bytes().all(|b| b.is_ascii_alphanumeric() || b"+-.".contains(&b))
    {
        (scheme, rest) = (&rest[..i], &rest[i + 1..]);
    }
    if let Some(r) = rest.strip_prefix("//") {
        rest = &r[r.find(['/', '?', '#']).unwrap_or(r.len())..];
    }
    let rest = rest.split(['#', '?']).next().unwrap_or_default();
    // `_splitparams`: cut at the first `;` after the last `/`.
    let from = rest.rfind('/').unwrap_or(0);
    match rest[from..].find(';') {
        Some(i) if USES_PARAMS.contains(&scheme.to_lowercase().as_str()) => {
            rest[..from + i].to_owned()
        }
        _ => rest.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: detecting on anything but `urlparse`'s path (`;params` only for schemes that use
    // them, query and fragment cut, `1a:` no scheme so its path ends in `/mcp`), or streamable HTTP for every URL.
    #[test]
    fn only_a_urlparse_path_ending_in_mcp_is_streamable_http() {
        let t = |u| http_transport(u, None).unwrap() == HttpTransport::StreamableHttp;
        for u in
            ["https://h/MCP///", "https://h/a/mcp?x=/sse#f", "https://h/mcp;v=1", "/mcp", "1a:/mcp"]
        {
            assert!(t(u), "{u}");
        }
        for u in
            ["https://h/sse", "https://h", "https://h/mcpx", "ws://h/mcp;v=1", "https://h?p=/mcp"]
        {
            assert!(!t(u), "{u}");
        }
    }
}
