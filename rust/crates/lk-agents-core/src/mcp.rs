//! MCP tool logic (decision 71): option resolution, schemas, setup and transport choice, and the
//! interpretation of tool results. The session and its transports are the runtime's I/O.

use std::sync::Arc;

use lk_agents_abi::llm::mcp::{
    McpCleanupStatus, McpError, McpToolOptions, McpToolOptionsPatch, McpTools as McpToolsPort,
};
use serde_json::{Map, Value, json};

/// `_resolve_tool_options`: `{**defaults, **config}`, key by key.
pub fn resolve_tool_options(config: Option<&McpToolOptionsPatch>) -> McpToolOptions {
    let (d, p) = (McpToolOptions::default(), config.copied().unwrap_or_default());
    McpToolOptions {
        flags: p.flags.unwrap_or(d.flags),
        on_duplicate: p.on_duplicate.unwrap_or(d.on_duplicate),
        duplicate_scope: p.duplicate_scope.unwrap_or(d.duplicate_scope),
        report_progress: p.report_progress.unwrap_or(d.report_progress),
    }
}

pub fn raw_schema(
    name: &str,
    description: Option<&str>,
    input_schema: &Map<String, Value>,
    meta: Option<&Map<String, Value>>,
) -> Value {
    let mut schema = json!({"name": name, "description": description, "parameters": input_schema});
    if let Some(meta) = meta.filter(|m| !m.is_empty()) {
        schema["meta"] = json!(meta);
    }
    schema
}

pub fn plan_setup(reload: bool, toolset_initialized: bool, server_connected: bool) -> SetupPlan {
    match (reload, toolset_initialized, server_connected) {
        (false, true, true) => SetupPlan::Skip,
        (_, _, false) => SetupPlan::ConnectThenList,
        (true, _, true) => SetupPlan::InvalidateThenList,
        (false, false, true) => SetupPlan::ListOnly,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupPlan {
    Skip,
    ConnectThenList,
    InvalidateThenList,
    ListOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpTransport {
    Sse,
    StreamableHttp,
}

/// `transport_type` when given, else streamable HTTP only for a `urlparse` path ending in `/mcp`
/// (case- and trailing-slash-insensitive); everything else is SSE.
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

/// The cleanup status a session reports: incomplete once anything was abandoned or failed, clean
/// only when the service and its reader settled.
pub fn cleanup_status(abandoned: bool, settled: bool, reader_settled: bool) -> McpCleanupStatus {
    if abandoned {
        McpCleanupStatus::Incomplete
    } else if settled && reader_settled {
        McpCleanupStatus::Clean
    } else {
        McpCleanupStatus::Pending
    }
}

fn decode_error(message: impl Into<String>) -> McpError {
    let error: Box<dyn std::error::Error + Send + Sync> = message.into().into();
    McpError::Decode(Arc::from(error))
}

/// A `CallToolResult` as the SDK serializes it: its content blocks and error flag.
fn content(result: &Value) -> Result<&[Value], McpError> {
    let malformed = || decode_error("a tool result is an object of `content` blocks and `isError`");
    let result = result.as_object().ok_or_else(malformed)?;
    let content = match result.get("content") {
        None => None,
        Some(content) => Some(content.as_array().ok_or_else(malformed)?),
    };
    let is_error = result.get("isError");
    if is_error.is_some_and(|flag| !flag.is_null() && !flag.is_boolean()) {
        return Err(malformed());
    }
    if is_error.and_then(Value::as_bool) == Some(true) {
        let parts = content.into_iter().flatten().map(|part| match part.get("text") {
            Some(Value::String(text)) if part.get("type") == Some(&json!("text")) => text.clone(),
            _ => part.to_string(),
        });
        return Err(McpError::ToolFailed(parts.collect::<Vec<_>>().join("\n")));
    }
    Ok(content.map(Vec::as_slice).unwrap_or_default())
}

pub fn check_result(result: &Value) -> Result<(), McpError> {
    content(result).map(drop)
}

pub fn render_result(tool_name: &str, result: &Value) -> Result<String, McpError> {
    Ok(match content(result)? {
        [] => return Err(McpError::EmptyResult(tool_name.into())),
        [one] => one.to_string(),
        many => Value::from(many.to_vec()).to_string(),
    })
}

#[cfg(test)]
mod transport_selection_tests {
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

pub struct McpTools;
impl McpToolsPort for McpTools {
    fn resolve_tool_options(config: Option<McpToolOptionsPatch>) -> McpToolOptions {
        resolve_tool_options(config.as_ref())
    }
    fn raw_schema(
        name: String,
        description: Option<String>,
        input_schema: Map<String, Value>,
        meta: Option<Map<String, Value>>,
    ) -> Value {
        raw_schema(&name, description.as_deref(), &input_schema, meta.as_ref())
    }
    fn plan_setup(reload: bool, toolset_initialized: bool, server_connected: bool) -> String {
        match plan_setup(reload, toolset_initialized, server_connected) {
            SetupPlan::Skip => "skip",
            SetupPlan::ConnectThenList => "connect_then_list",
            SetupPlan::InvalidateThenList => "invalidate_then_list",
            SetupPlan::ListOnly => "list_only",
        }
        .into()
    }
    fn http_transport(url: String, transport_type: Option<String>) -> Result<String, McpError> {
        Ok(match http_transport(&url, transport_type.as_deref())? {
            HttpTransport::Sse => "sse",
            HttpTransport::StreamableHttp => "streamable_http",
        }
        .into())
    }
    fn render_result(tool_name: String, result: Value) -> Result<String, McpError> {
        render_result(&tool_name, &result)
    }
    fn check_result(result: Value) -> Result<(), McpError> {
        check_result(&result)
    }
}
