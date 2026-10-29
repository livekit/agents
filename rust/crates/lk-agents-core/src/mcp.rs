//! MCP tool logic (decision 71): option resolution, schemas, setup and transport choice, and the
//! interpretation of tool results. The session and its transports are the runtime's I/O.

use std::sync::Arc;

use lk_agents_abi::llm::mcp::{
    McpCleanup, McpCleanupStatus, McpError, McpToolOptions, McpToolOptionsPatch,
    McpTools as McpToolsPort,
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

/// Incomplete once anything was abandoned or failed; clean only once everything settled.
fn cleanup_status(abandoned: bool, settled: bool) -> McpCleanupStatus {
    if abandoned {
        McpCleanupStatus::Incomplete
    } else if settled {
        McpCleanupStatus::Clean
    } else {
        McpCleanupStatus::Pending
    }
}

/// One transport's teardown as the runtime observes it: its readers, its child process and the
/// remote HTTP session the SDK may fail to delete. A cleanup that failed or overran its deadline
/// is never reported clean.
#[derive(Debug, Default)]
pub struct TransportSettlement {
    closed: bool,
    readers: usize,
    child_pending: bool,
    child_exit: Option<(Option<i32>, Option<i32>)>,
    failure: Option<Arc<McpError>>,
    cleanup_failed: bool,
    deadline_missed: bool,
    remote_session_pending: bool,
}

impl TransportSettlement {
    pub fn reader_opened(&mut self) {
        self.readers += 1;
    }
    pub fn reader_closed(&mut self) {
        self.readers -= 1;
    }
    pub fn child_spawned(&mut self) {
        self.child_pending = true;
    }
    pub fn child_exited(&mut self, code: Option<i32>, signal: Option<i32>) {
        self.child_exit = Some((code, signal));
        self.child_pending = false;
    }
    /// The first failure is the cause every later call reports.
    pub fn fail(&mut self, error: McpError) {
        self.failure.get_or_insert_with(|| Arc::new(error));
    }
    pub fn cleanup_failed(&mut self, error: McpError) {
        self.fail(error);
        self.cleanup_failed = true;
    }
    pub fn missed_deadline(&mut self) {
        self.deadline_missed = true;
    }
    pub fn remote_session(&mut self, pending: bool) {
        self.remote_session_pending = pending;
    }
    pub fn close(&mut self) {
        self.closed = true;
    }
    pub fn failure(&self) -> Option<Arc<McpError>> {
        self.failure.clone()
    }
    /// Closed, and no reader still holds the transport.
    pub fn drained(&self) -> bool {
        self.closed && self.readers == 0
    }
    fn abandoned(&self) -> bool {
        self.cleanup_failed
            || self.child_pending
            || self.deadline_missed
            || self.remote_session_pending
    }
    pub fn cleanup(&self) -> McpCleanup {
        let (child_exit_code, child_exit_signal) = self.child_exit.unwrap_or_default();
        McpCleanup {
            status: if self.cleanup_failed || self.deadline_missed || self.remote_session_pending {
                McpCleanupStatus::Incomplete
            } else {
                cleanup_status(false, self.drained() && !self.child_pending)
            },
            service_settled: self.closed,
            calls_settled: true,
            reader_settled: self.drained(),
            child_reaped: self.child_exit.is_some(),
            child_exit_code,
            child_exit_signal,
        }
    }
}

/// One established connection's lifecycle: calls borrow it until close, and it settles once the
/// SDK service joined and the last call returned. A call cancelled while its host progress
/// callback was pending leaves the cleanup incomplete.
#[derive(Debug, Default)]
pub struct ConnectionSettlement {
    closing: bool,
    calls: usize,
    settled: bool,
    incomplete: bool,
    service_settled: bool,
}

impl ConnectionSettlement {
    pub fn borrow(&mut self) -> Result<(), McpError> {
        if self.closing {
            return Err(McpError::Unavailable);
        }
        self.calls += 1;
        Ok(())
    }
    pub fn release(&mut self, host_callback_pending: bool) {
        self.calls -= 1;
        self.incomplete |= host_callback_pending;
    }
    pub fn close(&mut self) {
        self.closing = true;
    }
    /// `clean`: the service quit by cancellation or close.
    pub fn service_joined(&mut self, clean: bool, transport: &TransportSettlement) {
        self.service_settled = clean;
        self.incomplete |= !clean || transport.abandoned();
    }
    /// Settles once no call holds the connection; whether it is settled.
    pub fn settle_when_idle(&mut self) -> bool {
        self.settled |= self.calls == 0;
        self.settled
    }
    pub fn settled(&self) -> bool {
        self.settled
    }
    pub fn missed_deadline(&mut self) {
        self.incomplete = true;
    }
    pub fn cleanup(&self, transport: &TransportSettlement) -> McpCleanup {
        let mut cleanup = transport.cleanup();
        cleanup.service_settled = self.service_settled;
        cleanup.calls_settled = self.calls == 0;
        cleanup.status = cleanup_status(
            self.incomplete || transport.abandoned(),
            self.settled && cleanup.reader_settled,
        );
        cleanup
    }
}

/// The session's tool listing: the server's tools cached across runs, exposed through the allowed
/// names, fetched page by page when nothing is cached.
#[derive(Debug, Default)]
pub struct ToolListing {
    cache: Option<Vec<Value>>,
    allowed: Vec<String>,
}

/// The next step of one `list_tools` call.
#[derive(Debug)]
pub enum ListStep {
    /// Fetch the page after `cursor` (the first page when `None`), then feed it to
    /// [`ToolListing::page`] with `pages`.
    Fetch { pages: ToolPages, cursor: Option<String> },
    /// The tools the session exposes.
    Done(Vec<Value>),
}

/// The pages one fresh listing has fetched so far.
#[derive(Debug, Default)]
pub struct ToolPages(Vec<Value>);

impl ToolListing {
    pub fn set_allowed(&mut self, allowed: Vec<String>) {
        self.allowed = allowed;
    }
    pub fn invalidate(&mut self) {
        self.cache = None;
    }
    pub fn list(&self) -> ListStep {
        match &self.cache {
            Some(tools) => ListStep::Done(self.allowed(tools.clone())),
            None => ListStep::Fetch { pages: ToolPages::default(), cursor: None },
        }
    }
    /// One fetched page. The complete listing is cached unless its connection is closing: a
    /// closing connection must not repopulate the next run's cache.
    pub fn page(
        &mut self,
        mut pages: ToolPages,
        tools: Vec<Value>,
        next_cursor: Option<String>,
        closing: bool,
    ) -> ListStep {
        pages.0.extend(tools);
        if next_cursor.is_some() {
            return ListStep::Fetch { pages, cursor: next_cursor };
        }
        if !closing {
            self.cache = Some(pages.0.clone());
        }
        ListStep::Done(self.allowed(pages.0))
    }
    /// All the tools, or only the allowed names.
    fn allowed(&self, tools: Vec<Value>) -> Vec<Value> {
        tools
            .into_iter()
            .filter(|tool| {
                self.allowed.is_empty()
                    || tool
                        .get("name")
                        .and_then(Value::as_str)
                        .is_some_and(|n| self.allowed.iter().any(|a| a == n))
            })
            .collect()
    }
}

/// A progress notification reaches the host only when it carries a non-empty message.
pub fn progress_message(message: Option<&str>) -> Option<&str> {
    message.filter(|m| !m.is_empty())
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
