//! MCP tool logic: option resolution, schemas, setup decisions, and the
//! interpretation of tool results. The session and its transports are the runtime's I/O.

use std::sync::Arc;

use lk_agents_abi::llm::mcp::{
    McpError, McpToolOptions, McpToolOptionsPatch, McpTools as McpToolsPort,
};
use serde_json::{Map, Value, json};

pub mod http;

/// `_resolve_tool_options`: `{**defaults, **config}`, key by key.
pub fn resolve_tool_options(config: Option<&McpToolOptionsPatch>) -> McpToolOptions {
    let (d, p) = (McpToolOptions::default(), config.copied().unwrap_or_default());
    McpToolOptions {
        flags: p.flags.unwrap_or(d.flags),
        on_duplicate: p.on_duplicate.unwrap_or(d.on_duplicate),
        duplicate_scope: p.duplicate_scope.unwrap_or(d.duplicate_scope),
        report_progress: p.report_progress.unwrap_or(d.report_progress),
        retries: p.retries,
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

pub fn plan_setup(toolset_initialized: bool, server_connected: bool) -> McpSetup {
    match (toolset_initialized, server_connected) {
        (true, true) => McpSetup::Skip,
        (_, false) => McpSetup::ConnectThenList,
        (false, true) => McpSetup::ListOnly,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpSetup {
    Skip,
    ConnectThenList,
    ListOnly,
}

/// One transport's teardown as the runtime observes it: whether it closed, the readers still
/// holding it, and the first failure every later call reports.
#[derive(Debug, Default)]
pub struct TransportSettlement {
    closed: bool,
    readers: usize,
    failure: Option<Arc<McpError>>,
}

impl TransportSettlement {
    pub fn reader_opened(&mut self) {
        self.readers += 1;
    }
    pub fn reader_closed(&mut self) {
        self.readers -= 1;
    }
    /// The first failure is the cause every later call reports.
    pub fn fail(&mut self, error: McpError) {
        self.failure.get_or_insert_with(|| Arc::new(error));
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
}

/// One established connection's lifecycle: calls borrow it until close, and it settles once the
/// SDK service joined and the last call returned.
#[derive(Debug, Default)]
pub struct ConnectionSettlement {
    closing: bool,
    calls: usize,
    settled: bool,
}

impl ConnectionSettlement {
    pub fn borrow(&mut self) -> Result<(), McpError> {
        if self.closing {
            return Err(McpError::Unavailable);
        }
        self.calls += 1;
        Ok(())
    }
    pub fn release(&mut self) {
        self.calls -= 1;
    }
    pub fn close(&mut self) {
        self.closing = true;
    }
    /// Settles once no call holds the connection; whether it is settled.
    pub fn settle_when_idle(&mut self) -> bool {
        self.settled |= self.calls == 0;
        self.settled
    }
    pub fn settled(&self) -> bool {
        self.settled
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

/// A scalar MCP deadline within the shared connection-policy duration range.
pub fn timeout_duration(field: &str, seconds: f64) -> Result<std::time::Duration, McpError> {
    lk_agents_abi::types::connect::duration_from_seconds(seconds)
        .ok_or_else(|| McpError::InvalidTimeout(field.to_owned()))
}

pub fn render_result(tool_name: &str, result: &Value) -> Result<String, McpError> {
    Ok(match content(result)? {
        [] => return Err(McpError::EmptyResult(tool_name.into())),
        [one] => one.to_string(),
        many => Value::from(many.to_vec()).to_string(),
    })
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
    fn plan_setup(toolset_initialized: bool, server_connected: bool) -> String {
        match plan_setup(toolset_initialized, server_connected) {
            McpSetup::Skip => "skip",
            McpSetup::ConnectThenList => "connect_then_list",
            McpSetup::ListOnly => "list_only",
        }
        .into()
    }
    fn render_result(tool_name: String, result: Value) -> Result<String, McpError> {
        render_result(&tool_name, &result)
    }
    fn timeout_check(field: &str, seconds: f64) -> Result<(), McpError> {
        timeout_duration(field, seconds).map(drop)
    }
}
