use std::sync::Arc;

use lk_agents_abi::llm::mcp::{
    JsonObject, McpCallResult, McpContent, McpError, McpTimeout, McpTool, McpToolOptions,
    McpToolOptionsPatch, McpTools as McpToolsPort,
};
use serde_json::{Map, Value, json};

pub mod http;

/// `_resolve_tool_options`: `{**defaults, **config}`, key by key.
pub fn resolve_tool_options(
    config: Option<&McpToolOptionsPatch>,
) -> Result<McpToolOptions, McpError> {
    let (d, p) = (McpToolOptions::default(), config.copied().unwrap_or_default());
    if let Some(seconds) = p.timeout {
        timeout_duration(McpTimeout::Tool, seconds)?;
    }
    Ok(McpToolOptions {
        flags: p.flags.unwrap_or(d.flags),
        on_duplicate: p.on_duplicate.unwrap_or(d.on_duplicate),
        duplicate_scope: p.duplicate_scope.unwrap_or(d.duplicate_scope),
        report_progress: p.report_progress.unwrap_or(d.report_progress),
        retries: p.retries,
        timeout: p.timeout,
    })
}

/// The schema a listed tool is offered to the model as.
pub fn raw_schema(tool: &McpTool) -> Map<String, Value> {
    // MCP makes a tool's description optional; a tool spec's is a string.
    let description = tool.description.as_deref().unwrap_or_default();
    let mut schema = Map::new();
    schema.insert("name".into(), json!(tool.name));
    schema.insert("description".into(), json!(description));
    schema.insert("parameters".into(), Value::Object(tool.input_schema.clone()));
    if let Some(meta) = tool.meta.as_ref().filter(|m| !m.is_empty()) {
        schema.insert("meta".into(), Value::Object(meta.clone()));
    }
    schema
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

/// The session's tool listing: the server's tools cached across runs, refetched page by page after
/// each new connection, and the last complete listing while a refetch cannot complete.
#[derive(Debug, Default)]
pub struct ToolListing {
    cache: Option<Vec<McpTool>>,
    stale: bool,
}

/// The next step of one `list_tools` call.
#[derive(Debug)]
pub enum ListStep {
    /// Fetch the page after `cursor` (the first page when `None`), then feed it to
    /// [`ToolListing::page`] with `pages`.
    Fetch { pages: ToolPages, cursor: Option<String> },
    /// The tools the session exposes.
    Done(Vec<McpTool>),
}

/// The pages one fresh listing has fetched so far.
#[derive(Debug, Default)]
pub struct ToolPages(Vec<McpTool>);

impl ToolListing {
    /// A new connection: its server may list other tools.
    pub fn refresh(&mut self) {
        self.stale = true;
    }
    pub fn list(&self) -> ListStep {
        match &self.cache {
            Some(tools) if !self.stale => ListStep::Done(tools.clone()),
            _ => ListStep::Fetch { pages: ToolPages::default(), cursor: None },
        }
    }
    /// The last complete listing, for a refetch that failed.
    pub fn last_known(&self) -> Option<Vec<McpTool>> {
        self.cache.clone()
    }
    /// One fetched page. The complete listing is cached unless its connection is closing: a
    /// closing connection must not repopulate the next run's cache.
    pub fn page(
        &mut self,
        mut pages: ToolPages,
        tools: Vec<McpTool>,
        next_cursor: Option<String>,
        closing: bool,
    ) -> ListStep {
        pages.0.extend(tools);
        if next_cursor.is_some() {
            return ListStep::Fetch { pages, cursor: next_cursor };
        }
        if !closing {
            self.cache = Some(pages.0.clone());
            self.stale = false;
        }
        ListStep::Done(pages.0)
    }
}

/// When a session whose connection failed may connect again: at once after a success, then
/// after a backoff doubling from one second to thirty while attempts keep failing. Calls in
/// between fail fast instead of each waiting out a connect. Time is the session clock's `now`.
#[derive(Debug, Default)]
pub struct ReconnectGate {
    failures: u32,
    retry_at: Option<lk_agents_abi::Ts>,
}

impl ReconnectGate {
    const FIRST: std::time::Duration = std::time::Duration::from_secs(1);
    const MAX: std::time::Duration = std::time::Duration::from_secs(30);

    /// How long from `now` until an attempt is allowed; zero when it is.
    pub fn wait(&self, now: lk_agents_abi::Ts) -> std::time::Duration {
        self.retry_at.map_or(std::time::Duration::ZERO, |at| at.saturating_duration_since(now))
    }
    pub fn failed(&mut self, now: lk_agents_abi::Ts) {
        let backoff = Self::FIRST.saturating_mul(1 << self.failures.min(5)).min(Self::MAX);
        self.failures += 1;
        self.retry_at = Some(now.saturating_add(backoff));
    }
    pub fn succeeded(&mut self) {
        *self = Self::default();
    }
}

/// A progress notification reaches the host only when it carries a non-empty message.
pub fn progress_message(message: Option<&str>) -> Option<&str> {
    message.filter(|m| !m.is_empty())
}

/// A scalar MCP deadline within the shared connection-policy duration range.
pub fn timeout_duration(field: McpTimeout, seconds: f64) -> Result<std::time::Duration, McpError> {
    lk_agents_abi::types::connect::duration_from_seconds(seconds)
        .ok_or(McpError::InvalidTimeout(field))
}

/// What the model reads: a lone text block as its text; other content as its JSON blocks. An
/// error result fails the tool with its content, text blocks as their text.
pub fn render_result(tool_name: &str, result: &McpCallResult) -> Result<String, McpError> {
    if result.is_error {
        let parts = result.content.iter().map(|block| match block {
            McpContent::Text(text) => text.text.clone(),
            other => json(other),
        });
        return Err(McpError::ToolFailed(parts.collect::<Vec<_>>().join("\n")));
    }
    Ok(match result.content.as_slice() {
        [] => return Err(McpError::EmptyResult(tool_name.into())),
        [McpContent::Text(text)] => text.text.clone(),
        [one] => json(one),
        many => json(&many),
    })
}

/// A block as the server sent it; MCP content is plain JSON data.
fn json(blocks: &impl serde::Serialize) -> String {
    serde_json::to_string(blocks).expect("MCP content serializes")
}

pub struct McpTools;
impl McpToolsPort for McpTools {
    fn resolve_tool_options(
        config: Option<McpToolOptionsPatch>,
    ) -> Result<McpToolOptions, McpError> {
        resolve_tool_options(config.as_ref())
    }
    fn raw_schema(tool: McpTool) -> JsonObject {
        JsonObject(raw_schema(&tool))
    }
    fn render_result(tool_name: String, result: McpCallResult) -> Result<String, McpError> {
        render_result(&tool_name, &result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(wire: Value) -> McpCallResult {
        serde_json::from_value(wire).unwrap()
    }

    /// MCP text and resource results reach the model as literal Unicode.
    /// https://github.com/livekit/agents/pull/7565
    #[test]
    fn results_keep_unicode_literal() {
        let many = result(json!({"content": [
            {"type": "text", "text": "réservé à Zürich"},
            {"type": "text", "text": "予約済み"},
        ]}));
        let rendered = render_result("book", &many).unwrap();
        assert!(
            rendered.contains("réservé à Zürich") && rendered.contains("予約済み"),
            "{rendered}"
        );
        assert!(!rendered.contains("\\u"), "no ASCII escaping: {rendered}");

        let resource = result(json!({"content": [
            {"type": "resource", "resource": {"uri": "file:///menu", "text": "crème brûlée"}},
        ]}));
        let rendered = render_result("menu", &resource).unwrap();
        assert!(rendered.contains("crème brûlée") && !rendered.contains("\\u"), "{rendered}");
    }

    /// An MCP error result fails the tool with the result's content as its message.
    /// https://github.com/livekit/agents/pull/4582
    #[test]
    fn an_error_result_fails_with_its_content() {
        let failed = result(json!({"isError": true, "content": [
            {"type": "text", "text": "date indisponible"},
            {"type": "image", "data": "AA==", "mimeType": "image/png"},
        ]}));
        match render_result("book", &failed) {
            Err(McpError::ToolFailed(text)) => {
                let mut lines = text.lines();
                assert_eq!(lines.next(), Some("date indisponible"));
                let image: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
                assert_eq!(image["mimeType"], "image/png");
            }
            other => panic!("expected ToolFailed, got {other:?}"),
        }
    }

    /// A block the model reads as JSON keeps the wire form the server sent, every field in place.
    #[test]
    fn a_block_renders_as_its_wire_form() {
        let link = json!({
            "type": "resource_link",
            "uri": "booking://ABC/receipt",
            "name": "receipt",
            "mimeType": "application/pdf",
            "size": 2048,
            "icons": [{"src": "https://example.com/r.png", "sizes": ["48x48"], "theme": "dark"}],
            "_meta": {"trace": "t-1"},
            "annotations": {"audience": ["user"], "priority": 0.5, "lastModified": "2026-01-01T00:00:00Z"},
        });
        let rendered = render_result("receipt", &result(json!({"content": [link.clone()]})));
        assert_eq!(rendered.unwrap(), link.to_string());
    }
}
