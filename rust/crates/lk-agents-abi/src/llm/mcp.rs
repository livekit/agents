//! MCP host contracts. Runtime owns the SDK, transports, requests and cleanup.

use lk_agents_host_derive::Host;

// The generated module globs this one to resolve field types.
pub use crate::llm::provider_format::tools::{DuplicateMode, DuplicateScope, ToolFlag};

/// Resolved per-tool behavior passed to the host's function-tool decorator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Host)]
#[binding(emit, from_items)]
pub struct McpToolOptions {
    /// Passed through to the `@function_tool` decorator, using `ToolFlag`.
    pub flags: ToolFlag,
    pub on_duplicate: DuplicateMode,
    pub duplicate_scope: DuplicateScope,
    /// Forward the tool's MCP progress notifications to `ctx.update()`. Also selects the
    /// non-blocking call path (`_tool_called_nonblocking`, which takes a `RunContext`).
    pub report_progress: bool,
}

/// Per-tool behavior for MCP tools exposed by an MCPToolset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Host)]
#[python(
    typed_dict,
    total = "False",
    module = "mcp",
    path = "livekit.agents.llm.mcp",
    name = "MCPToolOptions"
)]
#[binding(extract, from_items)]
pub struct McpToolOptionsPatch {
    /// Flags passed to the @function_tool decorator, using ToolFlag.
    #[python(ty = "ToolFlag", import = "..llm.tool_context:ToolFlag")]
    pub flags: Option<ToolFlag>,
    /// Behavior when a tool is called multiple times in the same context.
    #[python(ty = "DuplicateMode", import = "..llm.tool_context:DuplicateMode")]
    pub on_duplicate: Option<DuplicateMode>,
    /// What counts as a duplicate, e.g. ``"name_and_args"`` to compare arguments too.
    #[python(ty = "DuplicateScope", import = "..llm.tool_context:DuplicateScope")]
    pub duplicate_scope: Option<DuplicateScope>,
    /// Whether to forward the tool's progress notifications to ctx.update().
    #[python(ty = "bool")]
    pub report_progress: Option<bool>,
}

/// MCP failures preserve their native cause, including the upstream SDK error.
#[derive(Debug, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
pub enum McpError {
    #[error("MCP service is unavailable")]
    #[host(kind = "livekit.agents.llm.tool_context.ToolError")]
    Unavailable,
    #[error("Tool '{0}' completed without producing a result")]
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
    #[error("{0} must be a finite number of seconds >= 0 and <= 100 years")]
    #[host(kind = "ValueError")]
    InvalidTimeout(String),
    #[error("MCP protocol error {code}: {message}")]
    Protocol {
        code: i32,
        message: String,
        data: Option<serde_json::Value>,
        #[source]
        cause: McpCause,
    },
    #[error("MCP request timed out after {timeout:?}")]
    #[host(kind = "TimeoutError")]
    Timeout {
        timeout: std::time::Duration,
        #[source]
        cause: McpCause,
    },
    #[error("MCP request cancelled")]
    Cancelled(#[source] McpCause),
    #[error("MCP transport failed: {0}")]
    #[host(kind = "ConnectionError")]
    Transport(#[source] McpCause),
    #[error("MCP initialization failed: {0}")]
    Initialization(#[source] McpCause),
    #[error("Invalid MCP response: {0}")]
    Decode(#[source] McpCause),
    #[error("Unexpected MCP response")]
    UnexpectedResponse,
    #[error(transparent)]
    #[host(delegate)]
    Host(#[from] crate::errors::HostError),
}

pub type McpCause = std::sync::Arc<dyn std::error::Error + Send + Sync>;

/// An observed cleanup outcome; a deadline never implies successful cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[crate::host(record(module = "mcp_session", emit))]
pub enum McpCleanupStatus {
    #[default]
    Pending,
    Clean,
    Incomplete,
}

/// Evidence retained by the session owner after close, including native child wait receipts.
#[derive(Debug, Clone, Default)]
#[crate::host(record(module = "mcp_session", emit))]
pub struct McpCleanup {
    pub status: McpCleanupStatus,
    pub service_settled: bool,
    pub calls_settled: bool,
    pub reader_settled: bool,
    pub child_exit_code: Option<i32>,
    pub child_exit_signal: Option<i32>,
    pub child_reaped: bool,
}

/// The host context captured at the MCP operation's entry.
#[derive(Debug, Clone, Default)]
#[crate::host(record(module = "mcp_session", emit))]
pub struct McpInvocation {
    pub host_context: Option<crate::utils::host_ref::HostRef>,
}

impl crate::agent::HostContext for McpInvocation {
    fn host_context(&self) -> Option<&crate::utils::host_ref::HostRef> {
        self.host_context.as_ref()
    }
}

/// The host's SDK streams. Dropping receive cancels the host coroutine through the shared bridge.
#[crate::host(proxy(host = "livekit.agents.llm.mcp._Streams"))]
#[async_trait::async_trait]
pub trait McpTransport: Send + Sync {
    #[host(await, name = "_core_start")]
    async fn start(
        &self,
        #[host(context)] invocation: &McpInvocation,
    ) -> Result<(), crate::errors::HostError>;
    #[host(await, name = "_core_send")]
    async fn send(
        &self,
        #[host(context)] invocation: &McpInvocation,
        message: &serde_json::Value,
    ) -> Result<(), crate::errors::HostError>;
    #[host(await, name = "_core_receive")]
    async fn receive(
        &self,
        #[host(context)] invocation: &McpInvocation,
    ) -> Result<serde_json::Value, crate::errors::HostError>;
    #[host(await, name = "_core_close")]
    async fn close(
        &self,
        #[host(context)] invocation: &McpInvocation,
    ) -> Result<(), crate::errors::HostError>;
}

#[crate::host(proxy(host = "livekit.agents.llm.mcp._Progress"))]
#[async_trait::async_trait]
pub trait McpProgress: Send + Sync {
    #[host(await, name = "_core_update")]
    async fn update(
        &self,
        #[host(context)] invocation: &McpInvocation,
        message: &str,
    ) -> Result<(), crate::errors::HostError>;
}

/// An SDK session whose owner cancels and joins independently of request borrowers.
#[crate::host(class(name = "McpSession"))]
#[async_trait::async_trait]
pub trait McpSession: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(client_session_timeout_seconds: f64) -> Result<Self, McpError>;
    #[host(getter)]
    fn connected(&self) -> bool;
    #[host(call)]
    fn set_allowed_tools(&self, allowed: Vec<String>);
    #[host(call)]
    fn invalidate_cache(&self);
    #[host(await)]
    async fn connect_http(
        &self,
        url: String,
        headers: Vec<(String, String)>,
        transport: String,
        timeout: f64,
        sse_read_timeout: f64,
    ) -> Result<(), McpError>;
    #[host(await)]
    async fn connect_stdio(
        &self,
        command: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
        cwd: Option<String>,
    ) -> Result<(), McpError>;
    #[host(await)]
    async fn connect_host(
        &self,
        transport: std::sync::Arc<dyn McpTransport>,
        host_context: Option<crate::utils::host_ref::HostRef>,
    ) -> Result<(), McpError>;
    #[host(await)]
    async fn list_tools(&self) -> Result<serde_json::Value, McpError>;
    #[host(await)]
    async fn call_tool(
        &self,
        name: String,
        arguments: serde_json::Map<String, serde_json::Value>,
        progress: Option<std::sync::Arc<dyn McpProgress>>,
        host_context: Option<crate::utils::host_ref::HostRef>,
    ) -> Result<serde_json::Value, McpError>;
    #[host(await)]
    async fn close(&self);
    #[host(getter)]
    fn cleanup(&self) -> McpCleanup;
}

/// Pure options and SDK result interpretation used by the Python authoring adapter.
#[crate::host(functions)]
pub trait McpTools {
    #[host(function, name = "resolve_mcp_tool_options")]
    fn resolve_tool_options(config: Option<McpToolOptionsPatch>) -> McpToolOptions;
    #[host(function, name = "mcp_raw_schema")]
    fn raw_schema(
        name: String,
        description: Option<String>,
        input_schema: serde_json::Map<String, serde_json::Value>,
        meta: Option<serde_json::Map<String, serde_json::Value>>,
    ) -> serde_json::Value;
    #[host(function, name = "mcp_plan_setup")]
    fn plan_setup(reload: bool, toolset_initialized: bool, server_connected: bool) -> String;
    #[host(function, name = "mcp_http_transport")]
    fn http_transport(url: String, transport_type: Option<String>) -> Result<String, McpError>;
    #[host(function, name = "mcp_render_result")]
    fn render_result(tool_name: String, result: serde_json::Value) -> Result<String, McpError>;
    #[host(function, name = "mcp_check_result")]
    fn check_result(result: serde_json::Value) -> Result<(), McpError>;
}

/// Validate a scalar MCP deadline with the shared connection-policy duration range.
pub fn timeout_duration(field: &str, seconds: f64) -> Result<std::time::Duration, McpError> {
    crate::types::connect::duration_from_seconds(seconds)
        .ok_or_else(|| McpError::InvalidTimeout(field.to_owned()))
}

/// Validate public MCP constructor arguments before a transport can be opened.
#[crate::host(export(name = "_core_mcp_timeout_check", pid = false))]
pub fn mcp_timeout_check(field: &str, seconds: f64) -> Result<(), McpError> {
    timeout_duration(field, seconds).map(|_| ())
}
