// The generated module globs this one to resolve field types.
pub use crate::llm::provider_format::tools::{DuplicateMode, DuplicateScope, ToolFlag};

#[crate::host(file(module = "mcp"))]
const _: () = ();

/// Resolved per-tool behavior passed to the host's function-tool decorator.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[crate::host(record(module = "mcp", name = "ResolvedToolOptions", emit, typed_dict))]
pub struct McpToolOptions {
    /// Passed through to the `@function_tool` decorator, using `ToolFlag`.
    pub flags: ToolFlag,
    pub on_duplicate: DuplicateMode,
    pub duplicate_scope: DuplicateScope,
    /// Forward the tool's MCP progress notifications to `ctx.update()`. Also selects the
    /// non-blocking call path (`_tool_called_nonblocking`, which takes a `RunContext`).
    pub report_progress: bool,
    /// The tool's `retries=`; `None` reads the run's tool budget.
    pub retries: Option<u32>,
    /// This tool's request timeout in seconds, in place of the server's `request_timeout`.
    pub timeout: Option<f64>,
}

/// Per-tool behavior for MCP tools exposed by an MCPToolset.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[crate::host(record(
    module = "mcp",
    path = "livekit.agents.llm.mcp",
    name = "MCPToolOptions",
    extract,
    typed_dict
))]
pub struct McpToolOptionsPatch {
    /// Flags passed to the @function_tool decorator, using ToolFlag.
    #[host(default = None)]
    pub flags: Option<ToolFlag>,
    /// Behavior when a tool is called multiple times in the same context.
    #[host(default = None)]
    pub on_duplicate: Option<DuplicateMode>,
    /// What counts as a duplicate, e.g. ``"name_and_args"`` to compare arguments too.
    #[host(default = None)]
    pub duplicate_scope: Option<DuplicateScope>,
    /// Whether to forward the tool's progress notifications to ctx.update().
    #[host(default = None)]
    pub report_progress: Option<bool>,
    /// The tool's retry budget, in place of the agent's ``retries['tools']``.
    #[host(default = None)]
    pub retries: Option<u32>,
    /// Seconds one call may wait without a reply or progress, in place of the server's
    /// ``request_timeout``.
    #[host(default = None)]
    pub timeout: Option<f64>,
}

/// Seconds a connection stays open after its toolset's last run ends, for the next run to reuse:
/// `MCPToolset(idle_timeout=)`'s default.
#[crate::host(constant)]
pub const MCP_IDLE_CLOSE_SECONDS: f64 = 60.0;

/// The MCP limits a caller sets, by their public argument names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[crate::host(wire_strs(msg = format!("unknown MCP limit {s:?}")))]
pub enum McpTimeout {
    #[host(rename = "init_timeout")]
    Init,
    #[host(rename = "request_timeout")]
    Request,
    #[host(rename = "read_timeout")]
    Read,
    /// A tool's own `MCPToolOptions(timeout=)`.
    #[host(rename = "timeout")]
    Tool,
    /// `MCPToolset(idle_timeout=)`.
    #[host(rename = "idle_timeout")]
    Idle,
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
    #[error("{0} must be a finite number of seconds >= 0 and <= 100 years")]
    #[host(kind = "ValueError")]
    InvalidTimeout(McpTimeout),
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

/// A `report_progress` tool's progress messages: its `RunContext.update()`, run in the host
/// context captured at the MCP operation's entry.
#[crate::host(proxy(host = "livekit.agents.voice.events.RunContext"))]
pub trait McpProgress: Send + Sync {
    #[host(await)]
    fn update<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] invocation: &'a crate::invoke::CallContext,
        message: &'a str,
    ) -> crate::BoxFuture<'a, Result<(), crate::errors::HostError>>;
}

/// An SDK session whose owner cancels and joins independently of request borrowers.
#[crate::host(class(name = "McpSession"))]
pub trait McpSession: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// `init_timeout` bounds connecting, the handshake and a listing; `request_timeout` one
    /// request without a reply or progress, `None` for no limit; `read_timeout` an idle HTTP
    /// reply stream.
    #[host(new)]
    fn new(
        env: &Self::Env,
        init_timeout: f64,
        request_timeout: Option<f64>,
        #[host(default = None)] read_timeout: Option<f64>,
    ) -> Result<Self, McpError>;
    #[host(getter)]
    fn connected(&self) -> bool;
    /// Seconds until a failed session may reconnect; zero when it may now.
    #[host(getter)]
    fn retry_in(&self) -> f64;
    #[host(await)]
    async fn connect_http(
        &self,
        env: &Self::Env,
        url: String,
        headers: std::collections::BTreeMap<String, String>,
        auth: Option<String>,
    ) -> Result<(), McpError>;
    #[host(await)]
    async fn connect_stdio(
        &self,
        command: String,
        args: Vec<String>,
        #[host(default = None)] env: Option<std::collections::BTreeMap<String, String>>,
        #[host(default = None)] cwd: Option<String>,
    ) -> Result<(), McpError>;
    #[host(await)]
    async fn list_tools(&self) -> Result<serde_json::Value, McpError>;
    #[host(await)]
    async fn call_tool(
        &self,
        name: String,
        arguments: serde_json::Map<String, serde_json::Value>,
        timeout: Option<f64>,
        progress: Option<std::sync::Arc<dyn McpProgress>>,
        host_context: Option<crate::utils::host_ref::HostRef>,
    ) -> Result<serde_json::Value, McpError>;
    #[host(await)]
    async fn close(&self);
}

/// `MCPServerStdio`: an MCP server run as a child process, over its stdio; restarted if it exits.
#[crate::host(facade(
    path = "livekit.agents.llm.mcp",
    name = "MCPServerStdio",
    native = McpSession,
    keep(command, args, env, cwd)
))]
pub trait McpServerStdioFacade {
    /// Args:
    ///     command: The program to run.
    ///     args: Its arguments.
    ///     env: Environment variables, over a minimal inherited set.
    ///     cwd: Its working directory.
    ///     init_timeout: Seconds to start it, complete the handshake and list the tools.
    ///     request_timeout: Seconds one request may go without a reply or progress; ``None`` for
    ///         no limit. ``MCPToolOptions(timeout=)`` sets one tool's.
    #[host(new)]
    fn new(
        command: String,
        args: Vec<String>,
        #[host(kw, default = None)] env: Option<std::collections::BTreeMap<String, String>>,
        #[host(default = None)] cwd: Option<String>,
        #[host(default = 5.0)] init_timeout: f64,
        #[host(default = 30.0)] request_timeout: Option<f64>,
    ) -> Self;

    /// Whether the server is connected.
    #[host(getter, native = connected)]
    fn initialized(&self) -> bool;

    /// Start the server and complete the MCP handshake.
    #[host(
        native = connect_stdio,
        bind(command = "_command", args = "_args", env = "_env", cwd = "_cwd")
    )]
    async fn initialize(&self) -> Result<(), McpError>;

    /// The server's tool listing, every page.
    async fn list_tools(&self) -> Result<serde_json::Value, McpError>;

    #[host(native = close)]
    async fn aclose(&self);
}

/// `MCPServerHTTP`'s connection: an MCP server over Streamable HTTP. The handwritten subclass
/// takes the credentials, which can be functions of a run, and connects with them.
#[crate::host(facade(
    path = "livekit.agents.llm.mcp",
    extended,
    name = "MCPServerHTTP",
    native = McpSession,
    keep(url)
))]
pub trait McpServerHttpFacade {
    /// Args:
    ///     url: The server's MCP endpoint.
    ///     init_timeout: Seconds to connect, complete the handshake and list the tools.
    ///     read_timeout: Seconds a reply stream may stay idle; ``None`` for no limit.
    ///     request_timeout: Seconds one request may go without a reply or progress; ``None`` for
    ///         no limit. ``MCPToolOptions(timeout=)`` sets one tool's.
    #[host(new)]
    fn new(
        url: String,
        #[host(kw, default = 5.0)] init_timeout: f64,
        #[host(default = 300.0)] read_timeout: Option<f64>,
        #[host(default = 30.0)] request_timeout: Option<f64>,
    ) -> Self;

    /// Whether the server is connected.
    #[host(getter, native = connected)]
    fn initialized(&self) -> bool;

    /// The server's tool listing, every page.
    async fn list_tools(&self) -> Result<serde_json::Value, McpError>;

    #[host(native = close)]
    async fn aclose(&self);
}

/// What an `MCPToolset`'s lifecycle runs on the host: its listing and its idle close. Each
/// runs in the context of the `setup()` or `aclose()` that started it.
#[crate::host(proxy(host = "livekit.agents.llm.mcp.MCPToolset"))]
pub trait McpToolsetHost: Send + Sync {
    /// Connect the server when it is not, list its tools and make them the toolset's.
    /// Cancelled when the toolset is closed while recovering.
    #[host(await, name = "_core_list")]
    fn list<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a crate::invoke::CallContext,
        #[host(cancel)] cancel: std::sync::Arc<dyn crate::invoke::Cancellation>,
    ) -> crate::BoxFuture<'a, Result<(), crate::errors::HostError>>;
    /// The connection idled: drop the tools and close the server, which a later `setup()`
    /// connects again.
    #[host(await, name = "_core_close_idle")]
    fn close_idle<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        #[host(context)] context: &'a crate::invoke::CallContext,
    ) -> crate::BoxFuture<'a, Result<(), crate::errors::HostError>>;
}

/// An `MCPToolset`'s lifecycle (docs/tools.md "MCP servers"): one listing at a time; an
/// unreachable server leaves the toolset without tools and is listed again in the background at
/// the session's backoff until it answers; the connection closes `idle_timeout` seconds after the
/// last run released the toolset unless a run sets it up again first. `host` is the toolset,
/// held only while a recovery or an idle close is pending; both end when the lifecycle drops.
#[crate::host(class(name = "_core_McpToolsetLifecycle"))]
pub trait McpToolsetLifecycle: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(class(name = "McpSession"))]
    type Session: Send + Sync + 'static;

    /// `server` names the server in logs; `session` owns the reconnect backoff; `idle_timeout`
    /// is the seconds an unused connection stays open ([`MCP_IDLE_CLOSE_SECONDS`] by default).
    #[host(new)]
    fn new(
        env: &Self::Env,
        server: String,
        session: std::sync::Arc<Self::Session>,
        idle_timeout: f64,
    ) -> Result<Self, McpError>;
    /// Disarm a pending idle close, then list the tools unless they are listed or a recovery
    /// is under way. A failed listing is logged and recovered in the background; a cancelled
    /// setup cancels its listing.
    #[host(await)]
    async fn setup(
        &self,
        host: std::sync::Arc<dyn McpToolsetHost>,
        #[host(context)] context: crate::invoke::CallContext,
        cx: crate::telemetry::Cx,
    );
    /// The last run released the toolset: stop a recovery (its listing cancelled and settled)
    /// and arm the idle close.
    #[host(await)]
    async fn release(
        &self,
        host: std::sync::Arc<dyn McpToolsetHost>,
        #[host(context)] context: crate::invoke::CallContext,
        cx: crate::telemetry::Cx,
    );
}

/// Pure options and SDK result interpretation used by the Python authoring adapter.
#[crate::host(functions)]
pub trait McpTools {
    #[host(function, name = "resolve_mcp_tool_options")]
    fn resolve_tool_options(
        config: Option<McpToolOptionsPatch>,
    ) -> Result<McpToolOptions, McpError>;
    #[host(function, name = "mcp_raw_schema")]
    fn raw_schema(
        name: String,
        description: Option<String>,
        input_schema: serde_json::Map<String, serde_json::Value>,
        meta: Option<serde_json::Map<String, serde_json::Value>>,
    ) -> serde_json::Value;
    #[host(function, name = "mcp_render_result")]
    fn render_result(tool_name: String, result: serde_json::Value) -> Result<String, McpError>;
}
