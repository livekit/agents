//! The process launcher's typed command line, and the agent server the dev channel reports on.

/// The command accepted by the process launcher.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "cli", emit, extract))]
pub enum CliArgs {
    Start {
        log_level: String,
        dev: bool,
        cli_addr: Option<String>,
        simulation: bool,
        ws_url: Option<String>,
        api_key: Option<String>,
        api_secret: Option<String>,
        shutdown_timeout: Option<f64>,
        drain_timeout: Option<f64>,
    },
    Console {
        connect_addr: String,
        record: bool,
    },
}

/// Parsing never prints or exits the hosting process.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "cli", emit))]
pub enum CliParseResult {
    Parsed { args: CliArgs },
    Display { message: String, exit_code: i32, stderr: bool },
}

/// Invalid launcher configuration.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[crate::host(error(kind = "livekit.agents._exceptions.CLIError"))]
pub enum CliError {
    #[error("{0}")]
    Invalid(String),
}

/// A language host's normalized process log record.
#[derive(Debug, Clone, Default, PartialEq)]
#[crate::host(record(module = "cli", extract))]
pub struct LogRecord {
    pub level: String,
    pub logger: String,
    pub process_id: Option<i64>,
    /// Unix seconds.
    pub timestamp: f64,
    pub message: String,
    pub fields: serde_json::Map<String, serde_json::Value>,
    pub exception: Option<String>,
    pub stack: Option<String>,
}

/// Shared command parsing, structured logging and the remote launcher connection.
#[crate::host(functions)]
pub trait Cli {
    #[host(function)]
    fn parse_cli(argv: Vec<String>) -> CliParseResult;

    /// One structured JSON line, including host-formatted exceptions and stack traces.
    #[host(function, pid = false)]
    fn format_log_record(record: LogRecord) -> String;

    /// The development launcher connection.
    #[host(function)]
    async fn dev_channel(
        cli_addr: String,
        agent_name: String,
        url: String,
        api_key: String,
        api_secret: String,
        server: std::sync::Arc<dyn DevServer>,
    );
}

/// Native enforcement of the process's one shutdown deadline.
#[crate::host(class(name = "Shutdown"))]
pub trait CliShutdown: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(timeout: f64, drain_timeout: f64) -> Result<Self, CliError>;

    /// Own process termination signals. Fails if another launcher is already listening.
    #[host(call)]
    fn listen(&self) -> Result<(), CliError>;

    /// Wait until shutdown starts, whether requested by the host or a native signal.
    #[host(await)]
    async fn wait(&self);

    /// Begin shutdown once. Repeated requests never extend the deadline.
    #[host(call)]
    fn begin(&self) -> bool;

    /// Seconds left until the deadline, or the configured budget before shutdown begins.
    #[host(call)]
    fn remaining(&self) -> f64;

    /// Drain time remaining while reserving the configured cleanup budget.
    #[host(call)]
    fn drain_remaining(&self) -> f64;

    /// Disarm only after host cleanup and event-loop closure finish.
    #[host(call)]
    fn closed(&self);
}

/// A job running under the dev server, kept across reloads: the `Job` protobuf as bytes, the
/// accept arguments (name, identity, metadata), the server URL, the token, the worker id, and
/// whether it is a fake job.
pub type DevJob = (Vec<u8>, String, String, String, String, String, String, bool);

/// The agent server the dev channel reports on (`cli.py`'s `_DevServer`).
#[crate::host(proxy(host = "livekit.agents.cli._DevServer"))]
pub trait DevServer: Send + Sync {
    /// The jobs running now, read when the CLI asks.
    #[host(call, name = "_core_running_jobs")]
    fn running_jobs(&self) -> Result<Vec<DevJob>, crate::errors::HostError>;
    /// Resume the jobs the CLI kept from before the reload, on this server's URL and a fresh token.
    #[host(await, name = "_core_reload_jobs")]
    fn reload_jobs(
        &self,
        cx: &crate::telemetry::Cx,
        jobs: Vec<DevJob>,
    ) -> crate::BoxFuture<'_, Result<(), crate::errors::HostError>>;
}
