//! The process launcher's typed command line, and the agent server the dev channel reports on.

/// The launcher's `--log-level`; `warning` and `critical` are aliases of `warn` and `fatal`.
#[crate::host(record(module = "cli", extract, emit))]
#[crate::host(wire_strs(msg = format!("unknown log level {s:?}")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
    Off,
}

impl LogLevel {
    /// The level on the host's numeric scale (Python `logging`'s): `trace` is 5, `off` is above
    /// `fatal`.
    pub const fn host_level(self) -> u32 {
        match self {
            LogLevel::Trace => 5,
            LogLevel::Debug => 10,
            LogLevel::Info => 20,
            LogLevel::Warn => 30,
            LogLevel::Error => 40,
            LogLevel::Fatal => 50,
            LogLevel::Off => 51,
        }
    }
}

/// The command accepted by the process launcher.
#[derive(Clone, PartialEq)]
#[crate::host(record(module = "cli", emit, extract))]
pub enum CliArgs {
    Start {
        log_level: LogLevel,
        dev: bool,
        cli_addr: Option<String>,
        simulation: bool,
        /// The server options the command line gives (`--url`, `--api-key`, `--api-secret`,
        /// `--shutdown-timeout`, `--drain-timeout`), over the `AgentServer`'s own.
        options: crate::worker::types::ServerOptions,
        /// The process tuning the command line gives.
        process: crate::worker::types::ProcessOptions,
    },
    Console {
        connect_addr: String,
        record: bool,
    },
}

/// Parsing never prints or exits the hosting process.
#[derive(Clone, PartialEq)]
#[crate::host(record(module = "cli", emit))]
pub enum CliParseResult {
    Parsed { args: CliArgs },
    Display { message: String, exit_code: i32, stderr: bool },
}

/// Invalid launcher configuration.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[crate::host(error(kind = "livekit.agents._bindings.CLIError"))]
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

    /// [`LogLevel::host_level`].
    #[host(function)]
    fn host_log_level(level: LogLevel) -> u32;

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
        server: std::sync::Arc<dyn crate::worker::server_host::ServerJobs>,
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
