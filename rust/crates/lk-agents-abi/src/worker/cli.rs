//! The process launcher's typed command line, and the agent server the dev channel reports on.

/// The command accepted by the process launcher.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "worker_cli", emit, extract))]
pub enum CliArgs {
    Start {
        entrypoint: Option<String>,
        log_level: String,
        log_format: String,
        log_style: String,
        dev: bool,
        cli_addr: Option<String>,
        simulation: bool,
        ws_url: Option<String>,
        api_key: Option<String>,
        api_secret: Option<String>,
    },
    Console {
        entrypoint: Option<String>,
        connect_addr: String,
        record: bool,
    },
    DownloadFiles,
}

/// A command line or entrypoint the launcher cannot run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[crate::host(error(kind = "livekit.agents._exceptions.CLIError"))]
pub enum CliError {
    #[error(
        "{0}\nusage: python -m livekit.agents {{start,console,download-files}} [entrypoint] [options]"
    )]
    Usage(String),
    #[error("Could not find a default file to run, please provide an explicit path")]
    NoDefaultFile,
    #[error("Path does not exist {0}")]
    NoPath(String),
}

/// The typed command parser and filesystem entrypoint discovery implemented by the launcher.
#[crate::host(functions)]
pub trait WorkerCli {
    #[host(function)]
    fn parse_cli(argv: Vec<String>) -> Result<CliArgs, CliError>;

    #[host(function)]
    fn entrypoint_module(path: Option<String>) -> Result<(String, String), CliError>;
}

/// The process launcher's signal and final event-loop cleanup owner.
#[crate::host(class(name = "ExitSignals"))]
pub trait CliSignals: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(#[host(default = true)] escalate: bool) -> Self;

    #[host(call)]
    fn signal(&self, exit_served: bool) -> String;

    #[host(call)]
    fn served(&self);

    #[host(call)]
    fn draining(&self);

    /// Arm the native deadline before the host cancels tasks or closes its executor.
    #[host(call)]
    fn closing(&self);

    /// Disarm the deadline only after the host event loop has actually closed.
    #[host(call)]
    fn closed(&self);
}

/// A job running under the dev server, kept across reloads: the `Job` protobuf as bytes, the
/// accept arguments (name, identity, metadata), the server URL, the token, the worker id, and
/// whether it is a fake job.
pub type DevJob = (Vec<u8>, String, String, String, String, String, String, bool);

/// The agent server the dev channel reports on (`cli.py`'s `_DevServer`).
#[crate::host(proxy(host = "livekit.agents.cli._DevServer"))]
#[async_trait::async_trait]
pub trait DevServer: Send + Sync {
    /// The jobs running now, read when the CLI asks.
    #[host(call, name = "_core_running_jobs")]
    fn running_jobs(&self) -> Result<Vec<DevJob>, crate::errors::HostError>;
    /// Resume the jobs the CLI kept from before the reload, on this server's URL and a fresh token.
    #[host(await, name = "_core_reload_jobs")]
    async fn reload_jobs(&self, jobs: Vec<DevJob>) -> Result<(), crate::errors::HostError>;
}
