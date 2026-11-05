//! A job runner's completion keeps its OS exit status separate from its control protocol's failure;
//! its supervision settings and its job's status.

#[derive(Clone, Debug, PartialEq, Eq)]
#[crate::host(record(module = "worker_ipc", emit))]
pub enum IpcFailure {
    Io { kind: String, os_error: Option<i32>, message: String },
    Malformed { message: String },
    TooLarge { size: u64 },
    Unexpected { message: String },
    Version { received: u32, expected: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[crate::host(record(module = "worker_ipc", emit))]
pub struct JobCompletion {
    pub exit_code: Option<i32>,
    pub failure: Option<IpcFailure>,
}

#[crate::host(record(
    module = "worker_ipc",
    path = "livekit.agents.ipc",
    enum,
    enum_base = "Enum",
    emit
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobStatus {
    #[host(rename = "running")]
    Running,
    #[host(rename = "failed")]
    Failed,
    #[host(rename = "success")]
    Success,
}

/// How the worker supervises one job runner. The defaults are the production values.
#[derive(Clone, Debug, PartialEq)]
#[crate::host(record(module = "worker_ipc", path = "livekit.agents.ipc", extract))]
pub struct JobRunnerOptions {
    /// The runner is a thread of this process, else a process. A thread has no pong deadline
    /// nor memory limit, whatever is set here: it cannot be stopped.
    #[host(default = false)]
    pub thread: bool,
    /// How long the runner has to be ready, and later to take its job.
    pub initialize_timeout: f64,
    /// How long a closing runner has to end before it is stopped.
    pub close_timeout: f64,
    /// The footprint past which a warning is logged; 0: none.
    #[host(default = 0.0)]
    pub memory_warn_mb: f64,
    /// The footprint past which the process is killed; 0: none.
    #[host(default = 0.0)]
    pub memory_limit_mb: f64,
    /// Seconds between pings; 0: none.
    #[host(default = 2.5)]
    pub ping_interval: f64,
    /// A process silent this long is killed; 0: a late pong is only reported.
    #[host(default = 60.0)]
    pub ping_timeout: f64,
    /// A pong later than this is reported.
    #[host(default = 0.5)]
    pub high_ping_threshold: f64,
    /// The job's HTTP proxy, handed over at initialization.
    #[host(default = None)]
    pub http_proxy: Option<String>,
    /// The runner's event loop runs in debug mode.
    #[host(default = false)]
    pub asyncio_debug: bool,
}
