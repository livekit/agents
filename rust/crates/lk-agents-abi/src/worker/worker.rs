//! Worker host contracts. Runtime behavior belongs to `lk-agents-worker`.

/// The worker's terminal drain state, independent of its display text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainOutcome {
    Finished,
    TimedOut,
}

/// Current worker state, available independently of lifecycle notification consumption.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    pub worker_id: String,
    pub draining: bool,
    pub active_jobs: usize,
    pub reserved_slots: usize,
    pub worker_load: f32,
    /// Registered on the current connection; false while it reconnects.
    pub registered: bool,
    pub target_idle_processes: Option<u32>,
    /// Set once the drain completes.
    pub drain_outcome: Option<DrainOutcome>,
    /// `run()` has returned. `Some` carries the reason, `None` while still running.
    pub finished: Option<String>,
}

/// `WorkerPermissions`, field-for-field, including the defaults.
#[crate::host(record(module = "worker_options", dataclass, extract, emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerPermissions {
    #[host(default = true)]
    pub can_publish: bool,
    #[host(default = true)]
    pub can_subscribe: bool,
    #[host(default = true)]
    pub can_publish_data: bool,
    #[host(default = true)]
    pub can_update_metadata: bool,
    /// `models.TrackSource` values, as raw enum numbers.
    #[host(default_factory = "list")]
    pub can_publish_sources: Vec<i32>,
    #[host(default = false)]
    pub hidden: bool,
}

impl Default for WorkerPermissions {
    fn default() -> Self {
        WorkerPermissions {
            can_publish: true,
            can_subscribe: true,
            can_publish_data: true,
            can_update_metadata: true,
            can_publish_sources: Vec::new(),
            hidden: false,
        }
    }
}

/// A job offered to the worker, which answers it with a [`JobDecision`].
#[derive(Debug, Clone, PartialEq)]
pub struct JobRequest {
    pub job: livekit_protocol::Job,
    /// `AvailabilityRequest.resuming` -- the job is migrating from another worker.
    pub resuming: bool,
}

impl JobRequest {
    pub fn room_name(&self) -> &str {
        self.job.room.as_ref().map_or("", |room| room.name.as_str())
    }
}

/// Acceptance returns before assignment and launch; the worker owns the assignment deadline.
/// A rejection leaves the job to another worker.
#[derive(Debug, Clone, PartialEq)]
pub enum JobDecision {
    Accept(crate::worker::job::AcceptArguments),
    Reject,
}

// --- the worker's notifications ------------------------------------------------------------

/// Registration accepted by the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredInfo {
    /// The id the server issued.
    pub worker_id: String,
    pub region: String,
    /// The protocol version the server chose.
    pub protocol: i32,
    pub deployment: String,
    pub agent_name: String,
    /// The server is LiveKit Cloud, which routes the agent's endpoints.
    pub cloud: bool,
}

/// `run()` returned, and why. `kind: "stopped"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerStopped {
    /// `closed`, `worker runtime cancelled`, or the runtime error's own sentence.
    pub reason: String,
}

// --- server launch configuration -----------------------------------------------------------

/// Process-pool settings resolved before the host launch callback.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "worker_server", emit))]
pub struct ProcessPoolOptions {
    pub num_idle_processes: u32,
    pub initialize_timeout: f64,
    pub close_timeout: f64,
    pub session_end_timeout: f64,
    pub memory_warn_mb: f64,
    pub memory_limit_mb: f64,
    pub http_proxy: Option<String>,
}

/// Resolved server settings handed to the process launch adapter.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "worker_server", emit))]
pub struct ProcessStartOptions {
    pub devmode: bool,
    pub job_processes: bool,
    pub env: Vec<(String, String)>,
    pub job_executor_type: crate::worker::job::JobExecutorType,
    pub multiprocessing_context: MultiprocessingContext,
    pub pool: ProcessPoolOptions,
}

/// How job processes start: `forkserver` (Linux only; elsewhere it falls back to `spawn`) or
/// `spawn`.
#[crate::host(record(module = "worker_options", extract, emit))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiprocessingContext {
    Spawn,
    Forkserver,
}

/// `AgentServer(listeners=)`: public traffic and operational information never share a port
/// (docs/server.md "Listeners"). A listener left out keeps its default.
#[crate::host(record(
    module = "worker_options",
    path = "livekit.agents",
    typed_dict,
    extract,
    emit
))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Listeners {
    /// The agent's endpoints and HTTP routes, where the worker serves them (a self-hosted
    /// LiveKit server, the console); `0.0.0.0:8080`.
    #[host(default = None)]
    pub public: Option<PublicListener>,
    /// `/`, `/ready` and `/worker`; `0.0.0.0:8081`.
    #[host(default = None)]
    pub health: Option<ListenerAddress>,
    /// `/metrics`; off unless given, then `0.0.0.0:9090`.
    #[host(default = None)]
    pub prometheus: Option<ListenerAddress>,
}

/// Where a listener binds; a key left out keeps its default.
#[crate::host(record(
    module = "worker_options",
    path = "livekit.agents",
    typed_dict,
    extract,
    emit
))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ListenerAddress {
    #[host(default = None)]
    pub host: Option<String>,
    #[host(default = None)]
    pub port: Option<u16>,
}

/// The public listener's address and how it reads the client.
#[crate::host(record(
    module = "worker_options",
    path = "livekit.agents",
    typed_dict,
    extract,
    emit
))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PublicListener {
    #[host(default = None)]
    pub host: Option<String>,
    #[host(default = None)]
    pub port: Option<u16>,
    /// Read the client address and scheme from `X-Forwarded-For` and `X-Forwarded-Proto`; only
    /// behind a proxy that always sets them.
    #[host(default = false)]
    pub proxy_headers: bool,
}

/// `AgentServer(...)`'s options: a sparse update to the server configuration, where omission and
/// an explicit null remain distinct. An option left out keeps its default.
#[crate::host(record(module = "worker_options", dataclass, extract, emit, kwargs))]
#[derive(Clone, Default, PartialEq)]
pub struct ServerOptions {
    /// The LiveKit server URL; default `LIVEKIT_URL`.
    pub ws_url: crate::Maybe<String>,
    /// Default `LIVEKIT_API_KEY`.
    pub api_key: crate::Maybe<String>,
    /// Default `LIVEKIT_API_SECRET`.
    pub api_secret: crate::Maybe<String>,
    /// The agent's name, its lane in a thread and the first segment of its endpoints; default
    /// `LIVEKIT_AGENT_NAME`. `LIVEKIT_AGENT_NAME_OVERRIDE` wins over it.
    pub agent_name: crate::Maybe<String>,
    /// The worker's `public`, `health` and `prometheus` listeners (docs/server.md "Listeners").
    pub listeners: crate::Maybe<Listeners>,
    /// Load at which the worker stops taking jobs; 0.7 in production.
    pub load_threshold: crate::Maybe<f64>,
    /// Job processes kept warm ahead of dispatch; `min(ceil(cpus), 4)` in production, none in
    /// development.
    pub num_idle_processes: crate::Maybe<u32>,
    /// Maximum seconds to wait for active work after admission stops; 3600. Zero skips waiting.
    pub drain_timeout: crate::Maybe<f64>,
    /// Total process shutdown budget in seconds, draining and cleanup included; 3610. Must
    /// exceed `drain_timeout`.
    pub shutdown_timeout: crate::Maybe<f64>,
    /// A job process's memory warning threshold in MB; 1000.
    pub job_memory_warn_mb: crate::Maybe<f64>,
    /// A job process's memory kill threshold in MB; 0, no limit.
    pub job_memory_limit_mb: crate::Maybe<f64>,
}

/// The server's process tuning: the launcher sets it (`CliArgs::Start`), never `AgentServer(...)`.
/// An option left out keeps its default.
#[crate::host(record(module = "worker_options", dataclass, extract, emit))]
#[derive(Clone, Default, PartialEq)]
pub struct ProcessOptions {
    /// Connection attempts to LiveKit before the worker fails; 16.
    pub max_retry: crate::Maybe<u32>,
    /// The permissions of the agent's participant in the rooms it joins.
    pub permissions: crate::Maybe<WorkerPermissions>,
    /// The proxy the job processes' HTTP clients use; default `HTTPS_PROXY`, else `HTTP_PROXY`.
    pub http_proxy: crate::Maybe<Option<String>>,
    /// Where jobs run; `"process"`.
    pub job_executor_type: crate::Maybe<crate::worker::job::JobExecutorType>,
    /// How job processes start; `"forkserver"` on Linux, else `"spawn"`.
    pub multiprocessing_context: crate::Maybe<MultiprocessingContext>,
    /// Seconds a job process has to exit once its job ended; 10.
    pub shutdown_process_timeout: crate::Maybe<f64>,
    /// Seconds a session's end callbacks have to finish; 300.
    pub session_end_timeout: crate::Maybe<f64>,
    /// Seconds a job process has to initialize (`prewarm`); 10.
    pub initialize_process_timeout: crate::Maybe<f64>,
}

/// What the agent server reads from where it runs: the CPU count its defaults are sized by.
#[crate::host(functions)]
pub trait ServerEnvironment {
    /// Logical CPUs, fractional under a cgroup quota. Reads, starts nothing.
    #[host(function)]
    fn cpu_count() -> f64;
}
