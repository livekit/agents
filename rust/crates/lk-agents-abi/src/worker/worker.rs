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
    pub target_idle_processes: Option<u32>,
    /// Set once the drain completes.
    pub drain_outcome: Option<DrainOutcome>,
    /// `run()` has returned. `Some` carries the reason, `None` while still running.
    pub finished: Option<String>,
}

/// `WorkerPermissions`, field-for-field, including the defaults.
#[crate::host(record(module = "worker_options", dataclass, extract))]
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

/// The scalar half of a `livekit.agent.Job`.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "worker_server", emit))]
pub struct JobInfo {
    pub id: String,
    pub dispatch_id: String,
    pub r#type: i32,
    /// The room's name, flattened out of `Job.room`.
    pub room: Option<String>,
    pub room_sid: Option<String>,
    pub agent_name: String,
    pub metadata: String,
    pub deployment: String,
    pub enable_recording: bool,
    pub enable_redaction: bool,
    /// The exact `agent.Job` bytes: the fields above are lossy, and this crosses IPC.
    pub proto: Vec<u8>,
}

/// A job offered to the worker, which answers it with a [`JobDecision`].
#[derive(Debug, Clone, PartialEq)]
pub struct JobRequest {
    pub job: JobInfo,
    /// `AvailabilityRequest.resuming` -- the job is migrating from another worker.
    pub resuming: bool,
}

impl JobRequest {
    pub fn room_name(&self) -> &str {
        self.job.room.as_deref().unwrap_or("")
    }
}

/// `RunningJobInfo`: what a launched job is handed.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "worker_server", emit))]
pub struct RunningJobInfo {
    pub accept_arguments: AcceptArguments,
    pub job: JobInfo,
    pub url: String,
    pub token: String,
    pub worker_id: String,
    pub fake_job: bool,
    /// The dispatch timeline in unix seconds, 0.0 for a job no dispatch offered (simulated):
    /// the `AvailabilityRequest` arrived,
    pub received_at: f64,
    /// the request was accepted,
    pub accepted_at: f64,
    /// and the `JobAssignment` arrived.
    pub assigned_at: f64,
}

/// Acceptance returns before assignment and launch; the worker owns the assignment deadline.
/// A rejection leaves the job to another worker.
#[derive(Debug, Clone, PartialEq)]
pub enum JobDecision {
    Accept(AcceptArguments),
    Reject,
}

/// Accepted participant fields; application attributes retain their own string keys.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(module = "worker_server", emit))]
pub struct AcceptArguments {
    pub name: String,
    pub identity: String,
    pub metadata: String,
    /// Merged after `lk.agent.name`, so a user attribute of that name wins.
    pub attributes: Vec<(String, String)>,
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
}

/// The worker notification and server callback share one registration schema.
pub type WorkerRegistered = RegisteredInfo;

/// `drain()` finished within its deadline. `kind: "drain_finished"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerDrainFinished {}

/// `drain()` hit its deadline; the caller is still expected to close the worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerDrainTimedOut {}

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
    pub job_executor_type: String,
    pub multiprocessing_context: String,
    pub pool: ProcessPoolOptions,
}

/// A sparse update to the server configuration; omission and an explicit null remain distinct.
#[crate::host(record(module = "worker_options", dataclass, extract))]
#[derive(Clone, Default)]
pub struct ServerOptions {
    pub ws_url: crate::Maybe<String>,
    pub api_key: crate::Maybe<String>,
    pub api_secret: crate::Maybe<String>,
    pub agent_name: crate::Maybe<String>,
    pub host: crate::Maybe<String>,
    pub port: crate::Maybe<u16>,
    pub load_threshold: crate::Maybe<f64>,
    pub num_idle_processes: crate::Maybe<u32>,
    pub drain_timeout: crate::Maybe<f64>,
    pub shutdown_timeout: crate::Maybe<f64>,
    pub max_retry: crate::Maybe<u32>,
    pub permissions: crate::Maybe<WorkerPermissions>,
    pub http_proxy: crate::Maybe<Option<String>>,
    pub prometheus_port: crate::Maybe<Option<u16>>,
    pub prometheus_multiproc_dir: crate::Maybe<Option<String>>,
    pub version: crate::Maybe<String>,
    pub job_executor_type: crate::Maybe<String>,
    pub multiprocessing_context: crate::Maybe<String>,
    pub job_memory_warn_mb: crate::Maybe<f64>,
    pub job_memory_limit_mb: crate::Maybe<f64>,
    pub shutdown_process_timeout: crate::Maybe<f64>,
    pub session_end_timeout: crate::Maybe<f64>,
    pub initialize_process_timeout: crate::Maybe<f64>,
}

/// What the agent server reads from where it runs: the CPU count its defaults are sized by, and
/// the base URL its text endpoints are reached at.
#[crate::host(functions)]
pub trait ServerEnvironment {
    /// Logical CPUs, fractional under a cgroup quota. Reads, starts nothing.
    #[host(function)]
    fn cpu_count() -> f64;

    /// `https://<project>/agents/<agent>/<deployment>` (`AgentServer._endpoint_url_base`), when
    /// the URL and the agent are known; a worker registered without a deployment is `default`.
    #[host(function)]
    fn endpoint_url_base(ws_url: String, agent_name: String, deployment: String) -> Option<String>;
}
