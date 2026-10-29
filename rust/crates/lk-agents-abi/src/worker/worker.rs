//! Worker host contracts. Runtime behavior belongs to `lk-agents-worker`.

use lk_agents_host_derive::Host;

/// Current worker state, available independently of lifecycle notification consumption.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    pub worker_id: String,
    pub draining: bool,
    pub active_jobs: usize,
    pub reserved_slots: usize,
    pub worker_load: f32,
    pub target_idle_processes: Option<u32>,
    /// `finished` or `timed_out`, once the drain completes.
    pub drain_outcome: Option<String>,
    /// `run()` has returned. `Some` carries the reason, `None` while still running.
    pub finished: Option<String>,
}

/// `WorkerPermissions`, field-for-field, including the defaults.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(dataclass, module = "worker_options")]
#[binding(extract)]
pub struct WorkerPermissions {
    #[python(default = "True")]
    #[binding(default = true)]
    pub can_publish: bool,
    #[python(default = "True")]
    #[binding(default = true)]
    pub can_subscribe: bool,
    #[python(default = "True")]
    #[binding(default = true)]
    pub can_publish_data: bool,
    #[python(default = "True")]
    #[binding(default = true)]
    pub can_update_metadata: bool,
    /// `models.TrackSource` values, as raw enum numbers.
    #[python(default_factory = "list")]
    pub can_publish_sources: Vec<i32>,
    #[python(default = "False")]
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
#[derive(Clone, Default, Host)]
#[python(dataclass, module = "worker_options")]
#[binding(extract)]
pub struct ServerOptions {
    #[python(
        ty = "NotGivenOr[str]",
        default = "NOT_GIVEN",
        import = "..types:NOT_GIVEN, NotGivenOr"
    )]
    pub ws_url: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[str]", default = "NOT_GIVEN")]
    pub api_key: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[str]", default = "NOT_GIVEN")]
    pub api_secret: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[str]", default = "NOT_GIVEN")]
    pub agent_name: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[str]", default = "NOT_GIVEN")]
    pub host: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[int]", default = "NOT_GIVEN")]
    pub port: crate::Maybe<u16>,
    #[python(ty = "NotGivenOr[float]", default = "NOT_GIVEN")]
    pub load_threshold: crate::Maybe<f32>,
    #[python(ty = "NotGivenOr[int]", default = "NOT_GIVEN")]
    pub num_idle_processes: crate::Maybe<u32>,
    #[python(ty = "NotGivenOr[float | None]", default = "NOT_GIVEN")]
    pub drain_timeout: crate::Maybe<Option<f64>>,
    #[python(ty = "NotGivenOr[int]", default = "NOT_GIVEN")]
    pub max_retry: crate::Maybe<u32>,
    #[python(ty = "NotGivenOr[WorkerPermissions]", default = "NOT_GIVEN")]
    pub permissions: crate::Maybe<WorkerPermissions>,
    #[python(ty = "NotGivenOr[str | None]", default = "NOT_GIVEN")]
    pub http_proxy: crate::Maybe<Option<String>>,
    #[python(ty = "NotGivenOr[int | None]", default = "NOT_GIVEN")]
    pub prometheus_port: crate::Maybe<Option<u16>>,
    #[python(ty = "NotGivenOr[str | None]", default = "NOT_GIVEN")]
    pub prometheus_multiproc_dir: crate::Maybe<Option<String>>,
    #[python(ty = "NotGivenOr[str]", default = "NOT_GIVEN")]
    pub version: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[str]", default = "NOT_GIVEN")]
    pub job_executor_type: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[str]", default = "NOT_GIVEN")]
    pub multiprocessing_context: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[float]", default = "NOT_GIVEN")]
    pub job_memory_warn_mb: crate::Maybe<f64>,
    #[python(ty = "NotGivenOr[float]", default = "NOT_GIVEN")]
    pub job_memory_limit_mb: crate::Maybe<f64>,
    #[python(ty = "NotGivenOr[float]", default = "NOT_GIVEN")]
    pub shutdown_process_timeout: crate::Maybe<f64>,
    #[python(ty = "NotGivenOr[float]", default = "NOT_GIVEN")]
    pub session_end_timeout: crate::Maybe<f64>,
    #[python(ty = "NotGivenOr[float]", default = "NOT_GIVEN")]
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
