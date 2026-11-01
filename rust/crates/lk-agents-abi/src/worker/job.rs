//! The job a process runs (`job.py`): its start fields, the session's view of it and its trace.

/// Dispatch stage timestamps and the durations between them.
pub type JobTimeline = (Vec<(String, i64)>, Vec<(String, f64)>);

/// The process's one job, shared by the generated host boundary and the IPC wire protocol.
#[derive(Debug, Clone, Default, PartialEq)]
#[crate::host(record(module = "worker_ipc", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StartJobFields {
    pub job: Vec<u8>,
    pub name: String,
    pub identity: String,
    pub metadata: String,
    pub url: String,
    pub token: String,
    pub worker_id: String,
    pub fake_job: bool,
    pub received_at: f64,
    pub accepted_at: f64,
    pub assigned_at: f64,
    pub launched_at: f64,
    #[host(default = None)]
    pub traceparent: Option<String>,
}

/// The job a session runs in, as `start()` sequences it: the room connected beside the opening,
/// before the remote session host starts on it.
#[crate::host(proxy(host = "livekit.agents.job.JobContext"))]
pub trait SessionJob: Send + Sync {
    /// `JobContext.connect()`: the room connected, once however many callers.
    #[host(await, name = "_core_connect")]
    fn connect(&self) -> crate::BoxFuture<'_, Result<(), crate::errors::HostError>>;
}

/// The trace of a job's dispatch.
#[crate::host(functions)]
pub trait JobTrace {
    /// A job's dispatch timeline, for its root span: one event per known stage instant (unix
    /// nanoseconds) and the seconds between adjacent stages as attributes (they sum to the dispatch
    /// latency). A zero instant is unknown (a simulation, the console, a resumed job) and skipped.
    /// `server_started_at` is the server's own start (`JobState.started_at`, unix nanoseconds, or
    /// milliseconds or seconds should that change).
    #[host(function)]
    fn job_timeline(
        received_at: f64,
        accepted_at: f64,
        assigned_at: f64,
        launched_at: f64,
        entrypoint_started_at: f64,
        server_started_at: i64,
    ) -> JobTimeline;
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(module = "job_records", path = "livekit.agents.job", dataclass))]
pub struct JobAcceptArguments {
    pub name: String,
    pub identity: String,
    pub metadata: String,
    #[host(default = None)]
    pub attributes: Option<std::collections::BTreeMap<String, String>>,
}

#[crate::host(record(module = "job_records", path = "livekit.agents.job", dataclass))]
pub struct RunningJobInfo {
    pub accept_arguments: JobAcceptArguments,
    #[host(annotation = "livekit.protocol.agent.Job")]
    pub job: crate::HostRef,
    pub url: String,
    pub token: String,
    pub worker_id: String,
    pub fake_job: bool,
    /// The worker received the availability request.
    #[host(default = 0.0)]
    pub received_at: f64,
    /// The request handler accepted the job.
    #[host(default = 0.0)]
    pub accepted_at: f64,
    /// The server's assignment (room token) arrived.
    #[host(default = 0.0)]
    pub assigned_at: f64,
    /// A process was acquired from the pool and handed the job.
    #[host(default = 0.0)]
    pub launched_at: f64,
}

#[crate::host(record(
    module = "job_records",
    path = "livekit.agents.job",
    enum,
    enum_base = "Enum"
))]
pub enum JobExecutorType {
    #[host(rename = "process")]
    Process,
    #[host(rename = "thread")]
    Thread,
}

#[crate::host(record(
    module = "job_records",
    path = "livekit.agents.job",
    enum,
    enum_base = "StrEnum"
))]
pub enum AutoSubscribe {
    #[host(rename = "subscribe_all")]
    SubscribeAll,
    #[host(rename = "subscribe_none")]
    SubscribeNone,
    #[host(rename = "audio_only")]
    AudioOnly,
    #[host(rename = "video_only")]
    VideoOnly,
}
