//! The job a process runs (`job.py`): its start fields, the session's view of it and its trace.

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

/// The job a session runs for (on the job's room, or the console's), as the session's owner
/// sequences its start: the primary claim, the recording, the shutdown, and the room connected
/// beside the opening, before the remote session host starts on it.
#[crate::host(proxy(host = "livekit.agents.job.JobContext"))]
pub trait SessionJob: Send + Sync {
    /// `JobContext.connect()`: the room connected, once however many callers; its `room_connect`
    /// opens under `trace` (the session's `session_start`).
    #[host(await, name = "_core_connect")]
    fn connect(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<(), crate::errors::HostError>>;
    /// Whether the job records its primary session by default.
    #[host(call, getter = "job.enable_recording")]
    fn enable_recording(&self) -> Result<bool, crate::errors::HostError>;
    /// `session` started for the job: whether it is the job's primary session, the first one
    /// claimed, which the job reports and whose close ends the job.
    #[host(call, name = "_core_claim_primary")]
    fn claim_primary(&self, session: crate::HostRef) -> Result<bool, crate::errors::HostError>;
    /// The directory the job's session files are written to.
    #[host(call, name = "_core_recording_dir")]
    fn recording_dir(&self) -> Result<String, crate::errors::HostError>;
    /// `JobContext.init_recording(options)`: the first call sets what the job records.
    #[host(call, name = "init_recording")]
    fn init_recording(
        &self,
        options: std::collections::BTreeMap<String, bool>,
    ) -> Result<(), crate::errors::HostError>;
    /// `session` closes as the job shuts down, in the shutdown's span.
    #[host(call, name = "_core_on_shutdown")]
    fn on_shutdown(&self, session: crate::HostRef) -> Result<(), crate::errors::HostError>;
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
