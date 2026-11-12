use crate::telemetry::decl::SpanValue;

/// A `livekit.agent.Job`, crossing as the SDK's message.
#[crate::host(codec(
    extract,
    emit,
    host = "livekit.protocol.agent.Job",
    py = "py_values::proto_job"
))]
pub type Job = livekit_protocol::Job;

/// The process's one job as the worker hands it over: what it runs, and the span it continues.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(module = "worker_ipc", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StartJob {
    pub info: RunningJobInfo,
    #[host(default = None)]
    pub traceparent: Option<String>,
}

/// The job a session runs for (on the job's room, or the console's), as the session's owner
/// sequences its start: the primary claim, the recording, the shutdown (on the job's run), and
/// the room connected beside the opening, before the remote session host starts on it. An avatar
/// started for the job closes on its shutdown and removes its participant with the job's API
/// credentials.
#[crate::host(proxy(host = "livekit.agents.job.JobContext"))]
pub trait SessionJob: Send + Sync {
    /// `JobContext.connect()`: the room connected, once however many callers; its `room_connect`
    /// opens under `trace` (the session's `session_start`).
    #[host(await, name = "connect")]
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
    #[host(call, getter = "_tempdir.name")]
    fn recording_dir(&self) -> Result<String, crate::errors::HostError>;
    /// `JobContext.init_recording(options)`: the first call sets what the job records.
    #[host(call, name = "init_recording")]
    fn init_recording(
        &self,
        options: std::collections::BTreeMap<String, bool>,
    ) -> Result<(), crate::errors::HostError>;
    /// The job's native run, where a session's and an avatar's closes register on its shutdown.
    #[host(read_once, getter = "_run")]
    fn run(&self) -> std::sync::Arc<dyn crate::worker::ipc_host::JobExecution>;
    /// The server's settings by variable name: `LIVEKIT_URL`, `LIVEKIT_API_KEY` and
    /// `LIVEKIT_API_SECRET` are what the job's API calls authenticate with.
    #[host(call, getter = "_server_env")]
    fn server_env(
        &self,
    ) -> Result<std::collections::BTreeMap<String, String>, crate::errors::HostError>;
}

// Public data contracts. Python behavior remains in its adapters.

/// The participant an accepted job joins as. Application attributes keep their own keys, merged
/// over `lk.agent.name`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[crate::host(record(
    module = "job_records",
    path = "livekit.agents.job",
    name = "JobAcceptArguments",
    dataclass,
    extract,
    emit
))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AcceptArguments {
    pub name: String,
    pub identity: String,
    pub metadata: String,
    #[host(default_factory = "dict")]
    pub attributes: std::collections::BTreeMap<String, String>,
}

/// What a launched job is handed.
#[derive(Debug, Clone, PartialEq)]
#[crate::host(record(
    module = "job_records",
    path = "livekit.agents.job",
    dataclass,
    extract,
    emit
))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RunningJobInfo {
    pub accept_arguments: AcceptArguments,
    #[cfg_attr(feature = "serde", serde(with = "proto_bytes"))]
    pub job: Job,
    pub url: String,
    pub token: String,
    pub worker_id: String,
    pub fake_job: bool,
    /// The worker received the availability request; absent for a job no dispatch offered.
    #[host(default = None)]
    pub received_at: Option<f64>,
    /// The request handler accepted the job.
    #[host(default = None)]
    pub accepted_at: Option<f64>,
    /// The server's assignment (room token) arrived.
    #[host(default = None)]
    pub assigned_at: Option<f64>,
    /// A process was acquired from the pool and handed the job.
    #[host(default = 0.0)]
    pub launched_at: f64,
}

/// The job's protobuf bytes, as the SDK serializes the message.
pub fn encode_job(job: &Job) -> Vec<u8> {
    prost::Message::encode_to_vec(job)
}

pub fn decode_job(bytes: &[u8]) -> Result<Job, prost::DecodeError> {
    <Job as prost::Message>::decode(bytes)
}

/// The thread a call's dispatch names: the job's `thread_id`; empty names none.
pub fn job_thread(job: &Job) -> Option<&str> {
    Some(job.thread_id.as_str()).filter(|id| !id.is_empty())
}

/// The job as its protobuf bytes on the job process channel.
#[cfg(feature = "serde")]
mod proto_bytes {
    use serde::{Deserialize as _, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(job: &super::Job, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(&super::encode_job(job))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<super::Job, D::Error> {
        let bytes = <Vec<u8>>::deserialize(d)?;
        super::decode_job(&bytes).map_err(serde::de::Error::custom)
    }
}

#[crate::host(record(
    module = "job_records",
    path = "livekit.agents",
    enum,
    enum_base = "StrEnum",
    extract,
    emit
))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// What ended a job. The application's own end wins over the session close it causes.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownCause {
    /// `JobContext.shutdown()`.
    Application,
    /// The simulator disconnected.
    SimulationCompleted,
    /// The job's primary session closed.
    SessionClosed,
    /// The room disconnected.
    RoomDisconnected,
    /// The entrypoint raised.
    EntrypointCrashed,
    /// The worker shut the job's process down.
    ProcessShutdown,
}

/// The simulation run a simulated job belongs to, from its `lk.simulator.dispatch` attribute.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Simulation {
    pub run_id: Option<String>,
    pub job_id: Option<String>,
}
