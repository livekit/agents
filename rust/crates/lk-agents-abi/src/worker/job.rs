//! The job a process runs (`job.py`): its start fields, the session's view of it and its trace.

use lk_agents_host_derive::Host;

/// The process's one job, shared by the generated host boundary and the IPC wire protocol.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[crate::class(from_py_object)]
#[binding(class, new)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StartJobFields {
    #[python(ty = "bytes")]
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
    #[binding(default)]
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
    ) -> (Vec<(String, i64)>, Vec<(String, f64)>);
}
