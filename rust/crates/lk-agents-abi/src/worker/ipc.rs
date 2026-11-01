//! A job runner's completion keeps its OS exit status separate from its control protocol's failure.

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
    enum_base = "Enum"
))]
pub enum JobStatus {
    #[host(rename = "running")]
    Running,
    #[host(rename = "failed")]
    Failed,
    #[host(rename = "success")]
    Success,
}
