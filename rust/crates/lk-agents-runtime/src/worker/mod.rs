//! Worker I/O, process resources, host callbacks and effect execution.

pub use lk_agents_worker::{LoadInputs, WorkerStatus};
/// Wire worker smoothing policy to the shared media filter.
pub type DefaultLoadCalc = lk_agents_worker::DefaultLoadCalc<lk_agents_media::MovingAverage>;

/// Log target for every line: the Python logger name, so `pyo3-log` routes them to it.
pub(crate) const LOG: &str = "livekit.agents";

pub mod auth;
pub mod cli;
#[path = "../../../lk-agents-worker/src/cli.rs"]
mod cli_exports;
pub mod config;
mod diagnostics;
pub mod endpoint;
pub mod hw;
pub mod ipc;

pub use lk_agents_worker::protocol;
#[cfg(feature = "python")]
#[path = "../../../lk-agents-worker/src/python.rs"]
pub mod python;
pub mod server;
#[cfg(any(feature = "python", test))]
mod server_driver;
pub use lk_agents_worker::server_lifecycle;
pub use lk_agents_worker::state;
pub mod transport;
mod worker;
#[cfg(test)]
mod worker_tests;

pub use auth::{TokenError, agent_join_token, extend_job_token, token_identity, worker_token};
pub use config::{
    ASSIGNMENT_TIMEOUT, DRAIN_TIMEOUT, EnvOption, UPDATE_LOAD_INTERVAL, UPDATE_STATUS_INTERVAL,
    WORKER_PROTOCOL_VERSION, WorkerConfig, WorkerOptions,
};
pub use hw::{CpuMonitor, CpuMonitorKind};
pub use lk_agents_abi::worker::worker::{
    AcceptArguments, JobDecision, JobInfo, JobRequest, RawJobDecision, RunningJobInfo, ServerType,
    WorkerDrainFinished, WorkerDrainTimedOut, WorkerPermissions, WorkerRegistered, WorkerStopped,
};
pub use state::{Slot, WorkerState, WorkerStateEffect, WorkerStateInput, WorkerTimer};
pub use transport::{
    AgentEndpointKind, Connected, EndpointReader, EndpointStream, EndpointWriter,
    HttpStreamResetCode, StreamPreamble, TransportConfig, TransportError, TransportEvent,
    WorkerSink, WorkerStream, WorkerTransport, WsTransport, WtTransport, agent_url,
};
pub use worker::{
    EndpointRequest, HEARTBEAT_INTERVAL, HostError, HostResult, JobId, RuntimeConfig, RuntimeError,
    Snapshot, Worker, WorkerCommand, WorkerHandle, WorkerHost, WorkerNotification, retry_delay,
};

#[cfg(test)]
mod test_support;
