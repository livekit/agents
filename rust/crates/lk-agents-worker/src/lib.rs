//! Agent worker control plane: registration, process pools, IPC and supervision.

mod load;
#[cfg(test)]
mod load_tests;
pub use load::{DefaultLoadCalc, LoadInputs, WorkerStatus};

/// Logger shared by native worker diagnostics and host adapters.
pub(crate) const LOG: &str = "livekit.agents";

pub mod auth;
pub mod cli;
pub mod cli_signals;
pub mod config;
pub mod diagnostics;
pub mod endpoint;
pub mod hw;
pub mod io;
pub mod ipc;

pub mod endpoint_budget;
pub mod protocol;
pub mod server;
pub mod state;
#[cfg(test)]
mod state_tests;
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
