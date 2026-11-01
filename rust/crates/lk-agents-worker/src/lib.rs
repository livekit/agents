//! Agent worker control plane: registration, process pools, IPC and supervision.

mod load;
pub use load::{DefaultLoadCalc, LoadInputs, WorkerStatus};

/// Logger shared by native worker diagnostics and host adapters.
pub(crate) const LOG: &str = "livekit.agents";

pub mod config;
pub mod http;
pub mod hw;
pub mod ipc;
pub mod server;
pub mod transport;
mod worker;
#[cfg(test)]
mod worker_tests;

pub use config::{
    ASSIGNMENT_TIMEOUT, DRAIN_TIMEOUT, EnvOption, UPDATE_LOAD_INTERVAL, UPDATE_STATUS_INTERVAL,
    WORKER_PROTOCOL_VERSION, WorkerConfig,
};
pub use hw::{CpuMonitor, CpuMonitorKind};
pub use lk_agents_abi::worker::types::{
    AcceptArguments, JobDecision, JobInfo, JobRequest, RunningJobInfo, WorkerDrainFinished,
    WorkerDrainTimedOut, WorkerPermissions, WorkerRegistered, WorkerStopped,
};
pub use transport::{
    AgentEndpointKind, Connected, EndpointReader, EndpointStream, EndpointWriter,
    HttpStreamResetCode, StreamPreamble, TransportConfig, TransportError, TransportEvent,
    WebSocketTransport, WorkerSink, WorkerStream, WorkerTransport, WtTransport, agent_url,
};
pub use worker::{
    DrainError, DrainOutcome, HEARTBEAT_INTERVAL, HostError, HostResult, JobId, RuntimeConfig,
    RuntimeError, Slot, Snapshot, Worker, WorkerClient, WorkerCommand, WorkerHost,
    WorkerNotification, retry_delay,
};

pub use http::IncomingRequest;
