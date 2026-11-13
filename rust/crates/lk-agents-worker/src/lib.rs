//! Agent worker control plane: registration, process pools, IPC and supervision.
#![cfg_attr(
    test,
    allow(
        clippy::disallowed_methods,
        reason = "tests drive Tokio's paused clock and spawn directly; the library is linted without"
    )
)]

mod load;
pub use load::{DefaultLoadCalc, LoadInputs, WorkerStatus};

pub mod config;
pub mod http;
pub mod hw;
pub mod ipc;
pub mod server;
pub mod transport;
mod worker;

pub use config::{
    ASSIGNMENT_TIMEOUT, DRAIN_TIMEOUT, EnvOption, UPDATE_LOAD_INTERVAL, UPDATE_STATUS_INTERVAL,
    WORKER_PROTOCOL_VERSION, WorkerConfig,
};
pub use hw::{CpuMonitor, CpuMonitorKind};
pub use lk_agents_abi::worker::job::{AcceptArguments, RunningJobInfo};
pub use lk_agents_abi::worker::types::{JobDecision, JobRequest, RegisteredInfo, WorkerStopped};
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

/// The test's executor and Tokio's clock (paused in a paused test), with a task set of its own.
#[cfg(test)]
pub(crate) fn test_rt() -> lk_agents_net::owned::Background {
    let spawner = std::sync::Arc::new(lk_agents_net::task::TokioSpawner::current());
    let clock = std::sync::Arc::new(lk_agents_net::clock::RealClock::new());
    lk_agents_net::owned::Background::new(Default::default(), spawner, clock)
}
