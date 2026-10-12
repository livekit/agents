//! Agent worker control plane. [`Worker::new`] takes [`WorkerOptions`] and a [`WorkerHost`] and
//! returns the worker and a [`WorkerHandle`] without starting anything; omitted options fall back
//! to the environment (an explicit empty string disables that).
//!
//! ```no_run
//! # use lk_agents_worker::{Worker, WorkerOptions, WorkerHost, RuntimeError};
//! # async fn start(host: impl WorkerHost + 'static) -> Result<(), RuntimeError> {
//! let options = WorkerOptions {
//!     agent_name: Some("support-agent".into()),
//!     ..Default::default() // URL, credentials, deployment, and token from the environment
//! };
//! let (mut worker, handle) = Worker::new(options, host);
//! let mut notifications = handle.subscribe();
//! worker.run().await?;
//! # Ok(())
//! # }
//! ```
//!
//! [`WorkerState`] is the sans-IO state machine. Pending status samples coalesce; availability
//! and job messages keep order; a write failure invalidates the connection, and a reconnect
//! re-registers without replaying. A successful write is transport completion, not a server ack.

/// Log target for every line: the Python logger name, so `pyo3-log` routes them to it.
pub(crate) const LOG: &str = "livekit.agents";

pub mod auth;
mod config;
pub mod hw;
mod load;
#[cfg(test)]
mod load_tests;
pub mod protocol;
mod state;
pub mod transport;
mod worker;
#[cfg(test)]
mod worker_tests;
#[cfg(feature = "python")]
pub mod python;

#[cfg(test)]
mod state_tests;

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
pub use load::{DefaultLoadCalc, LoadInputs, WorkerStatus};
pub use state::{Slot, WorkerState, WorkerStateEffect, WorkerStateInput, WorkerTimer};
pub use transport::{
    AgentEndpointKind, Connected, EndpointReader, EndpointStream, EndpointWriter,
    HttpStreamResetCode, StreamPreamble, TransportConfig, TransportError, TransportEvent,
    WorkerSink, WorkerStream, WorkerTransport, WsTransport, WtTransport, agent_url,
};
pub use worker::{
    EndpointRequest, HEARTBEAT_INTERVAL, HostError, HostResult, JobId, RuntimeConfig,
    RuntimeError, Snapshot, Worker, WorkerCommand, WorkerHandle, WorkerHost, WorkerNotification,
    retry_delay,
};
