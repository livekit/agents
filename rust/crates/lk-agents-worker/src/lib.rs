//! Synchronous worker registration, process allocation, supervision and lifecycle decisions.
pub mod config;
pub mod endpoint_budget;
pub mod ipc;
mod load;
#[cfg(test)]
mod load_tests;
pub mod protocol;
pub mod server_lifecycle;
pub mod state;
#[cfg(test)]
mod state_tests;
pub use config::*;
pub use lk_agents_abi::errors::{HostError, HostResult};
pub use lk_agents_abi::worker::worker::{
    AcceptArguments, JobDecision, JobInfo, JobRequest, RawJobDecision, RunningJobInfo, ServerType,
    WorkerPermissions,
};
pub use load::{DefaultLoadCalc, LoadInputs, WorkerStatus};
pub use state::{Slot, WorkerState, WorkerStateEffect, WorkerStateInput, WorkerTimer};
pub type JobId = String;
