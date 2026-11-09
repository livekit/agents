pub mod child;
pub mod executor;
pub mod ipc_channel;
pub mod job_run;
pub mod owner;
pub mod pool;
pub mod process;
pub mod process_pool;
pub mod supervisor;
pub(crate) mod telemetry;
#[cfg(test)]
#[path = "ipc_tests.rs"]
mod tests;
pub mod wire;

pub use child::ChildChannel;
pub use executor::{Exit, Runner, RunnerExecutor, ThreadExecutor};
pub use ipc_channel::{Channel, ChannelError, SendError};
pub use job_run::JobRun;
pub use owner::{JobSupervisor, ProcessLaunch, ThreadLaunch};
pub use process::{ProcessExecutor, ProcessReceipt};
pub use process_pool::ProcessPool;
pub use supervisor::secs;
pub use supervisor::{InitError, ProtocolError, StartJobError};
pub use supervisor::{RunnerFields, RunnerSupervisor, SupervisorCallbacks};
pub use wire::{Hello, StartJob, ToChild, ToParent};

// Spawning the OS process stays with the host language; `ipc_channel` is the only code touching
// the channel's bytes and `process` the only code touching the OS process.

/// The child's grace on top of the parent's `ping_timeout` before it calls itself orphaned.
pub const HEALTH_CHECK_GRACE: f64 = 10.0;
