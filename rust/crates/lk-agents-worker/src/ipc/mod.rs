//! The job processes: the worker and each job process over one channel.
//!
//! - [`channel`]: the connection, the only code touching its bytes: transport, framing, limits
//!   and the version check, with an in-memory variant for tests.
//! - [`wire`]: the protocol's messages, a typed one per direction.
//! - [`JobExecutor`] ([`executor`]): what a job runs in, a [`Runner`] launched with its channel;
//!   a thread of the worker ([`ThreadExecutor`]) or a process ([`ProcessExecutor`]).
//! - [`process`]: the process executor, the only code touching the OS process: spawn receipts,
//!   exit, kill, reaping receipts, memory.
//! - [`RunnerSupervisor`] ([`supervisor`]): the worker's policy for one job runner over its
//!   channel, its initialization, pings, memory sampling, shutdown and cleanup, whatever runs the
//!   job.
//! - [`ChildChannel`] ([`child`]): the job process's side, with the self health check. The child
//!   answers pings from its own event loop, so a pong still measures that loop's responsiveness.
//! - [`pool`]: the warm processes. Spawning them stays with the host language.
//! - [`JobSupervisor`], [`ProcessPool`], [`ChildChannel`] and [`JobRun`]: the classes the host
//!   drives them through, calling back into it only through declared proxies.

pub mod channel;
pub mod child;
pub mod executor;
pub mod job_run;
pub mod owner;
pub mod pool;
pub mod process;
pub mod process_pool;
pub mod supervisor;
#[cfg(test)]
mod tests;
pub mod wire;

pub use channel::{Channel, ChannelError, SendError};
pub use child::ChildChannel;
pub use executor::{Exit, JobExecutor, Runner, ThreadExecutor};
pub use job_run::JobRun;
pub use owner::JobSupervisor;
pub use process::ProcessExecutor;
pub use process_pool::ProcessPool;
pub use supervisor::secs;
pub use supervisor::{
    Extra, ExtraValue, InitError, Level, ProtocolError, StartJobError, SupervisorOptions,
    extra_fields,
};
pub use supervisor::{RunnerSupervisor, SupervisorCallbacks};
pub use wire::{Hello, StartJob, ToChild, ToParent};

/// The child's grace on top of the parent's `ping_timeout` before it calls itself orphaned.
pub const HEALTH_CHECK_GRACE: f64 = 10.0;
