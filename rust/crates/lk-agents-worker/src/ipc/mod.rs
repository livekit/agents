//! The job processes: the worker and each job process over one channel.
//!
//! - [`channel`]: the connection, the only code touching its bytes: transport, framing, limits
//!   and the version check, with an in-memory variant for tests.
//! - [`wire`]: the protocol's messages, a typed one per direction.
//! - [`process`]: the only code touching the OS process: exit, kill, reaping receipts, memory.
//! - [`ProcSupervisor`] ([`supervisor`]): the worker's policy for one job process over a channel,
//!   its initialization, pings, memory sampling, shutdown and cleanup.
//! - [`ChildChannel`] ([`child`]): the job process's side, with the self health check. The child
//!   answers pings from its own event loop, so a pong still measures that loop's responsiveness.
//! - [`pool`]: the warm processes. Spawning them stays with the host language.
//! - [`ProcessOwner`], [`ProcessPool`], [`ChildChannel`] and [`JobRun`]: the classes the host
//!   drives them through, calling back into it only through declared proxies.

pub mod channel;
pub mod child;
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
pub use job_run::JobRun;
pub use owner::ProcessOwner;
pub use process_pool::ProcessPool;
pub use supervisor::secs;
pub use supervisor::{
    Extra, ExtraValue, InitError, Level, ProtocolError, StartJobError, SupervisorOptions,
    extra_fields,
};
pub use supervisor::{MemoryError, ProcSupervisor, ProcessCallbacks};
pub use wire::{Hello, StartJob, ToChild, ToParent};

/// The child's grace on top of the parent's `ping_timeout` before it calls itself orphaned.
pub const HEALTH_CHECK_GRACE: f64 = 10.0;
