//! Async thread storage: shared execution over Cloud AgentDB or local SQLite.
#![cfg_attr(
    test,
    allow(
        clippy::disallowed_methods,
        reason = "tests drive Tokio's paused clock and spawn directly; the library is linted without"
    )
)]

pub mod backends;
pub mod context;
mod cursor;
pub mod http;
pub mod session_binding;
mod session_snapshot;
pub mod session_state;
pub mod sql;
pub mod storage;
mod telemetry;
#[cfg(feature = "testing")]
#[allow(clippy::unwrap_used, reason = "test fixtures: a failure fails the test")]
pub mod testing;
pub mod thread;
pub mod voice;

pub use backends::{Storage, ThreadDbConnection};
pub use lk_agents_abi::storage::StorageError;
pub use lk_agents_abi::storage::thread::{
    LeaseAfterCommit, PartitionAccess, TaskPickup, TaskWrite, WhenHeld, timestamp,
};
pub use lk_agents_net::strings::shortuuid;
pub use thread::db::{Commit, Committed, EventWrite, Fence, TaskQuery};
pub use thread::lease::{Holder, Lease};
pub use thread::run::{ItemRow, PartitionKind, RunOutcome, TaskState};
pub use thread::{CommitOptions, Partition, Run, RunOptions, Tasks, Thread};
