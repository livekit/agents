//! Async thread storage: shared execution over Cloud AgentDB or local SQLite.

pub mod backends;
pub mod context;
mod cursor;
pub mod http;
pub mod sql;
pub mod storage;
#[cfg(feature = "testing")]
#[allow(clippy::unwrap_used, reason = "test fixtures: a failure fails the test")]
pub mod testing;
pub mod thread;
pub mod voice;

pub use backends::{Storage, ThreadDbConnection};
pub use lk_agents_abi::storage::StorageError;
pub use lk_agents_abi::storage::thread::{
    LaneAccess, LeaseAfterCommit, TaskPickup, TaskWrite, WhenHeld, timestamp,
};
pub use lk_agents_net::strings::shortuuid;
pub use thread::db::{Commit, Committed, EventWrite, Fence, TaskQuery};
pub use thread::lease::Lease;
pub use thread::run::{ItemRow, RunOutcome, TaskState};
pub use thread::{CommitOptions, Lane, Run, RunOptions, Tasks, Thread};
