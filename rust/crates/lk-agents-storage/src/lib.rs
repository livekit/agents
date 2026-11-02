//! Async thread storage: shared execution over Cloud AgentDB or local SQLite.
//!
//! - [`backends`]: lifecycle and database connections for both backends.
//! - [`thread`]: thread/lane/run execution, leases, commits and typed database operations.
//! - [`context`], [`voice`]: persisted application context and voice-session storage.
//! - [`namespace`]: native host adapters for namespaces, threads and runs.
//! - [`http`]: the generated ThreadService contract and its local listener.
//! - [`sql`]: bound statements, results, row decoding and schema validation.

pub mod backends;
pub mod context;
pub mod http;
pub mod namespace;
pub mod sql;
#[cfg(feature = "testing")]
pub mod testing;
pub mod thread;
pub mod voice;

pub use backends::{Storage, ThreadDbConnection};
pub use lk_agents_abi::storage::StorageError;
pub use lk_agents_abi::storage::thread::{TaskWrite, timestamp};
pub use lk_agents_net::strings::shortuuid;
pub use thread::db::{Commit, Committed, EventWrite, Fence, TaskQuery};
pub use thread::lease::Lease;
pub use thread::run::{ItemRow, RunOutcome, TaskState};
pub use thread::{CommitOptions, Lane, Run, RunOptions, Tasks, Thread};
