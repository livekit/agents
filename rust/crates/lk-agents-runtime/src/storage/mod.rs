//! The runtime's storage wiring: the host's `Storage` class over either backend ([`host`]),
//! voice turns on a lane ([`voice`]) and persisted context fields ([`context`]). Every database
//! interaction is `lk_agents_storage`'s.

pub mod context;
pub mod host;
pub mod voice;

pub use lk_agents_storage::{
    CommitOptions, Lane, LaneDb, Lease, LocalStreams, QueuedStreams, ResponseStreams, Run,
    RunOptions, SqliteConnection, Storage, StreamRead, Tasks, TaskQuery, Thread, ThreadDb, agentdb,
    engine, now_ms, now_secs, shortuuid, sqlite,
};
pub use lk_agents_storage::run::{ItemRow, RunOutcome, TaskState};
