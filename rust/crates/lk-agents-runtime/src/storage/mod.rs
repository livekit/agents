//! The runtime's storage wiring: the host's `Storage` class over either backend ([`host`]). Every
//! database interaction is `lk_agents_storage`'s.

pub mod host;

pub use lk_agents_storage::context::store::ContextStore;
pub use lk_agents_storage::run::{ItemRow, RunOutcome, TaskState};
pub use lk_agents_storage::voice::VoiceStore;
pub use lk_agents_storage::{
    CommitOptions, Lane, LaneDb, Lease, LocalStreams, QueuedStreams, ResponseStreams, Run,
    RunOptions, SqliteConnection, Storage, StreamRead, TaskQuery, Tasks, Thread, ThreadDb, agentdb,
    engine, now_ms, now_secs, shortuuid, sqlite,
};
