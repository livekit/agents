//! The runtime's storage wiring: the host's `Storage` class opens this runtime's A2A door. Every
//! database interaction is `lk_agents_storage`'s.

use std::sync::Arc;

use lk_agents_abi::a2a::door::DoorConfig;
use lk_agents_abi::a2a::turn::TurnHost;
use lk_agents_abi::storage::StorageError;

use crate::a2a::door::Door;
use crate::process::Process;

/// The host's `Storage` class on this process, its A2A doors this runtime's.
pub type HostStorage = lk_agents_storage::host::HostStorage<Process>;

impl lk_agents_storage::host::Env for Process {
    type A2ADoor = Door;

    fn a2a_door(
        storage: Arc<dyn Storage>,
        config: DoorConfig,
        host: Arc<dyn TurnHost>,
        owner: tokio_util::task::TaskTracker,
    ) -> Result<Door, StorageError> {
        Door::new(storage, config, host, owner)
    }
}

pub use lk_agents_storage::context::store::ContextStore;
pub use lk_agents_storage::run::{ItemRow, RunOutcome, TaskState};
pub use lk_agents_storage::voice::VoiceStore;
pub use lk_agents_storage::{
    CommitOptions, Lane, LaneDb, Lease, LocalStreams, QueuedStreams, ResponseStreams, Run,
    RunOptions, SqliteConnection, Storage, StreamRead, TaskQuery, Tasks, Thread, ThreadDb, agentdb,
    engine, now_ms, now_secs, shortuuid, sqlite,
};
