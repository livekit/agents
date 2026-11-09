use lk_agents_abi::storage::StorageError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageOperation {
    Acquire,
    Settle,
    Abandon,
    ObserveLoss,
}

#[derive(Clone, Debug)]
pub enum StorageStep {
    Requested {
        speech: u64,
        operation: StorageOperation,
    },
    Answered {
        speech: u64,
        operation: StorageOperation,
        result: Result<(), StorageError>,
        delivered: bool,
    },
    /// A lease take found the lane held by a lease whose owner last sent a request (a renewal,
    /// commit or release) or took it this long ago.
    LaneHeld {
        idle_ms: u64,
    },
    /// What the lane held once the store closed: runs started and never ended, whether a lease
    /// is still held, and whether a fault that can cost ownership (an outage, a write never
    /// applied) was injected.
    LaneSettled {
        unended: u32,
        held: bool,
        faulted: bool,
    },
    /// A storage-layer fault was injected.
    Fault {
        fault: StorageFault,
    },
    /// A write carrying a lease token another worker stole, while that worker held the lane:
    /// whether it changed the lane (its version moved).
    StaleWrite {
        applied: bool,
    },
}

/// A storage-layer fault, as the database or another worker produces it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageFault {
    /// A statement refused as `sqlite_error` maps it: busy, locked, an I/O error or a full disk.
    Sqlite(&'static str),
    /// The wall clock `unixepoch()` reads jumped by this many ms (negative: backward).
    ClockJump(i64),
    /// The database's worker crashed with a write applied and unanswered; the next request
    /// reopens the file (WAL recovery).
    Crash,
    /// Another worker's connection took the free lane and held it, renewing.
    Held,
    /// Another worker's connection overwrote the lane's lease while the call held it.
    Stolen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryOperation {
    Unsubscribe,
    Resubscribe,
    BackgroundStart,
    BackgroundState,
    PreConnect,
    SpeechWaitGeneration,
    SpeechException,
    ToolWaitForPlayout,
    AvatarReceiverStart,
    AvatarRunnerStart,
    AvatarRunnerClose,
    AvatarStreamChunk,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryOutcome {
    pub operation: BoundaryOperation,
    pub error: Option<String>,
}
