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
    /// A lease take found the partition held by a lease whose owner last sent a request (a renewal,
    /// commit or release) or took it this long ago.
    PartitionHeld {
        idle_ms: u64,
    },
    /// What the partition held once the store closed: runs started and never ended, whether a lease
    /// is still held, and whether a fault that can cost ownership (an outage, a write never
    /// applied) was injected.
    PartitionSettled {
        unended: u32,
        held: bool,
        faulted: bool,
    },
    /// What the partition saved once the store closed: its items' ids, in the thread's order.
    Saved {
        items: Vec<String>,
    },
    /// The fake database holds a request's answer this long on purpose (a slow lease take).
    Slowed {
        ms: u64,
    },
    /// A storage-layer fault was injected.
    Fault {
        fault: StorageFault,
    },
    /// A write carrying a lease token another worker stole, while that worker held the partition:
    /// whether it changed the partition (its version moved).
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
    /// Another worker's connection took the free partition and held it, renewing.
    Held,
    /// Another worker's connection overwrote the partition's lease while the call held it.
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
