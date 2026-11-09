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
