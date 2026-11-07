//! The storage port's requests and receipts, independent of the runtime's journal.
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
