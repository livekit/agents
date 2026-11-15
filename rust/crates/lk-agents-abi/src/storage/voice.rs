use crate::StreamInput;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::{ChatItem, storage::StorageError};

/// Live conversation progress before an agent speech necessarily exists. Pending user input
/// is durable recognition evidence, kept separate from accepted model-visible history.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceConversationProgress {
    pub sequence: u64,
    pub agent: String,
    pub history: Vec<ChatItem>,
    pub pending_user_turns: Vec<crate::ChatMessage>,
    pub boundary: VoiceConversationBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceConversationBoundary {
    Progress,
    UserTurnCompleted,
}

pub type StorageWork<'a, T> = Pin<Box<dyn Future<Output = Result<T, StorageError>> + Send + 'a>>;

pub enum VoiceStorageCommand {
    /// `sequence` orders this conversation snapshot with every progress checkpoint in the run.
    Settle {
        speech: u64,
        sequence: u64,
        history: Vec<ChatItem>,
    },
    Abort {
        speech: u64,
    },
    ObserveLoss {
        speech: u64,
    },
}

pub enum VoiceStorageEvent {
    Settled { speech: u64, result: Result<(), StorageError>, lease_lost: bool },
    Abandoned { speech: u64, result: Result<(), StorageError> },
    LeaseLost { speech: u64 },
}

/// Runtime-owned lease and commit resource; host implementations cannot replace its policy.
#[crate::host(native(refused = "expected a native voice storage manager"))]
pub trait VoiceStorage: Send + Sync {
    /// Checkpoints live recognition, or settles an accepted user turn with no reply. Returns
    /// history reconciled under partition ownership, without exposing pending input as final.
    fn conversation(&self, progress: VoiceConversationProgress) -> StorageWork<'_, Vec<ChatItem>>;
    fn dispatch(
        self: Arc<Self>,
        command: VoiceStorageCommand,
        events: Arc<dyn StreamInput<VoiceStorageEvent>>,
    ) -> Result<(), StorageError>;
    fn acquire(&self, speech: u64, local: Vec<ChatItem>) -> StorageWork<'_, Vec<ChatItem>>;
    fn lease_lost(&self) -> bool;
    /// `speech`'s turn saves the conversation so far, numbered `sequence`.
    fn checkpoint(
        &self,
        speech: u64,
        sequence: u64,
        agent: String,
        history: Vec<ChatItem>,
    ) -> StorageWork<'_, ()>;
}

/// The stable context store and initial snapshot belonging to one live call's partition.
#[crate::host(class(name = "_core_VoiceStore", native = VoiceStorage))]
pub trait VoiceStore: Send + Sync + Sized + 'static {
    #[host(class(name = "_core_ContextActivation"))]
    type ContextActivation: Send + Sync + 'static;

    /// `None` preserves the application's explicit constructor history.
    #[host(getter)]
    fn initial_history(&self) -> Option<Vec<ChatItem>>;

    #[host(getter)]
    fn activation(&self) -> Self::ContextActivation;

    #[host(await)]
    async fn aclose(&self) -> Result<(), StorageError>;
}
