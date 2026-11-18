use crate::StreamInput;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::{ChatItem, storage::StorageError};

/// One immutable conversation snapshot and the database values it was read against.
/// A later snapshot's observations cannot turn this snapshot's old values into local edits.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VoiceHistory {
    /// Core-selected conversation items, never a rendered model prompt.
    pub items: Vec<ChatItem>,
    /// Ordered database values observed by this snapshot, before any local edits.
    pub stored: Vec<ChatItem>,
    /// Partition fence and local receipt order; older observations cannot rebase newer ones.
    pub revision: (i64, u64),
    /// Native storage identity for a late local conflict; never restored as execution authority.
    pub partition_key: Option<String>,
    /// Exact observed rows last changed by this live store, retained only while still unchanged.
    pub own_writes: std::collections::BTreeSet<String>,
}

impl VoiceHistory {
    /// New local content with no claim about previously stored values.
    pub fn new(items: Vec<ChatItem>) -> Self {
        Self { items, ..Self::default() }
    }
}

/// Live conversation progress before an agent speech necessarily exists. Pending user input
/// is durable recognition evidence, kept separate from accepted model-visible history.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceConversationProgress {
    pub sequence: u64,
    pub agent: String,
    pub history: VoiceHistory,
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
        history: VoiceHistory,
    },
    Abort {
        speech: u64,
    },
    ObserveLoss {
        speech: u64,
    },
}

pub enum VoiceStorageEvent {
    Settled { speech: u64, result: Result<VoiceHistory, StorageError>, lease_lost: bool },
    Abandoned { speech: u64, result: Result<(), StorageError> },
    LeaseLost { speech: u64 },
}

/// Runtime-owned lease and commit resource; host implementations cannot replace its policy.
#[crate::host(native(refused = "expected a native voice storage manager"))]
pub trait VoiceStorage: Send + Sync {
    /// Installed by this call's native executor before it accepts host work.
    fn bind_session_commands(&self, commands: std::sync::Arc<dyn crate::session::SessionCommands>);
    /// Checkpoints live recognition, or settles an accepted user turn with no reply. Returns
    /// history reconciled under partition ownership, without exposing pending input as final.
    fn conversation(&self, progress: VoiceConversationProgress) -> StorageWork<'_, VoiceHistory>;
    fn dispatch(
        self: Arc<Self>,
        command: VoiceStorageCommand,
        events: Arc<dyn StreamInput<VoiceStorageEvent>>,
    ) -> Result<(), StorageError>;
    fn acquire(&self, speech: u64, local: VoiceHistory) -> StorageWork<'_, VoiceHistory>;
    fn lease_lost(&self) -> bool;
    /// `speech`'s turn saves the conversation so far, numbered `sequence`.
    fn checkpoint(
        &self,
        speech: u64,
        sequence: u64,
        agent: String,
        history: VoiceHistory,
    ) -> StorageWork<'_, VoiceHistory>;
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
