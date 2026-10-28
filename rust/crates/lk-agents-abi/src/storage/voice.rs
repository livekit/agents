//! Native voice-turn storage capability. The voice executor owns when these effects run.

use crate::StreamSink;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::{ChatItem, storage::StorageError};

pub type StorageWork<'a, T> = Pin<Box<dyn Future<Output = Result<T, StorageError>> + Send + 'a>>;

pub enum VoiceStorageCommand {
    Settle { speech: u64, history: Vec<ChatItem> },
    Abort { speech: u64 },
    ObserveLoss { speech: u64 },
}

pub enum VoiceStorageEvent {
    Settled { speech: u64, result: Result<(), StorageError>, lease_lost: bool },
    Abandoned { speech: u64, result: Result<(), StorageError> },
    LeaseLost { speech: u64 },
}

/// Runtime-owned lease and commit resource; host implementations cannot replace its policy.
#[crate::host(native(refused = "expected a native voice storage manager"))]
pub trait VoiceStorage: Send + Sync {
    fn dispatch(
        self: Arc<Self>,
        command: VoiceStorageCommand,
        events: Arc<dyn StreamSink<VoiceStorageEvent>>,
    ) -> Result<(), StorageError>;
    fn acquire(&self, speech: u64, local: Vec<ChatItem>) -> StorageWork<'_, Vec<ChatItem>>;
    fn settle(&self, speech: u64, history: Vec<ChatItem>) -> StorageWork<'_, ()>;
    fn abort(&self, speech: u64) -> StorageWork<'_, ()>;
    fn wait_lost(&self) -> StorageWork<'_, bool>;
    fn lease_lost(&self) -> bool;
}

/// The stable context store and initial snapshot belonging to one live call's lane.
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
