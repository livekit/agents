//! Startup storage owned for the lifetime of one voice session.

use crate::{ChatItem, storage::StorageError};
use std::sync::Arc;

#[crate::host(class(name = "_core_SessionStorage"))]
pub trait SessionStorage: Send + Sync + Sized + 'static {
    #[host(env)]
    type Env;
    #[host(class(name = "Storage"))]
    type Namespace: Send + Sync + 'static;
    #[host(class(name = "_core_ContextActivation"))]
    type Activation: Send + Sync + 'static;
    #[host(class(name = "_core_VoiceStore"))]
    type VoiceStore: Send + Sync + 'static;

    #[host(new)]
    fn new(
        env: &Self::Env,
        namespace: Option<Arc<Self::Namespace>>,
        thread_id: Option<String>,
        partition_key: String,
        agent: String,
        initial_history: Option<Vec<ChatItem>>,
    ) -> Self;

    /// Reuse the context activation the application hydrated before voice startup.
    #[host(call)]
    fn use_activation(
        &self,
        activation: std::sync::Arc<Self::Activation>,
    ) -> Result<(), StorageError>;

    /// Load the supplied thread's partition; its failure is the call's (no other thread replaces it).
    /// Without a thread, the call runs in memory: nothing is written anywhere.
    #[host(await)]
    async fn open(&self) -> Result<(), StorageError>;
    /// The storage the call's thread lives in; `None` in memory.
    #[host(getter)]
    fn namespace(&self) -> Option<Arc<Self::Namespace>>;
    #[host(getter)]
    fn thread_id(&self) -> Option<String>;
    #[host(getter)]
    fn scope_id(&self) -> String;
    #[host(getter)]
    fn activation(&self) -> Self::Activation;
    #[host(getter)]
    fn voice_store(&self) -> Option<Arc<Self::VoiceStore>>;
    #[host(getter)]
    fn history(&self) -> Option<Vec<ChatItem>>;
    #[host(await)]
    async fn aclose(&self);
}
