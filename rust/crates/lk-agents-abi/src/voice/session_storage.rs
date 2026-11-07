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
        lane: String,
        fallback_lane: String,
        initial_history: Option<Vec<ChatItem>>,
    ) -> Self;

    /// Load the supplied thread's lane. Without one, or when it does not load, the call runs on
    /// a new thread of the local SQLite storage (`SQLiteStorage()`'s default directory).
    #[host(await)]
    async fn open(&self) -> Result<(), StorageError>;
    /// The storage the call's thread lives in.
    #[host(getter)]
    fn namespace(&self) -> Arc<Self::Namespace>;
    #[host(getter)]
    fn thread_id(&self) -> String;
    #[host(getter)]
    fn scope_id(&self) -> String;
    #[host(getter)]
    fn activation(&self) -> Self::Activation;
    #[host(getter)]
    fn voice_store(&self) -> Arc<Self::VoiceStore>;
    #[host(getter)]
    fn history(&self) -> Option<Vec<ChatItem>>;
    #[host(await)]
    async fn aclose(&self);
}
