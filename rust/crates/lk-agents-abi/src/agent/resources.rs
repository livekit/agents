//! Request-scoped resources shared by a direct invocation or all frames of a text session.

use std::sync::Arc;

use crate::{BoxFuture, errors::HostError, llm::toolsets::ToolsetHooks, storage::StorageError};

#[derive(Debug, Clone, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
pub enum ResourceError {
    #[error("run resources are closed")]
    Closed,
    #[error("run initialization was cancelled")]
    Cancelled,
    #[error("a supplied thread and its storage namespace must be provided together")]
    ThreadStorage,
    #[error("run resources have not finished opening")]
    NotReady,
    #[error(transparent)]
    #[host(delegate)]
    Storage(#[from] StorageError),
    #[error(transparent)]
    #[host(delegate)]
    Host(#[from] HostError),
}

/// Releases only language objects cached by an owned context adapter. Native code has already
/// ended its activation; this callback performs no storage or framework lifetime decisions.
#[crate::host(proxy(host = "livekit.agents.storage.context.ContextStore"))]
pub trait ContextHydration: Send + Sync {
    #[host(call, name = "_core_release_hydration")]
    fn release(&self) -> Result<(), HostError>;
}

#[crate::host(class(name = "_core_RunResources"))]
pub trait RunResources: Send + Sync + Sized + 'static {
    #[host(env)]
    type Env;
    #[host(class(name = "Storage"))]
    type Namespace: Send + Sync + 'static;
    #[host(class(name = "_core_ContextActivation"))]
    type Activation: Send + Sync + 'static;

    #[host(new)]
    fn new(
        env: &Self::Env,
        namespace: Option<Arc<Self::Namespace>>,
        thread_id: Option<String>,
        activation: Option<Self::Activation>,
        lane: String,
        toolsets: Vec<Arc<dyn ToolsetHooks>>,
    ) -> Result<Self, ResourceError>;

    /// Open storage and toolsets before any run starts. Cancelling the caller requests cleanup;
    /// the native owner retains every operation until it settles.
    #[host(await)]
    fn open(&self) -> Result<BoxFuture<'static, Result<(), ResourceError>>, ResourceError>;
    #[host(getter)]
    fn namespace(&self) -> Result<Option<Arc<Self::Namespace>>, ResourceError>;
    #[host(getter)]
    fn thread_id(&self) -> Result<Option<String>, ResourceError>;
    #[host(getter)]
    fn activation(&self) -> Result<Self::Activation, ResourceError>;
    #[host(getter)]
    fn closed(&self) -> bool;
    #[host(call)]
    fn retain_hydration(&self, hydration: Arc<dyn ContextHydration>);
    /// One owned cleanup; cancelled or repeated waiters can rejoin it.
    #[host(await)]
    fn aclose(&self) -> BoxFuture<'static, Result<(), ResourceError>>;
    /// Settle failed host-side initialization without replacing its original exception with a
    /// cleanup failure. Cleanup failures remain logged by the native owner.
    #[host(await)]
    fn rollback(&self) -> BoxFuture<'static, ()>;
}
