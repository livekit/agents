//! Shared toolset resource ownership for text invocations and voice activations.

use crate::{BoxFuture, errors::HostError};
use std::{any::Any, sync::Arc};

#[crate::host(native(refused = "expected a native toolset lifetime"))]
pub trait Lifetime: Any + Send + Sync {
    fn acquire(&self, hooks: Arc<dyn ToolsetHooks>) -> Arc<dyn Acquisition>;
}

pub trait Acquisition: Send + Sync {
    fn result(&self) -> BoxFuture<'_, Result<Arc<dyn Lease>, HostError>>;
    fn cancel(&self) -> BoxFuture<'_, Result<(), HostError>>;
}
pub trait Lease: Send + Sync {
    fn close(&self) -> BoxFuture<'_, Result<(), HostError>>;
}

#[crate::host(native(refused = "expected a native toolset lease group"))]
pub trait LeaseGroup: Any + Send + Sync {
    fn close(&self) -> BoxFuture<'_, Result<(), HostError>>;
}

#[crate::host(proxy(host = "livekit.agents.llm.Toolset"))]
pub trait ToolsetHooks: Send + Sync {
    #[host(read_once, name = "_core_lifetime")]
    fn lifetime(&self) -> &Arc<dyn Lifetime>;
    #[host(await, name = "setup")]
    fn setup(
        &self,
        #[host(cancel)] cancel: Arc<dyn crate::invoke::Cancellation>,
    ) -> BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "aclose")]
    fn close(&self) -> BoxFuture<'_, Result<(), HostError>>;
    /// Framework-owned children acquired by the default setup, including before a user override
    /// subsequently fails. A failed setup does not call the failed resource's authored close.
    #[host(call, name = "_core_children")]
    fn children(&self) -> Result<Option<Arc<dyn LeaseGroup>>, HostError>;
}

#[crate::host(class(name = "_core_ToolsetLifetime", native = Lifetime))]
pub trait ToolsetLifetime: Send + Sync + Sized + 'static {
    #[host(env)]
    type Env;
    #[host(new)]
    fn new(env: &Self::Env) -> Self;
}

#[crate::host(class(name = "_core_ToolsetGroup", native = LeaseGroup))]
pub trait ToolsetGroup: Send + Sync + Sized + 'static {
    #[host(env)]
    type Env;
    #[host(new)]
    fn new(env: &Self::Env, toolsets: Vec<Arc<dyn ToolsetHooks>>) -> Self;
    #[host(await)]
    async fn start(&self) -> Result<(), HostError>;
    #[host(await)]
    async fn aclose(&self) -> Result<(), HostError>;
}

#[crate::host(native(refused = "expected a native toolset scope registry"))]
pub trait ScopesPort: Any + Send + Sync {
    fn acquire(
        &self,
        scope: u64,
        toolsets: Vec<Arc<dyn ToolsetHooks>>,
    ) -> BoxFuture<'_, Result<u64, HostError>>;
    fn discard(&self, activation: u64) -> BoxFuture<'_, Result<(), HostError>>;
    fn release(&self, scope: Option<u64>) -> BoxFuture<'_, Result<(), HostError>>;
}

#[crate::host(class(name = "_core_ToolsetScopes", native = ScopesPort))]
pub trait ToolsetScopes: Send + Sync + Sized + 'static {
    #[host(env)]
    type Env;
    #[host(new)]
    fn new(env: &Self::Env) -> Self;
    /// Zero is the whole-session scope; other IDs are stable host agent identities.
    #[host(await)]
    async fn acquire(
        &self,
        scope: u64,
        toolsets: Vec<Arc<dyn ToolsetHooks>>,
    ) -> Result<u64, HostError>;
    /// End the oldest outstanding activation of that agent, or every scope on shutdown.
    #[host(await)]
    async fn release(&self, scope: Option<u64>) -> Result<(), HostError>;
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(module = "toolset_events", path = "livekit.agents.beta", dataclass))]
pub struct ToolCalledEvent {
    #[host(annotation = "livekit.agents.voice.events.RunContext")]
    pub ctx: crate::HostRef,
    #[host(annotation = "dict[str, object]")]
    pub arguments: std::collections::BTreeMap<String, crate::HostRef>,
}

#[crate::host(record(module = "toolset_events", path = "livekit.agents.beta", dataclass))]
pub struct ToolCompletedEvent {
    #[host(annotation = "livekit.agents.voice.events.RunContext")]
    pub ctx: crate::HostRef,
    #[host(annotation = "object")]
    pub output: crate::HostRef,
}

/// A search candidate derived from a single tool at index time.
#[crate::host(record(module = "toolset_events", path = "livekit.agents.beta", dataclass))]
pub struct SearchItem {
    #[host(annotation = "livekit.agents.llm.tool_context.Tool | livekit.agents.llm.Toolset")]
    pub source: crate::HostRef,
    pub name: String,
    pub description: String,
    #[host(default_factory = "dict")]
    pub parameters: std::collections::BTreeMap<String, String>,
    #[host(default = None, hide_repr, annotation = "typing.Any")]
    pub index_data: Option<crate::HostRef>,
}
