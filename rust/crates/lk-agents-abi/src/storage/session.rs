use std::collections::BTreeMap;

use super::StorageError;

#[crate::host(record(module = "storage_session", emit, extract))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionKind {
    Empty,
    Text,
    Voice,
}
use crate::text_session::{FrameId, SessionFrame, TextSessionState};

/// Read-only conversation state. Reads and results are detached; an explicitly bound
/// application field retains its identity while the native store updates its contents.
#[crate::host(class(name = "_core_PersistedSessionState", pid = false))]
pub trait PersistedSessionState: Clone + Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(#[host(default = None)] data: Option<String>) -> Result<Self, StorageError>;
    #[host(getter)]
    fn kind(&self) -> SessionKind;
    #[host(getter)]
    fn active_agent(&self) -> Option<String>;
    #[host(getter)]
    fn session_id(&self) -> Option<String>;
    #[host(getter)]
    fn execution_id(&self) -> Option<String>;
    #[host(getter)]
    fn root_frame_id(&self) -> Result<FrameId, StorageError>;
    #[host(getter)]
    fn active_frame_id(&self) -> Result<FrameId, StorageError>;
    #[host(getter)]
    fn ancestors(&self) -> Vec<FrameId>;
    #[host(getter)]
    fn frames(&self) -> BTreeMap<FrameId, SessionFrame>;
    #[host(call)]
    fn snapshot(&self) -> String;
    #[host(call)]
    fn _to_json(&self) -> String;
    #[host(call)]
    fn _text_state(&self) -> Result<Option<TextSessionState>, StorageError>;
    #[host(call)]
    fn _update(&self, data: String) -> Result<(), StorageError>;
    #[host(call)]
    fn _from_json(data: String) -> Result<String, StorageError>;
    #[host(call)]
    fn _from_text(state: TextSessionState) -> Result<String, StorageError>;
    #[host(call)]
    fn _from_voice(agent: String, history_ids: Vec<String>) -> Result<String, StorageError>;
}

/// A detached snapshot, or a view updated only by an explicit store binding.
#[crate::host(facade(path = "livekit.agents.storage.session_state", name = "SessionState", native = PersistedSessionState))]
pub trait SessionStateFacade: Sized {
    #[host(new)]
    fn new(#[host(default = None)] data: Option<String>) -> Result<Self, StorageError>;
    #[host(getter)]
    fn kind(&self) -> SessionKind;
    #[host(getter)]
    fn active_agent(&self) -> Option<String>;
    #[host(getter)]
    fn session_id(&self) -> Option<String>;
    #[host(getter)]
    fn execution_id(&self) -> Option<String>;
    #[host(getter)]
    fn root_frame_id(&self) -> Result<FrameId, StorageError>;
    #[host(getter)]
    fn active_frame_id(&self) -> Result<FrameId, StorageError>;
    #[host(getter)]
    fn ancestors(&self) -> Vec<FrameId>;
    #[host(getter)]
    fn frames(&self) -> BTreeMap<FrameId, SessionFrame>;
    #[host(returns = "..storage.session_state.SessionState")]
    fn snapshot(&self) -> String;
    fn _to_json(&self) -> String;
    fn _text_state(&self) -> Result<Option<TextSessionState>, StorageError>;
    fn _update(&self, data: String) -> Result<(), StorageError>;
    #[host(returns = "..storage.session_state.SessionState")]
    fn _from_json(data: String) -> Result<String, StorageError>;
    #[host(returns = "..storage.session_state.SessionState")]
    fn _from_text(state: TextSessionState) -> Result<String, StorageError>;
    #[host(returns = "..storage.session_state.SessionState")]
    fn _from_voice(agent: String, history_ids: Vec<String>) -> Result<String, StorageError>;
}

/// An inert execution declaration; tools receive the original context object.
#[crate::host(record(
    module = "storage_session",
    path = "livekit.agents.storage.session_state",
    dataclass,
    emit,
    frozen,
    generic = "Context_T"
))]
#[derive(Clone)]
pub struct BoundContext {
    #[host(annotation = "Context_T")]
    pub context: crate::HostRef,
    #[host(annotation = "typing.Any")]
    pub store: crate::HostRef,
}

/// Native invocation-local binding and checkpoint publisher.
#[crate::host(class(name = "_core_SessionBinding", native = crate::text_session::SessionStateSink))]
pub trait SessionBinding: Send + Sync + Sized + 'static {
    #[host(class(name = "_core_ContextActivation"))]
    type Activation: super::context::ContextActivation;
    #[host(class(name = "_core_PersistedSessionState"))]
    type View: PersistedSessionState;
    #[host(new)]
    fn new(activation: Self::Activation, view: Self::View) -> Result<Self, StorageError>;
    #[host(call)]
    fn update(&self, state: String) -> Result<(), StorageError>;
    #[host(call)]
    fn update_voice(&self, agent: String, history_ids: Vec<String>) -> Result<(), StorageError>;
    #[host(call)]
    fn validate_voice(&self, agent: String) -> Result<(), StorageError>;
    #[host(await)]
    async fn flush(
        &self,
        #[host(default = None)] history: Option<Vec<crate::ChatItem>>,
    ) -> Result<(), StorageError>;
    #[host(getter)]
    fn closed(&self) -> bool;
    #[host(call)]
    fn close(&self);
}

#[crate::host(facade(path = "livekit.agents.storage.context", name = "_SessionBinding", native = SessionBinding, extended, from_native))]
pub trait SessionBindingFacade {
    fn update(&self, state: String) -> Result<(), StorageError>;
    fn update_voice(&self, agent: String, history_ids: Vec<String>) -> Result<(), StorageError>;
    fn validate_voice(&self, agent: String) -> Result<(), StorageError>;
    #[host(bind(history = "_history()"))]
    async fn flush(&self) -> Result<(), StorageError>;
    #[host(getter)]
    fn closed(&self) -> bool;
    fn close(&self);
}
