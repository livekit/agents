//! Read-only conversation views; execution resources and command capabilities are not stored.

pub mod remote;
pub mod snapshot;

use std::collections::BTreeMap;

use serde_json::{Map, Value};

/// A private acquired checkpoint scope. Dropping it releases its storage ownership; it is
/// neither a saved snapshot nor a host API. The owner applies decisions before committing it.
pub trait SessionCheckpoint: Send + Sync {
    /// Rows reconciled under the acquired lease; memory-only checkpoints have no reload.
    fn history(&self) -> Option<crate::storage::voice::VoiceHistory>;
    fn commit(&self) -> crate::BoxFuture<'_, Result<Value, crate::storage::StorageError>>;
}

/// Owner-qualified input occurrence. Local continuation and remote transport remain distinct.
#[crate::host(record(module = "session_input", emit, extract))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[serde(tag = "owner", rename_all = "snake_case")]
pub enum InputRequestRef {
    Agent {
        agent_id: String,
        request_id: String,
    },
    RemoteTask {
        remote_id: String,
        task: crate::delegation::remote::RemoteTaskRef,
        request_id: String,
    },
}

impl InputRequestRef {
    /// Transport addressing exists only after the reference identifies a remote owner.
    pub fn remote(&self) -> Option<(&str, &crate::delegation::remote::RemoteTaskRef, &str)> {
        match self {
            Self::RemoteTask { remote_id, task, request_id } => Some((remote_id, task, request_id)),
            Self::Agent { .. } => None,
        }
    }
}

/// Native delivery and response admission. A reference never authorizes its own presentation.
#[derive(Debug, Clone)]
pub enum SessionInputCommand {
    Presented { requests: Vec<InputRequestRef>, message_id: String },
    Respond { requests: Vec<InputRequestRef>, message_id: String },
    Correct { request: InputRequestRef, previous_message_id: String, message_id: String },
}

/// Detector correlation only: matching a response does not approve a business operation.
#[derive(Debug, Clone, PartialEq)]
pub enum InputRoute {
    NoMatch,
    Matched(Vec<InputRequestRef>),
    Unclear,
}

/// Serialized conversation authority. No host policy callback or saved command capability.
#[crate::host(native(refused = "expected the active native conversation owner"))]
pub trait SessionCommands: Send + Sync {
    fn checkpoint(&self) -> crate::BoxFuture<'_, Result<Value, crate::storage::StorageError>>;

    fn input<'a>(
        &'a self,
        command: SessionInputCommand,
    ) -> crate::BoxFuture<'a, Result<(), crate::storage::StorageError>>;

    /// Canonical held human response, before any outgoing request has reserved it.
    fn response<'a>(
        &'a self,
        reference: &'a InputRequestRef,
    ) -> crate::BoxFuture<'a, Result<crate::ChatMessage, crate::storage::StorageError>>;

    /// One undispatched human response for this destination. Ambiguity requires explicit routing.
    fn held_input<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
    ) -> crate::BoxFuture<'a, Result<Option<InputRequestRef>, crate::storage::StorageError>>;

    fn recover<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
        origin: &'a ToolCallRef,
    ) -> crate::BoxFuture<
        'a,
        Result<Option<crate::delegation::remote::RemoteRecovery>, crate::storage::StorageError>,
    >;

    fn default_context<'a>(
        &'a self,
        remote_id: &'a str,
        endpoint: &'a str,
    ) -> crate::BoxFuture<'a, Result<Option<String>, crate::storage::StorageError>>;

    /// Returns an open input observation's exact occurrence only after its checkpoint.
    fn remote<'a>(
        &'a self,
        command: crate::delegation::remote::RemoteCommand,
    ) -> crate::BoxFuture<'a, Result<Option<InputRequestRef>, crate::storage::StorageError>>;
}

#[crate::host(record(
    module = "session",
    path = "livekit.agents.storage.session_state",
    emit,
    extract,
    dataclass,
    frozen,
    kw_only,
    slots
))]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolCallRef {
    /// Agent context containing the originating tool call.
    pub agent_id: String,
    /// Tool-call ID within that context, not the configured tool's name.
    pub call_id: String,
}

#[crate::host(record(module = "session", emit, extract))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentCallStatus {
    Approval,
    Ready,
    Running,
}

#[crate::host(record(
    module = "session",
    path = "livekit.agents.storage.session_state",
    emit,
    dataclass,
    frozen,
    kw_only,
    slots
))]
#[derive(Debug, Clone, PartialEq)]
pub struct AgentCallState {
    /// Parent tool call receiving the child's accepted output.
    pub origin: ToolCallRef,
    /// Parent model invocation that admitted this call.
    pub parent_invocation_id: String,
    /// Parent workflow/task correlation; independent of the child agent-state ID.
    pub parent_execution_id: String,
    /// Name of the agent-as-tool function exposed by the parent.
    pub parent_tool_name: String,
    /// Configured agent to execute after approval and validation.
    pub agent_name: String,
    /// Accepted JSON arguments passed to that agent-as-tool function.
    #[host(annotation = "dict[str, pydantic.JsonValue]")]
    pub validated_arguments: Map<String, Value>,
    /// Canonical validated-argument identity used by approval.
    pub identity: String,
    /// Local admission/execution state; never an A2A task status.
    pub status: AgentCallStatus,
    /// Child's entry in the session map, allocated only when it starts.
    pub agent_id: Option<String>,
}

#[crate::host(record(module = "session", emit, extract))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputRequestState {
    /// Waiting for a response to this occurrence.
    Open,
    /// A response is reserved; its remote admission is not yet confirmed.
    Responding,
    /// A successful response or observed task history confirmed admission.
    Responded,
    /// A replacement input occurrence retired this unsent selection.
    Superseded,
    /// Input is no longer requested; retained correlation does not prove admission.
    Closed,
}

#[crate::host(record(
    module = "session",
    path = "livekit.agents.storage.session_state",
    emit,
    extract,
    dataclass,
    frozen,
    kw_only,
    slots
))]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AgentInputRequest {
    /// This occurrence's response lifecycle, independent of its asking tool's completion.
    pub state: InputRequestState,
    /// The requested question; native delivery receipts separately establish presentation.
    pub source_message: crate::ChatMessage,
    /// Human response reserved for this occurrence, if one was selected.
    pub response_message: Option<crate::ChatMessage>,
    /// Asking call's provenance; it never directs the next input to a tool.
    pub origin: Option<ToolCallRef>,
    /// Canonical assistant delivery proving this request was presented, not merely queued.
    #[host(skip)]
    pub presented_message_id: Option<String>,
}

#[crate::host(record(
    module = "session",
    path = "livekit.agents.storage.session_state",
    emit,
    dataclass,
    frozen,
    kw_only,
    slots
))]
#[derive(Debug, Clone, PartialEq)]
pub struct AgentState {
    /// Configured behavior; no Python Agent instance is serialized.
    pub agent_name: String,
    /// This agent context's private model history.
    #[host(
        emit_class = "livekit.agents.llm.chat_context._ReadOnlyChatContext",
        annotation = "livekit.agents.llm.ChatContext"
    )]
    pub chat_ctx: Vec<crate::ChatItem>,
    /// Parent call receiving this agent's output; None for the root.
    pub origin: Option<ToolCallRef>,
    /// Unresolved local agent calls, indexed by the parent's tool-call ID.
    pub calls: BTreeMap<String, AgentCallState>,
    /// Retained input occurrences owned by this agent context.
    pub input_requests: BTreeMap<String, AgentInputRequest>,
    /// Latest invocation accepted an output; a JSON null is still a completed output.
    pub completed: bool,
    /// Accepted serialized output retained until its parent receives it.
    #[host(annotation = "pydantic.JsonValue | None")]
    pub output: Option<Value>,
    /// Diagnostic failure, separate from a successful output or cancellation.
    pub error: Option<String>,
}
