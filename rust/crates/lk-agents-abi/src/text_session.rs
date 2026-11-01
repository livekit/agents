//! Owned snapshots crossing the application load/save boundary. History uses the shared
//! ChatItem codec at that boundary; routing metadata contains no callbacks, tasks or host handles.

use std::collections::{BTreeMap, BTreeSet};

use crate::{ChatItem, ChatMessage};
use serde_json::Value;

/// A stable backend conversation frame, distinct from a call or invocation.
#[crate::host(record(module = "text_session", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct FrameId {
    pub value: String,
}

#[crate::host(record(module = "text_session", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParentCall {
    pub frame_id: FrameId,
    pub call_id: String,
}

#[crate::host(record(module = "text_session", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", serde(tag = "status", rename_all = "snake_case"))]
pub enum ChildCallStatus {
    Approval,
    Ready,
    Running { frame_id: FrameId },
}

#[crate::host(record(module = "text_session", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct PendingChildCall {
    pub parent: ParentCall,
    pub invocation_id: String,
    pub execution_id: String,
    pub tool_name: String,
    pub agent_name: String,
    pub validated_arguments: Value,
    /// The shared approval gate's canonical validated identity, never the raw model arguments.
    pub identity: String,
    pub status: ChildCallStatus,
}

#[crate::host(record(module = "text_session", emit, extract))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct SessionFrame {
    pub id: FrameId,
    pub agent_name: String,
    pub history: Vec<ChatItem>,
    pub parent: Option<ParentCall>,
    pub context_scope: String,
    pub completed: bool,
    pub pending_calls: Vec<PendingChildCall>,
    pub answer_tool: Option<String>,
    /// Delivery receipts, in conversation order, excluding the initial tool task and fork.
    pub visible_dialogue: Vec<ChatMessage>,
    pub starting_history_ids: BTreeSet<String>,
    pub transferred_message_ids: BTreeSet<String>,
    /// Frozen at creation: reconfiguration on a later request cannot change the return policy.
    pub merge_chat_history: bool,
}

#[crate::host(record(module = "text_session", emit, extract, persisted))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct TextSessionState {
    pub session_id: String,
    pub execution_id: String,
    pub root_frame_id: FrameId,
    pub active_frame_id: FrameId,
    /// Root first, immediate parent last. Completed frames never occur here.
    pub ancestors: Vec<FrameId>,
    pub frames: BTreeMap<FrameId, SessionFrame>,
    pub next_id: u64,
}

#[crate::host(record(module = "text_session", emit, extract))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatHistory {
    Fresh,
    Fork,
}

#[crate::host(record(module = "text_session", emit, extract))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalAgentTool {
    pub agent_name: String,
    pub chat_history: ChatHistory,
    pub merge_chat_history: bool,
    /// None shares the parent's scope. A supplied reference is application-owned.
    pub context_scope: Option<String>,
}

#[crate::host(record(module = "text_session", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingError {
    MaxDepth {
        max_depth: usize,
    },
    #[host(field = "message")]
    UnknownAgent(String),
    #[host(field = "message")]
    UnknownScope(String),
    #[host(field = "message")]
    UnknownTool(String),
    #[host(field = "message")]
    MissingOutputType(String),
    #[host(field = "message")]
    InvalidState(String),
    MissingAcceptedOutput,
}

impl std::fmt::Display for RoutingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxDepth { max_depth } => {
                write!(f, "agent-tool call exceeds max_depth={max_depth}")
            }
            Self::UnknownAgent(name) => write!(f, "unknown session agent: {name}"),
            Self::UnknownScope(scope) => write!(f, "unknown context scope: {scope}"),
            Self::UnknownTool(tool) => write!(f, "unknown local agent tool: {tool}"),
            Self::MissingOutputType(name) => {
                write!(f, "agent tool {name} requires explicit output_type")
            }
            Self::InvalidState(reason) => write!(f, "invalid text session state: {reason}"),
            Self::MissingAcceptedOutput => {
                f.write_str("child completed without a serialized accepted output")
            }
        }
    }
}

impl std::error::Error for RoutingError {}

use crate::agent::run::{RunError, RunResult};

#[crate::host(record(module = "text_session", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct TextSessionTurn {
    pub result: RunResult,
    pub state: TextSessionState,
}

#[crate::host(record(module = "text_session", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextSessionError {
    #[host(field = "error")]
    Routing(RoutingError),
    #[host(field = "error")]
    Run(RunError),
}

impl std::fmt::Display for TextSessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Routing(error) => error.fmt(f),
            Self::Run(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for TextSessionError {}

use crate::agent::run::{InvalidRun, StreamEvent};
use crate::agent::{AgentCallbacks, AgentDefinition};
use crate::utils::stream::StreamOutput;
use crate::{ApiConnectOptions, HostRef, Llm};
use std::sync::Arc;

/// Shared definitions and live resources; no host resource enters the saved state.
#[crate::host(record(module = "text_session", extract))]
#[derive(Clone)]
pub struct TextSessionAgent {
    pub definition: AgentDefinition,
    pub llm: Arc<dyn Llm>,
    pub host: Arc<dyn AgentCallbacks>,
    pub output_type_explicit: bool,
    pub agent_tools: BTreeMap<String, LocalAgentTool>,
    /// A callback context for this agent, captured at session entry.
    pub host_context: HostRef,
    /// Canonical tool pool for this agent, shared across its invocations.
    #[host(default = None)]
    pub tools_host: Option<HostRef>,
}

#[crate::host(record(module = "text_session", extract))]
#[derive(Clone)]
pub struct TextSessionRequest {
    pub session_id: String,
    pub execution_id: String,
    #[host(default = None)]
    pub task_id: Option<String>,
    pub root_scope: String,
    pub contexts: BTreeMap<String, HostRef>,
    /// Construct child application contexts in their frame-owned scopes through the host.
    #[host(default = false)]
    pub separate_contexts: bool,
    pub message: ChatItem,
    pub state: Option<TextSessionState>,
    pub delegated: bool,
    pub conn_options: ApiConnectOptions,
}

#[crate::host(record(module = "text_session", emit))]
#[derive(Debug, Clone)]
pub struct TextSessionSettled {
    pub turn: TextSessionTurn,
    pub output: Option<HostRef>,
}

#[crate::host(record(module = "text_session", emit))]
#[derive(Debug, Clone)]
pub enum TextSessionUpdate {
    Stream {
        sequence: u64,
        frame_id: FrameId,
        agent_name: String,
        invocation_id: String,
        event: StreamEvent,
    },
    #[host(field = "invocation_id")]
    InvocationSettled(String),
    #[host(field = "invocation_id")]
    Parked(String),
    #[host(field = "pending")]
    IncompleteCleanup(Vec<String>),
    #[host(field = "settled")]
    Finished(Box<TextSessionSettled>),
    #[host(field = "error")]
    Failed(TextSessionError),
}

#[crate::host(record(module = "text_session", extract))]
pub struct TextSessionDelivery {
    pub invocation_id: String,
    pub messages: BTreeMap<String, String>,
}

#[crate::host(class(name = "TextSessionRunner"))]
pub trait TextSessionRunner: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(class(name = "_core_RunStream"))]
    type Stream: Send + Sync + 'static;
    #[host(class(name = "_core_RunResources"))]
    type Resources: Send + Sync + 'static;

    #[host(source(item = TextSessionUpdate, class = "TextSessionUpdates"))]
    type Updates: StreamOutput<TextSessionUpdate> + 'static;

    #[host(new)]
    fn new(
        env: &Self::Env,
        root_agent: String,
        agents: Vec<TextSessionAgent>,
        max_depth: usize,
    ) -> Result<Self, InvalidRun>;

    #[host(call)]
    fn run(&self, request: TextSessionRequest) -> Result<Self::Updates, InvalidRun>;

    #[host(call)]
    fn run_stream(
        &self,
        request: TextSessionRequest,
        stream: Arc<Self::Stream>,
        resources: Arc<Self::Resources>,
    ) -> Result<(), InvalidRun>;

    /// The latest usage snapshot of the turn, across its local invocations; never advances it.
    #[host(getter)]
    fn usage(&self) -> crate::usage::Usage;

    /// Acknowledge selected presentation before the child can release its parent.
    #[host(call)]
    fn release(&self, delivery: TextSessionDelivery) -> Result<(), InvalidRun>;

    #[host(call)]
    fn progress(
        &self,
        invocation_id: &str,
        operation: u64,
        message: String,
    ) -> Result<(), InvalidRun>;
}
