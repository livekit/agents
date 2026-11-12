use std::collections::{BTreeMap, BTreeSet};

use crate::{ChatItem, ChatMessage};
use serde_json::Value;

// Routing metadata contains no callbacks, tasks or host handles.

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

#[crate::host(record(
    module = "text_session",
    path = "livekit.agents.text_session",
    emit,
    extract,
    persisted
))]
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
}

#[crate::host(exceptions(
    module = "text_session_errors",
    path = "livekit.agents.text_session",
    classes(TextSessionRoutingError(
        base = "ValueError",
        doc = "The supplied state or an agent-tool route cannot be used."
    ))
))]
const _: () = ();

#[crate::host(error(kind = "livekit.agents.text_session.TextSessionRoutingError"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingError {
    Agent(crate::agent::AgentError),
    MaxDepth { max_depth: usize },
    UnknownAgent(String),
    UnknownTool(String),
    MissingOutputType(String),
    InvalidState(String),
    MissingAcceptedOutput,
}

impl std::fmt::Display for RoutingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Agent(error) => error.fmt(f),
            Self::MaxDepth { max_depth } => {
                write!(f, "agent-tool call exceeds max_depth={max_depth}")
            }
            Self::UnknownAgent(name) => write!(f, "unknown session agent: {name}"),
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

impl std::error::Error for RoutingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Agent(error) => Some(error),
            _ => None,
        }
    }
}

use crate::agent::run::{RunError, RunResult};

#[derive(Debug, Clone, PartialEq)]
pub struct TextSessionTurn {
    pub result: RunResult,
    pub state: TextSessionState,
}

#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq)]
pub enum TextSessionError {
    #[host(delegate)]
    Routing(RoutingError),
    #[host(delegate)]
    Run(RunError),
    #[host(delegate)]
    Storage(crate::storage::StorageError),
}

impl std::fmt::Display for TextSessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Routing(error) => error.fmt(f),
            Self::Run(error) => error.fmt(f),
            Self::Storage(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for TextSessionError {}

use crate::agent::run::StreamEvent;
use crate::agent::{AgentCallbacks, AgentDefinition};
use crate::errors::Invalid;
use crate::utils::stream::StreamOutput;
use crate::{ApiConnectOptions, HostRef, Llm};
use std::sync::Arc;

/// Receives accepted session transitions before dependent execution proceeds.
#[crate::host(native(refused = "expected a native session binding"))]
pub trait SessionStateSink: Send + Sync {
    fn publish<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        state: TextSessionState,
    ) -> crate::BoxFuture<'a, Result<(), crate::storage::StorageError>>;
}

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
    #[host(default = None)]
    pub session_sink: Option<Arc<dyn SessionStateSink>>,
    pub session_id: String,
    /// What this request's chat item ids are drawn from; `None` draws a random one.
    #[host(default = None)]
    pub id_seed: Option<u64>,
    pub execution_id: String,
    #[host(default = None)]
    pub task_id: Option<String>,
    pub message: ChatItem,
    pub state: Option<TextSessionState>,
    pub delegated: bool,
    pub conn_options: ApiConnectOptions,
}

/// A settled turn, with the value the host validated when the output is one. The host reads it
/// as its [`TextSessionFinished`].
#[crate::host(codec(emit, host = "TextSessionFinished", via = crate::text_session::TextSessionFinished))]
#[derive(Debug, Clone)]
pub struct TextSessionSettled {
    pub turn: TextSessionTurn,
    pub output: Option<HostRef>,
}

/// A settled turn as the host reads it: the active frame's run and the state to save.
#[crate::host(record(module = "text_session", emit))]
#[derive(Debug, Clone)]
pub struct TextSessionFinished {
    pub run: crate::agent::run::RunFinished,
    pub state: TextSessionState,
}

impl From<TextSessionSettled> for TextSessionFinished {
    fn from(TextSessionSettled { turn, output }: TextSessionSettled) -> Self {
        let run = crate::agent::run::RunSettled { result: turn.result, output }.into();
        TextSessionFinished { run, state: turn.state }
    }
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
    Failed {
        error: TextSessionError,
        #[host(default_factory = "list")]
        input_requirements: Vec<crate::agent::run::RunInputRequirement>,
        /// The original model failure, retained beside its correlated outcome metadata.
        #[host(default = None)]
        provider_error: Option<crate::errors::ProviderError>,
    },
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
    ) -> Result<Self, Invalid>;

    #[host(call)]
    fn run(&self, request: TextSessionRequest) -> Result<Self::Updates, Invalid>;

    #[host(call)]
    fn run_stream(
        &self,
        request: TextSessionRequest,
        stream: Arc<Self::Stream>,
        resources: Arc<Self::Resources>,
    ) -> Result<(), Invalid>;

    /// The latest usage snapshot of the turn, across its local invocations; never advances it.
    #[host(getter)]
    fn usage(&self) -> crate::usage::Usage;

    /// Acknowledge selected presentation before the child can release its parent.
    #[host(call)]
    fn release(&self, delivery: TextSessionDelivery) -> Result<(), Invalid>;

    #[host(call)]
    fn progress(&self, invocation_id: &str, operation: u64, message: String)
    -> Result<(), Invalid>;
}

/// One call of an agent tool outside a TextSession: the child agent runs in process as a delegate
/// of the call (tools.md "Agents as tools"). Its decisions are `lk_agents_core::agent_tool`'s.
#[crate::host(class(name = "_core_LocalInteraction"))]
pub trait LocalInteraction: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// `child` runs for `function_call`, the parent's call of `tool`; `depth` counts the agent
    /// tools it runs under, itself included.
    #[host(new)]
    fn new(
        env: &Self::Env,
        child: TextSessionAgent,
        tool: LocalAgentTool,
        function_call: ChatItem,
        depth: usize,
    ) -> Result<Self, Invalid>;

    /// Run the child on `arguments` from the parent's `history`, under the caller's span: its
    /// tools' progress reaches `host.progress`; on a call (`live`) the request is announced
    /// through `host.delegated` and each question it asks is answered through `host.ask`, the
    /// answer resuming it. Returns the call's outcome; off a call, the child's question.
    #[host(await)]
    #[allow(clippy::too_many_arguments, reason = "Mirrors the host call's arguments")]
    async fn run(
        &self,
        env: &Self::Env,
        host: Arc<dyn crate::a2a::remote::DelegationHost>,
        arguments: Value,
        history: Vec<ChatItem>,
        live: bool,
        conn_options: ApiConnectOptions,
        #[host(context)] context: crate::invoke::CallContext,
        cx: crate::telemetry::Cx,
    ) -> Result<crate::llm::tool_executor::ToolOutcome, Invalid>;

    /// The tool call was cancelled: the running child is cancelled; `settled()` waits for it.
    #[host(call)]
    fn cancelled(&self);

    /// After `cancelled()`: the child's settlement, bounded by the teardown limit.
    #[host(await)]
    async fn settled(&self);

    /// The call's retained state: the child's own history while it waits on the person.
    #[host(getter)]
    fn extra(&self) -> serde_json::Map<String, Value>;

    /// What the child consumed across its invocations, the running one included.
    #[host(getter)]
    fn usage(&self) -> crate::usage::Usage;

    /// `ctx.update(message)` from a tool body of the child's invocation `invocation_id`.
    #[host(call)]
    fn progress(&self, invocation_id: &str, operation: u64, message: String)
    -> Result<(), Invalid>;
}
