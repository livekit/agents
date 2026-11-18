use std::collections::BTreeMap;

use crate::ChatItem;
use serde_json::Value;

// Routing metadata contains no callbacks, tasks or host handles.

pub use crate::session::snapshot::{
    ChildCallStatus, FrameId, FrameOptions, ParentCall, PendingChildCall, SessionFrame,
    SessionSnapshot,
};

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
    pub merge_chat_ctx: bool,
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
    pub state: SessionSnapshot,
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

/// Native transport ownership captured before a managed turn starts. Request receipts and
/// business tasks stay separate; a result delivery reuses its original business task IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveTurnOrigin {
    pub request_id: String,
    pub execution_id: String,
    pub task_ids: Vec<String>,
}

/// Native result propagation to the endpoint's existing leased owner. A managed handler may
/// still be awaiting its final result when an intermediate turn is committed as Working.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveTurnReport {
    pub turn: u64,
    pub origin: LiveTurnOrigin,
    pub result: Box<Result<TextSessionTurn, TextSessionError>>,
    pub working: bool,
}

/// Native commit receipt, admitted separately from a handler's eventual return.
pub type LiveTurnCommit = crate::BoxFuture<'static, Result<(), crate::storage::StorageError>>;

/// Framework checkpoints belong to the leased endpoint owner, rather than any one handler.
/// This native sender writes no application fields and grants no new execution authority.
pub trait SessionCheckpointPort: Send + Sync {
    fn publish(
        &self,
        state: SessionSnapshot,
    ) -> crate::BoxFuture<'_, Result<Value, crate::storage::StorageError>>;
}

/// Receives accepted session transitions before dependent execution proceeds.
#[crate::host(native(refused = "expected a native session binding"))]
pub trait SessionStateSink: Send + Sync {
    /// The explicit binding lends its current owner; saved snapshots never install one.
    fn commands(
        &self,
    ) -> Result<Option<Arc<dyn crate::session::SessionCommands>>, crate::storage::StorageError>
    {
        Ok(None)
    }

    fn checkpoint_port(
        &self,
    ) -> Result<Option<Arc<dyn SessionCheckpointPort>>, crate::storage::StorageError> {
        Ok(None)
    }

    /// Refresh a still-open bound view after the endpoint acknowledged its framework write.
    /// An expired handler view owes no update; it never becomes the checkpoint owner.
    fn acknowledge_checkpoint(&self, _state: Value) -> Result<(), crate::storage::StorageError> {
        Ok(())
    }

    /// Only a configured endpoint can opt managed execution into live scheduling.
    fn inbox_policy(&self) -> crate::a2a::options::Policy {
        crate::a2a::options::Policy::Queue
    }

    fn live_origin(&self) -> Option<LiveTurnOrigin> {
        None
    }

    /// Transfer this managed driver's task to the endpoint owner before any observer can
    /// drop it. The task retains its TaskSet and joins all of the driver's acquired work.
    fn retain_session(&self, _task: crate::OwnedTask) -> Result<(), crate::storage::StorageError> {
        Err(crate::storage::StorageError::Invalid("this binding has no live task owner".into()))
    }

    /// The outer future admits the report before its handler is allowed to finish. The
    /// returned owned receipt resolves only after the endpoint commits that exact turn.
    /// The endpoint owner commits this report before the conversation advances to another
    /// model turn. A binding outside a managed endpoint does not grant live scheduling.
    fn live_turn(
        &self,
        _cx: &crate::telemetry::Cx,
        _report: LiveTurnReport,
    ) -> crate::BoxFuture<'_, Result<LiveTurnCommit, crate::storage::StorageError>> {
        Box::pin(async {
            Err(crate::storage::StorageError::Invalid(
                "this binding has no live endpoint owner".into(),
            ))
        })
    }

    /// A live native executor lends a weak command capability separately from saved data.
    /// The binding clears it on close; decoding or assigning a snapshot never installs it.
    fn bind_commands(
        &self,
        commands: std::sync::Weak<dyn crate::session::SessionCommands>,
    ) -> Result<(), crate::storage::StorageError>;

    fn checkpoint_voice<'a>(
        &'a self,
        history: crate::storage::voice::VoiceHistory,
    ) -> crate::BoxFuture<
        'a,
        Result<Arc<dyn crate::session::SessionCheckpoint>, crate::storage::StorageError>,
    >;

    /// Read the bound conversation after its execution resources acquire and reload storage.
    fn restore(&self) -> Result<Option<SessionSnapshot>, crate::storage::StorageError>;

    /// Publish the live voice view without I/O or clearing unaccepted recognition progress.
    fn stage_voice(&self, state: SessionSnapshot) -> Result<(), crate::storage::StorageError>;
    fn publish<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        state: SessionSnapshot,
    ) -> crate::BoxFuture<'a, Result<Value, crate::storage::StorageError>>;
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
    pub output: Option<Arc<dyn crate::a2a::turn::HandlerPort>>,
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
    pub state: SessionSnapshot,
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

#[crate::host(class(name = "TextSessionRunner", native = crate::session::SessionCommands))]
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
        #[host(default = None)] delegation: Option<crate::delegation::DelegationOptions>,
    ) -> Result<Self, Invalid>;

    /// Runs the one request a runner serves; a second is refused.
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

    #[host(await)]
    fn progress(
        &self,
        invocation_id: &str,
        operation: u64,
        message: String,
        #[host(default = false)] silent: bool,
    ) -> crate::BoxFuture<'static, Result<(), crate::errors::ProviderError>>;

    /// Enqueue exact text now, or refuse it before admission. The private reader settles with
    /// the outcome at the source's local output boundary; dropping it never cancels the output.
    #[host(await)]
    fn say(
        &self,
        invocation_id: &str,
        operation: u64,
        message: String,
    ) -> Result<
        crate::BoxFuture<'static, crate::agent::run::SayOutcome>,
        crate::errors::ProviderError,
    >;
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
    #[host(await)]
    fn progress(
        &self,
        invocation_id: &str,
        operation: u64,
        message: String,
        #[host(default = false)] silent: bool,
    ) -> crate::BoxFuture<'static, Result<(), crate::errors::ProviderError>>;

    /// Enqueue exact text now, or refuse it before admission. The private reader settles with
    /// the outcome at the source's local output boundary; dropping it never cancels the output.
    #[host(await)]
    fn say(
        &self,
        invocation_id: &str,
        operation: u64,
        message: String,
    ) -> Result<
        crate::BoxFuture<'static, crate::agent::run::SayOutcome>,
        crate::errors::ProviderError,
    >;
}
