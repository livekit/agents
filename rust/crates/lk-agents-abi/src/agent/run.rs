use std::sync::Arc;

use serde_json::Value;

use super::{AgentCallbacks, AgentDefinition};
use crate::errors::Invalid;
use crate::utils::stream::StreamOutput;
use crate::{ApiConnectOptions, ChatItem, HostRef, Llm};

/// One output candidate's host validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValidationId(pub u64);

/// One tool call still awaiting human input. This is correlated outcome metadata, not a
/// presentation queue: consumers decide when and how to ask, and must retain the call identity.
#[crate::host(record(module = "agent_run", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct RunInputRequirement {
    pub invocation_id: String,
    pub call: crate::a2a::turn::PendingCall,
    pub question: String,
    pub verbatim: bool,
}

/// The settled invocation: its status, then the history and usage every branch exposes. The host
/// reads it as the caller's `RunResult` ([`RunFinished`]).
#[derive(Debug, Clone, PartialEq)]
pub struct RunResult {
    pub outcome: RunOutcome,
    /// The managed turn yielded at a safe boundary; history and usage still describe its work.
    pub interrupted: bool,
    /// The incoming message and everything this invocation recorded, in order.
    pub new_messages: Vec<ChatItem>,
    /// The supplied history plus this invocation, without instructions.
    pub all_messages: Vec<ChatItem>,
    /// This invocation's usage, settled (usage.md).
    pub usage: crate::usage::Usage,
    /// The part of the answer or question that reached the person exactly as written.
    pub verbatim: Option<String>,
    /// The selected tool call waiting on human input, supplied by the execution core.
    pub pending_call: Option<crate::a2a::turn::PendingCall>,
    /// Every tool-originated input requirement, in model-call order. The singular outcome and
    /// `pending_call` remain the compatibility projection; other requirements are not answered.
    pub input_requirements: Vec<RunInputRequirement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RunOutcome {
    Completed {
        output: Output,
    },
    /// The run ends on a question for the person.
    InputRequired(RunQuestion),
    /// The turn ended with no output: `StopResponse`, a tool result asking for no reply, an
    /// all-silent step or an interruption. Never `Completed`, whose output is the declared type.
    NoOutput,
}

/// The question a run ends on (`InputRequired`): asked as written when `verbatim`. The person's
/// next message goes to `answer_tool`, when one is named, as its first string parameter
/// (`InputRequired(tool=)`); the next invocation is told so.
#[derive(Debug, Clone, PartialEq)]
pub struct RunQuestion {
    pub question: String,
    pub verbatim: bool,
    pub answer_tool: Option<String>,
}

/// The accepted output of a completed invocation.
#[derive(Debug, Clone, PartialEq)]
pub enum Output {
    /// A `str` output.
    Text(String),
    /// The value the host validated for this candidate; the runtime retains it and the update
    /// carries it.
    Validated(ValidationId),
}

/// What every branch of the caller's `RunResult` exposes: its history, usage and correlated
/// input requests. The host builds a branch from a `RunSettled`.
#[crate::host(record(module = "run_result", name = "_Settled", emit, dataclass, kw_only))]
#[python(
    method(
        name = "new_messages",
        ret = "list[.chat_items.ChatItem]",
        hook = "run_new_messages",
        doc = "This invocation's input and everything it recorded, tool calls and results included."
    ),
    method(
        name = "all_messages",
        ret = "list[.chat_items.ChatItem]",
        hook = "run_all_messages",
        doc = "The supplied history plus this invocation, without instructions: the next direct\nrun's ``message_history``."
    ),
    property(
        name = "expect",
        ret = "livekit.agents.voice.run_result.RunAssert",
        hook = "run_expect",
        cached,
        doc = "Assertions over this invocation's events, in order (testing.md)."
    )
)]
#[derive(Debug, Clone)]
pub struct SettledRun {
    #[host(rename = "_new_messages", hide_repr)]
    pub new_messages: Vec<ChatItem>,
    #[host(rename = "_all_messages", hide_repr)]
    pub all_messages: Vec<ChatItem>,
    /// What this invocation consumed: an immutable, settled snapshot (usage.md).
    #[host(hide_repr)]
    pub usage: crate::usage::Usage,
    /// The part of the answer, or of the question, that reached the person exactly as written;
    /// `None` when nothing did (tools.md "Verbatim").
    #[host(hide_repr)]
    pub verbatim: Option<String>,
    #[host(rename = "_pending_call", default = None, hide_repr)]
    pub pending_call: Option<crate::a2a::turn::PendingCall>,
    /// All correlated tool requests awaiting input, in model order. This metadata does not select
    /// a presentation order or answer additional requests automatically.
    #[host(default_factory = "list", hide_repr)]
    pub input_requirements: Vec<RunInputRequirement>,
}

/// `RunResult` with `status == "completed"`: `output` is the accepted value.
#[crate::host(record(
    module = "run_result",
    path = "livekit.agents.voice.run_result",
    emit,
    dataclass,
    kw_only,
    base = "_Settled",
    generic = "Output_T",
    constant(status = "completed")
))]
#[derive(Debug, Clone)]
pub struct RunCompleted {
    #[host(flatten)]
    pub settled: SettledRun,
    #[host(annotation = "Output_T")]
    pub output: RunOutput,
}

/// A completed invocation's output: its text, or the value the host validated.
#[crate::host(record(module = "run_result", emit, transparent))]
#[derive(Debug, Clone)]
pub enum RunOutput {
    Text(String),
    Validated(HostRef),
}

/// `RunResult` with `status == "input_required"`: the invocation asks `question`.
#[crate::host(record(
    module = "run_result",
    path = "livekit.agents.voice.run_result",
    emit,
    dataclass,
    kw_only,
    base = "_Settled",
    constant(status = "input_required")
))]
#[derive(Debug, Clone)]
pub struct RunInputRequired {
    #[host(flatten)]
    pub settled: SettledRun,
    pub question: String,
}

/// `RunResult` with `status == "no_output"`: the turn ended without an output, on
/// `StopResponse`, a tool result asking for no reply, a silent step or an interruption.
#[crate::host(record(
    module = "run_result",
    path = "livekit.agents.voice.run_result",
    emit,
    dataclass,
    kw_only,
    base = "_Settled",
    constant(status = "no_output")
))]
#[derive(Debug, Clone)]
pub struct RunNoOutput {
    #[host(flatten)]
    pub settled: SettledRun,
}

/// One agent invocation (agents.md "Running a turn"): narrow on `status` for `output` or
/// `question`; every branch exposes `new_messages()`, `all_messages()` and `usage`.
#[crate::host(record(
    module = "run_result",
    name = "RunResult",
    path = "livekit.agents.voice.run_result",
    emit,
    transparent,
    generic = "Output_T",
    generic_default = "str"
))]
#[derive(Debug, Clone)]
pub enum CallerRunResult {
    #[host(annotation = "RunCompleted[Output_T]")]
    Completed(RunCompleted),
    InputRequired(RunInputRequired),
    NoOutput(RunNoOutput),
}

/// Why an invocation settled without a result: the exception its run raises.
#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// The agent has no model to run a step with.
    NoModel,
    #[host(kind = "asyncio.CancelledError")]
    Cancelled,
    /// The model request failed after the provider's own recovery.
    #[host(kind = "livekit.agents._bindings.APIError", kwargs(retryable = false))]
    Model(String),
    /// A host callback the invocation depends on (instructions, output validation) failed.
    Host(String),
    /// The model kept calling `tool` wrongly past its retry budget.
    #[host(kind = "livekit.agents._bindings.UnexpectedModelBehavior")]
    ToolRetriesExceeded { tool: String, retries: u32 },
    /// The model kept producing rejected final output past its retry budget.
    #[host(kind = "livekit.agents._bindings.UnexpectedModelBehavior")]
    OutputRetriesExceeded { retries: u32 },
    /// The voice session closed before the run settled.
    SessionClosed,
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::NoModel => f.write_str("the agent has no model"),
            RunError::Cancelled => f.write_str("the run was cancelled"),
            RunError::Model(e) => write!(f, "the model request failed: {e}"),
            RunError::Host(e) => write!(f, "{e}"),
            RunError::ToolRetriesExceeded { tool, retries } => {
                write!(f, "tool `{tool}` exceeded max retries count of {retries}")
            }
            RunError::OutputRetriesExceeded { retries } => {
                write!(f, "exceeded maximum retries ({retries}) for output validation")
            }
            RunError::SessionClosed => f.write_str("the session closed before the run settled"),
        }
    }
}

impl std::error::Error for RunError {}

/// The incremental view of an invocation (agents.md "Streaming runs"). The coordinator adds the
/// envelope (sequence, agent name, invocation and frame ids).
#[crate::host(record(module = "agent_run", emit))]
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    TextDelta {
        message_id: String,
        delta: String,
    },
    TextEnd {
        message_id: String,
        interrupted: bool,
    },
    /// A call whose arguments validated, with those arguments.
    ToolCall {
        call_id: String,
        tool_name: String,
        arguments: Value,
    },
    ToolResult {
        call_id: String,
        tool_name: String,
        status: ToolResultStatus,
        output: String,
    },
    /// `ctx.update(text)`.
    Progress {
        call_id: String,
        tool_name: String,
        message: String,
    },
}

#[crate::host(record(module = "agent_run", emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolResultStatus {
    Completed,
    Failed,
    Cancelled,
}

/// What one invocation is asked: the message, the history it continues, and the host context
/// captured at the `run()` entry (a `(loop, contextvars.Context)` pair) its callbacks run in.
#[crate::host(record(module = "agent_run", extract))]
#[derive(Debug, Clone)]
pub struct RunRequest {
    /// Prefixes every id the invocation mints.
    pub invocation_id: String,
    /// The transport task id, when this invocation belongs to an A2A task.
    #[host(default = None)]
    pub execution_id: Option<String>,
    /// A user message.
    pub message: ChatItem,
    pub history: Vec<ChatItem>,
    /// The previous invocation asked with `InputRequired(tool=)`: this message is that tool's.
    pub answer_tool: Option<String>,
    /// Another agent asked this run: questions pass up as written.
    pub delegated: bool,
    pub conn_options: ApiConnectOptions,
    pub host_context: HostRef,
    /// Canonical host tool objects offered by this invocation.
    #[host(default = None)]
    pub tools_host: Option<HostRef>,
    /// The caller's current span: the run's `agent_run` nests under it.
    #[host(default = None)]
    pub parent: Option<crate::telemetry::record::SpanContext>,
}

/// A settled invocation, with the value the host validated when the output is one. The host
/// reads it as its [`RunFinished`].
#[crate::host(codec(emit, host = "RunFinished", via = crate::agent::run::RunFinished))]
#[derive(Debug, Clone, PartialEq)]
pub struct RunSettled {
    pub result: RunResult,
    pub output: Option<HostRef>,
}

/// A settled invocation as the host reads it: the caller's `RunResult` branch, and whether the
/// managed turn yielded at a safe boundary.
#[crate::host(record(module = "agent_run", emit))]
#[derive(Debug, Clone)]
pub struct RunFinished {
    pub result: CallerRunResult,
    pub interrupted: bool,
}

impl From<RunSettled> for RunFinished {
    /// A validated output the runtime no longer holds reads as no output.
    fn from(RunSettled { result, output }: RunSettled) -> Self {
        let RunResult {
            outcome,
            interrupted,
            new_messages,
            all_messages,
            usage,
            verbatim,
            pending_call,
            input_requirements,
        } = result;
        let settled = SettledRun {
            new_messages,
            all_messages,
            usage,
            verbatim,
            pending_call,
            input_requirements,
        };
        let result = match (outcome, output) {
            (RunOutcome::Completed { output: Output::Text(text) }, _) => {
                CallerRunResult::Completed(RunCompleted { settled, output: RunOutput::Text(text) })
            }
            (RunOutcome::Completed { output: Output::Validated(_) }, Some(value)) => {
                let output = RunOutput::Validated(value);
                CallerRunResult::Completed(RunCompleted { settled, output })
            }
            (RunOutcome::InputRequired(RunQuestion { question, .. }), _) => {
                CallerRunResult::InputRequired(RunInputRequired { settled, question })
            }
            (RunOutcome::Completed { .. } | RunOutcome::NoOutput, _) => {
                CallerRunResult::NoOutput(RunNoOutput { settled })
            }
        };
        RunFinished { result, interrupted }
    }
}

/// What the consumer of a run receives, in order.
#[crate::host(record(module = "agent_run", emit))]
#[derive(Debug, Clone)]
pub enum RunUpdate {
    #[host(field = "event")]
    Stream(StreamEvent),
    /// Cancelled work missed its cleanup deadline; the runtime still owns it.
    #[host(field = "pending")]
    IncompleteCleanup(Vec<String>),
    /// The last update of a run that settled with a result.
    #[host(field = "settled")]
    Finished(Box<RunSettled>),
    /// The last update of a run that settled without one.
    Failed {
        error: RunError,
        #[host(default_factory = "list")]
        input_requirements: Vec<RunInputRequirement>,
        /// The original model failure, retained beside its correlated outcome metadata.
        #[host(default = None)]
        provider_error: Option<crate::errors::ProviderError>,
    },
}

/// Runs invocations of one agent definition: each `run` streams its updates, then its settled
/// result; closing the stream cancels the invocation, whose work still settles.
#[crate::host(class(name = "AgentRunner"))]
pub trait AgentRunner: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(class(name = "_core_RunStream"))]
    type Stream: Send + Sync + 'static;
    #[host(class(name = "_core_RunResources"))]
    type Resources: Send + Sync + 'static;

    #[host(source(item = RunUpdate, class = "AgentRunUpdates"))]
    type Updates: StreamOutput<RunUpdate> + 'static;

    /// The runner of `definition` on `llm`, its callbacks run by `host`.
    #[host(new)]
    fn new(
        env: &Self::Env,
        definition: AgentDefinition,
        llm: Arc<dyn Llm>,
        host: Arc<dyn AgentCallbacks>,
    ) -> Result<Self, Invalid>;

    /// Starts one invocation; nothing runs until the first pull.
    #[host(call)]
    fn run(&self, request: RunRequest) -> Result<Self::Updates, Invalid>;

    #[host(call)]
    fn run_stream(
        &self,
        request: RunRequest,
        stream: Arc<Self::Stream>,
        resources: Arc<Self::Resources>,
    ) -> Result<(), Invalid>;

    /// The latest usage snapshot of invocation `invocation_id`; never advances it.
    #[host(call)]
    fn usage(&self, invocation_id: &str) -> Result<crate::usage::Usage, Invalid>;

    /// `ctx.update(message)` from the tool body running as host `operation` of invocation
    /// `invocation_id`: the run streams it as progress. Never waits; an update racing its call's
    /// end is dropped. Refused for an invocation this runner is not running.
    #[host(call)]
    fn progress(&self, invocation_id: &str, operation: u64, message: String)
    -> Result<(), Invalid>;
}
