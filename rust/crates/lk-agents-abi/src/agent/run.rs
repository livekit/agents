//! One agent invocation as the host runs it (agents.md "Agent", "Streaming runs"): the request,
//! the updates it streams, its settled result, and [`AgentRunner`], the class that runs an
//! [`AgentDefinition`] against its model and [`AgentCallbacks`]. `lk-agents-core`'s `AgentRun`
//! decides; the runtime implements the runner.

use std::sync::Arc;

use serde_json::Value;

use super::{AgentCallbacks, AgentDefinition};
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

/// The settled invocation (`RunResult[Output]`): its status, then the history and usage both
/// branches expose.
#[crate::host(record(module = "agent_run", emit))]
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
    #[host(default_factory = "list")]
    pub input_requirements: Vec<RunInputRequirement>,
}

#[crate::host(record(module = "agent_run", emit))]
#[derive(Debug, Clone, PartialEq)]
pub enum RunOutcome {
    Completed {
        output: Output,
    },
    /// `answer_tool` takes the person's next message as its first string parameter
    /// (`InputRequired(tool=)`); the next invocation is told so.
    InputRequired {
        question: String,
        verbatim: bool,
        answer_tool: Option<String>,
    },
    /// The turn ended with no output: `StopResponse`, a tool result asking for no reply, an
    /// all-silent step or an interruption. Never `Completed`, whose output is the declared type.
    NoOutput,
}

/// The accepted output of a completed invocation.
#[crate::host(record(module = "agent_run", emit))]
#[derive(Debug, Clone, PartialEq)]
pub enum Output {
    /// A `str` output.
    #[host(field = "text")]
    Text(String),
    /// The value the host validated for this candidate; the runtime retains it and the update
    /// carries it.
    #[host(skip)]
    Validated(ValidationId),
}

/// Why an invocation settled without a result.
#[crate::host(record(module = "agent_run", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// The agent has no model to run a step with.
    NoModel,
    Cancelled,
    /// The model request failed after the provider's own recovery.
    #[host(field = "message")]
    Model(String),
    /// A host callback the invocation depends on (instructions, output validation) failed.
    #[host(field = "message")]
    Host(String),
    /// The model kept calling `tool` wrongly past its retry budget.
    ToolRetriesExceeded {
        tool: String,
        retries: u32,
    },
    /// The model kept producing rejected final output past its retry budget.
    OutputRetriesExceeded {
        retries: u32,
    },
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
}

/// A settled invocation, with the value the host validated when the output is one.
#[crate::host(record(module = "agent_run", emit))]
#[derive(Debug, Clone)]
pub struct RunSettled {
    pub result: RunResult,
    pub output: Option<HostRef>,
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
    },
}

/// The definition or the request cannot run: what is wrong with it.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct InvalidRun(pub String);

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
    ) -> Result<Self, InvalidRun>;

    /// Starts one invocation; nothing runs until the first pull.
    #[host(call)]
    fn run(&self, request: RunRequest) -> Result<Self::Updates, InvalidRun>;

    #[host(call)]
    fn run_stream(
        &self,
        request: RunRequest,
        stream: Arc<Self::Stream>,
        resources: Arc<Self::Resources>,
    ) -> Result<(), InvalidRun>;

    /// The latest usage snapshot of invocation `invocation_id`; never advances it.
    #[host(call)]
    fn usage(&self, invocation_id: &str) -> Result<crate::usage::Usage, InvalidRun>;

    /// `ctx.update(message)` from the tool body running as host `operation` of invocation
    /// `invocation_id`: the run streams it as progress. Never waits; an update racing its call's
    /// end is dropped. Refused for an invocation this runner is not running.
    #[host(call)]
    fn progress(
        &self,
        invocation_id: &str,
        operation: u64,
        message: String,
    ) -> Result<(), InvalidRun>;
}
