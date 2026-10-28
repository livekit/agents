//! What an invocation produces: its settled [`RunResult`], the [`StreamEvent`]s it emits as work
//! arrives, and the correlation ids its effects and events carry.

use lk_agents_abi::ChatItem;
use serde_json::Value;

macro_rules! ids {
    ($($(#[$doc:meta])* $name:ident),* $(,)?) => {$(
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u64);
    )*};
}

ids! {
    /// One model request.
    RequestId,
    /// One tool call the model made (or an answer handed to a tool), from preparation to settlement.
    CallId,
    /// One rendering of the dynamic instructions.
    RenderId,
    /// One output candidate's host validation.
    ValidationId,
}

/// The settled invocation (`RunResult[Output]`): its status, then the history and usage both
/// branches expose.
#[derive(Debug, Clone, PartialEq)]
pub struct RunResult {
    pub outcome: RunOutcome,
    /// The incoming message and everything this invocation recorded, in order.
    pub new_messages: Vec<ChatItem>,
    /// The supplied history plus this invocation, without instructions.
    pub all_messages: Vec<ChatItem>,
    /// This invocation's usage alone; the caller adds it to an accumulator it supplied.
    pub usage: RunUsage,
    /// The part of the answer or question that reached the person exactly as written.
    pub verbatim: Option<String>,
}

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
}

/// The accepted output of a completed invocation.
#[derive(Debug, Clone, PartialEq)]
pub enum Output {
    /// A `str` output.
    Text(String),
    /// The value the host validated for this candidate; the runtime retains it.
    Validated(ValidationId),
    /// `StopResponse`, or a turn ended by a tool that asks for no reply.
    Nothing,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunUsage {
    pub requests: u32,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub tool_calls: u32,
}

/// Why an invocation settled without a result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// The agent has no model to run a step with.
    NoModel,
    Cancelled,
    /// The model request failed after the provider's own recovery.
    Model(String),
    /// A host callback the invocation depends on (instructions, output validation) failed.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolResultStatus {
    Completed,
    Failed,
    Cancelled,
}
