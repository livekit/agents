//! A tool's outcome as the host classified it, and the texts the model reads around it.

// --- a tool's outcome: the one representation crossing from the host -------------------------

/// `make_function_call_output`'s text for an unexpected exception; says nothing about it.
pub const INTERNAL_ERROR_OUTPUT: &str = "An internal error occurred";

/// `make_function_call_output`'s text for a return value the model cannot be told about.
pub const INVALID_OUTPUT: &str = "the tool returned an invalid output";

/// How a call's output reads while it waits on the person; the answer replaces it.
pub const WAITING_PREFIX: &str = "waiting for the person's answer to: ";

/// What the model reads as the output of a verbatim result it did not write.
pub const DELIVERED: &str = "delivered to the person as written: ";

/// How the agent facing the person is told to ask a question that is not verbatim.
pub const ASK_IN_OWN_WORDS: &str =
    "Ask the person this, in your own words, as your whole reply, and add nothing else: ";

/// What a tool's run came to, as the host classified the value it returned or raised. The host
/// only classifies; the outcome table in `lk-agents-core` decides what it means.
#[crate::host(record(module = "tool_report", extract, emit))]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ToolOutcome {
    /// A value the model reads: `text` is a string as written, anything else as its JSON.
    /// `ToolResult(..)` sets `reply_required` and `verbatim`; `more` is what a delegate's
    /// verbatim result carries for the model alone.
    Value {
        text: String,
        #[host(default = true)]
        reply_required: bool,
        #[host(default = false)]
        verbatim: bool,
        #[host(default = None)]
        more: Option<String>,
    },
    /// `None`: nothing to say, no reply asked.
    Nothing,
    /// `ToolError(message)`: the author's words, for the model.
    #[host(field = "message")]
    ToolError(String),
    /// `ModelRetry(message)`: the reason, for the model to correct its call.
    #[host(field = "message")]
    ModelRetry(String),
    /// `StopResponse`: answered with nothing, no reply.
    StopResponse,
    /// Any other exception; the host logged it, the model is told nothing about it.
    Internal,
    /// A value the model can't be told about (`_is_valid_function_output`).
    Invalid,
    /// The host cancelled the body.
    Cancelled,
    /// `ctx.update(message)`: `message` rendered, `raw` as written; the tool runs on.
    Update { message: String, raw: String },
    /// `Completed(value)`: the value as text (`completed` for `None`).
    Completed {
        text: String,
        #[host(default = false)]
        verbatim: bool,
    },
    #[host(field = "error")]
    Failed(String),
    #[host(field = "reason")]
    Rejected(String),
    /// `InputRequired(question)`: the turn waits on the person; `tool` takes the answer.
    InputRequired {
        question: String,
        #[host(default = false)]
        verbatim: bool,
        #[host(default = None)]
        tool: Option<String>,
    },
    /// `Working(message=)`: waits on a system, nothing delivered.
    #[host(field = "message", default = None)]
    Working(Option<String>),
}

/// How a call of a tool that requires approval went: it asked for approval with its read-back,
/// or ran on the person's answer. Marked on the call for the next to read.
#[crate::host(record(module = "tool_report", extract, emit))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approval {
    Asked,
    Granted,
}

/// The host's report of one run: its outcome.
#[crate::host(record(module = "tool_report", extract))]
#[derive(Debug, Clone)]
pub struct ToolReport {
    pub outcome: ToolOutcome,
    /// For a `Completed` from a VoiceTask's tool, the value the host validated, which ends the
    /// task.
    #[host(default = None)]
    pub transition: Option<crate::HostRef>,
}

impl From<ToolOutcome> for ToolReport {
    fn from(outcome: ToolOutcome) -> Self {
        ToolReport { outcome, transition: None }
    }
}
