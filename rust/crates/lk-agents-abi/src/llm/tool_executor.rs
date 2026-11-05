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
#[crate::host(record(module = "tool_report", extract, emit))]
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

// Public data contracts. Python behavior remains in its adapters.

/// Args for the ``update`` template.
#[crate::host(record(
    module = "tool_templates",
    path = "livekit.agents.llm.tool_context",
    typed_dict
))]
pub struct UpdatePromptArgs {
    #[host(annotation = "str")]
    pub function_name: String,
    #[host(annotation = "str")]
    pub call_id: String,
    #[host(annotation = "str")]
    pub message: String,
}

/// Args for the ``duplicate_reject`` / ``duplicate_confirm`` templates.
#[crate::host(record(
    module = "tool_templates",
    path = "livekit.agents.llm.tool_context",
    typed_dict
))]
pub struct DuplicatePromptArgs {
    #[host(annotation = "str")]
    pub function_name: String,
    #[host(annotation = "list[str]")]
    pub fnc_calls_json: Vec<String>,
    #[host(annotation = "str")]
    pub fnc_calls_text: String,
}

/// Args for the ``reply_at_tail`` / ``reply_maybe_covered`` templates.
#[crate::host(record(
    module = "tool_templates",
    path = "livekit.agents.llm.tool_context",
    typed_dict
))]
pub struct ReplyPromptArgs {
    #[host(annotation = "list[str]")]
    pub call_ids: Vec<String>,
}

/// System-message templates injected around async tool dispatch.
///
///     Each field is either a ``str.format()`` template or a callable returning a string,
///     with the args typed as ``UpdatePromptArgs`` / ``DuplicatePromptArgs`` /
///     ``ReplyPromptArgs``. Unmentioned keys keep their defaults.
///
#[crate::host(record(
    module = "tool_templates",
    path = "livekit.agents.llm.tool_context",
    typed_dict
))]
pub struct AsyncToolOptions {
    /// Wraps a user-provided ``ctx.update(message)`` string before it lands in chat_ctx.
    #[host(annotation = "str | collections.abc.Callable[[UpdatePromptArgs], str]")]
    pub update_template: crate::Maybe<crate::HostRef>,
    /// Sent to the LLM when ``on_duplicate='reject'`` blocks a duplicate call.
    #[host(annotation = "str | collections.abc.Callable[[DuplicatePromptArgs], str]")]
    pub duplicate_reject_template: crate::Maybe<crate::HostRef>,
    /// Sent to the LLM when ``on_duplicate='confirm'`` requires re-call with confirmation.
    #[host(annotation = "str | collections.abc.Callable[[DuplicatePromptArgs], str]")]
    pub duplicate_confirm_template: crate::Maybe<crate::HostRef>,
    /// Instruction for the deferred reply when the pending update is still the tail of chat_ctx.
    #[host(annotation = "str | collections.abc.Callable[[ReplyPromptArgs], str]")]
    pub reply_at_tail_template: crate::Maybe<crate::HostRef>,
    /// Instruction for the deferred reply when newer items came after the pending update.
    #[host(annotation = "str | collections.abc.Callable[[ReplyPromptArgs], str]")]
    pub reply_maybe_covered_template: crate::Maybe<crate::HostRef>,
}

/// Configuration for the tool handling system.
///
///     Can be passed as a plain dict::
///
///         Agent(
///             instructions="...",
///             tool_handling={
///                 "async_options": {"update_template": "..."},
///             },
///         )
///
///     Set on ``Agent``, or on an ``AsyncToolset`` for its own tools (most specific wins).
///
#[crate::host(record(
    module = "tool_templates",
    path = "livekit.agents.llm.tool_context",
    typed_dict
))]
pub struct ToolHandlingOptions {
    /// Templates injected around async tool dispatch (``ctx.update()``, duplicate
    ///     handling, coalesced replies). Unmentioned keys keep their defaults.
    ///
    #[host(annotation = "AsyncToolOptions")]
    pub async_options: crate::Maybe<AsyncToolOptions>,
}
