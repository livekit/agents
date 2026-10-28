//! `voice/tool_executor.py`'s option `TypedDict`s and the marshal's wire dicts. The templates are
//! rendered host-side, so no core code reads them.

use lk_agents_host_derive::Host;

/// System-message templates injected around async tool dispatch.
///
/// Each field is either a ``str.format()`` template or a callable returning a string,
/// with the args typed as ``UpdatePromptArgs`` / ``DuplicatePromptArgs`` /
/// ``ReplyPromptArgs``. Unmentioned keys keep their defaults.
//
// `AsyncToolOptions`, `total=False`. Only the `str` branch of `str | Callable[..]` is carried: no
// core code reads a template. Each field spells `ty`, since `Option` here means absent, not
// nullable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[python(
    typed_dict,
    total = "False",
    module = "tool_executor",
    path = "livekit.agents.voice.tool_executor",
    name = "AsyncToolOptions"
)]
pub struct AsyncToolOptions {
    /// Wraps a user-provided ``ctx.update(message)`` string before it lands in chat_ctx.
    #[python(
        ty = "str | Callable[[UpdatePromptArgs], str]",
        import = "collections.abc:Callable",
        type_import = "..voice.tool_executor:UpdatePromptArgs"
    )]
    pub update_template: Option<String>,
    /// Sent to the LLM when ``on_duplicate='reject'`` blocks a duplicate call.
    #[python(
        ty = "str | Callable[[DuplicatePromptArgs], str]",
        import = "collections.abc:Callable",
        type_import = "..voice.tool_executor:DuplicatePromptArgs"
    )]
    pub duplicate_reject_template: Option<String>,
    /// Sent to the LLM when ``on_duplicate='confirm'`` requires re-call with confirmation.
    #[python(
        ty = "str | Callable[[DuplicatePromptArgs], str]",
        import = "collections.abc:Callable",
        type_import = "..voice.tool_executor:DuplicatePromptArgs"
    )]
    pub duplicate_confirm_template: Option<String>,
    /// Instruction for the deferred reply when the pending update is still the tail of chat_ctx.
    #[python(
        ty = "str | Callable[[ReplyPromptArgs], str]",
        import = "collections.abc:Callable",
        type_import = "..voice.tool_executor:ReplyPromptArgs"
    )]
    pub reply_at_tail_template: Option<String>,
    /// Instruction for the deferred reply when newer items came after the pending update.
    #[python(
        ty = "str | Callable[[ReplyPromptArgs], str]",
        import = "collections.abc:Callable",
        type_import = "..voice.tool_executor:ReplyPromptArgs"
    )]
    pub reply_maybe_covered_template: Option<String>,
}

/// Configuration for the tool handling system.
///
/// Can be passed as a plain dict::
///
#[cfg_attr(doctest, doc = "```text")]
///     AgentSession(
///         tool_handling={
///             "async_options": {"update_template": "..."},
///         },
///     )
#[cfg_attr(doctest, doc = "```")]
///
/// Set on ``AgentSession``, ``Agent``, or ``AsyncToolset`` (most specific wins).
//
// Also `total=False`. The nested annotation names the Python class.
// The `cfg_attr(doctest)` lines fence the Python sample for `cargo test --doc` only; the codegen
// reads `///` alone, so the generated docstring is unchanged.
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[python(
    typed_dict,
    total = "False",
    module = "tool_executor",
    path = "livekit.agents.voice.tool_executor",
    name = "ToolHandlingOptions"
)]
pub struct ToolHandlingOptions {
    /// Templates injected around async tool dispatch (``ctx.update()``, duplicate
    /// handling, coalesced replies). Unmentioned keys keep their defaults.
    #[python(ty = "AsyncToolOptions")]
    pub async_options: Option<AsyncToolOptions>,
}

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
/// only classifies; [`ToolOutcome::shape`] and the voice core decide what it means.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolOutcome {
    /// A value the model reads: `text` is a string as written, anything else as its JSON.
    /// `ToolResult(..)` sets `reply_required` and `verbatim`; `more` is what a delegate's
    /// verbatim result carries for the model alone.
    Value {
        text: String,
        reply_required: bool,
        verbatim: bool,
        more: Option<String>,
    },
    /// `None`: nothing to say, no reply asked.
    Nothing,
    /// `ToolError(message)`: the author's words, for the model.
    ToolError(String),
    /// `StopResponse`: answered with nothing, no reply.
    StopResponse,
    /// Any other exception; the host logged it, the model is told nothing about it.
    Internal,
    /// A value the model can't be told about (`_is_valid_function_output`).
    Invalid,
    /// The host cancelled the body.
    Cancelled,
    /// `ctx.update(message)`: `message` rendered, `raw` as written; the tool runs on.
    Update {
        message: String,
        raw: String,
    },
    /// `Completed(value)`: the value as text (`completed` for `None`).
    Completed {
        text: String,
        verbatim: bool,
    },
    Failed(String),
    Rejected(String),
    /// `InputRequired(question)`: the turn waits on the person; `tool` takes the answer.
    InputRequired {
        question: String,
        verbatim: bool,
        tool: Option<String>,
    },
    /// `Working(message=)`: waits on a system, nothing delivered.
    Working(Option<String>),
}

/// The output a [`ToolOutcome`] records, and whether it asks the model for a reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolShape {
    pub text: String,
    pub is_error: bool,
    pub reply_required: bool,
}

impl ToolOutcome {
    /// `make_tool_output` and the activity's transitions as one table: the call output and
    /// whether it asks for a reply. A verbatim result's delivery is the voice core's.
    pub fn shape(&self) -> ToolShape {
        let shape = |text: &str, is_error: bool, reply_required: bool| ToolShape {
            text: text.to_owned(),
            is_error,
            reply_required,
        };
        match self {
            ToolOutcome::Value { text, reply_required, .. } => shape(text, false, *reply_required),
            ToolOutcome::Nothing | ToolOutcome::StopResponse | ToolOutcome::Cancelled => {
                shape("", false, false)
            }
            ToolOutcome::ToolError(message) => shape(message, true, true),
            ToolOutcome::Internal => shape(INTERNAL_ERROR_OUTPUT, true, true),
            ToolOutcome::Invalid => shape(INVALID_OUTPUT, true, true),
            ToolOutcome::Update { message, .. } => shape(message, false, true),
            // The reference's `_apply_transition`: these ask for a reply.
            ToolOutcome::Completed { text, .. } => shape(text, false, true),
            ToolOutcome::Failed(text) | ToolOutcome::Rejected(text) => shape(text, true, true),
            // The question is asked, not replied to; the person's answer replaces it.
            ToolOutcome::InputRequired { question, .. } => {
                shape(&format!("{WAITING_PREFIX}{question}"), false, false)
            }
            ToolOutcome::Working(message) => {
                shape(message.as_deref().unwrap_or("working"), false, false)
            }
        }
    }

    /// The text the person gets exactly as written, when the outcome is verbatim.
    pub fn verbatim(&self) -> Option<&str> {
        match self {
            ToolOutcome::Value { text, verbatim: true, .. }
            | ToolOutcome::Completed { text, verbatim: true } => Some(text),
            _ => None,
        }
    }

    /// What the model reads once the verbatim text was said (`delivered`), then the delegate's
    /// note for the model.
    pub fn delivered(&self) -> Option<String> {
        let text = self.verbatim()?;
        Some(match self {
            ToolOutcome::Value { more: Some(more), .. } => format!("{DELIVERED}{text}\n{more}"),
            _ => format!("{DELIVERED}{text}"),
        })
    }
}

/// How a call of a tool that requires approval went (`approval_scope`): it asked for approval
/// with its read-back, or ran on the person's answer. Marked on the call for the next to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approval {
    Asked,
    Granted,
}

impl Approval {
    pub fn as_str(self) -> &'static str {
        match self {
            Approval::Asked => "asked",
            Approval::Granted => "granted",
        }
    }
}

/// The host's report of one run: its outcome, and how its approval went.
#[derive(Debug, Clone)]
pub struct ToolReport {
    pub outcome: ToolOutcome,
    pub approval: Option<Approval>,
    /// The `InputRequired` itself, the output of a run that ends on this call's speech; for a
    /// `Completed` from a VoiceTask's tool, the value the host validated, which ends the task.
    pub transition: Option<crate::HostRef>,
}

impl From<ToolOutcome> for ToolReport {
    fn from(outcome: ToolOutcome) -> Self {
        ToolReport { outcome, approval: None, transition: None }
    }
}

#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for ToolOutcome {
    type Error = pyo3::PyErr;

    /// The host's `{"kind": .., ..}` dict. An unknown kind, or one missing its key, is a
    /// `ValueError`, never a guess.
    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::{item, item_or};
        use pyo3::types::PyAnyMethods as _;
        const OWNER: &str = "ToolOutcome";
        let obj = &*obj;
        let kind: String = item(obj, OWNER, "kind")?;
        let text = || item::<String>(obj, OWNER, "text");
        let flag = |key: &str, default: bool| {
            item_or(obj, OWNER, key, |v| v.extract::<bool>(), || default)
        };
        let opt = |key: &str| item_or(obj, OWNER, key, |v| v.extract::<Option<String>>(), || None);
        Ok(match kind.as_str() {
            "value" => ToolOutcome::Value {
                text: text()?,
                reply_required: flag("reply_required", true)?,
                verbatim: flag("verbatim", false)?,
                more: opt("more")?,
            },
            "nothing" => ToolOutcome::Nothing,
            "tool_error" => ToolOutcome::ToolError(text()?),
            "stop_response" => ToolOutcome::StopResponse,
            "internal" => ToolOutcome::Internal,
            "invalid" => ToolOutcome::Invalid,
            "cancelled" => ToolOutcome::Cancelled,
            "update" => ToolOutcome::Update { message: text()?, raw: item(obj, OWNER, "raw")? },
            "completed" => {
                ToolOutcome::Completed { text: text()?, verbatim: flag("verbatim", false)? }
            }
            "failed" => ToolOutcome::Failed(text()?),
            "rejected" => ToolOutcome::Rejected(text()?),
            "input_required" => ToolOutcome::InputRequired {
                question: text()?,
                verbatim: flag("verbatim", false)?,
                tool: opt("tool")?,
            },
            "working" => ToolOutcome::Working(opt("text")?),
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown tool outcome kind {other:?}"
                )));
            }
        })
    }
}

#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for ToolReport {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::item_or;
        use pyo3::types::PyAnyMethods as _;
        const OWNER: &str = "ToolOutcome";
        let obj = &*obj;
        let transition = |v: &pyo3::Bound<'py, pyo3::PyAny>| {
            if v.is_none() { Ok(None) } else { v.extract::<crate::HostRef>().map(Some) }
        };
        let approval = |v: &pyo3::Bound<'py, pyo3::PyAny>| match v.extract::<Option<String>>()? {
            None => Ok(None),
            Some(s) if s == "asked" => Ok(Some(Approval::Asked)),
            Some(s) if s == "granted" => Ok(Some(Approval::Granted)),
            Some(other) => {
                Err(pyo3::exceptions::PyValueError::new_err(format!("unknown approval {other:?}")))
            }
        };
        Ok(ToolReport {
            outcome: obj.extract()?,
            approval: item_or(obj, OWNER, "approval", approval, || None)?,
            transition: item_or(obj, OWNER, "transition", transition, || None)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a transition read as silent (its output lost), `Failed`/`Rejected` not errors, an
    // `InputRequired` with no question, `ToolResult(reply_required=False)` losing its text, or
    // exception detail reaching the model.
    #[test]
    fn each_outcome_records_its_output_and_asks_for_a_reply_as_the_reference_does() {
        let value = |text: &str, reply_required| ToolOutcome::Value {
            text: text.into(),
            reply_required,
            verbatim: false,
            more: None,
        };
        let ask = ToolOutcome::InputRequired {
            question: "Which day?".into(),
            verbatim: false,
            tool: None,
        };
        let cases: &[(ToolOutcome, (&str, bool, bool))] = &[
            (value("12C", true), ("12C", false, true)),
            (value("sent DTMF 1", false), ("sent DTMF 1", false, false)),
            (ToolOutcome::Nothing, ("", false, false)),
            (ToolOutcome::StopResponse, ("", false, false)),
            (ToolOutcome::ToolError("nope".into()), ("nope", true, true)),
            (ToolOutcome::Internal, (INTERNAL_ERROR_OUTPUT, true, true)),
            (ToolOutcome::Invalid, (INVALID_OUTPUT, true, true)),
            (
                ToolOutcome::Completed { text: "booked".into(), verbatim: false },
                ("booked", false, true),
            ),
            (ToolOutcome::Failed("no slot".into()), ("no slot", true, true)),
            (ToolOutcome::Rejected("not mine".into()), ("not mine", true, true)),
            (ask, ("waiting for the person's answer to: Which day?", false, false)),
            (ToolOutcome::Working(None), ("working", false, false)),
        ];
        for (outcome, (text, is_error, reply)) in cases {
            let shape = outcome.shape();
            let got = (shape.text.as_str(), shape.is_error, shape.reply_required);
            assert_eq!(got, (*text, *is_error, *reply), "{outcome:?}");
        }
    }

    // Catches: a verbatim result's model text not saying it was delivered, or the delegate's note
    // dropped.
    #[test]
    fn a_verbatim_result_reads_as_delivered() {
        let v = ToolOutcome::Value {
            text: "Your code is 1234.".into(),
            reply_required: true,
            verbatim: true,
            more: Some("ask if they need more".into()),
        };
        assert_eq!(v.verbatim(), Some("Your code is 1234."));
        let delivered =
            "delivered to the person as written: Your code is 1234.\nask if they need more";
        assert_eq!(v.delivered().as_deref(), Some(delivered));
        assert_eq!(ToolOutcome::Nothing.delivered(), None);
    }
}
