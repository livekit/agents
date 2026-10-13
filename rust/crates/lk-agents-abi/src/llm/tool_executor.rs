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

// --- the marshal's report ---------------------------------------------------------------------

// The `kind` word the marshal writes; `from_wire` is `None` for one this build does not know.
#[crate::host(wire_strs(msg = format!("unknown tool answer kind {s:?}")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolAnswerKind {
    Returned,
    Raised,
    /// Something `_is_valid_function_output` rejects.
    Invalid,
    Cancelled,
    /// An async tool's first `ctx.update()`.
    Detached,
}

/// As the marshal's `isinstance` chain classified it.
#[crate::host(wire_strs(msg = format!("unknown tool raise kind {s:?}")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolRaiseKind {
    /// The author's message is for the model.
    ToolError,
    StopResponse,
    Internal,
}

/// `kind` selects the shape. A malformed dict is an error, which `HostTools` turns into the
/// reference's internal-error output.
#[derive(Debug, Clone)]
pub struct RawToolAnswer {
    pub kind: String,
    pub value: Option<serde_json::Value>,
    pub agents: Vec<lk_agents_abi::HostRef>,
    pub error: Option<String>,
    pub message: Option<String>,
    /// Before the template.
    pub raw: Option<String>,
}

#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for RawToolAnswer {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use crate::python::{intern_host, item, item_or, value_from_py};
        use pyo3::types::PyAnyMethods as _;
        const OWNER: &str = "ToolAnswerFromPy";
        let obj = &*obj;
        let extract = |v: &pyo3::Bound<'py, pyo3::PyAny>| v.extract::<Option<String>>();
        Ok(RawToolAnswer {
            kind: item(obj, OWNER, "kind")?,
            value: item_or(obj, OWNER, "value", |v| if v.is_none() { Ok(None) } else { value_from_py(v, "value").map(Some) }, || None)?,
            agents: item_or(
                obj,
                OWNER,
                "agents",
                |v| {
                    if v.is_none() {
                        return Ok(Vec::new());
                    }
                    v.try_iter()?.map(|a| Ok(intern_host(a?.unbind()))).collect()
                },
                Vec::new,
            )?,
            error: item_or(obj, OWNER, "error", extract, || None)?,
            message: item_or(obj, OWNER, "message", extract, || None)?,
            raw: item_or(obj, OWNER, "raw", extract, || None)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ToolAnswerKind, ToolRaiseKind};

    // Catches: a misspelled marshal word, or an unknown one accepted.
    #[test]
    fn the_marshal_words_parse_and_unknown_ones_are_refused() {
        // The marshal's own spellings, which `voice/tool_executor.py` writes and its tests read.
        assert_eq!(ToolAnswerKind::from_wire("detached"), Some(ToolAnswerKind::Detached));
        assert_eq!(ToolRaiseKind::from_wire("stop_response"), Some(ToolRaiseKind::StopResponse));
        assert_eq!(ToolAnswerKind::from_wire("done"), None);
        assert_eq!(ToolRaiseKind::from_wire("stop"), None, "`stop` is `ToolOutcome`'s word, not this wire's");
    }
}
