use serde_json::Value;

// `ToolFlag` stays a Python `Flag` (callers write `A | B`); Rust holds the bits.

/// The values are the reference's `auto()` order and the wire to the shim: never reorder.
#[crate::host(codec(
    extract,
    emit,
    host = "livekit.agents.llm.tool_context.ToolFlag",
    via = u8, attribute = "value", flag(module = "tool_flags")
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ToolFlag(u8);

impl ToolFlag {
    pub const NONE: ToolFlag = ToolFlag(0);
    pub const IGNORE_ON_ENTER: ToolFlag = ToolFlag(1);

    /// `None` for a bit this version does not know, so a newer host's flag is not reinterpreted.
    pub const fn from_bits(bits: u8) -> Option<ToolFlag> {
        if bits & !0b1 == 0 { Some(ToolFlag(bits)) } else { None }
    }
}

impl From<ToolFlag> for u8 {
    fn from(f: ToolFlag) -> u8 {
        f.0
    }
}

/// A bit this version does not know is refused, so a newer host's flag is not reinterpreted.
impl TryFrom<u8> for ToolFlag {
    type Error = crate::errors::Invalid;
    fn try_from(bits: u8) -> Result<ToolFlag, crate::errors::Invalid> {
        ToolFlag::from_bits(bits)
            .ok_or_else(|| crate::errors::Invalid(format!("unknown ToolFlag bits {bits}")))
    }
}

impl std::ops::BitOr for ToolFlag {
    type Output = ToolFlag;
    fn bitor(self, rhs: ToolFlag) -> ToolFlag {
        ToolFlag(self.0 | rhs.0)
    }
}

// No `///`: it would become a docstring on the host's `DuplicateMode = Literal[...]`, which has
// none. Variant order is the `Literal`'s member order, a public surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[crate::host(record(
    module = "tool_context",
    path = "livekit.agents.llm.tool_context",
    emit,
    extract
))]
#[crate::host(wire_strs(msg = format!("unknown on_duplicate mode {s:?}")))]
pub enum DuplicateMode {
    #[default]
    #[host(rename = "allow")]
    Allow,
    #[host(rename = "reject")]
    Reject,
    #[host(rename = "replace")]
    Replace,
    #[host(rename = "confirm")]
    Confirm,
}

/// What counts as a duplicate of an in-flight call, i.e. what ``on_duplicate`` acts on.
///
/// ``"name"``           any in-flight call of the same tool (default).
/// ``"name_and_args"``  same tool *and* same arguments, so concurrent calls of one
///                      tool with different arguments are not duplicates::
///
#[cfg_attr(doctest, doc = "```text")]
///                          @function_tool(on_duplicate="reject", duplicate_scope="name_and_args")
///                          async def check_order(ctx: RunContext, order_id: str) -> str: ...
///
///                      Arguments are compared *after* validation, so a parameter the
///                      LLM omitted on one call and passed explicitly on the next still
///                      reads as the same call, and ``1`` matches ``1.0`` for a ``float``
///                      parameter. Raw function tools (including MCP tools) have no
///                      per-parameter schema, so their arguments are compared as sent.
///
///                      Comparison fails open — an unrepresentable argument or a
///                      validation error leaves the call treated as sent rather than
///                      blocking it — so don't rely on this as an exactly-once
///                      guarantee; put that in the tool body.
#[cfg_attr(doctest, doc = "```")]
//
// The reference's docstring, verbatim. The `cfg_attr(doctest)` fences exist only for
// `cargo test --doc`, which the codegen never reads.
#[allow(
    rustdoc::invalid_rust_codeblocks,
    reason = "the RST `::` block is Python; fencing it would put backticks into the docstring"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[crate::host(record(
    module = "tool_context",
    path = "livekit.agents.llm.tool_context",
    emit,
    extract
))]
#[crate::host(wire_strs(msg = format!("unknown duplicate_scope {s:?}")))]
pub enum DuplicateScope {
    /// Any in-flight call of the same tool.
    #[default]
    #[host(rename = "name")]
    Name,
    /// Same tool *and* same arguments, so concurrent calls with different arguments coexist.
    #[host(rename = "name_and_args")]
    NameAndArgs,
}

#[allow(rustdoc::bare_urls, reason = "the docstring is Python's and writes the URL bare")]
/// Represents the raw function schema format used in LLM function calling APIs.
///
/// This structure directly maps to OpenAI's function definition format as documented at:
/// https://platform.openai.com/docs/guides/function-calling?api-mode=responses
///
/// It is also compatible with other LLM providers that support raw JSON Schema-based
/// function definitions.
#[crate::host(record(
    module = "tool_context",
    path = "livekit.agents.llm.tool_context",
    typed_dict,
    emit
))]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawFunctionDescription {
    pub name: String,
    #[host(annotation = "typing.NotRequired[str | None]")]
    pub description: Option<String>,
    #[host(annotation = "dict[str, object]")]
    pub parameters: Value,
}

pub type DuplicateKey = (String, Option<String>);

#[crate::host(file(module = "tool_context"))]
const _: () = ();

/// The schema parameter `@function_tool(on_duplicate="confirm")` adds, which the model sets to
/// re-assert a call.
#[crate::host(constant)]
pub const CONFIRM_DUPLICATE_PARAM: &str = "lk_agents_confirm_duplicate";

/// The interim return of a tool's `RunContext.update(message)` when no `update_template`
/// overrides it.
#[crate::host(constant)]
pub const UPDATE_TEMPLATE: &str = "The tool `{function_name}` has updated, message: {message}\nThe task is still running, so DON'T make up or give information not included in the message above.";

/// The description the model reads for that parameter. A prompt, so rewording it changes
/// behaviour: it must say the user asked for the repeat.
#[crate::host(constant)]
pub const CONFIRM_DUPLICATE_DESCRIPTION: &str = "Set this to True to confirm you want to run a duplicate. Only do this when user confirms the duplication is needed.";

// Public data contracts. Python behavior remains in its adapters.

/// The task failed: ``error`` is its last message.
#[crate::host(record(
    module = "tool_outcomes",
    path = "livekit.agents.llm.tool_context",
    dataclass,
    frozen
))]
pub struct Failed {
    pub error: String,
}

/// The task will not run this input, from ``on_enter``.
#[crate::host(record(
    module = "tool_outcomes",
    path = "livekit.agents.llm.tool_context",
    dataclass,
    frozen
))]
pub struct Rejected {
    pub reason: String,
}

/// The task waits on a system: nothing is delivered, ``message`` is its status.
#[crate::host(record(
    module = "tool_outcomes",
    path = "livekit.agents.llm.tool_context",
    dataclass,
    frozen
))]
pub struct Working {
    #[host(default = None)]
    pub message: Option<String>,
}

// Public data contracts. Python behavior remains in its adapters.

/// A function tool's output, whether it requests a follow-up reply, and whether the person
///     gets it as written.
///
///     Args:
///         output: The value to return to the model, such as a string, dict, list,
///             or None. For a handoff, return ``(agent, ToolResult(...))``.
///         reply_required: Whether this result requests a reply. Defaults to True.
///             False keeps the output in chat history without requesting a reply.
///             Other tool results can still request a reply, and interruption or
///             ``cancel_tool_reply()`` can still prevent one.
///         verbatim: The output is text the person gets exactly as written: spoken with ``say()``
///             on a call, sent unchanged in text, relayed unchanged by a delegate. With
///             ``reply_required=False`` it ends the turn; otherwise the model continues after it.
///
///     Example:
///         return ToolResult("Successfully sent DTMF events: 1", reply_required=False)
///
#[crate::host(record(
    module = "tool_outcomes",
    path = "livekit.agents.llm.tool_context",
    dataclass
))]
pub struct ToolResult {
    #[host(annotation = "object")]
    pub output: crate::HostRef,
    #[host(default = true, kw_only)]
    pub reply_required: bool,
    #[host(default = false, kw_only)]
    pub verbatim: bool,
}

#[crate::host(exceptions(
    module = "tool_errors",
    path = "livekit.agents.llm.tool_context",
    classes(
        ToolError(
            base = "Exception",
            doc = "A tool failure whose message is visible to the model.",
            fields(message(ty = "str", positional, readonly)),
            base_args(message)
        ),
        ModelRetry(
            base = "Exception",
            doc = "Send the model back with a reason, within its retry budget.",
            fields(message(ty = "str", positional, readonly)),
            base_args(message)
        ),
        StopResponse(
            base = "Exception",
            doc = "End this tool call without generating a response.",
            fields(),
            base_args()
        )
    )
))]
const _: () = ();

// Public data contracts. Python behavior remains in its adapters.

/// The turn or task waits on the person for an answer to ``question``.
///
///     The agent facing the person asks it in its own words, or exactly as written with
///     ``verbatim=True``. From a conversation tool the next message is written as this call's
///     output. ``tool`` names the tool that takes the answer.
///
#[crate::host(record(
    module = "function_tools",
    path = "livekit.agents.llm.tool_context",
    dataclass,
    deferred,
    frozen
))]
pub struct InputRequired {
    pub question: String,
    #[host(default = None, annotation = "..llm.tool_context.FunctionTool | ..llm.tool_context.RawFunctionTool | None")]
    pub tool: Option<crate::HostRef>,
    #[host(default = false)]
    pub verbatim: bool,
}

#[crate::host(record(
    module = "function_tools",
    path = "livekit.agents.llm.tool_context",
    dataclass,
    deferred
))]
pub struct FunctionToolInfo {
    pub name: String,
    pub description: Option<String>,
    #[host(annotation = ".tool_flags.ToolFlag")]
    pub flags: ToolFlag,
    #[host(default = "allow")]
    pub on_duplicate: DuplicateMode,
    #[host(default = "name")]
    pub duplicate_scope: DuplicateScope,
    #[host(default = false)]
    pub needs_approval: bool,
    /// ``needs_approval=`` as a function: the read-back the person approves.
    #[host(default = None, annotation = "..llm.tool_context.ReadBack | None")]
    pub read_back: Option<crate::HostRef>,
    /// ``retries=``: this tool's retry budget in place of the agent's ``retries['tools']``.
    #[host(default = None)]
    pub retries: Option<i64>,
}

#[crate::host(record(
    module = "function_tools",
    path = "livekit.agents.llm.tool_context",
    dataclass,
    deferred
))]
pub struct RawFunctionToolInfo {
    pub name: String,
    #[host(annotation = "dict[str, typing.Any]")]
    pub raw_schema: std::collections::BTreeMap<String, serde_json::Value>,
    #[host(annotation = ".tool_flags.ToolFlag")]
    pub flags: ToolFlag,
    #[host(default = "allow")]
    pub on_duplicate: DuplicateMode,
    #[host(default = "name")]
    pub duplicate_scope: DuplicateScope,
    #[host(default = None)]
    pub retries: Option<i64>,
}
