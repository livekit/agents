//! Tool metadata and duplicate-call vocabulary (`llm/tool_context.py`). `ToolFlag`
//! stays a Python `Flag` (callers write `A | B`); Rust holds the bits.

use lk_agents_host_derive::Host;
use serde_json::Value;

/// The values are the reference's `auto()` order and the wire to the shim: never reorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ToolFlag(u8);

impl ToolFlag {
    pub const NONE: ToolFlag = ToolFlag(0);
    pub const IGNORE_ON_ENTER: ToolFlag = ToolFlag(1);

    pub const fn bits(self) -> u8 {
        self.0
    }
    /// `None` for a bit this version does not know, so a newer host's flag is not reinterpreted.
    pub const fn from_bits(bits: u8) -> Option<ToolFlag> {
        if bits & !0b1 == 0 { Some(ToolFlag(bits)) } else { None }
    }
    pub const fn contains(self, other: ToolFlag) -> bool {
        self.0 & other.0 == other.0
    }
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for ToolFlag {
    type Target = pyo3::types::PyInt;
    type Output = pyo3::Bound<'py, pyo3::types::PyInt>;
    type Error = std::convert::Infallible;

    fn into_pyobject(
        self,
        py: pyo3::Python<'py>,
    ) -> Result<pyo3::Bound<'py, pyo3::types::PyInt>, std::convert::Infallible> {
        self.0.into_pyobject(py)
    }
}

/// Unknown bits raise: dropping one could make a cancellable tool uncancellable.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for ToolFlag {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        ToolFlag::try_from(obj.extract::<u8>()?).map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

impl From<ToolFlag> for u8 {
    fn from(f: ToolFlag) -> u8 {
        f.0
    }
}

/// A bit this version does not know is refused, so a newer host's flag is not reinterpreted.
impl TryFrom<u8> for ToolFlag {
    type Error = String;
    fn try_from(bits: u8) -> Result<ToolFlag, String> {
        ToolFlag::from_bits(bits).ok_or_else(|| format!("unknown ToolFlag bits {bits}"))
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Host)]
#[python(literal, module = "tool_context", name = "DuplicateMode")]
#[host(wire(msg = format!("unknown on_duplicate mode {s:?}")))]
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
// The reference's docstring, verbatim. The `allow`: RST's `::` block is Python, not Rust, and
// fencing it would put backticks into the generated Python docstring. The `cfg_attr(doctest)`
// fences exist only for `cargo test --doc`, which the codegen never reads.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Host)]
#[python(literal, module = "tool_context", name = "DuplicateScope")]
#[host(wire(msg = format!("unknown duplicate_scope {s:?}")))]
pub enum DuplicateScope {
    /// Any in-flight call of the same tool.
    #[default]
    #[host(rename = "name")]
    Name,
    /// Same tool *and* same arguments, so concurrent calls with different arguments coexist.
    #[host(rename = "name_and_args")]
    NameAndArgs,
}

// `llm.Function`, the inner half of a `NamedToolChoice`. `total=False` with a `Required` key is
// the reference's spelling.
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[python(
    typed_dict,
    total = "False",
    module = "tool_context",
    path = "livekit.agents.llm.tool_context",
    name = "Function"
)]
pub struct ToolChoiceFunction {
    // `Required[str]`: under `total=False` a bare `str` is an optional key.
    #[python(ty = "Required[str]", import = "typing_extensions:Required")]
    pub name: String,
}

// `bare_urls`: the `///` below is a Python docstring, and the reference writes the URL bare.
#[allow(rustdoc::bare_urls)]
/// Represents the raw function schema format used in LLM function calling APIs.
///
/// This structure directly maps to OpenAI's function definition format as documented at:
/// https://platform.openai.com/docs/guides/function-calling?api-mode=responses
///
/// It is also compatible with other LLM providers that support raw JSON Schema-based
/// function definitions.
//
// The reference's docstring, verbatim. `total=True`: `name` and `parameters` are required.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    typed_dict,
    module = "tool_context",
    path = "livekit.agents.llm.tool_context",
    name = "RawFunctionDescription"
)]
#[binding(emit)]
pub struct RawFunctionDescription {
    #[python(ty = "str")]
    pub name: String,
    // `NotRequired[str | None]`: absent and explicitly null are both legal.
    #[python(ty = "NotRequired[str | None]", import = "typing_extensions:NotRequired")]
    pub description: Option<String>,
    // As the reference says; the equivalence gate diffs annotation text.
    #[python(ty = "dict[str, object]")]
    pub parameters: Value,
}

pub type DuplicateKey = (String, Option<String>);

pub const CONFIRM_DUPLICATE_PARAM: &str = "lk_agents_confirm_duplicate";

#[crate::host(file(module = "tool_context"))]
const _: () = ();

/// The description the model reads for that parameter. A prompt, so rewording it changes
/// behaviour: it must say the user asked for the repeat.
#[crate::host(constant)]
pub const CONFIRM_DUPLICATE_DESCRIPTION: &str = "Set this to True to confirm you want to run a duplicate. Only do this when user confirms the duplication is needed.";

#[cfg(test)]
mod tests {
    use super::ToolFlag;
    // Catches: `from_bits` accepting a bit this version does not know.
    #[test]
    fn an_unknown_bit_is_refused() {
        assert_eq!(ToolFlag::from_bits(1).map(ToolFlag::bits), Some(1));
        assert!(ToolFlag::from_bits(2).is_none(), "CANCELLABLE is not a tool flag in 2.0");
        assert!(ToolFlag::from_bits(4).is_none());
        assert!(ToolFlag::from_bits(8).is_none());
    }
}
