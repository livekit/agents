//! `llm/tool_context.py`'s Rust half: tool metadata and the duplicate-call decision. `ToolFlag`
//! stays a Python `Flag` (callers write `A | B`); Rust holds the bits.

use std::collections::BTreeMap;

use lk_agents_host_derive::Host;
use serde_json::Value;

/// The values are the reference's `auto()` order and the wire to the shim: never reorder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ToolFlag(u8);

impl ToolFlag {
    pub const NONE: ToolFlag = ToolFlag(0);
    pub const IGNORE_ON_ENTER: ToolFlag = ToolFlag(1);
    pub const CANCELLABLE: ToolFlag = ToolFlag(2);

    pub const fn bits(self) -> u8 {
        self.0
    }
    /// `None` for a bit this version does not know, so a newer host's flag is not reinterpreted.
    pub const fn from_bits(bits: u8) -> Option<ToolFlag> {
        if bits & !0b11 == 0 { Some(ToolFlag(bits)) } else { None }
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

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<pyo3::Bound<'py, pyo3::types::PyInt>, std::convert::Infallible> {
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

/// Build the key a call is grouped under.
///
/// Arguments are compared after validation and canonicalised through a `BTreeMap`. Comparison
/// fails open: an unrepresentable argument groups the call under the name alone.
#[crate::host(export)]
pub fn duplicate_key(name: &str, args: Option<&Value>, scope: DuplicateScope) -> DuplicateKey {
    let args_key = match (scope, args) {
        (DuplicateScope::NameAndArgs, Some(Value::Object(m))) => {
            let sorted: BTreeMap<_, _> = m.iter().collect();
            serde_json::to_string(&sorted).ok()
        }
        (DuplicateScope::NameAndArgs, Some(v)) => serde_json::to_string(v).ok(),
        _ => None,
    };
    (name.to_owned(), args_key)
}

/// `_check_duplicate`'s decision (`tool_executor.py:649`) for the host, without its effects.
/// `in_flight` is `[(call_id, allow_cancellation)]` for calls with the same duplicate key. Returns
/// `("proceed", None)`, `("cancel_then_proceed", [call_id, ..])`, `("reject", None)`,
/// `("confirm", None)` or `("not_cancellable", name)`.
//
// `allow_cancellation` is the running task's, not the tool's flags. `confirm_duplicate` is the
// injected `lk_agents_confirm_duplicate` parameter the model sets to re-assert a call.
// `not_cancellable`: `replace` asked for, but an in-flight call refused (the reference raises).
#[crate::host(export(pid = false, name = "decide_duplicate"))]
pub fn decide_duplicate_wire(
    name: &str,
    mode: DuplicateMode,
    confirm_duplicate: Option<bool>,
    in_flight: Vec<(String, bool)>,
) -> (&'static str, Option<Value>) {
    // Checked before the in-flight lookup, matching the reference: `allow` never even looks.
    if mode == DuplicateMode::Allow || in_flight.is_empty() {
        return ("proceed", None);
    }
    match mode {
        DuplicateMode::Allow => ("proceed", None),
        DuplicateMode::Replace if in_flight.iter().any(|(_, cancellable)| !cancellable) => {
            ("not_cancellable", Some(name.into()))
        }
        DuplicateMode::Replace => {
            ("cancel_then_proceed", Some(in_flight.into_iter().map(|(id, _)| id).collect::<Vec<_>>().into()))
        }
        DuplicateMode::Reject => ("reject", None),
        // Truthiness, not presence: missing and `false` both mean "not confirmed".
        DuplicateMode::Confirm if !confirm_duplicate.unwrap_or(false) => ("confirm", None),
        DuplicateMode::Confirm => ("proceed", None),
    }
}

pub const CONFIRM_DUPLICATE_PARAM: &str = "lk_agents_confirm_duplicate";

/// The description the model reads for that parameter. A prompt, so rewording it changes
/// behaviour: it must say the user asked for the repeat.
#[crate::host(export(pid = false, name = "confirm_duplicate_description"))]
pub const CONFIRM_DUPLICATE_DESCRIPTION: &str = "Set this to True to confirm you want to run a duplicate. Only do this when user confirms the duplication is needed.";

/// `_inject_confirm_duplicate` (`llm/tool_context.py:499`): add the confirmation flag to a **raw**
/// JSON schema, strict-mode conformant.
///
/// The type is `["boolean", "null"]` and the name is appended to `required`: strict mode has no
/// optional property and rejects a `required` that disagrees with `properties`. Added even with
/// no `properties` (a no-argument tool needs it most). Re-injecting is a no-op.
#[crate::host(export(pid = false))]
pub fn inject_confirm_duplicate(parameters: &Value) -> Value {
    let mut params = match parameters {
        Value::Object(m) => m.clone(),
        // The reference would raise on a non-mapping; the caller has already validated a dict.
        _ => serde_json::Map::new(),
    };

    let mut properties = match params.get("properties") {
        Some(Value::Object(m)) => m.clone(),
        _ => serde_json::Map::new(),
    };
    properties.insert(
        CONFIRM_DUPLICATE_PARAM.to_owned(),
        serde_json::json!({
            "type": ["boolean", "null"],
            "description": CONFIRM_DUPLICATE_DESCRIPTION,
        }),
    );
    params.insert("properties".to_owned(), Value::Object(properties));

    let mut required = match params.get("required") {
        Some(Value::Array(a)) => a.clone(),
        _ => Vec::new(),
    };
    if !required.iter().any(|v| v.as_str() == Some(CONFIRM_DUPLICATE_PARAM)) {
        required.push(Value::String(CONFIRM_DUPLICATE_PARAM.to_owned()));
    }
    params.insert("required".to_owned(), Value::Array(required));

    Value::Object(params)
}

/// `make_function_call_output`'s table: `(output, is_error, reply_required)` for one outcome.
/// `output` is `None` where the host owns the text. An unknown outcome raises: a typo would
/// otherwise mean `is_error=False` on a failed call.
//
// The outcome is the host's `isinstance` verdict (`llm/utils.py:713`). The host's text is the
// exception's `message` for a `ToolError`, `str(output or "")` for `ok`. `StopResponse` is
// answered (`reply_required = false`, `is_error = false`): a realtime provider rejects an
// unanswered call, and an error would make the agent apologise. The error strings are
// model-facing prompts and deliberately leak nothing about the exception.
#[crate::host(export(pid = false))]
pub fn tool_call_output_shape(outcome: &str) -> Result<(Option<&'static str>, bool, bool), String> {
    Ok(match outcome {
        "tool_error" => (None, true, true),
        "stop" => (Some(""), false, false),
        "internal" => (Some("An internal error occurred"), true, true),
        "invalid" => (Some("the tool returned an invalid output"), true, true),
        "ok" => (None, false, true),
        _ => {
            return Err(format!(
                "unknown tool outcome {outcome:?} (expected tool_error, stop, internal, invalid or ok)"
            ));
        }
    })
}

#[cfg(test)]
mod tests {

    use super::*;
    use serde_json::json;

    fn decide(mode: DuplicateMode, confirm: Option<bool>, in_flight: &[(&str, bool)]) -> (&'static str, Option<Value>) {
        let in_flight = in_flight.iter().map(|(id, c)| ((*id).to_owned(), *c)).collect();
        decide_duplicate_wire("f", mode, confirm, in_flight)
    }

    // Catches: `allow` scanning the in-flight set, a first call rejected, `replace` cancelling a
    // non-cancellable task or only the first duplicate, or a missing `confirm_duplicate` read as yes.
    #[test]
    fn each_mode_decides_against_the_in_flight_set() {
        use DuplicateMode::*;
        let busy = [("a", true)];
        let cases: &[(DuplicateMode, Option<bool>, &[(&str, bool)], (&str, Option<Value>))] = &[
            (Allow, None, &[("a", false)], ("proceed", None)),
            (Reject, None, &[], ("proceed", None)),
            (Replace, None, &[], ("proceed", None)),
            (Confirm, None, &[], ("proceed", None)),
            (Replace, None, &[("a", true), ("b", false)], ("not_cancellable", Some(json!("f")))),
            (Replace, None, &[("a", true), ("b", true)], ("cancel_then_proceed", Some(json!(["a", "b"])))),
            (Confirm, None, &busy, ("confirm", None)),
            (Confirm, Some(false), &busy, ("confirm", None)),
            (Confirm, Some(true), &busy, ("proceed", None)),
            (Reject, Some(true), &busy, ("reject", None)),
        ];
        for (mode, confirm, in_flight, want) in cases {
            assert_eq!(&decide(*mode, *confirm, in_flight), want, "{mode:?} {confirm:?} {in_flight:?}");
        }
    }

    // Catches: injecting `{"type": "boolean"}`, or dropping the `required` entry.
    #[test]
    fn the_injected_flag_is_nullable_and_required() {
        let out = inject_confirm_duplicate(&json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }));
        assert_eq!(out["properties"][CONFIRM_DUPLICATE_PARAM]["type"], json!(["boolean", "null"]));
        assert_eq!(out["required"], json!(["city", CONFIRM_DUPLICATE_PARAM]));
        // The tool's own properties survive.
        assert_eq!(out["properties"]["city"], json!({"type": "string"}));
    }

    // Catches: skipping injection for a tool with no parameters, or appending the requirement
    // twice on a re-injected schema, which strict mode rejects.
    #[test]
    fn a_tool_with_no_parameters_gets_the_flag_once() {
        let once = inject_confirm_duplicate(&json!({"type": "object"}));
        assert!(once["properties"][CONFIRM_DUPLICATE_PARAM].is_object());
        assert_eq!(inject_confirm_duplicate(&once)["required"], json!([CONFIRM_DUPLICATE_PARAM]));
    }

    // Catches: `StopResponse` written as an error or left unanswered, a `ToolError` given a canned
    // message instead of the author's text, exception detail leaking to the model, or an unknown
    // tag defaulting instead of refusing.
    #[test]
    fn each_outcome_has_the_references_output_shape_and_an_unknown_one_is_refused() {
        assert_eq!(tool_call_output_shape("stop"), Ok((Some(""), false, false)));
        assert_eq!(tool_call_output_shape("tool_error"), Ok((None, true, true)));
        assert_eq!(tool_call_output_shape("internal"), Ok((Some("An internal error occurred"), true, true)));
        assert_eq!(tool_call_output_shape("invalid"), Ok((Some("the tool returned an invalid output"), true, true)));
        assert_eq!(tool_call_output_shape("ok"), Ok((None, false, true)));
        assert!(tool_call_output_shape("error").is_err());
    }

    // Catches: `name` scope including the arguments, hashing arguments in serialized order, or
    // treating "no arguments" as an empty-object key.
    #[test]
    fn duplicate_keys_follow_the_scope() {
        use DuplicateScope::*;
        let key = |args: Option<Value>, scope| duplicate_key("f", args.as_ref(), scope);
        let a = key(Some(json!({"x": 1})), Name);
        assert_eq!(a, key(Some(json!({"x": 2})), Name));
        assert_eq!(a.1, None);
        let ab = key(Some(json!({"a": 1, "b": 2})), NameAndArgs);
        assert_eq!(ab, key(Some(json!({"b": 2, "a": 1})), NameAndArgs));
        assert_ne!(ab, key(Some(json!({"a": 1})), NameAndArgs));
        assert_eq!(key(None, NameAndArgs).1, None);
    }

    // Catches: `from_bits` accepting a bit this version does not know.
    #[test]
    fn an_unknown_bit_is_refused() {
        assert_eq!(ToolFlag::from_bits(3).map(ToolFlag::bits), Some(3));
        assert!(ToolFlag::from_bits(4).is_none());
        assert!(ToolFlag::from_bits(8).is_none());
    }
}
