//! Tool duplicate grouping and confirmation schema policy.
pub use lk_agents_abi::llm::provider_format::tools::*;
use serde_json::Value;
use std::collections::BTreeMap;

/// Build the key a call is grouped under.
///
/// Arguments are compared after validation and canonicalised through a `BTreeMap`. Comparison
/// fails open: an unrepresentable argument groups the call under the name alone.
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
        DuplicateMode::Replace => (
            "cancel_then_proceed",
            Some(in_flight.into_iter().map(|(id, _)| id).collect::<Vec<_>>().into()),
        ),
        DuplicateMode::Reject => ("reject", None),
        // Truthiness, not presence: missing and `false` both mean "not confirmed".
        DuplicateMode::Confirm if !confirm_duplicate.unwrap_or(false) => ("confirm", None),
        DuplicateMode::Confirm => ("proceed", None),
    }
}

/// `_inject_confirm_duplicate` (`llm/tool_context.py:499`): add the confirmation flag to a **raw**
/// JSON schema, strict-mode conformant.
///
/// The type is `["boolean", "null"]` and the name is appended to `required`: strict mode has no
/// optional property and rejects a `required` that disagrees with `properties`. Added even with
/// no `properties` (a no-argument tool needs it most). Re-injecting is a no-op.
pub fn inject_confirm_duplicate(parameters: &Value) -> serde_json::Map<String, Value> {
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

    params
}

#[cfg(test)]
mod tests {

    use super::*;
    use serde_json::json;

    fn decide(
        mode: DuplicateMode,
        confirm: Option<bool>,
        in_flight: &[(&str, bool)],
    ) -> (&'static str, Option<Value>) {
        let in_flight = in_flight.iter().map(|(id, c)| ((*id).to_owned(), *c)).collect();
        decide_duplicate_wire("f", mode, confirm, in_flight)
    }

    // Catches: `allow` scanning the in-flight set, a first call rejected, `replace` cancelling a
    // non-cancellable task or only the first duplicate, or a missing `confirm_duplicate` read as yes.
    #[test]
    fn each_mode_decides_against_the_in_flight_set() {
        use DuplicateMode::*;
        let busy = [("a", true)];
        type Case<'a> =
            (DuplicateMode, Option<bool>, &'a [(&'a str, bool)], (&'a str, Option<Value>));
        let cases: &[Case<'_>] = &[
            (Allow, None, &[("a", false)], ("proceed", None)),
            (Reject, None, &[], ("proceed", None)),
            (Replace, None, &[], ("proceed", None)),
            (Confirm, None, &[], ("proceed", None)),
            (Replace, None, &[("a", true), ("b", false)], ("not_cancellable", Some(json!("f")))),
            (
                Replace,
                None,
                &[("a", true), ("b", true)],
                ("cancel_then_proceed", Some(json!(["a", "b"]))),
            ),
            (Confirm, None, &busy, ("confirm", None)),
            (Confirm, Some(false), &busy, ("confirm", None)),
            (Confirm, Some(true), &busy, ("proceed", None)),
            (Reject, Some(true), &busy, ("reject", None)),
        ];
        for (mode, confirm, in_flight, want) in cases {
            assert_eq!(
                &decide(*mode, *confirm, in_flight),
                want,
                "{mode:?} {confirm:?} {in_flight:?}"
            );
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
        assert_eq!(
            inject_confirm_duplicate(&Value::Object(once))["required"],
            json!([CONFIRM_DUPLICATE_PARAM])
        );
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
}
