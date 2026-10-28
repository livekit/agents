//! Approval (tools.md "Approval"): a `needs_approval` tool's call runs only as the repeat of a
//! read-back the person answered, with the same validated arguments, and each answer is consumed
//! once. The state lives in the history: calls carry [`APPROVAL_KEY`] (`asked`, `granted`) and the
//! [`APPROVAL_IDENTITY_KEY`] of the validated arguments they were decided on.

use lk_agents_abi::agent::ArgumentIdentity;
use lk_agents_abi::llm::tool_executor::{Approval, WAITING_PREFIX};
use lk_agents_abi::{ChatItem, FunctionCall};
use serde_json::Value;

/// `FunctionCall.extra` key: `asked` when the call returned the read-back, `granted` when it ran
/// on the person's answer.
pub const APPROVAL_KEY: &str = "lk.approval";

/// `FunctionCall.extra` key beside [`APPROVAL_KEY`]: the canonical identity of the validated
/// arguments the call was decided on. Raw model JSON is never the identity: defaults, coercions
/// and application validators change what the body would run with.
pub const APPROVAL_IDENTITY_KEY: &str = "lk.approval_identity";

/// What the model reads as the output of an answered read-back: the answer alone would read like
/// a call that ran. The history keeps the person's words.
pub fn answered_text(tool: &str, answer: &str) -> String {
    format!(
        "not run yet: the person answered {} to its read-back. If they agreed, call {tool} again \
         with exactly the same arguments to run it; if not, do not, and go on from what they said.",
        python_repr(answer)
    )
}

/// One string per identity: equal validated arguments compare equal whatever their key order.
pub fn canonical(identity: &ArgumentIdentity) -> String {
    match identity {
        ArgumentIdentity::Json(value) => canonical_json(value),
        ArgumentIdentity::Fingerprint(fingerprint) => format!("fingerprint:{fingerprint}"),
    }
}

/// Compact JSON with object keys sorted at every depth.
pub fn canonical_json(value: &Value) -> String {
    fn write(value: &Value, out: &mut String) {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                out.push('{');
                for (i, key) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&Value::String(key.clone()).to_string());
                    out.push(':');
                    write(&map[key], out);
                }
                out.push('}');
            }
            Value::Array(values) => {
                out.push('[');
                for (i, v) in values.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write(v, out);
                }
                out.push(']');
            }
            scalar => out.push_str(&scalar.to_string()),
        }
    }
    let mut out = String::new();
    write(value, &mut out);
    out
}

/// Whether a call of `tool` with `identity` repeats the latest read-back of that tool after the
/// person answered it, and no earlier repeat consumed that answer. `items` are the calls decided
/// before this one, in model order, with their outputs.
pub fn granted(items: &[ChatItem], tool: &str, identity: &str) -> bool {
    let answered = |call_id: &str| {
        items.iter().any(|i| {
            matches!(i, ChatItem::FunctionCallOutput(o)
                if o.call_id == call_id && !o.output.starts_with(WAITING_PREFIX))
        })
    };
    // The identity of the latest answered, unconsumed read-back.
    let mut open: Option<&str> = None;
    for item in items {
        let ChatItem::FunctionCall(call) = item else { continue };
        if call.name != tool {
            continue;
        }
        let decided = call.extra.get(APPROVAL_IDENTITY_KEY).and_then(Value::as_str);
        match mark(call) {
            Some(Approval::Granted) if decided.is_some() && decided == open => open = None,
            Some(Approval::Asked) => open = decided.filter(|_| answered(&call.call_id)),
            _ => {}
        }
    }
    open == Some(identity)
}

/// The call's approval mark, if it has one.
pub fn mark(call: &FunctionCall) -> Option<Approval> {
    match call.extra.get(APPROVAL_KEY).and_then(Value::as_str) {
        Some("asked") => Some(Approval::Asked),
        Some("granted") => Some(Approval::Granted),
        _ => None,
    }
}

/// Marks a call with how its approval went and the identity it was decided on.
pub fn set_mark(call: &mut FunctionCall, approval: Approval, identity: &str) {
    call.extra.insert(APPROVAL_KEY.to_owned(), Value::String(approval.as_str().to_owned()));
    call.extra.insert(APPROVAL_IDENTITY_KEY.to_owned(), Value::String(identity.to_owned()));
}

/// The default read-back: the first sentence of the tool's description and the validated
/// arguments, as the person is asked to confirm them.
pub fn read_back(description: &str, identity: &ArgumentIdentity) -> String {
    let first = description.trim().split(". ").next().unwrap_or_default();
    let plain = first.lines().next().unwrap_or_default().trim_end_matches('.');
    let plain = if plain.is_empty() { "this" } else { plain };
    let shown = match identity {
        ArgumentIdentity::Json(Value::Object(arguments)) if !arguments.is_empty() => arguments
            .iter()
            .map(|(name, value)| format!("{name}={}", python_literal(value)))
            .collect::<Vec<_>>()
            .join(", "),
        _ => return format!("Please confirm: {plain}. Yes or no?"),
    };
    format!("Please confirm: {plain} ({shown}). Yes or no?")
}

/// A JSON value as the Python literal the person reads in a read-back.
fn python_literal(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => python_repr(s),
        Value::Array(values) => {
            format!("[{}]", values.iter().map(python_literal).collect::<Vec<_>>().join(", "))
        }
        Value::Object(map) => {
            let entries: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", python_repr(k), python_literal(v)))
                .collect();
            format!("{{{}}}", entries.join(", "))
        }
    }
}

/// `repr(str)`: single quotes unless the text has one and no double quote.
pub fn python_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Catches: an identity that depends on the order the host serialised the validated fields.
    #[test]
    fn identity_ignores_key_order_at_every_depth() {
        let a =
            ArgumentIdentity::Json(json!({"b": 1, "a": {"y": [1, {"d": 1, "c": 2}], "x": "s"}}));
        let b =
            ArgumentIdentity::Json(json!({"a": {"x": "s", "y": [1, {"c": 2, "d": 1}]}, "b": 1}));
        assert_eq!(canonical(&a), canonical(&b));
        assert_ne!(canonical(&a), canonical(&ArgumentIdentity::Fingerprint(canonical(&a))));
    }

    // Catches: the read-back losing its first-sentence form or showing JSON instead of the
    // values the person recognises.
    #[test]
    fn read_back_names_the_action_and_the_validated_arguments() {
        let identity =
            ArgumentIdentity::Json(json!({"date": "Monday", "count": 5, "notify": true}));
        assert_eq!(
            read_back("Cancel the patient's appointment on a date. Then notify.", &identity),
            "Please confirm: Cancel the patient's appointment on a date (date='Monday', count=5, \
             notify=True). Yes or no?"
        );
        assert_eq!(
            read_back("", &ArgumentIdentity::Json(json!({}))),
            "Please confirm: this. Yes or no?"
        );
    }
}
