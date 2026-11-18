use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::agent::ArgumentIdentity;
use lk_agents_abi::llm::tool_executor::Approval;
use lk_agents_abi::{ChatItem, FunctionCall};
use serde_json::Value;

// A `needs_approval` call runs only as the repeat of a read-back the person answered in this
// invocation, with the same validated arguments; each answer is consumed once. The owning
// conversation supplies matched canonical call IDs; copied history supplies no response authority.

/// `FunctionCall.extra` key: `asked` when the call returned the read-back, `granted` when it ran
/// on the person's answer.
pub const APPROVAL_KEY: &str = "lk.approval";

/// `FunctionCall.extra` key beside [`APPROVAL_KEY`]: the canonical identity of the validated
/// arguments the call was decided on. Raw model JSON is never the identity: defaults, coercions
/// and application validators change what the body would run with.
pub const APPROVAL_IDENTITY_KEY: &str = "lk.approval_identity";

/// Private provenance of a grant: the canonical read-back call whose response it consumed.
pub const APPROVAL_SOURCE_KEY: &str = "lk.approval_source";

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
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                out.extend((i > 0).then_some(','));
                out.push_str(&Value::String(key.clone()).to_string());
                out.push(':');
                write_canonical(&map[key], out);
            }
            out.push('}');
        }
        Value::Array(values) => {
            out.push('[');
            for (i, v) in values.iter().enumerate() {
                out.extend((i > 0).then_some(','));
                write_canonical(v, out);
            }
            out.push(']');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

/// Whether a call of `tool` with `identity` repeats an explicitly answered read-back of that
/// tool, and no earlier repeat consumed that occurrence. `items` are the calls decided
/// before this one, with their outputs; `answered` identifies the pending calls this invocation
/// explicitly admitted as answered when it started. Output text and call metadata do not admit
/// a response by themselves.
pub fn granted<'a>(
    items: impl IntoIterator<Item = &'a ChatItem>,
    answered: &BTreeSet<String>,
    tool: &str,
    identity: &str,
) -> Option<String> {
    // Replies commit their own history groups, and persisted stamps can tie. Exact source
    // identities establish consumption without inferring causality from storage order.
    let calls: Vec<&FunctionCall> = items
        .into_iter()
        .filter_map(|item| match item {
            ChatItem::FunctionCall(call) if call.name == tool => Some(call),
            _ => None,
        })
        .collect();
    let mut readbacks = BTreeMap::new();
    for call in &calls {
        if mark(call) == Some(Approval::Asked)
            && let Some(identity) = call.extra.get(APPROVAL_IDENTITY_KEY).map(decided_identity)
        {
            if readbacks.get(&call.id).is_some_and(|previous| previous != &identity) {
                return None;
            }
            readbacks.insert(call.id.clone(), identity);
        }
    }
    let mut consumed = BTreeSet::new();
    for call in calls {
        if mark(call) == Some(Approval::Granted) {
            let source = call.extra.get(APPROVAL_SOURCE_KEY)?.as_str()?;
            let decided = call.extra.get(APPROVAL_IDENTITY_KEY).map(decided_identity)?;
            if readbacks.get(source) != Some(&decided) || !consumed.insert(source.to_owned()) {
                return None;
            }
        }
    }
    answered.iter().find_map(|id| {
        (!consumed.contains(id) && readbacks.get(id).is_some_and(|matched| matched == identity))
            .then(|| id.clone())
    })
}

/// The identity a call was decided on. Stored history keeps `extra` as strings and reads JSON
/// back as values, so a JSON identity may return as the arguments object it spells.
fn decided_identity(value: &Value) -> String {
    match value {
        Value::String(identity) => identity.clone(),
        arguments => canonical_json(arguments),
    }
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
    call.extra.remove(APPROVAL_SOURCE_KEY);
    call.extra.insert(APPROVAL_KEY.to_owned(), Value::String(approval.as_str().to_owned()));
    call.extra.insert(APPROVAL_IDENTITY_KEY.to_owned(), Value::String(identity.to_owned()));
}

/// A validated repeat records exactly which independently matched read-back it spent.
pub fn set_grant(call: &mut FunctionCall, identity: &str, source: String) {
    set_mark(call, Approval::Granted, identity);
    call.extra.insert(APPROVAL_SOURCE_KEY.into(), Value::String(source));
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

    #[test]
    fn a_matched_canonical_readback_survives_newer_questions_and_grants_once() {
        let at = |ms| std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_millis(ms);
        let identity = canonical(&ArgumentIdentity::Json(json!({})));
        let asked = |call_id: &str, created| {
            let mut call = FunctionCall {
                id: format!("item_{call_id}"),
                call_id: call_id.into(),
                arguments: "{}".into(),
                name: "transfer".into(),
                created_at: at(created),
                extra: Default::default(),
                group_id: None,
            };
            set_mark(&mut call, Approval::Asked, &identity);
            ChatItem::FunctionCall(call)
        };
        // Asked later, committed earlier in the history than the other reply's read-back.
        let history = [asked("answered", 2_000), asked("earlier", 1_000)];
        let answered = BTreeSet::from(["item_answered".to_owned()]);
        assert_eq!(
            granted(&history, &answered, "transfer", &identity),
            Some("item_answered".into())
        );
        let mut later = asked("answered", 3_000);
        let ChatItem::FunctionCall(call) = &mut later else { unreachable!() };
        call.id = "new-item-with-reused-wire-id".into();
        let history = [history[0].clone(), history[1].clone(), later];
        assert_eq!(
            granted(&history, &answered, "transfer", &identity),
            Some("item_answered".into())
        );
        assert!(
            granted(&history, &BTreeSet::from(["answered".into()]), "transfer", &identity)
                .is_none(),
            "a provider call ID is not a matched canonical occurrence"
        );
        let mut consumed = history[0].clone();
        let ChatItem::FunctionCall(call) = &mut consumed else { unreachable!() };
        call.id = "repeat".into();
        call.created_at = at(4_000);
        set_grant(call, &identity, "item_answered".into());
        let history = [history[0].clone(), history[1].clone(), history[2].clone(), consumed];
        assert!(
            granted(&history, &answered, "transfer", &identity).is_none(),
            "the unmatched question with a reused provider ID cannot grant another repeat"
        );
    }

    #[test]
    fn two_matched_confirmations_with_distinct_validated_arguments_grant_independently() {
        let first = canonical(&ArgumentIdentity::Json(json!({"date":"Monday", "notify":true})));
        let second = canonical(&ArgumentIdentity::Json(json!({"date":"Tuesday", "notify":true})));
        let decided = |id: &str, index, identity: &str, mark| {
            let mut call = FunctionCall {
                id: id.into(),
                call_id: "reused-provider-id".into(),
                name: "cancel".into(),
                arguments: "{}".into(),
                created_at: std::time::UNIX_EPOCH + std::time::Duration::from_secs(index),
                extra: Default::default(),
                group_id: None,
            };
            if mark == Approval::Granted {
                let source = if id == "repeat-first" { "first" } else { "second" };
                set_grant(&mut call, identity, source.into());
            } else {
                set_mark(&mut call, mark, identity);
            }
            ChatItem::FunctionCall(call)
        };
        let mut history = vec![
            decided("first", 1, &first, Approval::Asked),
            decided("second", 2, &second, Approval::Asked),
        ];
        let answered = BTreeSet::from(["first".into(), "second".into()]);
        assert_eq!(granted(&history, &answered, "cancel", &first), Some("first".into()));
        assert_eq!(granted(&history, &answered, "cancel", &second), Some("second".into()));
        history.push(decided("repeat-first", 3, &first, Approval::Granted));
        assert!(granted(&history, &answered, "cancel", &first).is_none());
        assert_eq!(granted(&history, &answered, "cancel", &second), Some("second".into()));
        history.push(decided("repeat-second", 4, &second, Approval::Granted));
        assert!(granted(&history, &answered, "cancel", &first).is_none());
        assert!(granted(&history, &answered, "cancel", &second).is_none());
    }

    #[test]
    fn granting_one_of_two_identical_readbacks_does_not_spend_the_other_later_answer() {
        let identity = canonical(&ArgumentIdentity::Json(json!({"target":"one"})));
        let asked = |id: &str, seconds| {
            let mut call = FunctionCall {
                id: id.into(),
                call_id: "same-provider-id".into(),
                name: "delete".into(),
                arguments: "{}".into(),
                extra: Default::default(),
                group_id: None,
                created_at: std::time::UNIX_EPOCH + std::time::Duration::from_secs(seconds),
            };
            set_mark(&mut call, Approval::Asked, &identity);
            call
        };
        let first = asked("first", 1);
        let second = asked("second", 2);
        let mut history =
            vec![ChatItem::FunctionCall(first.clone()), ChatItem::FunctionCall(second)];
        let first_answer = BTreeSet::from(["first".into()]);
        let source = granted(&history, &first_answer, "delete", &identity).unwrap();
        assert_eq!(source, "first");
        let mut repeat = asked("repeat-first", 3);
        set_grant(&mut repeat, &identity, source);
        history.push(repeat.into());
        let second_answer = BTreeSet::from(["second".into()]);
        let source = granted(&history, &second_answer, "delete", &identity).unwrap();
        assert_eq!(source, "second");
        let mut repeat = asked("repeat-second", 4);
        set_grant(&mut repeat, &identity, source);
        history.push(repeat.into());
        assert!(
            granted(
                &history,
                &BTreeSet::from(["first".into(), "second".into()]),
                "delete",
                &identity
            )
            .is_none()
        );
    }

    #[test]
    fn equal_time_history_groups_can_record_a_grant_before_its_readback() {
        let identity = canonical(&ArgumentIdentity::Json(json!({"target":"one"})));
        let asked = |id: &str| {
            let mut call = FunctionCall {
                id: id.into(),
                call_id: "reused-provider-id".into(),
                name: "delete".into(),
                arguments: "{}".into(),
                created_at: std::time::UNIX_EPOCH,
                extra: Default::default(),
                group_id: None,
            };
            set_mark(&mut call, Approval::Asked, &identity);
            call
        };
        let first = asked("first");
        let second = asked("second");
        let mut granted_first = asked("granted-first");
        set_grant(&mut granted_first, &identity, first.id.clone());
        let history = [granted_first.into(), second.into(), first.into()];
        assert_eq!(
            granted(&history, &BTreeSet::from(["second".into()]), "delete", &identity),
            Some("second".into()),
            "the grant spent only its exact source, independently of timestamp ties"
        );
        assert!(
            granted(&history, &BTreeSet::from(["first".into()]), "delete", &identity).is_none()
        );
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
