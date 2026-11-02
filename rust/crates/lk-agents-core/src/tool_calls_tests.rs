//! Duplicate admission uses session-owned live execution facts, including detached bodies.

use super::*;
use serde_json::json;
use std::time::SystemTime;

fn item(id: &str, arguments: Value) -> FunctionCall {
    FunctionCall {
        id: id.into(),
        call_id: id.into(),
        name: "lookup".into(),
        arguments: arguments.to_string(),
        created_at: SystemTime::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    }
}

fn tool(mode: DuplicateMode, scope: DuplicateScope) -> FunctionToolInfo {
    FunctionToolInfo {
        spec: ToolSpec {
            provider_tool: None,
            name: "lookup".into(),
            description: "Look up a record".into(),
            json_schema: json!({"type": "object", "properties": {"id": {"type": "string"}}}),
            flags: Default::default(),
        },
        raw_schema: false,
        needs_approval: None,
        on_duplicate: mode,
        duplicate_scope: scope,
        retries: None,
        remote: None,
    }
}

fn body(execution: u64, id: &str, identity: Value) -> RunningToolExecution {
    RunningToolExecution {
        execution,
        // The model's spelling is not the prepared identity used for duplicate comparison.
        call: item(id, json!({})),
        identity: ArgumentIdentity::Json(identity),
    }
}

fn prepare(
    tools: &[FunctionToolInfo],
    arguments: Value,
    identity: Value,
    running: &[RunningToolExecution],
) -> (ToolCalls, Vec<Effect>) {
    let mut calls = ToolCalls::new(BTreeSet::new());
    let mut fx = Vec::new();
    calls.add(CallId(1), item("new", arguments), tools, &mut fx);
    fx.clear();
    assert!(calls.prepared(
        CallId(1),
        (ResourceId(9), ArgumentIdentity::Json(identity)),
        tools,
        std::iter::empty(),
        running,
        &mut fx,
    ));
    (calls, fx)
}

#[test]
fn detached_execution_rejects_a_new_steps_duplicate_using_prepared_identity() {
    let tools = [tool(DuplicateMode::Reject, DuplicateScope::NameAndArgs)];
    let running = [body(41, "earlier-step", json!({"id": "same"}))];
    let (_, fx) = prepare(&tools, json!({}), json!({"id": "same"}), &running);
    assert!(fx.contains(&Effect::ReleaseArguments { prepared: ResourceId(9) }));
    let template = fx.iter().find_map(|e| match e {
        Effect::RenderTemplate { template, .. } => Some(template),
        _ => None,
    });
    let Some(ToolTemplate::DuplicateReject { fnc_calls_text, .. }) = template else {
        panic!("expected duplicate rejection, got {fx:?}")
    };
    assert!(fnc_calls_text.contains("earlier-step"));
    assert!(!fx.iter().any(|e| matches!(e, Effect::Execute { .. })));
}

#[test]
fn argument_scope_permits_a_different_identity_but_name_scope_does_not() {
    let running = [body(41, "other-reply", json!({"id": "old"}))];
    for (scope, executes) in [(DuplicateScope::NameAndArgs, true), (DuplicateScope::Name, false)] {
        let tools = [tool(DuplicateMode::Reject, scope)];
        let (_, fx) = prepare(&tools, json!({"id": "new"}), json!({"id": "new"}), &running);
        assert_eq!(fx.iter().any(|e| matches!(e, Effect::Execute { .. })), executes);
    }
}

#[test]
fn replace_cancels_each_matching_execution_even_when_model_call_ids_repeat() {
    let tools = [tool(DuplicateMode::Replace, DuplicateScope::NameAndArgs)];
    let running = [
        body(41, "reused-model-id", json!({"id": "same"})),
        body(42, "reused-model-id", json!({"id": "same"})),
        body(43, "unrelated", json!({"id": "different"})),
    ];
    let (_, fx) = prepare(&tools, json!({"id": "same"}), json!({"id": "same"}), &running);
    let cancelled: Vec<u64> = fx
        .iter()
        .filter_map(|e| match e {
            Effect::CancelExecution { execution } => Some(*execution),
            _ => None,
        })
        .collect();
    assert_eq!(cancelled, [41, 42]);
    assert!(fx.iter().any(|e| matches!(e, Effect::Execute { .. })));
}

#[test]
fn confirm_checks_live_execution_until_explicitly_reasserted() {
    let tools = [tool(DuplicateMode::Confirm, DuplicateScope::Name)];
    let running = [body(41, "earlier-step", json!({"id": "same"}))];
    for confirmed in [false, true] {
        let arguments = json!({"id": "same", CONFIRM_DUPLICATE_PARAM: confirmed});
        let (_, fx) = prepare(&tools, arguments, json!({"id": "same"}), &running);
        assert_eq!(fx.iter().any(|e| matches!(e, Effect::Execute { .. })), confirmed);
        assert_eq!(
            fx.iter().any(|e| matches!(
                e,
                Effect::RenderTemplate { template: ToolTemplate::DuplicateConfirm { .. }, .. }
            )),
            !confirmed
        );
    }
}

#[test]
fn ended_execution_is_not_retained_in_an_independent_duplicate_registry() {
    let tools = [tool(DuplicateMode::Reject, DuplicateScope::Name)];
    let running = [body(41, "earlier-step", json!({"id": "same"}))];
    let (_, blocked) = prepare(&tools, json!({}), json!({}), &running);
    assert!(!blocked.iter().any(|e| matches!(e, Effect::Execute { .. })));
    let (_, next) = prepare(&tools, json!({}), json!({}), &[]);
    assert!(next.iter().any(|e| matches!(e, Effect::Execute { .. })));
}

#[test]
fn cross_reply_requested_cancellation_does_not_become_an_unexpected_tool_cancel() {
    let tools = [tool(DuplicateMode::Allow, DuplicateScope::Name)];
    let (mut calls, mut fx) = prepare(&tools, json!({}), json!({}), &[]);
    calls.request_cancellation("new");
    assert!(!calls.body_settled(CallId(1), ToolOutcome::Cancelled, &mut fx));
    assert!(calls.all_settled());
}
