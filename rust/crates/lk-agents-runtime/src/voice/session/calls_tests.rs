use super::ToolBodies;
use lk_agents_abi::agent::Invocation;
use lk_agents_abi::llm::tool_executor::{ToolOutcome, ToolReport};

fn invocation(operation: u64) -> Invocation {
    Invocation {
        invocation_id: "run".into(),
        operation,
        function_call: Some(lk_agents_abi::FunctionCall {
            id: format!("call-{operation}"),
            call_id: "reused-call".into(),
            name: "lookup".into(),
            arguments: "{}".into(),
            created_at: std::time::UNIX_EPOCH,
            extra: Default::default(),
            group_id: None,
        }),
        agent_name: "agent".into(),
        execution_id: None,
        frame_id: None,
        history: Vec::new(),
        approval: None,
        speech: None,
        host_context: None,
    }
}

#[test]
fn releasing_first_progress_preserves_single_cancellation_until_body_settlement() {
    let bodies = ToolBodies::default();
    let invocation = invocation(1);
    let mut progress = bodies.tool_body(&invocation);
    let report: ToolReport = ToolOutcome::Nothing.into();
    assert!(
        bodies
            .first_tool_progress(
                &invocation.invocation_id,
                invocation.operation,
                "reused-call",
                report
            )
            .is_ok()
    );
    assert!(progress.try_recv().is_ok());
    assert!(bodies.claim_cancellation(&invocation));
    assert!(!bodies.claim_cancellation(&invocation), "core and close must not cancel twice");
    bodies.end_tool_body(&invocation);
    assert!(bodies.0.lock().is_empty());
}

#[test]
fn an_older_body_cannot_remove_or_cancel_a_reused_call_ids_new_invocation() {
    let bodies = ToolBodies::default();
    let old = invocation(1);
    let new = invocation(2);
    let _old_progress = bodies.tool_body(&old);
    assert!(
        bodies
            .first_tool_progress(
                &old.invocation_id,
                old.operation,
                "reused-call",
                ToolOutcome::Nothing.into()
            )
            .is_ok()
    );
    let mut new_progress = bodies.tool_body(&new);
    assert!(bodies.claim_cancellation(&old));
    assert!(bodies.claim_cancellation(&new), "old cancellation did not cancel the new invocation");
    bodies.end_tool_body(&old);
    assert!(
        !bodies.claim_cancellation(&new),
        "old completion did not remove the new cancellation claim"
    );
    assert!(
        bodies
            .first_tool_progress(
                &new.invocation_id,
                new.operation,
                "reused-call",
                ToolOutcome::Nothing.into()
            )
            .is_ok()
    );
    assert!(new_progress.try_recv().is_ok());
    bodies.end_tool_body(&new);
    assert!(bodies.0.lock().is_empty());
}

#[test]
fn late_old_progress_cannot_consume_a_replacement_invocations_sender() {
    let bodies = ToolBodies::default();
    let old = invocation(1);
    let new = invocation(2);
    let mut old_progress = bodies.tool_body(&old);
    assert!(
        bodies
            .first_tool_progress(
                &old.invocation_id,
                old.operation,
                "reused-call",
                ToolOutcome::Nothing.into()
            )
            .is_ok()
    );
    assert!(old_progress.try_recv().is_ok());
    let mut new_progress = bodies.tool_body(&new);
    assert!(
        bodies
            .first_tool_progress(
                &old.invocation_id,
                old.operation,
                "reused-call",
                ToolOutcome::Update { message: "old cleanup".into(), raw: "old cleanup".into() }
                    .into()
            )
            .is_err(),
        "old progress is routed only to its captured core owner"
    );
    assert!(matches!(
        new_progress.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
    assert!(
        bodies
            .first_tool_progress(
                &new.invocation_id,
                new.operation,
                "reused-call",
                ToolOutcome::Update { message: "new work".into(), raw: "new work".into() }.into()
            )
            .is_ok()
    );
    let report = new_progress.try_recv().expect("the replacement receives only its own update");
    assert!(matches!(report.outcome, ToolOutcome::Update { message, .. } if message == "new work"));
}

#[test]
fn updates_after_the_first_keep_their_owner_until_the_actual_body_finishes() {
    let bodies = ToolBodies::default();
    let invocation = invocation(1);
    let mut first = bodies.tool_body(&invocation);
    let report = || ToolOutcome::Update { message: "working".into(), raw: "working".into() };
    assert!(
        bodies
            .first_tool_progress(
                &invocation.invocation_id,
                invocation.operation,
                "reused-call",
                report().into()
            )
            .is_ok()
    );
    assert!(first.try_recv().is_ok());
    // Core decides whether this is an expert's live question or an ordinary terminal.
    assert!(
        bodies
            .first_tool_progress(
                &invocation.invocation_id,
                invocation.operation,
                "reused-call",
                ToolOutcome::InputRequired {
                    question: "Which city?".into(),
                    verbatim: false,
                    tool: None
                }
                .into()
            )
            .is_err()
    );
    assert!(
        bodies
            .first_tool_progress(
                &invocation.invocation_id,
                invocation.operation,
                "reused-call",
                report().into()
            )
            .is_err()
    );
    assert!(
        bodies
            .first_tool_progress(
                &invocation.invocation_id,
                invocation.operation,
                "reused-call",
                ToolOutcome::Nothing.into()
            )
            .is_err()
    );
    bodies.end_tool_body(&invocation);
    assert!(
        bodies
            .first_tool_progress(
                &invocation.invocation_id,
                invocation.operation,
                "reused-call",
                report().into()
            )
            .is_err(),
        "finished-body receipts go only to their captured core owner for rejection"
    );
}
#[test]
fn simultaneous_same_call_ids_route_first_progress_to_the_exact_invocation() {
    let bodies = ToolBodies::default();
    let old = invocation(1);
    let new = invocation(2);
    let mut old_progress = bodies.tool_body(&old);
    let mut new_progress = bodies.tool_body(&new);
    assert!(
        bodies
            .first_tool_progress(
                &new.invocation_id,
                new.operation,
                "reused-call",
                ToolOutcome::Nothing.into()
            )
            .is_ok()
    );
    assert!(matches!(
        old_progress.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
    assert!(new_progress.try_recv().is_ok());
    assert!(
        bodies
            .first_tool_progress(
                &old.invocation_id,
                old.operation,
                "reused-call",
                ToolOutcome::Nothing.into()
            )
            .is_ok()
    );
    assert!(old_progress.try_recv().is_ok());
}
