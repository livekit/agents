use super::*;
use lk_agents_abi::Role;
use serde_json::json;

fn origin(id: &str) -> LiveTurnOrigin {
    LiveTurnOrigin {
        request_id: format!("request_{id}"),
        execution_id: format!("execution_{id}"),
        task_ids: vec![format!("AST_{id}")],
    }
}

fn human(id: &str, words: &str) -> FrozenHuman {
    FrozenHuman {
        message: ChatMessage::text(id, Role::User, words, std::time::UNIX_EPOCH),
        candidates: Vec::new(),
        offered: Vec::new(),
        history: Vec::new(),
    }
}

fn body(live: &mut Live, invocation: &str, id: u64, canonical: &str) -> BodyKey {
    let key = (invocation.to_owned(), CallId(id));
    live.prepared(key.clone(), ArgumentIdentity::Json(json!({"booking":canonical})));
    live.start_body(
        key.clone(),
        FrameId { value: "root".into() },
        FunctionCall {
            id: canonical.into(),
            // Providers may reuse this ID; attribution still uses the canonical occurrence.
            call_id: "wire_reused".into(),
            name: "reserve".into(),
            arguments: json!({"booking":canonical}).to_string(),
            created_at: std::time::UNIX_EPOCH,
            extra: Default::default(),
            group_id: None,
        },
    )
    .unwrap();
    key
}

#[test]
fn side_turns_finish_before_a_held_body_delivers_to_its_original_task() {
    let root = FrameId { value: "root".into() };
    let mut live = Live::new(origin("booking"));
    let held = body(&mut live, "invocation_A", 1, "item_booking");
    assert!(live.detach(&held));
    let (turn_a, task_a, working) = live.commit_started().unwrap();
    assert_eq!(task_a.task_ids, ["AST_booking"]);
    assert!(working, "an acknowledgement does not complete the business task");
    live.enqueue(origin("question"), human("question", "When is check-in?")).unwrap();
    live.enqueue(origin("independent"), human("independent", "Book a taxi.")).unwrap();
    assert!(live.committed(turn_a));

    for task in ["AST_question", "AST_independent"] {
        let turn = live.next(&root, false).unwrap();
        assert_eq!(turn.origin.task_ids, [task]);
        let (id, _, working) = live.commit_started().unwrap();
        assert!(!working);
        assert!(live.committed(id));
        assert!(!live.idle(), "the original body still has its resource owner");
    }

    assert!(
        live.settle(
            &held,
            ToolOutcome::Value {
                text: "Reserved R1".into(),
                reply_required: true,
                verbatim: false,
                more: None,
            }
            .into()
        )
    );
    let turn = live.next(&root, false).unwrap();
    assert_eq!(turn.origin.task_ids, ["AST_booking"]);
    let TurnInput::Delivery(keys) = turn.input else { panic!("body result delivery") };
    let outcomes = live.delivered_outputs(&keys);
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].call.id, "item_booking");
    let (delivery, _, working) = live.commit_started().unwrap();
    assert!(!working);
    assert!(!live.idle(), "accepted output remains until exact task propagation commits");
    assert!(!live.committed(turn_a), "an older acknowledgement cannot release the result");
    assert!(live.committed(delivery));
    assert!(live.idle());
}

#[test]
fn opaque_progress_cannot_detach_an_unowned_body_and_repeated_settlement_delivers_once() {
    let root = FrameId { value: "root".into() };
    let mut live = Live::new(origin("A"));
    assert!(!live.detach(&("opaque".into(), CallId(1))));
    let held = body(&mut live, "invocation", 1, "item_A");
    assert!(live.detach(&held));
    let (initial, _, _) = live.commit_started().unwrap();
    assert!(live.committed(initial));
    assert!(live.settle(&held, ToolOutcome::Internal.into()));
    assert!(live.settle(&held, ToolOutcome::Internal.into()));
    let delivery = live.next(&root, false).unwrap();
    assert!(matches!(delivery.input, TurnInput::Delivery(ref keys) if keys == &[held]));
    let (id, _, _) = live.commit_started().unwrap();
    assert!(live.committed(id));
    assert!(live.next(&root, false).is_none());
}

#[test]
fn a_result_waits_for_its_own_frame_without_stealing_a_local_child_context() {
    let root = FrameId { value: "root".into() };
    let child = FrameId { value: "child".into() };
    let mut live = Live::new(origin("A"));
    let held = body(&mut live, "invocation", 1, "item_A");
    live.detach(&held);
    let (initial, _, _) = live.commit_started().unwrap();
    live.committed(initial);
    live.settle(&held, ToolOutcome::Internal.into());
    live.enqueue(origin("B"), human("B", "A side question")).unwrap();
    let side = live.next(&child, false).unwrap();
    assert_eq!(side.origin.task_ids, ["AST_B"]);
    let (side, _, _) = live.commit_started().unwrap();
    live.committed(side);
    assert!(live.next(&child, false).is_none());
    assert!(matches!(live.next(&root, false).unwrap().input, TurnInput::Delivery(_)));
}

#[test]
fn a_late_commit_cannot_release_a_newer_turn_or_change_its_human_words() {
    let root = FrameId { value: "root".into() };
    let mut live = Live::new(origin("A"));
    let (initial, _, _) = live.commit_started().unwrap();
    live.enqueue(origin("B"), human("human_B", "Friday, not Saturday.")).unwrap();
    live.committed(initial);
    let turn = live.next(&root, false).unwrap();
    let TurnInput::Human(human) = turn.input else { panic!("queued human input") };
    assert_eq!(human.message.raw_text_content().as_deref(), Some("Friday, not Saturday."));
    assert!(!live.committed(initial));
    assert_eq!(live.origin().unwrap().task_ids, ["AST_B"]);
}

#[test]
fn a_completed_child_delivers_its_bodies_before_replacing_its_accepted_output() {
    let root = FrameId { value: "root".into() };
    let mut live = Live::new(origin("A"));
    let held = body(&mut live, "invocation", 1, "item_A");
    live.detach(&held);
    let (initial, _, _) = live.commit_started().unwrap();
    live.committed(initial);
    live.enqueue(origin("side"), human("human_side", "When is check-in?")).unwrap();
    assert!(live.next(&root, true).is_none(), "the accepted child cannot be re-invoked");
    live.settle(&held, ToolOutcome::Nothing.into());
    assert!(matches!(live.next(&root, true).unwrap().input, TurnInput::Delivery(_)));
    assert!(!live.pending_for_frame(&root), "the selected final delivery can settle this frame");
    let (delivery, _, _) = live.commit_started().unwrap();
    live.committed(delivery);
    assert!(matches!(live.next(&root, false).unwrap().input, TurnInput::Human(_)));
}

#[test]
fn task_death_is_not_a_final_business_report() {
    let root = FrameId { value: "root".into() };
    let mut live = Live::new(origin("A"));
    let held = body(&mut live, "invocation", 1, "item_A");
    live.detach(&held);
    let (initial, _, _) = live.commit_started().unwrap();
    live.committed(initial);
    assert!(live.lost(&held));
    let TurnInput::Delivery(keys) = live.next(&root, false).unwrap().input else {
        panic!("one owned task-death delivery")
    };
    assert!(live.delivered_outputs(&keys)[0].report.is_none());
}

#[test]
fn canceling_the_task_that_started_a_child_leaves_the_task_now_continuing_it_running() {
    let child = FrameId { value: "child".into() };
    let mut live = Live::new(origin("A"));
    let held_a = body(&mut live, "invocation_A", 1, "item_A");
    live.detach(&held_a);
    let (initial, _, _) = live.commit_started().unwrap();
    live.committed(initial);
    live.enqueue(origin("B"), human("human_B", "Friday works.")).unwrap();
    let turn_b = live.next(&child, false).unwrap();
    assert_eq!(turn_b.origin.task_ids, ["AST_B"]);
    let running_b = body(&mut live, "invocation_B", 2, "item_B");

    let cancelled = live.cancel_tasks(&["AST_A".into()], Some("invocation_B"));

    assert_eq!(cancelled, [held_a], "only A's own detached body is canceled");
    assert!(!live.current_cancelled(), "B keeps the floor in the shared child");
    assert_eq!(live.origin().unwrap().task_ids, ["AST_B"]);
    assert!(live.output_source(&running_b).is_some(), "B's body keeps running");
    let (turn, origin, working) = live.commit_started().unwrap();
    assert_eq!(origin.task_ids, ["AST_B"]);
    assert!(working, "B still owns its running body");
    assert!(live.committed(turn));
}

#[test]
fn a_body_past_the_owned_limit_is_refused_as_retryable_pressure() {
    let mut live = Live::new(origin("A"));
    for id in 0..MAX_LIVE_BODIES as u64 {
        body(&mut live, "invocation", id, &format!("item_{id}"));
    }
    let key = ("invocation".to_owned(), CallId(MAX_LIVE_BODIES as u64));
    live.prepared(key.clone(), ArgumentIdentity::Json(json!({"booking":"over"})));
    let call = FunctionCall {
        id: "item_over".into(),
        call_id: "wire_over".into(),
        name: "reserve".into(),
        arguments: json!({"booking":"over"}).to_string(),
        created_at: std::time::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };

    let refused = live.start_body(key.clone(), FrameId { value: "root".into() }, call).unwrap_err();

    assert_eq!(refused.kind, lk_agents_abi::errors::HostErrorKind::Unavailable);
    assert!(matches!(
        crate::tool_outcome::host_failure(&refused),
        ToolOutcome::ToolError(text) if text == lk_agents_abi::llm::tool_executor::UNAVAILABLE_OUTPUT
    ));
    assert!(live.output_source(&key).is_none(), "a refused body is not owned");
}

#[test]
fn a_closed_owner_joins_its_cancelled_bodies_without_queueing_a_delivery() {
    let mut live = Live::new(origin("A"));
    let held = body(&mut live, "invocation", 1, "item_A");
    assert!(live.detach(&held));
    let (initial, _, _) = live.commit_started().unwrap();
    live.committed(initial);

    assert_eq!(live.close(), std::slice::from_ref(&held));
    assert!(!live.idle(), "the cancelled body is still owned until it ends");
    assert!(!live.settle(&held, ToolOutcome::Cancelled.into()));

    assert!(live.idle(), "a closed owner settles once its last body is joined");
}
