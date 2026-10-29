use super::*;

fn refunds() -> ExpertTool {
    ExpertTool::new("https://back.office/agents/refunds/", "refunds")
}

fn floor(tools: &[ExpertTool]) -> Floor<'_> {
    Floor { tools, free: true, task: false }
}

fn opened() -> Experts {
    let mut experts = Experts::default();
    assert!(experts.open("call_1", "dlg_1", refunds()));
    experts
}

fn question(request_id: &str) -> Presentation {
    Presentation::Question {
        call_id: "call_1".into(),
        request_id: request_id.into(),
        question: "What email should receive the receipt?".into(),
        verbatim: false,
    }
}

fn ask(experts: &mut Experts, request_id: &str) -> Asked {
    experts.ask("call_1", request_id, "What email should receive the receipt?".into(), false)
}

fn pending(request_id: &str) -> PendingRequest {
    PendingRequest { call_id: "call_1".into(), request_id: request_id.into() }
}

#[test]
fn progress_waits_for_the_floor_and_superseded_progress_is_coalesced() {
    let tools = [refunds()];
    let mut experts = opened();
    assert!(experts.progress("call_1", 1, "Looking up the booking".into()));
    assert!(experts.progress("call_1", 2, "Refund approved, issuing it".into()));
    let busy = Floor { free: false, ..floor(&tools) };
    assert_eq!(experts.next(busy), None);
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Progress {
            call_id: "call_1".into(),
            text: "Refund approved, issuing it".into()
        })
    );
    assert_eq!(experts.next(floor(&tools)), None);
}

#[test]
fn events_reach_only_an_agent_exposing_the_configured_remote_tool() {
    let mut experts = opened();
    assert!(experts.progress("call_1", 1, "Working on it".into()));
    // The same model-visible name on another endpoint is not this expert's tool.
    let lookalike = [ExpertTool::new("https://elsewhere/agents/refunds", "refunds")];
    assert_eq!(experts.next(floor(&lookalike)), None);
    assert_eq!(experts.next(floor(&[])), None);
    // A different configured agent exposing the same remote tool presents it.
    let replacement = [ExpertTool::new("https://back.office/agents/refunds", "refunds")];
    assert!(matches!(experts.next(floor(&replacement)), Some(Presentation::Progress { .. })));
}

#[test]
fn a_question_owns_the_reply_destination_and_takes_exactly_one_answer() {
    let tools = [refunds()];
    let mut experts = opened();
    assert_eq!(experts.collect(floor(&tools)), None);
    assert_eq!(ask(&mut experts, "r1"), Asked::Queued { superseded: None });
    // Recorded before it is presented: nothing to forward until it is asked.
    assert_eq!(experts.collect(floor(&tools)), None);
    assert_eq!(experts.next(floor(&tools)), Some(question("r1")));
    assert_eq!(experts.collect(floor(&tools)), Some(pending("r1")));
    assert_eq!(experts.answer(&pending("r1")), Ok(()));
    assert_eq!(experts.answer(&pending("r1")), Err(Stale));
    assert_eq!(experts.collect(floor(&tools)), None);
}

#[test]
fn only_one_expert_question_owns_the_reply_destination_at_a_time() {
    let tools = [refunds(), ExpertTool::new("https://back.office/agents/baggage", "baggage")];
    let mut experts = opened();
    assert!(experts.open("call_2", "dlg_2", tools[1].clone()));
    ask(&mut experts, "r1");
    experts.ask("call_2", "b1", "Which bag?".into(), true);
    assert!(experts.progress("call_2", 1, "Tracing the bag".into()));
    assert_eq!(experts.next(floor(&tools)), Some(question("r1")));
    // The second question waits; other events still present.
    assert!(matches!(experts.next(floor(&tools)), Some(Presentation::Progress { .. })));
    assert_eq!(experts.next(floor(&tools)), None);
    experts.answer(&pending("r1")).unwrap();
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Question {
            call_id: "call_2".into(),
            request_id: "b1".into(),
            question: "Which bag?".into(),
            verbatim: true
        })
    );
}

#[test]
fn a_voice_task_owns_the_interaction_and_the_question_is_presented_again_after_it() {
    let tools = [refunds()];
    let mut experts = opened();
    ask(&mut experts, "r1");
    experts.next(floor(&tools));
    let task = Floor { task: true, ..floor(&tools) };
    assert!(experts.progress("call_1", 1, "Still working".into()));
    assert_eq!(experts.collect(task), None);
    assert_eq!(experts.next(task), None);
    // The task returned its caller: the same request is restored, then answered.
    assert_eq!(experts.next(floor(&tools)), Some(question("r1")));
    assert_eq!(experts.collect(floor(&tools)), Some(pending("r1")));
}

#[test]
fn a_question_asked_before_switching_to_an_agent_without_its_tool_is_suspended() {
    let tools = [refunds()];
    let mut experts = opened();
    ask(&mut experts, "r1");
    experts.next(floor(&tools));
    // The replacement cannot interact with the expert: its turns are not forwarded.
    experts.observe(floor(&[]));
    assert_eq!(experts.collect(floor(&[])), None);
    // Back to an eligible agent: the question is re-presented before any answer is forwarded.
    assert_eq!(experts.collect(floor(&tools)), None);
    assert_eq!(experts.next(floor(&tools)), Some(question("r1")));
    assert_eq!(experts.collect(floor(&tools)), Some(pending("r1")));
}

#[test]
fn duplicate_events_are_never_presented_twice_nor_resolve_the_call_twice() {
    let tools = [refunds()];
    let mut experts = opened();
    assert!(!experts.open("call_1", "dlg_again", refunds()));
    assert!(experts.progress("call_1", 1, "Working".into()));
    experts.next(floor(&tools));
    assert!(!experts.progress("call_1", 1, "Working".into()));
    ask(&mut experts, "r1");
    assert_eq!(ask(&mut experts, "r1"), Asked::Duplicate);
    let done = ToolOutcome::Value {
        text: "Refunded".into(),
        reply_required: true,
        verbatim: false,
        more: None,
    };
    assert_eq!(
        experts.outcome("call_1", done.clone()),
        Settled::Queued { superseded: Some("r1".into()) }
    );
    assert_eq!(experts.outcome("call_1", done.clone()), Settled::Duplicate);
    assert_eq!(experts.outcome("call_9", done.clone()), Settled::NotOwned);
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Outcome { call_id: "call_1".into(), outcome: done })
    );
    assert_eq!(experts.next(floor(&tools)), None);
}

#[test]
fn a_settled_or_superseded_question_cannot_receive_a_late_answer() {
    let tools = [refunds()];
    let mut experts = opened();
    ask(&mut experts, "r1");
    experts.next(floor(&tools));
    // The expert asks again: the first question can no longer be answered.
    assert_eq!(ask(&mut experts, "r2"), Asked::Queued { superseded: Some("r1".into()) });
    assert_eq!(experts.answer(&pending("r1")), Err(Stale));
    experts.next(floor(&tools));
    // The application cancels: the pending question goes with it, and nothing is presented.
    assert_eq!(experts.cancel("call_1"), Some("r2".into()));
    assert_eq!(experts.answer(&pending("r2")), Err(Stale));
    assert_eq!(experts.outcome("call_1", ToolOutcome::Cancelled), Settled::Duplicate);
    assert_eq!(experts.next(floor(&tools)), None);
}

#[test]
fn retained_records_restore_without_presenting_a_delivered_occurrence_again() {
    let tools = [refunds()];
    let mut experts = opened();
    experts.progress("call_1", 1, "Working".into());
    experts.next(floor(&tools));
    ask(&mut experts, "r1");
    let saved = serde_json::to_value(&experts).unwrap();
    let mut restored: Experts = serde_json::from_value(saved).unwrap();
    // A replayed delivery after the restore is a duplicate; the queued question is presented once.
    assert!(!restored.progress("call_1", 1, "Working".into()));
    assert_eq!(ask(&mut restored, "r1"), Asked::Duplicate);
    assert_eq!(restored.next(floor(&tools)), Some(question("r1")));
    assert_eq!(restored.delegation_id("call_1"), Some("dlg_1"));
    assert_eq!(restored.next(floor(&tools)), None);
}
