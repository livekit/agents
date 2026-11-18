use super::*;

fn refunds() -> ExpertTool {
    ExpertTool::remote("https://back.office/agents/refunds", "refunds")
}

fn floor(tools: &[ExpertTool]) -> Floor<'_> {
    Floor { tools, free: true, task: false }
}

fn opened() -> Experts {
    let mut experts = Experts::default();
    assert!(experts.open("call_1", refunds()));
    experts
}

fn question(request_id: &str) -> Presentation {
    Presentation::Question {
        output_id: "call_1".into(),
        request_id: request_id.into(),
        question: "What email should receive the receipt?".into(),
        verbatim: false,
    }
}

fn ask(experts: &mut Experts, request_id: &str) -> bool {
    experts.ask("call_1", request_id, "What email should receive the receipt?".into(), false, None)
}

fn pending(request_id: &str) -> PendingRequest {
    PendingRequest { output_id: "call_1".into(), request_id: request_id.into() }
}

#[test]
fn an_answered_queued_question_is_withdrawn_without_losing_another_question() {
    let tools = [refunds()];
    let mut experts = opened();
    let reference = InputRequestRef::RemoteTask {
        remote_id: "refunds".into(),
        task: lk_agents_abi::delegation::remote::RemoteTaskRef {
            context_id: "conversation".into(),
            task_id: "refund".into(),
        },
        request_id: "input_1".into(),
    };
    experts.ask("call_1", "r1", "Which day?".into(), false, Some(reference.clone()));
    experts.open("call_2", refunds());
    experts.ask("call_2", "r2", "Which seat?".into(), false, None);
    // Shared admission can recognize an earlier presented occurrence before its live
    // question regains the foreground. It must not be asked again after the handoff.
    experts.reserve(
        reference.clone(),
        lk_agents_abi::ChatMessage::text(
            "human",
            lk_agents_abi::Role::User,
            "Friday",
            std::time::UNIX_EPOCH,
        ),
    );
    assert!(experts.response_queued(&reference));
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Question {
            output_id: "call_2".into(),
            request_id: "r2".into(),
            question: "Which seat?".into(),
            verbatim: false,
        })
    );
}

#[test]
fn a_late_speech_receipt_keeps_the_occurrence_it_actually_presented() {
    let tools = [refunds()];
    let mut experts = opened();
    let reference = |request: &str| InputRequestRef::RemoteTask {
        remote_id: "refunds".into(),
        task: lk_agents_abi::delegation::remote::RemoteTaskRef {
            context_id: "conversation".into(),
            task_id: "refund".into(),
        },
        request_id: request.into(),
    };
    experts.ask("call_1", "r1", "Which email?".into(), false, Some(reference("input_1")));
    experts.next(floor(&tools));
    experts.presenting(SpeechId(1), pending("r1"));

    experts.ask("call_1", "r2", "Which address?".into(), false, Some(reference("input_2")));
    assert_eq!(experts.delivered(SpeechId(1)), Some(reference("input_1")));
    experts.next(floor(&tools));
    experts.presenting(SpeechId(2), pending("r2"));
    assert_eq!(experts.delivered(SpeechId(2)), Some(reference("input_2")));
    assert_eq!(experts.delivered(SpeechId(2)), None, "one receipt presents an occurrence once");
}

#[test]
fn an_admitted_response_waits_for_its_remote_tool_and_cannot_answer_a_newer_request() {
    let mut experts = opened();
    let reference = |request: &str| InputRequestRef::RemoteTask {
        remote_id: "refunds".into(),
        task: lk_agents_abi::delegation::remote::RemoteTaskRef {
            context_id: "conversation".into(),
            task_id: "refund".into(),
        },
        request_id: request.into(),
    };
    let message = lk_agents_abi::ChatMessage::text(
        "human",
        lk_agents_abi::Role::User,
        "Friday",
        std::time::UNIX_EPOCH,
    );
    experts.ask("call_1", "r1", "Which date?".into(), false, Some(reference("input_1")));
    experts.next(floor(&[refunds()]));
    experts.observe(floor(&[]));
    experts.reserve(reference("input_1"), message.clone());
    assert!(experts.responses(floor(&[])).is_empty());
    assert_eq!(experts.responses(floor(&[refunds()])), [(pending("r1"), message.clone())]);
    experts.answer(&pending("r1")).unwrap();
    assert!(experts.responses(floor(&[refunds()])).is_empty());
    experts.ask("call_1", "r2", "Which address?".into(), false, Some(reference("input_2")));
    experts.reserve(reference("input_1"), message);
    assert!(experts.responses(floor(&[refunds()])).is_empty());
    assert!(experts.answer(&pending("r1")).is_err());
}

#[test]
fn an_unspoken_question_is_available_for_representing() {
    let tools = [refunds()];
    let mut experts = opened();
    assert!(ask(&mut experts, "r1"));
    assert_eq!(experts.next(floor(&tools)), Some(question("r1")));
    experts.presenting(SpeechId(1), pending("r1"));
    experts.presentation_ended(SpeechId(1));
    assert_eq!(experts.next(floor(&tools)), Some(question("r1")));
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
            output_id: "call_1".into(),
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
    let lookalike = [ExpertTool::remote("https://elsewhere/agents/refunds", "refunds")];
    assert_eq!(experts.next(floor(&lookalike)), None);
    assert_eq!(experts.next(floor(&[])), None);
    // A different configured agent exposing the same remote tool presents it.
    let replacement = [ExpertTool::remote("https://back.office/agents/refunds", "refunds")];
    assert!(matches!(experts.next(floor(&replacement)), Some(Presentation::Progress { .. })));
}

#[test]
fn a_question_owns_the_reply_destination_and_takes_exactly_one_answer() {
    let tools = [refunds()];
    let mut experts = opened();
    assert_eq!(experts.collect(floor(&tools)), None);
    assert!(ask(&mut experts, "r1"));
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
    let tools = [refunds(), ExpertTool::remote("https://back.office/agents/baggage", "baggage")];
    let mut experts = opened();
    assert!(experts.open("call_2", tools[1].clone()));
    ask(&mut experts, "r1");
    experts.ask("call_2", "b1", "Which bag?".into(), true, None);
    assert!(experts.progress("call_2", 1, "Tracing the bag".into()));
    assert_eq!(experts.next(floor(&tools)), Some(question("r1")));
    // The second question waits; other events still present.
    assert!(matches!(experts.next(floor(&tools)), Some(Presentation::Progress { .. })));
    assert_eq!(experts.next(floor(&tools)), None);
    experts.answer(&pending("r1")).unwrap();
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Question {
            output_id: "call_2".into(),
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
    assert!(!experts.open("call_1", refunds()));
    assert!(experts.progress("call_1", 1, "Working".into()));
    experts.next(floor(&tools));
    assert!(!experts.progress("call_1", 1, "Working".into()));
    ask(&mut experts, "r1");
    assert!(!ask(&mut experts, "r1"));
    let done = ToolOutcome::Value {
        text: "Refunded".into(),
        reply_required: true,
        verbatim: false,
        more: None,
    };
    assert!(experts.outcome("call_1", done.clone()));
    assert!(!experts.outcome("call_1", done.clone()));
    assert!(!experts.outcome("call_9", done.clone()));
    experts.recorded("call_1");
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Outcome { output_id: "call_1".into(), outcome: done })
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
    assert!(ask(&mut experts, "r2"));
    assert_eq!(experts.answer(&pending("r1")), Err(Stale));
    experts.next(floor(&tools));
    // The application cancels: the pending question goes with it, and nothing is presented.
    assert!(experts.cancel("call_1"));
    assert_eq!(experts.answer(&pending("r2")), Err(Stale));
    assert!(!experts.outcome("call_1", ToolOutcome::Cancelled));
    assert_eq!(experts.next(floor(&tools)), None);
}

#[test]
fn a_checkpointed_response_retires_only_its_exact_question_even_after_agent_switches() {
    use lk_agents_abi::delegation::remote::RemoteTaskRef;
    use lk_agents_abi::session::InputRequestRef;
    let tools = [refunds()];
    let mut experts = opened();
    let reference = InputRequestRef::RemoteTask {
        remote_id: "refunds".into(),
        task: RemoteTaskRef { context_id: "trip".into(), task_id: "refund".into() },
        request_id: "refund/input_1".into(),
    };
    assert!(experts.ask("call_1", "r1", "Which booking?".into(), false, Some(reference.clone())));
    experts.next(floor(&tools));
    let mut stale = reference.clone();
    let InputRequestRef::RemoteTask { request_id, .. } = &mut stale else { unreachable!() };
    *request_id = "refund/input_0".into();
    assert!(!experts.response_queued(&stale));
    assert_eq!(experts.collect(floor(&tools)), Some(pending("r1")));
    assert!(experts.response_queued(&reference));
    experts.observe(floor(&[]));
    assert_eq!(
        experts.next(floor(&tools)),
        None,
        "an admitted response cannot prompt its old question again"
    );
    assert!(experts.ask("call_1", "r2", "What budget?".into(), false, Some(stale)));
    assert!(
        !experts.response_queued(&reference),
        "a late admission cannot retire a replacement question"
    );
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Question {
            output_id: "call_1".into(),
            request_id: "r2".into(),
            question: "What budget?".into(),
            verbatim: false,
        })
    );
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
    assert!(!ask(&mut restored, "r1"));
    assert_eq!(restored.next(floor(&tools)), Some(question("r1")));
    assert_eq!(restored.next(floor(&tools)), None);
}

#[test]
fn an_outcome_resolves_the_originating_call_only_once_its_pending_output_is_recorded() {
    let tools = [refunds()];
    let mut experts = opened();
    experts.outcome("call_1", ToolOutcome::Failed("Refund declined".into()));
    assert_eq!(experts.next(floor(&tools)), None);
    experts.recorded("call_1");
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Outcome {
            output_id: "call_1".into(),
            outcome: ToolOutcome::Failed("Refund declined".into())
        })
    );
}

#[test]
fn exact_outputs_keep_order_and_receipts_when_later_progress_and_results_overtake_announcements() {
    use crate::agent_session::{RequestId, SpeechCall};
    let tools = [refunds()];
    let mut experts = opened();
    let first = SpeechCall { id: RequestId(50), speech: SpeechId(50) };
    let second = SpeechCall { id: RequestId(51), speech: SpeechId(51) };
    assert!(experts.exact("call_1", 1, "Reference ABC.".into(), Some(first.clone())));
    for occurrence in 2..12 {
        assert!(experts.progress("call_1", occurrence, format!("obsolete progress {occurrence}")));
    }
    assert!(experts.exact("call_1", 12, "Refund confirmed.".into(), Some(second.clone())));
    assert!(ask(&mut experts, "question"));
    let done = ToolOutcome::Value {
        text: "The business task completed.".into(),
        reply_required: true,
        verbatim: false,
        more: None,
    };
    assert!(experts.outcome("call_1", done.clone()));
    assert_eq!(experts.next(floor(&tools)), None, "source ownership is not recorded yet");
    experts.recorded("call_1");
    assert_eq!(
        experts.next(floor(&[])),
        None,
        "an unrelated agent cannot present exact expert output"
    );
    assert_eq!(
        experts.next(Floor { tools: &tools, free: true, task: true }),
        None,
        "a foreground VoiceTask keeps its floor"
    );
    for (text, call) in [("Reference ABC.", first), ("Refund confirmed.", second)] {
        assert_eq!(
            experts.next(floor(&tools)),
            Some(Presentation::Exact {
                output_id: "call_1".into(),
                text: text.into(),
                call: Some(call),
            })
        );
    }
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Outcome { output_id: "call_1".into(), outcome: done })
    );
    assert_eq!(experts.next(floor(&tools)), None, "an exact output cannot be presented twice");
}

#[test]
fn stable_remote_identity_reaches_a_replacement_agent_after_its_model_tool_is_renamed() {
    let mut experts = Experts::default();
    experts.open("original-output", ExpertTool::remote("flights-primary", "old_flight_tool"));
    assert!(experts.progress("original-output", 1, "Still searching".into()));
    let other_config = [ExpertTool::remote("flights-backup", "old_flight_tool")];
    assert_eq!(experts.next(floor(&other_config)), None);
    let replacement = [ExpertTool::remote("flights-primary", "search_flights")];
    assert!(matches!(experts.next(floor(&replacement)), Some(Presentation::Progress { .. })));
}

#[test]
fn remote_ids_cannot_alias_local_agent_tool_identity() {
    let mut experts = Experts::default();
    experts.open("local-output", ExpertTool::local("helper", "help"));
    assert!(experts.progress("local-output", 1, "Local work".into()));
    let remote_lookalike = [ExpertTool::remote("agent:helper", "help")];
    assert_eq!(experts.next(floor(&remote_lookalike)), None);
    let local = [ExpertTool::local("helper", "help")];
    assert!(matches!(experts.next(floor(&local)), Some(Presentation::Progress { .. })));
}

#[test]
fn a_refused_verbatim_question_keeps_its_reference_and_retries_the_same_occurrence() {
    let tools = [refunds()];
    let mut experts = opened();
    let reference =
        InputRequestRef::Agent { agent_id: "root".into(), request_id: "native-question".into() };
    assert!(experts.ask(
        "call_1",
        "r1",
        "Confirm the exact action?".into(),
        true,
        Some(reference.clone())
    ));
    let first = experts.next(floor(&tools)).unwrap();
    let request = pending("r1");
    experts.defer_question(&request);
    assert_eq!(experts.reference(&request), Some(&reference));
    assert!(experts.waits_for(&reference));
    assert!(
        experts.collect(floor(&tools)).is_none(),
        "an unheard question cannot own the next reply"
    );
    let second = experts.next(floor(&tools)).unwrap();
    assert_eq!(first, second, "retry preserves exact source and question identity");
    assert_eq!(experts.reference(&request), Some(&reference));
}
