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
    asked("call_1", request_id, "What email should receive the receipt?")
}

fn asked(output_id: &str, request_id: &str, question: &str) -> Presentation {
    Presentation::Questions(vec![Question {
        output_id: output_id.into(),
        request_id: request_id.into(),
        question: question.into(),
        verbatim: false,
    }])
}

fn result(output_id: &str, outcome: ToolOutcome) -> Presentation {
    Presentation::Outcomes(vec![Resolution { output_id: output_id.into(), outcome }])
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
    assert_eq!(experts.next(floor(&tools)), Some(asked("call_2", "r2", "Which seat?")));
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
    experts.presenting(SpeechId(1), vec![pending("r1")]);

    experts.ask("call_1", "r2", "Which address?".into(), false, Some(reference("input_2")));
    assert_eq!(experts.delivered(SpeechId(1)), [reference("input_1")]);
    experts.next(floor(&tools));
    experts.presenting(SpeechId(2), vec![pending("r2")]);
    assert_eq!(experts.delivered(SpeechId(2)), [reference("input_2")]);
    assert_eq!(experts.delivered(SpeechId(2)), [], "one receipt presents an occurrence once");
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
    experts.presenting(SpeechId(1), vec![pending("r1")]);
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
        Some(Presentation::Progress(vec![Update {
            output_id: "call_1".into(),
            text: "Refund approved, issuing it".into()
        }]))
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
    assert!(matches!(experts.next(floor(&replacement)), Some(Presentation::Progress(_))));
}

#[test]
fn a_question_owns_the_reply_destination_and_takes_exactly_one_answer() {
    let tools = [refunds()];
    let mut experts = opened();
    assert_eq!(experts.collect(floor(&tools)), []);
    assert!(ask(&mut experts, "r1"));
    // Recorded before it is presented: nothing to forward until it is asked.
    assert_eq!(experts.collect(floor(&tools)), []);
    assert_eq!(experts.next(floor(&tools)), Some(question("r1")));
    assert_eq!(experts.collect(floor(&tools)), [pending("r1")]);
    assert_eq!(experts.answer(&pending("r1")), Ok(()));
    assert_eq!(experts.answer(&pending("r1")), Err(Stale));
    assert_eq!(experts.collect(floor(&tools)), []);
}

#[test]
fn a_voice_task_owns_the_interaction_and_the_question_is_presented_again_after_it() {
    let tools = [refunds()];
    let mut experts = opened();
    ask(&mut experts, "r1");
    experts.next(floor(&tools));
    let task = Floor { task: true, ..floor(&tools) };
    assert!(experts.progress("call_1", 1, "Still working".into()));
    assert_eq!(experts.collect(task), []);
    assert_eq!(experts.next(task), None);
    // The task returned its caller: the same request is restored, then answered.
    assert_eq!(experts.next(floor(&tools)), Some(question("r1")));
    assert_eq!(experts.collect(floor(&tools)), [pending("r1")]);
}

#[test]
fn a_question_asked_before_switching_to_an_agent_without_its_tool_is_suspended() {
    let tools = [refunds()];
    let mut experts = opened();
    ask(&mut experts, "r1");
    experts.next(floor(&tools));
    // The replacement cannot interact with the expert: its turns are not forwarded.
    experts.observe(floor(&[]));
    assert_eq!(experts.collect(floor(&[])), []);
    // Back to an eligible agent: the question is re-presented before any answer is forwarded.
    assert_eq!(experts.collect(floor(&tools)), []);
    assert_eq!(experts.next(floor(&tools)), Some(question("r1")));
    assert_eq!(experts.collect(floor(&tools)), [pending("r1")]);
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
    assert_eq!(experts.next(floor(&tools)), Some(result("call_1", done)));
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
    assert_eq!(experts.collect(floor(&tools)), [pending("r1")]);
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
    assert_eq!(experts.next(floor(&tools)), Some(asked("call_1", "r2", "What budget?")));
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
        Some(result("call_1", ToolOutcome::Failed("Refund declined".into())))
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
    assert_eq!(experts.next(floor(&tools)), Some(result("call_1", done)));
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
    assert!(matches!(experts.next(floor(&replacement)), Some(Presentation::Progress(_))));
}

#[test]
fn remote_ids_cannot_alias_local_agent_tool_identity() {
    let mut experts = Experts::default();
    experts.open("local-output", ExpertTool::local("helper", "help"));
    assert!(experts.progress("local-output", 1, "Local work".into()));
    let remote_lookalike = [ExpertTool::remote("agent:helper", "help")];
    assert_eq!(experts.next(floor(&remote_lookalike)), None);
    let local = [ExpertTool::local("helper", "help")];
    assert!(matches!(experts.next(floor(&local)), Some(Presentation::Progress(_))));
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
        experts.collect(floor(&tools)).is_empty(),
        "an unheard question cannot own the next reply"
    );
    let second = experts.next(floor(&tools)).unwrap();
    assert_eq!(first, second, "retry preserves exact source and question identity");
    assert_eq!(experts.reference(&request), Some(&reference));
}

fn exports() -> ExpertTool {
    ExpertTool::remote("https://back.office/agents/exports", "exports")
}

fn export_question(output_id: &str, request_id: &str, question: &str) -> Question {
    Question {
        output_id: output_id.into(),
        request_id: request_id.into(),
        question: question.into(),
        verbatim: false,
    }
}

fn exports_pending(output_id: &str, request_id: &str) -> PendingRequest {
    PendingRequest { output_id: output_id.into(), request_id: request_id.into() }
}

/// Two exports are started; each asks before the frontend gets the floor.
fn two_exports_asking(sales: &str, stock: &str) -> Experts {
    let mut experts = Experts::default();
    experts.open("sales", exports());
    experts.open("stock", exports());
    experts.ask("sales", "s1", sales.into(), false, None);
    experts.ask("stock", "k1", stock.into(), false, None);
    experts
}

#[test]
fn questions_waiting_together_are_asked_together_and_each_takes_a_reply() {
    let tools = [exports()];
    let mut experts = two_exports_asking("Normal or high priority?", "Normal or high priority?");
    // The frontend asks both at the first opportunity: neither waits on the other's answer,
    // so the person's "high" has two candidates and response routing can ask which one.
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Questions(vec![
            export_question("sales", "s1", "Normal or high priority?"),
            export_question("stock", "k1", "Normal or high priority?"),
        ]))
    );
    assert_eq!(
        experts.collect(floor(&tools)),
        [exports_pending("sales", "s1"), exports_pending("stock", "k1")]
    );
    assert_eq!(experts.next(floor(&tools)), None, "an asked question is not asked again");
}

#[test]
fn a_partial_answer_leaves_the_other_question_open_without_asking_it_again() {
    let tools = [exports()];
    let mut experts = two_exports_asking("Normal or high priority?", "Which quarter?");
    experts.next(floor(&tools));
    // "High" was routed to the sales export only.
    experts.person_turn();
    experts.answer(&exports_pending("sales", "s1")).unwrap();
    assert_eq!(experts.collect(floor(&tools)), [exports_pending("stock", "k1")]);
    assert_eq!(experts.next(floor(&tools)), None, "the open quarter question is not repeated");
    // Then "Q3" answers the stock export.
    experts.answer(&exports_pending("stock", "k1")).unwrap();
    assert_eq!(experts.collect(floor(&tools)), []);
}

#[test]
fn a_question_arriving_while_another_is_open_is_asked_at_the_next_free_floor() {
    let tools = [exports()];
    let mut experts = Experts::default();
    experts.open("sales", exports());
    experts.open("stock", exports());
    experts.ask("sales", "s1", "Normal or high priority?".into(), false, None);
    experts.next(floor(&tools));
    experts.ask("stock", "k1", "Which quarter?".into(), false, None);
    let busy = Floor { free: false, ..floor(&tools) };
    assert_eq!(experts.next(busy), None, "a question still waits for the floor");
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Questions(vec![export_question("stock", "k1", "Which quarter?")])),
        "an unanswered question does not hold back another task's question"
    );
    assert_eq!(
        experts.collect(floor(&tools)),
        [exports_pending("sales", "s1"), exports_pending("stock", "k1")]
    );
}

#[test]
fn an_unheard_group_is_offered_again_together_and_its_receipt_presents_each_source() {
    let tools = [exports()];
    let reference = |request: &str| InputRequestRef::Agent {
        agent_id: "desk".into(),
        request_id: request.into(),
    };
    let mut experts = Experts::default();
    experts.open("sales", exports());
    experts.open("stock", exports());
    experts.ask("sales", "s1", "Normal or high priority?".into(), false, Some(reference("s")));
    experts.ask("stock", "k1", "Which quarter?".into(), false, Some(reference("k")));
    let group = experts.next(floor(&tools));
    let both = vec![exports_pending("sales", "s1"), exports_pending("stock", "k1")];
    experts.presenting(SpeechId(1), both.clone());
    experts.presentation_ended(SpeechId(1));
    assert_eq!(experts.collect(floor(&tools)), [], "unheard questions take no reply");
    assert_eq!(experts.next(floor(&tools)), group);
    experts.presenting(SpeechId(2), both);
    assert_eq!(experts.delivered(SpeechId(2)), [reference("s"), reference("k")]);
}

#[test]
fn a_verbatim_question_is_asked_alone_and_word_for_word() {
    let tools = [exports()];
    let mut experts = Experts::default();
    experts.open("sales", exports());
    experts.open("stock", exports());
    experts.ask("sales", "s1", "Say CONFIRM to export.".into(), true, None);
    experts.ask("stock", "k1", "Which quarter?".into(), false, None);
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Questions(vec![Question {
            verbatim: true,
            ..export_question("sales", "s1", "Say CONFIRM to export.")
        }]))
    );
    experts.person_turn();
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Questions(vec![export_question("stock", "k1", "Which quarter?")]))
    );
}

#[test]
fn a_burst_of_progress_across_tasks_is_one_summary_behind_exact_output() {
    let tools = [exports()];
    let mut experts = Experts::default();
    experts.open("sales", exports());
    experts.open("stock", exports());
    experts.progress("sales", 1, "Collecting rows".into());
    experts.exact("stock", 1, "Reference 42.".into(), None);
    experts.progress("stock", 2, "Collecting stock".into());
    experts.progress("sales", 2, "Rendering the PDF".into());
    experts.recorded("stock");
    // The stock update waits behind its own exact output; the sales updates coalesce.
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Progress(vec![Update {
            output_id: "sales".into(),
            text: "Rendering the PDF".into()
        }]))
    );
    assert!(matches!(experts.next(floor(&tools)), Some(Presentation::Exact { .. })));
    experts.progress("sales", 3, "Uploading".into());
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Progress(vec![
            Update { output_id: "stock".into(), text: "Collecting stock".into() },
            Update { output_id: "sales".into(), text: "Uploading".into() },
        ]))
    );
}

fn value(text: &str) -> ToolOutcome {
    ToolOutcome::Value { text: text.into(), reply_required: true, verbatim: false, more: None }
}

fn resolution(output_id: &str, outcome: ToolOutcome) -> Resolution {
    Resolution { output_id: output_id.into(), outcome }
}

/// `count` exports tasks, each started and recorded by one frontend turn.
fn exports_running(count: usize) -> Experts {
    let mut experts = Experts::default();
    for task in 0..count {
        let id = format!("export_{task}");
        experts.open(&id, exports());
        experts.recorded(&id);
    }
    experts
}

#[test]
fn results_ready_together_are_one_presentation_up_to_the_next_ordered_output() {
    let tools = [exports()];
    let mut experts = exports_running(4);
    // Every task finishes while the frontend is busy; export_1 also sent an exact reference
    // and export_3's result is verbatim.
    experts.outcome("export_0", value("Sales exported"));
    experts.exact("export_1", 1, "Reference 42.".into(), None);
    experts.outcome("export_2", value("Stock exported"));
    experts.outcome("export_1", value("Margins exported"));
    let verbatim = ToolOutcome::Value {
        text: "Say: all done.".into(),
        reply_required: true,
        verbatim: true,
        more: None,
    };
    experts.outcome("export_3", verbatim.clone());
    // Results never overtake exact output that arrived before them.
    assert_eq!(experts.next(floor(&tools)), Some(result("export_0", value("Sales exported"))));
    assert!(matches!(experts.next(floor(&tools)), Some(Presentation::Exact { .. })));
    // The two results ready behind it are one reply; each still resolves its own call.
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Outcomes(vec![
            resolution("export_2", value("Stock exported")),
            resolution("export_1", value("Margins exported")),
        ]))
    );
    // Verbatim text is never merged into a summary.
    assert_eq!(experts.next(floor(&tools)), Some(result("export_3", verbatim)));
    assert_eq!(experts.next(floor(&tools)), None, "each result is presented once");
}

#[test]
fn an_unheard_announcement_overtaken_by_a_question_is_withdrawn_and_others_keep_their_update() {
    let tools = [exports()];
    let mut experts = exports_running(2);
    experts.progress("export_0", 1, "Still collecting sales rows".into());
    experts.progress("export_1", 1, "Collecting stock".into());
    let Some(Presentation::Progress(updates)) = experts.next(floor(&tools)) else {
        panic!("progress is announced first");
    };
    experts.announcing(SpeechId(1), updates);
    // While that announcement is still being generated, export_0 asks.
    experts.ask("export_0", "q", "Which region?".into(), false, None);
    assert_eq!(experts.overtaken("export_0"), [SpeechId(1)]);
    assert_eq!(experts.overtaken("export_1"), [SpeechId(1)]);
    experts.withdraw_announcement(SpeechId(1));
    experts.presentation_ended(SpeechId(1));
    assert_eq!(
        experts.overtaken("export_0"),
        [],
        "a withdrawn announcement is not overtaken twice"
    );
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Questions(vec![export_question("export_0", "q", "Which region?")])),
        "the stale progress is not spoken before or after the question"
    );
    // The stock update waits for the person's turn, then is announced once.
    assert_eq!(experts.next(floor(&tools)), None);
    experts.person_turn();
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Progress(vec![Update {
            output_id: "export_1".into(),
            text: "Collecting stock".into()
        }]))
    );
    assert_eq!(experts.next(floor(&tools)), None);
}

#[test]
fn an_unheard_announcement_overtaken_by_its_result_is_withdrawn() {
    let tools = [exports()];
    let mut experts = exports_running(1);
    experts.progress("export_0", 1, "Still rendering".into());
    let Some(Presentation::Progress(updates)) = experts.next(floor(&tools)) else {
        panic!("progress is announced first");
    };
    experts.announcing(SpeechId(1), updates);
    experts.outcome("export_0", value("Exported"));
    assert_eq!(experts.overtaken("export_0"), [SpeechId(1)]);
    experts.withdraw_announcement(SpeechId(1));
    experts.presentation_ended(SpeechId(1));
    assert_eq!(experts.next(floor(&tools)), Some(result("export_0", value("Exported"))));
    assert_eq!(experts.next(floor(&tools)), None, "the finished task's progress is not re-queued");
}

fn expert(name: &str) -> ExpertTool {
    ExpertTool::remote(&format!("https://back.office/agents/{name}"), name)
}

fn trip_question(output_id: &str, question: &str) -> Question {
    export_question(output_id, &format!("{output_id}_q"), question)
}

fn trip_pending(output_id: &str) -> PendingRequest {
    exports_pending(output_id, &format!("{output_id}_q"))
}

/// Flights, payments and two hotel searches each ask before the frontend gets the floor.
fn five_questions() -> (Vec<ExpertTool>, Experts) {
    let tools = vec![expert("flights"), expert("hotels"), expert("payments"), expert("cars")];
    let mut experts = Experts::default();
    for (output_id, tool, question) in [
        ("flight", "flights", "What date would you like to fly?"),
        ("hotel", "hotels", "What is your nightly budget?"),
        ("payment", "payments", "Should I charge the card on file?"),
        ("hotel_2", "hotels", "How many rooms?"),
        ("car", "cars", "Which car size?"),
    ] {
        experts.open(output_id, expert(tool));
        experts.ask(output_id, &format!("{output_id}_q"), question.into(), false, None);
    }
    (tools, experts)
}

#[test]
fn five_waiting_questions_are_paced_one_expert_at_a_time_in_arrival_order() {
    let (tools, mut experts) = five_questions();
    let open = floor(&tools);
    assert_eq!(
        experts.next(open),
        Some(Presentation::Questions(vec![trip_question(
            "flight",
            "What date would you like to fly?"
        )])),
        "unrelated questions are not one utterance"
    );
    assert_eq!(experts.next(open), None, "the person gets a chance to answer first");
    experts.person_turn();
    experts.answer(&trip_pending("flight")).unwrap();
    assert_eq!(
        experts.next(open),
        Some(Presentation::Questions(vec![
            trip_question("hotel", "What is your nightly budget?"),
            trip_question("hotel_2", "How many rooms?"),
        ])),
        "the same expert's questions are asked together"
    );
    experts.person_turn();
    experts.answer(&trip_pending("hotel")).unwrap();
    experts.answer(&trip_pending("hotel_2")).unwrap();
    assert_eq!(
        experts.next(open),
        Some(Presentation::Questions(vec![trip_question(
            "payment",
            "Should I charge the card on file?"
        )]))
    );
    experts.person_turn();
    experts.answer(&trip_pending("payment")).unwrap();
    assert_eq!(
        experts.next(open),
        Some(Presentation::Questions(vec![trip_question("car", "Which car size?")]))
    );
}

#[test]
fn one_prompt_asks_at_most_three_of_an_experts_questions() {
    let tools = [exports()];
    let mut experts = exports_running(4);
    for task in 0..4 {
        let id = format!("export_{task}");
        experts.ask(&id, "q", format!("Format for export {task}?"), false, None);
    }
    let Some(Presentation::Questions(first)) = experts.next(floor(&tools)) else {
        panic!("questions are asked");
    };
    assert_eq!(first.len(), 3);
    assert_eq!(experts.next(floor(&tools)), None, "the fourth waits for the person's turn");
    experts.person_turn();
    for q in &first {
        experts.answer(&exports_pending(&q.output_id, "q")).unwrap();
    }
    assert_eq!(
        experts.next(floor(&tools)),
        Some(Presentation::Questions(vec![export_question(
            "export_3",
            "q",
            "Format for export 3?"
        )]))
    );
}

#[test]
fn a_passed_question_stays_open_is_not_repeated_at_once_and_newer_tasks_do_not_starve_it() {
    let tools = [expert("flights"), expert("hotels"), expert("payments")];
    let open = floor(&tools);
    let mut experts = Experts::default();
    experts.open("flight", expert("flights"));
    experts.ask("flight", "flight_q", "What date would you like to fly?".into(), false, None);
    let flight =
        Presentation::Questions(vec![trip_question("flight", "What date would you like to fly?")]);
    assert_eq!(experts.next(open), Some(flight.clone()));
    // The person changes topic: the question is passed, not answered.
    experts.person_turn();
    assert_eq!(experts.next(open), None, "a passed question is not asked again at once");
    assert_eq!(experts.collect(open), [trip_pending("flight")], "it stays open for an answer");
    // Two newer tasks ask.
    experts.open("hotel", expert("hotels"));
    experts.ask("hotel", "hotel_q", "What is your nightly budget?".into(), false, None);
    experts.open("payment", expert("payments"));
    experts.ask("payment", "payment_q", "Should I charge the card?".into(), false, None);
    assert_eq!(
        experts.next(open),
        Some(Presentation::Questions(vec![trip_question("hotel", "What is your nightly budget?")]))
    );
    experts.person_turn();
    experts.answer(&trip_pending("hotel")).unwrap();
    // One prompt later the passed question is due again, ahead of the newer payment task.
    assert_eq!(experts.next(open), Some(flight));
    experts.person_turn();
    assert_eq!(
        experts.next(open),
        Some(Presentation::Questions(vec![trip_question("payment", "Should I charge the card?")]))
    );
    experts.person_turn();
    experts.answer(&trip_pending("payment")).unwrap();
    // Passed twice: open for an answer, but not asked a third time.
    assert_eq!(experts.next(open), None);
    assert_eq!(experts.collect(open), [trip_pending("flight")]);
    experts.answer(&trip_pending("flight")).unwrap();
    assert_eq!(experts.collect(open), []);
}
