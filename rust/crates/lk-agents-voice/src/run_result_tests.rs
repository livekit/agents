use super::*;
use lk_agents_abi::ChatMessage;
use lk_agents_abi::agent::run::RunQuestion;
use std::time::{Duration, SystemTime};

fn capture() -> Runs {
    let mut runs = Runs::default();
    runs.open(&RunSpec { user_input: None, host: None }, false).unwrap();
    runs
}

#[test]
fn capture_waits_for_participation_and_then_for_owner_liveness() {
    let mut runs = capture();
    assert!(runs.decide(false).is_none());
    runs.participated();
    assert!(runs.decide(true).is_none());
    assert!(runs.decide(false).is_some());
    assert!(runs.decide(false).is_none(), "settlement is emitted once");
}

#[test]
fn result_precedence_releases_non_selected_values() {
    let mut runs = capture();
    runs.participated();
    let first = SpeechId(1);
    let last = SpeechId(2);
    runs.outputs.insert(
        first,
        RunOutcome::InputRequired(RunQuestion {
            question: "old?".into(),
            verbatim: true,
            answer_tool: None,
        }),
    );
    runs.speech_ended(first, None);
    assert!(runs.outputs.contains_key(&first));
    runs.speech_ended(last, None);
    assert!(!runs.outputs.contains_key(&first));
    let (open, output) = runs.decide(false).expect("settled");
    assert!(output.is_none());
    assert!(matches!(open.last, Some((id, None)) if id == last));
    assert!(runs.outputs.is_empty());
}

#[test]
fn close_releases_all_result_references_and_settles_once() {
    let mut runs = capture();
    runs.outputs.insert(SpeechId(1), RunOutcome::Failed(RunError::Cancelled));
    assert!(runs.close().is_some());
    assert!(runs.outputs.is_empty());
    assert!(runs.close().is_none());
}

#[test]
fn last_completed_participating_speech_preserves_failure_precedence() {
    let mut runs = capture();
    runs.participated();
    let speech = SpeechId(7);
    runs.outputs.insert(
        speech,
        RunOutcome::InputRequired(RunQuestion {
            question: "ignored by speech error?".into(),
            verbatim: true,
            answer_tool: None,
        }),
    );
    let error = ProviderError::Local(lk_agents_abi::errors::ApiError::new(
        lk_agents_abi::errors::ErrorKind::Server,
        "tts failed",
    ));
    runs.speech_ended(speech, Some(error.clone()));
    let (open, output) = runs.decide(false).expect("settled");
    assert!(runs.outputs.is_empty());
    let (_, result) = open.settle(output, Vec::new());
    let failure = result.expect_err("the speech's failure decides");
    assert_eq!(failure.provider_error, Some(error));
    assert!(matches!(failure.error, RunError::Model(_)));
}

fn reply(id: &str, text: &str, seconds: u64) -> ChatItem {
    let at = SystemTime::UNIX_EPOCH + Duration::from_secs(seconds);
    ChatMessage::text(id, Role::Assistant, text, at).into()
}

/// A settled run keeps its items in the order they happened and ignores items recorded after.
/// https://github.com/livekit/agents/pull/2787 https://github.com/livekit/agents/pull/4925
#[test]
fn terminal_results_keep_assistant_tool_order_despite_late_events() {
    let mut runs = capture();
    runs.participated();
    let text = reply("spoken", "Checking now.", 1);
    let call: ChatItem = lk_agents_abi::FunctionCall {
        id: "call-item".into(),
        call_id: "call-1".into(),
        name: "lookup".into(),
        arguments: "{}".into(),
        created_at: SystemTime::UNIX_EPOCH + Duration::from_secs(2),
        extra: Default::default(),
        group_id: None,
    }
    .into();
    runs.record(&call);
    runs.record(&text);
    let (open, output) = runs.decide(false).unwrap();
    let settled = open.settle(output, vec![text, call]).1.unwrap();
    let snapshot = settled.result.clone();
    runs.record(&reply("spoken", "late overwrite", 0));
    runs.record(&reply("late", "late append", 3));
    assert!(runs.current().is_none());
    assert_eq!(settled.result, snapshot);
    assert_eq!(
        settled.result.new_messages.iter().map(ChatItem::id).collect::<Vec<_>>(),
        ["spoken", "call-item"]
    );
}

#[test]
fn recording_replaces_identity_orders_equal_stamps_and_answers_the_last_reply() {
    let mut runs = capture();
    runs.participated();
    runs.record(&reply("later", "old", 2));
    runs.record(&reply("earlier", "first", 1));
    runs.record(&reply("later", "updated", 2));
    runs.record(&reply("equal", "last", 2));
    let (open, output) = runs.decide(false).expect("settled");
    let (_, result) = open.settle(output, Vec::new());
    let settled = result.expect("completed");
    assert_eq!(
        settled.result.new_messages.iter().map(ChatItem::id).collect::<Vec<_>>(),
        vec!["earlier", "later", "equal"]
    );
    assert!(settled.result.usage.settled);
    assert!(matches!(
        &settled.result.outcome,
        Outcome::Completed { output: Output::Text(text) } if text == "last"
    ));
}

#[test]
fn a_question_is_the_outcome_not_the_preceding_assistant_text() {
    let mut runs = capture();
    runs.participated();
    runs.record(&reply("message", "earlier reply", 0));
    let speech = SpeechId(3);
    let question = RunQuestion { question: "Confirm?".into(), verbatim: true, answer_tool: None };
    runs.outputs.insert(speech, RunOutcome::InputRequired(question.clone()));
    runs.speech_ended(speech, None);
    let (open, output) = runs.decide(false).expect("settled");
    let settled = open.settle(output, Vec::new()).1.expect("asks");
    assert_eq!(settled.result.outcome, Outcome::InputRequired(question));
    assert_eq!(settled.result.verbatim.as_deref(), Some("Confirm?"));
}

#[test]
fn a_last_assistant_message_without_text_is_no_output() {
    let mut runs = capture();
    runs.participated();
    runs.record(&reply("old", "older reply", 0));
    let mut latest = ChatMessage::text("new", Role::Assistant, "", SystemTime::UNIX_EPOCH);
    latest.content.clear();
    runs.record(&latest.into());
    let (open, output) = runs.decide(false).expect("settled");
    let settled = open.settle(output, Vec::new()).1.expect("settles");
    assert_eq!(settled.result.outcome, Outcome::NoOutput);
}

/// The open run's items, as `new_messages()` will list them.
fn recorded(driver: &VoiceDriver) -> Vec<(Role, String)> {
    let open = driver.session.runs.open.as_ref().expect("open");
    open.items
        .iter()
        .filter_map(|item| match item {
            ChatItem::Message(message) => Some((message.role, message.text_content()?)),
            _ => None,
        })
        .collect()
}

// Catches: a voice run's `new_messages()` leaving out the input it was run on, which a text
// run's starts with.
#[test]
fn a_run_records_its_input_first() {
    for mut driver in
        [crate::agent_session::tests::driver(), crate::agent_session::tests::realtime_driver()]
    {
        driver.on_run(Ts::ZERO, RunSpec { user_input: Some("hello".into()), host: None });
        assert_eq!(recorded(&driver).first(), Some(&(Role::User, "hello".to_owned())));
    }
}

#[test]
fn the_run_counts_only_its_own_speeches_requests() {
    use crate::agent_session::machine::activity::scheduling::SpeechKind;
    use crate::agent_session::machine::activity::speech::SpeechSpec;
    use crate::generation::ReplySource;
    let mut driver = crate::agent_session::tests::driver();
    driver.on_run(Ts::ZERO, RunSpec { user_input: None, host: None });
    let speech = driver.start_speech(
        Ts::ZERO,
        SpeechSpec::new(SpeechKind::Say, ReplySource::Text("owned".into()), true),
    );
    driver.start_request(Some(speech), ModelKind::Tts, "cartesia", "sonic");
    driver.start_request(None, ModelKind::Llm, "openai", "gpt");
    assert_eq!(driver.session.runs.usage().expect("open").snapshot().requests, 1);
    assert_eq!(driver.session.usage.snapshot().requests, 2);
}
#[test]
fn no_host_enter_completion_counts_as_participation_without_stranding_capture() {
    let mut driver = crate::agent_session::tests::driver();
    driver.on_run(Ts::ZERO, RunSpec { user_input: None, host: None });
    driver.issue_on_enter(Ts::ZERO);
    driver.run_check();
    let settled = driver
        .settle_step()
        .into_iter()
        .filter(|effect| {
            matches!(effect,
                crate::agent_session::effects::VoiceEffect::Emit(event)
                    if matches!(event, SessionEvent::RunSettled { .. })
            )
        })
        .count();
    assert_eq!(settled, 1);
    assert!(driver.session.runs.current().is_none());
}

#[test]
fn rejected_speech_is_attributed_before_its_immediate_terminal_release() {
    use crate::agent_session::machine::activity::scheduling::{PauseCause, SpeechKind};
    use crate::agent_session::machine::activity::speech::SpeechSpec;
    use crate::generation::ReplySource;
    let mut driver = crate::agent_session::tests::driver();
    driver.on_run(Ts::ZERO, RunSpec { user_input: None, host: None });
    driver.pause_scheduling(Ts::ZERO, PauseCause::Drain);
    let speech = driver.start_speech(
        Ts::ZERO,
        SpeechSpec::new(SpeechKind::Say, ReplySource::Text("hello".into()), true),
    );
    assert!(
        driver.session.runs.names(speech),
        "the rejected admission still determines its run result"
    );
    driver.run_check();
    assert!(driver.session.runs.current().is_none(), "a terminal speech must not remain pending");
    let settled = driver.settle_step().into_iter().filter(|effect| {
        matches!(
            effect,
            crate::agent_session::effects::VoiceEffect::Emit(SessionEvent::RunSettled { .. })
        )
    });
    assert_eq!(settled.count(), 1);
}

#[test]
fn retained_requirements_survive_result_precedence_and_close() {
    let mut runs = capture();
    let requirement = RunInputRequirement {
        invocation_id: "speech-1".into(),
        call: lk_agents_abi::a2a::turn::PendingCall {
            name: "flight".into(),
            call_id: "call-1".into(),
            answer_tool: Some("answer".into()),
        },
        question: "Which day?".into(),
        verbatim: true,
    };
    runs.retain_input_requirements(std::slice::from_ref(&requirement));
    // A coordinated step and its terminal envelope report the same request. Keep one
    // correlated obligation even when success/failure projections overlap.
    runs.retain_input_requirements(std::slice::from_ref(&requirement));
    runs.speech_ended(SpeechId(1), None);
    runs.speech_ended(SpeechId(2), None);
    assert_eq!(runs.close().unwrap().input_requirements, [requirement]);
}

#[test]
fn hold_membership_survives_waiting_to_granted_and_ends_on_release() {
    use crate::agent_session::{HoldId, IdleHolds, IdleWaiter};
    let run = RunId(3);
    let hold = HoldId(4);
    let mut idle = IdleHolds::default();
    idle.ask(IdleWaiter {
        run: Some(run),
        hold: None,
        speech: Some(SpeechId(2)),
        grant: Some(hold),
        call: None,
    });
    assert!(idle.has_run_work(run));
    assert!(!idle.has_run_work(RunId(99)));
    assert_eq!(idle.admit().len(), 1);
    assert!(idle.has_run_work(run), "granting must not lose attribution");
    assert!(idle.release(hold));
    assert!(!idle.has_run_work(run));
    assert!(!idle.release(hold), "duplicate release is inert");
}

#[test]
fn withdrawing_a_pending_hold_removes_its_run_obligation() {
    use crate::agent_session::{HoldId, IdleHolds, IdleWaiter};
    let run = RunId(3);
    let mut idle = IdleHolds::default();
    let speech = SpeechId(2);
    idle.ask(IdleWaiter {
        run: Some(run),
        hold: None,
        speech: Some(speech),
        grant: Some(HoldId(4)),
        call: None,
    });
    assert_eq!(idle.withdraw(speech).len(), 1);
    assert!(!idle.has_run_work(run));
}

#[test]
fn unrelated_speech_neither_extends_the_run_nor_replaces_its_result() {
    use crate::agent_session::machine::activity::scheduling::SpeechKind;
    use crate::agent_session::machine::activity::speech::SpeechSpec;
    use crate::generation::ReplySource;
    let mut driver = crate::agent_session::tests::driver();
    driver.on_run(Ts::ZERO, RunSpec { user_input: None, host: None });
    let host = driver.start_speech(
        Ts::ZERO,
        SpeechSpec::new(SpeechKind::Say, ReplySource::Text("owned".into()), true),
    );
    let unrelated = driver.start_speech(
        Ts::ZERO,
        SpeechSpec {
            voice_run:
                crate::agent_session::machine::activity::scheduling::VoiceRunParticipation::Outside,
            ..SpeechSpec::new(SpeechKind::Say, ReplySource::Text("incidental".into()), true)
        },
    );
    driver.session.runs.outputs.insert(
        host,
        RunOutcome::InputRequired(RunQuestion {
            question: "owned?".into(),
            verbatim: true,
            answer_tool: None,
        }),
    );
    driver.session.runs.outputs.insert(
        unrelated,
        RunOutcome::InputRequired(RunQuestion {
            question: "unrelated?".into(),
            verbatim: true,
            answer_tool: None,
        }),
    );
    driver.release_speech(Ts::ZERO, host);
    driver.run_check();
    assert!(driver.session.runs.current().is_none(), "unrelated speech cannot extend the run");
    driver.release_speech(Ts::ZERO, unrelated);
    let settled: Vec<_> = driver
        .settle_step()
        .into_iter()
        .filter_map(|effect| match effect {
            crate::agent_session::effects::VoiceEffect::Emit(SessionEvent::RunSettled {
                result: Ok(settled),
                ..
            }) => Some(settled.result.outcome),
            _ => None,
        })
        .collect();
    assert!(
        matches!(settled.as_slice(), [Outcome::InputRequired(ask)] if ask.question == "owned?")
    );
    assert!(driver.session.runs.outputs.is_empty());
}
