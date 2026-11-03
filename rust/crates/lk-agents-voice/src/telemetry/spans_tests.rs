use super::*;
use crate::audio_recognition::{EndOfTurnInfo, TurnDetectionMode, UserTurnCommitVerdict};
use crate::events::{EotPredictionEvent, SpeechOutcome, ToolEndStatus};
use lk_agents_abi::telemetry::record::{SpanRecord, Status, Value};
use lk_agents_abi::telemetry::spans::user_turn::{UserTurnOrigin, UserTurnOutcome};
use lk_agents_abi::voice::events::{AgentState, OverlappingSpeechEvent};
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

const AGENT: &str = "front";

fn ms(n: u64) -> Ts {
    Ts::from_nanos(n * 1_000_000)
}

fn opened() -> SessionSpans {
    let now = crate::agent_session::Now { at: Ts::ZERO, wall: UNIX_EPOCH + Duration::from_secs(1) };
    let mut spans = SessionSpans::new("AS_test", SessionTelemetry::default(), now);
    spans.open(Ts::ZERO, AGENT);
    spans.opened(Ts::ZERO);
    spans
}

fn stt() -> Stt {
    Stt { model: Some("nova-3".into()), provider: Some("deepgram".into()) }
}

fn endpointing() -> Endpointing {
    Endpointing { source: "vad", delay: 0.5 }
}

fn info(turn: u64, transcript: &str) -> EndOfTurnInfo {
    EndOfTurnInfo {
        turn_id: TurnId(turn),
        owner: TurnDetectionMode::Vad,
        stt_enabled: true,
        skip_reply: false,
        new_transcript: transcript.into(),
        transcript_confidence: 0.9,
        metrics: Default::default(),
        backchannel_over_agent: false,
        late: false,
        snapshot: Default::default(),
    }
}

/// The user speaks from `from` to `to` and says `text`.
fn speak(spans: &mut SessionSpans, from: u64, to: u64, text: &str) {
    spans.turns(ms(from), |trace, root, user, _| {
        user.speech_started(trace, root, ms(from), UserTurnOrigin::Speech, &stt());
        user.transcript(trace, root, ms(to), 1.0, true, text, "req_1", &stt());
        user.speech_ended(trace, ms(to), endpointing());
    });
}

fn decide(spans: &mut SessionSpans, at: u64, turn: u64, verdict: UserTurnCommitVerdict) {
    spans.turns(ms(at), |trace, root, user, _| {
        user.decided(trace, root, ms(at), &info(turn, "hello"), verdict, &stt())
    });
}

fn close(spans: &mut SessionSpans, at: u64) -> Vec<SpanRecord> {
    let reason = lk_agents_abi::voice::events::CloseReason::UserInitiated;
    spans.observe(ms(at), &SessionEvent::Close { reason, error: None });
    spans
        .drain()
        .into_iter()
        .filter_map(|r| match r {
            Record::Span(s) => Some(s),
            _ => None,
        })
        .collect()
}

fn named<'a>(spans: &'a [SpanRecord], name: &str) -> Vec<&'a SpanRecord> {
    spans.iter().filter(|s| s.name == name).collect()
}

fn one<'a>(spans: &'a [SpanRecord], name: &str) -> &'a SpanRecord {
    let found = named(spans, name);
    assert_eq!(found.len(), 1, "one {name} in {:?}", names(spans));
    found[0]
}

fn names(spans: &[SpanRecord]) -> Vec<String> {
    spans.iter().map(|s| s.name.to_string()).collect()
}

fn parent<'a>(spans: &'a [SpanRecord], child: &SpanRecord) -> &'a str {
    let id = child.parent.expect("a parent");
    spans.iter().find(|s| s.context.span_id == id).map_or("?", |s| &s.name)
}

fn attr<'a>(span: &'a SpanRecord, key: &str) -> Option<&'a Value> {
    span.attributes.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn s(v: &str) -> Value {
    Value::String(v.into())
}

/// Every span ended once, within its parent unless declared to outlive it, and none swept.
fn well_formed(spans: &[SpanRecord]) {
    let outlives = [OnEnter::NAME];
    for span in spans {
        assert_eq!(attr(span, "lk.cancelled"), None, "{} reached the close sweep", span.name);
        assert_eq!(
            spans.iter().filter(|o| o.context.span_id == span.context.span_id).count(),
            1,
            "{} ended once",
            span.name
        );
        let Some(p) = span.parent.and_then(|id| spans.iter().find(|s| s.context.span_id == id))
        else {
            continue;
        };
        assert!(span.start_unix_nanos >= p.start_unix_nanos, "{} starts in {}", span.name, p.name);
        if !outlives.contains(&&*span.name) {
            assert!(span.end_unix_nanos <= p.end_unix_nanos, "{} ends in {}", span.name, p.name);
        }
    }
}

// Catches: a reply's turn not a direct child of the session, its speaking outside it, a tool
// ending after its turn, generation events at the turn's start rather than each step's, or a
// speech id that repeats across sessions.
#[test]
fn a_pipeline_turn_with_a_tool_call() {
    let mut spans = opened();
    speak(&mut spans, 100, 900, "book a table");
    decide(&mut spans, 1_400, 1, UserTurnCommitVerdict::Reply);
    let hook = HookId(1);
    spans.hook_issued(
        ms(1_400),
        hook,
        HookKind::OnUserTurnCompleted,
        AGENT,
        Some(TurnId(1)),
        None,
        None,
    );
    spans.hook_finished(ms(1_450), hook, None, false);
    let user = lk_agents_abi::MetricsReport {
        stopped_speaking_at: Some(1.9),
        end_of_turn_delay: Some(0.5),
        ..Default::default()
    };
    let speech = SpeechId(3);
    let span = spans.speech_started(ms(1_500), speech, AGENT, Some(user)).expect("a span");
    spans.turns(ms(1_600), |trace, _, _, t| t.speaking(trace, ms(1_600), 2.6, speech));
    spans.turns(ms(1_700), |_, _, _, t| t.tool_started(speech, "call_1"));
    let stopped =
        SessionEvent::AgentStateChanged { old: AgentState::Speaking, new: AgentState::Thinking };
    spans.observe(ms(1_800), &stopped);
    spans.turns(ms(2_000), |trace, _, _, t| t.step(trace, ms(2_000), speech, 2));
    spans.turns(ms(2_100), |trace, _, _, t| {
        t.done(trace, ms(2_100), speech, SpeechOutcome::Completed, None)
    });
    let ended = ToolLifecycle::Ended {
        id: "u".into(),
        call_id: "call_1".into(),
        message: None,
        status: ToolEndStatus::Done,
    };
    spans.observe(ms(2_300), &SessionEvent::ToolExecutionUpdated(ended));
    let records = close(&mut spans, 3_000);
    well_formed(&records);

    let turn = one(&records, "agent_turn");
    assert_eq!(turn.context, span, "the runtime's work nests under this span");
    assert_eq!(parent(&records, turn), "agent_session");
    assert_eq!(parent(&records, one(&records, "agent_speaking")), "agent_turn");
    assert_eq!(attr(turn, "lk.speech_id"), Some(&s("AS_test_speech_3")));
    assert_eq!(turn.end_unix_nanos, 1_000_000_000 + 2_300_000_000, "ends with its tool");
    let generations: Vec<_> =
        turn.events.iter().map(|e| (e.time_unix_nanos, e.attributes.clone())).collect();
    assert_eq!(generations[0].0, 1_000_000_000 + 1_500_000_000);
    assert_eq!(generations[1].0, 1_000_000_000 + 2_000_000_000, "at the step's start");
    assert!(
        generations[1].1.contains(&("lk.parent_generation_id".into(), s("AS_test_speech_3_1")))
    );
    assert_eq!(attr(turn, "lk.generation_count"), Some(&Value::Int(2)));
    let Some(Value::Double(e2e)) = attr(turn, "lk.e2e_latency") else { panic!("{turn:?}") };
    assert!((e2e - 0.7).abs() < 1e-9, "first audio after the user stopped: {e2e}");
    assert_eq!(attr(turn, "lk.end_of_turn_delay"), Some(&Value::Double(0.5)));

    let user = one(&records, "user_turn");
    assert_eq!(parent(&records, user), "agent_session", "a sibling of the agent turn");
    assert_eq!(attr(user, "lk.user_turn.origin"), Some(&s("speech")));
    assert_eq!(attr(user, "lk.user_turn.outcome"), Some(&s("committed")));
    assert_eq!(attr(user, "gen_ai.request.model"), Some(&s("nova-3")));
    let hook = one(&records, "on_user_turn_completed");
    assert_eq!(parent(&records, hook), "user_turn");
    assert_eq!(user.end_unix_nanos, hook.end_unix_nanos, "the turn ends when its hook returns");
    assert_eq!(attr(one(&records, "eou_wait"), "lk.eou.outcome"), Some(&s("committed")));
}

// Catches: a backchannel verdict dropped from `user_speaking`, a verdict written as a native
// boolean, or an overlap ended without one left unmarked.
#[test]
fn every_overlap_verdict_is_written_as_a_string() {
    let mut spans = opened();
    let verdict = |is_interruption| {
        Arc::new(OverlappingSpeechEvent {
            created_at: 0.0,
            detected_at: 0.0,
            is_interruption,
            agent_ended: false,
            total_duration: 0.2,
            prediction_duration: 0.05,
            detection_delay: 0.1,
            overlap_started_at: None,
            speech_input: None,
            probabilities: None,
            probability: if is_interruption { 0.9 } else { 0.1 },
            num_requests: 1,
        })
    };
    for (from, outcome) in [(100, Some(false)), (1_000, Some(true)), (2_000, None)] {
        spans.turns(ms(from), |trace, root, user, _| {
            user.speech_started(trace, root, ms(from), UserTurnOrigin::Speech, &stt())
        });
        match outcome {
            Some(v) => spans.observe(ms(from + 100), &SessionEvent::OverlappingSpeech(verdict(v))),
            None => {
                spans.turns(ms(from + 100), |_, _, user, _| user.overlap_ended());
            }
        }
        spans.turns(ms(from + 300), |trace, _, user, _| {
            user.speech_ended(trace, ms(from + 300), endpointing())
        });
    }
    let records = close(&mut spans, 5_000);
    well_formed(&records);
    let verdicts: Vec<_> = named(&records, "user_speaking")
        .iter()
        .map(|s| attr(s, "lk.is_interruption").cloned())
        .collect();
    assert_eq!(verdicts, [Some(s("false")), Some(s("true")), Some(s("false"))]);
    assert_eq!(
        attr(named(&records, "user_speaking")[1], "lk.interruption.probability"),
        Some(&Value::Double(0.9))
    );
}

// Catches: a handoff's exit and entry not grouped under one `update_agent` naming both agents,
// the entry's activity not covering its `on_enter`, or a raised hook recorded without its
// exception.
#[test]
fn a_handoff_is_one_update_agent_and_a_raised_hook_records_its_exception() {
    let mut spans = opened();
    spans.hook_issued(ms(100), HookId(1), HookKind::OnExit, AGENT, None, Some("back"), None);
    let error = lk_agents_abi::errors::HostError {
        kind: "ValueError".into(),
        message: "no table".into(),
        host_ref: None,
        traceback: Some("Traceback (most recent call last): ...".into()),
    };
    spans.hook_finished(ms(200), HookId(1), Some(HookFailure::Raised(&error)), false);
    spans.hook_issued(ms(300), HookId(2), HookKind::OnEnter, "back", None, None, None);
    spans.hook_finished(ms(400), HookId(2), Some(HookFailure::TimedOut), false);
    let records = close(&mut spans, 1_000);
    well_formed(&records);
    let update = one(&records, "update_agent");
    assert_eq!(attr(update, "lk.previous_agent_label"), Some(&s(AGENT)));
    assert_eq!(attr(update, "lk.agent_label"), Some(&s("back")));
    assert_eq!(parent(&records, one(&records, "drain_agent_activity")), "update_agent");
    let activity = named(&records, "start_agent_activity");
    let entry = activity.iter().find(|a| parent(&records, a) == "update_agent").expect("the entry");
    let enter =
        named(&records, "on_enter").into_iter().find(|e| e.parent == Some(entry.context.span_id));
    assert!(entry.end_unix_nanos >= enter.expect("its on_enter").end_unix_nanos);
    assert!(update.end_unix_nanos <= 1_000_000_000 + 400_000_000, "the entry ends the handoff");
    let exit = one(&records, "on_exit");
    assert_eq!(exit.status, Status::Error("no table".into()));
    assert_eq!(attr(exit, "error.type"), Some(&s("ValueError")));
    assert_eq!(exit.events[0].name, "exception");
    let traceback = s("Traceback (most recent call last): ...");
    assert!(exit.events[0].attributes.contains(&("exception.stacktrace".into(), traceback)));
}

// Catches: a user turn opened by anything but its first evidence or not backdated to it, or a
// verdict without its outcome.
#[test]
fn a_user_turn_for_each_origin_and_outcome() {
    use UserTurnCommitVerdict as V;
    let mut spans = opened();
    let verdicts = [
        (V::TooFewWords, "too_few_words"),
        (V::Empty, "empty"),
        (V::RealtimeBackchannel, "backchannel"),
        (V::Dropped { append_user_message: false }, "dropped"),
        (V::LateTranscript, "committed"),
    ];
    let mut expected = Vec::new();
    for (i, (verdict, outcome)) in verdicts.into_iter().enumerate() {
        let at = 1_000 * (i as u64 + 1);
        speak(&mut spans, at, at + 300, "words");
        decide(&mut spans, at + 500, i as u64 + 1, verdict);
        expected.push(("speech", outcome));
    }
    // A committed turn with no hook to run ends as it is answered.
    spans.turns(ms(6_600), |trace, _, user, _| {
        user.retire(trace, ms(6_600), TurnId(5), UserTurnOutcome::Committed)
    });
    spans.turns(ms(7_000), |trace, root, user, _| {
        user.transcript(trace, root, ms(7_000), 7.0, false, "hi", "", &stt());
        user.abandon(trace, ms(7_100), UserTurnOutcome::Cleared);
        user.speech_started(trace, root, ms(7_200), UserTurnOrigin::ServerSpeech, &stt());
        user.abandon(trace, ms(7_300), UserTurnOutcome::TranscriptionTimeout);
    });
    expected.push(("transcript", "cleared"));
    expected.push(("server_speech", "transcription_timeout"));
    let typed = lk_agents_abi::ChatMessage::text("i", lk_agents_abi::Role::User, "hi", UNIX_EPOCH);
    let item = SessionEvent::ConversationItemAdded {
        item: Box::new(lk_agents_abi::ChatItem::Message(typed)),
        agent: AGENT.into(),
        source: crate::agent::Source::UserTyped,
        visibility: crate::agent::Visibility::Both,
        run: None,
    };
    spans.observe(ms(7_400), &item);
    expected.push(("text", "committed"));
    speak(&mut spans, 8_000, 8_100, "left open");
    expected.push(("speech", "closed"));
    let records = close(&mut spans, 9_000);
    well_formed(&records);
    let turns: Vec<_> = named(&records, "user_turn")
        .iter()
        .map(|t| {
            (attr(t, "lk.user_turn.origin").cloned(), attr(t, "lk.user_turn.outcome").cloned())
        })
        .collect();
    let expected: Vec<_> = expected.iter().map(|(o, e)| (Some(s(o)), Some(s(e)))).collect();
    assert_eq!(turns, expected);
    let first = named(&records, "user_turn")[0];
    assert_eq!(first.start_unix_nanos, 1_000_000_000 + 1_000_000_000, "the VAD's speech start");
}

// Catches: speech resumed before the decision opening a second turn, or the wait not ending
// `user_resumed` with its event.
#[test]
fn a_resumed_turn_stays_one_turn() {
    let mut spans = opened();
    speak(&mut spans, 100, 600, "i want");
    let prediction = EotPredictionEvent {
        probability: 0.2,
        threshold: 0.5,
        inference_duration: 0.05,
        delay: Duration::from_millis(500),
    };
    spans.turns(ms(700), |trace, _, user, _| {
        user.prediction(trace, ms(700), &prediction, endpointing())
    });
    speak(&mut spans, 800, 1_200, "a table");
    decide(&mut spans, 1_700, 1, UserTurnCommitVerdict::Empty);
    let records = close(&mut spans, 2_000);
    well_formed(&records);
    let turn = one(&records, "user_turn");
    assert_eq!(attr(turn, "lk.eou.resume_count"), Some(&Value::Int(1)));
    assert_eq!(named(&records, "user_speaking").len(), 2);
    let waits = named(&records, "eou_wait");
    assert_eq!(attr(waits[0], "lk.eou.outcome"), Some(&s("user_resumed")));
    assert_eq!(waits[0].events[0].name, "user_resumed");
    assert_eq!(attr(waits[0], "lk.eou.not_committed_count"), Some(&Value::Int(1)));
    assert_eq!(parent(&records, one(&records, "eou_detection")), "eou_wait");
}

// Catches: the close leaving a span for the sweep, ending a parent before its child, or the
// session not last.
#[test]
fn the_close_ends_every_open_span_itself() {
    let mut spans = opened();
    spans.closing(ms(50), lk_agents_abi::voice::events::CloseReason::UserInitiated, true, None);
    spans.hook_issued(ms(60), HookId(1), HookKind::OnExit, AGENT, None, None, None);
    speak(&mut spans, 100, 200, "wait");
    spans.speech_started(ms(150), SpeechId(1), AGENT, None);
    spans.amd(ms(160), None);
    spans.keyterm_requested(ms(170), None);
    let records = close(&mut spans, 1_000);
    well_formed(&records);
    assert_eq!(records.last().map(|s| s.name.to_string()), Some("agent_session".into()));
    assert_eq!(parent(&records, one(&records, "drain_agent_activity")), "session_close");
    assert_eq!(attr(one(&records, "user_turn"), "lk.user_turn.outcome"), Some(&s("closed")));
    for name in ["on_exit", "agent_turn", "amd", "keyterm_detection", "session_close"] {
        one(&records, name);
    }
    assert!(spans.drain().is_empty(), "nothing after the close");
}
