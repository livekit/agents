use super::*;
use crate::audio_recognition::{
    EndOfTurnInfo, EouDetected, EouPredicted, TurnDetectionMode, UserTurnCommitVerdict,
};
use crate::events::{SpeechOutcome, ToolEndStatus};
use crate::trace::InterruptCause;
use lk_agents_abi::telemetry::record::{Number, SpanRecord, Status, Value};
use lk_agents_abi::telemetry::spans::user_turn::{UserTurnOrigin, UserTurnOutcome};
use lk_agents_abi::voice::events::{AgentState, OverlappingSpeechEvent};
use lk_agents_abi::voice::rtc::{ConnectionState, DisconnectReason, ParticipantKind};
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

/// A span the runtime reserved in the session's trace at `at`.
fn reserved<D: SpanDecl>(spans: &SessionSpans, id: u64, at: Ts) -> Reserved<D> {
    let root = spans.root.as_ref().expect("the session is open").context();
    let context = lk_agents_abi::telemetry::record::SpanRef {
        span_id: lk_agents_abi::telemetry::record::SpanId(id),
        ..root
    };
    Reserved::new(context, 1_000_000_000 + at.as_nanos())
}

fn reply(user: Option<lk_agents_abi::MetricsReport>) -> Reply {
    Reply {
        agent: AGENT.into(),
        user,
        realtime: false,
        instructions: Some("be brief".into()),
        user_input: Some("book a table".into()),
        llm: Some(lk_agents_abi::ProviderId {
            label: "openai.LLM".into(),
            model: "gpt-4o".into(),
            provider: "api.openai.com".into(),
        }),
    }
}

fn info_of(identity: &str, sid: &str, kind: ParticipantKind) -> ParticipantInfo {
    ParticipantInfo { identity: identity.into(), sid: sid.into(), kind }
}

fn measurements(records: &[Record]) -> Vec<&lk_agents_abi::telemetry::record::Measurement> {
    records
        .iter()
        .filter_map(|r| match r {
            Record::Measurement(m) => Some(m),
            _ => None,
        })
        .collect()
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
        user.transcript(trace, root, ms(to), 1.0, "final_transcript", text, "req_1", &stt());
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
    let span = spans.speech_started(ms(1_500), speech, reply(Some(user))).expect("a span");
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
    assert_eq!(attr(turn, "lk.pii.instructions"), Some(&s("be brief")));
    assert_eq!(attr(turn, "lk.pii.user_input"), Some(&s("book a table")));
    assert_eq!(attr(turn, "lk.interrupted"), Some(&Value::Bool(false)));
    for absent in
        ["lk.interruption.source", "lk.playout.position", "lk.pii.response.function_calls"]
    {
        assert_eq!(attr(turn, absent), None, "{absent} on a completed pipeline reply");
    }

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
        user.transcript(trace, root, ms(7_000), 7.0, "interim_transcript", "hi", "", &stt());
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
    spans.turns(ms(700), |trace, _, user, _| {
        user.detection(trace, ms(700), &EouDetected::Awaiting, serde_json::Value::Null)
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
    assert_eq!(attr(waits[1], "lk.eou.not_committed_count"), Some(&Value::Int(1)), "empty");
    let detection = one(&records, "eou_detection");
    assert_eq!(parent(&records, detection), "eou_wait");
    assert_eq!(detection.events[0].name, "superseded", "the user resumed while it decided");
    assert!(detection.events[0].attributes.contains(&("lk.eou.outcome".into(), s("user_resumed"))));
    assert_eq!(attr(detection, "lk.eou.probability"), None);
}

// Catches: the close leaving a span for the sweep, ending a parent before its child, or the
// session not last.
#[test]
fn the_close_ends_every_open_span_itself() {
    let mut spans = opened();
    spans.closing(ms(50), lk_agents_abi::voice::events::CloseReason::UserInitiated, true, None);
    spans.hook_issued(ms(60), HookId(1), HookKind::OnExit, AGENT, None, None, None);
    speak(&mut spans, 100, 200, "wait");
    spans.speech_started(ms(150), SpeechId(1), reply(None));
    spans.amd(ms(160), &AmdSpan::Attached(reserved(&spans, 7, ms(160))));
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

// Catches: the queue wait, the interruption's cause or the played position missing from an
// interrupted reply, a later cause overwriting the first, or the e2e latency labeled with the
// GenAI keys instead of the LLM's `model_provider` / `model_name`.
#[test]
fn an_interrupted_reply_records_its_wait_cause_and_playout() {
    let mut spans = opened();
    let speech = SpeechId(1);
    let user =
        lk_agents_abi::MetricsReport { stopped_speaking_at: Some(1.0), ..Default::default() };
    spans.speech_started(ms(100), speech, reply(Some(user)));
    spans.turns(ms(400), |trace, _, _, t| {
        t.scheduled(ms(100), speech);
        t.authorized(ms(350), speech);
        t.authorized(ms(390), speech);
        t.speaking(trace, ms(400), 1.4, speech);
        t.interrupted(speech, InterruptCause::ByAudioActivity);
        t.interrupted(speech, InterruptCause::Host);
        t.played(speech, Duration::from_millis(300), false);
        t.played(speech, Duration::from_millis(1_200), true);
        t.done(trace, ms(1_700), speech, SpeechOutcome::Interrupted, None);
    });
    // Interrupted with no recorded cause: programmatic.
    spans.speech_started(ms(1_800), SpeechId(2), reply(None));
    spans.turns(ms(1_900), |trace, _, _, t| {
        t.done(trace, ms(1_900), SpeechId(2), SpeechOutcome::Interrupted, None)
    });
    let reason = lk_agents_abi::voice::events::CloseReason::UserInitiated;
    spans.observe(ms(2_000), &SessionEvent::Close { reason, error: None });
    let records = spans.drain();
    let turns: Vec<_> = records
        .iter()
        .filter_map(|r| match r {
            Record::Span(s) if s.name == "agent_turn" => Some(s),
            _ => None,
        })
        .collect();
    let turn = turns[0];
    assert_eq!(
        attr(turn, "lk.speech.queue_wait"),
        Some(&Value::Double(0.25)),
        "first authorization"
    );
    assert_eq!(attr(turn, "lk.interrupted"), Some(&Value::Bool(true)));
    assert_eq!(attr(turn, "lk.interruption.source"), Some(&s("audio_activity")));
    assert_eq!(attr(turn, "lk.playout.position"), Some(&Value::Double(1.2)));
    assert_eq!(attr(turns[1], "lk.interruption.source"), Some(&s("programmatic")));
    assert_eq!(attr(turns[1], "lk.playout.position"), None, "nothing played");
    assert_eq!(attr(turns[1], "lk.speech.queue_wait"), None, "never scheduled");
    let e2e: Vec<_> = measurements(&records)
        .into_iter()
        .filter(|m| m.instrument == "lk.agents.turn.e2e_latency")
        .collect();
    let [m] = e2e[..] else { panic!("{records:?}") };
    let Number::Double(v) = m.value else { panic!("{m:?}") };
    assert!((v - 0.4).abs() < 1e-9, "{v}");
    assert_eq!(
        m.labels,
        [("model_provider".into(), s("api.openai.com")), ("model_name".into(), s("gpt-4o"))]
    );
}

// Catches: a realtime turn without its function calls, or with a pipeline reply's input keys.
#[test]
fn a_realtime_reply_records_its_function_calls() {
    let mut spans = opened();
    let speech = SpeechId(1);
    spans.speech_started(ms(100), speech, Reply { realtime: true, ..reply(None) });
    let call = lk_agents_abi::FunctionCall {
        id: "item_1".into(),
        call_id: "call_1".into(),
        arguments: "{}".into(),
        name: "lookup".into(),
        created_at: UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    spans.turns(ms(200), |trace, _, _, t| {
        t.function_call(speech, &call);
        t.done(trace, ms(300), speech, SpeechOutcome::Completed, None);
    });
    spans.speech_started(ms(400), SpeechId(2), Reply { realtime: true, ..reply(None) });
    spans.turns(ms(500), |trace, _, _, t| {
        t.done(trace, ms(500), SpeechId(2), SpeechOutcome::Completed, None)
    });
    let records = close(&mut spans, 1_000);
    let turns = named(&records, "agent_turn");
    let Some(Value::String(calls)) = attr(turns[0], "lk.pii.response.function_calls") else {
        panic!("{:?}", turns[0])
    };
    let calls: serde_json::Value = serde_json::from_str(calls).expect("JSON");
    assert_eq!(calls[0]["call_id"], "call_1");
    assert_eq!(calls[0]["name"], "lookup");
    assert_eq!(calls[0].get("created_at"), None);
    assert_eq!(attr(turns[1], "lk.pii.response.function_calls"), Some(&s("[]")));
    assert_eq!(attr(turns[0], "lk.pii.instructions"), None, "a pipeline reply's");
    assert_eq!(attr(turns[0], "lk.pii.user_input"), None);
}

// Catches: a discarded preemptive generation ending a turn of its own instead of handing it to
// the speech that answers, the handover unmarked, or a lapsed handover adopted by a later speech.
#[test]
fn a_discarded_preemptive_generation_hands_its_turn_over() {
    let mut spans = opened();
    let (discarded, answer) = (SpeechId(1), SpeechId(2));
    spans.speech_started(ms(100), discarded, reply(None));
    spans.turns(ms(200), |_, _, _, t| {
        t.interrupted(discarded, InterruptCause::Superseded);
        t.discarded(discarded);
    });
    spans.speech_started(ms(200), answer, reply(None));
    spans.turns(ms(300), |trace, _, _, t| {
        t.done(trace, ms(300), discarded, SpeechOutcome::Interrupted, None);
        t.done(trace, ms(900), answer, SpeechOutcome::Completed, None);
    });
    // Unclaimed by the end of its step, a handover lapses.
    spans.speech_started(ms(1_000), SpeechId(3), reply(None));
    spans.turns(ms(1_000), |_, _, _, t| t.discarded(SpeechId(3)));
    let early: Vec<_> = spans
        .drain()
        .into_iter()
        .filter_map(|r| match r {
            Record::Span(s) => Some(s),
            _ => None,
        })
        .collect();
    spans.speech_started(ms(1_100), SpeechId(4), reply(None));
    let records = [early, close(&mut spans, 2_000)].concat();
    well_formed(&records);
    let turns = named(&records, "agent_turn");
    let turn = turns
        .iter()
        .find(|t| attr(t, "lk.speech_id") == Some(&s("AS_test_speech_2")))
        .unwrap_or_else(|| panic!("the answer's turn: {turns:?}"));
    assert_eq!(turn.start_unix_nanos, 1_000_000_000 + 100_000_000, "the discarded one's start");
    assert_eq!(attr(turn, "lk.interrupted"), Some(&Value::Bool(false)));
    let names: Vec<_> = turn.events.iter().map(|e| e.name.to_string()).collect();
    assert_eq!(names, ["generation", "preemptive_generation_discarded", "generation"]);
    assert!(turn.events[1].attributes.contains(&("lk.speech_id".into(), s("AS_test_speech_1"))));
    assert!(
        turn.events[2].attributes.contains(&("lk.generation_id".into(), s("AS_test_speech_2_1")))
    );
    assert_eq!(turns.len(), 3, "speeches 3 and 4 have their own turns");
}

// Catches: participant attributes missing or not in the protocol's spelling, the agent's spans
// described as the user, or the room's connection and disconnect events without their keys.
#[test]
fn spans_describe_their_participants_and_the_room() {
    let mut spans = opened();
    let user = info_of("caller", "PA_user", ParticipantKind::Sip);
    spans.participant_info(&user, false);
    spans.participant_info(&info_of("agent", "PA_agent", ParticipantKind::Agent), true);
    spans.connection(ms(10), ConnectionState::Connected);
    spans.participant(ms(20), "caller", None);
    speak(&mut spans, 100, 600, "hi");
    decide(&mut spans, 900, 1, UserTurnCommitVerdict::Empty);
    spans.speech_started(ms(1_000), SpeechId(1), reply(None));
    spans.turns(ms(1_100), |trace, _, _, t| {
        t.speaking(trace, ms(1_100), 2.1, SpeechId(1));
        t.done(trace, ms(1_500), SpeechId(1), SpeechOutcome::Completed, None);
    });
    spans.participant(ms(1_600), "caller", Some(DisconnectReason::ClientInitiated));
    speak(&mut spans, 1_700, 1_800, "gone");
    let records = close(&mut spans, 2_000);
    well_formed(&records);
    let is = |span: &SpanRecord, id: &str, identity: &str, kind: &str| {
        assert_eq!(attr(span, "lk.participant_id"), Some(&s(id)), "{}", span.name);
        assert_eq!(attr(span, "lk.pii.participant_identity"), Some(&s(identity)));
        assert_eq!(attr(span, "lk.participant_kind"), Some(&s(kind)));
    };
    let user_turn = named(&records, "user_turn")[0];
    is(user_turn, "PA_user", "caller", "PARTICIPANT_KIND_SIP");
    is(named(&records, "user_speaking")[0], "PA_user", "caller", "PARTICIPANT_KIND_SIP");
    is(one(&records, "agent_turn"), "PA_agent", "agent", "PARTICIPANT_KIND_AGENT");
    is(one(&records, "agent_speaking"), "PA_agent", "agent", "PARTICIPANT_KIND_AGENT");
    assert_eq!(attr(named(&records, "user_turn")[1], "lk.participant_id"), None, "after it left");
    let session = one(&records, "agent_session");
    let event = |name: &str| session.events.iter().find(|e| e.name == name).expect(name);
    let has = |name: &str, key: &str, value: &str| {
        assert!(event(name).attributes.contains(&(key.into(), s(value))), "{name} {key}");
    };
    has("connection_state_changed", "lk.connection_state", "CONN_CONNECTED");
    has("participant_linked", "lk.participant_id", "PA_user");
    has("participant_linked", "lk.participant_kind", "PARTICIPANT_KIND_SIP");
    has("participant_disconnected", "lk.participant_id", "PA_user");
    has("participant_disconnected", "lk.disconnect_reason", "CLIENT_INITIATED");
}

// Catches: the transcript events or provider request ids lost, an `eou_wait` not re-armed by a
// later trigger, or a prediction recorded without what the detector read.
#[test]
fn a_user_turn_records_its_transcripts_and_detections() {
    let mut spans = opened();
    spans.turns(ms(100), |trace, root, user, _| {
        user.speech_started(trace, root, ms(100), UserTurnOrigin::Speech, &stt());
        user.transcript(trace, root, ms(300), 1.25, "interim_transcript", "boo", "r1", &stt());
        user.transcript(trace, root, ms(500), 1.5, "preflight_transcript", "book", "r1", &stt());
        user.speech_ended(trace, ms(600), endpointing());
        user.detection(trace, ms(600), &EouDetected::Awaiting, serde_json::Value::Null);
        let predicted = EouDetected::Resolved {
            prediction: Some(EouPredicted {
                probability: 0.2,
                threshold: Some(0.5),
                detection_delay: Some(0.04),
            }),
            delay: Duration::from_secs(2),
            from_cache: false,
            language: None,
        };
        let ctx = serde_json::json!({"items": [{"role": "user", "content": ["book"]}]});
        user.detection(trace, ms(700), &predicted, ctx);
        user.transcript(trace, root, ms(800), 1.8, "final_transcript", "book it", "r2", &stt());
        let stt_trigger = Endpointing { source: "stt", delay: 0.5 };
        user.retriggered(trace, ms(800), stt_trigger);
    });
    decide(&mut spans, 1_200, 1, UserTurnCommitVerdict::Reply);
    spans.turns(ms(1_200), |trace, _, user, _| {
        user.retire(trace, ms(1_200), TurnId(1), UserTurnOutcome::Committed)
    });
    let records = close(&mut spans, 2_000);
    well_formed(&records);
    let turn = one(&records, "user_turn");
    assert_eq!(attr(turn, "lk.provider_request_ids"), Some(&Value::Array(vec![s("r1"), s("r2")])));
    let Some(Value::String(events)) = attr(turn, "lk.stt.events") else { panic!("{turn:?}") };
    let events: serde_json::Value = serde_json::from_str(events).expect("JSON");
    assert_eq!(
        events,
        serde_json::json!([
            {"received_at": 1.25, "type": "interim_transcript", "transcript_length": 3},
            {"received_at": 1.5, "type": "preflight_transcript", "transcript_length": 4},
            {"received_at": 1.8, "type": "final_transcript", "transcript_length": 7},
        ])
    );
    let wait = one(&records, "eou_wait");
    assert_eq!(attr(wait, "lk.eou.rearm_count"), Some(&Value::Int(1)));
    assert_eq!(attr(wait, "lk.eou.source"), Some(&s("stt")), "the latest trigger");
    assert_eq!(attr(wait, "lk.eou.not_committed_count"), Some(&Value::Int(0)));
    assert_eq!(wait.events[0].name, "rearmed");
    assert!(wait.events[0].attributes.contains(&("lk.eou.source".into(), s("stt"))));
    let detection = one(&records, "eou_detection");
    assert_eq!(detection.start_unix_nanos, 1_000_000_000 + 600_000_000, "from the request");
    let Some(Value::String(ctx)) = attr(detection, "lk.pii.chat_ctx") else { panic!() };
    assert!(ctx.contains("book"), "{ctx}");
    assert_eq!(attr(detection, "lk.eou.language"), Some(&s("")), "as 1.8.10 writes no language");
    assert_eq!(attr(detection, "lk.eou.from_cache"), Some(&Value::Bool(false)));
    assert_eq!(attr(detection, "lk.eou.probability"), Some(&Value::Double(0.2)));
    assert_eq!(attr(detection, "lk.eou.unlikely_threshold"), Some(&Value::Double(0.5)));
    assert_eq!(attr(detection, "lk.eou.detection_delay"), Some(&Value::Double(0.04)));
    assert_eq!(attr(detection, "lk.eou.endpointing_delay"), Some(&Value::Double(2.0)));
    assert_eq!(attr(detection, "lk.eou.source"), Some(&s("vad")));
    assert!(detection.events.is_empty(), "decided before the wait ended");
}

// Catches: a latency dropped when the model is unknown instead of recorded unlabeled, or an
// undescribed participant written with empty id and kind strings.
#[test]
fn unknown_metadata_is_omitted_not_blank() {
    let mut spans = opened();
    let user = lk_agents_abi::MetricsReport {
        stopped_speaking_at: Some(1.0),
        on_user_turn_completed_delay: Some(0.1),
        ..Default::default()
    };
    spans.speech_started(ms(100), SpeechId(1), Reply { llm: None, ..reply(Some(user)) });
    spans.turns(ms(200), |trace, _, _, t| {
        t.speaking(trace, ms(200), 1.2, SpeechId(1));
        t.done(trace, ms(300), SpeechId(1), SpeechOutcome::Completed, None);
    });
    spans.participant(ms(400), "stranger", None);
    let reason = lk_agents_abi::voice::events::CloseReason::UserInitiated;
    spans.observe(ms(500), &SessionEvent::Close { reason, error: None });
    let records = spans.drain();
    for name in ["lk.agents.turn.e2e_latency", "lk.agents.turn.on_user_turn_completed_delay"] {
        let m = measurements(&records).into_iter().find(|m| m.instrument == name);
        assert!(m.unwrap_or_else(|| panic!("{name}")).labels.is_empty(), "{name}");
    }
    let session = records
        .iter()
        .find_map(|r| match r {
            Record::Span(s) if s.name == "agent_session" => Some(s),
            _ => None,
        })
        .expect("the session");
    let linked = session.events.iter().find(|e| e.name == "participant_linked").expect("linked");
    let keys: Vec<_> = linked.attributes.iter().map(|(k, _)| k.to_string()).collect();
    assert_eq!(keys, ["lk.pii.participant_identity"]);
}

fn context(span: u64) -> SpanRef {
    let trace_id = lk_agents_abi::telemetry::record::TraceId(9);
    SpanRef { trace_id, span_id: lk_agents_abi::telemetry::record::SpanId(span), sampled: true }
}

// Catches: the driver drawing new spans for what `start()` reserved, so the room's connection and
// the first agent's toolsets (run under the reservations) hang off spans that never exist; or
// the first `on_enter` not handed to the host as its current span.
#[test]
fn the_session_opens_the_spans_start_reserved() {
    let now = crate::agent_session::Now { at: ms(500), wall: UNIX_EPOCH + Duration::from_secs(1) };
    let opening = Opening {
        session: Reserved::new(context(1), 1_000_000_000),
        start: Reserved::new(context(2), 1_000_000_000),
        activity: Some(Reserved::new(context(3), 1_200_000_000)),
    };
    let telemetry = SessionTelemetry {
        parent: Some(context(100)),
        opening: Some(opening),
        ..Default::default()
    };
    let mut spans = SessionSpans::new("AS_test", telemetry, now);
    spans.open(ms(500), AGENT);
    assert_eq!(spans.starting(), Some(context(2)));
    let hook = spans.hook_issued(ms(600), HookId(1), HookKind::OnEnter, AGENT, None, None, None);
    spans.hook_finished(ms(700), HookId(1), None, false);
    spans.opened(ms(700));
    let records = close(&mut spans, 1_000);
    well_formed(&records);
    let session = one(&records, "agent_session");
    assert_eq!((session.context, session.parent), (context(1), Some(context(100).span_id)));
    assert_eq!(session.start_unix_nanos, 1_000_000_000, "from start(), not the driver's opening");
    assert_eq!(one(&records, "session_start").context, context(2));
    let activity = one(&records, "start_agent_activity");
    assert_eq!(activity.context, context(3));
    assert_eq!(activity.start_unix_nanos, 1_200_000_000, "from its toolsets' setup");
    assert_eq!(hook, Some(one(&records, "on_enter").context), "the hook runs under its span");
}

// Catches: `agent_session` without the job's dispatch, worker and agent ids, which 1.8.10 repeated
// from `job_entrypoint`.
#[test]
fn the_session_carries_its_jobs_ids() {
    let now = crate::agent_session::Now { at: ms(0), wall: UNIX_EPOCH };
    let job_ids = JobIds {
        dispatch_id: Some("AD_1".into()),
        worker_id: Some("AW_1".into()),
        agent_id: Some("AG_1".into()),
    };
    let mut spans =
        SessionSpans::new("AS_test", SessionTelemetry { job_ids, ..Default::default() }, now);
    spans.open(ms(0), AGENT);
    spans.opened(ms(1));
    let records = close(&mut spans, 10);
    let session = one(&records, "agent_session");
    let string = |key| attr(session, key).cloned();
    assert_eq!(string("lk.dispatch_id"), Some(Value::String("AD_1".into())));
    assert_eq!(string("lk.job.worker_id"), Some(Value::String("AW_1".into())));
    assert_eq!(string("lk.job.agent_id"), Some(Value::String("AG_1".into())));
}

// Catches: a handoff's `update_agent` or arriving activity drawn anew instead of the spans its
// toolset setup ran under, or a reservation applied to an agent it was not made for.
#[test]
fn a_handoff_opens_the_spans_its_toolset_setup_reserved() {
    let mut spans = opened();
    let update = reserved(&spans, 50, ms(50));
    let activity = reserved(&spans, 51, ms(60));
    spans.activation("other", Some(Activation { update, activity }));
    spans.activation("back", Some(Activation { update, activity }));
    let exit =
        spans.hook_issued(ms(100), HookId(1), HookKind::OnExit, AGENT, None, Some("back"), None);
    spans.hook_finished(ms(200), HookId(1), None, false);
    let enter = spans.hook_issued(ms(300), HookId(2), HookKind::OnEnter, "back", None, None, None);
    spans.hook_finished(ms(400), HookId(2), None, false);
    let records = close(&mut spans, 1_000);
    well_formed(&records);
    assert_eq!(one(&records, "update_agent").context, update.context);
    let entry = named(&records, "start_agent_activity");
    let entry = entry.iter().find(|a| parent(&records, a) == "update_agent").expect("the entry");
    assert_eq!(entry.context, activity.context);
    assert_eq!(exit, Some(one(&records, "on_exit").context));
    assert_eq!(enter, Some(one(&records, "on_enter").context));
}

// Catches: stalls recorded only as their own spans, with the session's totals and its
// `event_loop_blocked` events missing; or totals written for a session that never stalled; or
// AMD's `amd` drawn anew instead of the span its classification runs under.
#[test]
fn event_loop_stalls_fold_into_the_session() {
    let mut spans = opened();
    spans.blocked(ms(100), Duration::from_millis(200));
    spans.blocked(ms(200), Duration::from_millis(600));
    let amd = reserved(&spans, 70, ms(300));
    spans.amd(ms(300), &AmdSpan::Attached(amd));
    let records = close(&mut spans, 1_000);
    well_formed(&records);
    let session = one(&records, "agent_session");
    assert_eq!(attr(session, "lk.blocking.count"), Some(&Value::Int(2)));
    assert_eq!(attr(session, "lk.blocking.total_duration"), Some(&Value::Double(0.8)));
    assert_eq!(attr(session, "lk.blocking.max_duration"), Some(&Value::Double(0.6)));
    let events: Vec<_> = session.events.iter().filter(|e| e.name == "event_loop_blocked").collect();
    assert_eq!(events.len(), 2);
    assert!(events[1].attributes.contains(&("lk.blocking.duration".into(), Value::Double(0.6))));
    assert_eq!(one(&records, "amd").context, amd.context);

    let records = close(&mut opened(), 1_000);
    assert_eq!(attr(one(&records, "agent_session"), "lk.blocking.count"), None);
}

// Catches: chat items reported with no link to their turn: the reply's message to its
// `agent_turn`, a typed user message to the `user_turn` it opened.
#[test]
fn each_chat_item_knows_its_turn() {
    let mut spans = opened();
    let turn = spans.speech_started(ms(100), SpeechId(1), reply(None)).expect("a turn");
    let added = |id: &str, role, source| SessionEvent::ConversationItemAdded {
        item: Box::new(lk_agents_abi::ChatItem::Message(lk_agents_abi::ChatMessage::text(
            id, role, "hi", UNIX_EPOCH,
        ))),
        agent: AGENT.into(),
        source,
        visibility: crate::agent::Visibility::Both,
        run: None,
    };
    let reply_source = crate::agent::Source::Reply { speech: SpeechId(1), step: 1 };
    spans.observe(ms(200), &added("item_a", lk_agents_abi::Role::Assistant, reply_source));
    spans.observe(
        ms(300),
        &added("item_u", lk_agents_abi::Role::User, crate::agent::Source::UserTyped),
    );
    let turns = spans.item_turns().clone();
    let records = close(&mut spans, 1_000);
    assert_eq!(turns.get("item_a"), Some(&turn));
    assert_eq!(turns.get("item_u"), Some(&one(&records, "user_turn").context));
}
