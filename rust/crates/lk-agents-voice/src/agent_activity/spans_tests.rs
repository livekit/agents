use crate::agent::{Source, Visibility};
use crate::agent_activity::scheduling::{SpeechId, SpeechKind};
use crate::agent_session::room::RoomLinkConfig;
use crate::agent_session::{AmdSpan, Close, HookKind, Now, SessionParts, VoiceDriver};
use crate::audio_recognition::{
    EndOfTurnInfo, EouDetected, EouPredicted, TurnDetectionMode, TurnId, UserTurnCommitVerdict,
};
use crate::events::SpeechOutcome;
use crate::telemetry::{
    Activation, Endpointing, EouSource, HookFailure, JobIds, Opening, Reply, Reserved,
    SessionTelemetry, Stt, Trace, UserTurnSpan,
};
use crate::trace::InterruptCause;
use lk_agents_abi::Ts;
use lk_agents_abi::telemetry::decl::SpanDecl;
use lk_agents_abi::telemetry::record::{Number, Record, SpanRecord, SpanRef, Status, Value};
use lk_agents_abi::telemetry::spans::session::OnEnter;
use lk_agents_abi::telemetry::spans::user_turn::{UserTurnOrigin, UserTurnOutcome};
use lk_agents_abi::voice::events::{AgentState, OverlappingSpeechEvent};
use lk_agents_abi::voice::room_io::{ParticipantInfo, ParticipantLinked, ParticipantUnlinked};
use lk_agents_abi::voice::room_owner::RoomIoEvent;
use lk_agents_abi::voice::rtc::{ConnectionState, DisconnectReason, ParticipantKind};
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

const AGENT: &str = "front";

fn ms(n: u64) -> Ts {
    Ts::from_nanos(n * 1_000_000)
}

/// A session's decision core with what the activity owns of its user turns: the one being heard
/// and the decided ones waiting for their hook.
struct Session {
    d: VoiceDriver,
    heard: Option<UserTurnSpan>,
    decided: std::collections::BTreeMap<u64, UserTurnSpan>,
}

impl Session {
    /// The step at `n` ms.
    fn at(&mut self, n: u64) -> &mut Trace {
        self.d.telemetry.step(ms(n));
        &mut self.d.telemetry
    }

    /// The heard turn, opened by evidence at `n` ms when none is.
    fn hear(&mut self, n: u64, origin: UserTurnOrigin) -> &mut UserTurnSpan {
        if self.heard.is_none() {
            self.d.telemetry.step(ms(n));
            let participant = self.d.session.room.linked_participant();
            let trace = &mut self.d.telemetry;
            let turn = UserTurnSpan::open(trace, ms(n), origin, participant, &stt());
            self.heard = Some(turn);
        }
        self.heard.as_mut().expect("heard")
    }

    fn user_speaks(&mut self, n: u64, origin: UserTurnOrigin) {
        self.hear(n, origin);
        let participant = self.d.session.room.linked_participant();
        let turn = self.heard.as_mut().expect("heard");
        turn.speech_started(&mut self.d.telemetry, ms(n), participant);
    }

    fn speech_ended(&mut self, n: u64, endpointing: Endpointing) {
        self.d.telemetry.step(ms(n));
        if let Some(turn) = &mut self.heard {
            turn.speech_ended(&mut self.d.telemetry, ms(n), endpointing);
        }
    }

    /// The turn decided at `n` ms: ended now unless committed.
    fn decide(&mut self, n: u64, turn: u64, verdict: UserTurnCommitVerdict) {
        let info = info(turn, "hello");
        let mut span = match self.heard.take() {
            Some(span) => span,
            None => {
                let trace = self.at(n);
                UserTurnSpan::open(trace, ms(n), UserTurnOrigin::Transcript, None, &stt())
            }
        };
        let trace = self.at(n);
        match span.decide(trace, &info, verdict) {
            UserTurnOutcome::Committed => {
                if let Some(context) = span.context(trace) {
                    self.d.session.item_turns.user_turn(TurnId(turn), context);
                }
                self.decided.insert(turn, span);
            }
            outcome => span.end(trace, outcome),
        }
    }

    /// A decided turn ends without its hook.
    fn retire(&mut self, n: u64, turn: u64, outcome: UserTurnOutcome) {
        let mut span = self.decided.remove(&turn).expect("decided");
        span.end(self.at(n), outcome);
    }

    /// The turn being heard ends undecided.
    fn abandon(&mut self, n: u64, outcome: UserTurnOutcome) {
        if let Some(mut span) = self.heard.take() {
            span.end(self.at(n), outcome);
        }
    }

    /// Speech `id` is scheduled at `n` ms and its first generation opens its turn.
    fn speech(&mut self, n: u64, id: u64, reply: Reply) -> Option<SpanRef> {
        self.at(n);
        let (speech, state) = self.d.new_speech(Some(SpeechId(id)), true);
        let admitted = self.d.schedule_speech(
            ms(n),
            speech,
            state,
            SpeechKind::GenerateReply,
            true,
            false,
            false,
        );
        assert!(admitted, "the speech is queued");
        self.d.agent_turn_started(speech, reply)
    }

    /// Speech `id`'s turn.
    fn turn(&mut self, id: u64) -> &mut telemetry::AgentTurnSpan {
        self.d.activity.agent_turn(SpeechId(id)).expect("its turn")
    }

    /// Speech `id` is done at `n` ms with `outcome`, and released.
    fn release(&mut self, n: u64, id: u64, outcome: SpeechOutcome) {
        self.at(n);
        self.d.agent_turn_done(SpeechId(id), outcome, None);
        self.d.release_speech(ms(n), SpeechId(id));
    }

    /// The records ended so far.
    fn drain(&mut self) -> Vec<Record> {
        self.d.telemetry.drain()
    }

    /// The close: every user turn the harness owns ends `closed`, then the session's. Every
    /// record since the last drain.
    fn close_records(&mut self, n: u64) -> Vec<Record> {
        self.at(n);
        let mut owned: Vec<UserTurnSpan> = self.heard.take().into_iter().collect();
        owned.extend(std::mem::take(&mut self.decided).into_values());
        for mut turn in owned {
            turn.end(&mut self.d.telemetry, UserTurnOutcome::Closed);
        }
        self.d.close_spans();
        self.drain()
    }

    /// [`Session::close_records`]' spans.
    fn close(&mut self, n: u64) -> Vec<SpanRecord> {
        spans_of(self.close_records(n))
    }
}

use crate::telemetry;

fn spans_of(records: Vec<Record>) -> Vec<SpanRecord> {
    records
        .into_iter()
        .filter_map(|r| match r {
            Record::Span(s) => Some(s),
            _ => None,
        })
        .collect()
}

/// A session built at `at` (one second on the wall clock) with `telemetry`, hooks hosted.
fn session_with(telemetry: SessionTelemetry, at: Ts) -> Session {
    let room = RoomLinkConfig {
        close_on_disconnect: false,
        delete_room_on_close: false,
        room: "r".into(),
    };
    let parts = SessionParts {
        agent: AGENT.into(),
        session_id: "AS_test".into(),
        max_unrecoverable_errors: 3,
        room_link: Some(room),
        telemetry,
        ..Default::default()
    };
    let now = Now { at, wall: UNIX_EPOCH + Duration::from_secs(1) };
    let d = crate::agent_session::tests::driver_with(parts, Some(Duration::from_secs(5)), now);
    Session { d, heard: None, decided: Default::default() }
}

/// A session opened at the zero instant, its opening done.
fn opened() -> Session {
    let mut session = session_with(SessionTelemetry::default(), Ts::ZERO);
    session.d.session_opened();
    session
}

/// A span the runtime reserved in the session's trace at `at`.
fn reserved<D: SpanDecl>(session: &Session, id: u64, at: Ts) -> Reserved<D> {
    let root = session.d.session.span.as_ref().expect("the session is open");
    let root = session.d.telemetry.context(root).expect("recording");
    let context = SpanRef { span_id: lk_agents_abi::telemetry::record::SpanId(id), ..root };
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
    Endpointing { source: EouSource::Vad, delay: 0.5 }
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
fn speak(s: &mut Session, from: u64, to: u64, text: &str) {
    s.user_speaks(from, UserTurnOrigin::Speech);
    s.hear(to, UserTurnOrigin::Speech).transcript(1.0, "final_transcript", text, "req_1");
    s.speech_ended(to, endpointing());
}

fn room(s: &mut Session, n: u64, ev: RoomIoEvent) {
    s.at(n);
    s.d.on_room_io(ms(n), ev);
}

fn linked(identity: &str) -> RoomIoEvent {
    let participant = ParticipantLinked { identity: identity.into() };
    RoomIoEvent::ParticipantLinked { participant, outbound_sip: false }
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

fn st(v: &str) -> Value {
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
    let mut s = opened();
    speak(&mut s, 100, 900, "book a table");
    s.decide(1_400, 1, UserTurnCommitVerdict::Reply);
    let turn = s.decided.get_mut(&1).expect("committed");
    turn.hook_issued(&mut s.d.telemetry, AGENT);
    s.at(1_450);
    let mut turn = s.decided.remove(&1).expect("committed");
    turn.hook_finished(&mut s.d.telemetry, None, false);
    let user = lk_agents_abi::MetricsReport {
        stopped_speaking_at: Some(1.9),
        end_of_turn_delay: Some(0.5),
        ..Default::default()
    };
    let span = s.speech(1_500, 3, reply(Some(user))).expect("a span");
    s.at(1_600);
    s.d.agent_turn_speaking(SpeechId(3), 2.6);
    s.at(1_700);
    s.turn(3).tool_started("call_1");
    s.at(1_800);
    s.d.agent_state_span_event(AgentState::Speaking, AgentState::Thinking);
    s.at(2_000);
    let turn = s.d.activity.agent_turn(SpeechId(3)).expect("its turn");
    turn.step(&mut s.d.telemetry, 2);
    s.release(2_100, 3, SpeechOutcome::Completed);
    assert!(s.d.activity.tools.turns.iter().any(|t| t.speech() == SpeechId(3)), "its tool runs");
    s.at(2_300);
    s.d.agent_turn_tool_ended("call_1");
    assert!(s.d.activity.tools.turns.is_empty(), "the last tool ended the turn");
    let records = s.close(3_000);
    well_formed(&records);

    let turn = one(&records, "agent_turn");
    assert_eq!(turn.context, span, "the runtime's work nests under this span");
    assert_eq!(parent(&records, turn), "agent_session");
    assert_eq!(parent(&records, one(&records, "agent_speaking")), "agent_turn");
    assert_eq!(attr(turn, "lk.speech_id"), Some(&st("AS_test_speech_3")));
    assert_eq!(turn.end_unix_nanos, 1_000_000_000 + 2_300_000_000, "ends with its tool");
    let generations: Vec<_> =
        turn.events.iter().map(|e| (e.time_unix_nanos, e.attributes.clone())).collect();
    assert_eq!(generations[0].0, 1_000_000_000 + 1_500_000_000);
    assert_eq!(generations[1].0, 1_000_000_000 + 2_000_000_000, "at the step's start");
    assert!(
        generations[1].1.contains(&("lk.parent_generation_id".into(), st("AS_test_speech_3_1")))
    );
    assert_eq!(attr(turn, "lk.generation_count"), Some(&Value::Int(2)));
    let Some(Value::Double(e2e)) = attr(turn, "lk.e2e_latency") else { panic!("{turn:?}") };
    assert!((e2e - 0.7).abs() < 1e-9, "first audio after the user stopped: {e2e}");
    assert_eq!(attr(turn, "lk.end_of_turn_delay"), Some(&Value::Double(0.5)));
    assert_eq!(attr(turn, "lk.pii.instructions"), Some(&st("be brief")));
    assert_eq!(attr(turn, "lk.pii.user_input"), Some(&st("book a table")));
    assert_eq!(attr(turn, "lk.interrupted"), Some(&Value::Bool(false)));
    for absent in
        ["lk.interruption.source", "lk.playout.position", "lk.pii.response.function_calls"]
    {
        assert_eq!(attr(turn, absent), None, "{absent} on a completed pipeline reply");
    }

    let user = one(&records, "user_turn");
    assert_eq!(parent(&records, user), "agent_session", "a sibling of the agent turn");
    assert_eq!(attr(user, "lk.user_turn.origin"), Some(&st("speech")));
    assert_eq!(attr(user, "lk.user_turn.outcome"), Some(&st("committed")));
    assert_eq!(attr(user, "gen_ai.request.model"), Some(&st("nova-3")));
    let hook = one(&records, "on_user_turn_completed");
    assert_eq!(parent(&records, hook), "user_turn");
    assert_eq!(user.end_unix_nanos, hook.end_unix_nanos, "the turn ends when its hook returns");
    assert_eq!(attr(one(&records, "eou_wait"), "lk.eou.outcome"), Some(&st("committed")));
}

// Catches: a backchannel verdict dropped from `user_speaking`, a verdict written as a native
// boolean, or an overlap ended without one left unmarked.
#[test]
fn every_overlap_verdict_is_written_as_a_string() {
    let mut s = opened();
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
        s.user_speaks(from, UserTurnOrigin::Speech);
        let turn = s.heard.as_mut().expect("heard");
        match outcome {
            Some(v) => turn.overlap(&verdict(v)),
            None => turn.overlap_ended(),
        }
        s.speech_ended(from + 300, endpointing());
    }
    let records = s.close(5_000);
    well_formed(&records);
    let verdicts: Vec<_> = named(&records, "user_speaking")
        .iter()
        .map(|s| attr(s, "lk.is_interruption").cloned())
        .collect();
    assert_eq!(verdicts, [Some(st("false")), Some(st("true")), Some(st("false"))]);
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
    let mut s = opened();
    s.at(100);
    let exit = s.d.hook_span(HookKind::OnExit, Some("back")).expect("its span");
    let error = lk_agents_abi::errors::HostError {
        kind: "ValueError".into(),
        message: "no table".into(),
        host_ref: None,
        traceback: Some("Traceback (most recent call last): ...".into()),
    };
    s.at(200);
    s.d.hook_span_ended(exit, Some(&HookFailure::Raised(&error)));
    s.at(300);
    s.d.activity.agent = "back".into();
    let enter = s.d.hook_span(HookKind::OnEnter, None).expect("its span");
    s.at(400);
    s.d.hook_span_ended(enter, Some(&HookFailure::TimedOut));
    let records = s.close(1_000);
    well_formed(&records);
    let update = one(&records, "update_agent");
    assert_eq!(attr(update, "lk.previous_agent_label"), Some(&st(AGENT)));
    assert_eq!(attr(update, "lk.agent_label"), Some(&st("back")));
    assert_eq!(parent(&records, one(&records, "drain_agent_activity")), "update_agent");
    let activity = named(&records, "start_agent_activity");
    let entry = activity.iter().find(|a| parent(&records, a) == "update_agent").expect("the entry");
    let enter =
        named(&records, "on_enter").into_iter().find(|e| e.parent == Some(entry.context.span_id));
    assert!(entry.end_unix_nanos >= enter.expect("its on_enter").end_unix_nanos);
    assert!(update.end_unix_nanos <= 1_000_000_000 + 400_000_000, "the entry ends the handoff");
    let exit = one(&records, "on_exit");
    assert_eq!(exit.status, Status::Error("no table".into()));
    assert_eq!(attr(exit, "error.type"), Some(&st("exception")));
    assert_eq!(exit.events[0].name, "exception");
    assert!(exit.events[0].attributes.contains(&("exception.type".into(), st("ValueError"))));
    let traceback = st("Traceback (most recent call last): ...");
    assert!(exit.events[0].attributes.contains(&("exception.stacktrace".into(), traceback)));
}

// Catches: a user turn opened by anything but its first evidence or not backdated to it, or a
// verdict without its outcome.
#[test]
fn a_user_turn_for_each_origin_and_outcome() {
    use UserTurnCommitVerdict as V;
    let mut s = opened();
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
        speak(&mut s, at, at + 300, "words");
        s.decide(at + 500, i as u64 + 1, verdict);
        expected.push(("speech", outcome));
    }
    // A committed turn with no hook to run ends as it is answered.
    s.retire(6_600, 5, UserTurnOutcome::Committed);
    s.hear(7_000, UserTurnOrigin::Transcript).transcript(7.0, "interim_transcript", "hi", "");
    s.abandon(7_100, UserTurnOutcome::Cleared);
    s.user_speaks(7_200, UserTurnOrigin::ServerSpeech);
    s.abandon(7_300, UserTurnOutcome::TranscriptionTimeout);
    expected.push(("transcript", "cleared"));
    expected.push(("server_speech", "transcription_timeout"));
    let typed = lk_agents_abi::ChatMessage::text("i", lk_agents_abi::Role::User, "hi", UNIX_EPOCH);
    s.at(7_400);
    s.d.item_turn(&lk_agents_abi::ChatItem::Message(typed), &Source::UserTyped);
    expected.push(("text", "committed"));
    speak(&mut s, 8_000, 8_100, "left open");
    expected.push(("speech", "closed"));
    let records = s.close(9_000);
    well_formed(&records);
    let turns: Vec<_> = named(&records, "user_turn")
        .iter()
        .map(|t| {
            (attr(t, "lk.user_turn.origin").cloned(), attr(t, "lk.user_turn.outcome").cloned())
        })
        .collect();
    let expected: Vec<_> = expected.iter().map(|(o, e)| (Some(st(o)), Some(st(e)))).collect();
    assert_eq!(turns, expected);
    let first = named(&records, "user_turn")[0];
    assert_eq!(first.start_unix_nanos, 1_000_000_000 + 1_000_000_000, "the VAD's speech start");
}

// Catches: speech resumed before the decision opening a second turn, or the wait not ending
// `user_resumed` with its event.
#[test]
fn a_resumed_turn_stays_one_turn() {
    let mut s = opened();
    speak(&mut s, 100, 600, "i want");
    s.at(700);
    let turn = s.heard.as_mut().expect("heard");
    turn.detection(&mut s.d.telemetry, &EouDetected::Awaiting, serde_json::Value::Null);
    speak(&mut s, 800, 1_200, "a table");
    s.decide(1_700, 1, UserTurnCommitVerdict::Empty);
    let records = s.close(2_000);
    well_formed(&records);
    let turn = one(&records, "user_turn");
    assert_eq!(attr(turn, "lk.eou.resume_count"), Some(&Value::Int(1)));
    assert_eq!(named(&records, "user_speaking").len(), 2);
    let waits = named(&records, "eou_wait");
    assert_eq!(attr(waits[0], "lk.eou.outcome"), Some(&st("user_resumed")));
    assert_eq!(waits[0].events[0].name, "user_resumed");
    assert_eq!(attr(waits[1], "lk.eou.not_committed_count"), Some(&Value::Int(1)), "empty");
    let detection = one(&records, "eou_detection");
    assert_eq!(parent(&records, detection), "eou_wait");
    assert_eq!(detection.events[0].name, "superseded", "the user resumed while it decided");
    assert!(
        detection.events[0].attributes.contains(&("lk.eou.outcome".into(), st("user_resumed")))
    );
    assert_eq!(attr(detection, "lk.eou.probability"), None);
}

// Catches: the close leaving a span for the sweep, ending a parent before its child, or the
// session not last.
#[test]
fn the_close_ends_every_open_span_itself() {
    let mut s = opened();
    s.speech(40, 1, reply(None));
    s.at(50);
    let reason = lk_agents_abi::voice::events::CloseReason::UserInitiated;
    s.d.begin_closing(ms(50), Close { reason, drain: true, error: None });
    speak(&mut s, 100, 200, "wait");
    let amd = reserved(&s, 7, ms(160));
    s.at(160);
    s.d.amd_span(&AmdSpan::Attached(amd));
    let records = s.close(1_000);
    well_formed(&records);
    assert_eq!(records.last().map(|s| s.name.to_string()), Some("agent_session".into()));
    assert_eq!(parent(&records, one(&records, "drain_agent_activity")), "session_close");
    assert_eq!(attr(one(&records, "user_turn"), "lk.user_turn.outcome"), Some(&st("closed")));
    for name in ["on_exit", "agent_turn", "amd", "session_close"] {
        one(&records, name);
    }
    assert!(s.drain().is_empty(), "nothing after the close");
    assert_eq!(s.d.spans_swept(), 0, "every owner ended its span");
}

// Catches: the queue wait, the interruption's cause or the played position missing from an
// interrupted reply, a later cause overwriting the first, or the e2e latency labeled with the
// GenAI keys instead of the LLM's `model_provider` / `model_name`.
#[test]
fn an_interrupted_reply_records_its_wait_cause_and_playout() {
    let mut s = opened();
    let user =
        lk_agents_abi::MetricsReport { stopped_speaking_at: Some(1.0), ..Default::default() };
    s.speech(100, 1, reply(Some(user)));
    s.turn(1).scheduled(ms(100));
    s.turn(1).authorized(ms(350));
    s.turn(1).authorized(ms(390));
    s.at(400);
    s.d.agent_turn_speaking(SpeechId(1), 1.4);
    s.turn(1).interrupted(InterruptCause::ByAudioActivity);
    s.turn(1).interrupted(InterruptCause::Host);
    s.turn(1).played(Duration::from_millis(300), false);
    s.turn(1).played(Duration::from_millis(1_200), true);
    s.release(1_700, 1, SpeechOutcome::Interrupted);
    // Interrupted with no recorded cause: programmatic.
    s.speech(1_800, 2, reply(None));
    s.release(1_900, 2, SpeechOutcome::Interrupted);
    let records = s.close_records(2_000);
    let spans = spans_of(records.clone());
    let turns = named(&spans, "agent_turn");
    let turn = turns[0];
    assert_eq!(
        attr(turn, "lk.speech.queue_wait"),
        Some(&Value::Double(0.25)),
        "first authorization"
    );
    assert_eq!(attr(turn, "lk.interrupted"), Some(&Value::Bool(true)));
    assert_eq!(attr(turn, "lk.interruption.source"), Some(&st("audio_activity")));
    assert_eq!(attr(turn, "lk.playout.position"), Some(&Value::Double(1.2)));
    assert_eq!(attr(turns[1], "lk.interruption.source"), Some(&st("programmatic")));
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
        [("model_provider".into(), st("api.openai.com")), ("model_name".into(), st("gpt-4o"))]
    );
}

// Catches: a realtime turn without its function calls, or with a pipeline reply's input keys.
#[test]
fn a_realtime_reply_records_its_function_calls() {
    let mut s = opened();
    s.speech(100, 1, Reply { realtime: true, ..reply(None) });
    let call = lk_agents_abi::FunctionCall {
        id: "item_1".into(),
        call_id: "call_1".into(),
        arguments: "{}".into(),
        name: "lookup".into(),
        created_at: UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    s.turn(1).function_call(&call);
    s.release(300, 1, SpeechOutcome::Completed);
    s.speech(400, 2, Reply { realtime: true, ..reply(None) });
    s.release(500, 2, SpeechOutcome::Completed);
    let records = s.close(1_000);
    let turns = named(&records, "agent_turn");
    let Some(Value::String(calls)) = attr(turns[0], "lk.pii.response.function_calls") else {
        panic!("{:?}", turns[0])
    };
    let calls: serde_json::Value = serde_json::from_str(calls).expect("JSON");
    assert_eq!(calls[0]["call_id"], "call_1");
    assert_eq!(calls[0]["name"], "lookup");
    assert_eq!(calls[0].get("created_at"), None);
    assert_eq!(attr(turns[1], "lk.pii.response.function_calls"), Some(&st("[]")));
    assert_eq!(attr(turns[0], "lk.pii.instructions"), None, "a pipeline reply's");
    assert_eq!(attr(turns[0], "lk.pii.user_input"), None);
}

// Catches: a discarded preemptive generation ending a turn of its own instead of handing it to
// the speech that answers, the handover unmarked, or a lapsed handover adopted by a later speech.
#[test]
fn a_discarded_preemptive_generation_hands_its_turn_over() {
    let mut s = opened();
    s.speech(100, 1, reply(None));
    s.at(200);
    s.turn(1).interrupted(InterruptCause::Superseded);
    s.d.activity.speech_scheduler.hand_over(SpeechId(1));
    s.speech(200, 2, reply(None));
    s.release(300, 1, SpeechOutcome::Interrupted);
    s.release(900, 2, SpeechOutcome::Completed);
    // Unclaimed when the next session event goes out, a handover lapses.
    s.speech(1_000, 3, reply(None));
    s.d.activity.speech_scheduler.hand_over(SpeechId(3));
    s.d.activity.lapse_agent_turn_handover();
    let early = spans_of(s.drain());
    s.speech(1_100, 4, reply(None));
    let records = [early, s.close(2_000)].concat();
    well_formed(&records);
    let turns = named(&records, "agent_turn");
    let turn = turns
        .iter()
        .find(|t| attr(t, "lk.speech_id") == Some(&st("AS_test_speech_2")))
        .unwrap_or_else(|| panic!("the answer's turn: {turns:?}"));
    assert_eq!(turn.start_unix_nanos, 1_000_000_000 + 100_000_000, "the discarded one's start");
    assert_eq!(attr(turn, "lk.interrupted"), Some(&Value::Bool(false)));
    let names: Vec<_> = turn.events.iter().map(|e| e.name.to_string()).collect();
    assert_eq!(names, ["generation", "preemptive_generation_discarded", "generation"]);
    assert!(turn.events[1].attributes.contains(&("lk.speech_id".into(), st("AS_test_speech_1"))));
    assert!(
        turn.events[2].attributes.contains(&("lk.generation_id".into(), st("AS_test_speech_2_1")))
    );
    assert_eq!(turns.len(), 3, "speeches 3 and 4 have their own turns");
}

// Catches: participant attributes missing or not in the protocol's spelling, the agent's spans
// described as the user, or the room's connection and disconnect events without their keys.
#[test]
fn spans_describe_their_participants_and_the_room() {
    let mut s = opened();
    let user = info_of("caller", "PA_user", ParticipantKind::Sip);
    room(&mut s, 0, RoomIoEvent::ParticipantInfo { participant: user, local: false });
    let agent = info_of("agent", "PA_agent", ParticipantKind::Agent);
    room(&mut s, 0, RoomIoEvent::ParticipantInfo { participant: agent, local: true });
    room(&mut s, 10, RoomIoEvent::ConnectionStateChanged(ConnectionState::Connected));
    room(&mut s, 20, linked("caller"));
    speak(&mut s, 100, 600, "hi");
    s.decide(900, 1, UserTurnCommitVerdict::Empty);
    s.speech(1_000, 1, reply(None));
    s.at(1_100);
    s.d.agent_turn_speaking(SpeechId(1), 2.1);
    s.release(1_500, 1, SpeechOutcome::Completed);
    let reason = DisconnectReason::ClientInitiated;
    let unlinked = ParticipantUnlinked { identity: "caller".into(), reason };
    room(&mut s, 1_600, RoomIoEvent::ParticipantUnlinked(unlinked));
    speak(&mut s, 1_700, 1_800, "gone");
    let records = s.close(2_000);
    well_formed(&records);
    let is = |span: &SpanRecord, id: &str, identity: &str, kind: &str| {
        assert_eq!(attr(span, "lk.participant_id"), Some(&st(id)), "{}", span.name);
        assert_eq!(attr(span, "lk.pii.participant_identity"), Some(&st(identity)));
        assert_eq!(attr(span, "lk.participant_kind"), Some(&st(kind)));
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
        assert!(event(name).attributes.contains(&(key.into(), st(value))), "{name} {key}");
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
    let mut s = opened();
    s.user_speaks(100, UserTurnOrigin::Speech);
    let turn = s.heard.as_mut().expect("heard");
    turn.transcript(1.25, "interim_transcript", "boo", "r1");
    turn.transcript(1.5, "preflight_transcript", "book", "r1");
    s.speech_ended(600, endpointing());
    {
        let turn = s.heard.as_mut().expect("heard");
        let trace = &mut s.d.telemetry;
        turn.detection(trace, &EouDetected::Awaiting, serde_json::Value::Null);
        trace.step(ms(700));
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
        turn.detection(trace, &predicted, ctx);
        trace.step(ms(800));
        turn.transcript(1.8, "final_transcript", "book it", "r2");
        let stt_trigger = Endpointing { source: EouSource::Stt, delay: 0.5 };
        turn.retriggered(trace, stt_trigger);
    }
    s.decide(1_200, 1, UserTurnCommitVerdict::Reply);
    s.retire(1_200, 1, UserTurnOutcome::Committed);
    let records = s.close(2_000);
    well_formed(&records);
    let turn = one(&records, "user_turn");
    let ids = Value::Array(vec![st("r1"), st("r2")]);
    assert_eq!(attr(turn, "lk.provider_request_ids"), Some(&ids));
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
    assert_eq!(attr(wait, "lk.eou.source"), Some(&st("stt")), "the latest trigger");
    assert_eq!(attr(wait, "lk.eou.not_committed_count"), Some(&Value::Int(0)));
    assert_eq!(wait.events[0].name, "rearmed");
    assert!(wait.events[0].attributes.contains(&("lk.eou.source".into(), st("stt"))));
    let detection = one(&records, "eou_detection");
    assert_eq!(detection.start_unix_nanos, 1_000_000_000 + 600_000_000, "from the request");
    let Some(Value::String(ctx)) = attr(detection, "lk.pii.chat_ctx") else { panic!() };
    assert!(ctx.contains("book"), "{ctx}");
    assert_eq!(attr(detection, "lk.eou.language"), Some(&st("")), "as 1.8.10 writes no language");
    assert_eq!(attr(detection, "lk.eou.from_cache"), Some(&Value::Bool(false)));
    assert_eq!(attr(detection, "lk.eou.probability"), Some(&Value::Double(0.2)));
    assert_eq!(attr(detection, "lk.eou.unlikely_threshold"), Some(&Value::Double(0.5)));
    assert_eq!(attr(detection, "lk.eou.detection_delay"), Some(&Value::Double(0.04)));
    assert_eq!(attr(detection, "lk.eou.endpointing_delay"), Some(&Value::Double(2.0)));
    assert_eq!(attr(detection, "lk.eou.source"), Some(&st("vad")));
    assert!(detection.events.is_empty(), "decided before the wait ended");
}

// Catches: a latency dropped when the model is unknown instead of recorded unlabeled, or an
// undescribed participant written with empty id and kind strings.
#[test]
fn unknown_metadata_is_omitted_not_blank() {
    let mut s = opened();
    let user = lk_agents_abi::MetricsReport {
        stopped_speaking_at: Some(1.0),
        on_user_turn_completed_delay: Some(0.1),
        ..Default::default()
    };
    s.speech(100, 1, Reply { llm: None, ..reply(Some(user)) });
    s.at(200);
    s.d.agent_turn_speaking(SpeechId(1), 1.2);
    s.release(300, 1, SpeechOutcome::Completed);
    room(&mut s, 400, linked("stranger"));
    let records = s.close_records(500);
    for name in ["lk.agents.turn.e2e_latency", "lk.agents.turn.on_user_turn_completed_delay"] {
        let m = measurements(&records).into_iter().find(|m| m.instrument == name);
        assert!(m.unwrap_or_else(|| panic!("{name}")).labels.is_empty(), "{name}");
    }
    let spans = spans_of(records);
    let session = one(&spans, "agent_session");
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
    let mut s = session_with(telemetry, ms(500));
    assert_eq!(s.d.session_start_span(), Some(context(2)));
    // The input streams run under the first activity: reserved, then running.
    assert_eq!(s.d.activity_span(), Some(context(3)));
    s.at(600);
    let enter = s.d.hook_span(HookKind::OnEnter, None).expect("its span");
    let hook = enter.context(&s.d.telemetry);
    assert_eq!(s.d.activity_span(), Some(context(3)));
    s.at(700);
    s.d.hook_span_ended(enter, None);
    s.d.session_opened();
    let records = s.close(1_000);
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
    let job_ids = JobIds {
        dispatch_id: Some("AD_1".into()),
        worker_id: Some("AW_1".into()),
        agent_id: Some("AG_1".into()),
    };
    let mut s = session_with(SessionTelemetry { job_ids, ..Default::default() }, ms(0));
    s.at(1);
    s.d.session_opened();
    let records = s.close(10);
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
    let mut s = opened();
    let update = reserved(&s, 50, ms(50));
    let activity = reserved(&s, 51, ms(60));
    s.d.arriving("other", Some(Activation { update, activity }));
    s.d.arriving("back", Some(Activation { update, activity }));
    s.at(100);
    let exit = s.d.hook_span(HookKind::OnExit, Some("back")).expect("its span");
    let exit_context = exit.context(&s.d.telemetry);
    s.at(200);
    s.d.hook_span_ended(exit, None);
    s.at(300);
    s.d.activity.agent = "back".into();
    let enter = s.d.hook_span(HookKind::OnEnter, None).expect("its span");
    let enter_context = enter.context(&s.d.telemetry);
    s.at(400);
    s.d.hook_span_ended(enter, None);
    let records = s.close(1_000);
    well_formed(&records);
    assert_eq!(one(&records, "update_agent").context, update.context);
    let entry = named(&records, "start_agent_activity");
    let entry = entry.iter().find(|a| parent(&records, a) == "update_agent").expect("the entry");
    assert_eq!(entry.context, activity.context);
    assert_eq!(exit_context, Some(one(&records, "on_exit").context));
    assert_eq!(enter_context, Some(one(&records, "on_enter").context));
}

// Catches: stalls recorded only as their own spans, with the session's totals and its
// `event_loop_blocked` events missing; or totals written for a session that never stalled; or
// AMD's `amd` drawn anew instead of the span its classification runs under.
#[test]
fn event_loop_stalls_fold_into_the_session() {
    let mut s = opened();
    s.at(100);
    s.d.event_loop_blocked(Duration::from_millis(200));
    s.at(200);
    s.d.event_loop_blocked(Duration::from_millis(600));
    let amd = reserved(&s, 70, ms(300));
    s.at(300);
    s.d.amd_span(&AmdSpan::Attached(amd));
    let records = s.close(1_000);
    well_formed(&records);
    let session = one(&records, "agent_session");
    assert_eq!(attr(session, "lk.blocking.count"), Some(&Value::Int(2)));
    assert_eq!(attr(session, "lk.blocking.total_duration"), Some(&Value::Double(0.8)));
    assert_eq!(attr(session, "lk.blocking.max_duration"), Some(&Value::Double(0.6)));
    let events: Vec<_> = session.events.iter().filter(|e| e.name == "event_loop_blocked").collect();
    assert_eq!(events.len(), 2);
    assert!(events[1].attributes.contains(&("lk.blocking.duration".into(), Value::Double(0.6))));
    assert_eq!(one(&records, "amd").context, amd.context);

    let records = opened().close(1_000);
    assert_eq!(attr(one(&records, "agent_session"), "lk.blocking.count"), None);
}

// Catches: chat items reported with no link to their turn: the reply's message to its
// `agent_turn`, a typed user message to the `user_turn` it opened.
#[test]
fn each_chat_item_knows_its_turn() {
    let mut s = opened();
    let turn = s.speech(100, 1, reply(None)).expect("a turn");
    let message = |id: &str, role| {
        lk_agents_abi::ChatItem::Message(lk_agents_abi::ChatMessage::text(
            id, role, "hi", UNIX_EPOCH,
        ))
    };
    let reply_source = Source::Reply { speech: SpeechId(1), step: 1 };
    s.at(200);
    s.d.item_turn(&message("item_a", lk_agents_abi::Role::Assistant), &reply_source);
    s.at(300);
    s.d.item_turn(&message("item_u", lk_agents_abi::Role::User), &Source::UserTyped);
    let turns = s.d.session.item_turns.items().clone();
    let records = s.close(1_000);
    assert_eq!(turns.get("item_a"), Some(&turn));
    assert_eq!(turns.get("item_u"), Some(&one(&records, "user_turn").context));
    let _ = Visibility::Both;
}
