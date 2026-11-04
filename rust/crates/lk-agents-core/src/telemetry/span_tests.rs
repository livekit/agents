use super::*;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::telemetry::decl::{Content, Json};
use lk_agents_abi::telemetry::decl::{Measured, Seconds};
use lk_agents_abi::telemetry::record::Value;
use lk_agents_abi::telemetry::spans::common::GenAiOperation;
use lk_agents_abi::telemetry::spans::session::{
    AgentSession, AgentSessionEnd, AgentStateChanged, OnEnter, SessionStart, StartAgentActivity,
};
use lk_agents_abi::telemetry::spans::user_turn::{
    EouDetection, EouDetectionEnd, EouOutcome, EouWait, EouWaitEnd, UserSpeaking, UserSpeakingEnd,
    UserTurn, UserTurnEnd, UserTurnOrigin, UserTurnOutcome,
};

const EPOCH_S: u64 = 1_700_000_000;

fn trace() -> Trace {
    let wall = SystemTime::UNIX_EPOCH + Duration::from_secs(EPOCH_S);
    let job = Some(JobKey("job".into()));
    Trace::new(seeded_ids(7), Ts::from_nanos(1_000), wall, job, Some("AS_1".into()))
}

fn ts(ms: u64) -> Ts {
    Ts::from_nanos(1_000 + ms * 1_000_000)
}

fn wall_ms(ms: u64) -> u64 {
    EPOCH_S * 1_000_000_000 + ms * 1_000_000
}

fn session() -> AgentSession {
    AgentSession {
        agent_label: "a".into(),
        operation: GenAiOperation::InvokeWorkflow,
        workflow: AgentSession::NAME.into(),
        dispatch_id: None,
        worker_id: None,
        agent_id: None,
    }
}

fn session_end() -> AgentSessionEnd {
    AgentSessionEnd {
        blocking_count: None,
        blocking_total: None,
        blocking_max: None,
        sip: Default::default(),
        sip_phone_number: None,
    }
}

fn turn_end(delay: Option<f64>) -> UserTurnEnd {
    UserTurnEnd {
        outcome: UserTurnOutcome::Committed,
        transcript: Content("hi".into()),
        confidence: None,
        transcription_delay: Measured(delay.map(Seconds)),
        end_of_turn_delay: Measured(None),
        provider_request_ids: Vec::new(),
        stt_events: Json(serde_json::Value::Array(Vec::new())),
        resumes: None,
    }
}

fn user_turn() -> UserTurn {
    UserTurn {
        started_at: ts(0),
        origin: UserTurnOrigin::Speech,
        participant: None,
        stt_model: Some("nova".into()),
        stt_provider: Some("deepgram".into()),
    }
}

fn spans(records: &[Record]) -> Vec<&SpanRecord> {
    records
        .iter()
        .filter_map(|r| match r {
            Record::Span(s) => Some(s),
            _ => None,
        })
        .collect()
}

fn attr<'a>(span: &'a SpanRecord, key: &str) -> Option<&'a Value> {
    span.attributes.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

// Catches: a child not linked to its parent or trace, wall times not derived from the anchor, a
// root not continuing its external parent, or the session's keys missing.
#[test]
fn spans_nest_under_their_parent_on_the_anchor_timeline() {
    let mut t = trace();
    let external = SpanRef { trace_id: TraceId(42), span_id: SpanId(9), sampled: true };
    t.step(ts(0));
    let root = t.open_root(Some(external), None, session());
    t.step(ts(5));
    let start = t.open_in(&root, SessionStart);
    t.step(ts(20));
    t.end(start, ());
    t.step(ts(30));
    t.end(root, session_end());
    let records = t.drain();
    let [start, root] = spans(&records)[..] else { panic!("{records:?}") };
    assert_eq!(root.parent, Some(SpanId(9)));
    assert_eq!(root.context.trace_id, TraceId(42));
    assert_eq!(start.parent, Some(root.context.span_id));
    assert_eq!(start.context.trace_id, TraceId(42));
    assert_eq!((start.start_unix_nanos, start.end_unix_nanos), (wall_ms(5), wall_ms(20)));
    assert_eq!((root.start_unix_nanos, root.end_unix_nanos), (wall_ms(0), wall_ms(30)));
    assert_eq!(root.name, "agent_session");
    assert_eq!(attr(root, "gen_ai.conversation.id"), Some(&Value::String("AS_1".into())));
    assert_eq!(root.job, Some(JobKey("job".into())));
    assert!(t.drain().is_empty(), "drained once");
}

// Catches: a backdated child starting before its parent, or an end before the start.
#[test]
fn a_backdated_start_is_clamped_to_the_parent() {
    let mut t = trace();
    t.step(ts(100));
    let root = t.open_root(None, None, session());
    let turn = t.open_in(&root, UserTurn { started_at: ts(40), ..user_turn() });
    t.step(ts(120));
    let speaking = t.open_in(&turn, UserSpeaking { started_at: ts(120), participant: None });
    let verdict = UserSpeakingEnd {
        ended_at: ts(110),
        is_interruption: None,
        probability: None,
        total_duration: None,
        prediction_duration: None,
        detection_delay: None,
    };
    t.end(speaking, verdict);
    t.step(ts(130));
    t.end(turn, turn_end(None));
    t.step(ts(200));
    t.end(root, session_end());
    let records = t.drain();
    let [speaking, turn, _] = spans(&records)[..] else { panic!("{records:?}") };
    assert_eq!(turn.start_unix_nanos, wall_ms(100), "clamped to the session's start");
    assert_eq!(speaking.start_unix_nanos, wall_ms(120), "after the parent: kept");
    assert_eq!(speaking.end_unix_nanos, wall_ms(120), "never before its start");
}

// Catches: the close sweep ending a parent before its child, missing a span, or not marking
// what it ended cancelled.
#[test]
fn the_close_sweep_ends_what_is_left_child_first_as_cancelled() {
    let mut t = trace();
    t.step(ts(0));
    let root = t.open_root(None, None, session());
    let start = t.open_in(&root, SessionStart);
    let activity = t.open_in(
        &start,
        StartAgentActivity {
            agent_label: "a".into(),
            operation: GenAiOperation::CreateAgent,
            agent_name: "a".into(),
            provider: None,
            model: None,
        },
    );
    let enter = t.open_in(&activity, OnEnter { agent_label: "a".into() });
    assert_eq!(t.close(ts(9)), 4);
    assert_eq!(t.swept(), 4);
    let records = t.drain();
    let names: Vec<_> = spans(&records).iter().map(|s| s.name.to_string()).collect();
    assert_eq!(names, ["on_enter", "start_agent_activity", "session_start", "agent_session"]);
    assert!(spans(&records).iter().all(|s| attr(s, "lk.cancelled") == Some(&Value::Bool(true))));
    // The owners still end their handles; nothing is recorded twice.
    t.end(enter, ());
    t.end(activity, ());
    t.end(start, ());
    t.end(root, session_end());
    assert!(t.drain().is_empty());
}

// Catches: an explicitly ended span marked cancelled, events dropped or misdated, or a failure
// recorded without its status, type and exception event.
#[test]
fn events_and_failures_are_recorded_on_the_span() {
    let mut t = trace();
    t.step(ts(0));
    let root = t.open_root(None, None, session());
    t.step(ts(4));
    t.event(&root, AgentStateChanged { old: "listening".into(), new: "thinking".into() });
    let error = HostError::new("boom");
    let error = HostError { kind: "ValueError".into(), ..error };
    t.step(ts(6));
    t.fail(&root, &error);
    t.step(ts(8));
    t.end(root, session_end());
    let records = t.drain();
    let [root] = spans(&records)[..] else { panic!("{records:?}") };
    assert_eq!(root.status, Status::Error("boom".into()));
    assert_eq!(attr(root, "error.type"), Some(&Value::String("ValueError".into())));
    assert_eq!(attr(root, "lk.cancelled"), None);
    let names: Vec<_> =
        root.events.iter().map(|e| (e.name.to_string(), e.time_unix_nanos)).collect();
    assert_eq!(
        names,
        [("agent_state_changed".to_owned(), wall_ms(4)), ("exception".to_owned(), wall_ms(6))]
    );
    let exception = &root.events[1].attributes;
    assert!(exception.contains(&(Key::Borrowed("exception.type"), "ValueError".into())));
    assert!(exception.contains(&(Key::Borrowed("exception.message"), "boom".into())));
}

// Catches: a measured end field not recorded with the span's labels under the instrument's label
// keys, or an unmeasured one recorded.
#[test]
fn a_measured_end_field_records_its_instrument() {
    let mut t = trace();
    t.step(ts(0));
    let root = t.open_root(None, None, session());
    let turn = t.open_in(&root, user_turn());
    t.end(turn, turn_end(Some(0.25)));
    t.end(root, session_end());
    let records = t.drain();
    let measured: Vec<_> = records
        .iter()
        .filter_map(|r| match r {
            Record::Measurement(m) => Some(m),
            _ => None,
        })
        .collect();
    let [m] = measured[..] else { panic!("{records:?}") };
    assert_eq!(m.instrument, "lk.agents.turn.transcription_delay");
    assert_eq!(m.value, Number::Double(0.25));
    assert_eq!(
        m.labels,
        [
            (Key::Borrowed("model_name"), "nova".into()),
            (Key::Borrowed("model_provider"), "deepgram".into())
        ]
    );
}

// Catches: ids that repeat or depend on anything but the injected source.
#[test]
fn ids_come_from_the_injected_source() {
    let ids = |seed| {
        let wall = SystemTime::UNIX_EPOCH;
        let mut t = Trace::new(seeded_ids(seed), Ts::ZERO, wall, None, None);
        t.step(Ts::ZERO);
        let root = t.open_root(None, None, session());
        let child = t.open_in(&root, SessionStart);
        let out = (root.context(), child.context());
        t.end(child, ());
        t.end(root, session_end());
        out
    };
    assert_eq!(ids(1), ids(1));
    assert_ne!(ids(1), ids(2));
    let (root, child) = ids(1);
    assert_ne!(root.span_id, child.span_id);
}

// Catches: a handle dropped without `end` going unnoticed in debug builds.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "dropped without end")]
fn a_span_dropped_without_end_fails_in_debug() {
    let mut t = trace();
    t.step(ts(0));
    let _root = t.open_root(None, None, session());
}

// Catches: a parent ended at a backdated instant (the end of speech) closing before a child
// that ended later.
#[test]
fn a_parent_never_ends_before_its_children() {
    let mut t = trace();
    t.step(ts(0));
    let root = t.open_root(None, None, session());
    let turn = t.open(user_turn());
    t.step(ts(1));
    let decl = EouWait { started_at: ts(1), source: "vad".into(), endpointing_delay: Seconds(0.5) };
    let wait = t.open_in(&turn, decl);
    let detection = t.open_in(&wait, EouDetection);
    t.step(ts(50));
    t.end(detection, EouDetectionEnd { ended_at: ts(50), prediction: None });
    let end = EouWaitEnd {
        ended_at: ts(20),
        outcome: EouOutcome::Committed,
        wait_duration: Seconds(0.019),
        rearm_count: 0,
        not_committed_count: 0,
    };
    t.end(wait, end);
    t.end(turn, turn_end(None));
    t.end(root, session_end());
    let records = t.drain();
    let [detection, wait, ..] = spans(&records)[..] else { panic!("{records:?}") };
    assert_eq!(wait.end_unix_nanos, detection.end_unix_nanos);
}

// Catches: `set` appending a second copy of the start attributes instead of replacing them, or
// the measurement keeping the labels the span opened with.
#[test]
fn set_replaces_the_start_attributes_and_labels() {
    let mut t = trace();
    t.step(ts(0));
    let root = t.open_root(None, None, session());
    let turn = t.open_in(&root, user_turn());
    t.set(&turn, UserTurn { stt_model: Some("nova-3".into()), ..user_turn() });
    t.end(turn, turn_end(Some(0.25)));
    t.end(root, session_end());
    let records = t.drain();
    let span = spans(&records).into_iter().find(|s| s.name == "user_turn").expect("user_turn");
    let models: Vec<_> =
        span.attributes.iter().filter(|(k, _)| k == "gen_ai.request.model").collect();
    assert_eq!(models, [&(Key::Borrowed("gen_ai.request.model"), "nova-3".into())]);
    let labels = records.iter().find_map(|r| match r {
        Record::Measurement(m) => Some(&m.labels),
        _ => None,
    });
    assert!(
        labels.expect("a measurement").contains(&(Key::Borrowed("model_name"), "nova-3".into()))
    );
}

// Catches: `open` not implying the root as parent, a span not dated by the step's instant, or
// `end` not consuming at the step's instant.
#[test]
fn the_trace_api_opens_under_the_root_at_the_steps_instant() {
    let mut t = trace();
    t.step(ts(0));
    let root = t.open_root(None, None, session());
    t.step(ts(10));
    let turn = t.open(UserTurn { started_at: ts(10), ..user_turn() });
    t.step(ts(20));
    let speaking = t.open_in(&turn, UserSpeaking { started_at: ts(15), participant: None });
    let verdict = UserSpeakingEnd {
        ended_at: ts(20),
        is_interruption: None,
        probability: None,
        total_duration: None,
        prediction_duration: None,
        detection_delay: None,
    };
    t.end(speaking, verdict);
    t.end(turn, turn_end(None));
    t.step(ts(30));
    t.end(root, session_end());
    let records = t.drain();
    let [speaking, turn, root] = spans(&records)[..] else { panic!("{records:?}") };
    assert_eq!(turn.parent, Some(root.context.span_id));
    assert_eq!(speaking.parent, Some(turn.context.span_id));
    assert_eq!((turn.start_unix_nanos, turn.end_unix_nanos), (wall_ms(10), wall_ms(20)));
    assert_eq!((speaking.start_unix_nanos, speaking.end_unix_nanos), (wall_ms(15), wall_ms(20)));
}

// Catches: a span opened after the root ended (a late event at close) being exported, or its
// context reaching an effect.
#[test]
fn a_span_opened_after_the_root_ended_records_nothing() {
    let mut t = trace();
    t.step(ts(0));
    let root = t.open_root(None, None, session());
    t.end(root, session_end());
    assert_eq!(t.close_root(), 0);
    let _ = t.drain();
    let turn = t.open(user_turn());
    assert_eq!(t.context(&turn), None);
    t.end(turn, turn_end(None));
    assert!(t.drain().is_empty());
}

// Catches: an end attribute appended beside the start's value under the same key instead of
// replacing it, so the span carries the key twice.
#[test]
fn an_end_attribute_replaces_the_start_value_under_its_key() {
    use lk_agents_abi::telemetry::spans::user_turn::Rearmed;
    let mut t = trace();
    t.step(ts(0));
    let root = t.open_root(None, None, session());
    let turn = t.open(user_turn());
    let decl = EouWait { started_at: ts(0), source: "vad".into(), endpointing_delay: Seconds(0.5) };
    let mut wait = t.open_in(&turn, decl);
    wait.ended = true;
    let open = t.open.remove(&wait.context.span_id).expect("open");
    t.finish(open, wall_ms(1), Some(&Rearmed { at: ts(1), source: "model".into() }));
    t.end(turn, turn_end(None));
    t.end(root, session_end());
    let records = t.drain();
    let wait = spans(&records).into_iter().find(|s| s.name == "eou_wait").expect("eou_wait");
    let sources: Vec<_> = wait.attributes.iter().filter(|(k, _)| k == "lk.eou.source").collect();
    assert_eq!(sources.len(), 1, "{:?}", wait.attributes);
    assert_eq!(attr(wait, "lk.eou.source"), Some(&Value::String("model".into())));
}
