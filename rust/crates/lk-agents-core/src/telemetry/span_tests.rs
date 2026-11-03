use super::*;
use lk_agents_abi::telemetry::decl::{Content, Json};
use lk_agents_abi::telemetry::decl::{Measured, Seconds};
use lk_agents_abi::telemetry::record::Value;
use lk_agents_abi::telemetry::spans::common::GenAiOperation;
use lk_agents_abi::telemetry::spans::session::{
    AgentSession, AgentSessionEnd, AgentStateChanged, OnEnter, SessionStart, StartAgentActivity,
};
use lk_agents_abi::telemetry::spans::user_turn::{
    UserSpeaking, UserSpeakingEnd, UserTurn, UserTurnEnd, UserTurnOrigin, UserTurnOutcome,
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
    let root = t.open(ts(0), Some(external), session());
    let start = root.child(&mut t, ts(5), SessionStart);
    start.end(&mut t, ts(20), ());
    root.end(&mut t, ts(30), session_end());
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
    let root = t.open(ts(100), None, session());
    let turn = root.child_at(&mut t, ts(40), user_turn());
    let speaking = turn.child_at(&mut t, ts(120), UserSpeaking { participant: None });
    let verdict = UserSpeakingEnd {
        is_interruption: None,
        probability: None,
        total_duration: None,
        prediction_duration: None,
        detection_delay: None,
    };
    speaking.end(&mut t, ts(110), verdict);
    turn.end(&mut t, ts(130), turn_end(None));
    root.end(&mut t, ts(200), session_end());
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
    let root = t.open(ts(0), None, session());
    let start = root.child(&mut t, ts(1), SessionStart);
    let activity = start.child(
        &mut t,
        ts(2),
        StartAgentActivity {
            agent_label: "a".into(),
            operation: GenAiOperation::CreateAgent,
            agent_name: "a".into(),
            provider: None,
            model: None,
        },
    );
    let enter = activity.child(&mut t, ts(3), OnEnter { agent_label: "a".into() });
    assert_eq!(t.close(ts(9)), 4);
    assert_eq!(t.swept(), 4);
    let records = t.drain();
    let names: Vec<_> = spans(&records).iter().map(|s| s.name.to_string()).collect();
    assert_eq!(names, ["on_enter", "start_agent_activity", "session_start", "agent_session"]);
    assert!(spans(&records).iter().all(|s| attr(s, "lk.cancelled") == Some(&Value::Bool(true))));
    // The owners still end their handles; nothing is recorded twice.
    enter.end(&mut t, ts(10), ());
    activity.end(&mut t, ts(10), ());
    start.end(&mut t, ts(10), ());
    root.end(&mut t, ts(10), session_end());
    assert!(t.drain().is_empty());
}

// Catches: an explicitly ended span marked cancelled, events dropped or misdated, or a failure
// recorded without its status, type and exception event.
#[test]
fn events_and_failures_are_recorded_on_the_span() {
    let mut t = trace();
    let root = t.open(ts(0), None, session());
    root.event(
        &mut t,
        ts(4),
        AgentStateChanged { old: "listening".into(), new: "thinking".into() },
    );
    let error = HostError::new("boom");
    let error = HostError { kind: "ValueError".into(), ..error };
    root.fail(&mut t, ts(6), &error);
    root.end(&mut t, ts(8), session_end());
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

// Catches: a measured end field not recorded with the span's labels, or an unmeasured one
// recorded.
#[test]
fn a_measured_end_field_records_its_instrument() {
    let mut t = trace();
    let root = t.open(ts(0), None, session());
    root.child(&mut t, ts(1), user_turn()).end(&mut t, ts(2), turn_end(Some(0.25)));
    root.end(&mut t, ts(3), session_end());
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
            (Key::Borrowed("gen_ai.request.model"), "nova".into()),
            (Key::Borrowed("gen_ai.provider.name"), "deepgram".into())
        ]
    );
}

// Catches: ids that repeat or depend on anything but the injected source.
#[test]
fn ids_come_from_the_injected_source() {
    let ids = |seed| {
        let wall = SystemTime::UNIX_EPOCH;
        let mut t = Trace::new(seeded_ids(seed), Ts::ZERO, wall, None, None);
        let root = t.open(Ts::ZERO, None, session());
        let child = root.child(&mut t, Ts::ZERO, SessionStart);
        let out = (root.context(), child.context());
        child.end(&mut t, Ts::ZERO, ());
        root.end(&mut t, Ts::ZERO, session_end());
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
    let _root = t.open(ts(0), None, session());
}

// Catches: a parent ended at a backdated instant (the end of speech) closing before a child
// that ended later.
#[test]
fn a_parent_never_ends_before_its_children() {
    let mut t = trace();
    let root = t.open(ts(0), None, session());
    let start = root.child(&mut t, ts(1), SessionStart);
    start.end(&mut t, ts(50), ());
    root.end(&mut t, ts(20), session_end());
    let records = t.drain();
    let [start, root] = spans(&records)[..] else { panic!("{records:?}") };
    assert_eq!(root.end_unix_nanos, start.end_unix_nanos);
}
