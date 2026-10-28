use std::time::{Duration, SystemTime};

use lk_agents_abi::metrics::{LlmModelUsage, ModelUsage, TtsModelUsage};
use lk_agents_abi::{ChatItem, ChatMessage, Role, VirtualClock};
use serde_json::{Value, json};

use super::*;
use crate::worker::worker::SessionReportInput;

// ============================================================================================
// records
// ============================================================================================

fn view(usage: &[ModelUsage], items: &[ChatItem], opts: RecordingOptions) -> SessionReportInput {
    SessionReportInput {
        job_id: "J1".into(),
        room_id: "RM_1".into(),
        room: "room-1".into(),
        agent_name: "my-agent".into(),
        sdk_version: "1.0.0".into(),
        started_at: Some(100.0),
        timestamp: 200.0,
        options: json!({"turn_detection": "vad"}),
        recording_options: opts,
        model_usage: usage.to_vec(),
        chat_history: items.to_vec(),
    }
}

fn all_on() -> RecordingOptions {
    RecordingOptions { audio: true, traces: true, logs: true, transcript: true, redaction: false }
}

fn tagger() -> Tagger {
    Tagger::new(std::sync::Arc::new(VirtualClock::new()))
}

fn find<'a>(recs: &'a [LogRecord], body: &str) -> Vec<&'a LogRecord> {
    recs.iter().filter(|r| r.body == body).collect()
}

fn attr<'a>(r: &'a LogRecord, k: &str) -> &'a Value {
    r.attributes.get(k).expect("missing attribute")
}

// Catches: serializing zero-valued usage fields.
#[test]
fn zero_valued_usage_fields_are_stripped() {
    let u = ModelUsage::Llm(LlmModelUsage {
        provider: "openai".into(),
        model: "gpt-4o".into(),
        input_tokens: 100,
        output_tokens: 50,
        ..Default::default()
    });
    let v = usage_attributes(&u);
    let o = v.as_object().unwrap();
    assert_eq!(o.get("input_tokens"), Some(&json!(100)));
    assert_eq!(o.get("output_tokens"), Some(&json!(50)));
    assert!(o.get("input_cached_tokens").is_none(), "{v}");
    assert!(o.get("session_duration").is_none(), "float zeros go too: {v}");
    assert_eq!(o.get("type"), Some(&json!("llm_usage")));
    assert_eq!(o.len(), 5, "type, provider, model, input_tokens, output_tokens: {v}");
}

// Catches: stripping empty strings along with the zeros.
#[test]
fn empty_provider_and_model_strings_survive_the_zero_strip() {
    let u = ModelUsage::Tts(TtsModelUsage::default());
    let o = usage_attributes(&u);
    let o = o.as_object().unwrap();
    assert_eq!(o.get("provider"), Some(&json!("")));
    assert_eq!(o.get("model"), Some(&json!("")));
    assert_eq!(o.len(), 3, "only type, provider, model: {o:?}");
}

#[test]
fn denied_content_never_reaches_session_projection_or_report_records() {
    let mut t = tagger();
    t.add("customer-tag", Some(json!({"private": "customer metadata"})));
    t.fail(Some("customer outcome reason"));
    t.record_judgment("booking", "fail", "customer reasoning", "");
    let at = SystemTime::UNIX_EPOCH;
    let items = [ChatItem::Message(ChatMessage::text("i", Role::User, "customer transcript", at))];
    for (capture_content, redaction) in [(false, false), (true, true)] {
        let mut report = view(&[], &items, RecordingOptions { redaction, ..all_on() });
        report.options = json!({"instructions": "customer prompt"});
        let records = session_records(
            &UploadGate::new(),
            &report,
            &t,
            &|_| panic!("content policy must run before projecting the transcript"),
            capture_content,
        );
        let rendered = format!("{records:?}");
        for private in [
            "customer metadata",
            "customer outcome reason",
            "customer reasoning",
            "customer prompt",
            "customer transcript",
        ] {
            assert!(!rendered.contains(private), "{private} leaked before export");
        }
        assert!(records.iter().all(|record| {
            record.attributes["job_id"] == "J1"
                && record.attributes["room_id"] == "RM_1"
                && record.attributes["lk.telemetry.capture_content"] == false
                && record.attributes["lk.telemetry.safe_body"] == true
        }));
        let evaluations = find(&records, "evaluation");
        assert_eq!(attr(evaluations[0], "evaluation")["verdict"], "fail");
    }
}

// Catches: a failing judge at the same severity as a passing one.
#[test]
fn a_failing_verdict_bumps_the_evaluation_severity() {
    let mut t = tagger();
    t.record_judgment("politeness", "pass", "fine", "");
    t.record_judgment("booked", "fail", "never booked", "");
    t.record_judgment("unsure", "maybe", "hmm", "");

    let recs =
        session_records(&UploadGate::new(), &view(&[], &[], all_on()), &t, &|_| json!({}), true);
    let evals = find(&recs, "evaluation");
    assert_eq!(evals.len(), 3);
    assert_eq!(
        evals.iter().map(|r| (r.severity_number, r.severity_text.as_str())).collect::<Vec<_>>(),
        vec![(0, "unspecified"), (17, "error"), (0, "unspecified")],
        "only `fail` bumps -- `maybe` does not"
    );
    assert!(evals.iter().all(|r| r.logger == "evaluations"));
    assert_eq!(attr(evals[1], "evaluation")["tag"], json!("lk.judge.booked:fail"));
}

// Catches: the outcome record losing its severity bump, or a `reason` key with no reason.
#[test]
fn a_failed_outcome_bumps_severity_and_an_empty_reason_is_omitted() {
    let mut t = tagger();
    t.fail(Some("user hung up"));
    let recs =
        session_records(&UploadGate::new(), &view(&[], &[], all_on()), &t, &|_| json!({}), true);
    let o = find(&recs, "outcome");
    assert_eq!(o.len(), 1);
    assert_eq!(o[0].severity_number, 17);
    assert_eq!(attr(o[0], "outcome"), &json!({"outcome": "fail", "reason": "user hung up"}));

    let mut t = tagger();
    t.success(None::<String>);
    let recs =
        session_records(&UploadGate::new(), &view(&[], &[], all_on()), &t, &|_| json!({}), true);
    let o = find(&recs, "outcome");
    assert_eq!(o[0].severity_number, 0);
    assert_eq!(attr(o[0], "outcome"), &json!({"outcome": "success"}));
}

// Catches: emitting `[]` for an empty tag set or usage list instead of `None`.
#[test]
fn empty_tags_and_usage_are_null_not_empty_lists() {
    let recs = session_records(
        &UploadGate::new(),
        &view(&[], &[], all_on()),
        &tagger(),
        &|_| json!({}),
        true,
    );
    let s = find(&recs, "session report")[0];
    assert_eq!(attr(s, "session.tags"), &Value::Null);
    assert_eq!(attr(s, "usage"), &Value::Null);
}

// Catches: shipping the tags unsorted.
#[test]
fn session_tags_are_sorted() {
    let mut t = tagger();
    t.add("zebra:1", None);
    t.add("apple:2", None);
    t.add("mango:3", None);
    let recs =
        session_records(&UploadGate::new(), &view(&[], &[], all_on()), &t, &|_| json!({}), true);
    let s = find(&recs, "session report")[0];
    assert_eq!(attr(s, "session.tags"), &json!(["apple:2", "mango:3", "zebra:1"]));
}

// Catches: collapsing the two gates (summary on `recording_enabled`, items on `transcript`).
#[test]
fn the_summary_and_transcript_gates_are_independent() {
    let t = SystemTime::UNIX_EPOCH + Duration::from_secs(1);
    let items = vec![ChatItem::Message(ChatMessage::text("i", Role::User, "hi", t))];
    let logs_only = RecordingOptions { logs: true, ..Default::default() };
    assert!(logs_only.recording_enabled());
    let recs = session_records(
        &UploadGate::new(),
        &view(&[], &items, logs_only),
        &tagger(),
        &|_| json!({}),
        true,
    );
    assert_eq!(find(&recs, "session report").len(), 1);
    assert_eq!(find(&recs, "chat item").len(), 0);

    // Nothing enabled: no summary, no transcript -- but see the next test.
    let recs = session_records(
        &UploadGate::new(),
        &view(&[], &items, RecordingOptions::default()),
        &tagger(),
        &|_| json!({}),
        true,
    );
    assert!(recs.is_empty(), "{recs:?}");
}

// Catches: putting the tag/outcome records behind `recording_enabled`.
#[test]
fn tag_and_outcome_records_are_emitted_with_recording_off() {
    let mut t = tagger();
    t.add("voicemail:true", Some(json!({"slot_id": "abc"})));
    t.fail(Some("no answer"));
    let recs = session_records(
        &UploadGate::new(),
        &view(&[], &[], RecordingOptions::default()),
        &t,
        &|_| json!({}),
        true,
    );
    assert_eq!(find(&recs, "session report").len(), 0);
    assert_eq!(find(&recs, "tag").len(), 1);
    assert_eq!(find(&recs, "outcome").len(), 1);
    assert!(t.has_evals(), "a bare fail() must trip the upload gate");
}

// Catches: a tag record for every tag instead of those carrying metadata.
#[test]
fn only_tags_with_non_empty_metadata_produce_a_record() {
    let mut t = tagger();
    t.add("plain:1", None);
    t.add("empty:1", Some(json!({})));
    t.add("rich:1", Some(json!({"k": "v"})));
    let recs =
        session_records(&UploadGate::new(), &view(&[], &[], all_on()), &t, &|_| json!({}), true);
    let tags = find(&recs, "tag");
    assert_eq!(tags.len(), 1, "{tags:?}");
    assert_eq!(attr(tags[0], "tag"), &json!({"name": "rich:1", "metadata": {"k": "v"}}));
}

// Catches: the `started_at or timestamp` chain handled wrongly, including 0.0.
#[test]
fn the_summary_timestamp_falls_through_a_zero_started_at() {
    let mut v = view(&[], &[], all_on());
    let recs = session_records(&UploadGate::new(), &v, &tagger(), &|_| json!({}), true);
    assert_eq!(find(&recs, "session report")[0].timestamp, (100, 0));

    v.started_at = Some(0.0);
    let recs = session_records(&UploadGate::new(), &v, &tagger(), &|_| json!({}), true);
    assert_eq!(find(&recs, "session report")[0].timestamp, (200, 0));
}

// ============================================================================================
// tagger
// ============================================================================================

// Catches: `success()` and `fail()` accumulating instead of replacing.
#[test]
fn the_outcome_tags_are_mutually_exclusive() {
    let mut t = tagger();
    t.success(Some("done"));
    t.fail(Some("actually no"));
    assert_eq!(t.sorted_tags(), vec![TAG_FAIL]);
    assert_eq!(t.outcome(), Some(Outcome::Fail));
    assert_eq!(t.outcome_reason(), Some("actually no"));
}

// Catches: re-adding a tag moving it to the end of the insertion order.
#[test]
fn re_adding_a_tag_keeps_its_position() {
    let mut t = tagger();
    t.add("a:1", Some(json!({"v": 1})));
    t.add("b:1", Some(json!({"v": 2})));
    t.add("a:1", Some(json!({"v": 3})));
    let names: Vec<&str> = t.entries().map(|(n, _)| n).collect();
    assert_eq!(names, vec!["a:1", "b:1"]);
    assert_eq!(t.entries().next().unwrap().1.metadata, Some(json!({"v": 3})));
}

// Catches: judge tags not landing in the tag set.
#[test]
fn recording_an_evaluation_writes_both_a_tag_and_a_record() {
    let mut t = tagger();
    t.record_judgment("booked", "fail", "no", "");
    assert_eq!(t.sorted_tags(), vec!["lk.judge.booked:fail"]);
    assert_eq!(t.evaluations().len(), 1);
    assert_eq!(t.evaluations()[0].name, "booked");
    assert!(t.has_evals());
}

// Catches: tag timestamps from the wall clock rather than the session clock.
#[test]
fn tag_timestamps_come_from_the_session_clock() {
    let clock = std::sync::Arc::new(VirtualClock::new());
    let mut t = Tagger::new(clock.clone());
    t.add("a:1", None);
    clock.advance(Duration::from_secs(10));
    t.add("b:1", None);
    let ts: Vec<SystemTime> = t.entries().map(|(_, e)| e.timestamp).collect();
    // Not exactly 10s: `Clock::wall` nudges by 1 µs per read.
    let delta = ts[1].duration_since(ts[0]).unwrap();
    assert!(delta > Duration::from_millis(9_990) && delta < Duration::from_secs(11), "{ts:?}");
}
