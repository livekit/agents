use std::time::{Duration, SystemTime};

use lk_agents_abi::usage::{LlmUsage, ModelUsage, TokenUsage, Usage};
use lk_agents_abi::{ChatItem, ChatMessage, Role};
use serde_json::{Value, json};

use lk_agents_abi::telemetry::report::UploadGate as _;

use super::*;

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
        usage: Usage { models: usage.to_vec(), ..Default::default() },
        chat_history: items.to_vec(),
        omitted_events: 0,
    }
}

fn all_on() -> RecordingOptions {
    RecordingOptions { audio: true, traces: true, logs: true, transcript: true, redaction: false }
}

fn tagger() -> Tagger {
    Tagger::default()
}

const AT: SystemTime = SystemTime::UNIX_EPOCH;

fn find<'a>(recs: &'a [LogRecord], body: &str) -> Vec<&'a LogRecord> {
    recs.iter().filter(|r| r.body == body).collect()
}

fn attr<'a>(r: &'a LogRecord, k: &str) -> &'a Value {
    r.attributes.get(k).expect("missing attribute")
}

// usage.md "Implementation boundaries": the report carries the normalized record, an absent
// breakdown left out and a reported zero kept.
#[test]
fn report_usage_omits_absent_fields_and_keeps_reported_zeros() {
    let u = ModelUsage::Llm(LlmUsage {
        provider: Some("openai".into()),
        model: None,
        requests: 1,
        tokens: TokenUsage { input_tokens: 100, cache_read_tokens: Some(0), ..Default::default() },
        ..Default::default()
    });
    let v = usage_attributes(&u);
    assert_eq!(
        v,
        json!({"type": "llm", "provider": "openai", "requests": 1, "unmeasured_requests": 0,
               "tokens": {"input_tokens": 100, "output_tokens": 0, "cache_read_tokens": 0}})
    );
}

#[test]
fn denied_content_never_reaches_session_projection_or_report_records() {
    let mut t = tagger();
    t.add("customer-tag".into(), Some(json!({"private": "customer metadata"})), AT);
    t.fail(Some("customer outcome reason".into()), AT);
    t.record_judgment("booking".into(), "fail".into(), "customer reasoning".into(), "".into(), AT);
    let at = SystemTime::UNIX_EPOCH;
    let items = [ChatItem::Message(ChatMessage::text("i", Role::User, "customer transcript", at))];
    for (capture_content, redaction) in [(false, false), (true, true)] {
        let mut report = view(&[], &items, RecordingOptions { redaction, ..all_on() });
        report.options = json!({"instructions": "customer prompt"});
        let records = session_records(
            false,
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
    t.record_judgment("politeness".into(), "pass".into(), "fine".into(), "".into(), AT);
    t.record_judgment("booked".into(), "fail".into(), "never booked".into(), "".into(), AT);
    t.record_judgment("unsure".into(), "maybe".into(), "hmm".into(), "".into(), AT);

    let recs = session_records(false, &view(&[], &[], all_on()), &t, &|_| json!({}), true);
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
    t.fail(Some("user hung up".into()), AT);
    let recs = session_records(false, &view(&[], &[], all_on()), &t, &|_| json!({}), true);
    let o = find(&recs, "outcome");
    assert_eq!(o.len(), 1);
    assert_eq!(o[0].severity_number, 17);
    assert_eq!(attr(o[0], "outcome"), &json!({"outcome": "fail", "reason": "user hung up"}));

    let mut t = tagger();
    t.success(None, AT);
    let recs = session_records(false, &view(&[], &[], all_on()), &t, &|_| json!({}), true);
    let o = find(&recs, "outcome");
    assert_eq!(o[0].severity_number, 0);
    assert_eq!(attr(o[0], "outcome"), &json!({"outcome": "success"}));
}

// Catches: emitting `[]` for an empty tag set or usage list instead of `None`.
#[test]
fn empty_tags_and_usage_are_null_not_empty_lists() {
    let recs = session_records(false, &view(&[], &[], all_on()), &tagger(), &|_| json!({}), true);
    let s = find(&recs, "session report")[0];
    assert_eq!(attr(s, "session.tags"), &Value::Null);
    assert_eq!(attr(s, "usage"), &Value::Null);
}

// Catches: shipping the tags unsorted.
#[test]
fn session_tags_are_sorted() {
    let mut t = tagger();
    t.add("zebra:1".into(), None, AT);
    t.add("apple:2".into(), None, AT);
    t.add("mango:3".into(), None, AT);
    let recs = session_records(false, &view(&[], &[], all_on()), &t, &|_| json!({}), true);
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
    let recs =
        session_records(false, &view(&[], &items, logs_only), &tagger(), &|_| json!({}), true);
    assert_eq!(find(&recs, "session report").len(), 1);
    assert_eq!(find(&recs, "chat item").len(), 0);

    // Nothing enabled: no summary, no transcript -- but see the next test.
    let recs = session_records(
        false,
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
    t.add("voicemail:true".into(), Some(json!({"slot_id": "abc"})), AT);
    t.fail(Some("no answer".into()), AT);
    let recs = session_records(
        false,
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
    t.add("plain:1".into(), None, AT);
    t.add("empty:1".into(), Some(json!({})), AT);
    t.add("rich:1".into(), Some(json!({"k": "v"})), AT);
    let recs = session_records(false, &view(&[], &[], all_on()), &t, &|_| json!({}), true);
    let tags = find(&recs, "tag");
    assert_eq!(tags.len(), 1, "{tags:?}");
    assert_eq!(attr(tags[0], "tag"), &json!({"name": "rich:1", "metadata": {"k": "v"}}));
}

// Catches: the `started_at or timestamp` chain handled wrongly, including 0.0.
#[test]
fn the_summary_timestamp_falls_through_a_zero_started_at() {
    let mut v = view(&[], &[], all_on());
    let recs = session_records(false, &v, &tagger(), &|_| json!({}), true);
    assert_eq!(find(&recs, "session report")[0].timestamp, (100, 0));

    v.started_at = Some(0.0);
    let recs = session_records(false, &v, &tagger(), &|_| json!({}), true);
    assert_eq!(find(&recs, "session report")[0].timestamp, (200, 0));
}

// ============================================================================================
// tagger
// ============================================================================================

// Catches: `success()` and `fail()` accumulating instead of replacing.
#[test]
fn the_outcome_tags_are_mutually_exclusive() {
    let mut t = tagger();
    t.success(Some("done".into()), AT);
    t.fail(Some("actually no".into()), AT);
    assert_eq!(t.sorted_tags(), vec![TAG_FAIL]);
    assert_eq!(t.outcome(), Some(Outcome::Fail));
    assert_eq!(t.outcome_reason(), Some("actually no"));
}

// Catches: re-adding a tag moving it to the end of the insertion order.
#[test]
fn re_adding_a_tag_keeps_its_position() {
    let mut t = tagger();
    t.add("a:1".into(), Some(json!({"v": 1})), AT);
    t.add("b:1".into(), Some(json!({"v": 2})), AT);
    t.add("a:1".into(), Some(json!({"v": 3})), AT);
    let names: Vec<String> = t.entries().into_iter().map(|(n, _, _)| n).collect();
    assert_eq!(names, vec!["a:1", "b:1"]);
    assert_eq!(t.entries()[0].1, Some(json!({"v": 3})));
}

// ============================================================================================
// the upload gate
// ============================================================================================

#[test]
fn incremental_chat_records_keep_supplied_identity_time_and_enforce_recording_consent() {
    let item = ChatItem::Message(lk_agents_abi::ChatMessage::new(
        "message-a",
        lk_agents_abi::Role::User,
        vec![lk_agents_abi::ChatContent::Text("private".into())],
        std::time::UNIX_EPOCH + std::time::Duration::from_nanos(123_456),
    ));
    let identity: (String, String, String) = ("job-a".into(), "room-a".into(), "room-name".into());
    let projected = json!({"id": "message-a", "content": ["private"]});
    let mut gate = UploadGate::new();
    let mut options = RecordingOptions { transcript: true, ..Default::default() };
    let record = session_chat_record(
        gate.disabled(),
        &options,
        item.clone(),
        projected.clone(),
        true,
        identity.clone(),
    )
    .unwrap();
    assert_eq!(record.timestamp_nanos, 123_456);
    assert_eq!(record.attributes["chat.item"], projected);
    assert_eq!(record.attributes["job_id"], "job-a");
    assert_eq!(record.attributes["lk.telemetry.capture_content"], true);
    assert!(
        session_chat_record(
            gate.disabled(),
            &options,
            item.clone(),
            projected.clone(),
            false,
            identity.clone()
        )
        .is_none()
    );
    options.redaction = true;
    assert!(
        session_chat_record(
            gate.disabled(),
            &options,
            item.clone(),
            projected.clone(),
            true,
            identity.clone()
        )
        .is_none()
    );
    options.redaction = false;
    options.transcript = false;
    assert!(
        session_chat_record(
            gate.disabled(),
            &options,
            item.clone(),
            projected.clone(),
            true,
            identity.clone()
        )
        .is_none()
    );
    options.transcript = true;
    gate.disable();
    assert!(
        session_chat_record(gate.disabled(), &options, item, projected, true, identity.clone())
            .is_none()
    );
}

#[test]
fn large_tag_state_cannot_create_an_unbounded_sdk_report_or_remove_the_business_outcome() {
    let mut tagger = Tagger::default();
    for index in 0..1_000 {
        tagger.add(format!("tag-{index}"), Some(json!({"detail": "value"})), AT);
    }
    tagger.success(Some("completed".into()), AT);
    let report = SessionReportInput {
        recording_options: RecordingOptions { logs: true, ..Default::default() },
        ..Default::default()
    };
    let records = session_records(false, &report, &tagger, &|_| Value::Null, true);
    assert!(records.len() <= super::policy::SDK_PENDING_RECORDS);
    assert!(
        records.iter().any(|record| record.body == "outcome"
            && record.attributes["outcome"]["outcome"] == "success")
    );
    assert!(records[0].attributes["lk.telemetry.omitted_report_entries"].as_u64().unwrap() > 0);
    assert_eq!(tagger.sorted_tags().len(), 1_001, "business tag state remains application-owned");
    assert!(records.iter().all(|record| report_value_fits(&record.attributes)));
}

// Catches: gating on the status code alone (a missing-grant 401 would disable uploads).
#[test]
fn only_the_recording_disabled_401_closes_the_gate() {
    assert!(UploadGate::is_disabled_response(
        401,
        b"data recording is disabled for project".to_vec()
    ));
    assert!(UploadGate::is_disabled_response(401, b"Disabled By Owner".to_vec()));
    assert!(!UploadGate::is_disabled_response(401, b"missing project id".to_vec()));
    assert!(!UploadGate::is_disabled_response(
        401,
        b"operation requires observability write grant".to_vec()
    ));
    // Right text, wrong code -- the reference checks the code first.
    assert!(!UploadGate::is_disabled_response(403, b"data recording is disabled".to_vec()));
    assert!(!UploadGate::is_disabled_response(500, b"data recording is disabled".to_vec()));
}
