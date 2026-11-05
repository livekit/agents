use super::*;
use lk_agents_abi::telemetry::record::SpanContext;
use lk_agents_abi::telemetry::record::{LogRecord, Value as Otlp};
use lk_agents_abi::telemetry::report::{
    EvaluationRecord, Outcome, RecordingOptions, ReportOutcome, ReportTag, SessionReport,
};
use lk_agents_abi::usage::{LlmUsage, ModelUsage, TokenUsage, Usage};
use lk_agents_abi::{
    AgentHandoffItem, ChatContent, ChatItem, ChatMessage, FunctionCall, FunctionCallOutput,
    MetricsReport, Role,
};
use serde_json::{Value, json};
use std::time::{Duration, UNIX_EPOCH};

fn at(secs: f64) -> std::time::SystemTime {
    UNIX_EPOCH + Duration::from_secs_f64(secs)
}

fn message(id: &str, role: Role, text: &str, created: f64) -> ChatItem {
    ChatItem::Message(ChatMessage {
        id: id.into(),
        role,
        content: vec![ChatContent::Text(text.into())],
        interrupted: false,
        transcript_confidence: None,
        extra: Default::default(),
        metrics: MetricsReport::default(),
        created_at: at(created),
    })
}

fn report() -> SessionReport {
    let mut user = message("item_1", Role::User, "hello", 1_700_000_001.25);
    if let ChatItem::Message(m) = &mut user {
        m.transcript_confidence = Some(0.5);
        m.extra =
            serde_json::from_value(json!({"n": 1, "s": "x", "b": true, "o": {"k": [1, "it's"]}}))
                .unwrap();
        m.metrics.started_speaking_at = Some(1_700_000_000.5);
        m.metrics.e2e_latency = Some(1.25);
    }
    SessionReport {
        job_id: "AJ_1".into(),
        room_id: "RM_1".into(),
        room: "room".into(),
        agent_name: "front-desk".into(),
        options: json!({
            "interruption": {"enabled": true, "min_duration": 0.5},
            "max_tool_steps": 3,
            "stt_context_options": {"keyterms": ["LiveKit"], "keyterm_detection": {"instructions": "private"}},
            "recording_options": {"audio": true, "transcript": true},
        }),
        events: vec![
            json!({"type": "error", "error": {"type": "llm_error"}}),
            json!({"type": "error", "error": {"type": "stt_error"}}),
            json!({"type": "close", "reason": "task_completed"}),
        ],
        chat_history: vec![
            user,
            ChatItem::FunctionCall(FunctionCall {
                id: "item_2".into(),
                call_id: "call_1".into(),
                arguments: "{}".into(),
                name: "book".into(),
                created_at: at(1_700_000_002.0),
                extra: Default::default(),
                group_id: None,
            }),
            ChatItem::FunctionCallOutput(FunctionCallOutput {
                id: "item_3".into(),
                name: "book".into(),
                call_id: "call_1".into(),
                output: "no slot".into(),
                is_error: true,
                created_at: at(1_700_000_002.5),
                reply_required: false,
            }),
            ChatItem::AgentHandoff(AgentHandoffItem {
                id: "item_4".into(),
                old_agent_id: None,
                new_agent_id: "desk".into(),
                created_at: at(1_700_000_003.0),
            }),
        ],
        turns: [("item_1".to_owned(), SpanContext { trace_id: 5, span_id: 6, sampled: true })]
            .into(),
        usage: Usage {
            models: vec![ModelUsage::Llm(LlmUsage {
                provider: Some("openai".into()),
                model: Some("gpt".into()),
                requests: 2,
                tokens: TokenUsage { input_tokens: 10, output_tokens: 5, ..Default::default() },
                ..Default::default()
            })],
            ..Default::default()
        },
        tags: vec![
            ReportTag { name: "z:1".into(), metadata: None, timestamp: 1_700_000_004.0 },
            ReportTag {
                name: "a:1".into(),
                metadata: Some(json!({"slot": "abc"})),
                timestamp: 1_700_000_005.0,
            },
            ReportTag { name: "e:1".into(), metadata: Some(json!({})), timestamp: 1_700_000_006.0 },
        ],
        evaluations: vec![EvaluationRecord {
            name: "task".into(),
            tag: "lk.judge.task:fail".into(),
            verdict: "fail".into(),
            reasoning: "why".into(),
            instructions: "how".into(),
        }],
        outcome: Some(ReportOutcome { outcome: Outcome::Fail, reason: Some("hung up".into()) }),
        started_at: Some(1_700_000_000.0),
        timestamp: 1_700_000_010.0,
        duration: Some(9.5),
        audio_recording_path: Some("/tmp/recording.ogg".into()),
        audio_recording_started_at: Some(1_700_000_000.25),
        sdk_version: "2.0.0".into(),
    }
}

/// An OTLP value as JSON, to compare whole attributes.
fn json_of(v: &Otlp) -> Value {
    match v {
        Otlp::Bool(b) => json!(b),
        Otlp::Int(i) => json!(i),
        Otlp::Double(d) => json!(d),
        Otlp::String(s) => json!(s),
        Otlp::Bytes(b) => json!(b),
        Otlp::Array(a) => Value::Array(a.iter().map(json_of).collect()),
        Otlp::KvList(kv) => {
            Value::Object(kv.iter().map(|(k, v)| (k.clone(), json_of(v))).collect())
        }
    }
}

fn attrs(r: &LogRecord) -> Value {
    Value::Object(r.attributes.iter().map(|(k, v)| (k.to_string(), json_of(v))).collect())
}

fn shape(records: &[LogRecord]) -> Vec<(String, String)> {
    records
        .iter()
        .map(|r| (r.scope.to_string(), json_of(&r.body).as_str().unwrap().to_owned()))
        .collect()
}

// Catches: a record's scope, body, order, timestamp or attribute shape drifting from 1.8.10's.
#[test]
fn records_match_the_1_8_10_shapes() {
    let records = records(&report(), &RecordingOptions::default());
    let expected: Vec<(String, String)> = [
        ("chat_history", "session report"),
        ("chat_history", "chat item"),
        ("chat_history", "chat item"),
        ("chat_history", "chat item"),
        ("chat_history", "chat item"),
        ("evaluations", "evaluation"),
        ("evaluations", "tag"),
        ("evaluations", "outcome"),
    ]
    .iter()
    .map(|(s, b)| (s.to_string(), b.to_string()))
    .collect();
    assert_eq!(shape(&records), expected);
    assert!(records.iter().all(|r| r.job.as_ref().unwrap().0 == "AJ_1"));

    let summary = &records[0];
    assert_eq!(summary.time_unix_nanos, 1_700_000_000_000_000_000);
    assert_eq!(
        attrs(summary),
        json!({
            "session.options": {
                "interruption": {"enabled": true, "min_duration": 0.5},
                "max_tool_steps": 3,
                "stt_context_options": {"lk.pii.keyterms": ["LiveKit"], "keyterm_detection": {}},
                "recording_options": {"audio": true, "transcript": true},
            },
            "session.report_timestamp": 1_700_000_010.0,
            "session.tags": ["a:1", "e:1", "z:1"],
            "agent_name": "front-desk",
            "sdk_version": "2.0.0",
            "usage": [{
                "type": "llm", "provider": "openai", "model": "gpt", "requests": 2,
                "unmeasured_requests": 0, "tokens": {"input_tokens": 10, "output_tokens": 5},
            }],
        })
    );

    let user = &records[1];
    assert_eq!(user.time_unix_nanos, 1_700_000_001_250_000_000);
    // Catches: a chat item not linked to the span of the turn it belongs to.
    let turn = user.span.expect("the item's turn");
    assert_eq!((turn.trace_id, turn.span_id), (5, 6));
    assert_eq!(records[2].span, None, "an item with no turn");
    assert_eq!(user.severity_number, 0);
    assert_eq!(user.severity_text, "unspecified");
    assert_eq!(
        attrs(user),
        json!({"chat.item": {"message": {
            "id": "item_1",
            "role": "USER",
            "content": [{"text": "hello"}],
            "transcriptConfidence": 0.5,
            "extra": {"n": "1", "s": "\"x\"", "b": "true", "o": "{\"k\":[1,\"it's\"]}"},
            "metrics": {"startedSpeakingAt": "2023-11-14T22:13:20.500+00:00", "e2eLatency": 1.25},
            "createdAt": "2023-11-14T22:13:21.250+00:00",
        }}})
    );
    assert_eq!(
        attrs(&records[2]),
        json!({"chat.item": {"functionCall": {
            "id": "item_2", "callId": "call_1", "arguments": "{}", "name": "book",
            "createdAt": "2023-11-14T22:13:22+00:00",
        }}})
    );
    let output = &records[3];
    assert_eq!((output.severity_number, output.severity_text.as_ref()), (17, "error"));
    assert_eq!(
        attrs(output),
        json!({"chat.item": {"functionCallOutput": {
            "id": "item_3", "name": "book", "callId": "call_1", "output": "no slot",
            "isError": true, "createdAt": "2023-11-14T22:13:22.500+00:00",
        }}})
    );
    assert_eq!(
        attrs(&records[4]),
        json!({"chat.item": {"agentHandoff": {
            "id": "item_4", "newAgentId": "desk", "createdAt": "2023-11-14T22:13:23+00:00",
        }}})
    );

    let evaluation = &records[5];
    assert_eq!(
        (evaluation.severity_number, evaluation.time_unix_nanos),
        (17, 1_700_000_010_000_000_000)
    );
    assert_eq!(
        attrs(evaluation),
        json!({"evaluation": {
            "name": "task", "tag": "lk.judge.task:fail", "verdict": "fail",
            "reasoning": "why", "instructions": "how",
        }})
    );
    // Only the tag with metadata; an empty mapping is no metadata.
    assert_eq!(records[6].time_unix_nanos, 1_700_000_005_000_000_000);
    assert_eq!(attrs(&records[6]), json!({"tag": {"name": "a:1", "metadata": {"slot": "abc"}}}));
    assert_eq!(records[7].severity_number, 17);
    assert_eq!(attrs(&records[7]), json!({"outcome": {"outcome": "fail", "reason": "hung up"}}));
}

// Catches: a recorded part's switch turning on or off an output it does not own.
#[test]
fn recording_options_decide_the_outputs() {
    let report = report();
    let off = RecordingOptions {
        audio: false,
        traces: false,
        logs: false,
        transcript: false,
        redaction: false,
    };
    let cases = [
        // (options, summary record, chat item records, upload transcript, upload audio)
        (RecordingOptions::default(), true, true, true, true),
        (RecordingOptions { audio: false, ..Default::default() }, true, true, true, false),
        (RecordingOptions { transcript: false, ..Default::default() }, true, false, false, true),
        (RecordingOptions { traces: true, ..off }, true, false, false, false),
        (off, false, false, false, false),
        (RecordingOptions { redaction: true, ..off }, false, false, true, false),
        (RecordingOptions { redaction: true, ..Default::default() }, true, true, true, true),
    ];
    for (options, summary, items, transcript, audio) in cases {
        let records = records(&report, &options);
        let bodies: Vec<String> = shape(&records).into_iter().map(|(_, b)| b).collect();
        assert_eq!(bodies.contains(&"session report".to_owned()), summary, "{options:?}");
        assert_eq!(bodies.contains(&"chat item".to_owned()), items, "{options:?}");
        // Evaluations, tags and the outcome never depend on the recorded parts.
        assert!(bodies.ends_with(&["evaluation".into(), "tag".into(), "outcome".into()]));
        let upload = upload(&report, &options, false);
        assert_eq!(
            upload.as_ref().is_some_and(|u| u.chat_history.is_some()),
            transcript,
            "{options:?}"
        );
        assert_eq!(upload.as_ref().is_some_and(|u| u.audio_path.is_some()), audio, "{options:?}");
        if let Some(upload) = upload {
            assert_eq!(upload.header.redaction_enabled, options.redaction);
        }
    }
}

// Catches: a redacted session that records no transcript leaving Cloud nothing to rebuild the
// transcript (and mute the audio) from.
#[test]
fn a_redacted_session_always_uploads_chat_history() {
    let options = RecordingOptions { transcript: false, redaction: true, ..Default::default() };
    let upload = upload(&report(), &options, true).unwrap();
    let history: Value = serde_json::from_str(upload.chat_history.as_ref().unwrap()).unwrap();
    assert_eq!(history["items"].as_array().unwrap().len(), 4);
    assert!(upload.header.redaction_enabled && upload.header.simulated);
    assert!(upload.audio_path.is_some());
}

// Catches: the transcript, header or summary drifting from the ingest contract.
#[test]
fn upload_matches_the_ingest_contract() {
    let upload = upload(&report(), &RecordingOptions::default(), false).unwrap();
    let header = &upload.header;
    assert_eq!((header.room_id.as_str(), header.job_id.as_str()), ("RM_1", "AJ_1"));
    assert_eq!(header.start_time_millis, 1_700_000_000_250);
    assert_eq!(header.duration_millis, 9_500);
    assert_eq!(upload.audio_path.as_deref(), Some("/tmp/recording.ogg"));

    let history: Value = serde_json::from_str(upload.chat_history.as_ref().unwrap()).unwrap();
    let items = history["items"].as_array().unwrap();
    let types: Vec<&str> = items.iter().map(|i| i["type"].as_str().unwrap()).collect();
    assert_eq!(types, ["message", "function_call", "function_call_output", "agent_handoff"]);
    let user = &items[0];
    let mut keys: Vec<&str> = user.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort();
    assert_eq!(
        keys,
        [
            "content",
            "created_at",
            "extra",
            "id",
            "interrupted",
            "metrics",
            "role",
            "transcript_confidence",
            "type"
        ]
    );
    assert_eq!(user["created_at"], json!(1_700_000_001.25));
    assert_eq!(user["metrics"]["started_speaking_at"], json!(1_700_000_000.5));

    let summary: Value = serde_json::from_str(&header.session_summary).unwrap();
    assert_eq!(
        summary,
        json!({
            "job_id": "AJ_1",
            "room_id": "RM_1",
            "room": "room",
            "audio_recording_path": "/tmp/recording.ogg",
            "audio_recording_started_at": 1_700_000_000.25,
            "options": {
                "allow_interruptions": true,
                "discard_audio_if_uninterruptible": null,
                "min_interruption_duration": 0.5,
                "min_interruption_words": null,
                "min_endpointing_delay": null,
                "max_endpointing_delay": null,
                "max_tool_steps": 3,
                "user_away_timeout": null,
                "min_consecutive_speech_delay": null,
                "preemptive_generation": null,
                "recording_options": {"audio": true, "transcript": true},
            },
            "timestamp": 1_700_000_010.0,
            "usage": [{
                "type": "llm", "provider": "openai", "model": "gpt", "requests": 2,
                "unmeasured_requests": 0, "tokens": {"input_tokens": 10, "output_tokens": 5},
            }],
            "sdk_version": "2.0.0",
            "agent_name": "front-desk",
            "tags": ["a:1", "e:1", "z:1"],
            "duration_ms": 9_500,
            "user_turns": 1,
            "errors": {"llm": 1, "stt": 1, "tts": 0, "realtime": 0, "interruption": 0},
            "close_reason": "CR_COMPLETED",
        })
    );
}

// Catches: an oversized transcript failing the upload, dropping the audio, or dropping newer
// items before older ones without saying so.
#[test]
fn an_oversized_transcript_drops_its_oldest_items() {
    let mut report = report();
    let big = "x".repeat(300 * 1024);
    report.chat_history =
        (0..10).map(|n| message(&format!("item_{n}"), Role::User, &big, n as f64)).collect();
    let uploaded = upload(&report, &RecordingOptions::default(), false).unwrap();
    let json = uploaded.chat_history.unwrap();
    assert!(json.len() <= TRANSCRIPT_BYTES);
    let history: Value = serde_json::from_str(&json).unwrap();
    let ids: Vec<&str> =
        history["items"].as_array().unwrap().iter().map(|i| i["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["item_4", "item_5", "item_6", "item_7", "item_8", "item_9"]);
    let summary: Value = serde_json::from_str(&uploaded.header.session_summary).unwrap();
    assert_eq!(summary["omitted_chat_items"], 4);
    assert_eq!(summary["user_turns"], 10);
    assert!(uploaded.audio_path.is_some());

    let mut many = self::report();
    many.chat_history = (0..TRANSCRIPT_ITEMS + 5)
        .map(|n| message(&format!("item_{n}"), Role::User, "hi", n as f64))
        .collect();
    let upload = upload(&many, &RecordingOptions::default(), false).unwrap();
    let summary: Value = serde_json::from_str(&upload.header.session_summary).unwrap();
    assert_eq!(summary["omitted_chat_items"], 5);
}

// Catches: a summary past the 16 KiB the header's consumer accepts.
#[test]
fn the_summary_stays_within_its_bound() {
    let mut report = report();
    report.options["recording_options"] = json!({"padding": "x".repeat(20_000)});
    let upload = upload(&report, &RecordingOptions::default(), false).unwrap();
    let summary = &upload.header.session_summary;
    assert!(summary.len() <= lk_agents_abi::telemetry::report::SESSION_SUMMARY_BYTES);
    let summary: Value = serde_json::from_str(summary).unwrap();
    assert_eq!(summary["options"], Value::Null);
    assert_eq!(summary["tags"], json!(["a:1", "e:1", "z:1"]));
}
