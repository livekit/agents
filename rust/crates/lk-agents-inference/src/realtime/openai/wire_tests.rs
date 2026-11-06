use super::*;
use crate::realtime::dialect::{OPENAI, VOICE_LIVE, XAI};
use crate::realtime::machine::{OptionsPatch, SessionOptions};
use crate::realtime::openai::{
    encode,
    testfix::{TIME, msg},
};
use crate::testkit::NoImages;
use lk_agents_abi::Maybe;
use serde_json::json;

fn value(event: impl serde::Serialize) -> Value {
    serde_json::to_value(event).unwrap()
}

#[test]
fn the_base_url_becomes_a_realtime_websocket_url() {
    for (base, want) in [
        ("wss://api.x.ai/v1/realtime", "wss://api.x.ai/v1/realtime?model=m"),
        ("https://api.openai.com/v1", "wss://api.openai.com/v1/realtime?model=m"),
        ("https://gw.example", "wss://gw.example/realtime?model=m"),
        ("https://gw/v1?next=http://x", "wss://gw/v1/realtime?next=http://x&model=m"),
        ("wss://gw/v1?model=pinned", "wss://gw/v1/realtime?model=pinned"),
    ] {
        assert_eq!(process_base_url(base, "m", None), want);
    }
    let azure = Azure { deployment: Some("dep".into()) };
    assert_eq!(
        process_base_url("https://r.azure.com/openai/v1", "m", Some(&azure)),
        "wss://r.azure.com/openai/v1/realtime?model=dep"
    );
}

#[test]
fn beta_sessions_use_the_flat_schema_and_assistant_text_parts() {
    let opts = SessionOptions::from_dialect(&VOICE_LIVE, false);
    let mut event = encode::session_options("e", &opts, &VOICE_LIVE);
    let shape = WireShape { protocol: Protocol::Beta, beta_event_names: true, drop_model: true };
    shape.outbound(&mut event);
    let event = value(event);
    let session = &event["session"];
    assert_eq!(session["modalities"], json!(["text", "audio"]));
    assert_eq!(session["input_audio_format"], "pcm16");
    assert_eq!(session["output_audio_format"], "pcm16");
    assert_eq!(session["max_response_output_tokens"], "inf");
    assert!(session.get("model").is_none());
    assert!(session.get("audio").is_none());
    assert!(session.get("type").is_none());
    assert!(session.get("tracing").is_none());

    let mut assistant = msg("a", "hi");
    assistant.as_message_mut().unwrap().role = Role::Assistant;
    let mut event = ClientEvent::ItemCreate {
        event_id: "e".into(),
        item: to_wire_item(&assistant, &NoImages).unwrap(),
        previous_item_id: None,
    };
    shape.outbound(&mut event);
    assert_eq!(value(event)["item"]["content"][0]["type"], "text");
}

#[test]
fn beta_event_aliases_are_scoped_to_azure_and_beta() {
    for (legacy, ga, fields) in [
        ("response.text.delta", "response.output_text.delta", json!({"item_id":"m", "delta":"hi"})),
        (
            "response.audio.delta",
            "response.output_audio.delta",
            json!({"item_id":"m", "delta":"AAA="}),
        ),
        (
            "response.audio_transcript.delta",
            "response.output_audio_transcript.delta",
            json!({"item_id":"m", "delta":"hi"}),
        ),
        (
            "conversation.item.created",
            "conversation.item.added",
            json!({"item":{"type":"message","id":"m","role":"user"}}),
        ),
    ] {
        let mut event = fields.clone();
        event["type"] = json!(legacy);
        let beta = decode_text(
            &event.to_string(),
            &WireShape { beta_event_names: true, ..Default::default() },
            TIME,
        )
        .unwrap();
        assert!(matches!(decode(&event), Incoming::Unhandled { .. }));
        event["type"] = json!(ga);
        assert_eq!(beta, decode(&event));
    }
}

#[test]
fn explicit_nulls_clear_settings_and_absent_fields_leave_them_alone() {
    let event = value(encode::options_update(
        "e",
        OptionsPatch {
            transcription: Maybe::Given(None),
            turn_detection: Maybe::Given(None),
            tracing: Maybe::Given(None),
            ..Default::default()
        },
        &OPENAI,
    ));
    let session = &event["session"];
    assert_eq!(session["audio"]["input"], json!({"transcription":null,"turn_detection":null}));
    assert_eq!(session["tracing"], Value::Null);
    assert!(session.get("reasoning").is_none());
    assert!(session["audio"].get("output").is_none());
    let event = value(encode::options_update(
        "e",
        OptionsPatch { tool_choice: Maybe::Given(None), ..Default::default() },
        &OPENAI,
    ));
    assert!(event["session"].get("audio").is_none());
    assert_eq!(event["session"]["tool_choice"], "auto");
}

#[test]
fn xai_sessions_match_its_schema_including_nested_function_tools() {
    let opts = SessionOptions::from_dialect(&XAI, false);
    let event = value(encode::session_options("e", &opts, &XAI));
    let session = &event["session"];
    assert_eq!(session["voice"], "Ara");
    assert_eq!(session["turn_detection"]["type"], "server_vad");
    assert!(session["audio"]["output"].get("voice").is_none());
    assert!(session["audio"]["input"].get("turn_detection").is_none());
    assert!(session["audio"]["input"].get("noise_reduction").is_none());
    for key in ["type", "output_modalities", "max_output_tokens", "tool_choice", "tracing"] {
        assert!(session.get(key).is_none(), "undocumented GA field {key}");
    }
    let tool = lk_agents_abi::ToolSpec {
        name: "weather".into(),
        description: "forecast".into(),
        json_schema: json!({"type":"object"}),
        flags: Default::default(),
        provider_tool: None,
    };
    let event = value(encode::tools_update("e", &[tool], &opts.model, &XAI));
    assert_eq!(
        event["session"]["tools"][0],
        json!({"type":"function", "function":{
            "name":"weather", "description":"forecast", "parameters":{"type":"object"}
        }})
    );
    let event = value(encode::options_update(
        "e",
        OptionsPatch { voice: Maybe::Given(json!("Eve")), ..Default::default() },
        &XAI,
    ));
    assert_eq!(event["session"], json!({"voice":"Eve"}));
}

#[test]
fn item_roles_and_call_ids_survive_replay() {
    for (role, wire_role, part) in [
        (Role::User, "user", "input_text"),
        (Role::System, "system", "input_text"),
        (Role::Developer, "system", "input_text"),
        (Role::Assistant, "assistant", "output_text"),
    ] {
        let mut item = msg("a", "hi");
        item.as_message_mut().unwrap().role = role;
        let item = value(to_wire_item(&item, &NoImages).unwrap());
        assert_eq!(item["role"], wire_role);
        assert_eq!(item["content"][0]["type"], part);
    }
    let long = "a".repeat(64);
    assert_eq!(shorten_call_id(&long).len(), MAX_CALL_ID_LEN);
    assert_eq!(shorten_call_id(&long), shorten_call_id(&long));
    assert_ne!(shorten_call_id(&long), shorten_call_id(&"b".repeat(64)));
    assert_eq!(shorten_call_id("short"), "short");
}

#[test]
fn unknown_events_and_unknown_items_are_ignored_without_requiring_item_fields() {
    for kind in ["mcp_call", "mcp_list_tools", "mcp_approval_request", "future_item"] {
        for event in
            ["conversation.item.added", "response.output_item.added", "response.output_item.done"]
        {
            assert_eq!(
                decode(&json!({"type":event, "item":{"type":kind}})),
                Incoming::Unhandled { event_type: event.into() }
            );
        }
    }
    assert_eq!(
        decode(&json!({"type":"future.event", "arbitrary":true})),
        Incoming::Unhandled { event_type: "future.event".into() }
    );
    assert!(decode_at(&json!({}), TIME).is_err());
}

#[test]
fn null_metadata_and_null_audio_transcripts_are_supported() {
    assert_eq!(
        decode(&json!({"type":"response.created", "response":{"id":"r", "metadata":null}})),
        Incoming::ResponseCreated { response_id: "r".into(), client_event_id: None }
    );
    let Incoming::ItemAdded { item, .. } = decode(
        &json!({"type":"conversation.item.added", "item":{
            "type":"message", "id":"m", "role":"user", "content":[{"type":"input_audio", "transcript":null}]
        }}),
    ) else {
        panic!("item")
    };
    assert!(item.as_message().unwrap().content.is_empty());
    let Incoming::ItemAdded { item, .. } = decode(
        &json!({"type":"conversation.item.added", "item":{
            "type":"message", "id":"m", "role":"user", "content":[{"type":"input_audio", "transcript":"hello"}]
        }}),
    ) else {
        panic!("item")
    };
    assert_eq!(item.as_message().unwrap().raw_text_content().as_deref(), Some("hello"));
}

#[test]
fn malformed_known_fields_fail_with_the_event_identity() {
    for mut event in [
        json!({"type":"response.created", "response":{"id":"r", "metadata":[]}}),
        json!({"type":"conversation.item.added", "item":{
            "type":"message", "id":"m", "role":"user", "content":[{"type":"input_text", "text":5}]
        }}),
        json!({"type":"conversation.item.input_audio_transcription.completed",
            "item_id":"m", "transcript":"hello", "status":5}),
        json!({"type":"conversation.item.input_audio_transcription.completed",
            "item_id":"m", "transcript":"hello", "logprobs":[{}]}),
    ] {
        event["event_id"] = json!("malformed");
        let error = decode_at(&event, TIME).unwrap_err();
        assert_eq!(error.operation, event["type"].as_str().unwrap());
        assert_eq!(error.event_id.as_deref(), Some("malformed"));
    }
}

#[test]
fn ga_output_text_parts_are_text_only() {
    let event = decode(
        &json!({"type":"response.content_part.added", "item_id":"m", "part":{"type":"output_text"}}),
    );
    assert!(matches!(
        event,
        Incoming::ContentPartAdded {
            modalities: lk_agents_abi::llm::realtime::Modalities::TextOnly,
            ..
        }
    ));
}

#[test]
fn transcript_confidence_is_the_geometric_mean_and_xai_progress_is_preserved() {
    let Incoming::TranscriptionCompleted(transcript) = decode(&json!({
        "type":"conversation.item.input_audio_transcription.completed", "item_id":"m", "transcript":"hello",
        "status":"in_progress", "logprobs":[{"logprob":-0.1},{"logprob":-2.0}], "usage":{"type":"duration","seconds":1.5}
    })) else {
        panic!("transcript")
    };
    assert!(transcript.in_progress);
    assert!((transcript.confidence.unwrap() - ((-0.1f64 - 2.0) / 2.0).exp()).abs() < 1e-12);
    assert_eq!(transcript.usage, Some(TranscriptionUsage::Duration { seconds: 1.5 }));
    assert!(
        decode_at(
            &json!({"type":"conversation.item.input_audio_transcription.completed",
        "item_id":"m", "transcript":"", "usage":{"type":"duration","seconds":-1}}),
            TIME
        )
        .is_err()
    );
}

#[test]
fn nested_usage_and_both_error_discriminants_survive_decoding() {
    let Incoming::ResponseDone(done) = decode(&json!({"type":"response.done", "response":{
        "id":"r", "status":"failed", "status_details":{"error":{"type":"invalid_request_error", "code":"quota"}},
        "usage":{"input_tokens":100,"output_tokens":20,"input_token_details":{
            "audio_tokens":80,"text_tokens":20,"cached_tokens":64,"cached_tokens_details":{"audio_tokens":60,"text_tokens":4}
        },"output_token_details":{"audio_tokens":18,"text_tokens":2}}
    }})) else {
        panic!("done")
    };
    assert_eq!(done.usage.input_tokens, 100);
    assert_eq!(done.usage.cache_read_tokens, Some(64));
    assert_eq!(done.usage.cache_read_audio_tokens, Some(60));
    assert_eq!(done.usage.cache_read_image_tokens, None);
    assert_eq!(done.usage.output_audio_tokens, Some(18));
    assert_eq!(done.error_type.as_deref(), Some("invalid_request_error"));
    assert_eq!(done.error_code.as_deref(), Some("quota"));
    let Incoming::Error(error) = decode(&json!({"type":"error", "error":{
        "type":"invalid_request_error","code":"quota","event_id":"e","message":"out of credits"
    }})) else {
        panic!("error")
    };
    assert_eq!(error.message, "out of credits");
    assert_eq!(error.event_id.as_deref(), Some("e"));
    assert_eq!(error.type_.as_deref(), Some("invalid_request_error"));
    assert_eq!(error.code.as_deref(), Some("quota"));
}
