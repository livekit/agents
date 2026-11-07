use std::sync::Arc;

use crate::RefRegistry;
use serde_json::json;

use super::*;
use crate::llm::chat_context::{
    AgentConfigUpdate, AgentHandoffItem, AudioContent, FunctionCall, FunctionCallOutput,
    ImageContent, MetricsMetadata, MetricsReport,
};
use crate::testkit::{t, text};
use crate::utils::video::FakeFrame;

struct NoopTable;
impl RefRegistry for NoopTable {
    fn release(&self, _id: u64) {}
}

fn media_ref(id: u64) -> MediaRef {
    let registry: Arc<dyn RefRegistry> = Arc::new(NoopTable);
    MediaRef::new(id, &registry)
}

/// Stands in for the host SDK's own frame serializer.
fn fake_media(r: &MediaRef) -> Value {
    json!({ "ref": r.id() })
}

fn fake_frame(frame: &dyn RtcVideoFrame) -> Value {
    json!({ "ref": FakeFrame::id_of(frame) })
}

fn host() -> ToDictHost<'static> {
    ToDictHost { media: Some(&fake_media), frame: Some(&fake_frame), strip_expr_markup: None }
}

fn dump_with(items: Vec<ChatItem>, opts: ToDictOptions, host: &ToDictHost<'_>) -> Dumped {
    items.model_dump(&DumpCx { opts: &opts, host }, "ChatItem")
}

fn dump(items: Vec<ChatItem>, opts: ToDictOptions) -> Value {
    json!({ "items": dump_with(items, opts, &host()).unwrap() })
}

/// Serialized text: an `IndexMap`'s `PartialEq` ignores key order, so every order assertion
/// compares through here.
fn as_text(v: &Value) -> String {
    serde_json::to_string(v).unwrap()
}

fn msg(role: Role, content: Vec<ChatContent>) -> ChatItem {
    ChatMessage::new("item_1", role, content, t(7)).into()
}

// Catches: fields emitted in anything but declaration order.
#[test]
fn keys_come_out_in_pydantic_declaration_order() {
    let opts = ToDictOptions { exclude_timestamp: false, ..Default::default() };
    let d = dump(vec![msg(Role::User, vec![text("hi")])], opts);
    assert_eq!(
        as_text(&d),
        r#"{"items":[{"id":"item_1","type":"message","role":"user","content":["hi"],"interrupted":false,"extra":{},"metrics":{},"created_at":7.0}]}"#
    );

    let call = FunctionCall {
        id: "fc".into(),
        call_id: "c".into(),
        arguments: "{}".into(),
        name: "n".into(),
        created_at: t(7),
        extra: Default::default(),
        // Set, so that `group_id` moving is a failure rather than a `None` cancelling out.
        group_id: Some("grp".into()),
    };
    let d = dump(vec![call.into()], opts);
    assert_eq!(
        as_text(&d),
        r#"{"items":[{"id":"fc","type":"function_call","call_id":"c","arguments":"{}","name":"n","created_at":7.0,"extra":{},"group_id":"grp"}]}"#
    );

    let out = FunctionCallOutput::new("fo", "c", "done", false, t(7));
    let d = dump(vec![out.into()], opts);
    assert_eq!(
        as_text(&d),
        r#"{"items":[{"id":"fo","type":"function_call_output","name":"","call_id":"c","output":"done","is_error":false,"created_at":7.0,"reply_required":true}]}"#
    );

    let h = AgentHandoffItem {
        id: "h".into(),
        old_agent_id: Some("a".into()),
        new_agent_id: "b".into(),
        created_at: t(7),
    };
    let d = dump(vec![h.into()], opts);
    assert_eq!(
        as_text(&d),
        r#"{"items":[{"id":"h","type":"agent_handoff","old_agent_id":"a","new_agent_id":"b","created_at":7.0}]}"#
    );

    let u = AgentConfigUpdate {
        id: "u".into(),
        instructions: Some("be brief".into()),
        tools_added: Some(vec![]),
        tools_removed: Some(vec!["x".into()]),
        created_at: t(7),
        tools: None,
    };
    let d = dump(vec![u.into()], opts);
    assert_eq!(
        as_text(&d),
        r#"{"items":[{"id":"u","type":"agent_config_update","instructions":"be brief","tools_added":[],"tools_removed":["x"],"created_at":7.0}]}"#
    );
}

// Catches: `metrics` emitted in population order. All fifteen keys are set on purpose: a
// partial set let a reorder survive.
#[test]
fn metrics_follow_declaration_order_not_assignment_order() {
    let meta = |n: &str| {
        Some(MetricsMetadata {
            // Reversed against declaration order at the construction site as well.
            model_provider: Some(format!("{n}_p")),
            model_name: Some(format!("{n}_n")),
        })
    };
    let mut m = ChatMessage::new("item_1", Role::Assistant, vec![], t(7));
    // Field-init order here is deliberately not declaration order either.
    m.metrics = MetricsReport {
        stt_metadata: meta("stt"),
        tts_metadata: meta("tts"),
        llm_metadata: meta("llm"),
        provider_request_ids: Some(vec!["r1".into()]),
        e2e_latency: Some(11.0),
        playback_latency: Some(10.0),
        tts_node_ttfb: Some(9.0),
        llm_node_ttfs: Some(8.0),
        llm_node_tps: Some(7.0),
        llm_node_ttft: Some(6.0),
        on_user_turn_completed_delay: Some(5.0),
        end_of_turn_delay: Some(4.0),
        transcription_delay: Some(3.0),
        stopped_speaking_at: Some(2.0),
        started_speaking_at: Some(1.0),
    };
    let d = dump(vec![m.into()], ToDictOptions::default());
    assert!(
        as_text(&d).contains(concat!(
            r#""metrics":{"started_speaking_at":1.0,"stopped_speaking_at":2.0,"#,
            r#""transcription_delay":3.0,"end_of_turn_delay":4.0,"#,
            r#""on_user_turn_completed_delay":5.0,"llm_node_ttft":6.0,"llm_node_tps":7.0,"#,
            r#""llm_node_ttfs":8.0,"tts_node_ttfb":9.0,"playback_latency":10.0,"#,
            r#""e2e_latency":11.0,"provider_request_ids":["r1"],"#,
            r#""llm_metadata":{"model_name":"llm_n","model_provider":"llm_p"},"#,
            r#""tts_metadata":{"model_name":"tts_n","model_provider":"tts_p"},"#,
            r#""stt_metadata":{"model_name":"stt_n","model_provider":"stt_p"}}"#
        )),
        "{}",
        as_text(&d)
    );
}

// Catches: collapsing `Some(vec![])` and `None` on `provider_request_ids` (and `tools_added`).
#[test]
fn empty_list_and_absent_list_are_different_answers() {
    let mut m = ChatMessage::new("item_1", Role::Assistant, vec![], t(7));
    m.metrics.provider_request_ids = Some(vec![]);
    assert!(
        as_text(&dump(vec![m.clone().into()], ToDictOptions::default()))
            .contains(r#""provider_request_ids":[]"#)
    );

    m.metrics.provider_request_ids = None;
    assert!(
        !as_text(&dump(vec![m.into()], ToDictOptions::default())).contains("provider_request_ids")
    );
}

// Catches: `exclude_timestamp`/`exclude_metrics` implemented as an item filter.
#[test]
fn timestamp_and_metrics_exclusions_are_field_wide_not_message_only() {
    let items = vec![
        msg(Role::User, vec![]),
        crate::testkit::call("fc", "n", 7),
        crate::testkit::handoff("h", 7),
    ];
    let opts =
        ToDictOptions { exclude_timestamp: true, exclude_metrics: true, ..Default::default() };
    let text = as_text(&dump(items, opts));
    assert!(!text.contains("created_at"), "{text}");
    assert!(!text.contains("metrics"), "{text}");
}

// Catches: a hand-rolled `ImageContent`/`AudioContent` shape.
#[test]
fn media_content_matches_the_two_models_field_for_field() {
    // Every optional field set, so swapped fields cannot cancel out as two `None`s.
    let img = ImageContent {
        id: "img_1".into(),
        image: ImageSource::Frame(FakeFrame::shared(4)),
        inference_width: Some(10),
        inference_height: Some(20),
        inference_detail: InferenceDetail::High,
        mime_type: Some("image/png".into()),
    };
    let audio = AudioContent { frame: media_ref(5), transcript: Some("said".into()) };
    let items = vec![msg(Role::User, vec![ChatContent::Image(img), ChatContent::Audio(audio)])];
    let opts = ToDictOptions { exclude_image: false, exclude_audio: false, ..Default::default() };
    let d = dump(items, opts);
    assert_eq!(
        as_text(&d["items"][0]["content"]),
        r#"[{"id":"img_1","type":"image_content","image":{"ref":4},"inference_width":10,"inference_height":20,"inference_detail":"high","mime_type":"image/png"},{"type":"audio_content","frame":{"ref":5},"transcript":"said"}]"#
    );
}

// Catches: rendering a frame as `null` when no renderer was supplied.
#[test]
fn a_frame_without_a_renderer_is_an_error_not_a_null() {
    let img = ImageContent {
        id: "img_1".into(),
        image: ImageSource::Frame(FakeFrame::shared(4)),
        inference_width: None,
        inference_height: None,
        inference_detail: InferenceDetail::Auto,
        mime_type: None,
    };
    let items = vec![msg(Role::User, vec![ChatContent::Image(img)])];
    let bare = ToDictHost::default();
    let opts = ToDictOptions { exclude_image: false, ..Default::default() };
    assert_eq!(
        dump_with(items.clone(), opts, &bare),
        Err(ToDictError::MediaNotRenderable { kind: "ImageContent.image" })
    );
    // ...and the default flags never reach it.
    assert!(dump_with(items, ToDictOptions::default(), &bare).is_ok());
}
