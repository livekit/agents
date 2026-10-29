//! `ChatContext.to_dict` (`chat_context.py:629`) records: `model_dump(mode="json", exclude_none=True)`,
//! generated per record (`#[binding(model_dump)]`), and the declared host function. The item
//! filters are `lk_agents_core::llm::chat_dict`.

use std::time::SystemTime;

use crate::MediaRef;
use serde_json::{Map, Value};

use crate::llm::chat_context::{
    ChatContent, ChatItem, ChatMessage, Extra, ImageSource, InferenceDetail, Role,
};

/// `chat_context.py:630`. The seventh flag, `strip_markup`, is [`ToDictHost::strip_expr_markup`]:
/// the flag and the stripper are one decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToDictOptions {
    pub exclude_image: bool,
    pub exclude_audio: bool,
    /// `exclude={"created_at"}`, not a filter.
    pub exclude_timestamp: bool,
    /// Calls and outputs, as one unit.
    pub exclude_function_call: bool,
    pub exclude_metrics: bool,
    pub exclude_config_update: bool,
}

impl Default for ToDictOptions {
    fn default() -> Self {
        ToDictOptions {
            exclude_image: true,
            exclude_audio: true,
            exclude_timestamp: true,
            exclude_function_call: false,
            exclude_metrics: false,
            exclude_config_update: false,
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct ToDictHost<'a> {
    /// `None` is fine while media is excluded; reaching media without it is
    /// [`ToDictError::MediaNotRenderable`], never `null`.
    pub media: Option<&'a dyn Fn(&MediaRef) -> Value>,
    /// `Some` *is* `strip_markup=True`.
    pub strip_expr_markup: Option<&'a dyn Fn(&str) -> String>,
}

/// A binding contract with no reference raise.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToDictError {
    /// A frame-backed image or an audio part had to be serialized without [`ToDictHost::media`].
    #[error("{kind} needs a media renderer: pass ToDictHost::media, or exclude it")]
    MediaNotRenderable { kind: &'static str },
}

/// An item's dump as its own record; the item filters are the core's.
impl ModelDump for ChatItem {
    fn model_dump(&self, cx: &DumpCx<'_>, _: &'static str) -> Dumped {
        match self {
            ChatItem::Message(m) => m.model_dump(cx, "ChatMessage"),
            ChatItem::FunctionCall(f) => f.model_dump(cx, "FunctionCall"),
            ChatItem::FunctionCallOutput(f) => f.model_dump(cx, "FunctionCallOutput"),
            ChatItem::AgentHandoff(h) => h.model_dump(cx, "AgentHandoff"),
            ChatItem::AgentConfigUpdate(c) => c.model_dump(cx, "AgentConfigUpdate"),
        }
    }
}

/// A value's `model_dump(mode="json")`; `None` is a key `exclude_none` drops.
/// `#[binding(model_dump)]` derives it for a record; `at` (`Record.field`) names a media value in
/// its error.
pub trait ModelDump {
    fn model_dump(&self, cx: &DumpCx<'_>, at: &'static str) -> Dumped;
}

pub type Dumped = Result<Option<Value>, ToDictError>;
pub use serde_json::{Map as DumpMap, Value as DumpValue};

/// One `to_dict` call's flags and host renderers.
pub struct DumpCx<'a> {
    pub opts: &'a ToDictOptions,
    pub host: &'a ToDictHost<'a>,
}

impl DumpCx<'_> {
    /// `exclude={"created_at", "metrics"}` names item keys; no nested record declares either.
    pub fn insert(&self, d: &mut Map<String, Value>, key: &str, v: Option<Value>) {
        let excluded = match key {
            "created_at" => self.opts.exclude_timestamp,
            "metrics" => self.opts.exclude_metrics,
            _ => false,
        };
        if let (false, Some(v)) = (excluded, v) {
            d.insert(key.to_owned(), v);
        }
    }

    fn media(&self, r: &MediaRef, kind: &'static str) -> Dumped {
        match self.host.media {
            Some(render) => Ok(Some(render(r))),
            None => Err(ToDictError::MediaNotRenderable { kind }),
        }
    }
}

macro_rules! dump_as_value {
    ($($t:ty),*) => {$(
        impl ModelDump for $t {
            fn model_dump(&self, _: &DumpCx<'_>, _: &'static str) -> Dumped {
                Ok(Some(Value::from(self.clone())))
            }
        }
    )*};
}
dump_as_value!(String, bool, f64, u32);

macro_rules! dump_as_str {
    ($($t:ty),*) => {$(
        impl ModelDump for $t {
            fn model_dump(&self, _: &DumpCx<'_>, _: &'static str) -> Dumped {
                Ok(Some(self.as_str().into()))
            }
        }
    )*};
}
dump_as_str!(Role, InferenceDetail);

impl<T: ModelDump> ModelDump for Option<T> {
    fn model_dump(&self, cx: &DumpCx<'_>, at: &'static str) -> Dumped {
        self.as_ref().map_or(Ok(None), |v| v.model_dump(cx, at))
    }
}

impl<T: ModelDump> ModelDump for Vec<T> {
    fn model_dump(&self, cx: &DumpCx<'_>, at: &'static str) -> Dumped {
        let items = self.iter().map(|v| Ok(v.model_dump(cx, at)?.unwrap_or(Value::Null)));
        Ok(Some(Value::Array(items.collect::<Result<_, ToDictError>>()?)))
    }
}

/// Verbatim, nulls included: `exclude_none` does not reach inside a dict.
impl ModelDump for Extra {
    fn model_dump(&self, _: &DumpCx<'_>, _: &'static str) -> Dumped {
        Ok(Some(Value::Object(self.clone())))
    }
}

impl ModelDump for SystemTime {
    fn model_dump(&self, _: &DumpCx<'_>, _: &'static str) -> Dumped {
        Ok(Some(crate::unix_secs(*self).into()))
    }
}

/// A frame is the host's to render; reaching one without a renderer is an error, never `null`.
impl ModelDump for MediaRef {
    fn model_dump(&self, cx: &DumpCx<'_>, at: &'static str) -> Dumped {
        cx.media(self, at)
    }
}

impl ModelDump for ImageSource {
    fn model_dump(&self, cx: &DumpCx<'_>, at: &'static str) -> Dumped {
        match self {
            ImageSource::Url(u) => Ok(Some(u.as_str().into())),
            ImageSource::Frame(r) => cx.media(r, at),
        }
    }
}

/// `ChatMessage.content`, filtered (`chat_context.py:653`). Markup is stripped from assistant
/// messages only (`:658`): a user turn may contain the literal text of a tag.
pub(crate) fn content_dump(m: &ChatMessage, cx: &DumpCx<'_>) -> Dumped {
    let mut content = Vec::with_capacity(m.content.len());
    for part in &m.content {
        content.push(match part {
            ChatContent::Image(_) if cx.opts.exclude_image => continue,
            ChatContent::Audio(_) if cx.opts.exclude_audio => continue,
            ChatContent::Text(t) => match cx.host.strip_expr_markup {
                Some(strip) if m.role == Role::Assistant => strip(t).into(),
                _ => Value::from(t.as_str()),
            },
            ChatContent::Image(img) => img.model_dump(cx, "ImageContent")?.unwrap_or_default(),
            ChatContent::Audio(a) => a.model_dump(cx, "AudioContent")?.unwrap_or_default(),
        });
    }
    Ok(Some(Value::Array(content)))
}

/// `ChatContext.to_dict` (`chat_context.py:629`) as the host calls it.
#[crate::host(functions)]
pub trait ChatContextDict {
    /// ``{"items": [...]}``, each item's JSON dump. Media content needs its exclusion; with
    /// ``strip_markup`` the ``<expr/>`` dialect is stripped from assistant messages.
    #[host(function)]
    #[allow(clippy::too_many_arguments)]
    fn chat_ctx_to_dict(
        items: Vec<crate::ChatItem>,
        #[host(kw, default = true)] exclude_image: bool,
        #[host(default = true)] exclude_audio: bool,
        #[host(default = true)] exclude_timestamp: bool,
        #[host(default = false)] exclude_function_call: bool,
        #[host(default = false)] exclude_metrics: bool,
        #[host(default = false)] exclude_config_update: bool,
        #[host(default = false)] strip_markup: bool,
    ) -> Result<Map<String, Value>, ToDictError>;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::RefRegistry;
    use serde_json::json;

    use super::*;
    use crate::llm::chat_context::{
        AgentConfigUpdate, AgentHandoffItem, AudioContent, FunctionCall, FunctionCallOutput,
        ImageContent, MetricsMetadata, MetricsReport,
    };
    use crate::testkit::{t, text};

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

    fn host() -> ToDictHost<'static> {
        ToDictHost { media: Some(&fake_media), strip_expr_markup: None }
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
            !as_text(&dump(vec![m.into()], ToDictOptions::default()))
                .contains("provider_request_ids")
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
            image: ImageSource::Frame(media_ref(4)),
            inference_width: Some(10),
            inference_height: Some(20),
            inference_detail: InferenceDetail::High,
            mime_type: Some("image/png".into()),
        };
        let audio = AudioContent { frames: media_ref(5), transcript: Some("said".into()) };
        let items = vec![msg(Role::User, vec![ChatContent::Image(img), ChatContent::Audio(audio)])];
        let opts =
            ToDictOptions { exclude_image: false, exclude_audio: false, ..Default::default() };
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
            image: ImageSource::Frame(media_ref(4)),
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
}
