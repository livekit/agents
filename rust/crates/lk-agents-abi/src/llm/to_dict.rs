//! `ChatContext.to_dict` (`chat_context.py:629`): `model_dump(mode="json", exclude_none=True)`, pinned
//! against pydantic 2.13 by a differential proptest. The records' dumps are generated
//! (`#[binding(model_dump)]`); the filters and the markup strip are here.

use std::time::SystemTime;

use crate::MediaRef;
use serde_json::{Map, Value};

use crate::llm::chat_context::ChatContext;
use crate::llm::chat_context::{
    ChatContent, ChatItem, ChatMessage, Extra, ImageSource, InferenceDetail, ItemKind, Role,
};

/// The `time.time()` float the reference stores. Pre-epoch stamps come back negative.
pub fn epoch_secs(at: SystemTime) -> f64 {
    match at.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => d.as_secs_f64(),
        Err(before) => -before.duration().as_secs_f64(),
    }
}

/// `None` for a value no wall clock returns (NaN, infinities, out of range); never panics.
pub fn wall_from_secs(secs: f64) -> Option<SystemTime> {
    let magnitude = std::time::Duration::try_from_secs_f64(secs.abs()).ok()?;
    if secs < 0.0 {
        SystemTime::UNIX_EPOCH.checked_sub(magnitude)
    } else {
        SystemTime::UNIX_EPOCH.checked_add(magnitude)
    }
}

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
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToDictError {
    /// A frame-backed image or an audio part had to be serialized without [`ToDictHost::media`].
    #[error("{kind} needs a media renderer: pass ToDictHost::media, or exclude it")]
    MediaNotRenderable { kind: &'static str },
}

impl ChatContext {
    /// `{"items": [...]}` and nothing else: `from_dict` and the session report read that envelope.
    pub fn to_dict(
        &self,
        opts: &ToDictOptions,
        host: &ToDictHost<'_>,
    ) -> Result<Value, ToDictError> {
        let mut items = Vec::with_capacity(self.items.len());
        for item in &self.items {
            if let Some(v) = item_to_dict(item, opts, host)? {
                items.push(v);
            }
        }
        let mut out = Map::new();
        out.insert("items".to_owned(), Value::Array(items));
        Ok(Value::Object(out))
    }
}

/// The item filters (`chat_context.py:642`), then the item's generated [`ModelDump`].
pub fn item_to_dict(
    item: &ChatItem,
    opts: &ToDictOptions,
    host: &ToDictHost<'_>,
) -> Result<Option<Value>, ToDictError> {
    if opts.exclude_function_call && item.kind().is_function_call_pair() {
        return Ok(None);
    }
    if opts.exclude_config_update && item.kind() == ItemKind::AgentConfigUpdate {
        return Ok(None);
    }
    let cx = DumpCx { opts, host };
    match item {
        ChatItem::Message(m) => m.model_dump(&cx, "ChatMessage"),
        ChatItem::FunctionCall(f) => f.model_dump(&cx, "FunctionCall"),
        ChatItem::FunctionCallOutput(f) => f.model_dump(&cx, "FunctionCallOutput"),
        ChatItem::AgentHandoff(h) => h.model_dump(&cx, "AgentHandoff"),
        ChatItem::AgentConfigUpdate(c) => c.model_dump(&cx, "AgentConfigUpdate"),
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
        Ok(Some(epoch_secs(*self).into()))
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

/// `ChatContext.to_dict` (`chat_context.py:629`) over the core's projection.
///
/// `serialize_media` renders a host media object (the RTC SDK owns that serializer); it is
/// required only when media exclusions are off. Passing `strip_expr_markup` is `strip_markup=True`.
#[cfg(feature = "python")]
#[pyo3::pyfunction]
#[pyo3(signature = (
    items,
    *,
    exclude_image = true,
    exclude_audio = true,
    exclude_timestamp = true,
    exclude_function_call = false,
    exclude_metrics = false,
    exclude_config_update = false,
    serialize_media = None,
    strip_expr_markup = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn chat_ctx_to_dict<'py>(
    py: pyo3::Python<'py>,
    items: Vec<crate::ChatItem>,
    exclude_image: bool,
    exclude_audio: bool,
    exclude_timestamp: bool,
    exclude_function_call: bool,
    exclude_metrics: bool,
    exclude_config_update: bool,
    serialize_media: Option<pyo3::Bound<'py, pyo3::PyAny>>,
    strip_expr_markup: Option<pyo3::Bound<'py, pyo3::PyAny>>,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    let ctx = ChatContext::new(items);
    let opts = ToDictOptions {
        exclude_image,
        exclude_audio,
        exclude_timestamp,
        exclude_function_call,
        exclude_metrics,
        exclude_config_update,
    };

    // The core has no error channel for host raises, so the first one is parked and re-raised
    // after the projection: a `Null` standing in for a frame would be silent data loss.
    let raised: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
    let park = |e: PyErr| {
        let mut slot = raised.borrow_mut();
        if slot.is_none() {
            *slot = Some(e);
        }
        Value::Null
    };

    let render = |r: &MediaRef| -> Value {
        let Some(f) = serialize_media.as_ref() else {
            return park(PyValueError::new_err("serialize_media is required for media content"));
        };
        let out = crate::python::resolve_key(py, r.id())
            .and_then(|obj| f.call1((obj,)))
            .and_then(|v| crate::python::value_from_py(&v, "serialize_media(...)"));
        match out {
            Ok(v) => v,
            Err(e) => park(e),
        }
    };
    let strip = |s: &str| -> String {
        let Some(f) = strip_expr_markup.as_ref() else { return s.to_owned() };
        match f.call1((s,)).and_then(|v| v.extract::<String>()) {
            Ok(v) => v,
            Err(e) => {
                park(e);
                s.to_owned()
            }
        }
    };

    let host = ToDictHost {
        media: Some(&render),
        strip_expr_markup: strip_expr_markup.as_ref().map(|_| &strip as &dyn Fn(&str) -> String),
    };
    let value = ctx.to_dict(&opts, &host).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if let Some(e) = raised.into_inner() {
        return Err(e);
    }
    crate::python::value_to_py(py, &value)
}

#[cfg(feature = "python")]
crate::register_host!(fn chat_ctx_to_dict);

#[cfg(test)]
mod wall_tests {
    use super::{epoch_secs, wall_from_secs};

    // Catches: converting `created_at` through anything coarser than f64 (the 1e-6 `_summarize`
    // nudge must survive).
    #[test]
    fn a_wall_clock_stamp_survives_the_round_trip_including_the_summarize_nudge() {
        let now = 1_767_225_600.123_456_7_f64; // a real `time.time()` magnitude
        assert_eq!(epoch_secs(wall_from_secs(now).unwrap()), now);

        let nudged = now + 1e-6;
        assert!(nudged > now, "the f64 itself must be distinguishable");
        assert!(
            wall_from_secs(nudged).unwrap() > wall_from_secs(now).unwrap(),
            "a 1us nudge collapsed on the way into SystemTime"
        );
        assert_eq!(epoch_secs(wall_from_secs(nudged).unwrap()), nudged);
    }

    // Below ~4.5e6 s an f64 is finer than `SystemTime`'s nanosecond, so such stamps quantize; no
    // real `time.time()` is there.
    // Catches: "fixing" that by rounding or a coarser representation.
    #[test]
    fn sub_nanosecond_stamps_quantize_and_real_wall_clocks_do_not() {
        // Below the boundary: an f64 with more precision than a nanosecond loses the excess.
        let tiny = 0.745_466_620_754_925_9_f64;
        assert_ne!(epoch_secs(wall_from_secs(tiny).unwrap()), tiny, "the boundary moved");
        assert_eq!(epoch_secs(wall_from_secs(tiny).unwrap()), 0.745_466_621);

        // Above it -- every value `time.time()` can return -- the round trip is exact.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..100_000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let now = 1.7e9 + (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0e8;
            assert_eq!(epoch_secs(wall_from_secs(now).unwrap()), now, "{now} did not survive");
        }
    }

    // Catches: `Duration::from_secs_f64` panicking on a host-supplied nan/inf/negative.
    #[test]
    fn a_nonsensical_created_at_is_none_not_a_panic() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e30] {
            assert!(wall_from_secs(bad).is_none(), "{bad} should not be accepted");
        }
        // ...but a pre-epoch stamp is representable, and round-trips.
        assert_eq!(epoch_secs(wall_from_secs(-1.5).unwrap()), -1.5);
    }
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

    fn dump(items: Vec<ChatItem>, opts: ToDictOptions) -> Value {
        ChatContext::new(items).to_dict(&opts, &host()).unwrap()
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
        let ctx = ChatContext::new(vec![msg(Role::User, vec![ChatContent::Image(img)])]);
        let bare = ToDictHost::default();
        let opts = ToDictOptions { exclude_image: false, ..Default::default() };
        assert_eq!(
            ctx.to_dict(&opts, &bare),
            Err(ToDictError::MediaNotRenderable { kind: "ImageContent.image" })
        );
        // ...and the default flags never reach it.
        assert!(ctx.to_dict(&ToDictOptions::default(), &bare).is_ok());
    }




}
