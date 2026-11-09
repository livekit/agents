use std::time::SystemTime;

use crate::MediaRef;
use crate::utils::video::RtcVideoFrame;
use serde_json::{Map, Value};

use crate::llm::chat_context::{
    ChatContent, ChatItem, ChatMessage, Extra, ImageSource, InferenceDetail, Role,
};

/// The seventh flag, `strip_markup`, is [`ToDictHost::strip_expr_markup`]:
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

/// Renders an image's host frame as its dump.
pub type FrameRenderer = dyn Fn(&dyn RtcVideoFrame) -> Value;

#[derive(Default, Clone, Copy)]
pub struct ToDictHost<'a> {
    /// `None` is fine while media is excluded; reaching media without it is
    /// [`ToDictError::MediaNotRenderable`], never `null`.
    pub media: Option<&'a dyn Fn(&MediaRef) -> Value>,
    /// The same for an image's host frame.
    pub frame: Option<&'a FrameRenderer>,
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

    fn frame(&self, frame: &dyn RtcVideoFrame, kind: &'static str) -> Dumped {
        match self.host.frame {
            Some(render) => Ok(Some(render(frame))),
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
            ImageSource::Frame(frame) => cx.frame(frame.as_ref(), at),
        }
    }
}

/// `ChatMessage.content`, filtered. Markup is stripped from assistant
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

/// `ChatContext.to_dict` as the host calls it.
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
#[path = "to_dict_tests.rs"]
mod tests;
