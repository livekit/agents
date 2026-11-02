//! Schema and message formatting (`llm/_strict.py`, `llm/_provider_format/*`). All pure: images
//! arrive through an [`ImageResolver`], and the reference's warnings come back as [`Dropped`].

use lk_agents_abi::ImageContent;
use serde_json::Value;

pub mod anthropic;
pub mod aws;
pub mod google;
pub mod group;
pub mod mistralai;
pub mod openai;

pub mod host;
mod strict;
pub mod tools;
mod turns;

#[cfg(test)]
mod testkit;

pub use group::{
    ChatItemGroup, DropReason, Dropped, GroupItem, Grouping, convert_mid_conversation_instructions,
    group_tool_calls,
};
pub use lk_agents_abi::llm::provider_format::FormatError;
pub use strict::{Object, StrictError, resolve_ref, to_strict_json_schema};

#[derive(Debug, Clone, PartialEq)]
pub struct Formatted<D> {
    /// `messages` / `contents` / `entries`, per provider.
    pub messages: Vec<Value>,
    /// `None` for OpenAI, the hoisted system prompt for the three that keep it out of band,
    /// `instructions` for Mistral.
    pub data: D,
    /// The binding logs each at warning.
    pub dropped: Vec<Dropped>,
}

/// The host builds the OpenAI-shaped schema; the reference's `strict` flag becomes the variant.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolSchema {
    /// `info.raw_schema`, verbatim.
    Raw { name: String, raw_schema: Value },
    /// `build_legacy_openai_schema(tool, internally_tagged=True)`: for Anthropic, Bedrock, Gemini
    /// and the Responses API.
    Legacy(Value),
    /// `build_strict_openai_schema(tool)`.
    Strict(Value),
    /// Only the Responses API accepts one.
    Provider(Value),
}

/// `null`, `{}`, `[]`, `0` and `""` are falsy, which decides whether keys appear in the body.
pub fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// A missing key is the reference's `KeyError`.
pub(crate) fn field<'v>(schema: &'v Value, key: &str) -> Result<&'v Value, FormatError> {
    schema
        .get(key)
        .ok_or_else(|| FormatError::MalformedToolSchema { detail: format!("missing {key:?}") })
}

/// `parse_tool_call_arguments`: a stored call's arguments as the JSON object Anthropic, Bedrock and
/// Gemini take, repaired when malformed. History the model never parsed (restored, or written by a
/// custom `llm_node`) must not fail every later turn, so an unrecoverable one becomes `{}`.
pub(crate) fn parse_arguments(call: &lk_agents_abi::FunctionCall) -> Value {
    if call.arguments.is_empty() {
        return Value::Object(serde_json::Map::new());
    }
    match crate::llm::tool_arguments::parse_logged(&call.arguments) {
        Ok(arguments) => Value::Object(arguments),
        Err(_) => {
            log::warn!(
                target: "livekit.agents",
                call_id = call.call_id.as_str(),
                tool_name = call.name.as_str();
                "could not parse stored tool call arguments as a JSON object, using empty arguments"
            );
            Value::Object(serde_json::Map::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: writing `extra.get(k)` as a presence test.
    #[test]
    fn truthiness_matches_pythons_for_every_json_shape() {
        for falsy in [
            serde_json::json!(null),
            serde_json::json!(false),
            serde_json::json!(0),
            serde_json::json!(0.0),
            serde_json::json!(-0.0),
            serde_json::json!(""),
            serde_json::json!([]),
            serde_json::json!({}),
        ] {
            assert!(!is_truthy(&falsy), "{falsy} should be falsy");
        }
        for truthy in [
            serde_json::json!(true),
            serde_json::json!(1),
            serde_json::json!(-1),
            serde_json::json!(0.5),
            serde_json::json!(" "),
            serde_json::json!("0"),
            serde_json::json!([null]),
            serde_json::json!({ "a": null }),
        ] {
            assert!(is_truthy(&truthy), "{truthy} should be truthy");
        }
    }

    // Catches: a malformed stored call raising on every later turn instead of being repaired, or
    // emptied when it is not an object.
    #[test]
    fn stored_arguments_are_repaired_or_emptied_never_raised() {
        let args = |raw: &str| {
            let lk_agents_abi::ChatItem::FunctionCall(call) = testkit::call("c", "c0", "f", raw)
            else {
                unreachable!()
            };
            parse_arguments(&call)
        };
        assert_eq!(args(""), serde_json::json!({}));
        assert_eq!(args("{\"a\":1}"), serde_json::json!({ "a": 1 }));
        assert_eq!(args("{'a': 1,}"), serde_json::json!({ "a": 1 }));
        assert_eq!(
            args(&serde_json::to_string("{\"a\":1}").unwrap()),
            serde_json::json!({ "a": 1 })
        );
        for unrecoverable in ["5", "[1]", "   ", "not json"] {
            assert_eq!(args(unrecoverable), serde_json::json!({}), "{unrecoverable:?}");
        }
    }
}

pub use lk_agents_abi::llm::provider_format::image::{BYTES_TAG, ImageResolver, SerializedImage};

/// Keyed by `ImageContent.id`: one frame under two `ImageContent`s may have different resize hints.
#[derive(Debug, Clone, Default)]
pub struct PreResolved(pub std::collections::BTreeMap<String, SerializedImage>);

impl PreResolved {
    pub fn from_pairs<K: Into<String>>(
        pairs: impl IntoIterator<Item = (K, SerializedImage)>,
    ) -> Self {
        PreResolved(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }
}

impl ImageResolver for PreResolved {
    fn resolve(&self, image: &ImageContent) -> Result<SerializedImage, String> {
        self.0
            .get(&image.id)
            .cloned()
            .ok_or_else(|| format!("no SerializedImage was resolved for {}", image.id))
    }
}

/// Gemini's and Bedrock's image leaves are `bytes`: they emit `{"$lk_bytes_b64": "<base64>"}` and
/// the binding materialises it.
pub fn bytes_value(data: &[u8]) -> Value {
    serde_json::json!({ BYTES_TAG: b64_encode(data) })
}

/// `base64.b64encode`.
pub fn b64_encode(data: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// A `None` mime_type formats as the reference's `data:None;...`, which only a driver bug reaches.
pub(crate) fn data_uri(mime_type: Option<&str>, data: &[u8]) -> String {
    format!("data:{};base64,{}", mime_type.unwrap_or("None"), b64_encode(data))
}
