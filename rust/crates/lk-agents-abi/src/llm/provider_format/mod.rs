//! Schema and message formatting (`llm/_strict.py`, `llm/_provider_format/*`). All pure: images
//! arrive through an [`ImageResolver`], and the reference's warnings come back as [`Dropped`].

pub mod anthropic;
pub mod aws;
pub mod google;
pub mod group;
pub mod image;
pub mod mistralai;
pub mod openai;

mod error;
pub mod host;
mod strict;
pub mod tools;
mod turns;

#[cfg(test)]
mod testkit;

pub use error::FormatError;
pub use group::{
    ChatItemGroup, DropReason, Dropped, GroupItem, Grouping, convert_mid_conversation_instructions,
    group_tool_calls,
};
pub use image::{BYTES_TAG, ImageResolver, PreResolved, SerializedImage};
pub use strict::{Object, StrictError, resolve_ref, to_strict_json_schema};

use serde_json::Value;

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
pub(crate) fn is_truthy(value: &Value) -> bool {
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
    schema.get(key).ok_or_else(|| FormatError::MalformedToolSchema { detail: format!("missing {key:?}") })
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
            let lk_agents_abi::ChatItem::FunctionCall(call) = testkit::call("c", "c0", "f", raw) else {
                unreachable!()
            };
            parse_arguments(&call)
        };
        assert_eq!(args(""), serde_json::json!({}));
        assert_eq!(args("{\"a\":1}"), serde_json::json!({ "a": 1 }));
        assert_eq!(args("{'a': 1,}"), serde_json::json!({ "a": 1 }));
        assert_eq!(args(&serde_json::to_string("{\"a\":1}").unwrap()), serde_json::json!({ "a": 1 }));
        for unrecoverable in ["5", "[1]", "   ", "not json"] {
            assert_eq!(args(unrecoverable), serde_json::json!({}), "{unrecoverable:?}");
        }
    }
}
