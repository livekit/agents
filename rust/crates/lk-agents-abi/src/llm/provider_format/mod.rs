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

/// `json.loads(msg.arguments or "{}")`: only `""` becomes `{}`.
pub(crate) fn parse_arguments(arguments: &str) -> Result<Value, FormatError> {
    if arguments.is_empty() {
        return Ok(Value::Object(serde_json::Map::new()));
    }
    serde_json::from_str(arguments).map_err(|e| FormatError::MalformedToolSchema {
        detail: format!("tool call arguments are not JSON: {e}"),
    })
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

    // Catches: treating any falsy-looking `arguments` as `{}` instead of only `""`.
    #[test]
    fn only_empty_arguments_default_to_an_empty_object() {
        assert_eq!(parse_arguments("").unwrap(), serde_json::json!({}));
        assert_eq!(parse_arguments("{\"a\":1}").unwrap(), serde_json::json!({ "a": 1 }));
        // Not an object, and the reference passes it straight through as `input`/`args`.
        assert_eq!(parse_arguments("5").unwrap(), serde_json::json!(5));
        assert!(parse_arguments("   ").is_err());
        assert!(parse_arguments("{").is_err());
    }
}
