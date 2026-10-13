//! `llm/utils.py`'s OpenAI wrappers, minus the pydantic prologue. Key order and the
//! `description or ""` fallback are behaviour.

use crate::llm::provider_format::to_strict_json_schema;
use serde_json::{Value, json};

use crate::llm::schema::defaults::SchemaError;

/// `build_legacy_openai_schema` (`utils.py:203`) — non-strict tool description.
///
/// `internally_tagged` gives the flat form the Responses API and Bedrock want; the default is
/// `{"type": "function", "function": {...}}`. A `None` or empty description emits `""`.
#[crate::host(export(pid = false, kw_only = "internally_tagged", defaults(internally_tagged = false)))]
pub fn build_legacy_openai_schema(
    name: &str,
    description: Option<&str>,
    parameters: Value,
    internally_tagged: bool,
) -> Value {
    let description = description.unwrap_or("");
    if internally_tagged {
        json!({
            "name": name,
            "description": description,
            "parameters": parameters,
            "type": "function",
        })
    } else {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        })
    }
}

/// `build_strict_openai_schema` (`utils.py:229`) — strict-mode tool description. Defaulted fields
/// become nullable so the model can decline them; [`inject_schema_defaults`](crate::llm::schema::inject_schema_defaults)
/// swaps the null back.
#[crate::host(export(pid = false))]
pub fn build_strict_openai_schema(
    name: &str,
    description: Option<&str>,
    parameters: &Value,
) -> Result<Value, SchemaError> {
    let parameters = to_strict_json_schema(parameters, true)?;
    Ok(json!({
        "type": "function",
        "function": {
            "name": name,
            "strict": true,
            "description": description.unwrap_or(""),
            "parameters": parameters,
        },
    }))
}

/// `to_openai_response_format` (`utils.py:288`) — structured-output response format. No null
/// sentinel: nothing resolves it on the decode side, so defaulted fields stay required.
#[crate::host(export(pid = false))]
pub fn to_openai_response_format(name: &str, schema: &Value) -> Result<Value, SchemaError> {
    let schema = to_strict_json_schema(schema, false)?;
    Ok(json!({
        "type": "json_schema",
        "json_schema": {
            "schema": schema,
            "name": name,
            "strict": true,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape pydantic emits for `def f(a: str, b: int = 3)`, trimmed to what matters here.
    fn parameters() -> Value {
        json!({
            "type": "object",
            "title": "FArgs",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer", "default": 3},
            },
            "required": ["a"],
        })
    }

    // Catches: an envelope in any other shape or key order than the reference's.
    #[test]
    fn each_envelope_serializes_in_the_references_shape_and_key_order() {
        let params = json!({"type": "object"});
        for (out, want) in [
            (
                build_legacy_openai_schema("f", Some("does f"), params.clone(), false),
                r#"{"type":"function","function":{"name":"f","description":"does f","parameters":{"type":"object"}}}"#,
            ),
            (
                build_legacy_openai_schema("f", Some("does f"), params.clone(), true),
                r#"{"name":"f","description":"does f","parameters":{"type":"object"},"type":"function"}"#,
            ),
            (
                build_strict_openai_schema("f", Some("does f"), &params).unwrap(),
                r#"{"type":"function","function":{"name":"f","strict":true,"description":"does f","parameters":{"type":"object","additionalProperties":false}}}"#,
            ),
            (
                to_openai_response_format("F", &params).unwrap(),
                r#"{"type":"json_schema","json_schema":{"schema":{"type":"object","additionalProperties":false},"name":"F","strict":true}}"#,
            ),
        ] {
            assert_eq!(serde_json::to_string(&out).unwrap(), want);
        }
    }

    // Catches: emitting `null` for a missing description.
    #[test]
    fn a_missing_description_becomes_an_empty_string_not_null() {
        for description in [None, Some("")] {
            let legacy = build_legacy_openai_schema("f", description, json!({}), false);
            assert_eq!(legacy["function"]["description"], json!(""));
            let strict = build_strict_openai_schema("f", description, &json!({})).unwrap();
            assert_eq!(strict["function"]["description"], json!(""));
        }
    }

    // Catches: passing the same `null_sentinel_for_defaults` to both strict builders.
    #[test]
    fn only_the_tool_builder_nulls_a_defaulted_field() {
        let tool = build_strict_openai_schema("f", None, &parameters()).unwrap();
        assert_eq!(
            tool["function"]["parameters"]["properties"]["b"]["type"],
            json!(["integer", "null"]),
            "a tool's defaulted field must accept the null sentinel"
        );

        let response = to_openai_response_format("F", &parameters()).unwrap();
        assert_eq!(
            response["json_schema"]["schema"]["properties"]["b"]["type"],
            json!("integer"),
            "a response format's defaulted field must stay non-nullable"
        );

        for schema in [&tool["function"]["parameters"], &response["json_schema"]["schema"]] {
            assert_eq!(schema["required"], json!(["a", "b"]), "strict mode requires every field");
            assert!(schema["properties"]["b"].get("default").is_none(), "`default` is never legal");
        }
    }

    // Catches: swallowing a malformed schema.
    #[test]
    fn a_schema_that_is_not_an_object_propagates_out_of_both_strict_builders() {
        assert!(build_strict_openai_schema("f", None, &json!("nope")).is_err());
        assert!(to_openai_response_format("F", &json!(["nope"])).is_err());
    }
}
