//! `_provider_format/google.py`. Results accumulate under a synthetic `tool` role (so they never
//! share a turn with a user message), later rewritten to `user`. Gemini needs a final user turn.

use std::collections::BTreeMap;

use lk_agents_abi::ChatContext;
use serde_json::{Map, Value};

use crate::llm::provider_format::image::{ImageResolver, SerializedImage, bytes_value};
use crate::llm::provider_format::turns::{Dialect, Turns, merge_turns};
use crate::llm::provider_format::{FormatError, Formatted, ToolSchema, is_truthy, parse_arguments};

/// `google.py:15`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GoogleFormatData {
    pub system_messages: Vec<String>,
}

/// Keyed by `call_id`; the follow-up is rejected without them.
pub type ThoughtSignatures = BTreeMap<String, Vec<u8>>;

pub fn to_chat_ctx(
    chat_ctx: &ChatContext,
    images: &dyn ImageResolver,
    inject_dummy_user_message: bool,
    thought_signatures: Option<&ThoughtSignatures>,
) -> Result<Formatted<GoogleFormatData>, FormatError> {
    let dialect = Dialect {
        assistant: "model",
        // "tool output shouldn't be mixed with other messages"
        output: "tool",
        key: "parts",
        text: |text| serde_json::json!({ "text": text }),
        image: |img| Ok(to_image_part(img)),
        call: &|call| {
            let mut fc = Map::new();
            fc.insert(
                "function_call".into(),
                serde_json::json!({
                    "id": call.call_id,
                    "name": call.name,
                    "args": parse_arguments(call),
                }),
            );
            if let Some(sig) = thought_signatures
                .and_then(|sigs| sigs.get(&call.call_id))
                // An empty signature is falsy and is not echoed.
                .filter(|sig| !sig.is_empty())
            {
                fc.insert("thought_signature".into(), bytes_value(sig));
            }
            Ok(Value::Object(fc))
        },
        result: |output| {
            // Gemini marks a failed tool by the key inside `response`.
            let response = if output.is_error {
                serde_json::json!({ "error": output.output })
            } else {
                serde_json::json!({ "output": output.output })
            };
            serde_json::json!({
                "function_response": {
                    "id": output.call_id,
                    "name": output.name,
                    "response": response,
                }
            })
        },
    };
    let Turns { mut turns, system_messages, last_role, dropped } =
        merge_turns(chat_ctx, images, &dialect)?;

    for turn in &mut turns {
        if turn["role"] == "tool" {
            turn["role"] = "user".into();
        }
    }

    // Tests the last role seen, not the last turn emitted: a final item with no parts still counts.
    if inject_dummy_user_message && !matches!(last_role, Some("user") | Some("tool")) {
        turns.push(serde_json::json!({ "role": "user", "parts": [{ "text": "." }] }));
    }

    Ok(Formatted { messages: turns, data: GoogleFormatData { system_messages }, dropped })
}

/// `google.py:100`.
fn to_image_part(img: &SerializedImage) -> Value {
    if let Some(url) = &img.external_url {
        // The reference logs at debug and defaults to jpeg.
        let mime_type = img.mime_type.as_deref().filter(|m| !m.is_empty()).unwrap_or("image/jpeg");
        return serde_json::json!({ "file_data": { "file_uri": url, "mime_type": mime_type } });
    }
    // No assert here, unlike other dialects: neither arm set emits `data: null`.
    let data = match &img.data_bytes {
        Some(bytes) => bytes_value(bytes),
        None => Value::Null,
    };
    serde_json::json!({ "inline_data": { "data": data, "mime_type": img.mime_type } })
}

/// `google.py:121`. An object schema without properties means no arguments. Gemini Live rejects
/// `parameters_json_schema` (googleapis/python-genai#1147), so without `use_parameters_json_schema`
/// the schema is simplified to Gemini's own `parameters`.
pub fn to_fnc_ctx(
    tools: &[ToolSchema],
    // `google.py:118`, checked by the caller.
    tool_behavior: Option<&str>,
    use_parameters_json_schema: bool,
) -> Result<Vec<Value>, FormatError> {
    let missing =
        |key: &str| FormatError::MalformedToolSchema { detail: format!("missing {key:?}") };
    let mut out = Vec::new();
    for tool in tools {
        let (name, description, json_schema) = match tool {
            // A raw schema is author-written and goes through as it is.
            ToolSchema::Raw { name, raw_schema } => (
                Value::String(name.clone()),
                raw_schema.get("description").cloned().unwrap_or_else(|| "".into()),
                raw_schema.get("parameters").cloned().unwrap_or(Value::Null),
            ),
            ToolSchema::Legacy(built) => {
                let parameters = built.get("parameters").cloned().unwrap_or(Value::Null);
                let has_properties = parameters.get("properties").is_some_and(is_truthy);
                (
                    built.get("name").cloned().ok_or_else(|| missing("name"))?,
                    // Not coerced to `""` as Anthropic and Bedrock do: null reaches Gemini as null.
                    built.get("description").cloned().ok_or_else(|| missing("description"))?,
                    if has_properties { parameters } else { Value::Null },
                )
            }
            ToolSchema::Strict(_) | ToolSchema::Provider(_) => continue,
        };
        let mut schema = Map::new();
        schema.insert("name".into(), name);
        schema.insert("description".into(), description);
        if use_parameters_json_schema {
            schema.insert("parameters_json_schema".into(), none_if_falsy(json_schema));
        } else {
            let json_schema =
                if is_truthy(&json_schema) { json_schema } else { Value::Object(Map::new()) };
            schema.insert("parameters".into(), none_if_falsy(simplify_schema(json_schema)?));
        }
        if let Some(behavior) = tool_behavior {
            schema.insert("behavior".into(), behavior.into());
        }
        out.push(Value::Object(schema));
    }
    Ok(out)
}

fn none_if_falsy(value: Value) -> Value {
    if is_truthy(&value) { value } else { Value::Null }
}

/// Every key Gemini's `types.Schema` accepts (field name and camelCase alias), and the JSON Schema
/// keywords the simplifier still reads. Gemini refuses any other key.
const GEMINI_SCHEMA_KEYS: &[&str] = &[
    "additional_properties",
    "additionalProperties",
    "any_of",
    "anyOf",
    "default",
    "defs",
    "description",
    "enum",
    "example",
    "format",
    "items",
    "max_items",
    "maxItems",
    "max_length",
    "maxLength",
    "max_properties",
    "maxProperties",
    "maximum",
    "min_items",
    "minItems",
    "min_length",
    "minLength",
    "min_properties",
    "minProperties",
    "minimum",
    "nullable",
    "pattern",
    "properties",
    "property_ordering",
    "propertyOrdering",
    "ref",
    "required",
    "title",
    "type",
    "$ref",
    "prefixItems",
    "const",
];

const GEMINI_TYPES: &[(&str, &str)] = &[
    ("string", "STRING"),
    ("number", "NUMBER"),
    ("integer", "INTEGER"),
    ("boolean", "BOOLEAN"),
    ("array", "ARRAY"),
    ("object", "OBJECT"),
];

/// The google plugin's `_GeminiJsonSchema(schema).simplify()`: a JSON Schema in Gemini's dialect,
/// `$ref`s inlined, or null for an object without properties.
pub fn simplify_schema(mut schema: Value) -> Result<Value, FormatError> {
    let defs = schema.as_object_mut().and_then(|o| o.remove("$defs")).unwrap_or_default();
    simplify(&mut schema, &defs, &mut Vec::new())?;
    if schema.get("type").and_then(Value::as_str) == Some("OBJECT")
        && !schema.get("properties").is_some_and(is_truthy)
    {
        return Ok(Value::Null);
    }
    Ok(schema)
}

fn unsupported(detail: String) -> FormatError {
    FormatError::MalformedToolSchema { detail }
}

fn simplify(schema: &mut Value, defs: &Value, refs: &mut Vec<String>) -> Result<(), FormatError> {
    let Some(obj) = schema.as_object_mut() else {
        return Err(unsupported(format!("a JSON Schema is an object, got {schema}")));
    };
    obj.retain(|k, _| GEMINI_SCHEMA_KEYS.contains(&k.as_str()));
    for key in ["title", "default", "additionalProperties"] {
        obj.remove(key);
    }
    if let Some(constant) = obj.remove("const").filter(|c| !c.is_null()) {
        // Gemini has no const, but takes an enum of one value.
        obj.insert("enum".into(), Value::Array(vec![constant]));
    }

    if let Some(reference) = obj.remove("$ref").filter(is_truthy) {
        let reference = reference.as_str().unwrap_or_default();
        let key = reference.strip_prefix("#/$defs/").unwrap_or(reference).to_owned();
        if refs.contains(&key) {
            return Err(unsupported(
                "Recursive `$ref`s in JSON Schema are not supported by Gemini".into(),
            ));
        }
        let mut def =
            defs.get(&key).cloned().ok_or_else(|| unsupported(format!("unknown $ref {key:?}")))?;
        refs.push(key);
        let simplified = simplify(&mut def, defs, refs);
        refs.pop();
        simplified?;
        if let Value::Object(def) = def {
            obj.extend(def);
        }
        return Ok(());
    }

    if !obj.contains_key("type")
        && let Some(first) = obj.get("enum").and_then(|e| e.get(0))
    {
        // Python's `isinstance(value, int)` holds for a bool too.
        let inferred = match first {
            Value::Number(n) if n.is_f64() => "number",
            Value::Number(_) | Value::Bool(_) => "integer",
            Value::String(_) => "string",
            other => return Err(unsupported(format!("Unsupported type in Schema: {other}"))),
        };
        obj.insert("type".into(), inferred.into());
    }

    if let Some(json_type) = obj.get("type").filter(|t| t.as_str() != Some("null")) {
        let gemini = json_type.as_str().and_then(|t| {
            GEMINI_TYPES
                .iter()
                .find(|(json, gemini)| *json == t || *gemini == t)
                .map(|(_, gemini)| *gemini)
                .or(matches!(t, "TYPE_UNSPECIFIED" | "NULL").then_some(t))
        });
        let Some(gemini) = gemini else {
            return Err(unsupported(format!("Unsupported type in JSON Schema: {json_type}")));
        };
        obj.insert("type".into(), gemini.to_owned().into());
    }

    for (json, gemini) in [
        ("minLength", "min_length"),
        ("maxLength", "max_length"),
        ("minItems", "min_items"),
        ("maxItems", "max_items"),
        ("minProperties", "min_properties"),
        ("maxProperties", "max_properties"),
    ] {
        if let Some(value) = obj.remove(json) {
            obj.insert(gemini.into(), value);
        }
    }

    if let Some(Value::Array(mut any_of)) = obj.remove("anyOf").filter(is_truthy) {
        let total = any_of.len();
        let mut has_null = false;
        let mut non_null = None;
        let mut mapped = Vec::new();
        for item in &mut any_of {
            simplify(item, defs, refs)?;
            if *item == serde_json::json!({"type": "null"}) {
                has_null = true;
            } else {
                non_null = Some(item.clone());
                mapped.push(item.clone());
            }
        }
        match non_null.filter(is_truthy) {
            Some(Value::Object(non_null)) if has_null && total == 2 => {
                obj.extend(non_null);
                obj.insert("nullable".into(), true.into());
            }
            _ => {
                obj.insert("any_of".into(), Value::Array(mapped));
            }
        }
    }

    match obj.get("type").and_then(Value::as_str) {
        Some("OBJECT") => {
            if let Some(Value::Object(properties)) =
                obj.get_mut("properties").filter(|p| is_truthy(p))
            {
                for value in properties.values_mut() {
                    simplify(value, defs, refs)?;
                }
                let order: Vec<Value> = properties.keys().cloned().map(Value::String).collect();
                if !obj.contains_key("property_ordering") && !obj.contains_key("propertyOrdering") {
                    obj.insert("property_ordering".into(), Value::Array(order));
                }
            }
        }
        Some("ARRAY") => {
            if let Some(Value::Array(prefix)) = obj.get_mut("prefixItems") {
                for item in prefix {
                    simplify(item, defs, refs)?;
                }
            }
            if let Some(items) = obj.get_mut("items").filter(|i| is_truthy(i)) {
                simplify(items, defs, refs)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use lk_agents_abi::{ChatItem, InferenceDetail, Role};

    use super::*;
    use crate::llm::provider_format::testkit::{
        call, ctx, image, inline_images, msg, msg_with, no_images, out, url_images,
    };

    fn run(items: Vec<ChatItem>, images: &dyn ImageResolver) -> Formatted<GoogleFormatData> {
        to_chat_ctx(&ctx(items), images, true, None).unwrap()
    }

    fn text_of(f: &Formatted<GoogleFormatData>) -> String {
        serde_json::to_string(&f.messages).unwrap()
    }

    // Catches: leaving the synthetic `tool` role in the output, or never using it.
    #[test]
    fn tool_results_get_their_own_turn_and_are_relabelled_user() {
        let items = vec![
            call("c", "c0", "f", "{}"),
            out("o", "c0", "f", "ok"),
            msg("u", Role::User, &["thanks"]),
        ];
        assert_eq!(
            text_of(&run(items, &no_images())),
            r#"[{"role":"model","parts":[{"function_call":{"id":"c0","name":"f","args":{}}}]},{"role":"user","parts":[{"function_response":{"id":"c0","name":"f","response":{"output":"ok"}}}]},{"role":"user","parts":[{"text":"thanks"}]}]"#
        );
    }

    // Catches: prepending the dummy turn the way Anthropic and Bedrock do.
    #[test]
    fn the_dummy_turn_is_appended_not_prepended() {
        assert_eq!(
            text_of(&run(vec![msg("a", Role::Assistant, &["hi"])], &no_images())),
            r#"[{"role":"model","parts":[{"text":"hi"}]},{"role":"user","parts":[{"text":"."}]}]"#
        );
    }

    // Catches: deciding the dummy turn from the last emitted turn instead of the last role seen.
    #[test]
    fn the_dummy_turn_follows_the_last_role_seen_not_the_last_turn_emitted() {
        let items = vec![
            msg("u", Role::User, &["q"]),
            msg("a", Role::Assistant, &[""]), // contributes no parts, emits no turn
        ];
        assert_eq!(
            text_of(&run(items, &no_images())),
            r#"[{"role":"user","parts":[{"text":"q"}]},{"role":"user","parts":[{"text":"."}]}]"#
        );
        // A tool role counts as user for this test -- "allow tool role since we update it above".
        let items = vec![call("c", "c0", "f", "{}"), out("o", "c0", "f", "ok")];
        assert!(!text_of(&run(items, &no_images())).contains(r#"{"text":"."}"#));
    }

    // Catches: emitting `response: {"output": ...}` for a failed tool.
    #[test]
    fn a_failed_tool_result_moves_from_the_output_key_to_the_error_key() {
        let mut o = out("o", "c0", "f", "boom");
        if let ChatItem::FunctionCallOutput(x) = &mut o {
            x.is_error = true;
        }
        let items = vec![call("c", "c0", "f", "{}"), o];
        assert!(text_of(&run(items, &no_images())).contains(r#""response":{"error":"boom"}"#));
    }

    // Catches: attaching an empty thought signature, or one to the wrong call.
    #[test]
    fn a_thought_signature_is_attached_by_call_id_and_only_when_non_empty() {
        let items = vec![
            call("c", "c0", "f", "{}"),
            out("o", "c0", "f", "ok"),
            call("d", "c1", "g", "{}"),
            out("p", "c1", "g", "ok"),
        ];
        let sigs: ThoughtSignatures =
            [("c0".to_string(), vec![1u8, 2]), ("c1".to_string(), Vec::new())]
                .into_iter()
                .collect();
        let f = to_chat_ctx(&ctx(items), &no_images(), true, Some(&sigs)).unwrap();
        let text = serde_json::to_string(&f.messages).unwrap();
        assert!(text.contains(r#""thought_signature":{"$lk_bytes_b64":"AQI="}"#), "{text}");
        assert_eq!(
            text.matches("thought_signature").count(),
            1,
            "an empty signature is not echoed"
        );
    }

    // Catches: base64 in `inline_data.data`, or losing the `image/jpeg` fallback.
    #[test]
    fn inline_images_are_raw_bytes_and_a_url_image_falls_back_to_jpeg() {
        let m = msg_with("m", Role::User, vec![image("i")]);
        assert!(text_of(&run(vec![m.clone()], &inline_images(&["i"]))).contains(
            r#"{"inline_data":{"data":{"$lk_bytes_b64":"/9j/"},"mime_type":"image/jpeg"}}"#
        ));

        let no_mime = SerializedImage::external("https://x/y", None, InferenceDetail::Auto);
        assert_eq!(
            serde_json::to_string(&to_image_part(&no_mime)).unwrap(),
            r#"{"file_data":{"file_uri":"https://x/y","mime_type":"image/jpeg"}}"#
        );
        assert!(
            text_of(&run(vec![m], &url_images(&["i"]))).contains(r#""mime_type":"image/png""#),
            "a real mime type is not overwritten"
        );
    }

    // Catches: coercing a null description to `""`, or `{}` instead of `null`.
    #[test]
    fn a_falsy_parameters_schema_becomes_null_and_a_null_description_stays_null() {
        let legacy = ToolSchema::Legacy(
            serde_json::json!({ "name": "t", "description": null, "parameters": {} }),
        );
        assert_eq!(
            serde_json::to_string(&to_fnc_ctx(&[legacy], None, true).unwrap()).unwrap(),
            r#"[{"name":"t","description":null,"parameters_json_schema":null}]"#
        );
    }

    // Catches: sending `parameters_json_schema` to Gemini Live.
    #[test]
    fn use_parameters_json_schema_selects_the_key_for_raw_tools() {
        let raw = ToolSchema::Raw {
            name: "t".into(),
            raw_schema: serde_json::json!({ "description": "d", "parameters": { "type": "object" } }),
        };
        assert_eq!(
            serde_json::to_string(&to_fnc_ctx(std::slice::from_ref(&raw), None, true).unwrap())
                .unwrap(),
            r#"[{"name":"t","description":"d","parameters_json_schema":{"type":"object"}}]"#
        );
        assert_eq!(
            serde_json::to_string(
                &to_fnc_ctx(std::slice::from_ref(&raw), Some("NON_BLOCKING"), false).unwrap()
            )
            .unwrap(),
            r#"[{"name":"t","description":"d","parameters":null,"behavior":"NON_BLOCKING"}]"#
        );
    }

    // Catches: hoisting `developer`, or hoisting an empty system message.
    #[test]
    fn the_system_prompt_is_hoisted_only_for_a_non_empty_system_role_message() {
        let f = to_chat_ctx(&ctx(vec![msg("s", Role::System, &["p"])]), &no_images(), false, None)
            .unwrap();
        assert_eq!(f.data.system_messages, ["p"]);

        let f =
            to_chat_ctx(&ctx(vec![msg("d", Role::Developer, &["p"])]), &no_images(), false, None)
                .unwrap();
        assert!(f.data.system_messages.is_empty());
        assert!(serde_json::to_string(&f.messages).unwrap().contains(r#"{"text":"p"}"#));

        let f = to_chat_ctx(&ctx(vec![msg("s", Role::System, &[""])]), &no_images(), false, None)
            .unwrap();
        assert!(f.data.system_messages.is_empty());
        assert!(f.messages.is_empty());
    }

    // Catches: emitting an empty `{"text": ""}` part.
    #[test]
    fn empty_text_parts_are_dropped_rather_than_emitted_as_empty_parts() {
        assert_eq!(
            text_of(&run(vec![msg("m", Role::User, &["a", "", "b"])], &no_images())),
            r#"[{"role":"user","parts":[{"text":"a"},{"text":"b"}]}]"#
        );
    }

    // Catches: sending `parameters` where Gemini takes the schema as written, or a schema for a
    // tool without arguments.
    #[test]
    fn function_tools_send_parameters_json_schema_and_null_without_arguments() {
        let with = ToolSchema::Legacy(serde_json::json!({
            "name": "f", "description": "d",
            "parameters": {"type": "object", "properties": {"x": {"type": "string", "title": "X"}}},
        }));
        let without = ToolSchema::Legacy(serde_json::json!({
            "name": "g", "description": "d", "parameters": {"type": "object", "properties": {}},
        }));
        let tools = to_fnc_ctx(&[with, without], None, true).unwrap();
        assert_eq!(
            tools[0]["parameters_json_schema"],
            serde_json::json!({"type": "object", "properties": {"x": {"type": "string", "title": "X"}}})
        );
        assert!(tools[0].get("parameters").is_none());
        assert_eq!(tools[1]["parameters_json_schema"], Value::Null);
    }

    // Catches: Gemini Live getting a JSON Schema it rejects instead of the simplified one.
    #[test]
    fn live_tools_get_the_simplified_schema() {
        let raw = ToolSchema::Raw {
            name: "f".into(),
            raw_schema: serde_json::json!({"parameters": {
                "type": "object",
                "$defs": {"Unit": {"enum": ["c", "f"], "title": "Unit"}},
                "properties": {
                    "unit": {"$ref": "#/$defs/Unit"},
                    "n": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}], "default": null},
                    "tags": {"type": "array", "items": {"type": "string", "maxLength": 5}, "readOnly": true},
                    "kind": {"const": "a"},
                },
                "required": ["unit"],
                "additionalProperties": false,
            }}),
        };
        let tools = to_fnc_ctx(&[raw], Some("NON_BLOCKING"), false).unwrap();
        assert_eq!(
            tools[0],
            serde_json::json!({
                "name": "f", "description": "",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "unit": {"enum": ["c", "f"], "type": "STRING"},
                        "n": {"type": "INTEGER", "minimum": 1, "nullable": true},
                        "tags": {"type": "ARRAY", "items": {"type": "STRING", "max_length": 5}},
                        "kind": {"enum": ["a"], "type": "STRING"},
                    },
                    "required": ["unit"],
                    "property_ordering": ["unit", "n", "tags", "kind"],
                },
                "behavior": "NON_BLOCKING",
            })
        );
        let empty = ToolSchema::Raw { name: "g".into(), raw_schema: serde_json::json!({}) };
        assert_eq!(to_fnc_ctx(&[empty], None, false).unwrap()[0]["parameters"], Value::Null);
    }

    // Catches: a recursive model looping instead of failing.
    #[test]
    fn a_recursive_ref_is_refused() {
        let schema = serde_json::json!({
            "type": "object",
            "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}}},
            "properties": {"root": {"$ref": "#/$defs/Node"}},
        });
        assert!(simplify_schema(schema).is_err());
    }
}
