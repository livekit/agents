//! Gemini contents. Function results occupy their own user turn.

use super::turns::{Dialect, ResultsTurn, Speaker, Turns, merge_turns};
use super::wire::{FunctionDefinition, HostBytes};
use super::{
    FormatError, Formatted, ImageResolver, SerializedImage, ToolSchema, is_truthy, parse_arguments,
};
use lk_agents_abi::llm::provider_format::vocabulary::{
    GeminiRole as ContentRole, GeminiToolBehavior as Behavior,
};
use lk_agents_abi::{ChatContext, FunctionCall, FunctionCallOutput};
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct GoogleFormatData {
    pub system_messages: Vec<String>,
}

pub type ThoughtSignatures = BTreeMap<String, Vec<u8>>;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Content {
    pub role: ContentRole,
    pub parts: Vec<Part>,
}

// The Python SDK accepts its snake_case names and their REST camelCase aliases.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Part {
    #[serde(flatten)]
    pub data: PartData,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<HostBytes>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PartData {
    Text(String),
    InlineData(Blob),
    FileData(FileData),
    FunctionCall(FunctionCallPart),
    FunctionResponse(FunctionResponse),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Blob {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<HostBytes>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FileData {
    pub file_uri: String,
    pub mime_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FunctionCallPart {
    pub id: String,
    pub name: String,
    pub args: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FunctionResponse {
    pub id: String,
    pub name: String,
    pub response: FunctionResult,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FunctionResult {
    Output(String),
    Error(String),
}

struct Gemini<'a> {
    thought_signatures: Option<&'a ThoughtSignatures>,
}

impl Dialect for Gemini<'_> {
    type Block = Part;
    const RESULTS: ResultsTurn = ResultsTurn::Separate;

    fn text(&self, text: &str) -> Part {
        Part { data: PartData::Text(text.into()), thought_signature: None }
    }
    fn image(&self, image: &SerializedImage) -> Result<Part, FormatError> {
        Ok(to_image_part(image))
    }
    fn call(&self, call: &FunctionCall) -> Result<Part, FormatError> {
        Ok(Part {
            data: PartData::FunctionCall(FunctionCallPart {
                id: call.call_id.clone(),
                name: call.name.clone(),
                args: parse_arguments(call),
            }),
            thought_signature: self
                .thought_signatures
                .and_then(|signatures| signatures.get(&call.call_id))
                .filter(|signature| !signature.is_empty())
                .cloned()
                .map(HostBytes),
        })
    }
    fn result(&self, output: &FunctionCallOutput) -> Part {
        Part {
            data: PartData::FunctionResponse(FunctionResponse {
                id: output.call_id.clone(),
                name: output.name.clone(),
                response: if output.is_error {
                    FunctionResult::Error(output.output.clone())
                } else {
                    FunctionResult::Output(output.output.clone())
                },
            }),
            thought_signature: None,
        }
    }
}

pub fn to_chat_ctx(
    chat_ctx: &ChatContext,
    images: &dyn ImageResolver,
    inject_dummy_user_message: bool,
    thought_signatures: Option<&ThoughtSignatures>,
) -> Result<Formatted<Content, GoogleFormatData>, FormatError> {
    let dialect = Gemini { thought_signatures };
    let Turns { turns, system_messages, last_speaker, dropped } =
        merge_turns(chat_ctx, images, &dialect)?;
    let mut messages: Vec<_> = turns
        .into_iter()
        .map(|turn| Content {
            role: if turn.speaker == Speaker::Assistant {
                ContentRole::Model
            } else {
                ContentRole::User
            },
            parts: turn.blocks,
        })
        .collect();
    if inject_dummy_user_message
        && !matches!(last_speaker, Some(Speaker::User | Speaker::ToolResults))
    {
        messages.push(Content { role: ContentRole::User, parts: vec![dialect.text(".")] });
    }
    Ok(Formatted { messages, data: GoogleFormatData { system_messages }, dropped })
}

fn to_image_part(image: &SerializedImage) -> Part {
    let data = if let Some(url) = &image.external_url {
        PartData::FileData(FileData {
            file_uri: url.clone(),
            mime_type: image
                .mime_type
                .clone()
                .filter(|mime| !mime.is_empty())
                .unwrap_or_else(|| "image/jpeg".into()),
        })
    } else {
        PartData::InlineData(Blob {
            data: image.data_bytes.clone().map(HostBytes),
            mime_type: image.mime_type.clone(),
        })
    };
    Part { data, thought_signature: None }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters_json_schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub behavior: Option<Behavior>,
}

pub fn to_fnc_ctx(
    tools: &[ToolSchema],
    tool_behavior: Option<Behavior>,
    use_parameters_json_schema: bool,
) -> Result<Vec<FunctionDeclaration>, FormatError> {
    let behavior = tool_behavior.filter(|behavior| *behavior != Behavior::Unspecified);
    let mut declarations = Vec::new();
    for tool in tools {
        if matches!(tool, ToolSchema::Strict(_) | ToolSchema::Provider(_)) {
            continue;
        }
        let mut definition = FunctionDefinition::from_tool(tool)?;
        if matches!(tool, ToolSchema::Legacy(_))
            && !definition
                .parameters
                .as_ref()
                .and_then(|schema| schema.get("properties"))
                .is_some_and(is_truthy)
        {
            definition.parameters = None;
        }
        let schema = definition.parameters.filter(is_truthy);
        let (parameters, parameters_json_schema) = if use_parameters_json_schema {
            (None, schema)
        } else {
            let simplified = simplify_schema(schema.unwrap_or_else(|| Value::Object(Map::new())))?;
            (is_truthy(&simplified).then_some(simplified), None)
        };
        declarations.push(FunctionDeclaration {
            name: definition.name,
            description: if matches!(tool, ToolSchema::Raw { .. }) {
                Some(definition.description.unwrap_or_default())
            } else {
                definition.description
            },
            parameters,
            parameters_json_schema,
            behavior,
        });
    }
    Ok(declarations)
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
        call, ctx, image, inline_images, msg, msg_with, no_images, out, text_of, url_images,
    };

    fn run(
        items: Vec<ChatItem>,
        images: &dyn ImageResolver,
    ) -> Formatted<Content, GoogleFormatData> {
        to_chat_ctx(&ctx(items), images, true, None).unwrap()
    }

    // Catches: merging tool responses with user content or emitting a non-Gemini role.
    #[test]
    fn tool_results_get_their_own_user_turn() {
        let items = vec![
            call("c", "c0", "f", "{}"),
            out("o", "c0", "f", "ok"),
            msg("u", Role::User, &["thanks"]),
        ];
        assert_eq!(
            text_of(&run(items, &no_images()).messages),
            r#"[{"role":"model","parts":[{"function_call":{"id":"c0","name":"f","args":{}}}]},{"role":"user","parts":[{"function_response":{"id":"c0","name":"f","response":{"output":"ok"}}}]},{"role":"user","parts":[{"text":"thanks"}]}]"#
        );
    }

    // Catches: prepending the dummy turn the way Anthropic and Bedrock do.
    #[test]
    fn the_dummy_turn_is_appended_not_prepended() {
        assert_eq!(
            text_of(&run(vec![msg("a", Role::Assistant, &["hi"])], &no_images()).messages),
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
            text_of(&run(items, &no_images()).messages),
            r#"[{"role":"user","parts":[{"text":"q"}]},{"role":"user","parts":[{"text":"."}]}]"#
        );
        // A tool role counts as user for this test -- "allow tool role since we update it above".
        let items = vec![call("c", "c0", "f", "{}"), out("o", "c0", "f", "ok")];
        assert!(!text_of(&run(items, &no_images()).messages).contains(r#"{"text":"."}"#));
    }

    // Catches: emitting `response: {"output": ...}` for a failed tool.
    #[test]
    fn a_failed_tool_result_moves_from_the_output_key_to_the_error_key() {
        let mut o = out("o", "c0", "f", "boom");
        if let ChatItem::FunctionCallOutput(x) = &mut o {
            x.is_error = true;
        }
        let items = vec![call("c", "c0", "f", "{}"), o];
        assert!(
            text_of(&run(items, &no_images()).messages).contains(r#""response":{"error":"boom"}"#)
        );
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
        assert!(text_of(&run(vec![m.clone()], &inline_images(&["i"])).messages).contains(
            r#"{"inline_data":{"data":{"$lk_bytes_b64":"/9j/"},"mime_type":"image/jpeg"}}"#
        ));

        let no_mime = SerializedImage::external("https://x/y", None, InferenceDetail::Auto);
        assert_eq!(
            serde_json::to_string(&to_image_part(&no_mime)).unwrap(),
            r#"{"file_data":{"file_uri":"https://x/y","mime_type":"image/jpeg"}}"#
        );
        assert!(
            text_of(&run(vec![m], &url_images(&["i"])).messages)
                .contains(r#""mime_type":"image/png""#),
            "a real mime type is not overwritten"
        );
    }

    // Catches: sending absent optional fields as null.
    #[test]
    fn absent_parameters_and_description_are_omitted() {
        let legacy = ToolSchema::Legacy(
            serde_json::json!({ "name": "t", "description": null, "parameters": {} }),
        );
        assert_eq!(
            serde_json::to_string(&to_fnc_ctx(&[legacy], None, true).unwrap()).unwrap(),
            r#"[{"name":"t"}]"#
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
                &to_fnc_ctx(std::slice::from_ref(&raw), Some(Behavior::NonBlocking), false)
                    .unwrap()
            )
            .unwrap(),
            r#"[{"name":"t","description":"d","behavior":"NON_BLOCKING"}]"#
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
            text_of(&run(vec![msg("m", Role::User, &["a", "", "b"])], &no_images()).messages),
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
        let tools = serde_json::to_value(tools).unwrap();
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
        let tools = to_fnc_ctx(&[raw], Some(Behavior::NonBlocking), false).unwrap();
        let tools = serde_json::to_value(tools).unwrap();
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
        assert!(to_fnc_ctx(&[empty], None, false).unwrap()[0].parameters.is_none());
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
