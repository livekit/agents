//! `_provider_format/anthropic.py`. System messages are hoisted; consecutive items of the same
//! effective role coalesce into one turn.

use lk_agents_abi::ChatContext;
use serde_json::{Map, Value};

use crate::llm::provider_format::image::{ImageResolver, SerializedImage};
use crate::llm::provider_format::turns::{Dialect, Turns, merge_turns};
use crate::llm::provider_format::{FormatError, Formatted, field, parse_arguments};

/// `anthropic.py:14`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnthropicFormatData {
    pub system_messages: Vec<String>,
}

/// `anthropic.py:19`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// Prepend an `"(empty)"` user turn when `messages` would not open on one.
    pub inject_dummy_user_message: bool,
    /// After a trailing assistant turn (#4973: Claude 4.6+ forbids prefilling). Off by default.
    pub inject_trailing_user_message: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { inject_dummy_user_message: true, inject_trailing_user_message: false }
    }
}

pub fn to_chat_ctx(
    chat_ctx: &ChatContext,
    images: &dyn ImageResolver,
    opts: Options,
) -> Result<Formatted<AnthropicFormatData>, FormatError> {
    let dialect = Dialect {
        assistant: "assistant",
        output: "user",
        key: "content",
        text: |text| serde_json::json!({ "text": text, "type": "text" }),
        image: |img| Ok(to_image_content(img)),
        call: &|call| {
            Ok(serde_json::json!({
                "id": call.call_id,
                "type": "tool_use",
                "name": call.name,
                "input": parse_arguments(&call.arguments)?,
            }))
        },
        result: |output| {
            // A JSON list result is a list of content blocks (#4882, computer use); anything
            // else stays the raw string.
            let result_content = match serde_json::from_str::<Value>(&output.output) {
                Ok(parsed @ Value::Array(_)) => parsed,
                _ => Value::String(output.output.clone()),
            };
            serde_json::json!({
                "tool_use_id": output.call_id,
                "type": "tool_result",
                "content": result_content,
                "is_error": output.is_error,
            })
        },
    };
    let Turns { turns: mut messages, system_messages, dropped, .. } =
        merge_turns(chat_ctx, images, &dialect)?;

    if opts.inject_dummy_user_message && first_role(&messages) != Some("user") {
        messages.insert(0, dummy_turn("(empty)"));
    }
    if opts.inject_trailing_user_message && last_role(&messages) == Some("assistant") {
        messages.push(dummy_turn("."));
    }

    Ok(Formatted { messages, data: AnthropicFormatData { system_messages }, dropped })
}

/// Reversed key order from other dialects, observable.
fn dummy_turn(text: &str) -> Value {
    serde_json::json!({ "role": "user", "content": [{ "text": text, "type": "text" }] })
}

fn first_role(messages: &[Value]) -> Option<&str> {
    messages.first()?.get("role")?.as_str()
}

fn last_role(messages: &[Value]) -> Option<&str> {
    messages.last()?.get("role")?.as_str()
}

/// `anthropic.py:107`.
fn to_image_content(img: &SerializedImage) -> Value {
    if let Some(url) = &img.external_url {
        return serde_json::json!({ "type": "image", "source": { "type": "url", "url": url } });
    }
    serde_json::json!({
        "type": "image",
        "source": {
            "type": "base64",
            "data": crate::llm::provider_format::image::b64_encode(img.data_bytes.as_deref().unwrap_or(&[])),
            "media_type": img.mime_type,
        },
    })
}

/// `anthropic.py:130`. Strict schemas are unpacked from `function` and keep `strict: true`.
pub fn to_fnc_ctx(tools: &[crate::llm::provider_format::ToolSchema]) -> Result<Vec<Value>, FormatError> {
    let mut schemas = Vec::new();
    for tool in tools {
        let entry = match tool {
            crate::llm::provider_format::ToolSchema::Strict(schema) => {
                let function = field(schema, "function")?;
                let mut out = Map::new();
                out.insert("name".into(), field(function, "name")?.clone());
                // `.get("description") or ""` -- a null *or an empty-ish* description becomes "".
                out.insert("description".into(), or_empty_string(function.get("description")));
                out.insert("input_schema".into(), field(function, "parameters")?.clone());
                out.insert("strict".into(), Value::Bool(true));
                Value::Object(out)
            }
            crate::llm::provider_format::ToolSchema::Legacy(schema) => {
                let mut out = Map::new();
                out.insert("name".into(), field(schema, "name")?.clone());
                // `fnc["description"] or ""` -- indexed, not `.get`, so a missing key raises.
                out.insert(
                    "description".into(),
                    or_empty_string(Some(field(schema, "description")?)),
                );
                out.insert("input_schema".into(), field(schema, "parameters")?.clone());
                Value::Object(out)
            }
            crate::llm::provider_format::ToolSchema::Raw { name, raw_schema } => serde_json::json!({
                "name": name,
                "description": raw_schema.get("description").cloned().unwrap_or_else(|| "".into()),
                "input_schema": raw_schema
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Map::new())),
            }),
            // The reference iterates `tool_ctx.function_tools`, which cannot contain one.
            crate::llm::provider_format::ToolSchema::Provider(_) => continue,
        };
        schemas.push(entry);
    }
    Ok(schemas)
}

fn or_empty_string(value: Option<&Value>) -> Value {
    match value {
        Some(v) if crate::llm::provider_format::is_truthy(v) => v.clone(),
        _ => Value::String(String::new()),
    }
}

#[cfg(test)]
mod tests {
    use lk_agents_abi::{ChatItem, Role};

    use super::*;
    use crate::llm::provider_format::ToolSchema;
    use crate::llm::provider_format::testkit::{
        call, ctx, image, inline_images, msg, msg_with, no_images, out, text_content, url_images,
    };

    fn run(items: Vec<ChatItem>, images: &dyn ImageResolver) -> Formatted<AnthropicFormatData> {
        to_chat_ctx(&ctx(items), images, Options::default()).unwrap()
    }

    fn text_of(f: &Formatted<AnthropicFormatData>) -> String {
        serde_json::to_string(&f.messages).unwrap()
    }

    // Catches: emitting one turn per item instead of coalescing.
    #[test]
    fn consecutive_items_of_one_effective_role_coalesce_into_a_single_turn() {
        let items = vec![
            msg("u", Role::User, &["question"]),
            msg("t", Role::Assistant, &["thinking"]),
            call("t/0", "c0", "f", "{}"),
            out("o", "c0", "f", "ok"),
            msg("u2", Role::User, &["more"]),
        ];
        assert_eq!(
            text_of(&run(items, &no_images())),
            r#"[{"role":"user","content":[{"text":"question","type":"text"}]},{"role":"assistant","content":[{"text":"thinking","type":"text"},{"id":"c0","type":"tool_use","name":"f","input":{}}]},{"role":"user","content":[{"tool_use_id":"c0","type":"tool_result","content":"ok","is_error":false},{"text":"more","type":"text"}]}]"#
        );
    }

    // Catches: hoisting `developer` into `system_messages` alongside `system`.
    #[test]
    fn only_the_system_role_is_hoisted_out_of_band_not_developer() {
        let f = run(vec![msg("d", Role::Developer, &["dev preamble"])], &no_images());
        assert!(f.data.system_messages.is_empty());
        assert_eq!(
            text_of(&f),
            r#"[{"role":"user","content":[{"text":"dev preamble","type":"text"}]}]"#
        );

        let f = run(vec![msg("s", Role::System, &["sys preamble"])], &no_images());
        assert_eq!(f.data.system_messages, ["sys preamble"]);
    }

    // Catches: reading OpenAI's text rule into this dialect.
    #[test]
    fn empty_text_parts_are_dropped_rather_than_concatenated() {
        assert_eq!(
            text_of(&run(vec![msg("m", Role::User, &["a", "", "b"])], &no_images())),
            r#"[{"role":"user","content":[{"text":"a","type":"text"},{"text":"b","type":"text"}]}]"#
        );
    }

    // Catches: stringifying a list tool result, or parsing an object one.
    #[test]
    fn only_a_json_list_tool_result_is_unwrapped_into_content_blocks() {
        let unwrap = |output: &str| {
            let items = vec![call("c", "c0", "f", "{}"), out("o", "c0", "f", output)];
            text_of(&run(items, &no_images()))
        };
        assert!(unwrap(r#"[{"type":"image"}]"#).contains(r#""content":[{"type":"image"}]"#));
        assert!(unwrap(r#"{"k":1}"#).contains(r#""content":"{\"k\":1}""#));
        assert!(unwrap("42").contains(r#""content":"42""#));
        assert!(unwrap("not json").contains(r#""content":"not json""#));
    }

    // Catches: dropping `is_error`.
    #[test]
    fn a_failed_tool_result_carries_is_error() {
        let mut o = out("o", "c0", "f", "boom");
        if let ChatItem::FunctionCallOutput(x) = &mut o {
            x.is_error = true;
        }
        let items = vec![call("c", "c0", "f", "{}"), o];
        assert!(text_of(&run(items, &no_images())).contains(r#""is_error":true"#));
    }

    // Catches: appending the "(empty)" turn, or prepending it when the conversation opens on user.
    #[test]
    fn the_dummy_user_turn_is_prepended_and_only_when_needed() {
        let f = run(vec![msg("a", Role::Assistant, &["hi"])], &no_images());
        assert_eq!(
            text_of(&f),
            r#"[{"role":"user","content":[{"text":"(empty)","type":"text"}]},{"role":"assistant","content":[{"text":"hi","type":"text"}]}]"#
        );
        assert_eq!(
            text_of(&run(vec![msg("u", Role::User, &["hi"])], &no_images())),
            r#"[{"role":"user","content":[{"text":"hi","type":"text"}]}]"#
        );
        // An empty conversation still gets one -- `not messages` is the first disjunct.
        assert_eq!(run(vec![], &no_images()).messages.len(), 1);
        // ...unless the caller turned it off.
        let opts = Options { inject_dummy_user_message: false, ..Options::default() };
        assert!(to_chat_ctx(&ctx(vec![]), &no_images(), opts).unwrap().messages.is_empty());
    }

    // Catches: enabling the trailing turn unconditionally, or appending it after a user turn.
    #[test]
    fn the_trailing_user_turn_is_off_by_default_and_only_follows_an_assistant_turn() {
        let items = vec![msg("u", Role::User, &["q"]), msg("a", Role::Assistant, &["partial"])];
        assert!(
            !text_of(&run(items.clone(), &no_images())).contains(r#"{"text":".","type":"text"}"#)
        );

        let opts = Options { inject_dummy_user_message: true, inject_trailing_user_message: true };
        let f = to_chat_ctx(&ctx(items), &no_images(), opts).unwrap();
        assert!(
            text_of(&f).ends_with(r#"{"role":"user","content":[{"text":".","type":"text"}]}]"#)
        );

        let f = to_chat_ctx(&ctx(vec![msg("u", Role::User, &["q"])]), &no_images(), opts).unwrap();
        assert_eq!(f.messages.len(), 1, "a trailing user turn needs no help");
    }

    // Catches: emitting OpenAI's `data:` URI shape.
    #[test]
    fn images_use_anthropics_source_object_not_a_data_uri() {
        let m = msg_with("m", Role::User, vec![image("i")]);
        assert!(text_of(&run(vec![m.clone()], &inline_images(&["i"]))).contains(
            r#"{"type":"image","source":{"type":"base64","data":"/9j/","media_type":"image/jpeg"}}"#
        ));
        assert!(text_of(&run(vec![m], &url_images(&["i"]))).contains(
            r#"{"type":"image","source":{"type":"url","url":"https://example.test/i.png"}}"#
        ));
    }

    // Catches: emitting an empty `content: []` turn for an item with no blocks.
    #[test]
    fn a_turn_that_contributes_no_blocks_is_not_emitted() {
        let items = vec![
            msg("u", Role::User, &["q"]),
            msg_with("a", Role::Assistant, vec![text_content("")]),
            msg("u2", Role::User, &["more"]),
        ];
        assert_eq!(
            text_of(&run(items, &no_images())),
            r#"[{"role":"user","content":[{"text":"q","type":"text"}]},{"role":"user","content":[{"text":"more","type":"text"}]}]"#
        );
    }

    // Catches: unpacking a strict schema as if flat, or forgetting `strict: true`.
    #[test]
    fn strict_and_legacy_tool_schemas_unpack_from_different_levels() {
        let strict = ToolSchema::Strict(serde_json::json!({
            "type": "function",
            "function": { "name": "t", "description": null, "parameters": { "type": "object" } },
        }));
        assert_eq!(
            serde_json::to_string(&to_fnc_ctx(&[strict]).unwrap()).unwrap(),
            r#"[{"name":"t","description":"","input_schema":{"type":"object"},"strict":true}]"#
        );

        let legacy = ToolSchema::Legacy(serde_json::json!({
            "name": "t", "description": "does a thing", "parameters": { "type": "object" },
        }));
        assert_eq!(
            serde_json::to_string(&to_fnc_ctx(&[legacy]).unwrap()).unwrap(),
            r#"[{"name":"t","description":"does a thing","input_schema":{"type":"object"}}]"#
        );
    }
}
