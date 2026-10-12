//! `_provider_format/openai.py`. `extra` is JSON because [`EXTRA_CONTENT_KEYS`] are inlined into
//! the body and `extra["openai"]["phase"]` drives control flow.

use lk_agents_abi::{ChatContent, ChatContext, ChatMessage, Extra, FunctionCallOutput, Role};
use serde_json::{Map, Value};

use crate::llm::provider_format::group::group_tool_calls;
use crate::llm::provider_format::image::{ImageResolver, SerializedImage, data_uri};
use crate::llm::provider_format::{FormatError, Formatted, is_truthy};

/// `openai.py:13`, in the emitted key order.
pub const EXTRA_CONTENT_KEYS: [&str; 3] = ["google", "livekit", "xai"];

/// `openai.py:16`. Truthiness: `{}`, `None` and `0` are dropped, and an empty result emits no key.
fn filter_extra(extra: &Extra) -> Map<String, Value> {
    let mut out = Map::new();
    for key in EXTRA_CONTENT_KEYS {
        if let Some(value) = extra.get(key)
            && is_truthy(value)
        {
            out.insert(key.to_string(), value.clone());
        }
    }
    out
}

/// `openai.py:18`. A group without a message still emits a bare `{"role": "assistant"}`.
pub fn to_chat_ctx(
    chat_ctx: &ChatContext,
    images: &dyn ImageResolver,
) -> Result<Formatted<()>, FormatError> {
    let grouping = group_tool_calls(&chat_ctx.items)?;
    let mut messages = Vec::new();

    for group in &grouping.groups {
        if group.is_empty() {
            continue;
        }

        let mut msg = match group.message {
            Some(m) => message_to_chat_item(m, images)?,
            None => {
                let mut m = Map::new();
                m.insert("role".into(), "assistant".into());
                m
            }
        };

        let mut tool_calls = Vec::new();
        for call in &group.tool_calls {
            let mut tc = Map::new();
            tc.insert("id".into(), call.call_id.clone().into());
            tc.insert("type".into(), "function".into());
            tc.insert(
                "function".into(),
                serde_json::json!({ "name": call.name, "arguments": call.arguments }),
            );
            let extra_content = filter_extra(&call.extra);
            if !extra_content.is_empty() {
                tc.insert("extra_content".into(), Value::Object(extra_content));
            }
            tool_calls.push(Value::Object(tc));
        }
        if !tool_calls.is_empty() {
            msg.insert("tool_calls".into(), Value::Array(tool_calls));
        }
        messages.push(Value::Object(msg));

        for output in &group.tool_outputs {
            messages.push(serde_json::json!({
                "role": "tool",
                "tool_call_id": output.call_id,
                "content": output.output,
            }));
        }
    }

    Ok(Formatted { messages, data: (), dropped: grouping.dropped })
}

/// `openai.py:49`; `to_chat_ctx` inlines the other arms.
fn message_to_chat_item(
    msg: &ChatMessage,
    images: &dyn ImageResolver,
) -> Result<Map<String, Value>, FormatError> {
    let (list_content, text_content) = split_content(msg, images, to_image_content, "text")?;

    let mut result = Map::new();
    result.insert("role".into(), msg.role.as_str().into());
    result.insert(
        "content".into(),
        match list_content {
            // "certain providers require text-only content in a string vs a list. for
            // max-compatibility, we will combine all text content into a single string."
            None => Value::String(text_content),
            Some(list) => Value::Array(list),
        },
    );

    let extra_content = filter_extra(&msg.extra);
    if !extra_content.is_empty() {
        result.insert("extra_content".into(), Value::Object(extra_content));
    }
    Ok(result)
}

/// Shared by both OpenAI dialects and Mistral. `None` for the list means "collapse to a bare
/// string"; otherwise the text is one trailing part, after any images. Audio parts are skipped.
pub(crate) fn split_content(
    msg: &ChatMessage,
    images: &dyn ImageResolver,
    to_image: fn(&SerializedImage) -> Value,
    text_part_type: &str,
) -> Result<(Option<Vec<Value>>, String), FormatError> {
    let mut list_content: Vec<Value> = Vec::new();
    let mut text_content = String::new();

    for content in &msg.content {
        match content {
            ChatContent::Image(image) => list_content.push(to_image(&resolve(image, images)?)),
            ChatContent::Audio(_) => {}
            ChatContent::Text(text) => {
                // Separator before the part: ["a", ""] becomes "a\n".
                if !text_content.is_empty() {
                    text_content.push('\n');
                }
                text_content.push_str(text);
            }
        }
    }

    if list_content.is_empty() {
        return Ok((None, text_content));
    }
    if !text_content.is_empty() {
        list_content.push(serde_json::json!({ "type": text_part_type, "text": text_content }));
    }
    Ok((Some(list_content), text_content))
}

pub(crate) fn resolve(
    image: &lk_agents_abi::ImageContent,
    images: &dyn ImageResolver,
) -> Result<SerializedImage, FormatError> {
    images
        .resolve(image)
        .map_err(|reason| FormatError::Image { image_id: image.id.clone(), reason })
}

/// `openai.py:104`.
fn to_image_content(img: &SerializedImage) -> Value {
    let url = match (&img.external_url, &img.data_bytes) {
        (Some(url), _) => url.clone(),
        // The reference asserts; an empty payload beats a panic.
        (None, bytes) => data_uri(img.mime_type.as_deref(), bytes.as_deref().unwrap_or(&[])),
    };
    serde_json::json!({
        "type": "image_url",
        "image_url": { "url": url, "detail": img.inference_detail.as_str() },
    })
}

/// `openai.py:122`. Flat, where Chat Completions nests.
fn to_responses_image_content(img: &SerializedImage) -> Value {
    let url = match (&img.external_url, &img.data_bytes) {
        (Some(url), _) => url.clone(),
        (None, bytes) => data_uri(img.mime_type.as_deref(), bytes.as_deref().unwrap_or(&[])),
    };
    serde_json::json!({
        "type": "input_image",
        "image_url": url,
        "detail": img.inference_detail.as_str(),
    })
}

/// `openai.py:137`: tool calls are top-level items, no `extra_content`, text parts `input_text`.
pub fn to_responses_chat_ctx(
    chat_ctx: &ChatContext,
    images: &dyn ImageResolver,
) -> Result<Formatted<()>, FormatError> {
    let grouping = group_tool_calls(&chat_ctx.items)?;
    let mut items = Vec::new();

    for group in &grouping.groups {
        if group.is_empty() {
            continue;
        }

        if let Some(message) = group.message {
            items.push(Value::Object(message_to_responses_item(message, images)?));
        }
        for call in &group.tool_calls {
            items.push(serde_json::json!({
                "call_id": call.call_id,
                "type": "function_call",
                "name": call.name,
                "arguments": call.arguments,
            }));
        }
        for output in &group.tool_outputs {
            items.push(output_to_responses_item(output));
        }
    }

    Ok(Formatted { messages: items, data: (), dropped: grouping.dropped })
}

fn message_to_responses_item(
    msg: &ChatMessage,
    images: &dyn ImageResolver,
) -> Result<Map<String, Value>, FormatError> {
    let (list_content, text_content) =
        split_content(msg, images, to_responses_image_content, "input_text")?;

    let mut item = Map::new();
    item.insert("role".into(), msg.role.as_str().into());
    item.insert(
        "content".into(),
        match list_content {
            None => Value::String(text_content),
            Some(list) => Value::Array(list),
        },
    );

    // Re-attach the assistant `phase` (#5975): dropping it degrades gpt-5.3-codex. Assistant only.
    if msg.role == Role::Assistant
        && let Some(phase) = msg
            .extra
            .get("openai")
            .and_then(Value::as_object)
            .and_then(|openai| openai.get("phase"))
        // An explicit null is `None` to the reference's `if phase is not None`.
        && !phase.is_null()
    {
        item.insert("phase".into(), phase.clone());
    }

    Ok(item)
}

fn output_to_responses_item(output: &FunctionCallOutput) -> Value {
    serde_json::json!({
        "type": "function_call_output",
        "call_id": output.call_id,
        "output": output.output,
    })
}

/// `openai.py:201`. Provider tools are not accepted.
pub fn to_fnc_ctx(tools: &[crate::llm::provider_format::ToolSchema]) -> Vec<Value> {
    tools
        .iter()
        .filter_map(|tool| match tool {
            crate::llm::provider_format::ToolSchema::Raw { raw_schema, .. } => {
                Some(serde_json::json!({ "type": "function", "function": raw_schema }))
            }
            crate::llm::provider_format::ToolSchema::Legacy(schema) | crate::llm::provider_format::ToolSchema::Strict(schema) => {
                Some(schema.clone())
            }
            crate::llm::provider_format::ToolSchema::Provider(_) => None,
        })
        .collect()
}

/// `openai.py:222`: a raw schema is spread flat and tagged `type: "function"`, function tools use
/// the legacy build, and a `ProviderTool` contributes its `to_dict()`.
pub fn to_responses_fnc_ctx(tools: &[crate::llm::provider_format::ToolSchema]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| match tool {
            crate::llm::provider_format::ToolSchema::Raw { raw_schema, .. } => {
                let mut schema = raw_schema
                    .as_object()
                    .cloned()
                    .unwrap_or_else(|| Map::from_iter([("_".into(), raw_schema.clone())]));
                schema.insert("type".into(), "function".into());
                Value::Object(schema)
            }
            crate::llm::provider_format::ToolSchema::Legacy(schema)
            | crate::llm::provider_format::ToolSchema::Strict(schema)
            | crate::llm::provider_format::ToolSchema::Provider(schema) => schema.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::provider_format::ToolSchema;
    use crate::llm::provider_format::testkit::{
        call, ctx, image, inline_images, msg, msg_with, no_images, out, url_images,
    };

    fn chat(items: Vec<lk_agents_abi::ChatItem>, images: &dyn ImageResolver) -> Vec<Value> {
        to_chat_ctx(&ctx(items), images).unwrap().messages
    }

    fn responses(items: Vec<lk_agents_abi::ChatItem>, images: &dyn ImageResolver) -> Vec<Value> {
        to_responses_chat_ctx(&ctx(items), images).unwrap().messages
    }

    /// Tests compare serialized text: `IndexMap`'s `PartialEq` ignores key order.
    fn text_of(v: &[Value]) -> String {
        serde_json::to_string(v).unwrap()
    }

    // Catches: `_filter_extra` copying every key present, or emitting `extra_content` for `{}`.
    #[test]
    fn extra_content_carries_only_truthy_vendor_keys_in_the_constants_order() {
        let mut m = msg("m", Role::User, &["hi"]);
        let e = &mut m.as_message_mut().unwrap().extra;
        e.insert("xai".into(), serde_json::json!({ "x": 1 }));
        e.insert("is_summary".into(), Value::Bool(true));
        e.insert("google".into(), serde_json::json!({ "g": 1 }));
        e.insert("livekit".into(), serde_json::json!({}));
        e.insert("anthropic".into(), serde_json::json!({ "a": 1 }));
        assert_eq!(
            text_of(&chat(vec![m], &no_images())),
            r#"[{"role":"user","content":"hi","extra_content":{"google":{"g":1},"xai":{"x":1}}}]"#
        );

        // ...and nothing truthy means no key, not an empty object.
        let mut m = msg("m", Role::User, &["hi"]);
        m.as_message_mut().unwrap().extra.insert("google".into(), serde_json::json!(0));
        assert_eq!(text_of(&chat(vec![m], &no_images())), r#"[{"role":"user","content":"hi"}]"#);
    }

    // Catches: emitting a one-element content list for a text-only message.
    #[test]
    fn text_only_content_collapses_to_a_string_and_mixed_content_does_not() {
        assert_eq!(
            text_of(&chat(vec![msg("m", Role::User, &["a", "b"])], &no_images())),
            r#"[{"role":"user","content":"a\nb"}]"#
        );
        let m = msg_with("m", Role::User, vec![image("img_1")]);
        assert_eq!(
            text_of(&chat(vec![m], &url_images(&["img_1"]))),
            r#"[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/img_1.png","detail":"high"}}]}]"#
        );
    }

    // Catches: appending the text where it appeared instead of at the end, or when empty.
    #[test]
    fn accumulated_text_is_appended_after_every_image_and_omitted_when_empty() {
        use crate::llm::provider_format::testkit::text_content;
        let m = msg_with(
            "m",
            Role::User,
            vec![text_content("before"), image("i"), text_content("after")],
        );
        assert_eq!(
            text_of(&chat(vec![m], &url_images(&["i"]))),
            r#"[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/i.png","detail":"high"}},{"type":"text","text":"before\nafter"}]}]"#
        );
        let m = msg_with("m", Role::User, vec![image("i")]);
        assert!(!text_of(&chat(vec![m], &url_images(&["i"]))).contains(r#""type":"text""#));
    }

    // Catches: joining with the separator after each part, or skipping empty parts (Anthropic's rule).
    #[test]
    fn an_empty_trailing_text_part_still_contributes_its_newline() {
        assert_eq!(
            text_of(&chat(vec![msg("m", Role::User, &["a", ""])], &no_images())),
            r#"[{"role":"user","content":"a\n"}]"#
        );
        assert_eq!(
            text_of(&chat(vec![msg("m", Role::User, &["", "a"])], &no_images())),
            r#"[{"role":"user","content":"a"}]"#,
            "a leading empty part contributes nothing, because the separator comes first"
        );
    }

    // Catches: dropping a message-less tool-call group's placeholder.
    #[test]
    fn a_message_less_group_still_emits_a_bare_assistant_message_to_hang_calls_off() {
        let items = vec![call("c", "c0", "lookup", "{\"q\":1}"), out("o", "c0", "lookup", "ok")];
        assert_eq!(
            text_of(&chat(items, &no_images())),
            r#"[{"role":"assistant","tool_calls":[{"id":"c0","type":"function","function":{"name":"lookup","arguments":"{\"q\":1}"}}]},{"role":"tool","tool_call_id":"c0","content":"ok"}]"#
        );
    }

    // Catches: emitting `tool_calls` before `content`/`extra_content`.
    #[test]
    fn tool_calls_are_appended_after_content_and_extra_content() {
        let mut m = msg("t", Role::Assistant, &["thinking"]);
        m.as_message_mut().unwrap().extra.insert("xai".into(), serde_json::json!(1));
        let items = vec![m, call("t/0", "c0", "f", "{}"), out("o", "c0", "f", "ok")];
        let out_text = text_of(&chat(items, &no_images()));
        let content_at = out_text.find(r#""content""#).unwrap();
        let extra_at = out_text.find(r#""extra_content""#).unwrap();
        let calls_at = out_text.find(r#""tool_calls""#).unwrap();
        assert!(content_at < extra_at && extra_at < calls_at, "{out_text}");
    }

    // Catches: filtering a tool call's extra with a different rule than a message's.
    #[test]
    fn a_tool_calls_extra_content_is_filtered_the_same_way() {
        let mut c = call("c", "c0", "f", "{}");
        if let lk_agents_abi::ChatItem::FunctionCall(f) = &mut c {
            f.extra.insert("google".into(), serde_json::json!({ "sig": "x" }));
            f.extra.insert("is_summary".into(), Value::Bool(true));
        }
        let items = vec![c, out("o", "c0", "f", "ok")];
        assert!(
            text_of(&chat(items, &no_images()))
                .contains(r#""extra_content":{"google":{"sig":"x"}}}]"#)
        );
    }

    // Catches: base64-ing an image that already has an external URL, or dropping `detail`.
    #[test]
    fn an_inline_image_becomes_a_data_uri_and_a_url_image_is_passed_through() {
        let m = msg_with("m", Role::User, vec![image("i")]);
        assert_eq!(
            text_of(&chat(vec![m.clone()], &inline_images(&["i"]))),
            r#"[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,/9j/","detail":"auto"}}]}]"#
        );
        assert!(
            text_of(&chat(vec![m], &url_images(&["i"]))).contains("https://example.test/i.png")
        );
    }

    // Catches: emitting the Chat Completions shapes from the Responses converter.
    #[test]
    fn the_responses_dialect_emits_top_level_calls_and_no_extra_content() {
        let mut m = msg("t", Role::Assistant, &["thinking"]);
        m.as_message_mut().unwrap().extra.insert("google".into(), serde_json::json!(1));
        let items = vec![m, call("t/0", "c0", "f", "{}"), out("o", "c0", "f", "ok")];
        assert_eq!(
            text_of(&responses(items, &no_images())),
            r#"[{"role":"assistant","content":"thinking"},{"call_id":"c0","type":"function_call","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"c0","output":"ok"}]"#
        );
    }

    // Catches: emitting a placeholder assistant item for a message-less group in Responses.
    #[test]
    fn the_responses_dialect_emits_no_placeholder_for_a_message_less_group() {
        let items = vec![call("c", "c0", "f", "{}"), out("o", "c0", "f", "ok")];
        assert!(!text_of(&responses(items, &no_images())).contains(r#""role""#));
    }

    // Catches: echoing `phase` back on a user item, or on a null phase.
    #[test]
    fn phase_is_echoed_only_for_an_assistant_item_and_only_when_present() {
        let with_phase = |role, phase: Value| {
            let mut m = msg("m", role, &["x"]);
            m.as_message_mut()
                .unwrap()
                .extra
                .insert("openai".into(), serde_json::json!({ "phase": phase }));
            text_of(&responses(vec![m], &no_images()))
        };
        assert!(
            with_phase(Role::Assistant, "commentary".into()).contains(r#""phase":"commentary""#)
        );
        assert!(!with_phase(Role::User, "commentary".into()).contains("phase"));
        assert!(!with_phase(Role::Assistant, Value::Null).contains("phase"));
        // The key lands after `content`, which is where the reference's assignment puts it.
        let out_text = with_phase(Role::Assistant, "final_answer".into());
        assert!(out_text.find(r#""content""#) < out_text.find(r#""phase""#), "{out_text}");
    }

    // Pins a deliberate deviation: a non-dict `extra["openai"]` raises in the reference; here it is
    // "no phase" and the request is still formatted.
    #[test]
    fn a_non_object_openai_extra_yields_no_phase_rather_than_the_references_attributeerror() {
        for garbage in
            [serde_json::json!("commentary"), serde_json::json!(7), serde_json::json!([])]
        {
            let mut m = msg("m", Role::Assistant, &["x"]);
            m.as_message_mut().unwrap().extra.insert("openai".into(), garbage);
            let text = text_of(&responses(vec![m], &no_images()));
            assert_eq!(text, r#"[{"role":"assistant","content":"x"}]"#);
        }
    }

    // Catches: nesting a raw schema under `function` in Responses, or not tagging it.
    #[test]
    fn the_two_fnc_ctx_dialects_wrap_a_raw_schema_differently() {
        let raw = ToolSchema::Raw {
            name: "t".into(),
            raw_schema: serde_json::json!({ "name": "t", "parameters": {} }),
        };
        assert_eq!(
            serde_json::to_string(&to_fnc_ctx(std::slice::from_ref(&raw))).unwrap(),
            r#"[{"type":"function","function":{"name":"t","parameters":{}}}]"#
        );
        assert_eq!(
            serde_json::to_string(&to_responses_fnc_ctx(std::slice::from_ref(&raw))).unwrap(),
            r#"[{"name":"t","parameters":{},"type":"function"}]"#
        );
    }

    // Catches: dropping provider tools from Responses, or letting them into Chat Completions.
    #[test]
    fn provider_tools_reach_the_responses_dialect_only() {
        let tools = [ToolSchema::Provider(serde_json::json!({ "type": "web_search" }))];
        assert!(to_fnc_ctx(&tools).is_empty());
        assert_eq!(to_responses_fnc_ctx(&tools), [serde_json::json!({ "type": "web_search" })]);
    }

    // Catches: reusing the Chat Completions image shape in Responses (flat `image_url` string).
    #[test]
    fn the_responses_dialect_flattens_the_image_url_and_renames_the_part_type() {
        let m = msg_with("m", Role::User, vec![image("i")]);
        assert_eq!(
            text_of(&responses(vec![m.clone()], &url_images(&["i"]))),
            r#"[{"role":"user","content":[{"type":"input_image","image_url":"https://example.test/i.png","detail":"high"}]}]"#
        );
        assert_eq!(
            text_of(&responses(vec![m], &inline_images(&["i"]))),
            r#"[{"role":"user","content":[{"type":"input_image","image_url":"data:image/jpeg;base64,/9j/","detail":"auto"}]}]"#
        );
    }

    // Catches: emitting `{"type":"text"}` for a Responses text part.
    #[test]
    fn the_responses_dialect_names_its_trailing_text_part_input_text() {
        use crate::llm::provider_format::testkit::text_content;
        let m = msg_with("m", Role::User, vec![image("i"), text_content("caption")]);
        assert!(
            text_of(&responses(vec![m], &url_images(&["i"])))
                .contains(r#"{"type":"input_text","text":"caption"}"#)
        );
    }
}
