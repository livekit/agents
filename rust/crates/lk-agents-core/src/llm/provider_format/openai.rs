//! OpenAI Chat Completions and Responses request items.

use super::group::group_tool_calls;
use super::wire::{Content, FunctionDefinition};
use super::{
    FormatError, Formatted, ImageResolver, SerializedImage, ToolSchema, data_uri, is_truthy,
};
use lk_agents_abi::llm::provider_format::vocabulary::ResponsesPhase as Phase;
use lk_agents_abi::{ChatContent, ChatContext, ChatMessage, Extra, InferenceDetail, Role};
use serde::Serialize;
use serde_json::Value;

pub const EXTRA_CONTENT_KEYS: [&str; 4] = ["google", "livekit", "xai", "openai"];

/// Namespaced vendor extensions; their payloads are opaque to the chat schema.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ExtraContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub livekit: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xai: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai: Option<Value>,
}

fn filter_extra(extra: &Extra) -> Option<ExtraContent> {
    let [google, livekit, xai, openai] =
        EXTRA_CONTENT_KEYS.map(|key| extra.get(key).filter(|value| is_truthy(value)).cloned());
    let content = ExtraContent { google, livekit, xai, openai };
    (content != ExtraContent::default()).then_some(content)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        extra_content: Option<ExtraContent>,
    },
    Developer {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        extra_content: Option<ExtraContent>,
    },
    User {
        content: Content<UserPart>,
        #[serde(skip_serializing_if = "Option::is_none")]
        extra_content: Option<ExtraContent>,
    },
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        extra_content: Option<ExtraContent>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(serialize_with = "super::wire::serialize_wire")]
    pub detail: InferenceDetail,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(flatten)]
    pub body: FunctionCallBody,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_content: Option<ExtraContent>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FunctionCallBody {
    Function { function: FunctionCallArguments },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FunctionCallArguments {
    pub name: String,
    pub arguments: String,
}

pub fn to_chat_ctx(
    chat_ctx: &ChatContext,
    images: &dyn ImageResolver,
) -> Result<Formatted<Message, ()>, FormatError> {
    let grouping = group_tool_calls(&chat_ctx.items)?;
    let mut messages = Vec::new();
    for group in &grouping.groups {
        if group.is_empty() {
            continue;
        }
        let calls = group
            .tool_calls
            .iter()
            .map(|call| ToolCall {
                id: call.call_id.clone(),
                body: FunctionCallBody::Function {
                    function: FunctionCallArguments {
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    },
                },
                extra_content: filter_extra(&call.extra),
            })
            .collect();
        let message = match group.message {
            Some(message) => message_to_chat_item(message, images, calls)?,
            None => Message::Assistant { content: None, extra_content: None, tool_calls: calls },
        };
        messages.push(message);
        messages.extend(group.tool_outputs.iter().map(|output| Message::Tool {
            tool_call_id: output.call_id.clone(),
            content: output.output.clone(),
        }));
    }
    Ok(Formatted { messages, data: (), dropped: grouping.dropped })
}

fn message_to_chat_item(
    message: &ChatMessage,
    images: &dyn ImageResolver,
    tool_calls: Vec<ToolCall>,
) -> Result<Message, FormatError> {
    let extra_content = filter_extra(&message.extra);
    Ok(match message.role {
        Role::System => Message::System { content: text_content(message), extra_content },
        Role::Developer => Message::Developer { content: text_content(message), extra_content },
        Role::User => Message::User {
            content: split_content(message, images, to_image_content, |text| UserPart::Text {
                text,
            })?,
            extra_content,
        },
        Role::Assistant => {
            Message::Assistant { content: Some(text_content(message)), extra_content, tool_calls }
        }
    })
}

pub(crate) fn text_content(message: &ChatMessage) -> String {
    let mut text_content = String::new();
    for content in &message.content {
        if let ChatContent::Text(text) = content {
            if !text_content.is_empty() {
                text_content.push('\n');
            }
            text_content.push_str(text);
        }
    }
    text_content
}

/// Images first, joined text last; a text-only message stays a string.
pub(crate) fn split_content<P>(
    message: &ChatMessage,
    images: &dyn ImageResolver,
    to_image: impl Fn(&SerializedImage) -> P,
    to_text: impl Fn(String) -> P,
) -> Result<Content<P>, FormatError> {
    let mut parts = Vec::new();
    for content in &message.content {
        if let ChatContent::Image(image) = content {
            parts.push(to_image(&resolve(image, images)?));
        }
    }
    let text = text_content(message);
    if parts.is_empty() {
        return Ok(Content::Text(text));
    }
    if !text.is_empty() {
        parts.push(to_text(text));
    }
    Ok(Content::Parts(parts))
}

pub(crate) fn resolve(
    image: &lk_agents_abi::ImageContent,
    images: &dyn ImageResolver,
) -> Result<SerializedImage, FormatError> {
    images
        .resolve(image)
        .map_err(|reason| FormatError::Image { image_id: image.id.clone(), reason })
}

pub(crate) fn image_url(image: &SerializedImage) -> String {
    image.external_url.clone().unwrap_or_else(|| {
        data_uri(image.mime_type.as_deref(), image.data_bytes.as_deref().unwrap_or_default())
    })
}

fn to_image_content(image: &SerializedImage) -> UserPart {
    UserPart::ImageUrl {
        image_url: ImageUrl { url: image_url(image), detail: image.inference_detail },
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum InputItem {
    // EasyInputMessage's type discriminator is optional.
    Message {
        #[serde(serialize_with = "super::wire::serialize_wire")]
        role: Role,
        content: Content<InputPart>,
        #[serde(skip_serializing_if = "Option::is_none")]
        phase: Option<Phase>,
    },
    Event(InputEvent),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputEvent {
    FunctionCall { call_id: String, name: String, arguments: String },
    FunctionCallOutput { call_id: String, output: String },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputPart {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        #[serde(serialize_with = "super::wire::serialize_wire")]
        detail: InferenceDetail,
    },
}

#[derive(serde::Deserialize)]
struct ResponsesExtra {
    phase: Option<Phase>,
}

pub fn to_responses_chat_ctx(
    chat_ctx: &ChatContext,
    images: &dyn ImageResolver,
) -> Result<Formatted<InputItem, ()>, FormatError> {
    let grouping = group_tool_calls(&chat_ctx.items)?;
    let mut items = Vec::new();
    for group in &grouping.groups {
        if group.is_empty() {
            continue;
        }
        if let Some(message) = group.message {
            let content = if message.role == Role::User {
                split_content(
                    message,
                    images,
                    |image| InputPart::InputImage {
                        image_url: image_url(image),
                        detail: image.inference_detail,
                    },
                    |text| InputPart::InputText { text },
                )?
            } else {
                Content::Text(text_content(message))
            };
            let phase = (message.role == Role::Assistant)
                .then(|| {
                    message.extra.get("openai").cloned().and_then(|extra| {
                        serde_json::from_value::<ResponsesExtra>(extra).ok()?.phase
                    })
                })
                .flatten();
            items.push(InputItem::Message { role: message.role, content, phase });
        }
        items.extend(group.tool_calls.iter().map(|call| {
            InputItem::Event(InputEvent::FunctionCall {
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            })
        }));
        items.extend(group.tool_outputs.iter().map(|output| {
            InputItem::Event(InputEvent::FunctionCallOutput {
                call_id: output.call_id.clone(),
                output: output.output.clone(),
            })
        }));
    }
    Ok(Formatted { messages: items, data: (), dropped: grouping.dropped })
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename = "function")]
pub struct ChatTool {
    pub function: FunctionDefinition,
}

pub fn to_fnc_ctx(tools: &[ToolSchema]) -> Result<Vec<ChatTool>, FormatError> {
    tools
        .iter()
        .filter(|tool| !matches!(tool, ToolSchema::Provider(_)))
        .map(|tool| Ok(ChatTool { function: FunctionDefinition::from_tool(tool)? }))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ResponsesTool {
    Function(ResponsesFunction),
    Provider(Value),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename = "function")]
pub struct ResponsesFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Option<Value>,
    pub strict: Option<bool>,
}

pub fn to_responses_fnc_ctx(tools: &[ToolSchema]) -> Result<Vec<ResponsesTool>, FormatError> {
    tools
        .iter()
        .map(|tool| {
            if let ToolSchema::Provider(schema) = tool {
                return Ok(ResponsesTool::Provider(schema.clone()));
            }
            let definition = FunctionDefinition::from_tool(tool)?;
            Ok(ResponsesTool::Function(ResponsesFunction {
                name: definition.name,
                description: definition.description,
                parameters: definition.parameters,
                strict: definition.strict,
            }))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::provider_format::ToolSchema;
    use crate::llm::provider_format::testkit::{
        call, ctx, image, inline_images, msg, msg_with, no_images, out, text_of, url_images,
    };

    fn chat(items: Vec<lk_agents_abi::ChatItem>, images: &dyn ImageResolver) -> Vec<Message> {
        to_chat_ctx(&ctx(items), images).unwrap().messages
    }

    fn responses(
        items: Vec<lk_agents_abi::ChatItem>,
        images: &dyn ImageResolver,
    ) -> Vec<InputItem> {
        to_responses_chat_ctx(&ctx(items), images).unwrap().messages
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
            r#"[{"role":"assistant","content":"thinking"},{"type":"function_call","call_id":"c0","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"c0","output":"ok"}]"#
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
            serde_json::to_string(&to_fnc_ctx(std::slice::from_ref(&raw)).unwrap()).unwrap(),
            r#"[{"type":"function","function":{"name":"t","parameters":{}}}]"#
        );
        assert_eq!(
            serde_json::to_string(&to_responses_fnc_ctx(std::slice::from_ref(&raw)).unwrap())
                .unwrap(),
            r#"[{"type":"function","name":"t","parameters":{},"strict":null}]"#
        );
    }

    // Catches: dropping provider tools from Responses, or letting them into Chat Completions.
    #[test]
    fn provider_tools_reach_the_responses_dialect_only() {
        let tools = [ToolSchema::Provider(serde_json::json!({ "type": "web_search" }))];
        assert!(to_fnc_ctx(&tools).unwrap().is_empty());
        assert_eq!(
            serde_json::to_value(to_responses_fnc_ctx(&tools).unwrap()).unwrap(),
            serde_json::json!([{ "type": "web_search" }])
        );
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
