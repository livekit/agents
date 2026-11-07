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
            parts.push(to_image(&images.resolve(image)?));
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
#[path = "openai_tests.rs"]
mod tests;
