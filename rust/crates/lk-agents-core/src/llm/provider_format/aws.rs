use super::turns::{Dialect, ResultsTurn, Speaker, Turns, merge_turns};
use super::wire::{FunctionDefinition, HostBytes};
use super::{
    FormatError, Formatted, ImageResolver, SerializedImage, ToolSchema, is_truthy, parse_arguments,
};
use lk_agents_abi::llm::provider_format::vocabulary::{
    BedrockImageFormat as ImageFormat, ConversationRole as MessageRole, ImageMime, ToolResultStatus,
};
use lk_agents_abi::{ChatContext, FunctionCall, FunctionCallOutput};
use serde::Serialize;
use serde_json::Value;

const PROVIDER: &str = "AWS Bedrock";

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct BedrockFormatData {
    pub system_messages: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ContentBlock {
    Text(String),
    Image(ImageBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolUseBlock {
    pub tool_use_id: String,
    pub name: String,
    pub input: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultBlock {
    pub tool_use_id: String,
    pub content: Vec<ToolResultContent>,
    pub status: ToolResultStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ToolResultContent {
    Text(String),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ImageBlock {
    pub format: ImageFormat,
    pub source: ImageSource,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ImageSource {
    Bytes(HostBytes),
}

struct Bedrock;

impl Dialect for Bedrock {
    type Block = ContentBlock;
    const RESULTS: ResultsTurn = ResultsTurn::WithUser;
    const USER_IMAGES_ONLY: bool = true;

    fn text(&self, text: &str) -> ContentBlock {
        ContentBlock::Text(text.into())
    }
    fn image(&self, image: &SerializedImage) -> Result<ContentBlock, FormatError> {
        build_image(image)
    }
    fn call(&self, call: &FunctionCall) -> Result<ContentBlock, FormatError> {
        Ok(ContentBlock::ToolUse(ToolUseBlock {
            tool_use_id: call.call_id.clone(),
            name: call.name.clone(),
            input: parse_arguments(call),
        }))
    }
    fn result(&self, output: &FunctionCallOutput) -> ContentBlock {
        ContentBlock::ToolResult(ToolResultBlock {
            tool_use_id: output.call_id.clone(),
            content: vec![ToolResultContent::Text(output.output.clone())],
            status: if output.is_error {
                ToolResultStatus::Error
            } else {
                ToolResultStatus::Success
            },
        })
    }
}

pub fn to_chat_ctx(
    chat_ctx: &ChatContext,
    images: &dyn ImageResolver,
    inject_dummy_user_message: bool,
) -> Result<Formatted<Message, BedrockFormatData>, FormatError> {
    let Turns { turns, system_messages, dropped, .. } = merge_turns(chat_ctx, images, &Bedrock)?;
    let mut messages: Vec<_> = turns
        .into_iter()
        .map(|turn| Message {
            role: if turn.speaker == Speaker::Assistant {
                MessageRole::Assistant
            } else {
                MessageRole::User
            },
            content: turn.blocks,
        })
        .collect();
    if inject_dummy_user_message && messages.first().map(|m| m.role) != Some(MessageRole::User) {
        messages.insert(
            0,
            Message {
                role: MessageRole::User,
                content: vec![ContentBlock::Text("(empty)".into())],
            },
        );
    }
    Ok(Formatted { messages, data: BedrockFormatData { system_messages }, dropped })
}

fn build_image(image: &SerializedImage) -> Result<ContentBlock, FormatError> {
    if image.external_url.is_some() {
        return Err(FormatError::ExternalUrlUnsupported { provider: PROVIDER });
    }
    Ok(ContentBlock::Image(ImageBlock {
        format: image_format(image.mime_type.as_deref())?,
        source: ImageSource::Bytes(HostBytes(image.data_bytes.clone().unwrap_or_default())),
    }))
}

fn image_format(mime_type: Option<&str>) -> Result<ImageFormat, FormatError> {
    let mime = mime_type
        .filter(|mime| !mime.is_empty())
        .map(|mime_type| {
            ImageMime::from_wire(mime_type).ok_or_else(|| FormatError::UnsupportedImageMime {
                provider: PROVIDER,
                mime_type: mime_type.into(),
            })
        })
        .transpose()?
        .unwrap_or(ImageMime::Jpeg);
    Ok(match mime {
        ImageMime::Jpeg => ImageFormat::Jpeg,
        ImageMime::Png => ImageFormat::Png,
        ImageMime::Gif => ImageFormat::Gif,
        ImageMime::Webp => ImageFormat::Webp,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Tool {
    ToolSpec(ToolSpec),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolSpec {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: InputSchema,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InputSchema {
    pub json: Value,
}

pub fn to_fnc_ctx(tools: &[ToolSchema]) -> Result<Vec<Tool>, FormatError> {
    tools
        .iter()
        .map(|tool| {
            if !matches!(tool, ToolSchema::Raw { .. } | ToolSchema::Legacy(_)) {
                return Err(FormatError::MalformedToolSchema {
                    detail: format!("{PROVIDER} builds legacy schemas only, got {tool:?}"),
                });
            }
            let definition = FunctionDefinition::from_tool(tool)?;
            Ok(Tool::ToolSpec(ToolSpec {
                name: definition.name,
                description: definition.description.filter(|description| !description.is_empty()),
                input_schema: InputSchema {
                    json: definition
                        .parameters
                        .filter(is_truthy)
                        .unwrap_or_else(|| Value::Object(Default::default())),
                },
            }))
        })
        .collect()
}

#[cfg(test)]
#[path = "aws_tests.rs"]
mod tests;
