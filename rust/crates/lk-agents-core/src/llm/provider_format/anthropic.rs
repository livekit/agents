use super::turns::{Dialect, ResultsTurn, Speaker, Turns, merge_turns};
use super::wire::FunctionDefinition;
use super::{
    FormatError, Formatted, ImageResolver, SerializedImage, ToolSchema, b64_encode, parse_arguments,
};
use lk_agents_abi::llm::provider_format::vocabulary::{ConversationRole as MessageRole, ImageMime};
use lk_agents_abi::{ChatContext, FunctionCall, FunctionCallOutput};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct AnthropicFormatData {
    pub system_messages: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    pub inject_dummy_user_message: bool,
    pub inject_trailing_user_message: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { inject_dummy_user_message: true, inject_trailing_user_message: false }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: Vec<Block>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    Text { text: String },
    Image { source: ImageSource },
    ToolUse { id: String, name: String, input: serde_json::Map<String, Value> },
    ToolResult { tool_use_id: String, content: ToolResultContent, is_error: bool },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 {
        data: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        media_type: Option<ImageMime>,
    },
    Url {
        url: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultBlock>),
}

/// Decode only supported tool-result blocks. Other JSON stays a text result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ToolResultBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<Value>>,
    },
    Image {
        source: ResultImageSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResultImageSource {
    Base64 { data: String, media_type: ImageMime },
    Url { url: String },
    File { file_id: String },
}

struct Anthropic;

impl Dialect for Anthropic {
    type Block = Block;
    const RESULTS: ResultsTurn = ResultsTurn::WithUser;

    fn text(&self, text: &str) -> Block {
        Block::Text { text: text.into() }
    }
    fn image(&self, image: &SerializedImage) -> Result<Block, FormatError> {
        Ok(to_image_content(image))
    }
    fn call(&self, call: &FunctionCall) -> Result<Block, FormatError> {
        Ok(Block::ToolUse {
            id: call.call_id.clone(),
            name: call.name.clone(),
            input: parse_arguments(call),
        })
    }
    fn result(&self, output: &FunctionCallOutput) -> Block {
        let content = serde_json::from_str::<Vec<ToolResultBlock>>(&output.output)
            .map(ToolResultContent::Blocks)
            .unwrap_or_else(|_| ToolResultContent::Text(output.output.clone()));
        Block::ToolResult {
            tool_use_id: output.call_id.clone(),
            content,
            is_error: output.is_error,
        }
    }
}

pub fn to_chat_ctx(
    chat_ctx: &ChatContext,
    images: &dyn ImageResolver,
    opts: Options,
) -> Result<Formatted<Message, AnthropicFormatData>, FormatError> {
    let Turns { turns, system_messages, dropped, .. } = merge_turns(chat_ctx, images, &Anthropic)?;
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
    if opts.inject_dummy_user_message && messages.first().map(|m| m.role) != Some(MessageRole::User)
    {
        messages.insert(0, dummy_turn("(empty)"));
    }
    if opts.inject_trailing_user_message
        && messages.last().map(|m| m.role) == Some(MessageRole::Assistant)
    {
        messages.push(dummy_turn("."));
    }
    Ok(Formatted { messages, data: AnthropicFormatData { system_messages }, dropped })
}

fn dummy_turn(text: &str) -> Message {
    Message { role: MessageRole::User, content: vec![Block::Text { text: text.into() }] }
}

fn to_image_content(image: &SerializedImage) -> Block {
    let source = if let Some(url) = &image.external_url {
        ImageSource::Url { url: url.clone() }
    } else {
        ImageSource::Base64 {
            data: b64_encode(image.data_bytes.as_deref().unwrap_or_default()),
            media_type: image.mime_type.as_deref().and_then(ImageMime::from_wire),
        }
    };
    Block::Image { source }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

pub fn to_fnc_ctx(tools: &[ToolSchema]) -> Result<Vec<Tool>, FormatError> {
    tools
        .iter()
        .filter(|tool| !matches!(tool, ToolSchema::Provider(_)))
        .map(|tool| {
            let definition = FunctionDefinition::from_tool(tool)?;
            Ok(Tool {
                name: definition.name,
                description: definition.description.unwrap_or_default(),
                input_schema: definition
                    .parameters
                    .unwrap_or_else(|| Value::Object(Default::default())),
                strict: matches!(tool, ToolSchema::Strict(_)).then_some(true),
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "anthropic_tests.rs"]
mod tests;
