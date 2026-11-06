//! Anthropic Messages content blocks, with the system prompt out of band.

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
mod tests {
    use lk_agents_abi::{ChatItem, Role};

    use super::*;
    use crate::llm::provider_format::ToolSchema;
    use crate::llm::provider_format::testkit::{
        call, ctx, image, inline_images, msg, msg_with, no_images, out, text_content, text_of,
        url_images,
    };

    fn run(
        items: Vec<ChatItem>,
        images: &dyn ImageResolver,
    ) -> Formatted<Message, AnthropicFormatData> {
        to_chat_ctx(&ctx(items), images, Options::default()).unwrap()
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
            text_of(&run(items, &no_images()).messages),
            r#"[{"role":"user","content":[{"type":"text","text":"question"}]},{"role":"assistant","content":[{"type":"text","text":"thinking"},{"type":"tool_use","id":"c0","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"c0","content":"ok","is_error":false},{"type":"text","text":"more"}]}]"#
        );
    }

    // Catches: hoisting `developer` into `system_messages` alongside `system`.
    #[test]
    fn only_the_system_role_is_hoisted_out_of_band_not_developer() {
        let f = run(vec![msg("d", Role::Developer, &["dev preamble"])], &no_images());
        assert!(f.data.system_messages.is_empty());
        assert_eq!(
            text_of(&f.messages),
            r#"[{"role":"user","content":[{"type":"text","text":"dev preamble"}]}]"#
        );

        let f = run(vec![msg("s", Role::System, &["sys preamble"])], &no_images());
        assert_eq!(f.data.system_messages, ["sys preamble"]);
    }

    // Catches: reading OpenAI's text rule into this dialect.
    #[test]
    fn empty_text_parts_are_dropped_rather_than_concatenated() {
        assert_eq!(
            text_of(&run(vec![msg("m", Role::User, &["a", "", "b"])], &no_images()).messages),
            r#"[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]"#
        );
    }

    // Catches: stringifying valid blocks, or interpreting arbitrary JSON as content blocks.
    #[test]
    fn only_a_json_list_tool_result_is_unwrapped_into_content_blocks() {
        let unwrap = |output: &str| {
            let items = vec![call("c", "c0", "f", "{}"), out("o", "c0", "f", output)];
            text_of(&run(items, &no_images()).messages)
        };
        assert!(
            unwrap(r#"[{"type":"text","text":"ok"}]"#)
                .contains(r#""content":[{"type":"text","text":"ok"}]"#)
        );
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
        assert!(text_of(&run(items, &no_images()).messages).contains(r#""is_error":true"#));
    }

    // Catches: appending the "(empty)" turn, or prepending it when the conversation opens on user.
    #[test]
    fn the_dummy_user_turn_is_prepended_and_only_when_needed() {
        let f = run(vec![msg("a", Role::Assistant, &["hi"])], &no_images());
        assert_eq!(
            text_of(&f.messages),
            r#"[{"role":"user","content":[{"type":"text","text":"(empty)"}]},{"role":"assistant","content":[{"type":"text","text":"hi"}]}]"#
        );
        assert_eq!(
            text_of(&run(vec![msg("u", Role::User, &["hi"])], &no_images()).messages),
            r#"[{"role":"user","content":[{"type":"text","text":"hi"}]}]"#
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
            !text_of(&run(items.clone(), &no_images()).messages)
                .contains(r#"{"type":"text","text":"."}"#)
        );

        let opts = Options { inject_dummy_user_message: true, inject_trailing_user_message: true };
        let f = to_chat_ctx(&ctx(items), &no_images(), opts).unwrap();
        assert!(
            text_of(&f.messages)
                .ends_with(r#"{"role":"user","content":[{"type":"text","text":"."}]}]"#)
        );

        let f = to_chat_ctx(&ctx(vec![msg("u", Role::User, &["q"])]), &no_images(), opts).unwrap();
        assert_eq!(f.messages.len(), 1, "a trailing user turn needs no help");
    }

    // Catches: emitting OpenAI's `data:` URI shape.
    #[test]
    fn images_use_anthropics_source_object_not_a_data_uri() {
        let m = msg_with("m", Role::User, vec![image("i")]);
        assert!(text_of(&run(vec![m.clone()], &inline_images(&["i"])).messages).contains(
            r#"{"type":"image","source":{"type":"base64","data":"/9j/","media_type":"image/jpeg"}}"#
        ));
        assert!(text_of(&run(vec![m], &url_images(&["i"])).messages).contains(
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
            text_of(&run(items, &no_images()).messages),
            r#"[{"role":"user","content":[{"type":"text","text":"q"}]},{"role":"user","content":[{"type":"text","text":"more"}]}]"#
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
