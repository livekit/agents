//! The turn-merging loop `anthropic.py`, `aws.py` and `google.py` each write out: hoist the system
//! prompt, then coalesce consecutive items of one effective role into a single turn. Each dialect
//! supplies a [`Dialect`] table of role names and leaf builders, and runs its own final pass.

use std::borrow::Cow;

use lk_agents_abi::{ChatContent, ChatContext, FunctionCall, FunctionCallOutput, Role};
use serde_json::Value;

use crate::llm::provider_format::FormatError;
use crate::llm::provider_format::group::{
    DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE, Dropped, GroupItem,
    convert_mid_conversation_instructions, group_tool_calls,
};
use crate::llm::provider_format::image::{ImageResolver, SerializedImage};
use crate::llm::provider_format::openai::resolve;

pub(crate) struct Dialect<'a> {
    /// The role of an assistant message and of a tool call.
    pub assistant: &'static str,
    /// The role a tool result accumulates under (Gemini's synthetic `tool`).
    pub output: &'static str,
    /// `content` or `parts`.
    pub key: &'static str,
    pub text: fn(&str) -> Value,
    pub image: fn(&SerializedImage) -> Result<Value, FormatError>,
    pub call: &'a dyn Fn(&FunctionCall) -> Result<Value, FormatError>,
    pub result: fn(&FunctionCallOutput) -> Value,
}

pub(crate) struct Turns {
    pub turns: Vec<Value>,
    pub system_messages: Vec<String>,
    /// The last role seen, including one whose item contributed no blocks.
    pub last_role: Option<&'static str>,
    pub dropped: Vec<Dropped>,
}

pub(crate) fn merge_turns(
    chat_ctx: &ChatContext,
    images: &dyn ImageResolver,
    dialect: &Dialect<'_>,
) -> Result<Turns, FormatError> {
    let converted = convert_mid_conversation_instructions(
        chat_ctx,
        Role::User,
        DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE,
    );
    let grouping = group_tool_calls(converted.iter().map(Cow::as_ref))?;

    let mut turns: Vec<Value> = Vec::new();
    let mut system_messages: Vec<String> = Vec::new();
    let mut current_role: Option<&'static str> = None;
    let mut content: Vec<Value> = Vec::new();
    let flush = |role: &'static str, content: Vec<Value>, turns: &mut Vec<Value>| {
        if !content.is_empty() {
            let mut turn = serde_json::Map::new();
            turn.insert("role".into(), role.into());
            turn.insert(dialect.key.into(), Value::Array(content));
            turns.push(Value::Object(turn));
        }
    };

    for item in grouping.groups.iter().flat_map(|g| g.flatten()) {
        let role = match item {
            // Only `role == "system"` with truthy text is hoisted; `developer` and an empty
            // system message fall through to a user turn, as in the reference.
            GroupItem::Message(m) if m.role == Role::System => {
                match m.raw_text_content().filter(|t| !t.is_empty()) {
                    Some(text) => {
                        system_messages.push(text);
                        continue;
                    }
                    None => "user",
                }
            }
            GroupItem::Message(m) if m.role == Role::Assistant => dialect.assistant,
            GroupItem::Message(_) => "user",
            GroupItem::Call(_) => dialect.assistant,
            GroupItem::Output(_) => dialect.output,
        };

        if Some(role) != current_role {
            // An empty accumulator emits no turn but still flips `current_role`.
            if let Some(previous) = current_role {
                flush(previous, std::mem::take(&mut content), &mut turns);
            }
            current_role = Some(role);
        }

        match item {
            GroupItem::Message(m) => {
                for part in &m.content {
                    match part {
                        ChatContent::Image(image) => {
                            content.push((dialect.image)(&resolve(image, images)?)?);
                        }
                        ChatContent::Audio(_) => {}
                        // `elif c:`: empty parts are dropped, the rest stay separate blocks.
                        ChatContent::Text(text) if !text.is_empty() => {
                            content.push((dialect.text)(text));
                        }
                        ChatContent::Text(_) => {}
                    }
                }
            }
            GroupItem::Call(call) => content.push((dialect.call)(call)?),
            GroupItem::Output(output) => content.push((dialect.result)(output)),
        }
    }

    if let Some(role) = current_role {
        flush(role, content, &mut turns);
    }

    Ok(Turns { turns, system_messages, last_role: current_role, dropped: grouping.dropped })
}
