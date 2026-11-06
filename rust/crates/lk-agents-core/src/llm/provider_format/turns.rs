//! Coalesce by speaker, independently of a provider's wire role and message shape.

use super::group::{
    DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE, Dropped, GroupItem,
    convert_mid_conversation_instructions, group_tool_calls,
};
use super::openai::resolve;
use super::{FormatError, ImageResolver, SerializedImage};
use lk_agents_abi::{
    ChatContent, ChatContext, ChatMessage, FunctionCall, FunctionCallOutput, Role,
};
use std::borrow::Cow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Speaker {
    User,
    Assistant,
    ToolResults,
}

pub(crate) enum ResultsTurn {
    WithUser,
    Separate,
}

pub(crate) trait Dialect {
    type Block;
    const RESULTS: ResultsTurn;
    const USER_IMAGES_ONLY: bool = false;
    fn text(&self, text: &str) -> Self::Block;
    fn image(&self, image: &SerializedImage) -> Result<Self::Block, FormatError>;
    fn call(&self, call: &FunctionCall) -> Result<Self::Block, FormatError>;
    fn result(&self, output: &FunctionCallOutput) -> Self::Block;
}

pub(crate) struct Turn<B> {
    pub speaker: Speaker,
    pub blocks: Vec<B>,
}

pub(crate) struct Turns<B> {
    pub turns: Vec<Turn<B>>,
    pub system_messages: Vec<String>,
    /// Includes the last item even when it contributed no blocks.
    pub last_speaker: Option<Speaker>,
    pub dropped: Vec<Dropped>,
}

pub(crate) fn merge_turns<D: Dialect>(
    chat_ctx: &ChatContext,
    images: &dyn ImageResolver,
    dialect: &D,
) -> Result<Turns<D::Block>, FormatError> {
    let converted = convert_mid_conversation_instructions(
        chat_ctx,
        Role::User,
        DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE,
    );
    let grouping = group_tool_calls(converted.iter().map(Cow::as_ref))?;
    let mut turns = Vec::new();
    let mut system_messages = Vec::new();
    let mut current_speaker = None;
    let mut content = Vec::new();

    for item in grouping.groups.iter().flat_map(|g| g.flatten()) {
        let speaker = match placement(item) {
            Placement::System(text) => {
                system_messages.push(text);
                continue;
            }
            Placement::Turn(Speaker::ToolResults)
                if matches!(D::RESULTS, ResultsTurn::WithUser) =>
            {
                Speaker::User
            }
            Placement::Turn(speaker) => speaker,
        };
        if Some(speaker) != current_speaker {
            if let Some(previous) = current_speaker {
                flush(previous, std::mem::take(&mut content), &mut turns);
            }
            current_speaker = Some(speaker);
        }
        match item {
            GroupItem::Message(m) => push_message_blocks(m, images, dialect, &mut content)?,
            GroupItem::Call(call) => content.push(dialect.call(call)?),
            GroupItem::Output(output) => content.push(dialect.result(output)),
        }
    }
    if let Some(speaker) = current_speaker {
        flush(speaker, content, &mut turns);
    }
    Ok(Turns { turns, system_messages, last_speaker: current_speaker, dropped: grouping.dropped })
}

fn flush<B>(speaker: Speaker, blocks: Vec<B>, turns: &mut Vec<Turn<B>>) {
    if !blocks.is_empty() {
        turns.push(Turn { speaker, blocks });
    }
}

enum Placement {
    System(String),
    Turn(Speaker),
}

fn placement(item: GroupItem<'_>) -> Placement {
    match item {
        GroupItem::Message(m) if m.role == Role::System => {
            match m.raw_text_content().filter(|t| !t.is_empty()) {
                Some(text) => Placement::System(text),
                None => Placement::Turn(Speaker::User),
            }
        }
        GroupItem::Message(m) if m.role == Role::Assistant => Placement::Turn(Speaker::Assistant),
        GroupItem::Message(_) => Placement::Turn(Speaker::User),
        GroupItem::Call(_) => Placement::Turn(Speaker::Assistant),
        GroupItem::Output(_) => Placement::Turn(Speaker::ToolResults),
    }
}

fn push_message_blocks<D: Dialect>(
    message: &ChatMessage,
    images: &dyn ImageResolver,
    dialect: &D,
    content: &mut Vec<D::Block>,
) -> Result<(), FormatError> {
    for part in &message.content {
        match part {
            ChatContent::Image(image) if !D::USER_IMAGES_ONLY || message.role == Role::User => {
                content.push(dialect.image(&resolve(image, images)?)?);
            }
            ChatContent::Text(text) if !text.is_empty() => content.push(dialect.text(text)),
            ChatContent::Audio(_) | ChatContent::Image(_) | ChatContent::Text(_) => {}
        }
    }
    Ok(())
}
