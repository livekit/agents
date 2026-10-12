//! The expressive-off chat edits (`agent_activity.py:3195`). Both mutate the stored history.

use lk_agents_abi::{ChatContent, ChatContext, ChatItem, Role};

use crate::tts::markup::transcript::strip_all_markup;

/// The ID of the expressive TTS markup-guide message in the chat context. Must not change: it is
/// matched against persisted contexts.
pub const EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID: &str = "lk.expressive.instructions";

pub fn remove_expressive_instructions(chat_ctx: &mut ChatContext) -> usize {
    let before = chat_ctx.items.len();
    chat_ctx.items.retain(|item| item.id() != EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID);
    before - chat_ctx.items.len()
}

pub fn positions_of(item_ids: impl IntoIterator<Item = impl AsRef<str>>, id: &str) -> Vec<usize> {
    item_ids
        .into_iter()
        .enumerate()
        .filter(|(_, item_id)| item_id.as_ref() == id)
        .map(|(index, _)| index)
        .collect()
}

/// Left in history, tags few-shot the LLM into markup nothing converts.
pub fn strip_assistant_markup(chat_ctx: &mut ChatContext) -> usize {
    let mut changed = 0;
    for item in &mut chat_ctx.items {
        let ChatItem::Message(msg) = item else { continue };
        if msg.role != Role::Assistant {
            continue;
        }
        // Markup is XML-only here, so a message with no `<` is skipped.
        if !msg.content.iter().any(|c| matches!(c, ChatContent::Text(t) if t.contains('<'))) {
            continue;
        }
        for part in &mut msg.content {
            if let ChatContent::Text(t) = part {
                *t = strip_all_markup(t);
            }
        }
        changed += 1;
    }
    changed
}
