use lk_agents_abi::{ChatContext, ChatMessage, Role};

/// The ID of the expressive TTS markup-guide message in the chat context. Must not change: it is
/// matched against persisted contexts.
pub const EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID: &str = "lk.expressive.instructions";

pub fn remove_expressive_instructions(chat_ctx: &mut ChatContext) -> usize {
    let before = chat_ctx.items.len();
    chat_ctx.items.retain(|item| item.id() != EXPRESSIVE_INSTRUCTIONS_MESSAGE_ID);
    before - chat_ctx.items.len()
}

/// Display text strips speech markup from assistant messages.
pub trait ChatMessageExt {
    fn text_content(&self) -> Option<String>;
}

impl ChatMessageExt for ChatMessage {
    fn text_content(&self) -> Option<String> {
        self.raw_text_content().map(|text| {
            if self.role == Role::Assistant { super::split_expr_markup(&text).0 } else { text }
        })
    }
}

pub fn chat_message_text_content(parts: Vec<String>, assistant: bool) -> Option<String> {
    let text = lk_agents_abi::chat_message_raw_text_content(parts)?;
    Some(if assistant { super::split_expr_markup(&text).0 } else { text })
}
