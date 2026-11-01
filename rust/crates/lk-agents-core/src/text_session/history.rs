//! Frame history selection and once-only transfer of delivered dialogue.

use crate::llm::history::ChatContextExt;
use lk_agents_abi::{ChatContext, ChatItem, ChatMessage, CopyOptions, Role};

use super::{ChatHistory, SessionFrame};

pub(super) fn initial(history: &[ChatItem], mode: ChatHistory) -> Vec<ChatItem> {
    match mode {
        ChatHistory::Fresh => Vec::new(),
        ChatHistory::Fork => {
            let calls: std::collections::BTreeSet<_> = history
                .iter()
                .filter_map(|item| match item {
                    ChatItem::FunctionCall(call) => Some(call.call_id.as_str()),
                    _ => None,
                })
                .collect();
            let answered: std::collections::BTreeSet<_> = history
                .iter()
                .filter_map(|item| match item {
                    ChatItem::FunctionCallOutput(output)
                        if calls.contains(output.call_id.as_str()) =>
                    {
                        Some(output.call_id.as_str())
                    }
                    _ => None,
                })
                .collect();
            let complete = history
                .iter()
                .filter(|item| match item {
                    ChatItem::FunctionCall(call) => answered.contains(call.call_id.as_str()),
                    ChatItem::FunctionCallOutput(output) => {
                        answered.contains(output.call_id.as_str())
                    }
                    _ => true,
                })
                .cloned()
                .collect();
            ChatContext::new(complete)
                .copy(&CopyOptions {
                    exclude_instructions: true,
                    exclude_handoff: true,
                    exclude_config_update: true,
                    ..Default::default()
                })
                .items
        }
    }
}

pub(super) fn delivered(frame: &mut SessionFrame, message: ChatMessage) {
    if !matches!(message.role, Role::User | Role::Assistant)
        || frame.starting_history_ids.contains(&message.id)
        || frame.visible_dialogue.iter().any(|m| m.id == message.id)
    {
        return;
    }
    frame.visible_dialogue.push(message);
}

pub(super) fn transfer(child: &mut SessionFrame, parent: &mut SessionFrame) {
    if !child.merge_chat_history {
        return;
    }
    for message in &child.visible_dialogue {
        if child.transferred_message_ids.insert(message.id.clone()) {
            if !parent.history.iter().any(|item| item.id() == message.id) {
                parent.history.push(ChatItem::Message(message.clone()));
            }
            delivered(parent, message.clone());
        }
    }
}
