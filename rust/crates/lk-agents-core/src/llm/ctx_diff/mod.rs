//! Minimal edit script between two `ChatContext`s (`llm/utils.py`): a realtime server can only be
//! told "insert this after that" or "delete this".

use lk_agents_abi::{ChatContext, ChatItem};
use std::collections::BTreeMap;

pub use lk_agents_abi::llm::chat_context::Anchored;

mod incremental;
pub use incremental::ChatContextDiff;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DiffOps {
    pub to_remove: Vec<String>,
    pub to_create: Vec<Anchored>,
    /// Patched in place, keeping the item's server-side position.
    pub to_update: Vec<Anchored>,
}

/// `compute_chat_ctx_diff` (`llm/utils.py:120`) over the host's items.
pub struct ChatDiffs;

impl lk_agents_abi::llm::chat_context::ChatContextDiffs for ChatDiffs {
    fn compute_chat_ctx_diff(
        old: Vec<ChatItem>,
        new: Vec<ChatItem>,
    ) -> lk_agents_abi::llm::chat_context::DiffOps {
        use lk_agents_abi::llm::chat_context::AnchoredItem;
        let anchored = |items: Vec<Anchored>| {
            items
                .into_iter()
                .map(|(previous_item_id, item_id)| AnchoredItem { previous_item_id, item_id })
                .collect()
        };
        let ops = compute_chat_ctx_diff(&ChatContext::new(old), &ChatContext::new(new));
        lk_agents_abi::llm::chat_context::DiffOps {
            to_remove: ops.to_remove,
            to_create: anchored(ops.to_create),
            to_update: anchored(ops.to_update),
        }
    }
}

pub fn compute_chat_ctx_diff(old: &ChatContext, new: &ChatContext) -> DiffOps {
    let mut old_by_id: BTreeMap<&str, &ChatItem> =
        old.items.iter().map(|item| (item.id(), item)).collect();
    if old_by_id.len() == old.len()
        && old.items.iter().zip(&new.items).all(|(before, after)| before.id() == after.id())
    {
        // Validate appended ids with the existing index. Repeated ids use the general edit
        // engine's historical set semantics.
        let unique = new
            .items
            .iter()
            .skip(old.len())
            .all(|item| old_by_id.insert(item.id(), item).is_none());
        if unique {
            return prefix_diff(old, new);
        }
    }
    drop(old_by_id);
    let mut diff = ChatContextDiff::default();
    while !diff.advance(old, new, 1024) {}
    diff.take_operations()
}

/// Appending or trimming a context whose ids already line up needs no retained subsequence or
/// owned id index. Duplicate host ids keep the general path's historical set semantics.
fn prefix_diff(old: &ChatContext, new: &ChatContext) -> DiffOps {
    let shared = old.len().min(new.len());
    let mut diff = DiffOps::default();
    for (index, (before, after)) in old.items.iter().zip(&new.items).enumerate() {
        if item_changed(before, after) {
            diff.to_update.push((
                index.checked_sub(1).map(|prev| new.items[prev].id().to_owned()),
                after.id().to_owned(),
            ));
        }
    }
    diff.to_remove.extend(old.items[shared..].iter().map(|item| item.id().to_owned()));
    diff.to_create.extend(new.items.iter().enumerate().skip(shared).map(|(index, item)| {
        (index.checked_sub(1).map(|prev| new.items[prev].id().to_owned()), item.id().to_owned())
    }));
    diff
}

/// A message's text or a call output's result changed under the same id (an expert's outcome
/// replaces its call's pending output).
fn item_changed(old: &ChatItem, new: &ChatItem) -> bool {
    use lk_agents_abi::ChatContent;
    if let (ChatItem::FunctionCallOutput(old), ChatItem::FunctionCallOutput(new)) = (old, new) {
        return old.output != new.output || old.is_error != new.is_error;
    }
    let (Some(old), Some(new)) = (old.as_message(), new.as_message()) else {
        return false;
    };
    fn text(content: &ChatContent) -> Option<&str> {
        match content {
            ChatContent::Text(text) => Some(text.as_str()),
            _ => None,
        }
    }
    // Equal parts need no joined String; differing boundaries may still join to the same text.
    if old.content.iter().filter_map(text).eq(new.content.iter().filter_map(text)) {
        return false;
    }
    old.raw_text_content() != new.raw_text_content()
}

#[cfg(test)]
#[path = "ctx_diff_tests.rs"]
mod tests;
