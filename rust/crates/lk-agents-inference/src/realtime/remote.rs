//! `llm/remote_chat_context.py` -- the mirror of what the server believes, diffed against the
//! agent's context to produce the edit script. A `Vec` rather than the reference's linked list:
//! it is read whole on every update. The id map, the `previous_item_id: None` => head convention
//! and both `ValueError` cases are preserved.

use std::collections::HashMap;

use lk_agents_abi::{ChatContext, ChatItem};

/// The two `ValueError`s the reference raises; callers log them.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RemoteError {
    #[error("Item with ID {0} already exists.")]
    DuplicateId(String),
    #[error("previous_item_id `{0}` not found")]
    UnknownAnchor(String),
    #[error("item_id `{0}` not found")]
    UnknownItem(String),
}

/// `RemoteChatContext`.
#[derive(Debug, Clone, Default)]
pub struct RemoteChatContext {
    items: Vec<ChatItem>,
    /// id -> index, rebuilt on every structural change.
    index: HashMap<String, usize>,
}

impl RemoteChatContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn to_chat_ctx(&self) -> ChatContext {
        ChatContext::new(self.items.clone())
    }

    pub fn items(&self) -> &[ChatItem] {
        &self.items
    }

    pub fn get(&self, item_id: &str) -> Option<&ChatItem> {
        self.index.get(item_id).map(|&i| &self.items[i])
    }

    pub fn get_mut(&mut self, item_id: &str) -> Option<&mut ChatItem> {
        let i = *self.index.get(item_id)?;
        Some(&mut self.items[i])
    }

    /// `tail_id` -- the last item's id. The fallback anchor when `conversation.item.added` names a
    /// deleted item.
    pub fn tail_id(&self) -> Option<&str> {
        self.items.last().map(ChatItem::id)
    }

    /// `insert(previous_item_id, message)`. `None` inserts at the head (the wire's `"root"`).
    pub fn insert(
        &mut self,
        previous_item_id: Option<&str>,
        item: ChatItem,
    ) -> Result<(), RemoteError> {
        let id = item.id().to_owned();
        if self.index.contains_key(&id) {
            return Err(RemoteError::DuplicateId(id));
        }

        let at = match previous_item_id {
            None => 0,
            Some(prev) => match self.index.get(prev) {
                Some(&i) => i + 1,
                None => return Err(RemoteError::UnknownAnchor(prev.to_owned())),
            },
        };

        self.items.insert(at, item);
        self.reindex_from(at);
        Ok(())
    }

    pub fn delete(&mut self, item_id: &str) -> Result<ChatItem, RemoteError> {
        let Some(&at) = self.index.get(item_id) else {
            return Err(RemoteError::UnknownItem(item_id.to_owned()));
        };
        let removed = self.items.remove(at);
        self.index.remove(item_id);
        self.reindex_from(at);
        Ok(removed)
    }

    fn reindex_from(&mut self, from: usize) {
        for (i, item) in self.items.iter().enumerate().skip(from) {
            self.index.insert(item.id().to_owned(), i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::testfix::msg;

    fn ids(ctx: &RemoteChatContext) -> Vec<&str> {
        ctx.items().iter().map(ChatItem::id).collect()
    }

    // Catches: a `delete` that forgets to repair the index map.
    #[test]
    fn deleting_from_the_middle_keeps_later_anchors_correct() {
        let mut c = RemoteChatContext::new();
        for (id, t) in [("a", "1"), ("b", "2"), ("c", "3")] {
            let prev = c.tail_id().map(str::to_owned);
            c.insert(prev.as_deref(), msg(id, t)).unwrap();
        }
        assert_eq!(ids(&c), ["a", "b", "c"]);

        c.delete("b").unwrap();
        assert_eq!(ids(&c), ["a", "c"]);
        // "c" must still be findable at its NEW index; anchoring to it must land after it.
        c.insert(Some("c"), msg("d", "4")).unwrap();
        assert_eq!(ids(&c), ["a", "c", "d"]);
        assert_eq!(c.tail_id(), Some("d"));
    }

    // Catches: silently accepting a duplicate id.
    #[test]
    fn a_duplicate_id_is_rejected_rather_than_inserted_twice() {
        let mut c = RemoteChatContext::new();
        c.insert(None, msg("a", "1")).unwrap();
        assert_eq!(c.insert(None, msg("a", "2")), Err(RemoteError::DuplicateId("a".into())));
        assert_eq!(ids(&c), ["a"]);
        // the surviving item is the original, not the rejected one
        assert_eq!(
            c.get("a").unwrap().as_message().unwrap().raw_text_content().as_deref(),
            Some("1")
        );
    }

    // Catches: an unknown anchor treated as "at the head".
    #[test]
    fn an_unknown_anchor_is_an_error_not_a_head_insert() {
        let mut c = RemoteChatContext::new();
        c.insert(None, msg("a", "1")).unwrap();
        assert_eq!(
            c.insert(Some("nope"), msg("b", "2")),
            Err(RemoteError::UnknownAnchor("nope".into()))
        );
        assert_eq!(ids(&c), ["a"]);
    }

    // Catches: `delete` of an absent id returning Ok.
    #[test]
    fn deleting_an_absent_id_is_an_error() {
        let mut c = RemoteChatContext::new();
        assert_eq!(c.delete("ghost"), Err(RemoteError::UnknownItem("ghost".into())));
    }
}
