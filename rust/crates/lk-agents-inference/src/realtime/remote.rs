//! The acknowledged server conversation, ordered by links and indexed by item identity.

use std::collections::BTreeMap;
use std::sync::Arc;

use lk_agents_abi::{ChatContext, ChatItem};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RemoteError {
    #[error("Item with ID {0} already exists.")]
    DuplicateId(String),
    #[error("previous_item_id `{0}` not found")]
    UnknownAnchor(String),
    #[error("item_id `{0}` not found")]
    UnknownItem(String),
}

#[derive(Debug, Clone)]
struct RemoteItem {
    item: ChatItem,
    previous: Option<Arc<str>>,
    next: Option<Arc<str>>,
}

/// Insertion and acknowledgement never shift or rebuild the retained history.
#[derive(Debug, Clone, Default)]
pub struct RemoteChatContext {
    nodes: BTreeMap<Arc<str>, RemoteItem>,
    calls: BTreeMap<String, usize>,
    head: Option<Arc<str>>,
    tail: Option<Arc<str>>,
    revision: u64,
}

pub struct RemoteItems<'a> {
    context: &'a RemoteChatContext,
    front: Option<&'a str>,
    back: Option<&'a str>,
    remaining: usize,
}

impl<'a> Iterator for RemoteItems<'a> {
    type Item = &'a ChatItem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let node = &self.context.nodes[self.front?];
        self.front = node.next.as_deref();
        self.remaining -= 1;
        Some(&node.item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl DoubleEndedIterator for RemoteItems<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let node = &self.context.nodes[self.back?];
        self.back = node.previous.as_deref();
        self.remaining -= 1;
        Some(&node.item)
    }
}

impl ExactSizeIterator for RemoteItems<'_> {}

impl RemoteChatContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn retained_units(&self) -> usize {
        self.nodes.len() + self.calls.len() + 1
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn to_chat_ctx(&self) -> ChatContext {
        ChatContext::new(self.items().cloned().collect())
    }

    pub fn items(&self) -> RemoteItems<'_> {
        RemoteItems {
            context: self,
            front: self.head.as_deref(),
            back: self.tail.as_deref(),
            remaining: self.nodes.len(),
        }
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    pub fn get(&self, item_id: &str) -> Option<&ChatItem> {
        self.nodes.get(item_id).map(|node| &node.item)
    }

    pub(crate) fn contains_call(&self, call_id: &str) -> bool {
        self.calls.contains_key(call_id)
    }

    pub fn get_mut(&mut self, item_id: &str) -> Option<&mut ChatItem> {
        let node = self.nodes.get_mut(item_id)?;
        Some(&mut node.item)
    }

    pub fn tail_id(&self) -> Option<&str> {
        self.tail.as_deref()
    }

    pub(crate) fn first_id(&self) -> Option<&str> {
        self.head.as_deref()
    }

    pub(crate) fn previous_id(&self, id: &str) -> Option<&str> {
        self.nodes.get(id)?.previous.as_deref()
    }

    pub(crate) fn next_id(&self, id: &str) -> Option<&str> {
        self.nodes.get(id)?.next.as_deref()
    }

    pub fn insert(
        &mut self,
        previous_item_id: Option<&str>,
        item: ChatItem,
    ) -> Result<(), RemoteError> {
        if self.nodes.contains_key(item.id()) {
            return Err(RemoteError::DuplicateId(item.id().to_owned()));
        }
        let previous = match previous_item_id {
            None => None,
            Some(id) => Some(
                self.nodes
                    .get_key_value(id)
                    .ok_or_else(|| RemoteError::UnknownAnchor(id.to_owned()))?
                    .0
                    .clone(),
            ),
        };
        let next =
            previous.as_deref().map_or_else(|| self.head.clone(), |id| self.nodes[id].next.clone());
        let id: Arc<str> = Arc::from(item.id());
        if let Some(previous) = previous.as_deref() {
            self.nodes.get_mut(previous).unwrap().next = Some(id.clone());
        } else {
            self.head = Some(id.clone());
        }
        if let Some(next) = next.as_deref() {
            self.nodes.get_mut(next).unwrap().previous = Some(id.clone());
        } else {
            self.tail = Some(id.clone());
        }
        if let ChatItem::FunctionCall(call) = &item {
            *self.calls.entry(call.call_id.clone()).or_default() += 1;
        }
        self.nodes.insert(id, RemoteItem { item, previous, next });
        self.revision = self.revision.checked_add(1).expect("remote context revision exhausted");
        Ok(())
    }

    pub fn delete(&mut self, item_id: &str) -> Result<ChatItem, RemoteError> {
        let node = self
            .nodes
            .remove(item_id)
            .ok_or_else(|| RemoteError::UnknownItem(item_id.to_owned()))?;
        if let Some(previous) = node.previous.as_deref() {
            self.nodes.get_mut(previous).unwrap().next = node.next.clone();
        } else {
            self.head = node.next.clone();
        }
        if let Some(next) = node.next.as_deref() {
            self.nodes.get_mut(next).unwrap().previous = node.previous;
        } else {
            self.tail = node.previous;
        }
        if let ChatItem::FunctionCall(call) = &node.item
            && let Some(count) = self.calls.get_mut(&call.call_id)
        {
            *count -= 1;
            if *count == 0 {
                self.calls.remove(&call.call_id);
            }
        }
        self.revision = self.revision.checked_add(1).expect("remote context revision exhausted");
        Ok(node.item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::testfix::msg;

    fn ids(ctx: &RemoteChatContext) -> Vec<&str> {
        ctx.items().map(ChatItem::id).collect()
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
