//! The acknowledged server conversation, ordered by links and indexed by item identity.

use std::collections::BTreeMap;
use std::sync::Arc;

use lk_agents_abi::{ChatContext, ChatItem};

pub use lk_agents_abi::llm::chat_context::RemoteError;

/// The ordering engine retains values without converting or taking ownership of their contents.
pub trait RemoteValue {
    fn id(&self) -> &str;
    fn call_id(&self) -> Option<&str> {
        None
    }
}

impl RemoteValue for ChatItem {
    fn id(&self) -> &str {
        ChatItem::id(self)
    }
    fn call_id(&self) -> Option<&str> {
        match self {
            ChatItem::FunctionCall(call) => Some(&call.call_id),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct RemoteItem<T> {
    item: T,
    previous: Option<Arc<str>>,
    next: Option<Arc<str>>,
}

/// The native realtime conversation; the generic history also retains host-owned items.
pub type RemoteChatContext = RemoteHistory<ChatItem>;

/// Insertion and acknowledgement never shift or rebuild the retained history.
#[derive(Debug, Clone)]
pub struct RemoteHistory<T> {
    nodes: BTreeMap<Arc<str>, RemoteItem<T>>,
    calls: BTreeMap<String, usize>,
    head: Option<Arc<str>>,
    tail: Option<Arc<str>>,
    revision: u64,
}

pub struct RemoteItems<'a, T = ChatItem> {
    context: &'a RemoteHistory<T>,
    front: Option<&'a str>,
    back: Option<&'a str>,
    remaining: usize,
}

impl<'a, T> Iterator for RemoteItems<'a, T> {
    type Item = &'a T;

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

impl<T> DoubleEndedIterator for RemoteItems<'_, T> {
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

impl<T> ExactSizeIterator for RemoteItems<'_, T> {}

impl<T> Default for RemoteHistory<T> {
    fn default() -> Self {
        Self { nodes: BTreeMap::new(), calls: BTreeMap::new(), head: None, tail: None, revision: 0 }
    }
}

impl RemoteHistory<ChatItem> {
    pub fn to_chat_ctx(&self) -> ChatContext {
        ChatContext::new(self.items().cloned().collect())
    }
}

impl<T: RemoteValue> RemoteHistory<T> {
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

    pub fn items(&self) -> RemoteItems<'_, T> {
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

    pub fn get(&self, item_id: &str) -> Option<&T> {
        self.nodes.get(item_id).map(|node| &node.item)
    }

    pub(crate) fn contains_call(&self, call_id: &str) -> bool {
        self.calls.contains_key(call_id)
    }

    pub fn get_mut(&mut self, item_id: &str) -> Option<&mut T> {
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

    pub fn insert(&mut self, previous_item_id: Option<&str>, item: T) -> Result<(), RemoteError> {
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
        if let Some(call_id) = item.call_id() {
            *self.calls.entry(call_id.to_owned()).or_default() += 1;
        }
        self.nodes.insert(id, RemoteItem { item, previous, next });
        self.revision = self.revision.checked_add(1).expect("remote context revision exhausted");
        Ok(())
    }

    pub fn delete(&mut self, item_id: &str) -> Result<T, RemoteError> {
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
        if let Some(call_id) = node.item.call_id()
            && let Some(count) = self.calls.get_mut(call_id)
        {
            *count -= 1;
            if *count == 0 {
                self.calls.remove(call_id);
            }
        }
        self.revision = self.revision.checked_add(1).expect("remote context revision exhausted");
        Ok(node.item)
    }
}

#[derive(Debug, Clone)]
struct HostItem {
    id: String,
    value: lk_agents_abi::HostRef,
}
impl RemoteValue for HostItem {
    fn id(&self) -> &str {
        &self.id
    }
}

/// The same ordering implementation, retaining a host item rather than a native ChatItem.
pub struct HostRemoteChatContext(RemoteHistory<HostItem>);
impl lk_agents_abi::llm::chat_context::RemoteChatContext for HostRemoteChatContext {
    fn new() -> Self {
        Self(RemoteHistory::new())
    }
    fn tail_id(&self) -> Option<String> {
        self.0.tail_id().map(str::to_owned)
    }
    fn get(&self, item_id: String) -> Option<lk_agents_abi::HostRef> {
        self.0.get(&item_id).map(|item| item.value.clone())
    }
    fn items(&self) -> Vec<lk_agents_abi::HostRef> {
        self.0.items().map(|item| item.value.clone()).collect()
    }
    fn insert(
        &mut self,
        previous_item_id: Option<String>,
        item_id: String,
        item: lk_agents_abi::HostRef,
    ) -> Result<(), RemoteError> {
        self.0.insert(previous_item_id.as_deref(), HostItem { id: item_id, value: item })
    }
    fn delete(&mut self, item_id: String) -> Result<(), RemoteError> {
        self.0.delete(&item_id).map(|_| ())
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
