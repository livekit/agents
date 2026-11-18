//! Session-owned provider identities. Application history keeps its original IDs; only the
//! provider-facing names change. Retain bindings across reconnects so late events still correlate.

use std::collections::BTreeMap;

use lk_agents_abi::ChatItem;

use super::{machine::Outgoing, protocol::Incoming};

pub(crate) const MAX_ID_LEN: usize = 32;

pub(crate) fn valid(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_ID_LEN
        && id.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}

/// Stable, ASCII-only fingerprint; it names a pair consistently without truncation collisions.
pub(crate) fn fingerprint(id: &str) -> String {
    let mut a: u64 = 0xcbf2_9ce4_8422_2325;
    let mut b: u64 = 0x9e37_79b9_7f4a_7c15;
    for (i, byte) in id.bytes().enumerate() {
        a = (a ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
        b = (b ^ (u64::from(byte) << (i % 8))).wrapping_mul(0x100_0000_01b3);
    }
    format!("{a:016x}{b:016x}")
}

#[derive(Default)]
struct Names {
    outgoing: BTreeMap<String, String>,
    incoming: BTreeMap<String, String>,
}

impl Names {
    fn outbound(&mut self, logical: &str) -> String {
        if let Some(wire) = self.outgoing.get(logical) {
            return wire.clone();
        }
        // Empty and reserved item identities remain invalid requests, rather than becoming
        // ambiguous with the protocol's root anchor. Their rejection must reach the caller.
        let mut wire = if valid(logical) || logical.is_empty() {
            logical.to_owned()
        } else {
            fingerprint(logical)
        };
        let mut collision = 0_u64;
        while self.incoming.contains_key(&wire) {
            collision += 1;
            wire = fingerprint(&format!("{logical}:{collision}"));
        }
        self.outgoing.insert(logical.to_owned(), wire.clone());
        self.incoming.insert(wire.clone(), logical.to_owned());
        wire
    }

    fn inbound(&mut self, wire: &str) -> String {
        if let Some(logical) = self.incoming.get(wire) {
            return logical.clone();
        }
        // Provider-generated IDs are valid on their originating connection, even when another
        // compatible provider would reject their shape. Preserve them on replies and replay.
        let mut logical = wire.to_owned();
        let mut collision = 0_u64;
        while self.outgoing.contains_key(&logical) {
            collision += 1;
            logical = format!("native_{}_{}", fingerprint(wire), collision);
        }
        self.outgoing.insert(logical.clone(), wire.to_owned());
        self.incoming.insert(wire.to_owned(), logical.clone());
        logical
    }
}

#[derive(Default)]
pub(crate) struct ProviderIds {
    items: Names,
    calls: Names,
}

impl ProviderIds {
    pub(crate) fn outgoing(&mut self, frame: &mut Outgoing) {
        match frame {
            Outgoing::ItemCreate { item, previous_item_id, .. } => {
                self.item(item, true);
                if let Some(previous) = previous_item_id
                    && previous != super::protocol::ROOT_ITEM
                {
                    *previous = self.items.outbound(previous);
                }
            }
            Outgoing::ItemDelete { item_id, .. } | Outgoing::ItemTruncate { item_id, .. } => {
                *item_id = self.items.outbound(item_id);
            }
            _ => {}
        }
    }

    pub(crate) fn incoming(&mut self, event: &mut Incoming) {
        match event {
            Incoming::ItemAdded { item, previous_item_id } => {
                self.item(item, false);
                if let Some(previous) = previous_item_id {
                    *previous = self.items.inbound(previous);
                }
            }
            Incoming::OutputItemDone { item } => self.item(item, false),
            Incoming::SpeechStarted { item_id: Some(id) }
            | Incoming::OutputItemAdded { item_id: id, .. }
            | Incoming::ContentPartAdded { item_id: id, .. }
            | Incoming::ItemDeleted { item_id: id }
            | Incoming::TranscriptionDelta { item_id: id, .. }
            | Incoming::TranscriptionFailed { item_id: id, .. }
            | Incoming::TranscriptionUpdated { item_id: id, .. }
            | Incoming::TextDelta { item_id: id, .. }
            | Incoming::AudioTranscriptDelta { item_id: id, .. }
            | Incoming::AudioDelta { item_id: id, .. } => *id = self.items.inbound(id),
            Incoming::TranscriptionCompleted(t) => t.item_id = self.items.inbound(&t.item_id),
            _ => {}
        }
    }

    fn item(&mut self, item: &mut ChatItem, outgoing: bool) {
        let bind = |names: &mut Names, id: &mut String| {
            *id = if outgoing { names.outbound(id) } else { names.inbound(id) };
        };
        match item {
            ChatItem::Message(m) => bind(&mut self.items, &mut m.id),
            ChatItem::FunctionCall(c) => {
                bind(&mut self.items, &mut c.id);
                bind(&mut self.calls, &mut c.call_id);
            }
            ChatItem::FunctionCallOutput(o) => {
                bind(&mut self.items, &mut o.id);
                bind(&mut self.calls, &mut o.call_id);
            }
            ChatItem::AgentHandoff(_) | ChatItem::AgentConfigUpdate(_) => {}
        }
    }
}

#[cfg(test)]
#[path = "ids_tests.rs"]
mod tests;
