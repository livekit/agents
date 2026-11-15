use std::collections::BTreeMap;

use lk_agents_abi::llm::chat_context::protocol::encode_chat_item;
use lk_agents_abi::{ChatItem, ChatMessage, Role};
use lk_agents_voice::agent::{Source, Visibility};
use lk_agents_voice::events::SessionEvent;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Observed, Recording, StorageOperation, StorageStep},
};

/// A save's receipt protects the accepted item revisions that preceded its request.
/// Read database payloads, not the arguments the runtime submitted to the store.
pub struct StoragePreservesItemPayloads;

impl Rule for StoragePreservesItemPayloads {
    const CONTRACT: Contract = Contract {
        id: "storage_preserves_item_payloads",
        guarantee: "A successfully settled conversation preserves accepted item payloads through its save request, including interrupted assistant prefixes and flags. Reloads replace the expected database view; later unsaved revisions are not treated as durable.",
        examples: &[a_delayed_reload_cannot_undo_foreign_compaction, a_reload_does_not_accept_private_model_context, cut_saved, unsaved_revision, foreign_reload, unseen_foreign_item_kept],
        counterexamples: &[a_delayed_reload_cannot_hide_a_lost_foreign_summary, generated_tail_saved, interruption_flag_lost, accepted_item_lost, unseen_foreign_item_lost],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        let through = durable_through(r);
        let Some(through) = through else { return };
        // Reloads also carry private model-context items, which were never accepted into
        // conversation history. Only public acceptance or an observed foreign commit makes
        // an identity part of this durability contract.
        let mut accepted_ids = std::collections::BTreeSet::new();
        for (_, event) in host.events() {
            if let SessionEvent::ConversationItemAdded { item, visibility, .. } = event
                && visibility.history()
            {
                accepted_ids.insert(item.id());
            }
        }
        for (_, step) in host.storage() {
            if let StorageStep::ForeignCommitted { items } = step {
                accepted_ids.extend(items.iter().map(ChatItem::id));
            }
        }
        let mut expected = BTreeMap::new();
        let mut retired = std::collections::BTreeSet::new();
        let mut foreign_current = std::collections::BTreeSet::new();
        let mut changes = BTreeMap::new();
        for (at, event) in host.events() {
            changes.insert(at, Ok(event));
        }
        for (at, step) in host.storage() {
            if let StorageStep::ForeignCommitted { items } = step {
                changes.insert(*at, Err(items));
            }
        }
        for (at, change) in changes {
            if at > through {
                break;
            }
            if let Err(items) = change {
                // A foreign compaction is authoritative. A delayed reload of an older read
                // cannot accept its removed rows again; only a new local item revision can.
                let incoming: std::collections::BTreeSet<_> = items.iter().map(ChatItem::id).collect();
                retired.extend(expected.keys().copied().filter(|id| !incoming.contains(id)));
                retired.retain(|id| !incoming.contains(id));
                foreign_current = incoming;
                expected.clear();
                for item in items {
                    expected.insert(item.id(), (at, item));
                }
                continue;
            }
            let event = change.expect("foreign changes were handled");
            match event {
                SessionEvent::HistoryReloaded { items } => {
                    // An older read may omit a row the external writer just added. Keep
                    // that authoritative row until a later save or foreign baseline changes it.
                    expected.retain(|id, _| foreign_current.contains(id) && !items.iter().any(|item| item.id() == *id));
                    for item in items.iter().filter(|item| accepted_ids.contains(item.id()) && !retired.contains(item.id())) {
                        expected.insert(item.id(), (at, item));
                    }
                }
                SessionEvent::ConversationItemAdded { item, visibility, .. } if visibility.history() => {
                    retired.remove(item.id());
                    foreign_current.remove(item.id());
                    expected.insert(item.id(), (at, item.as_ref()));
                }
                _ => {}
            }
        }
        for (_, step) in host.storage() {
            let StorageStep::Saved { items } = step else { continue };
            for (id, (accepted, item)) in &expected {
                if *accepted > through || !durable(item) {
                    continue;
                }
                let saved = items.iter().find(|saved| saved.id() == *id);
                check.expect_with(Subject::Session, saved.is_some_and(|saved| {
                        payload(saved) == payload(item)
                }), || format!("item {id} was accepted before its save but read back as {saved:?}; expected {item:?}"));
            }
        }
    }
}

pub(super) fn durable_through(r: &Recording) -> Option<Observed> {
    let host = r.host_view();
    let mut pending = BTreeMap::new();
    let mut conversations = BTreeMap::new();
    let mut through = None::<Observed>;
    for (at, step) in host.storage() {
        match step {
            StorageStep::ConversationRequested { progress } => {
                conversations.insert(progress.sequence, *at);
            }
            StorageStep::ConversationAnswered { sequence, result } => {
                if let Some(requested) = conversations.remove(sequence)
                    && result.is_ok()
                {
                    through = Some(through.map_or(requested, |held| held.max(requested)));
                }
            }
            StorageStep::ForeignCommitted { .. } => {
                through = Some(through.map_or(*at, |held| held.max(*at)));
            }
            StorageStep::Requested { speech, operation } if matches!(operation, StorageOperation::Settle | StorageOperation::Checkpoint) => {
                pending.insert((*speech, *operation), *at);
            }
            StorageStep::Answered { speech, operation, result, .. } if matches!(operation, StorageOperation::Settle | StorageOperation::Checkpoint) => {
                if let Some(requested) = pending.remove(&(*speech, *operation))
                    && result.is_ok()
                {
                    through = Some(through.map_or(requested, |held| held.max(requested)));
                }
            }
            _ => {}
        }
    }
    through
}

fn payload(item: &ChatItem) -> impl PartialEq {
    let mut item = item.clone();
    if let ChatItem::Message(message) = &mut item {
        // This contract checks conversation content, identity and interruption accounting.
        // Telemetry uses a lossy protocol projection and JSON floating-point storage.
        message.metrics = Default::default();
    }
    encode_chat_item(&item)
}

fn durable(item: &ChatItem) -> bool {
    match item {
        ChatItem::Message(m) => !matches!(m.role, Role::System | Role::Developer),
        ChatItem::AgentConfigUpdate(_) => false,
        _ => true,
    }
}

fn message(text: &str, interrupted: bool) -> ChatItem {
    let mut message = ChatMessage::text("reply", Role::Assistant, text, std::time::UNIX_EPOCH);
    message.interrupted = interrupted;
    ChatItem::Message(message)
}

fn accepted(item: ChatItem) -> SessionEvent {
    SessionEvent::ConversationItemAdded {
        item: Box::new(item), agent: "desk".into(), source: Source::HostCtx,
        visibility: Visibility::Both,
    }
}

fn settled(a: Authored) -> Authored {
    a.storage(StorageStep::Requested { speech: 1, operation: StorageOperation::Settle })
        .storage(StorageStep::Answered {
            speech: 1, operation: StorageOperation::Settle, result: Ok(()), delivered: true,
        })
}

fn cut_saved() -> Recording {
    settled(Authored::new().session(accepted(message("heard", true))))
        .storage(StorageStep::Saved { items: vec![message("heard", true)] }).done()
}

fn unsaved_revision() -> Recording {
    let anchor = ChatItem::Message(ChatMessage::text("anchor", Role::User, "hello", std::time::UNIX_EPOCH));
    settled(Authored::new().session(accepted(anchor.clone())).session(accepted(message("heard", true))))
        .session(accepted(message("later", true)))
        .storage(StorageStep::Saved { items: vec![anchor, message("heard", true)] })
        .done()
}

fn foreign_reload() -> Recording {
    settled(Authored::new().session(accepted(message("heard", true)))
        .session(SessionEvent::HistoryReloaded { items: vec![message("foreign edit", false)] }))
        .storage(StorageStep::Saved { items: vec![message("foreign edit", false)] }).done()
}

fn generated_tail_saved() -> Recording {
    settled(Authored::new().session(accepted(message("heard", true))))
        .storage(StorageStep::Saved { items: vec![message("heard and unheard", true)] }).done()
}

fn interruption_flag_lost() -> Recording {
    settled(Authored::new().session(accepted(message("heard", true))))
        .storage(StorageStep::Saved { items: vec![message("heard", false)] }).done()
}

fn accepted_item_lost() -> Recording {
    settled(Authored::new().session(accepted(message("heard", true))))
        .storage(StorageStep::Saved { items: vec![] }).done()
}

fn foreign_and_local() -> Authored {
    Authored::new()
        .storage(StorageStep::ForeignCommitted { items: vec![message("foreign", false)] })
        .session(accepted(ChatItem::Message(ChatMessage::text("local", Role::User, "voice", std::time::UNIX_EPOCH))))
}

fn unseen_foreign_item_kept() -> Recording {
    settled(foreign_and_local()).storage(StorageStep::Saved {
        items: vec![message("foreign", false), ChatItem::Message(ChatMessage::text("local", Role::User, "voice", std::time::UNIX_EPOCH))],
    }).done()
}

fn unseen_foreign_item_lost() -> Recording {
    settled(foreign_and_local()).storage(StorageStep::Saved {
        items: vec![ChatItem::Message(ChatMessage::text("local", Role::User, "voice", std::time::UNIX_EPOCH))],
    }).done()
}

fn a_reload_does_not_accept_private_model_context() -> Recording {
    let private = ChatItem::Message(ChatMessage::text("private", Role::User, "prompt only", std::time::UNIX_EPOCH));
    settled(Authored::new().session(accepted(message("heard", true)))
        .session(SessionEvent::HistoryReloaded { items: vec![private, message("heard", true)] }))
        .storage(StorageStep::Saved { items: vec![message("heard", true)] }).done()
}

fn summary(id: &str) -> ChatItem {
    ChatItem::Message(ChatMessage::text(id, Role::Assistant, id, std::time::UNIX_EPOCH))
}

fn delayed_compaction() -> Authored {
    settled(Authored::new()
        .storage(StorageStep::ForeignCommitted { items: vec![summary("old")] })
        .storage(StorageStep::ForeignCommitted { items: vec![summary("new")] })
        .session(SessionEvent::HistoryReloaded { items: vec![summary("old")] }))
}

fn a_delayed_reload_cannot_undo_foreign_compaction() -> Recording {
    delayed_compaction().storage(StorageStep::Saved { items: vec![summary("new")] }).done()
}

fn a_delayed_reload_cannot_hide_a_lost_foreign_summary() -> Recording {
    delayed_compaction().storage(StorageStep::Saved { items: vec![] }).done()
}
