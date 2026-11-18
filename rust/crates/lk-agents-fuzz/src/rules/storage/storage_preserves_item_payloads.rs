use std::collections::BTreeMap;

use lk_agents_abi::llm::chat_context::protocol::encode_chat_item;
use lk_agents_abi::{ChatItem, ChatMessage, Role};
use lk_agents_voice::agent::{Source, Visibility};
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::{CoreStep, GenerationStep, Step};

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
        examples: &[an_unread_foreign_edit_survives_a_stale_private_edit, an_unread_foreign_edit_survives_a_stale_public_edit, an_unchanged_snapshot_does_not_supersede_a_foreign_edit, a_changed_snapshot_can_supersede_a_foreign_edit, an_explicit_private_edit_revises_a_stored_item, a_later_foreign_edit_supersedes_an_earlier_private_edit, a_foreign_write_does_not_save_unacknowledged_local_edits, an_unrelated_foreign_append_keeps_a_new_local_revision, a_late_host_callback_keeps_the_snapshot_causal_order, an_earlier_checkpoint_receipt_covers_only_its_snapshot, a_later_accepted_revision_can_already_be_saved, a_queued_snapshot_does_not_include_later_input, a_delayed_reload_cannot_undo_foreign_compaction, a_reload_does_not_accept_private_model_context, cut_saved, unsaved_revision, foreign_reload, unseen_foreign_item_kept],
        counterexamples: &[a_stale_private_edit_cannot_erase_an_unread_foreign_edit, a_stale_public_edit_cannot_erase_an_unread_foreign_edit, an_unchanged_snapshot_cannot_excuse_losing_a_foreign_edit, a_changed_snapshot_cannot_excuse_losing_the_new_revision, an_explicit_private_edit_cannot_be_lost, a_private_edit_cannot_replace_a_later_foreign_edit, a_foreign_write_still_protects_its_own_addition, an_older_unsaved_local_edit_cannot_replace_a_foreign_edit, an_unrelated_foreign_append_cannot_excuse_losing_a_local_revision, a_late_host_callback_does_not_excuse_the_old_payload, a_queued_snapshot_still_protects_its_captured_input, a_delayed_reload_cannot_hide_a_lost_foreign_summary, generated_tail_saved, interruption_flag_lost, accepted_item_lost, unseen_foreign_item_lost],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        let through = durable_through(r);
        let events = history_events(r);
        let contexts: Vec<_> = r
            .internal_view()
            .steps()
            .iter()
            .filter_map(|(at, step)| match step {
                Step::Generation(GenerationStep::ChatCtxUpdated { history }) => Some((*at, history)),
                _ => None,
            })
            .collect();
        // An explicit context update publishes its resulting history next. Associate that
        // snapshot with the actual baseline captured at acceptance, including in authored cases.
        let context_snapshots: BTreeMap<_, _> = contexts.iter().filter_map(|(at, history)| {
            let (published, event) = events.iter().find(|(published, _)| published > at)?;
            matches!(event, SessionEvent::HistoryReloaded { .. }).then_some((*published, *history))
        }).collect();
        let context_edits: Vec<_> = contexts.into_iter()
            .flat_map(|(at, history)| history.items.iter().map(move |item| (at, item, &history.stored)))
            .collect();
        // A reload alone does not establish acceptance. Public acceptance or an observed
        // foreign commit brings an identity into this durability contract; subsequent
        // explicit private-context edits can revise it without creating human dialogue.
        let mut accepted_ids = std::collections::BTreeSet::new();
        for (_, event) in events.iter().copied() {
            if let SessionEvent::ConversationItemAdded { item, visibility, .. } = event
                && visibility.history()
            {
                accepted_ids.insert(item.id());
            }
        }
        for (_, step) in host.storage() {
            if let StorageStep::ForeignCommitted { items, .. } = step {
                accepted_ids.extend(items.iter().map(ChatItem::id));
            }
        }
        let mut expected = BTreeMap::new();
        let mut observed_history = BTreeMap::<&str, &ChatItem>::new();
        let mut retired = std::collections::BTreeSet::new();
        let mut foreign_current = std::collections::BTreeSet::new();
        let mut changes = BTreeMap::new();
        for (at, event) in events.iter().copied() {
            changes.insert(at, Ok(event));
        }
        for (at, step) in host.storage() {
            if let StorageStep::ForeignCommitted { before, items } = step {
                changes.insert(*at, Err((before, items)));
            }
        }
        for (at, change) in changes {
            if let Err((before, items)) = change {
                // Observe that writer's actual edits. Its unchanged stored rows do not replace
                // newer local revisions that have not reached storage yet; omissions only
                // retire rows that writer had actually read.
                let incoming: std::collections::BTreeSet<_> = items.iter().map(ChatItem::id).collect();
                let previous: BTreeMap<_, _> = before.iter().map(|item| (item.id(), item)).collect();
                let removed: std::collections::BTreeSet<_> = previous.keys().copied().filter(|id| !incoming.contains(id)).collect();
                retired.extend(removed.iter().copied());
                expected.retain(|id, _| !removed.contains(id));
                foreign_current.retain(|id| incoming.contains(id));
                for item in items {
                    if previous.get(item.id()).is_some_and(|before| payload(before) == payload(item)) {
                        continue;
                    }
                    retired.remove(item.id());
                    foreign_current.insert(item.id());
                    expected.insert(item.id(), (at, item));
                }
                continue;
            }
            if !through.is_some_and(|through| at <= through) {
                continue;
            }
            let event = change.expect("foreign changes were handled");
            match event {
                SessionEvent::HistoryReloaded { items } => {
                    // An older read may omit a row the external writer just added. Keep
                    // that authoritative row until a later save or foreign baseline changes it.
                    expected.retain(|id, _| foreign_current.contains(id) || items.iter().any(|item| item.id() == *id));
                    for item in items.iter().filter(|item| accepted_ids.contains(item.id()) && !retired.contains(item.id())) {
                        // Publishing another row or position also copies unchanged members.
                        // That copy is not a fresh edit superseding an unseen foreign write.
                        let unobserved_conflict = foreign_current.contains(item.id())
                            && context_snapshots.get(&at).is_some_and(|history| {
                                expected.get(item.id()).is_some_and(|(_, held)| {
                                    !history.stored.iter().any(|read| read.id() == item.id() && payload(read) == payload(held))
                                })
                            });
                        if !unobserved_conflict && observed_history.get(item.id()).is_none_or(|before| payload(before) != payload(item)) {
                            expected.insert(item.id(), (at, item));
                            foreign_current.remove(item.id());
                        }
                    }
                    observed_history = items.iter().map(|item| (item.id(), item)).collect();
                }
                SessionEvent::ConversationItemAdded { item, visibility, .. } if visibility.history() => {
                    retired.remove(item.id());
                    foreign_current.remove(item.id());
                    expected.insert(item.id(), (at, item.as_ref()));
                    observed_history.insert(item.id(), item.as_ref());
                }
                _ => {}
            }
        }
        // An explicit private-context edit can revise a previously stored identity without
        // publishing it as human dialogue. Read the native acceptance, never write arguments;
        // a later observed foreign/public revision still wins by causal order.
        let mut later = BTreeMap::<_, Vec<_>>::new();
        for (at, item, stored) in context_edits {
            let Some((accepted, held)) = expected.get_mut(item.id()) else { continue };
            if at <= *accepted {
                continue;
            }
            // Global test order is not the owner's read order. An edit based on an older
            // snapshot cannot overwrite a conflicting foreign revision it has not observed.
            if foreign_current.contains(item.id())
                && !stored.iter().any(|read| read.id() == item.id() && payload(read) == payload(held))
            {
                continue;
            }
            foreign_current.remove(item.id());
            if through.is_some_and(|through| at <= through) {
                *accepted = at;
                *held = item;
            } else {
                later.entry(item.id()).or_default().push((at, item));
            }
        }
        for (at, event) in events.iter().copied() {
            if through.is_none_or(|through| at > through)
                && let SessionEvent::ConversationItemAdded { item, visibility, .. } = event
                && visibility.history()
            {
                later.entry(item.id()).or_default().push((at, item.as_ref()));
            }
        }
        for (_, step) in host.storage() {
            let StorageStep::Saved { items } = step else { continue };
            for (id, (accepted, item)) in &expected {
                if !durable(item) {
                    continue;
                }
                let saved = items.iter().find(|saved| saved.id() == *id);
                let preserved = saved.is_some_and(|saved| {
                    payload(saved) == payload(item)
                        || later.get(id).is_some_and(|versions| {
                            versions.iter().any(|(at, later)| *at > *accepted && payload(saved) == payload(later))
                        })
                });
                check.expect_with(Subject::Session, preserved, || {
                    format!("item {id} was accepted before its save but read back as {saved:?}; expected {item:?}")
                });
            }
        }
    }
}

/// History revisions are selected before their host callbacks run. Compare captured snapshots
/// with that causal order, even when the callback arrives after the snapshot's preparation.
pub(super) fn history_events(r: &Recording) -> Vec<(Observed, &SessionEvent)> {
    let events: Vec<_> = r
        .internal_view()
        .steps()
        .iter()
        .filter_map(|(at, step)| match step {
            Step::Emitted(event) => Some((*at, event)),
            _ => None,
        })
        .collect();
    if events.is_empty() { r.host_view().events() } else { events }
}

pub(super) fn durable_through(r: &Recording) -> Option<Observed> {
    let host = r.host_view();
    let prepared: BTreeMap<_, _> = r
        .internal_view()
        .steps()
        .iter()
        .filter_map(|(at, step)| match step {
            Step::Core(CoreStep::ConversationCheckpointPrepared { sequence }) => {
                Some((*sequence, *at))
            }
            _ => None,
        })
        .collect();
    let mut submitted = BTreeMap::new();
    let mut checkpoints = BTreeMap::new();
    let mut pending = BTreeMap::new();
    let mut conversations = BTreeMap::new();
    let mut through = None::<Observed>;
    for (at, step) in host.storage() {
        match step {
            StorageStep::ConversationRequested { progress } => {
                // A queued snapshot was captured before its I/O task starts. Its receipt
                // cannot make input accepted in that intervening time durable.
                let captured = prepared.get(&progress.sequence).copied().unwrap_or(*at);
                conversations.insert(progress.sequence, captured);
            }
            StorageStep::ConversationAnswered { sequence, result } => {
                if let Some(requested) = conversations.remove(sequence)
                    && result.is_ok()
                {
                    through = Some(through.map_or(requested, |held| held.max(requested)));
                }
            }
            StorageStep::HistorySubmitted { speech, sequence, .. } => {
                let captured = prepared.get(sequence).copied().unwrap_or(*at);
                submitted.insert(*speech, captured);
                checkpoints.insert(*sequence, captured);
            }
            // A speech settles once; its receipt carries no sequence.
            StorageStep::Requested { speech, operation: StorageOperation::Settle } => {
                pending.insert(*speech, submitted.get(speech).copied().unwrap_or(*at));
            }
            StorageStep::Answered { speech, operation: StorageOperation::Settle, result, .. } => {
                if let Some(requested) = pending.remove(speech)
                    && result.is_ok()
                {
                    through = Some(through.map_or(requested, |held| held.max(requested)));
                }
            }
            // One speech can have several checkpoints in flight, answered in any order: the
            // database's read-back names the sequence that applied.
            StorageStep::ConversationCheckpoint { sequence, .. } => {
                if let Some(requested) = checkpoints.remove(sequence) {
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

/// A later write can apply without a final receipt, for example when its answer is lost.
/// It may advance an item to another accepted revision; it cannot introduce generated text.
fn a_later_accepted_revision_can_already_be_saved() -> Recording {
    settled(Authored::new().session(accepted(message("heard", true))))
        .session(accepted(message("later", true)))
        .storage(StorageStep::Saved { items: vec![message("later", true)] })
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
        .storage(StorageStep::ForeignCommitted { before: vec![], items: vec![message("foreign", false)] })
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
        .storage(StorageStep::ForeignCommitted { before: vec![], items: vec![summary("old")] })
        .storage(StorageStep::ForeignCommitted { before: vec![summary("old")], items: vec![summary("new")] })
        .session(SessionEvent::HistoryReloaded { items: vec![summary("old")] }))
}

fn a_delayed_reload_cannot_undo_foreign_compaction() -> Recording {
    delayed_compaction().storage(StorageStep::Saved { items: vec![summary("new")] }).done()
}

fn a_delayed_reload_cannot_hide_a_lost_foreign_summary() -> Recording {
    delayed_compaction().storage(StorageStep::Saved { items: vec![] }).done()
}

fn foreign_append_with_a_local_edit() -> Authored {
    Authored::new()
        .session(accepted(message("new local words", false)))
        .storage(StorageStep::ForeignCommitted {
            before: vec![message("old stored words", false)],
            items: vec![message("old stored words", false), summary("foreign")],
        })
}

/// Appending an unrelated text message never rewrites this owner's unsaved new voice words.
fn an_unrelated_foreign_append_keeps_a_new_local_revision() -> Recording {
    settled(foreign_append_with_a_local_edit())
        .storage(StorageStep::Saved { items: vec![message("new local words", false), summary("foreign")] })
        .done()
}

fn an_unrelated_foreign_append_cannot_excuse_losing_a_local_revision() -> Recording {
    settled(foreign_append_with_a_local_edit())
        .storage(StorageStep::Saved { items: vec![message("old stored words", false), summary("foreign")] })
        .done()
}

fn foreign_append_after_a_local_receipt() -> Authored {
    settled(Authored::new().session(accepted(message("saved", false))))
        .session(accepted(message("not saved yet", false)))
        .storage(StorageStep::ForeignCommitted {
            before: vec![message("saved", false)],
            items: vec![message("saved", false), summary("foreign")],
        })
}

/// A foreign append acknowledges its own rows, not another owner's newer unsaved words.
fn a_foreign_write_does_not_save_unacknowledged_local_edits() -> Recording {
    foreign_append_after_a_local_receipt()
        .storage(StorageStep::Saved { items: vec![message("saved", false), summary("foreign")] })
        .done()
}

fn a_foreign_write_still_protects_its_own_addition() -> Recording {
    foreign_append_after_a_local_receipt()
        .storage(StorageStep::Saved { items: vec![message("saved", false)] })
        .done()
}

fn an_older_unsaved_local_edit_cannot_replace_a_foreign_edit() -> Recording {
    settled(Authored::new().session(accepted(message("saved", false))))
        .session(accepted(message("older unsaved local edit", false)))
        .storage(StorageStep::ForeignCommitted {
            before: vec![message("saved", false)],
            items: vec![message("actual foreign edit", false)],
        })
        .storage(StorageStep::Saved { items: vec![message("older unsaved local edit", false)] })
        .done()
}

/// The owner captures one revision, accepts another input, then its queued I/O task starts.
/// Only the captured revision is protected by that task's successful receipt.
fn queued_snapshot() -> Authored {
    let captured = message("captured", false);
    let later = ChatItem::Message(ChatMessage::text(
        "later", Role::User, "later input", std::time::UNIX_EPOCH,
    ));
    Authored::new()
        .session(accepted(captured.clone()))
        .journal(Step::Core(CoreStep::ConversationCheckpointPrepared { sequence: 7 }))
        .session(accepted(later))
        .storage(StorageStep::ConversationRequested {
            progress: lk_agents_abi::storage::voice::VoiceConversationProgress {
                sequence: 7,
                agent: "desk".into(),
                history: lk_agents_abi::storage::voice::VoiceHistory::new(vec![captured]),
                pending_user_turns: Vec::new(),
                boundary: lk_agents_abi::storage::voice::VoiceConversationBoundary::UserTurnCompleted,
            },
        })
        .storage(StorageStep::ConversationAnswered { sequence: 7, result: Ok(()) })
}

fn a_queued_snapshot_does_not_include_later_input() -> Recording {
    queued_snapshot().storage(StorageStep::Saved { items: vec![message("captured", false)] }).done()
}

fn a_queued_snapshot_still_protects_its_captured_input() -> Recording {
    queued_snapshot().storage(StorageStep::Saved { items: vec![] }).done()
}

fn delayed_history_callback() -> Authored {
    let before = message("old", false);
    let update = SessionEvent::HistoryReloaded { items: vec![message("edited", false)] };
    Authored::new()
        .journal(Step::Emitted(accepted(before.clone())))
        .session(accepted(before))
        .journal(Step::Emitted(update.clone()))
        .journal(Step::Core(CoreStep::ConversationCheckpointPrepared { sequence: 4 }))
        .session(update)
        .storage(StorageStep::ConversationRequested {
            progress: lk_agents_abi::storage::voice::VoiceConversationProgress {
                sequence: 4,
                agent: "desk".into(),
                history: lk_agents_abi::storage::voice::VoiceHistory::new(vec![message("edited", false)]),
                pending_user_turns: Vec::new(),
                boundary: lk_agents_abi::storage::voice::VoiceConversationBoundary::UserTurnCompleted,
            },
        })
        .storage(StorageStep::ConversationAnswered { sequence: 4, result: Ok(()) })
}

fn a_late_host_callback_keeps_the_snapshot_causal_order() -> Recording {
    delayed_history_callback().storage(StorageStep::Saved { items: vec![message("edited", false)] }).done()
}

fn a_late_host_callback_does_not_excuse_the_old_payload() -> Recording {
    delayed_history_callback().storage(StorageStep::Saved { items: vec![message("old", false)] }).done()
}

fn edited_private_context() -> Authored {
    Authored::new()
        .storage(StorageStep::ForeignCommitted { before: Vec::new(), items: vec![message("stored", false)] })
        .journal(Step::Generation(GenerationStep::ChatCtxUpdated {
            history: lk_agents_abi::storage::voice::VoiceHistory {
                items: vec![message("private edit", false)],
                stored: vec![message("stored", false)], revision: (1, 1), ..Default::default()
            },
        }))
        // Replacing the public history does not expose the private edit as human dialogue.
        .session(SessionEvent::HistoryReloaded { items: Vec::new() })
}

fn an_explicit_private_edit_revises_a_stored_item() -> Recording {
    settled(edited_private_context())
        .storage(StorageStep::Saved { items: vec![message("private edit", false)] }).done()
}

fn an_explicit_private_edit_cannot_be_lost() -> Recording {
    settled(edited_private_context())
        .storage(StorageStep::Saved { items: vec![message("stored", false)] }).done()
}

fn later_foreign_edit() -> Authored {
    edited_private_context().storage(StorageStep::ForeignCommitted {
        before: vec![message("stored", false)], items: vec![message("later foreign edit", false)],
    })
}

fn a_later_foreign_edit_supersedes_an_earlier_private_edit() -> Recording {
    settled(later_foreign_edit())
        .storage(StorageStep::Saved { items: vec![message("later foreign edit", false)] }).done()
}

fn a_private_edit_cannot_replace_a_later_foreign_edit() -> Recording {
    settled(later_foreign_edit())
        .storage(StorageStep::Saved { items: vec![message("private edit", false)] }).done()
}

fn snapshot_after_a_foreign_edit(changed: bool) -> Authored {
    let before = message("original", false);
    Authored::new()
        .session(accepted(before.clone()))
        .storage(StorageStep::ForeignCommitted {
            before: vec![before.clone()], items: vec![message("foreign edit", false)],
        })
        .session(SessionEvent::HistoryReloaded {
            items: vec![if changed { message("later local edit", true) } else { before }],
        })
}

fn an_unchanged_snapshot_does_not_supersede_a_foreign_edit() -> Recording {
    settled(snapshot_after_a_foreign_edit(false))
        .storage(StorageStep::Saved { items: vec![message("foreign edit", false)] }).done()
}

fn an_unchanged_snapshot_cannot_excuse_losing_a_foreign_edit() -> Recording {
    settled(snapshot_after_a_foreign_edit(false))
        .storage(StorageStep::Saved { items: vec![message("original", false)] }).done()
}

fn a_changed_snapshot_can_supersede_a_foreign_edit() -> Recording {
    settled(snapshot_after_a_foreign_edit(true))
        .storage(StorageStep::Saved { items: vec![message("later local edit", true)] }).done()
}

fn a_changed_snapshot_cannot_excuse_losing_the_new_revision() -> Recording {
    settled(snapshot_after_a_foreign_edit(true))
        .storage(StorageStep::Saved { items: vec![message("foreign edit", false)] }).done()
}

fn an_edit_based_on_an_older_read(public: bool) -> Authored {
    let original = message("original", false);
    let edited = message("local edit against an older read", false);
    let mut story = Authored::new().storage(StorageStep::ForeignCommitted {
        before: Vec::new(), items: vec![original.clone()],
    });
    if public {
        story = story.session(accepted(original.clone()));
    }
    story.storage(StorageStep::ForeignCommitted {
        before: vec![original.clone()], items: vec![message("unread foreign edit", false)],
    }).journal(Step::Generation(GenerationStep::ChatCtxUpdated {
        history: lk_agents_abi::storage::voice::VoiceHistory {
            items: vec![edited.clone()], stored: vec![original], revision: (1, 1), ..Default::default()
        },
    })).session(SessionEvent::HistoryReloaded { items: if public { vec![edited] } else { Vec::new() } })
}

fn an_unread_foreign_edit_survives_a_stale_private_edit() -> Recording {
    settled(an_edit_based_on_an_older_read(false))
        .storage(StorageStep::Saved { items: vec![message("unread foreign edit", false)] }).done()
}

fn an_unread_foreign_edit_survives_a_stale_public_edit() -> Recording {
    settled(an_edit_based_on_an_older_read(true))
        .storage(StorageStep::Saved { items: vec![message("unread foreign edit", false)] }).done()
}

fn a_stale_private_edit_cannot_erase_an_unread_foreign_edit() -> Recording {
    settled(an_edit_based_on_an_older_read(false))
        .storage(StorageStep::Saved { items: vec![message("local edit against an older read", false)] }).done()
}

fn a_stale_public_edit_cannot_erase_an_unread_foreign_edit() -> Recording {
    settled(an_edit_based_on_an_older_read(true))
        .storage(StorageStep::Saved { items: vec![message("local edit against an older read", false)] }).done()
}

/// One speech checkpoints twice before either answers: the first applies, the second fails.
/// The first receipt protects its own snapshot, never the revision only the second carried.
fn overlapping_checkpoints() -> Authored {
    let checkpoint = |sequence, item: ChatItem| {
        move |a: Authored| {
            a.journal(Step::Core(CoreStep::ConversationCheckpointPrepared { sequence }))
                .storage(StorageStep::HistorySubmitted { speech: 1, sequence, history: vec![item] })
                .storage(StorageStep::Requested { speech: 1, operation: StorageOperation::Checkpoint })
        }
    };
    let first = Authored::new().session(accepted(message("captured", false)));
    let second = checkpoint(4, message("captured", false))(first)
        .session(accepted(message("revised", false)));
    checkpoint(5, message("revised", false))(second)
        .storage(StorageStep::ConversationCheckpoint {
            sequence: 4, items: vec![message("captured", false)], pending_user_turns: Vec::new(),
        })
        .storage(StorageStep::Answered {
            speech: 1, operation: StorageOperation::Checkpoint, result: Ok(()), delivered: true,
        })
        .storage(StorageStep::Answered {
            speech: 1,
            operation: StorageOperation::Checkpoint,
            result: Err(lk_agents_abi::storage::StorageError::LeaseLost("taken".into())),
            delivered: true,
        })
}

fn an_earlier_checkpoint_receipt_covers_only_its_snapshot() -> Recording {
    overlapping_checkpoints().storage(StorageStep::Saved { items: vec![message("captured", false)] }).done()
}

fn an_earlier_checkpoint_receipt_still_protects_its_snapshot() -> Recording {
    overlapping_checkpoints().storage(StorageStep::Saved { items: vec![] }).done()
}
