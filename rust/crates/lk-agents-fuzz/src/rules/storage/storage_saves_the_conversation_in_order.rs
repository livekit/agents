use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, StorageStep},
};
use lk_agents_abi::{
    AgentConfigUpdate, AgentHandoffItem, ChatContent, ChatItem, ChatMessage, FunctionCall,
    FunctionCallOutput, Role,
};
use lk_agents_voice::agent::{Source, Visibility};
use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
use lk_agents_voice::events::SessionEvent;
use std::time::{Duration, SystemTime};

pub struct StorageSavesTheConversationInOrder;

impl Rule for StorageSavesTheConversationInOrder {
    const CONTRACT: Contract = Contract {
        id: "storage_saves_the_conversation_in_order",
        guarantee: "Persisted conversation items have unique IDs. The call's saved items retain their relative conversation order, including when foreign items share the partition or a fault leaves only a saved subset. Tool outputs remain with their call group even when they arrive after a newer turn.",
        examples: &[a_reloaded_call_keeps_its_late_output, saved_as_held, saved_short, tools_updated_while_they_ran, output_landed_late, shared_with_another_writer, fault_kept_a_prefix],
        counterexamples: &[a_reloaded_call_does_not_excuse_misplaced_output, handoff_saved_late, output_saved_after_the_next_turn, foreign_item_hides_reordering, duplicate_saved_item],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let held = conversation(r);
        let stored: Vec<&str> =
            held.iter().filter(|item| saved_kind(item)).map(ChatItem::id).collect();
        for (_, step) in r.host_view().storage() {
            let StorageStep::Saved { items, .. } = step else { continue };
            let unique: std::collections::BTreeSet<_> = items.iter().map(ChatItem::id).collect();
            check.expect(Subject::Session, unique.len() == items.len(), "saved duplicate item IDs");
            // Foreign rows have their own writer's order; project this call's rows without
            // excluding the whole recording. A failed save may leave any earlier subset.
            let order: Vec<&str> =
                stored.iter().copied().filter(|id| items.iter().any(|saved| saved.id() == *id)).collect();
            let saved: Vec<&str> = items.iter().map(ChatItem::id).filter(|id| stored.contains(id)).collect();
            check.expect_with(Subject::Session, saved == order, || {
                format!("saved {saved:?}, the conversation holds {stored:?}")
            });
        }
    }
}

/// The session's history as the host holds it: each item inserted after the last one created no
/// later, a told-again item replacing its earlier telling, and nothing between a call and its
/// output. An output sits right after its call's group, ahead of what was said, heard or changed
/// while its tool ran, whatever instant it carries.
fn conversation(r: &Recording) -> Vec<ChatItem> {
    let mut held: Vec<ChatItem> = Vec::new();
    let through = super::storage_preserves_item_payloads::durable_through(r);
    for (at, event) in r.host_view().events() {
        if through.is_some_and(|through| at > through) { break; }
        if let SessionEvent::HistoryReloaded { items } = event {
            // The host replaces its ledger on a reload, including calls restored from a
            // checkpoint before their late outputs arrive. The oracle must do the same.
            held = items.clone();
            continue;
        }
        let SessionEvent::ConversationItemAdded { item, visibility, .. } = event else {
            continue;
        };
        if !visibility.history() {
            continue;
        }
        if let Some(told) = held.iter_mut().find(|told| told.id() == item.id()) {
            *told = item.as_ref().clone();
            continue;
        }
        let tool = |i: &ChatItem| matches!(i, ChatItem::FunctionCall(_) | ChatItem::FunctionCallOutput(_));
        let group_end = |held: &[ChatItem], from: usize| {
            from + held[from..].iter().take_while(|i| tool(i)).count()
        };
        let call = match item.as_ref() {
            ChatItem::FunctionCallOutput(output) => held.iter().position(
                |i| matches!(i, ChatItem::FunctionCall(call) if call.call_id == output.call_id),
            ),
            _ => None,
        };
        let at = match call {
            Some(call) => group_end(&held, call),
            None => {
                let at = held
                    .iter()
                    .rposition(|i| i.created_at() <= item.created_at())
                    .map_or(0, |i| i + 1);
                let inside = at > 0
                    && tool(&held[at - 1])
                    && matches!(held.get(at), Some(ChatItem::FunctionCallOutput(_)));
                if inside { group_end(&held, at) } else { at }
            }
        };
        held.insert(at, item.as_ref().clone());
    }
    held
}

/// What a thread row keeps: not instructions, not a config note to the model.
fn saved_kind(item: &ChatItem) -> bool {
    match item {
        ChatItem::AgentConfigUpdate(_) => false,
        ChatItem::Message(message) => !matches!(message.role, Role::System | Role::Developer),
        _ => true,
    }
}

fn at(ms: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_millis(ms)
}

fn added(item: ChatItem) -> SessionEvent {
    added_from(item, Source::HostCtx)
}

fn added_from(item: ChatItem, source: Source) -> SessionEvent {
    SessionEvent::ConversationItemAdded {
        item: Box::new(item),
        agent: "agent".into(),
        source,
        visibility: Visibility::Both,
    }
}

fn message(id: &str, ms: u64) -> ChatItem {
    ChatItem::Message(ChatMessage::new(id, Role::User, vec![ChatContent::Text(id.into())], at(ms)))
}

/// A handoff at 0, then a turn's message and its reply's call.
fn cut_call() -> Authored {
    let handoff = AgentHandoffItem {
        id: "handoff".into(),
        old_agent_id: None,
        new_agent_id: "agent".into(),
        created_at: at(0),
    };
    let call = FunctionCall {
        id: "call".into(),
        call_id: "c1".into(),
        arguments: "{}".into(),
        name: "book".into(),
        created_at: at(200),
        extra: Default::default(),
        group_id: None,
    };
    Authored::new()
        .session(added(ChatItem::AgentHandoff(handoff)))
        .session(added(message("book it", 100)))
        .session(added_from(
            ChatItem::FunctionCall(call),
            Source::ToolCall { speech: SpeechId(1), step: 1 },
        ))
}

/// The call's turn, the next turn taken while the tool ran, and the call's output, stamped as
/// `output_at`.
fn held_with(output_at: u64) -> Authored {
    let output = FunctionCallOutput::new("output", "c1", "booked", false, at(output_at));
    cut_call()
        .session(added(message("are you there?", 400)))
        .session(added(message("Yes.", 500)))
        .session(added_from(
            ChatItem::FunctionCallOutput(output),
            Source::ToolOutput { speech: SpeechId(1), step: 1 },
        ))
}

fn held() -> Authored {
    held_with(200)
}

fn saved(items: &[&str]) -> StorageStep {
    StorageStep::Saved {
        items: items.iter().map(|id| message(id, 0)).collect(),
    }
}

fn saved_as_held() -> Recording {
    held()
        .storage(saved(&["handoff", "book it", "call", "output", "are you there?", "Yes."]))
        .done()
}

/// The turn after the cut one was abandoned: the thread kept what came before it.
fn saved_short() -> Recording {
    held().storage(saved(&["handoff", "book it", "call", "output"])).done()
}

/// The tools changed while the step's call ran: the update sits after the step's output.
fn tools_updated_while_they_ran() -> Recording {
    let update = AgentConfigUpdate {
        id: "update".into(),
        instructions: None,
        tools_added: Some(vec!["cancel".into()]),
        tools_removed: None,
        created_at: at(300),
        tools: None,
    };
    let output = FunctionCallOutput::new("output", "c1", "booked", false, at(350));
    cut_call()
        .session(added_from(ChatItem::AgentConfigUpdate(update), Source::ConfigUpdate))
        .session(added_from(
            ChatItem::FunctionCallOutput(output),
            Source::ToolOutput { speech: SpeechId(1), step: 1 },
        ))
        .storage(saved(&["handoff", "book it", "call", "output"]))
        .done()
}

/// The output was stamped when it landed, after the turn the person took while the tool ran: it
/// still sits right after its call, the turn after it (seed 1242).
fn output_landed_late() -> Recording {
    held_with(600)
        .storage(saved(&["handoff", "book it", "call", "output", "are you there?", "Yes."]))
        .done()
}

fn handoff_saved_late() -> Recording {
    held()
        .storage(saved(&["book it", "call", "output", "are you there?", "handoff", "Yes."]))
        .done()
}

fn output_saved_after_the_next_turn() -> Recording {
    held_with(600)
        .storage(saved(&["handoff", "book it", "call", "are you there?", "Yes.", "output"]))
        .done()
}

fn shared_with_another_writer() -> Recording {
    held().storage(saved(&["foreign", "handoff", "book it", "call", "output", "are you there?", "Yes."])).done()
}

fn fault_kept_a_prefix() -> Recording {
    held()
        .storage(StorageStep::PartitionSettled { unended: 0, held: false, faulted: true })
        .storage(saved(&["handoff", "book it", "call", "output"]))
        .done()
}

fn foreign_item_hides_reordering() -> Recording {
    held().storage(saved(&["foreign", "book it", "handoff", "call", "output"])).done()
}

fn duplicate_saved_item() -> Recording {
    held().storage(saved(&["handoff", "book it", "book it"])).done()
}

fn reloaded_call() -> Authored {
    let call = ChatItem::FunctionCall(FunctionCall {
        id: "restored".into(), call_id: "c1".into(), name: "lookup".into(),
        arguments: "{}".into(), created_at: at(10), extra: Default::default(), group_id: None,
    });
    Authored::new().session(SessionEvent::HistoryReloaded { items: vec![call, message("newer", 20)] })
        .session(added(ChatItem::FunctionCallOutput(FunctionCallOutput::new("late", "c1", "done", false, at(30)))))
}

fn a_reloaded_call_keeps_its_late_output() -> Recording {
    reloaded_call().storage(saved(&["restored", "late", "newer"])).done()
}

fn a_reloaded_call_does_not_excuse_misplaced_output() -> Recording {
    reloaded_call().storage(saved(&["restored", "newer", "late"])).done()
}
