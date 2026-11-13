use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Recording, StorageStep},
};
use lk_agents_abi::{
    AgentHandoffItem, ChatContent, ChatItem, ChatMessage, FunctionCall, FunctionCallOutput, Role,
};
use lk_agents_voice::agent::{Source, Visibility};
use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::{SpeechStep, Step};
use std::time::{Duration, SystemTime};

pub struct StorageSavesTheConversationInOrder;

impl Rule for StorageSavesTheConversationInOrder {
    const CONTRACT: Contract = Contract {
        id: "storage_saves_the_conversation_in_order",
        guarantee: "The thread saves the call's conversation in the order the session holds it: each turn's items together, in the order they happened, a handoff where it happened. A turn the person cut while its tools ran keeps its calls and their late outputs together, right after what was said before them and before the next turn. A save may stop short (an abandoned turn, a lost partition); a thread another writer shares, or a faulted store, is not judged.",
        examples: &[saved_as_held, saved_short],
        counterexamples: &[handoff_saved_late, output_saved_after_the_next_turn, output_held_late],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let held = conversation(r);
        let cut: Vec<SpeechId> = r
            .journal()
            .iter()
            .filter_map(|(_, step)| match step {
                Step::Speech(SpeechStep::ReleasedToTools { speech }) => Some(*speech),
                _ => None,
            })
            .collect();
        // A cut turn's tool outputs follow their call's group: only calls and outputs between.
        for (at, (item, source)) in held.iter().enumerate() {
            let ChatItem::FunctionCallOutput(output) = item else { continue };
            if !source.speech().is_some_and(|speech| cut.contains(&speech)) {
                continue;
            }
            let Some(call) = held.iter().position(
                |(c, _)| matches!(c, ChatItem::FunctionCall(call) if call.call_id == output.call_id),
            ) else {
                continue;
            };
            let together = call < at
                && held[call..at].iter().all(|(between, _)| {
                    matches!(between, ChatItem::FunctionCall(_) | ChatItem::FunctionCallOutput(_))
                });
            check.expect_with(Subject::Session, together, || {
                format!(
                    "the output of {} is not right after its call: {:?}",
                    output.call_id,
                    ids(&held)
                )
            });
        }
        let stored: Vec<&str> =
            held.iter().filter(|(item, _)| saved_kind(item)).map(|(item, _)| item.id()).collect();
        let faulted =
            r.host_view().storage().iter().any(|(_, step)| {
                matches!(step, StorageStep::PartitionSettled { faulted: true, .. })
            });
        for (_, step) in r.host_view().storage() {
            let StorageStep::Saved { items } = step else { continue };
            // Only this call's rows, from a store that met no fault: their order is the call's.
            if faulted || items.iter().any(|id| !stored.contains(&id.as_str())) {
                continue;
            }
            let order: Vec<&str> =
                stored.iter().copied().filter(|id| items.iter().any(|saved| saved == id)).collect();
            let saved: Vec<&str> = items.iter().map(String::as_str).collect();
            check.expect_with(Subject::Session, saved == order, || {
                format!("saved {saved:?}, the conversation holds {stored:?}")
            });
        }
    }
}

/// The session's history as the host holds it: each item inserted after the last one created no
/// later, a told-again item replacing its earlier telling.
fn conversation(r: &Recording) -> Vec<(ChatItem, Source)> {
    let mut held: Vec<(ChatItem, Source)> = Vec::new();
    for (_, event) in r.host_view().events() {
        let SessionEvent::ConversationItemAdded { item, visibility, source, .. } = event else {
            continue;
        };
        if !visibility.history() {
            continue;
        }
        if let Some(told) = held.iter_mut().find(|(told, _)| told.id() == item.id()) {
            told.0 = item.as_ref().clone();
            continue;
        }
        let at = held
            .iter()
            .rposition(|(i, _)| i.created_at() <= item.created_at())
            .map_or(0, |i| i + 1);
        held.insert(at, (item.as_ref().clone(), source.clone()));
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

fn ids(items: &[(ChatItem, Source)]) -> Vec<&str> {
    items.iter().map(|(item, _)| item.id()).collect()
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

/// A handoff at 0, the cut turn's message and call, the next turn, and the call's late output,
/// stamped as `output_at`.
fn held_with(output_at: u64) -> Authored {
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
    let output = FunctionCallOutput::new("output", "c1", "booked", false, at(output_at));
    let cut = SpeechId(1);
    Authored::new()
        .journal(Step::Speech(SpeechStep::ReleasedToTools { speech: cut }))
        .session(added(ChatItem::AgentHandoff(handoff)))
        .session(added(message("book it", 100)))
        .session(added_from(
            ChatItem::FunctionCall(call),
            Source::ToolCall { speech: cut, step: 1 },
        ))
        .session(added(message("are you there?", 400)))
        .session(added(message("Yes.", 500)))
        .session(added_from(
            ChatItem::FunctionCallOutput(output),
            Source::ToolOutput { speech: cut, step: 1 },
        ))
}

fn held() -> Authored {
    held_with(200)
}

fn saved(items: &[&str]) -> StorageStep {
    StorageStep::Saved { items: items.iter().map(|id| (*id).to_owned()).collect() }
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

/// The late output was stamped when it landed: the session holds it after the next turn.
fn output_held_late() -> Recording {
    held_with(600)
        .storage(saved(&["handoff", "book it", "call", "are you there?", "Yes.", "output"]))
        .done()
}

fn handoff_saved_late() -> Recording {
    held()
        .storage(saved(&["book it", "call", "output", "are you there?", "handoff", "Yes."]))
        .done()
}

fn output_saved_after_the_next_turn() -> Recording {
    held()
        .storage(saved(&["handoff", "book it", "call", "are you there?", "Yes.", "output"]))
        .done()
}
