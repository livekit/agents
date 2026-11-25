//! The authoritative session conversation and each agent's view of it.
//!
//! Stable IDs update existing items in place. New items use their logical instant, ties after;
//! tool results stay with their call group even when another turn happened while they ran.
//! Generation/receipt arrival order therefore does not define conversation order. Interrupted
//! speech contributes its accepted heard prefix and interruption flag, not its generated tail.
//! Acceptance here is distinct from durability: unsaved edits survive reload until the save
//! carrying them is acknowledged. Storage reconciliation preserves unseen foreign items.

use crate::agent_session::machine::activity::scheduling::SpeechId;
use crate::audio_recognition::TurnId;
use crate::generation::INSTRUCTIONS_MESSAGE_ID;
use lk_agents_abi::llm::chat_context::ordering::insertion_index;
pub use lk_agents_abi::voice::events::Visibility;
use lk_agents_abi::{ChatContext, ChatItem, Vocabulary};
use std::collections::{BTreeMap, BTreeSet};

/// Why an item exists. Carried on `ItemAdded`, in the trace, and on the record.
#[derive(Debug, Clone, PartialEq, Eq, Vocabulary)]
pub enum Source {
    /// One per merged turn, stamped at the turn's onset.
    UserTurn {
        turn: TurnId,
    },
    /// `generate_reply(user_input=..)`: a user message the host supplied for one reply.
    UserInput {
        speech: SpeechId,
    },
    /// A typed user turn on a realtime session, committed when pushed: no `TurnId`, no `SpeechId` yet.
    UserTyped,
    /// The assistant message of one step of a reply, stamped at its first frame.
    Reply {
        speech: SpeechId,
        step: u32,
    },
    Say {
        speech: SpeechId,
    },
    ToolCall {
        speech: SpeechId,
        step: u32,
    },
    ToolOutput {
        speech: SpeechId,
        step: u32,
    },
    /// A background tool's progress or final pair, committed when the update lands.
    ToolUpdate {
        speech: SpeechId,
        step: u32,
    },
    Handoff,
    ConfigUpdate,
    /// Private framework context describing response admission, never delivered dialogue.
    InputRouting,
    /// `UpdateChatCtx`, `Agent(chat_ctx=..)`.
    HostCtx,
    /// An item the realtime server created; `item_id` is the server's.
    Remote {
        item_id: String,
    },
}

impl Source {
    pub fn speech(&self) -> Option<SpeechId> {
        match self {
            Source::UserInput { speech }
            | Source::Reply { speech, .. }
            | Source::Say { speech }
            | Source::ToolCall { speech, .. }
            | Source::ToolOutput { speech, .. }
            | Source::ToolUpdate { speech, .. } => Some(*speech),
            Source::UserTurn { .. }
            | Source::UserTyped
            | Source::Handoff
            | Source::ConfigUpdate
            | Source::InputRouting
            | Source::HostCtx
            | Source::Remote { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Vocabulary)]
pub enum CommitOutcome {
    /// New to the record.
    Added,
    /// A model-only placeholder entered history. Publish its native position before the item
    /// notification; timestamps alone cannot reconstruct its order among equal instants.
    Promoted,
    /// Replaced in place. Tool revisions are republished; a realtime mirror's repeated message
    /// does not announce another conversation item.
    Replaced,
    /// The item is the fixed-id instructions message of an older session: filtered, not recorded.
    Filtered,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Committed {
    pub item: ChatItem,
    pub source: Source,
    pub visibility: Visibility,
    pub agent: String,
    /// A local replacement retains the exact database value it edited until it rebases.
    revision: HistoryRevision,
}

#[derive(Debug, Clone, PartialEq)]
enum HistoryRevision {
    Stored,
    Local { base: Option<Box<ChatItem>> },
}

#[derive(Debug)]
pub(crate) struct HistoryConflict {
    pub(crate) items: Vec<String>,
}

fn same_payload(left: &ChatItem, right: &ChatItem) -> bool {
    use lk_agents_abi::llm::chat_context::protocol::encode_chat_item;
    encode_chat_item(left) == encode_chat_item(right)
}

/// One edit the end-of-turn hook asks for, applied to that turn's request.
#[derive(Debug, Clone, PartialEq, Vocabulary)]
pub enum TurnEdit {
    UpdateInstructions(String),
    /// Plan this request against these items instead of the record; the record is untouched.
    ReplaceItems(Vec<ChatItem>),
    /// `StopResponse`: no reply, and no user item in history.
    StopResponse,
    /// The hook edited `new_message`'s text: the committed user item and the request carry it.
    UpdateUserMessage(String),
}

/// The session history and each agent's view as one record: sorted by `created_at`, ties after; each id once. Items are stamped with their agent; the
/// agent's view is the current agent's items, history is every agent's.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionChatCtx {
    items: Vec<Committed>,
    ids: BTreeSet<String>,
    agent: String,
    /// Bumped by every commit that adds or replaces; a preemptive generation planned against an
    /// older version is discarded.
    version: u64,
    /// Includes view-only changes that do not invalidate a preemptive request.
    view_revision: u64,
    /// Items rewritten here, each with the edit that last rewrote it, until a save that carried
    /// that edit is answered: a reload keeps them as they are here.
    unsaved: BTreeMap<String, u64>,
    /// The number of the last edit.
    edits: u64,
    /// User items whose turn a realtime server holds as audio: the realtime projection leaves
    /// them out, as it leaves out the server's own items.
    server_audio: BTreeSet<String>,
}

impl SessionChatCtx {
    pub fn new(agent: &str) -> Self {
        SessionChatCtx { agent: agent.to_owned(), ..Default::default() }
    }

    /// A handoff's carry to an agent with `tools`: a call to a tool the new agent lacks, and its
    /// output, stay the previous agent's (in the history, out of the new agent's view).
    pub fn hand_to(&mut self, agent: &str, tools: &BTreeSet<String>) -> Vec<String> {
        let valid: BTreeSet<String> = self
            .items
            .iter()
            .filter_map(|c| match &c.item {
                ChatItem::FunctionCall(call) if tools.contains(&call.name) => {
                    Some(call.call_id.clone())
                }
                _ => None,
            })
            .collect();
        let keep = |item: &ChatItem| match item {
            ChatItem::FunctionCall(call) => tools.contains(&call.name),
            ChatItem::FunctionCallOutput(out) if !out.name.is_empty() => tools.contains(&out.name),
            ChatItem::FunctionCallOutput(out) => valid.contains(&out.call_id),
            _ => true,
        };
        let mut carried = Vec::new();
        let from = std::mem::replace(&mut self.agent, agent.to_owned());
        if from == agent {
            return carried;
        }
        self.view_revision += 1;
        for c in &mut self.items {
            if c.agent == from && c.visibility.model() && keep(&c.item) {
                c.agent = agent.to_owned();
                carried.push(c.item.id().to_owned());
            }
        }
        carried
    }

    pub fn agent(&self) -> &str {
        &self.agent
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    /// `id` was rewritten here (a provider's correction): the thread still holds it as it was.
    pub(crate) fn edited(&mut self, id: &str) {
        self.edits += 1;
        self.unsaved.insert(id.to_owned(), self.edits);
    }

    /// A save carries the record as it is now: the edits it carries, for its answer.
    pub(crate) fn saving(&self) -> u64 {
        self.edits
    }

    /// `id` stands in for a turn whose audio the realtime server holds: it stays in the agent's
    /// view and history, and never goes to the server.
    pub(crate) fn server_holds_audio(&mut self, id: &str) {
        self.server_audio.insert(id.to_owned());
    }

    /// The save that carried the edits up to `edits` was answered: the thread holds them. Items
    /// gone from the record are forgotten.
    pub(crate) fn saved(&mut self, edits: u64) {
        let ids = &self.ids;
        self.unsaved.retain(|id, edit| *edit > edits && ids.contains(id));
    }

    pub(crate) fn view_revision(&self) -> u64 {
        self.view_revision
    }

    /// The only write, inserted by the item's own `created_at`; an existing id is replaced in
    /// place. The fixed-id instructions message is never recorded.
    #[must_use = "a commit is performed by the driver: the speech is told, the event is emitted"]
    pub fn commit(
        &mut self,
        item: ChatItem,
        source: Source,
        visibility: Visibility,
        agent: &str,
    ) -> (Committed, CommitOutcome) {
        let commit = self.insert(item, source, visibility, agent);
        if commit.1 != CommitOutcome::Filtered {
            self.version += 1;
            self.view_revision += 1;
        }
        commit
    }

    fn insert(
        &mut self,
        item: ChatItem,
        source: Source,
        visibility: Visibility,
        agent: &str,
    ) -> (Committed, CommitOutcome) {
        let mut entry = Committed {
            item,
            source,
            visibility,
            agent: agent.to_owned(),
            revision: HistoryRevision::Local { base: None },
        };
        if entry.item.id() == INSTRUCTIONS_MESSAGE_ID {
            return (entry, CommitOutcome::Filtered);
        }
        if matches!(entry.source, Source::ToolCall { .. } | Source::ToolOutput { .. }) {
            let canonical = self
                .items
                .iter()
                .find(|held| held.item.id() == entry.item.id() && held.visibility.history());
            if let Some(held) = canonical {
                // A checkpoint can restore a projected call or output before execution reports
                // it. Raw execution time cannot replace its accepted conversation timestamp.
                match &mut entry.item {
                    ChatItem::FunctionCall(call) => call.created_at = held.item.created_at(),
                    ChatItem::FunctionCallOutput(output) => {
                        output.created_at = held.item.created_at();
                    }
                    _ => {}
                }
            } else if let Source::ToolCall { speech, .. } = entry.source
                && !self.ids.contains(entry.item.id())
            {
                let held = self.items.iter().map(|c| (&c.item, c.source.speech()));
                with_its_turn(&mut entry.item, speech, held);
            }
        }
        let id = entry.item.id();
        let outcome = if self.ids.insert(id.to_owned()) {
            let idx = self.find_insertion_index(&mut entry.item);
            self.items.insert(idx, entry.clone());
            CommitOutcome::Added
        } else {
            // Replaced under the committing agent; an item in history stays where it is.
            let idx = self
                .items
                .iter()
                .position(|c| c.item.id() == id)
                .expect("an id in the set is on the record");
            entry.revision = if same_payload(&self.items[idx].item, &entry.item) {
                self.items[idx].revision.clone()
            } else {
                match &self.items[idx].revision {
                    HistoryRevision::Stored => HistoryRevision::Local {
                        base: Some(Box::new(self.items[idx].item.clone())),
                    },
                    HistoryRevision::Local { base } => {
                        HistoryRevision::Local { base: base.clone() }
                    }
                }
            };
            // A replaced item keeps every projection it was in and gains the replacement's.
            let was = self.items[idx].visibility;
            let visibility =
                match (was.model() || visibility.model(), was.history() || visibility.history()) {
                    (true, true) => Visibility::Both,
                    (true, false) => Visibility::Model,
                    (false, _) => Visibility::History,
                };
            if !was.history() && visibility.history() {
                // A message's known instant keeps its place among equal timestamps. Heard
                // assistant words can carry a new instant; calls join their heard turn.
                let same_message_instant = matches!(entry.item, ChatItem::Message(_))
                    && self.items[idx].item.created_at() == entry.item.created_at();
                if same_message_instant {
                    self.items[idx] = Committed { visibility, ..entry.clone() };
                } else {
                    self.items.remove(idx);
                    if let Source::ToolCall { speech, .. } = entry.source {
                        let held = self.items.iter().map(|c| (&c.item, c.source.speech()));
                        with_its_turn(&mut entry.item, speech, held);
                    }
                    let at = self.find_insertion_index(&mut entry.item);
                    self.items.insert(at, Committed { visibility, ..entry.clone() });
                }
                CommitOutcome::Promoted
            } else {
                self.items[idx] = Committed { visibility, ..entry.clone() };
                CommitOutcome::Replaced
            }
        };
        (entry, outcome)
    }

    /// The partition as storage reloaded it for a turn that read `read`. The record keeps its own
    /// order: storage holds each row at the position the record gave it, so its order is never
    /// the record's to take. Rows the turn read are the partition's to change: another turn's
    /// edits replace them in place (a row given back as the turn read it is the turn's own
    /// rendering, never an edit), and rows it removed, compacted away or abandoned leave. Rows
    /// the turn did not read stay unless their exact current revision was already observed in
    /// storage: a newer observation can update those too. Local edits remain protected. New
    /// stored rows keep their neighbors when those neighbors agree with the record;
    /// without that anchor they join by instant. A divergent foreign/local row is refused
    /// before mutation, preserving the exact local intent and its original base.
    pub(crate) fn reload_transport(
        &mut self,
        read: &[ChatItem],
        items: Vec<ChatItem>,
        stored: &[ChatItem],
        own_writes: &BTreeSet<String>,
        agent: &str,
    ) -> Result<bool, HistoryConflict> {
        let read: BTreeMap<&str, &ChatItem> = read.iter().map(|item| (item.id(), item)).collect();
        let stored: BTreeMap<&str, &ChatItem> =
            stored.iter().map(|item| (item.id(), item)).collect();
        let conflicts: Vec<_> = self
            .items
            .iter()
            .filter_map(|entry| {
                let HistoryRevision::Local { base } = &entry.revision else { return None };
                let id = entry.item.id();
                if own_writes.contains(id) {
                    return None;
                }
                let current = stored.get(id).copied();
                if current.is_some_and(|row| same_payload(&entry.item, row)) {
                    return None;
                }
                let observed_base = base.is_some();
                let base = base.as_deref().or_else(|| read.get(id).copied());
                let local_changed = base.is_none_or(|before| !same_payload(before, &entry.item));
                let foreign_changed = match (base, current) {
                    (Some(before), Some(row)) => !same_payload(before, row),
                    (_, None) if !observed_base => false,
                    (None, None) => false,
                    _ => true,
                };
                (local_changed && foreign_changed).then(|| id.to_owned())
            })
            .collect();
        if !conflicts.is_empty() {
            return Err(HistoryConflict { items: conflicts });
        }
        let before: Vec<ChatItem> = self.items.iter().map(|entry| entry.item.clone()).collect();
        let order: Vec<_> = items
            .iter()
            .filter(|item| item.id() != INSTRUCTIONS_MESSAGE_ID)
            .map(|item| item.id().to_owned())
            .collect();
        let mut partition: Vec<Option<ChatItem>> = items
            .into_iter()
            .filter(|item| item.id() != INSTRUCTIONS_MESSAGE_ID)
            .map(Some)
            .collect();
        let mut take = |id: &str| {
            partition.iter_mut().find(|row| row.as_ref().is_some_and(|row| row.id() == id))?.take()
        };
        let unsaved = &self.unsaved;
        self.items.retain_mut(|entry| {
            let id = entry.item.id().to_owned();
            let row = take(&id);
            let observed = stored.get(id.as_str()).copied();
            if matches!(entry.revision, HistoryRevision::Stored) {
                let Some(observed) = observed else { return false };
                // Rendered items can contain unsaved proposals from an older request. Only
                // the actual database observation supersedes a previously stored revision.
                entry.item = observed.clone();
            } else {
                match (row, read.get(id.as_str())) {
                    // A read row given back as read is the turn's own rendering. A newer local
                    // revision or a correction whose save is unanswered remains ours.
                    (Some(row), Some(read))
                        if *read != &row && &entry.item == *read && !unsaved.contains_key(&id) =>
                    {
                        entry.item = row;
                    }
                    (Some(_), _) | (None, None) => {}
                    (None, Some(read)) if &entry.item == *read && !unsaved.contains_key(&id) => {
                        return false;
                    }
                    (None, Some(_)) => {}
                }
            }
            entry.revision = if observed.is_some_and(|row| same_payload(row, &entry.item)) {
                HistoryRevision::Stored
            } else {
                HistoryRevision::Local { base: observed.cloned().map(Box::new) }
            };
            true
        });
        self.ids = self.items.iter().map(|entry| entry.item.id().to_owned()).collect();
        let ids = &self.ids;
        self.unsaved.retain(|id, _| ids.contains(id));
        // A row the turn read that the record does not hold is the request's own (its prompt, its
        // new message): never the partition's to add.
        for (index, row) in partition.into_iter().enumerate() {
            let Some(row) = row.filter(|row| !read.contains_key(row.id())) else { continue };
            let mut entry = Committed {
                revision: HistoryRevision::Local { base: None },
                item: row,
                source: Source::HostCtx,
                visibility: Visibility::Both,
                agent: agent.to_owned(),
            };
            let next = order[index + 1..]
                .iter()
                .find_map(|id| self.items.iter().position(|entry| entry.item.id() == id));
            let previous = order[..index].iter().rev().find_map(|id| {
                self.items.iter().position(|entry| entry.item.id() == id).map(|at| at + 1)
            });
            let anchor = match (previous, next) {
                (Some(previous), Some(next)) if previous > next => None,
                (_, Some(next)) => Some(next),
                (previous, None) => previous,
            };
            let at = if matches!(entry.item, ChatItem::FunctionCallOutput(_)) {
                self.find_insertion_index(&mut entry.item)
            } else {
                anchor.unwrap_or_else(|| self.find_insertion_index(&mut entry.item))
            };
            entry.revision = match stored.get(entry.item.id()).copied() {
                Some(row) if same_payload(row, &entry.item) => HistoryRevision::Stored,
                observed => HistoryRevision::Local { base: observed.cloned().map(Box::new) },
            };
            self.ids.insert(entry.item.id().to_owned());
            self.items.insert(at, entry);
        }
        let changed = !before.iter().eq(self.items.iter().map(|entry| &entry.item));
        if changed {
            self.version += 1;
            self.view_revision += 1;
        }
        Ok(changed)
    }

    /// `agent`'s view becomes exactly `items` (the version is kept).
    /// Dropped items stay in history; known items are promoted in place; new ones are `HostCtx`,
    /// frame-only.
    pub fn update_agent_chat_ctx(&mut self, items: Vec<ChatItem>, agent: &str) {
        self.view_revision += 1;
        self.agent = agent.to_owned();
        let incoming: BTreeSet<&str> = items.iter().map(ChatItem::id).collect();
        // A host-supplied item nobody else was told about leaves the record entirely.
        let ids = &mut self.ids;
        self.items.retain_mut(|c| {
            if c.agent != agent || !c.visibility.model() || incoming.contains(c.item.id()) {
                return true;
            }
            if c.visibility.history() {
                c.visibility = Visibility::History;
                true
            } else {
                ids.remove(c.item.id());
                false
            }
        });
        for item in items {
            let _ = self.insert(item, Source::HostCtx, Visibility::Model, agent);
        }
    }

    /// `Agent.chat_ctx`: the driving agent's view.
    pub fn agent_chat_ctx(&self) -> ChatContext {
        ChatContext::new(
            self.items
                .iter()
                .filter(|c| c.agent == self.agent && c.visibility.model())
                .map(|c| c.item.clone())
                .collect(),
        )
    }

    /// The id of the driving agent's last item, without building its view.
    pub fn last_agent_item_id(&self) -> Option<&str> {
        self.items
            .iter()
            .rev()
            .find(|c| c.agent == self.agent && c.visibility.model())
            .map(|c| c.item.id())
    }

    /// A realtime server's mirror: the agent's view minus items the server created or holds as
    /// audio, and minus agent records.
    pub fn for_realtime(&self) -> ChatContext {
        ChatContext::new(
            self.items
                .iter()
                .filter(|c| {
                    c.agent == self.agent
                        && c.visibility.model()
                        && !matches!(c.source, Source::Remote { .. })
                        && !self.server_audio.contains(c.item.id())
                        && !matches!(
                            c.item,
                            ChatItem::AgentHandoff(_) | ChatItem::AgentConfigUpdate(_)
                        )
                })
                .map(|c| c.item.clone())
                .collect(),
        )
    }

    pub fn contains(&self, id: &str) -> bool {
        self.ids.contains(id)
    }

    /// `session.history`: every item the history shows, in record order.
    pub fn history(&self) -> impl Iterator<Item = &ChatItem> + Clone {
        self.items.iter().filter(|c| c.visibility.history()).map(|c| &c.item)
    }

    /// Every item of every agent, in record order.
    pub fn items(&self) -> impl Iterator<Item = &ChatItem> + Clone {
        self.items.iter().map(|c| &c.item)
    }

    /// The user message `turn` committed.
    pub(crate) fn user_turn_message(&self, turn: TurnId) -> Option<&lk_agents_abi::ChatMessage> {
        self.items.iter().find_map(|c| match (&c.item, &c.source) {
            (ChatItem::Message(m), Source::UserTurn { turn: t }) if *t == turn => Some(m),
            _ => None,
        })
    }

    pub fn entry(&self, id: &str) -> Option<&Committed> {
        self.contains(id).then(|| self.items.iter().find(|c| c.item.id() == id)).flatten()
    }

    /// Where `item` joins the record, an output stamped at its call's group.
    fn find_insertion_index(&self, item: &mut ChatItem) -> usize {
        insertion_index(self.items.iter().map(|c| &c.item), item)
    }

    /// Every item as the record will hold it once `pending`, each with the reply it belongs to,
    /// is committed: a new one inserted as a commit inserts it, one already held kept as held.
    pub fn projected(
        &self,
        pending: impl IntoIterator<Item = (ChatItem, Option<SpeechId>)>,
    ) -> Vec<ChatItem> {
        let mut items: Vec<(ChatItem, Option<SpeechId>)> =
            self.items.iter().map(|c| (c.item.clone(), c.source.speech())).collect();
        for (mut item, speech) in pending {
            if items.iter().any(|(held, _)| held.id() == item.id()) {
                continue;
            }
            if let Some(speech) = speech {
                with_its_turn(&mut item, speech, items.iter().map(|(i, s)| (i, *s)));
            }
            let at = insertion_index(items.iter().map(|(i, _)| i), &mut item);
            items.insert(at, (item, speech));
        }
        items.into_iter().map(|(item, _)| item).collect()
    }
}

/// A reply's tool call sits with its turn: right after the last item the reply holds (its step's
/// heard words, or the calls and outputs before it), at that item's instant. Never heard, a call
/// never takes its place by when it started: a step's calls started before and after its words
/// were first heard would straddle them.
fn with_its_turn<'a>(
    item: &mut ChatItem,
    speech: SpeechId,
    mut held: impl DoubleEndedIterator<Item = (&'a ChatItem, Option<SpeechId>)>,
) {
    let ChatItem::FunctionCall(call) = item else { return };
    if let Some((turn, _)) = held.rfind(|(_, of)| *of == Some(speech)) {
        call.created_at = turn.created_at();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::{ChatMessage, FunctionCall, FunctionCallOutput, Role};
    use std::time::{Duration, SystemTime};

    // Catches: a provider's correction reverted by the next turn's reload once a save was only
    // asked (and then failed), or kept against the thread after the save that carried it was
    // answered.
    #[test]
    fn a_reload_keeps_a_correction_until_its_save_is_answered() {
        let said = |text: &str| {
            ChatItem::Message(ChatMessage::text("u", Role::User, text, SystemTime::UNIX_EPOCH))
        };
        let reload = |ctx: &mut SessionChatCtx| {
            // The turn read the corrected record; the thread still holds the words it replaced.
            ctx.reload_transport(
                &[said("book a table for two")],
                vec![said("book a table")],
                &[],
                &Default::default(),
                "agent",
            )
            .unwrap()
        };
        let mut ctx = SessionChatCtx::new("agent");
        let _ = ctx.commit(said("book a table"), Source::HostCtx, Visibility::Both, "agent");
        ctx.edited("u");
        let _ =
            ctx.commit(said("book a table for two"), Source::HostCtx, Visibility::Both, "agent");
        // A save asked, never answered (it failed): the thread may still hold the old words.
        let asked = ctx.saving();
        assert!(!reload(&mut ctx));
        let kept = ctx.history().next().and_then(|item| item.as_message()?.raw_text_content());
        assert_eq!(kept.as_deref(), Some("book a table for two"));
        // An answered save carried it: the thread's row is the partition's again.
        ctx.saved(asked);
        assert!(reload(&mut ctx));
    }

    #[test]
    fn a_recovered_speech_prefix_keeps_its_stored_place_before_its_tool_group() {
        let at = |n| SystemTime::UNIX_EPOCH + Duration::from_secs(n);
        let prefix =
            ChatItem::Message(ChatMessage::text("prefix", Role::Assistant, "Checking.", at(2)));
        let call = ChatItem::FunctionCall(FunctionCall {
            id: "call".into(),
            call_id: "lookup".into(),
            arguments: "{}".into(),
            name: "lookup".into(),
            created_at: at(1),
            extra: Default::default(),
            group_id: None,
        });
        let mut context = SessionChatCtx::new("agent");
        let _ = context.commit(call.clone(), Source::HostCtx, Visibility::Both, "agent");
        context
            .reload_transport(
                std::slice::from_ref(&call),
                vec![prefix, call.clone()],
                &[],
                &Default::default(),
                "agent",
            )
            .unwrap();
        let ids: Vec<_> = context.history().map(ChatItem::id).collect();
        assert_eq!(
            ids,
            ["prefix", "call"],
            "recovery preserves the stored conversation's neighbors"
        );
    }

    #[test]
    fn a_late_reload_cannot_replace_or_remove_a_newer_accepted_revision() {
        let item = |text| {
            ChatItem::Message(ChatMessage::text("user", Role::User, text, SystemTime::UNIX_EPOCH))
        };
        let read = item("draft");
        let final_input = item("final words");
        let mut context = SessionChatCtx::new("agent");
        let _ = context.commit(read.clone(), Source::HostCtx, Visibility::Both, "agent");
        let _ = context.commit(final_input.clone(), Source::HostCtx, Visibility::Both, "agent");
        context
            .reload_transport(
                std::slice::from_ref(&read),
                vec![item("an older foreign edit")],
                &[],
                &Default::default(),
                "agent",
            )
            .unwrap();
        assert_eq!(
            context.history().cloned().collect::<Vec<_>>(),
            std::slice::from_ref(&final_input)
        );
        context.reload_transport(&[read], Vec::new(), &[], &Default::default(), "agent").unwrap();
        assert_eq!(context.history().cloned().collect::<Vec<_>>(), [final_input]);
    }

    #[test]
    fn a_newer_storage_observation_updates_a_row_missing_from_its_request() {
        let item = |text| {
            ChatItem::Message(ChatMessage::text(
                "text",
                Role::Assistant,
                text,
                SystemTime::UNIX_EPOCH,
            ))
        };
        let mut context = SessionChatCtx::new("agent");
        context
            .reload_transport(
                &[],
                vec![item("Sent.")],
                &[item("Sent.")],
                &Default::default(),
                "agent",
            )
            .unwrap();
        // Both reads began before this row arrived. The second observes its foreign edit.
        context
            .reload_transport(
                &[],
                vec![item("Edited.")],
                &[item("Edited.")],
                &Default::default(),
                "agent",
            )
            .unwrap();
        assert_eq!(context.history().cloned().collect::<Vec<_>>(), [item("Edited.")]);
        context.reload_transport(&[], Vec::new(), &[], &Default::default(), "agent").unwrap();
        assert_eq!(context.history().count(), 0, "a later foreign deletion is also observed");
    }

    #[test]
    fn a_storage_observation_cannot_replace_a_locally_edited_stored_row() {
        let item = |text| {
            ChatItem::Message(ChatMessage::text(
                "text",
                Role::Assistant,
                text,
                SystemTime::UNIX_EPOCH,
            ))
        };
        let mut context = SessionChatCtx::new("agent");
        context
            .reload_transport(
                &[],
                vec![item("Stored.")],
                &[item("Stored.")],
                &Default::default(),
                "agent",
            )
            .unwrap();
        let _ = context.commit(item("New words."), Source::HostCtx, Visibility::Both, "agent");
        let conflict = context
            .reload_transport(
                &[],
                vec![item("Foreign edit.")],
                &[item("Foreign edit.")],
                &Default::default(),
                "agent",
            )
            .unwrap_err();
        assert_eq!(conflict.items, ["text"]);
        assert_eq!(context.history().cloned().collect::<Vec<_>>(), [item("New words.")]);
    }

    #[test]
    fn a_handoff_leaves_calls_to_tools_the_new_agent_lacks() {
        let at = |s: u64| SystemTime::UNIX_EPOCH + Duration::from_secs(s);
        let call = FunctionCall {
            id: "fc".into(),
            call_id: "call_1".into(),
            arguments: "{}".into(),
            name: "to_billing".into(),
            created_at: at(2),
            extra: Default::default(),
            group_id: None,
        };
        let output = FunctionCallOutput::new("fo", "call_1", "", false, at(3));
        let items = [
            ChatItem::Message(ChatMessage::text("u", Role::User, "billing please", at(1))),
            ChatItem::FunctionCall(call),
            ChatItem::FunctionCallOutput(output),
        ];
        let mut ctx = SessionChatCtx::new("front_desk");
        for item in items {
            let _ = ctx.commit(item, Source::HostCtx, Visibility::Both, "front_desk");
        }
        let carried = ctx.hand_to("billing", &BTreeSet::new());
        assert_eq!(carried, ["u"]);
        let ids: Vec<_> = ctx.agent_chat_ctx().items.iter().map(|i| i.id().to_owned()).collect();
        assert_eq!(ids, ["u"]);
    }

    /// Seed 1227: a question another agent asked was answered after the last commit; a reload
    /// for a turn that did not read it gives back the committed, waiting row, and the answer
    /// stays. A row the turn read and the partition changed is replaced.
    #[test]
    fn a_reload_changes_only_the_rows_the_turn_read() {
        let at = SystemTime::UNIX_EPOCH;
        let output = |text: &str| {
            ChatItem::FunctionCallOutput(FunctionCallOutput::new("fo", "call_1", text, false, at))
        };
        let said = |text: &str| ChatItem::Message(ChatMessage::text("u", Role::User, text, at));
        let mut ctx = SessionChatCtx::new("caller");
        let _ = ctx.commit(output("Monday"), Source::HostCtx, Visibility::History, "agent");
        let _ = ctx.commit(said("hello"), Source::HostCtx, Visibility::Both, "caller");
        let read = [said("hello")];
        assert!(
            ctx.reload_transport(
                &read,
                vec![output("waiting"), said("hello, edited")],
                &[],
                &Default::default(),
                "caller"
            )
            .unwrap()
        );
        let items: Vec<_> = ctx.items().cloned().collect();
        assert_eq!(items, [output("Monday"), said("hello, edited")]);
    }

    // Catches (seed 127): a reload took the partition's order for the rows the turn read, so a
    // message the session held before the reply moved after it, in `session.history` and, once
    // saved again, in the thread. The record keeps its order; another turn's row joins by its
    // instant.
    #[test]
    fn a_reload_keeps_the_records_order() {
        let at = |s: u64| SystemTime::UNIX_EPOCH + Duration::from_secs(s);
        let said = |id: &str, s| ChatItem::Message(ChatMessage::text(id, Role::User, id, at(s)));
        let mut ctx = SessionChatCtx::new("caller");
        for item in [said("asked", 1), said("reply", 3)] {
            let _ = ctx.commit(item, Source::HostCtx, Visibility::Both, "caller");
        }
        let read = [said("asked", 1), said("reply", 3)];
        let partition = vec![said("reply", 3), said("text", 2), said("asked", 1)];
        assert!(
            ctx.reload_transport(&read, partition, &[], &Default::default(), "caller").unwrap()
        );
        let ids: Vec<_> = ctx.items().map(|item| item.id().to_owned()).collect();
        assert_eq!(ids, ["asked", "text", "reply"]);
    }

    // Catches (seed 289): a realtime reply the model held before it was heard joined the history
    // where its earlier telling sat, ahead of the person's words said before it was heard, so
    // the history and the thread held it out of order. Joining the history, an item takes its
    // place by its own instant.
    #[test]
    fn an_item_joining_the_history_takes_its_place_by_its_instant() {
        let at = |s: u64| SystemTime::UNIX_EPOCH + Duration::from_secs(s);
        let msg = |id: &str, role, s| ChatItem::Message(ChatMessage::text(id, role, id, at(s)));
        let mut ctx = SessionChatCtx::new("caller");
        let _ = ctx.commit(
            msg("reply", Role::Assistant, 1),
            Source::HostCtx,
            Visibility::Model,
            "caller",
        );
        let _ =
            ctx.commit(msg("asked", Role::User, 2), Source::HostCtx, Visibility::Both, "caller");
        let (_, outcome) = ctx.commit(
            msg("reply", Role::Assistant, 3),
            Source::HostCtx,
            Visibility::Both,
            "caller",
        );
        assert_eq!(outcome, CommitOutcome::Promoted);
        let ids: Vec<_> = ctx.history().map(|item| item.id().to_owned()).collect();
        assert_eq!(ids, ["asked", "reply"]);
    }

    #[test]
    fn a_model_only_call_joins_its_heard_words_when_it_enters_history() {
        let speech = SpeechId(7);
        let started = SystemTime::UNIX_EPOCH + Duration::from_secs(1);
        let heard = started + Duration::from_secs(1);
        let call = ChatItem::FunctionCall(FunctionCall {
            id: "call".into(),
            call_id: "call".into(),
            arguments: "{}".into(),
            name: "lookup".into(),
            created_at: started,
            extra: Default::default(),
            group_id: None,
        });
        let mut ctx = SessionChatCtx::new("agent");
        let source = Source::ToolCall { speech, step: 1 };
        let _ = ctx.commit(call.clone(), source.clone(), Visibility::Model, "agent");
        let words =
            ChatItem::Message(ChatMessage::text("words", Role::Assistant, "Checking.", heard));
        let _ = ctx.commit(words, Source::Reply { speech, step: 1 }, Visibility::Both, "agent");
        let _ = ctx.commit(call, source, Visibility::Both, "agent");
        let items: Vec<_> = ctx.history().collect();
        assert_eq!(items.iter().map(|item| item.id()).collect::<Vec<_>>(), ["words", "call"]);
        assert_eq!(items[1].created_at(), heard);
    }

    #[test]
    fn a_restored_tool_call_keeps_its_checkpointed_timestamp_when_execution_reports_it() {
        let speech = SpeechId(7);
        let started = SystemTime::UNIX_EPOCH + Duration::from_secs(1);
        let heard = started + Duration::from_secs(1);
        let call = |at| {
            ChatItem::FunctionCall(FunctionCall {
                id: "call".into(),
                call_id: "call".into(),
                arguments: "{}".into(),
                name: "lookup".into(),
                created_at: at,
                extra: Default::default(),
                group_id: None,
            })
        };
        let words =
            ChatItem::Message(ChatMessage::text("words", Role::Assistant, "Checking.", heard));
        let mut ctx = SessionChatCtx::new("agent");
        ctx.reload_transport(&[], vec![words, call(heard)], &[], &Default::default(), "agent")
            .unwrap();
        let _ = ctx.commit(
            call(started),
            Source::ToolCall { speech, step: 1 },
            Visibility::Both,
            "agent",
        );
        let items: Vec<_> = ctx.history().collect();
        assert_eq!(items.iter().map(|item| item.id()).collect::<Vec<_>>(), ["words", "call"]);
        assert_eq!(items[1].created_at(), heard);
    }

    #[test]
    fn a_restored_tool_output_keeps_its_checkpointed_timestamp_when_execution_reports_it() {
        let speech = SpeechId(7);
        let heard = SystemTime::UNIX_EPOCH + Duration::from_secs(1);
        let finished = heard + Duration::from_secs(2);
        let output = |at| {
            ChatItem::FunctionCallOutput(FunctionCallOutput::new(
                "output", "call", "Found.", false, at,
            ))
        };
        let mut context = SessionChatCtx::new("agent");
        context
            .reload_transport(
                &[],
                vec![output(heard)],
                &[output(heard)],
                &Default::default(),
                "agent",
            )
            .unwrap();
        let _ = context.commit(
            output(finished),
            Source::ToolOutput { speech, step: 1 },
            Visibility::Both,
            "agent",
        );
        assert_eq!(context.history().cloned().collect::<Vec<_>>(), [output(heard)]);
    }

    // Catches (survey seed 317): a step's calls kept the instants they started at while its words
    // took the instant they were first heard, so the words sat between the calls started before
    // and after it, and a cut turn's late outputs ended up apart from their calls. A call sits
    // with its turn, and a checkpoint projects it where the commit puts it.
    #[test]
    fn a_steps_calls_sit_together_right_after_its_words() {
        let at = |ms: u64| SystemTime::UNIX_EPOCH + Duration::from_millis(ms);
        let speech = SpeechId(7);
        let said = |id: &str, role, ms| ChatItem::Message(ChatMessage::text(id, role, id, at(ms)));
        let call = |id: &str, ms| {
            ChatItem::FunctionCall(FunctionCall {
                id: id.into(),
                call_id: id.into(),
                arguments: "{}".into(),
                name: "lookup".into(),
                created_at: at(ms),
                extra: Default::default(),
                group_id: None,
            })
        };
        let mut ctx = SessionChatCtx::new("agent");
        let commit = |ctx: &mut SessionChatCtx, item, source| {
            let _ = ctx.commit(item, source, Visibility::Both, "agent");
        };
        commit(&mut ctx, said("asked", Role::User, 10_000), Source::HostCtx);
        let words = Source::Reply { speech, step: 1 };
        commit(&mut ctx, said("words", Role::Assistant, 50_000), words);
        // the person cut the reply while its tools ran
        commit(&mut ctx, said("cut", Role::User, 51_500), Source::HostCtx);
        // one call started before the words were heard, one after the person spoke
        let calls = [call("a", 15_000), call("b", 51_700)];
        let projected = ctx.projected(calls.iter().map(|c| (c.clone(), Some(speech)))).into_iter();
        let projected: Vec<String> = projected.map(|i| i.id().to_owned()).collect();
        for item in calls {
            commit(&mut ctx, item, Source::ToolCall { speech, step: 1 });
        }
        // each output stamped when it landed, after the person spoke
        for id in ["a", "b"] {
            let output = FunctionCallOutput::new(format!("{id}_out"), id, "", false, at(60_000));
            commit(&mut ctx, output.into(), Source::ToolOutput { speech, step: 1 });
        }
        let ids: Vec<_> = ctx.history().map(|item| item.id().to_owned()).collect();
        assert_eq!(ids, ["asked", "words", "a", "b", "a_out", "b_out", "cut"]);
        assert_eq!(projected, ["asked", "words", "a", "b", "cut"]);
        // the host reads the order back from the instants
        assert!(
            ctx.history().zip(ctx.history().skip(1)).all(|(a, b)| a.created_at() <= b.created_at())
        );
    }
    #[test]
    fn a_late_foreign_read_rebases_disjoint_local_intent_atomically() {
        let item = |id: &str, text: &str| {
            ChatItem::Message(ChatMessage::text(id, Role::User, text, SystemTime::UNIX_EPOCH))
        };
        let before = vec![item("foreign", "old foreign"), item("local", "old local")];
        let mut context = SessionChatCtx::new("agent");
        context
            .reload_transport(&[], before.clone(), &before, &Default::default(), "agent")
            .unwrap();
        let _ =
            context.commit(item("local", "new local"), Source::HostCtx, Visibility::Both, "agent");
        let remote = vec![
            item("foreign", "new foreign"),
            item("local", "old local"),
            item("new", "foreign addition"),
        ];
        assert!(
            context
                .reload_transport(&before, remote.clone(), &remote, &Default::default(), "agent")
                .unwrap()
        );
        let held: Vec<_> = context.history().cloned().collect();
        assert!(held.contains(&item("foreign", "new foreign")));
        assert!(held.contains(&item("local", "new local")));
        assert!(held.contains(&item("new", "foreign addition")));
    }

    #[test]
    fn a_late_foreign_collision_keeps_the_local_intent_and_its_original_base() {
        let item = |text: &str| {
            ChatItem::Message(ChatMessage::text("item", Role::User, text, SystemTime::UNIX_EPOCH))
        };
        let original = item("original");
        let mut context = SessionChatCtx::new("agent");
        context
            .reload_transport(
                &[],
                vec![original.clone()],
                std::slice::from_ref(&original),
                &Default::default(),
                "agent",
            )
            .unwrap();
        let _ = context.commit(item("local intent"), Source::HostCtx, Visibility::Both, "agent");
        let before = context.clone();
        let foreign = item("foreign intent");
        let conflict = context
            .reload_transport(
                std::slice::from_ref(&original),
                vec![foreign.clone()],
                &[foreign],
                &Default::default(),
                "agent",
            )
            .unwrap_err();
        assert_eq!(conflict.items, ["item"]);
        assert_eq!(
            context, before,
            "a refused rebase cannot relabel or partially mutate the local intent"
        );
    }

    #[test]
    fn an_own_checkpoint_receipt_does_not_conflict_with_later_local_final_words() {
        let item = |text: &str| {
            ChatItem::Message(ChatMessage::text("input", Role::User, text, SystemTime::UNIX_EPOCH))
        };
        let original = item("draft");
        let mut context = SessionChatCtx::new("agent");
        context
            .reload_transport(
                &[],
                vec![original.clone()],
                std::slice::from_ref(&original),
                &Default::default(),
                "agent",
            )
            .unwrap();
        let _ = context.commit(item("final words"), Source::HostCtx, Visibility::Both, "agent");
        let ours = item("intermediate draft");
        context
            .reload_transport(
                &[original],
                vec![ours.clone()],
                &[ours],
                &BTreeSet::from(["input".into()]),
                "agent",
            )
            .unwrap();
        assert_eq!(context.history().cloned().collect::<Vec<_>>(), [item("final words")]);
    }
}
