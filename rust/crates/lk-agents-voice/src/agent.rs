use crate::agent_activity::scheduling::SpeechId;
use crate::audio_recognition::TurnId;
use crate::generation::INSTRUCTIONS_MESSAGE_ID;
use lk_agents_abi::{ChatContext, ChatItem, Vocabulary};
use std::collections::BTreeSet;

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
    /// `UpdateChatCtx`, `Agent(chat_ctx=..)`, a task's context merged back.
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
            | Source::HostCtx
            | Source::Remote { .. } => None,
        }
    }
}

/// Carried on `ItemAdded` so the host's projections decide nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Vocabulary)]
pub enum Visibility {
    /// The frame's context only: a host-supplied item, not in `session.history`.
    Model,
    History,
    /// The frame's context and `session.history`: everything said, and the config records.
    Both,
}

impl Visibility {
    pub fn model(self) -> bool {
        match self {
            Visibility::Model | Visibility::Both => true,
            Visibility::History => false,
        }
    }

    pub fn history(self) -> bool {
        match self {
            Visibility::History | Visibility::Both => true,
            Visibility::Model => false,
        }
    }

    /// A replaced item keeps every projection it was in and gains the replacement's.
    fn union(self, other: Visibility) -> Visibility {
        match (self.model() || other.model(), self.history() || other.history()) {
            (true, true) => Visibility::Both,
            (true, false) => Visibility::Model,
            (false, true) => Visibility::History,
            // Unreachable: every variant is in at least one projection.
            (false, false) => Visibility::History,
        }
    }

    /// The wire spelling (`ItemAdded.visibility` as the host reads it).
    pub fn as_str(self) -> &'static str {
        match self {
            Visibility::Model => "model",
            Visibility::History => "history",
            Visibility::Both => "both",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Vocabulary)]
pub enum CommitOutcome {
    Added,
    /// Replaced in place (a realtime server re-finalising one `item_id`); no second `ItemAdded`.
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

/// `AgentSession._chat_ctx` (the history) and each agent's `chat_ctx` (its view) as one record:
/// sorted by `created_at`, ties after; each id once. Items are stamped with their frame; the
/// agent's view is the current frame's items, history is every frame's.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionChatCtx {
    items: Vec<Committed>,
    ids: BTreeSet<String>,
    agent: String,
    /// Bumped by every commit that adds or replaces; a preemptive generation planned against an
    /// older version is discarded.
    version: u64,
}

impl SessionChatCtx {
    pub fn new() -> Self {
        SessionChatCtx::default()
    }

    /// Set the driving frame. `carry`: the frame brought no context, so the previous frame's
    /// conversation becomes its own (a resume after an inline task). Returns the ids carried.
    pub fn set_agent(&mut self, agent: &str, carry: bool) -> Vec<String> {
        let mut carried = Vec::new();
        if carry {
            let from = std::mem::replace(&mut self.agent, agent.to_owned());
            if from == agent {
                return carried;
            }
            for c in &mut self.items {
                if c.agent == from && c.visibility.model() {
                    c.agent = agent.to_owned();
                    carried.push(c.item.id().to_owned());
                }
            }
        } else {
            self.agent = agent.to_owned();
        }
        carried
    }

    pub fn agent(&self) -> &str {
        &self.agent
    }

    pub fn version(&self) -> u64 {
        self.version
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
        let entry = Committed { item, source, visibility, agent: agent.to_owned() };
        if entry.item.id() == INSTRUCTIONS_MESSAGE_ID {
            return (entry, CommitOutcome::Filtered);
        }
        let id = entry.item.id();
        let outcome = if self.ids.insert(id.to_owned()) {
            let idx = self.find_insertion_index(&entry.item);
            self.items.insert(idx, entry.clone());
            CommitOutcome::Added
        } else {
            // Replaced in place under the committing frame; an item in history stays there.
            let idx = self
                .items
                .iter()
                .position(|c| c.item.id() == id)
                .expect("an id in the set is on the record");
            let visibility = self.items[idx].visibility.union(visibility);
            self.items[idx] = Committed { visibility, ..entry.clone() };
            CommitOutcome::Replaced
        };
        (entry, outcome)
    }

    /// `Agent.update_chat_ctx`: `agent`'s view becomes exactly `items` (the version is kept).
    /// Dropped items stay in history; known items are promoted in place; new ones are `HostCtx`,
    /// frame-only.
    pub fn update_agent_chat_ctx(&mut self, items: Vec<ChatItem>, agent: &str) {
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

    /// An inline task hands back with `exclude_function_call`: its calls and outputs leave every
    /// agent view and stay in history. Returns how many were demoted.
    pub fn exclude_task_calls(&mut self, task_agent: &str) -> usize {
        let mut n = 0;
        for c in &mut self.items {
            if c.agent == task_agent
                && c.visibility.model()
                && matches!(c.item, ChatItem::FunctionCall(_) | ChatItem::FunctionCallOutput(_))
            {
                c.visibility = Visibility::History;
                n += 1;
            }
        }
        n
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

    /// A realtime server's mirror: the agent's view minus items the server created and minus frame
    /// records (`_detach_reusable_resources`).
    pub fn for_realtime(&self) -> ChatContext {
        ChatContext::new(
            self.items
                .iter()
                .filter(|c| {
                    c.agent == self.agent
                        && c.visibility.model()
                        && !matches!(c.source, Source::Remote { .. })
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

    /// After the last item stamped at or before this one: ties go after.
    fn find_insertion_index(&self, item: &ChatItem) -> usize {
        let at = item.created_at();
        for (i, c) in self.items.iter().enumerate().rev() {
            if c.item.created_at() <= at {
                return i + 1;
            }
        }
        0
    }
}
