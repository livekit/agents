//! Live recognition durability, independent of agent speech ownership.
//!
//! Pending user turns remain separate from accepted conversation. One checkpoint is in flight;
//! new observations replace its queued successor. Database work and reconciliation stay in the
//! shared storage crate. Every result (including task death) returns through the session queue.

use std::collections::BTreeMap;

use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::voice::{VoiceConversationBoundary, VoiceConversationProgress};
use lk_agents_abi::{ChatContent, ChatItem, ChatMessage, Role, Ts};

use crate::agent_session::VoiceDriver;
use crate::agent_session::effects::VoiceEffect;
use crate::audio_recognition::TurnId;

#[derive(Default)]
pub(super) struct ConversationSaves {
    state: SaveState,
    last: Option<Version>,
    inputs: BTreeMap<TurnId, ChatMessage>,
    failed: bool,
    dirty: bool,
}

#[derive(PartialEq)]
struct Version {
    history: u64,
    pending: Vec<ChatMessage>,
}

struct Queued {
    progress: VoiceConversationProgress,
    edits: u64,
}

#[derive(Default)]
enum SaveState {
    #[default]
    Idle,
    InFlight {
        sequence: u64,
        edits: u64,
        queued: Option<Box<Queued>>,
    },
}

impl VoiceDriver {
    pub(crate) fn conversation_changed(&mut self) {
        if self.session.storage.enabled && !self.session.storage.conversation.failed {
            self.session.storage.conversation.dirty = true;
        }
    }

    pub(crate) fn forget_user_input(&mut self, turn: TurnId) {
        self.session.storage.conversation.inputs.remove(&turn);
    }
    pub(crate) fn user_input_item_id(&mut self, turn: TurnId) -> String {
        if let Some(message) = self.session.chat_ctx.user_turn_message(turn) {
            return message.id.clone();
        }
        if let Some(message) = self.session.storage.conversation.inputs.get(&turn) {
            return message.id.clone();
        }
        let id = self.session.ids.mint("item");
        if !self.session.storage.enabled {
            return id;
        }
        self.session
            .storage
            .conversation
            .inputs
            .insert(turn, ChatMessage::text(id.clone(), Role::User, "", self.step.wall()));
        id
    }

    pub(crate) fn checkpoint_conversation(&mut self) {
        if !self.session.storage.enabled || self.session.storage.conversation.failed || self.stopped
        {
            return;
        }
        self.session.storage.conversation.dirty = false;
        let mut pending = Vec::new();
        for (turn, started, text) in self.activity.pending_user_inputs() {
            if text.trim().is_empty() || self.session.chat_ctx.user_turn_message(turn).is_some() {
                continue;
            }
            self.user_input_item_id(turn);
            let message = self
                .session
                .storage
                .conversation
                .inputs
                .get_mut(&turn)
                .expect("input identity exists");
            if let Some(started) = started {
                message.created_at = self.step.wall_at(started);
            }
            message.content = vec![ChatContent::Text(text)];
            pending.push(message.clone());
        }
        let version = Version { history: self.session.chat_ctx.view_revision(), pending };
        let previous = self.session.storage.conversation.last.as_ref();
        // Startup publishes the agent handoff before any user evidence or speech exists.
        // Establish that local baseline without acquiring a partition just to store setup.
        if previous.is_none() && version.pending.is_empty() {
            self.session.storage.conversation.last = Some(version);
            return;
        }
        if previous == Some(&version) {
            return;
        }
        let input_changed = previous.is_some_and(|previous| previous.pending != version.pending)
            || !version.pending.is_empty();
        // Existing speech barriers already persist ordinary assistant/tool history. This owner
        // covers recognition changes and accepted conversation with no speech to save it.
        if !input_changed && !self.session.storage.turns.is_empty() {
            self.session.storage.conversation.last = Some(version);
            return;
        }
        let sequence = self.session.storage.next_sequence();
        let boundary = if version.pending.is_empty() {
            VoiceConversationBoundary::UserTurnCompleted
        } else {
            VoiceConversationBoundary::Progress
        };
        let progress = VoiceConversationProgress {
            sequence,
            agent: self.activity.agent_name().to_owned(),
            history: self.session.chat_ctx.items().cloned().collect(),
            pending_user_turns: version.pending.clone(),
            boundary,
        };
        let edits = self.session.chat_ctx.saving();
        self.session.storage.conversation.last = Some(version);
        match &mut self.session.storage.conversation.state {
            SaveState::Idle => {
                self.session.storage.conversation.state =
                    SaveState::InFlight { sequence, edits, queued: None };
                self.record_effect(VoiceEffect::SaveConversation(Box::new(progress)));
            }
            SaveState::InFlight { queued, .. } => {
                *queued = Some(Box::new(Queued { progress, edits }))
            }
        }
    }

    pub(crate) fn conversation_save_pending(&self) -> bool {
        self.session.storage.enabled
            && !self.session.storage.conversation.failed
            && (self.session.storage.conversation.dirty
                || matches!(self.session.storage.conversation.state, SaveState::InFlight { .. }))
    }

    pub(crate) fn conversation_saved(
        &mut self,
        now: Ts,
        sequence: u64,
        read: Vec<ChatItem>,
        result: Result<Vec<ChatItem>, StorageError>,
    ) {
        if !matches!(self.session.storage.conversation.state, SaveState::InFlight { sequence: held, .. } if held == sequence)
        {
            return;
        }
        let SaveState::InFlight { edits, queued, .. } =
            std::mem::take(&mut self.session.storage.conversation.state)
        else {
            return;
        };
        match result {
            Ok(history) => {
                self.session.chat_ctx.saved(edits);
                self.storage_reloaded(read, history);
                if let Some(queued) = queued {
                    let Queued { progress, edits } = *queued;
                    let sequence = progress.sequence;
                    self.session.storage.conversation.state =
                        SaveState::InFlight { sequence, edits, queued: None };
                    self.record_effect(VoiceEffect::SaveConversation(Box::new(progress)));
                }
            }
            Err(error) => {
                self.session.storage.conversation.failed = true;
                let error = crate::events::ErrorEvent::provider(
                    lk_agents_abi::voice::events::ErrorSource::Storage,
                    false,
                    lk_agents_abi::errors::ProviderError::Local(error.into()),
                    None,
                );
                self.begin_closing(
                    now,
                    crate::agent_session::commands::Close {
                        reason: lk_agents_abi::voice::events::CloseReason::Error,
                        drain: false,
                        error: Some(error),
                    },
                );
            }
        }
    }
}
