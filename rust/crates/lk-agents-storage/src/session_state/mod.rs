//! Native conversation state shared by text and voice. Clones share the live view;
//! serializing a snapshot detaches it. One schema serves every execution mode.

use std::collections::BTreeMap;
use std::sync::Arc;

use lk_agents_abi::storage::session::{PersistedSessionState, SessionKind};
use lk_agents_abi::text_session::SessionSnapshot;
use lk_agents_abi::{ChatItem, ChatMessage, Role};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::StorageError;

mod host;
pub mod remote;
#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;
mod views;

#[derive(Clone, Serialize, Deserialize)]
struct Envelope {
    version: u32,
    #[serde(flatten)]
    state: State,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum State {
    Empty,
    Conversation {
        state: Box<SessionSnapshot>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pending_user_turns: Vec<livekit_protocol::agent::chat_context::ChatItem>,
    },
}

#[derive(Clone)]
pub struct SessionState(Arc<Mutex<Envelope>>);

impl SessionState {
    pub(crate) fn from_value(value: Value) -> Result<Self, StorageError> {
        Ok(Self(Arc::new(Mutex::new(decode(value)?))))
    }

    pub(crate) fn assign(&self, other: &Self) {
        let state = other.0.lock().clone();
        *self.0.lock() = state;
    }

    pub(crate) fn root_history(&self) -> Result<Vec<ChatItem>, StorageError> {
        match &self.0.lock().state {
            State::Conversation { state, .. } => state
                .frames
                .get(&state.root_frame_id)
                .map(|frame| frame.history.clone())
                .ok_or_else(|| invalid("session state has no root agent")),
            _ => Ok(Vec::new()),
        }
    }

    pub(crate) fn same_view(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

fn initial_voice(agent: String, history: Vec<ChatItem>) -> SessionSnapshot {
    let session_id = crate::shortuuid("session_");
    let mut state = SessionSnapshot::new(session_id.clone(), session_id, agent);
    let chat_history = dialogue(&history);
    let root = state.frames.get_mut(&state.root_frame_id).expect("new conversations have a root");
    root.history = history;
    root.visible_dialogue = chat_history.clone();
    state.chat_history = chat_history;
    state
}

fn dialogue(history: &[ChatItem]) -> Vec<ChatMessage> {
    history
        .iter()
        .filter_map(|item| match item {
            ChatItem::Message(message) if matches!(message.role, Role::User | Role::Assistant) => {
                Some(message.clone())
            }
            _ => None,
        })
        .collect()
}

fn decode(value: Value) -> Result<Envelope, StorageError> {
    let envelope: Envelope = serde_json::from_value(value).map_err(invalid)?;
    if envelope.version != 1 {
        return Err(invalid("unsupported session state version"));
    }
    Ok(envelope)
}

pub(crate) fn snapshot_value(state: SessionSnapshot) -> Result<Value, StorageError> {
    serde_json::to_value(Envelope {
        version: 1,
        state: State::Conversation { state: Box::new(state), pending_user_turns: Vec::new() },
    })
    .map_err(invalid)
}

/// Update the selected conversation in place, preserving agent identity and remote work.
pub(crate) fn voice_value(
    agent: String,
    history: Vec<ChatItem>,
    current: Option<&Value>,
) -> Result<Value, StorageError> {
    let previous = current.cloned().map(decode).transpose()?;
    let mut state = match previous.as_ref().map(|envelope| &envelope.state) {
        Some(State::Conversation { state, .. }) => state.clone(),
        _ => Box::new(initial_voice(agent.clone(), history.clone())),
    };
    let frame = state
        .frames
        .get_mut(&state.active_frame_id)
        .ok_or_else(|| invalid("session state has no current agent"))?;
    // Live handoffs and VoiceTasks change the speaking role, never the durable local graph.
    // The configured root is established only when the conversation is first created.
    frame.history = history.clone();
    let delivered = dialogue(&history);
    frame.visible_dialogue = delivered.clone();
    for message in delivered {
        match state.chat_history.iter_mut().find(|seen| seen.id == message.id) {
            Some(seen) => *seen = message,
            None => state.chat_history.push(message),
        }
    }
    let pending_user_turns = match previous.map(|envelope| envelope.state) {
        Some(State::Conversation { pending_user_turns, .. }) => pending_user_turns,
        _ => Vec::new(),
    };
    serde_json::to_value(Envelope {
        version: 1,
        state: State::Conversation { state, pending_user_turns },
    })
    .map_err(invalid)
}

pub(crate) fn voice_progress_value(
    agent: String,
    history: Vec<ChatItem>,
    pending: &[ChatMessage],
    current: Option<&Value>,
) -> Result<Value, StorageError> {
    let mut state = voice_value(agent, history, current)?;
    let items: Vec<_> = pending
        .iter()
        .map(|message| {
            lk_agents_abi::llm::chat_context::protocol::encode_chat_item(&ChatItem::Message(
                message.clone(),
            ))
        })
        .collect();
    state["pending_user_turns"] = serde_json::to_value(items).map_err(invalid)?;
    Ok(state)
}

pub(crate) fn pending_voice_inputs(state: &Value) -> Result<Vec<ChatMessage>, StorageError> {
    let Some(value) = state.get("pending_user_turns") else { return Ok(Vec::new()) };
    let encoded: Vec<livekit_protocol::agent::chat_context::ChatItem> =
        serde_json::from_value(value.clone()).map_err(invalid)?;
    encoded
        .into_iter()
        .map(|item| match lk_agents_abi::llm::chat_context::protocol::decode_chat_item(item)? {
            ChatItem::Message(message) if message.role == Role::User => Ok(message),
            _ => Err(invalid("pending voice input must be a user message")),
        })
        .collect()
}

pub(crate) fn agent_name(state: &Value) -> Result<Option<String>, StorageError> {
    Ok(SessionState::from_value(state.clone())?.current_agent()?.map(|agent| agent.agent_name))
}

fn invalid(error: impl std::fmt::Display) -> StorageError {
    StorageError::Invalid(error.to_string())
}
