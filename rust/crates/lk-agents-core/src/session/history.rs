//! Projection of live voice history into the shared conversation, without a saved voice role.

use lk_agents_abi::errors::Invalid;
use lk_agents_abi::session::snapshot::SessionSnapshot;
use lk_agents_abi::{ChatItem, Role};

pub fn voice(
    state: Option<SessionSnapshot>,
    session_id: String,
    agent: String,
    history: Vec<ChatItem>,
) -> Result<SessionSnapshot, Invalid> {
    let mut state = state
        .unwrap_or_else(|| SessionSnapshot::new(session_id.clone(), session_id, agent.clone()));
    super::resume::validate_voice(&state, &agent)?;
    let root = state
        .frames
        .get_mut(&state.root_frame_id)
        .ok_or_else(|| Invalid("the conversation has no root agent".into()))?;
    let delivered: Vec<_> = history
        .iter()
        .filter_map(|item| match item {
            ChatItem::Message(message) if matches!(message.role, Role::User | Role::Assistant) => {
                Some(message.clone())
            }
            _ => None,
        })
        .collect();
    root.history = history;
    root.visible_dialogue = delivered.clone();
    for message in delivered {
        match state.chat_history.iter_mut().find(|seen| seen.id == message.id) {
            Some(seen) => *seen = message,
            None => state.chat_history.push(message),
        }
    }
    Ok(state)
}
