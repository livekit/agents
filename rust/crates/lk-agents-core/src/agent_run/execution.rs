//! Conversational identity travels with saved history, independently of invocation ids.

use lk_agents_abi::{ChatItem, ChatMessage, Role};
use serde::{Deserialize, Serialize};

const EXECUTION_KEY: &str = "lk.execution";

#[derive(Serialize, Deserialize)]
struct Execution {
    id: String,
    pending: bool,
}

fn stored(message: &ChatMessage) -> Option<Execution> {
    serde_json::from_value(message.extra.get(EXECUTION_KEY)?.clone()).ok()
}

/// The pending conversational identity recorded by the latest user turn.
pub fn pending(history: &[ChatItem]) -> Option<String> {
    history
        .iter()
        .rev()
        .find_map(|item| match item {
            ChatItem::Message(message) if message.role == Role::User => Some(message),
            _ => None,
        })
        .and_then(stored)
        .filter(|state| state.pending)
        .map(|state| state.id)
}

/// The transport owns A2A identity. A direct human reply continues the last pending execution;
/// a completed conversation or an unrelated history starts at the supplied invocation id.
pub fn bind(
    message: &mut ChatMessage,
    history: &[ChatItem],
    transport_id: Option<&str>,
    invocation_id: &str,
) -> String {
    let execution = transport_id
        .map(str::to_owned)
        .or_else(|| stored(message).map(|state| state.id))
        .or_else(|| pending(history))
        .unwrap_or_else(|| invocation_id.to_owned());
    message.extra.insert(
        EXECUTION_KEY.into(),
        serde_json::to_value(Execution { id: execution.clone(), pending: false })
            .expect("execution identity contains only a string and boolean"),
    );
    execution
}

pub(super) fn settle(message: &mut ChatMessage, pending: bool) {
    if let Some(mut state) = stored(message) {
        state.pending = pending;
        message.extra.insert(
            EXECUTION_KEY.into(),
            serde_json::to_value(state)
                .expect("execution identity contains only a string and boolean"),
        );
    }
}
