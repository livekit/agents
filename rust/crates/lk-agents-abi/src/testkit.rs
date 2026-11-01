//! Builders shared by the test modules. Nothing here is compiled into the library.

use std::time::{Duration, SystemTime};

use crate::llm::chat_context::{AgentHandoffItem, ChatContent, ChatItem, FunctionCall};

/// A wall stamp `secs` after the epoch. Distinct integers are distinct instants.
pub fn t(secs: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
}

pub fn call(id: &str, name: &str, at: u64) -> ChatItem {
    FunctionCall {
        id: id.into(),
        call_id: format!("call_{id}"),
        arguments: "{}".into(),
        name: name.into(),
        created_at: t(at),
        extra: Default::default(),
        group_id: None,
    }
    .into()
}

pub fn handoff(id: &str, at: u64) -> ChatItem {
    AgentHandoffItem {
        id: id.into(),
        old_agent_id: None,
        new_agent_id: "b".into(),
        created_at: t(at),
    }
    .into()
}

pub fn text(s: &str) -> ChatContent {
    ChatContent::Text(s.into())
}
