//! Builders shared by the test modules. Nothing here is compiled into the library.

use std::time::{Duration, SystemTime};

use lk_agents_abi::llm::chat_context::{ChatItem, ChatMessage, FunctionCall, Role};

/// A wall stamp `secs` after the epoch. Distinct integers are distinct instants.
pub fn t(secs: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
}

pub fn msg(id: &str, role: Role, text: &str, at: u64) -> ChatItem {
    ChatMessage::text(id, role, text, t(at)).into()
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

