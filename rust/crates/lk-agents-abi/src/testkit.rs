//! Builders shared by the test modules. Nothing here is compiled into the library.

use std::time::{Duration, SystemTime};

use crate::llm::chat_context::ChatContext;
use crate::llm::chat_context::{
    AgentConfigUpdate, AgentHandoffItem, ChatContent, ChatItem, ChatMessage, FunctionCall,
    FunctionCallOutput, Role,
};

/// A wall stamp `secs` after the epoch. Distinct integers are distinct instants.
pub fn t(secs: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
}

pub fn msg(id: &str, role: Role, text: &str, at: u64) -> ChatItem {
    ChatMessage::text(id, role, text, t(at)).into()
}

/// A message with no content parts at all -- what `exclude_empty_message` targets.
pub fn empty_msg(id: &str, at: u64) -> ChatItem {
    ChatMessage::new(id, Role::Assistant, vec![], t(at)).into()
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

pub fn out(id: &str, name: &str, at: u64) -> ChatItem {
    let mut o = FunctionCallOutput::new(id, format!("call_{id}"), "done", false, t(at));
    o.name = name.into();
    o.into()
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

pub fn config_update(id: &str, at: u64) -> ChatItem {
    AgentConfigUpdate {
        id: id.into(),
        instructions: None,
        tools_added: None,
        tools_removed: None,
        created_at: t(at),
        tools: None,
    }
    .into()
}

pub fn text(s: &str) -> ChatContent {
    ChatContent::Text(s.into())
}

/// Item ids in order -- what nearly every ordering assertion here compares.
pub fn ids(ctx: &ChatContext) -> Vec<&str> {
    ctx.items.iter().map(ChatItem::id).collect()
}
