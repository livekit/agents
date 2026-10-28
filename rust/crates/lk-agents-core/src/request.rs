//! Request planning (rust-core "Agent execution", request planning): one model request built from
//! an explicit snapshot of the allowed history, the instructions and the tools. Pure; no session
//! state is read.

use std::time::SystemTime;

use lk_agents_abi::{ChatContext, ChatItem, ChatMessage, Role, ToolSpec};

use crate::approval;

/// The instructions message's id; rendered per request, never part of the history.
pub const INSTRUCTIONS_MESSAGE_ID: &str = "lk.agent_task.instructions";

/// What one request is planned from.
#[derive(Debug, Clone)]
pub struct RequestSnapshot<'a> {
    pub history: &'a [ChatItem],
    /// The instructions for this request, static and rendered dynamic ones joined.
    pub instructions: &'a str,
    pub tools: &'a [ToolSpec],
    /// The request is made inside the agent's `on_enter`: `IGNORE_ON_ENTER` tools are withheld.
    pub on_enter: bool,
    /// A system message for this request alone, after the history.
    pub turn_instructions: Option<&'a str>,
    /// The wall stamp of the messages the plan creates.
    pub created_at: SystemTime,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlannedRequest {
    pub chat_ctx: ChatContext,
    pub tools: Vec<ToolSpec>,
}

pub fn plan_request(snapshot: RequestSnapshot<'_>) -> PlannedRequest {
    let mut items = Vec::with_capacity(snapshot.history.len() + 2);
    if !snapshot.instructions.is_empty() {
        items.push(ChatItem::Message(ChatMessage::text(
            INSTRUCTIONS_MESSAGE_ID,
            Role::System,
            snapshot.instructions,
            snapshot.created_at,
        )));
    }
    items.extend(model_view(snapshot.history));
    if let Some(text) = snapshot.turn_instructions {
        let id = format!("{INSTRUCTIONS_MESSAGE_ID}.turn");
        items.push(ChatItem::Message(ChatMessage::text(
            id,
            Role::System,
            text,
            snapshot.created_at,
        )));
    }
    let tools = snapshot
        .tools
        .iter()
        .filter(|t| !(snapshot.on_enter && t.flags.ignore_on_enter))
        .cloned()
        .collect();
    PlannedRequest { chat_ctx: ChatContext::new(items), tools }
}

/// The history as the model reads it: no stored instructions; a call without an output yet is
/// left out (a request cannot carry it); an answered read-back says the call has not run and how
/// to run it, while the stored output stays the person's words.
pub fn model_view(history: &[ChatItem]) -> Vec<ChatItem> {
    let answered = |call_id: &str| {
        history.iter().any(|i| matches!(i, ChatItem::FunctionCallOutput(o) if o.call_id == call_id))
    };
    let asked = |call_id: &str| {
        history.iter().find_map(|i| match i {
            ChatItem::FunctionCall(c)
                if c.call_id == call_id
                    && approval::mark(c)
                        == Some(lk_agents_abi::llm::tool_executor::Approval::Asked) =>
            {
                Some(c.name.as_str())
            }
            _ => None,
        })
    };
    history
        .iter()
        .filter(|item| item.id() != INSTRUCTIONS_MESSAGE_ID)
        .filter(|item| !matches!(item, ChatItem::FunctionCall(c) if !answered(&c.call_id)))
        .map(|item| match item {
            ChatItem::FunctionCallOutput(o)
                if !o.output.starts_with(lk_agents_abi::llm::tool_executor::WAITING_PREFIX) =>
            {
                match asked(&o.call_id) {
                    Some(tool) => {
                        let mut read = o.clone();
                        read.output = approval::answered_text(tool, &o.output);
                        ChatItem::FunctionCallOutput(read)
                    }
                    None => item.clone(),
                }
            }
            _ => item.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use lk_agents_abi::llm::llm::ToolFlags;
    use serde_json::json;

    use super::*;

    // Catches (decision 42): `IGNORE_ON_ENTER` withholding a tool outside `on_enter`, or offering it
    // inside; stored instructions reaching the model twice.
    #[test]
    fn ignore_on_enter_tools_are_withheld_only_inside_on_enter() {
        let tool = |name: &str, ignore_on_enter| ToolSpec {
            provider_tool: None,
            name: name.into(),
            description: String::new(),
            json_schema: json!({"type": "object"}),
            flags: ToolFlags { ignore_on_enter },
        };
        let tools = [tool("lookup", false), tool("collect_dob", true)];
        let stale = ChatItem::Message(ChatMessage::text(
            INSTRUCTIONS_MESSAGE_ID,
            Role::System,
            "old",
            SystemTime::UNIX_EPOCH,
        ));
        let history = [stale];
        let plan = |on_enter| {
            plan_request(RequestSnapshot {
                history: &history,
                instructions: "new",
                tools: &tools,
                on_enter,
                turn_instructions: None,
                created_at: SystemTime::UNIX_EPOCH,
            })
        };
        let names = |p: &PlannedRequest| p.tools.iter().map(|t| t.name.clone()).collect::<Vec<_>>();
        assert_eq!(names(&plan(false)), ["lookup", "collect_dob"]);
        assert_eq!(names(&plan(true)), ["lookup"]);
        let texts: Vec<_> = plan(false)
            .chat_ctx
            .items
            .iter()
            .map(|i| i.as_message().and_then(|m| m.raw_text_content()))
            .collect();
        assert_eq!(texts, [Some("new".to_owned())]);
    }
}
