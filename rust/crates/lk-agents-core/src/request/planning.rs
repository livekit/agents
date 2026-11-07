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
pub struct ModelRequest {
    pub chat_ctx: ChatContext,
    pub tools: Vec<ToolSpec>,
}

pub fn plan_request(snapshot: RequestSnapshot<'_>) -> ModelRequest {
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
    ModelRequest { chat_ctx: ChatContext::new(items), tools }
}

/// Marks the call whose question the run's outcome selected. Selection does not prove delivery
/// or audibility. Other waiting calls remain pending until a coordinator explicitly selects them.
pub const INPUT_REQUESTED_KEY: &str = "lk.input_requested";

/// Answer only the selected call, never every waiting output in a tool batch.
pub fn answer_pending_input(
    history: &mut [ChatItem],
    answer: &str,
) -> std::collections::BTreeSet<String> {
    let marked: std::collections::BTreeSet<_> = history
        .iter()
        .filter_map(|item| match item {
            ChatItem::FunctionCall(call) if selected(call) => Some(call.call_id.clone()),
            _ => None,
        })
        .collect();
    let waiting: Vec<_> = history
        .iter()
        .filter_map(|item| match item {
            ChatItem::FunctionCallOutput(output)
                if output.output.starts_with(lk_agents_abi::llm::tool_executor::WAITING_PREFIX) =>
            {
                Some(output.call_id.clone())
            }
            _ => None,
        })
        .collect();
    let selected = waiting.iter().find(|call| marked.contains(*call)).cloned();
    let mut answered = std::collections::BTreeSet::new();
    for item in history {
        if let ChatItem::FunctionCallOutput(output) = item
            && Some(&output.call_id) == selected.as_ref()
            && output.output.starts_with(lk_agents_abi::llm::tool_executor::WAITING_PREFIX)
        {
            output.output = answer.to_owned();
            answered.insert(output.call_id.clone());
        }
    }
    answered
}

/// Recover an approval answer written by a transport before the invocation starts: a selected
/// read-back whose output holds an answer, not the waiting placeholder. A retained sibling
/// question is not answered merely because it belongs to the same turn.
pub(crate) fn answered_approval_calls(history: &[ChatItem]) -> std::collections::BTreeSet<String> {
    history
        .iter()
        .filter_map(|item| match item {
            ChatItem::FunctionCall(call)
                if selected(call)
                    && approval::mark(call)
                        == Some(lk_agents_abi::llm::tool_executor::Approval::Asked) =>
            {
                Some(call)
            }
            _ => None,
        })
        .filter(|call| {
            history.iter().any(|item| {
                matches!(item, ChatItem::FunctionCallOutput(output)
                    if output.call_id == call.call_id
                    && !output.output.starts_with(lk_agents_abi::llm::tool_executor::WAITING_PREFIX))
            })
        })
        .map(|call| call.call_id.clone())
        .collect()
}

fn selected(call: &lk_agents_abi::FunctionCall) -> bool {
    call.extra
        .get(INPUT_REQUESTED_KEY)
        .is_some_and(|value| value.as_bool() == Some(true) || value.as_str() == Some("true"))
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
    use lk_agents_abi::llm::types::ToolFlags;
    use serde_json::json;

    use super::*;

    // Catches: `IGNORE_ON_ENTER` withholding a tool outside `on_enter`, or offering it
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
        let names = |p: &ModelRequest| p.tools.iter().map(|t| t.name.clone()).collect::<Vec<_>>();
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
