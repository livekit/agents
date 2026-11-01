//! A conversation as the one-line-per-item transcript every judge prompt embeds.

use lk_agents_abi::{ChatItem, ChatMessage, CopyOptions, Role};
use lk_agents_core::llm::history::ChatContextExt;
use lk_agents_core::text::py_repr;

/// The items in list order, one line each.
pub fn format_items(items: &[ChatItem]) -> String {
    let mut lines = Vec::with_capacity(items.len());
    for item in items {
        lines.push(match item {
            ChatItem::Message(m) => {
                let text = text_content(m).unwrap_or_default();
                let role = m.role.as_str();
                if m.interrupted {
                    format!("{role}: {text} [interrupted]")
                } else {
                    format!("{role}: {text}")
                }
            }
            ChatItem::FunctionCall(f) => format!("[function call: {}({})]", f.name, f.arguments),
            ChatItem::FunctionCallOutput(o) if o.is_error => {
                format!("[function error: {}]", o.output)
            }
            ChatItem::FunctionCallOutput(o) => format!("[function output: {}]", o.output),
            ChatItem::AgentHandoff(h) => {
                let old = h.old_agent_id.as_deref().unwrap_or("None");
                format!("[agent handoff: {old} -> {}]", h.new_agent_id)
            }
            ChatItem::AgentConfigUpdate(c) => {
                let mut fields = Vec::with_capacity(3);
                if let Some(instructions) = c.instructions.as_deref().filter(|s| !s.is_empty()) {
                    fields.push(format!("instructions={}", py_repr(instructions)));
                }
                if let Some(added) = c.tools_added.as_deref().filter(|t| !t.is_empty()) {
                    fields.push(format!("tools_added={}", py_list_repr(added)));
                }
                if let Some(removed) = c.tools_removed.as_deref().filter(|t| !t.is_empty()) {
                    fields.push(format!("tools_removed={}", py_list_repr(removed)));
                }
                format!("[agent config: {}]", fields.join(", "))
            }
        });
    }
    lines.join("\n")
}

/// A reference conversation without its instructions: the judge never sees two sets.
pub fn reference_transcript(reference: &[ChatItem]) -> String {
    let ctx = lk_agents_abi::ChatContext::new(reference.to_vec());
    format_items(&ctx.copy(&CopyOptions { exclude_instructions: true, ..Default::default() }).items)
}

/// The newest non-empty instructions of an agent config update.
pub fn latest_instructions(items: &[ChatItem]) -> Option<&str> {
    items.iter().rev().find_map(|item| match item {
        ChatItem::AgentConfigUpdate(c) => c.instructions.as_deref().filter(|s| !s.is_empty()),
        _ => None,
    })
}

/// Whether one agent handed off to another; the session's first assignment is not a handoff.
pub fn has_handoffs(items: &[ChatItem]) -> bool {
    items.iter().any(|item| matches!(item, ChatItem::AgentHandoff(h) if h.old_agent_id.is_some()))
}

/// Assistant text without its expressive markup, which would read as verbosity.
fn text_content(m: &ChatMessage) -> Option<String> {
    let raw = m.raw_text_content()?;
    if m.role != Role::Assistant {
        return Some(raw);
    }
    Some(lk_agents_core::tts::markup::strip_expr_markup(&raw))
}

fn py_list_repr(items: &[String]) -> String {
    let inner: Vec<String> = items.iter().map(|s| py_repr(s)).collect();
    format!("[{}]", inner.join(", "))
}
