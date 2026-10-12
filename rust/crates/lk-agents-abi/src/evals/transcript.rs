//! `evals/judge.py:60`: a `ChatContext` as the line-per-item transcript every judge prompt embeds.

use lk_agents_abi::{ChatContext, ChatItem, ChatMessage, Role};

/// In list order (not re-sorted).
pub fn format_chat_ctx(chat_ctx: &ChatContext) -> String {
    format_items(&chat_ctx.items)
}

pub fn format_items(items: &[ChatItem]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(items.len());
    for item in items {
        match item {
            ChatItem::Message(m) => {
                let text = text_content(m).unwrap_or_default();
                let role = m.role.as_str();
                if m.interrupted {
                    parts.push(format!("{role}: {text} [interrupted]"));
                } else {
                    parts.push(format!("{role}: {text}"));
                }
            }
            ChatItem::FunctionCall(f) => {
                parts.push(format!("[function call: {}({})]", f.name, f.arguments));
            }
            ChatItem::FunctionCallOutput(o) => {
                if o.is_error {
                    parts.push(format!("[function error: {}]", o.output));
                } else {
                    parts.push(format!("[function output: {}]", o.output));
                }
            }
            ChatItem::AgentHandoff(h) => {
                // `None` on the first assignment prints as the literal `None`.
                let old = h.old_agent_id.as_deref().unwrap_or("None");
                parts.push(format!("[agent handoff: {} -> {}]", old, h.new_agent_id));
            }
            ChatItem::AgentConfigUpdate(c) => {
                let mut config_parts: Vec<String> = Vec::with_capacity(3);
                // `if item.instructions:`: empty is absent.
                if let Some(instructions) = &c.instructions
                    && !instructions.is_empty()
                {
                    config_parts.push(format!("instructions={}", py_repr(instructions)));
                }
                if let Some(added) = &c.tools_added
                    && !added.is_empty()
                {
                    config_parts.push(format!("tools_added={}", py_list_repr(added)));
                }
                if let Some(removed) = &c.tools_removed
                    && !removed.is_empty()
                {
                    config_parts.push(format!("tools_removed={}", py_list_repr(removed)));
                }
                // Emitted even when empty, as in the reference.
                parts.push(format!("[agent config: {}]", config_parts.join(", ")));
            }
        }
    }
    parts.join("\n")
}

/// `<expr/>` stripped from assistant messages: unstripped markup would count as verbosity.
fn text_content(m: &ChatMessage) -> Option<String> {
    let raw = m.raw_text_content()?;
    if m.role != Role::Assistant {
        return Some(raw);
    }
    Some(crate::tts::markup::strip_expr_markup(&raw))
}

/// Keeps multi-line instructions on one line. Unlike CPython, unprintable non-ASCII is not escaped.
pub(crate) fn py_repr(s: &str) -> String {
    // CPython picks `'` unless that would need escaping and `"` would not.
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn py_list_repr(items: &[String]) -> String {
    let inner: Vec<String> = items.iter().map(|s| py_repr(s)).collect();
    format!("[{}]", inner.join(", "))
}

/// `judge.py:92`. System messages are not a source.
pub fn latest_instructions(chat_ctx: &ChatContext) -> Option<&str> {
    chat_ctx.items.iter().rev().find_map(|item| match item {
        ChatItem::AgentConfigUpdate(c) => match c.instructions.as_deref() {
            // `if ... and item.instructions:` -- an empty string does not count.
            Some(s) if !s.is_empty() => Some(s),
            _ => None,
        },
        _ => None,
    })
}

/// `judge.py:103`: excludes the initial assignment every session has.
pub fn has_handoffs(chat_ctx: &ChatContext) -> bool {
    chat_ctx
        .items
        .iter()
        .any(|item| matches!(item, ChatItem::AgentHandoff(h) if h.old_agent_id.is_some()))
}
