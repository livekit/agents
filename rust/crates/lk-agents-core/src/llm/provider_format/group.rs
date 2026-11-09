use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::{ChatContext, ChatItem, ChatMessage, FunctionCall, FunctionCallOutput, Role};

use crate::llm::provider_format::FormatError;

pub use lk_agents_abi::llm::provider_format::host::DEFAULT_INLINE_INSTRUCTIONS_TEMPLATE;

// The two rewrites every dialect runs before it formats; neither clones the context.

/// A system/developer message is the preamble only when it is the first one and no message or
/// tool call precedes it. Every other one is rewritten as a `role` message wrapped in `template`,
/// in place, and dropped when it has no text. `{content}` only, not full `str.format`.
pub fn convert_mid_conversation_instructions<'a>(
    chat_ctx: &'a ChatContext,
    role: Role,
    template: &str,
) -> Vec<Cow<'a, ChatItem>> {
    let mut preamble_allowed = true;
    let mut items: Vec<Cow<'a, ChatItem>> = Vec::with_capacity(chat_ctx.items.len());

    for item in &chat_ctx.items {
        if let ChatItem::Message(message) = item
            && item.is_instruction()
        {
            if preamble_allowed {
                preamble_allowed = false;
                items.push(Cow::Borrowed(item));
            } else if let Some(text) = message.raw_text_content().filter(|t| !t.is_empty()) {
                items.push(Cow::Owned(ChatItem::Message(ChatMessage::text(
                    message.id.clone(),
                    role,
                    template.replace("{content}", &text),
                    message.created_at,
                ))));
            }
            continue;
        }
        if matches!(
            item,
            ChatItem::Message(_) | ChatItem::FunctionCall(_) | ChatItem::FunctionCallOutput(_)
        ) {
            preamble_allowed = false;
        }
        items.push(Cow::Borrowed(item));
    }

    items
}

/// One assistant turn: at most one message, its parallel tool calls, and the matching results.
#[derive(Debug, Default, Clone)]
pub struct ChatItemGroup<'a> {
    pub message: Option<&'a ChatMessage>,
    pub tool_calls: Vec<&'a FunctionCall>,
    pub tool_outputs: Vec<&'a FunctionCallOutput>,
}

#[derive(Debug, Clone, Copy)]
pub enum GroupItem<'a> {
    Message(&'a ChatMessage),
    Call(&'a FunctionCall),
    Output(&'a FunctionCallOutput),
}

impl<'a> ChatItemGroup<'a> {
    /// `utils.py:161`: message, then calls, then outputs, as Anthropic, Bedrock and Gemini require.
    pub fn flatten(&self) -> impl Iterator<Item = GroupItem<'a>> + '_ {
        self.message
            .map(GroupItem::Message)
            .into_iter()
            .chain(self.tool_calls.iter().copied().map(GroupItem::Call))
            .chain(self.tool_outputs.iter().copied().map(GroupItem::Output))
    }

    /// Empty for an `agent_handoff`/`agent_config_update` key: the reference drops those items.
    pub fn is_empty(&self) -> bool {
        self.message.is_none() && self.tool_calls.is_empty() && self.tool_outputs.is_empty()
    }

    /// `utils.py:117`'s assert; reachable when two messages share a `/`-prefix.
    fn set_assistant_message(
        &mut self,
        message: &'a ChatMessage,
        group_id: &str,
    ) -> Result<(), FormatError> {
        if self.message.is_some() {
            return Err(FormatError::MultipleMessagesInGroup { group_id: group_id.to_string() });
        }
        self.message = Some(message);
        Ok(())
    }
}

/// The binding logs the reference's warnings (`utils.py:99`, `:140`, `:152`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dropped {
    pub reason: DropReason,
    pub call_id: String,
    pub tool_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropReason {
    /// "function output missing the corresponding function call, ignoring"
    OutputWithoutCall,
    /// "function call missing the corresponding function output, ignoring"
    CallWithoutOutput,
}

#[derive(Debug, Default, Clone)]
pub struct Grouping<'a> {
    pub groups: Vec<ChatItemGroup<'a>>,
    pub dropped: Vec<Dropped>,
}

/// `utils.py:54`. Messages and calls group by `group_id` when set (#4170), else by the `id` before
/// the first `/`; outputs attach by `call_id`.
pub fn group_tool_calls<'a>(
    items: impl IntoIterator<Item = &'a ChatItem>,
) -> Result<Grouping<'a>, FormatError> {
    let mut keys: Vec<ChatItemGroup<'a>> = Vec::new();
    // `item_groups` is an `OrderedDict`: re-assigning keeps position, and a non-groupable item
    // whose `id` collides replaces that group. An index over a `Vec` reproduces both.
    let mut index: BTreeMap<&'a str, usize> = BTreeMap::new();
    let mut tool_outputs: Vec<&'a FunctionCallOutput> = Vec::new();
    let mut dropped: Vec<Dropped> = Vec::new();

    for item in items {
        match item {
            ChatItem::Message(m) if m.role == Role::Assistant => {
                // The `/`-prefix key puts `item_7` with `item_7/0`, `item_7/1`.
                let group_id = m.id.split('/').next().unwrap_or(&m.id);
                upsert(&mut keys, &mut index, group_id).set_assistant_message(m, group_id)?;
            }
            ChatItem::FunctionCall(f) => {
                let group_id = match &f.group_id {
                    Some(g) if !g.is_empty() => g.as_str(),
                    _ => f.id.split('/').next().unwrap_or(&f.id),
                };
                upsert(&mut keys, &mut index, group_id).tool_calls.push(f);
            }
            ChatItem::FunctionCallOutput(o) => tool_outputs.push(o),
            other => {
                // An assignment: a user message reusing a group's key wipes that group.
                let id = other.id();
                let slot = upsert(&mut keys, &mut index, id);
                *slot = ChatItemGroup::default();
                if let ChatItem::Message(m) = other {
                    slot.message = Some(m);
                }
            }
        }
    }

    attach_outputs(&mut keys, tool_outputs, &mut dropped);
    for group in &mut keys {
        remove_invalid_tool_calls(group, &mut dropped);
    }

    Ok(Grouping { groups: keys, dropped })
}

/// Each output joins the group holding its call; one without a call is dropped.
fn attach_outputs<'a>(
    groups: &mut [ChatItemGroup<'a>],
    outputs: Vec<&'a FunctionCallOutput>,
    dropped: &mut Vec<Dropped>,
) {
    // One flat `call_id -> group` map first, so with a shared `call_id` the last group wins.
    let mut call_id_to_group: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, group) in groups.iter().enumerate() {
        for call in &group.tool_calls {
            call_id_to_group.insert(call.call_id.as_str(), i);
        }
    }
    for output in outputs {
        match call_id_to_group.get(output.call_id.as_str()) {
            Some(&i) => groups[i].tool_outputs.push(output),
            None => dropped.push(Dropped {
                reason: DropReason::OutputWithoutCall,
                call_id: output.call_id.clone(),
                tool_name: output.name.clone(),
            }),
        }
    }
}

fn upsert<'g, 'a>(
    keys: &'g mut Vec<ChatItemGroup<'a>>,
    index: &mut BTreeMap<&'a str, usize>,
    id: &'a str,
) -> &'g mut ChatItemGroup<'a> {
    let i = *index.entry(id).or_insert_with(|| {
        keys.push(ChatItemGroup::default());
        keys.len() - 1
    });
    &mut keys[i]
}

/// `utils.py:126`. Keep the equal-length early return: a balanced but mismatched group is left as
/// it is.
fn remove_invalid_tool_calls(group: &mut ChatItemGroup<'_>, dropped: &mut Vec<Dropped>) {
    if group.tool_calls.len() == group.tool_outputs.len() {
        return;
    }

    let call_ids: BTreeSet<&str> = group.tool_calls.iter().map(|c| c.call_id.as_str()).collect();
    let output_ids: BTreeSet<&str> =
        group.tool_outputs.iter().map(|o| o.call_id.as_str()).collect();

    group.tool_calls.retain(|c| {
        let keep = output_ids.contains(c.call_id.as_str());
        if !keep {
            dropped.push(Dropped {
                reason: DropReason::CallWithoutOutput,
                call_id: c.call_id.clone(),
                tool_name: c.name.clone(),
            });
        }
        keep
    });
    group.tool_outputs.retain(|o| {
        let keep = call_ids.contains(o.call_id.as_str());
        if !keep {
            dropped.push(Dropped {
                reason: DropReason::OutputWithoutCall,
                call_id: o.call_id.clone(),
                tool_name: o.name.clone(),
            });
        }
        keep
    });
}

#[cfg(test)]
#[path = "group_tests.rs"]
mod tests;
