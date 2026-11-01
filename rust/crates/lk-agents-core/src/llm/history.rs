//! Pure copy, truncation and merge plans over a conversation's item facts.
use std::collections::BTreeSet;
use std::time::SystemTime;

use lk_agents_abi::{ChatContext, ChatItem, CopyOptions, ItemKind, MergeOptions, Role};

/// What a host that owns its items reports; the rules over them stay here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ItemFacts<'a> {
    pub kind: ItemKind,
    pub role: Option<Role>,
    /// `len(item.content)`: the reference tests the list, so one empty string survives.
    pub content_len: usize,
    pub tool_name: Option<&'a str>,
    /// A function call's or output's `call_id`: an output with no name pairs with its call by it.
    pub call_id: Option<&'a str>,
    pub id: &'a str,
    pub created_at: SystemTime,
}

impl<'a> ItemFacts<'a> {
    pub fn of(item: &'a ChatItem) -> Self {
        ItemFacts {
            kind: item.kind(),
            role: item.as_message().map(|m| m.role),
            content_len: item.as_message().map_or(0, |m| m.content.len()),
            tool_name: item.tool_name(),
            call_id: match item {
                ChatItem::FunctionCall(f) => Some(&f.call_id),
                ChatItem::FunctionCallOutput(f) => Some(&f.call_id),
                _ => None,
            },
            id: item.id(),
            created_at: item.created_at(),
        }
    }

    fn is_instruction(&self) -> bool {
        self.kind == ItemKind::Message && matches!(self.role, Some(Role::System | Role::Developer))
    }

    fn is_function_call_pair(&self) -> bool {
        self.kind.is_function_call_pair()
    }
}

/// `valid_call_ids`: the calls `opts.tools` keeps, which an output with no name is paired by.
pub fn copy_excludes_facts(
    f: &ItemFacts<'_>,
    opts: &CopyOptions,
    valid_call_ids: &BTreeSet<&str>,
) -> bool {
    let is_message = f.kind == ItemKind::Message;

    if opts.exclude_function_call && f.is_function_call_pair() {
        return true;
    }
    // `role in ("system", "developer")`, and a *message* only -- the other kinds have no role.
    if opts.exclude_instructions && f.is_instruction() {
        return true;
    }
    // `not item.content` -- an empty content *list*. A message holding one empty string survives.
    if opts.exclude_empty_message && is_message && f.content_len == 0 {
        return true;
    }
    if opts.exclude_handoff && f.kind == ItemKind::AgentHandoff {
        return true;
    }
    if opts.exclude_config_update && f.kind == ItemKind::AgentConfigUpdate {
        return true;
    }
    if let Some(valid) = &opts.tools {
        let named = |name: &str| valid.iter().any(|t| t == name);
        match f.kind {
            ItemKind::FunctionCall => return !f.tool_name.is_some_and(named),
            // `FunctionCallOutput.name` is optional: one with none goes with its call
            ItemKind::FunctionCallOutput => {
                return match f.tool_name.filter(|n| !n.is_empty()) {
                    Some(name) => !named(name),
                    None => !f.call_id.is_some_and(|c| valid_call_ids.contains(c)),
                };
            }
            _ => {}
        }
    }
    false
}

fn merge_excludes_facts(f: &ItemFacts<'_>, opts: &MergeOptions) -> bool {
    (opts.exclude_function_call && f.is_function_call_pair())
        || (opts.exclude_instructions && f.is_instruction())
        || (opts.exclude_config_update && f.kind == ItemKind::AgentConfigUpdate)
}

// The operations as index plans over the facts, applied by whoever owns the items. Positions, not
// ids, so duplicate ids pick the right object.

pub fn copy_plan(facts: &[ItemFacts<'_>], opts: &CopyOptions) -> Vec<usize> {
    let valid_call_ids: BTreeSet<&str> = match &opts.tools {
        Some(valid) => facts
            .iter()
            .filter(|f| {
                f.kind == ItemKind::FunctionCall
                    && f.tool_name.is_some_and(|n| valid.iter().any(|t| t == n))
            })
            .filter_map(|f| f.call_id)
            .collect(),
        None => BTreeSet::new(),
    };
    (0..facts.len()).filter(|&i| !copy_excludes_facts(&facts[i], opts, &valid_call_ids)).collect()
}

/// `None` when under the limit. Leading function calls are dropped; the first instruction message
/// comes back unless the window holds its id (#4845). Sliced like `self._items[-max_items:]`, but a
/// zero budget keeps no conversational item (only that instruction); a negative `n` drops the
/// first `|n|` (the host refuses one).
pub fn truncate_plan(facts: &[ItemFacts<'_>], max_items: i64) -> Option<Vec<usize>> {
    let n = facts.len();
    if (n as i64) <= max_items {
        return None;
    }
    let instructions = facts.iter().position(ItemFacts::is_instruction);
    let start = match max_items {
        0 => n,
        m if m > 0 => n - m as usize,
        m => m.unsigned_abs().min(n as u64) as usize,
    };
    let mut kept: Vec<usize> = (start..n).collect();
    let lead = kept.iter().take_while(|&&i| facts[i].is_function_call_pair()).count();
    kept.drain(..lead);
    if let Some(ins) = instructions
        && !kept.iter().any(|&i| facts[i].id == facts[ins].id)
    {
        kept.insert(0, ins);
    }
    Some(kept)
}

/// `merge`: `(index, j)` pairs -- insert `theirs[j]` at `index` of the growing list, in order.
/// De-duplicated by id; ties go after.
pub fn merge_plan(
    mine: &[ItemFacts<'_>],
    theirs: &[ItemFacts<'_>],
    opts: &MergeOptions,
) -> Vec<(usize, usize)> {
    let mut existing: BTreeSet<&str> = mine.iter().map(|f| f.id).collect();
    let mut stamps: Vec<SystemTime> = mine.iter().map(|f| f.created_at).collect();
    let mut plan = Vec::new();
    for (j, f) in theirs.iter().enumerate() {
        if merge_excludes_facts(f, opts) || !existing.insert(f.id) {
            continue;
        }
        let idx = stamps.iter().rposition(|&at| at <= f.created_at).map_or(0, |i| i + 1);
        stamps.insert(idx, f.created_at);
        plan.push((idx, j));
    }
    plan
}

/// Conversation selection policy, separate from the ABI's data representation.
pub trait ChatContextExt {
    fn copy(&self, options: &CopyOptions) -> ChatContext;
}

impl ChatContextExt for ChatContext {
    fn copy(&self, options: &CopyOptions) -> ChatContext {
        let facts: Vec<_> = self.items.iter().map(ItemFacts::of).collect();
        ChatContext {
            items: copy_plan(&facts, options)
                .into_iter()
                .map(|index| self.items[index].clone())
                .collect(),
        }
    }
}

#[cfg(test)]
#[path = "history_tests.rs"]
mod tests;
