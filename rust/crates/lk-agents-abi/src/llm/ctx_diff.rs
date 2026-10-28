//! Minimal edit script between two `ChatContext`s (`llm/utils.py`): a realtime server can only be
//! told "insert this after that" or "delete this".

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::collections::{HashMap, HashSet};

use crate::{ChatContext, ChatItem};

/// `(previous_item_id, id)`; `None` is first in the conversation.
pub type Anchored = (Option<String>, String);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DiffOps {
    pub to_remove: Vec<String>,
    pub to_create: Vec<Anchored>,
    /// Patched in place, keeping the item's server-side position.
    pub to_update: Vec<Anchored>,
}

/// Ids only, never content, so an edited message is an update. Stable item ids turn the LCS
/// into a longest increasing subsequence of their positions in the new context.
fn lcs(old: &[&str], new: &[&str]) -> Vec<String> {
    let prefix = old.iter().zip(new).take_while(|(a, b)| a == b).count();
    let mut out: Vec<String> = old[..prefix].iter().map(|id| (*id).to_owned()).collect();
    let (old, new) = (&old[prefix..], &new[prefix..]);
    if old.is_empty() || new.is_empty() {
        return out;
    }

    let mut positions = std::collections::BTreeMap::new();
    for (index, id) in new.iter().enumerate() {
        if positions.insert(*id, index).is_some() {
            // Host-authored contexts can repeat ids. Keep their established diff semantics too,
            // without allocating a quadratic table for that exceptional input.
            lcs_repeated_ids(old, new, &mut out);
            return out;
        }
    }

    let mut tails = Vec::<usize>::new();
    let mut tail_items = Vec::<usize>::new();
    let mut previous = vec![None; old.len()];
    for (index, id) in old.iter().enumerate() {
        let Some(&position) = positions.get(id) else { continue };
        let length = tails.partition_point(|&tail| tail < position);
        previous[index] = length.checked_sub(1).map(|i| tail_items[i]);
        if length == tails.len() {
            tails.push(position);
            tail_items.push(index);
        } else {
            // Scanning old items and replacing equal-length tails keeps the old backtracking
            // tie rule: discard from the new side when both choices have the same length.
            tails[length] = position;
            tail_items[length] = index;
        }
    }
    let mut suffix = Vec::with_capacity(tails.len());
    let mut cursor = tail_items.last().copied();
    while let Some(index) = cursor {
        suffix.push(old[index].to_owned());
        cursor = previous[index];
    }
    out.extend(suffix.into_iter().rev());
    out
}

/// Hirschberg's linear-space fallback for repeated new ids. The normal, uniquely keyed path
/// above is O((n + m) log m); this fallback is O(nm) time and O(n + m) memory.
fn lcs_repeated_ids(old: &[&str], new: &[&str], out: &mut Vec<String>) {
    if old.is_empty() || new.is_empty() {
        return;
    }
    if old.len() == 1 {
        if new.contains(&old[0]) {
            out.push(old[0].to_owned());
        }
        return;
    }
    let middle = old.len() / 2;
    let left = lcs_lengths(&old[..middle], new, false);
    let right = lcs_lengths(&old[middle..], new, true);
    let mut split = 0;
    let mut longest = 0;
    for index in 0..=new.len() {
        let length = left[index] + right[new.len() - index];
        if length > longest {
            longest = length;
            split = index;
        }
    }
    // Free rows before recursing so live scratch space remains linear.
    drop(left);
    drop(right);
    lcs_repeated_ids(&old[..middle], &new[..split], out);
    lcs_repeated_ids(&old[middle..], &new[split..], out);
}

fn lcs_lengths(old: &[&str], new: &[&str], reverse: bool) -> Vec<usize> {
    let mut lengths = vec![0; new.len() + 1];
    for i in 0..old.len() {
        let old_id = old[if reverse { old.len() - 1 - i } else { i }];
        let mut diagonal = 0;
        for j in 0..new.len() {
            let new_id = new[if reverse { new.len() - 1 - j } else { j }];
            let previous = lengths[j + 1];
            lengths[j + 1] = if old_id == new_id { diagonal + 1 } else { lengths[j].max(previous) };
            diagonal = previous;
        }
    }
    lengths
}

/// `compute_chat_ctx_diff` (`llm/utils.py:120`) over the host's items, as
/// `(to_remove, to_create, to_update)`.
#[crate::host(export(name = "compute_chat_ctx_diff"))]
pub fn compute_chat_ctx_diff_of(
    old: Vec<crate::ChatItem>,
    new: Vec<crate::ChatItem>,
) -> (Vec<String>, Vec<Anchored>, Vec<Anchored>) {
    let ops = compute_chat_ctx_diff(&ChatContext::new(old), &ChatContext::new(new));
    (ops.to_remove, ops.to_create, ops.to_update)
}

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
pub fn compute_chat_ctx_diff(old: &ChatContext, new: &ChatContext) -> DiffOps {
    let mut old_by_id: HashMap<&str, &crate::ChatItem> =
        old.items.iter().map(|item| (item.id(), item)).collect();
    if old_by_id.len() == old.len()
        && old.items.iter().zip(&new.items).all(|(before, after)| before.id() == after.id())
    {
        // Reuse the existing id index to validate the appended suffix. A duplicate restores the
        // old-only index before falling back to the historical set semantics.
        let unique = new
            .items
            .iter()
            .skip(old.len())
            .all(|item| old_by_id.insert(item.id(), item).is_none());
        if unique {
            return prefix_diff(old, new);
        }
        old_by_id = old.items.iter().map(|item| (item.id(), item)).collect();
    }
    let old_ids: Vec<&str> = old.items.iter().map(|i| i.id()).collect();
    let new_ids: Vec<&str> = new.items.iter().map(|i| i.id()).collect();
    let kept: HashSet<String> = lcs(&old_ids, &new_ids).into_iter().collect();

    let to_remove =
        old.items.iter().filter(|i| !kept.contains(i.id())).map(|i| i.id().to_owned()).collect();

    let (mut to_create, mut to_update) = (Vec::new(), Vec::new());
    let mut prev: Option<String> = None;

    for item in &new.items {
        if !kept.contains(item.id()) {
            to_create.push((prev.clone(), item.id().to_owned()));
        } else if let Some(old_item) = old_by_id.get(item.id())
            && message_changed(old_item, item)
        {
            to_update.push((prev.clone(), item.id().to_owned()));
        }
        // Advances for created items too, so a run of new items chains in order.
        prev = Some(item.id().to_owned());
    }

    DiffOps { to_remove, to_create, to_update }
}

/// Appending or trimming a context whose ids already line up needs no retained subsequence or
/// owned id index. Duplicate host ids keep the general path's historical set semantics.
fn prefix_diff(old: &ChatContext, new: &ChatContext) -> DiffOps {
    let shared = old.len().min(new.len());
    let mut diff = DiffOps::default();
    for (index, (before, after)) in old.items.iter().zip(&new.items).enumerate() {
        if message_changed(before, after) {
            diff.to_update.push((
                index.checked_sub(1).map(|prev| new.items[prev].id().to_owned()),
                after.id().to_owned(),
            ));
        }
    }
    diff.to_remove.extend(old.items[shared..].iter().map(|item| item.id().to_owned()));
    diff.to_create.extend(new.items.iter().enumerate().skip(shared).map(|(index, item)| {
        (index.checked_sub(1).map(|prev| new.items[prev].id().to_owned()), item.id().to_owned())
    }));
    diff
}

fn message_changed(old: &ChatItem, new: &ChatItem) -> bool {
    use crate::ChatContent;
    let (Some(old), Some(new)) = (old.as_message(), new.as_message()) else {
        return false;
    };
    fn text(content: &ChatContent) -> Option<&str> {
        match content {
            ChatContent::Text(text) => Some(text.as_str()),
            _ => None,
        }
    }
    // Equal parts need no joined String; differing boundaries may still join to the same text.
    if old.content.iter().filter_map(text).eq(new.content.iter().filter_map(text)) {
        return false;
    }
    old.raw_text_content() != new.raw_text_content()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::chat_context::Role;
    use crate::testkit::msg;

    fn ctx(items: &[(&str, &str)]) -> ChatContext {
        let mut c = ChatContext::empty();
        for (n, (id, text)) in items.iter().enumerate() {
            c.push(msg(id, Role::User, text, n as u64 * 10));
        }
        c
    }
    fn ids(v: &[Anchored]) -> Vec<(Option<&str>, &str)> {
        v.iter().map(|(p, i)| (p.as_deref(), i.as_str())).collect()
    }

    // Catches: diffing content instead of ids.
    #[test]
    fn an_edited_message_is_an_update_not_a_remove_and_create() {
        let d = compute_chat_ctx_diff(
            &ctx(&[("a", "hi"), ("b", "old")]),
            &ctx(&[("a", "hi"), ("b", "new")]),
        );
        assert_eq!(d.to_remove, Vec::<String>::new());
        assert_eq!(d.to_create, vec![]);
        assert_eq!(ids(&d.to_update), vec![(Some("a"), "b")]);

        let before = ctx(&[("a", "hello\nthere")]);
        let mut after = before.clone();
        after.items[0].as_message_mut().unwrap().content = vec![
            crate::ChatContent::Text("hello".into()),
            crate::ChatContent::Text("there".into()),
        ];
        assert_eq!(compute_chat_ctx_diff(&before, &after), DiffOps::default());
    }

    // Catches: anchoring every new item to the last surviving id (a run arrives reversed).
    #[test]
    fn consecutive_inserts_chain_onto_each_other() {
        let d = compute_chat_ctx_diff(
            &ctx(&[("a", "1")]),
            &ctx(&[("a", "1"), ("x", "2"), ("y", "3"), ("z", "4")]),
        );
        assert_eq!(ids(&d.to_create), vec![(Some("a"), "x"), (Some("x"), "y"), (Some("y"), "z")]);
    }

    // Catches: losing the root anchor for a prepended item.
    #[test]
    fn an_item_prepended_at_the_head_anchors_to_none() {
        let d = compute_chat_ctx_diff(&ctx(&[("a", "1")]), &ctx(&[("z", "0"), ("a", "1")]));
        assert_eq!(ids(&d.to_create), vec![(None, "z")]);
    }

    // Catches: a naive "ids not in new" removal that also drops items which merely moved.
    #[test]
    fn a_reorder_removes_and_recreates_only_the_moved_item() {
        let d = compute_chat_ctx_diff(
            &ctx(&[("a", "1"), ("b", "2"), ("c", "3")]),
            &ctx(&[("b", "2"), ("a", "1"), ("c", "3")]),
        );
        assert_eq!(d.to_remove, vec!["a".to_owned()]);
        assert_eq!(ids(&d.to_create), vec![(Some("b"), "a")]);
        assert!(d.to_update.is_empty());
    }

    #[test]
    fn identical_contexts_produce_no_ops() {
        let c = ctx(&[("a", "1"), ("b", "2")]);
        assert_eq!(compute_chat_ctx_diff(&c, &c), DiffOps::default());
    }

    // Catches: an empty old or new context panicking on the DP table's zero dimension.
    #[test]
    fn empty_contexts_are_handled_at_both_ends() {
        let empty = ChatContext::empty();
        let full = ctx(&[("a", "1"), ("b", "2")]);
        assert_eq!(
            ids(&compute_chat_ctx_diff(&empty, &full).to_create),
            vec![(None, "a"), (Some("a"), "b")]
        );
        assert_eq!(
            compute_chat_ctx_diff(&full, &empty).to_remove,
            vec!["a".to_owned(), "b".to_owned()]
        );
    }

    // Catches: a wrong DP table. Two common subsequences of different lengths; only the table picks
    // the longer one.
    #[test]
    fn an_lcs_choice_between_two_common_subsequences_is_resolved_by_the_table() {
        // "a,b,c" survives as a block; "z" was dropped and "q" prepended.
        let d = compute_chat_ctx_diff(
            &ctx(&[("a", "1"), ("b", "2"), ("c", "3"), ("z", "4")]),
            &ctx(&[("q", "0"), ("a", "1"), ("b", "2"), ("c", "3")]),
        );
        assert_eq!(d.to_remove, vec!["z".to_owned()]);
        assert_eq!(ids(&d.to_create), vec![(None, "q")]);
        assert!(d.to_update.is_empty());

        // Keeping the tail is longer, so the head is removed and recreated.
        let d = compute_chat_ctx_diff(
            &ctx(&[("a", "1"), ("b", "2"), ("c", "3"), ("d", "4"), ("e", "5")]),
            &ctx(&[("c", "3"), ("d", "4"), ("e", "5"), ("a", "1"), ("b", "2")]),
        );
        assert_eq!(d.to_remove, vec!["a".to_owned(), "b".to_owned()], "not c, d and e");
        assert_eq!(ids(&d.to_create), vec![(Some("e"), "a"), (Some("a"), "b")]);

        // A genuine tie: `max(dp[i-1][j], dp[i][j-1])` must break it as `llm/utils.py:60` does.
        let d = compute_chat_ctx_diff(
            &ctx(&[("a", "1"), ("b", "2"), ("c", "3")]),
            &ctx(&[("a", "1"), ("c", "3"), ("d", "4"), ("b", "2")]),
        );
        assert_eq!(d.to_remove, vec!["b".to_owned()], "the reference keeps a and c, not a and b");
        assert_eq!(ids(&d.to_create), vec![(Some("c"), "d"), (Some("d"), "b")]);
    }

    #[test]
    fn repeated_host_ids_preserve_the_existing_reorder_tie() {
        let d = compute_chat_ctx_diff(
            &ctx(&[("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")]),
            &ctx(&[("b", "2"), ("a", "1"), ("b", "2"), ("d", "4"), ("c", "3")]),
        );
        assert_eq!(d.to_remove, vec!["c"]);
        assert_eq!(ids(&d.to_create), vec![(Some("d"), "c")]);
        assert!(d.to_update.is_empty());

        let appended_duplicate =
            compute_chat_ctx_diff(&ctx(&[("a", "old")]), &ctx(&[("a", "old"), ("a", "new")]));
        assert!(appended_duplicate.to_create.is_empty());
        assert_eq!(ids(&appended_duplicate.to_update), vec![(Some("a"), "a")]);
    }

    // Pins the reference's gap: a same-id change of non-text content is invisible.
    #[test]
    fn only_text_changes_are_detected_as_updates() {
        let mut a = ChatContext::empty();
        let mut m1 = msg("a", Role::User, "same", 0);
        let mut m2 = msg("a", Role::User, "same", 0);
        if let Some(m) = m1.as_message_mut() {
            m.extra.insert("google".into(), serde_json::json!({"sig": "one"}));
        }
        if let Some(m) = m2.as_message_mut() {
            m.extra.insert("google".into(), serde_json::json!({"sig": "two"}));
        }
        a.push(m1);
        let mut b = ChatContext::empty();
        b.push(m2);
        assert!(
            compute_chat_ctx_diff(&a, &b).to_update.is_empty(),
            "reference does not detect this"
        );
    }
}
