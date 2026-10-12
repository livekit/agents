//! Minimal edit script between two `ChatContext`s (`llm/utils.py`): a realtime server can only be
//! told "insert this after that" or "delete this".

use std::collections::{HashMap, HashSet};

use crate::{ChatContext, ItemKind};

/// `(previous_item_id, id)`; `None` is first in the conversation.
pub type Anchored = (Option<String>, String);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DiffOps {
    pub to_remove: Vec<String>,
    pub to_create: Vec<Anchored>,
    /// Patched in place, keeping the item's server-side position.
    pub to_update: Vec<Anchored>,
}

/// Ids only, never content, so an edited message is an update. Several index expressions have
/// equivalent mutants because ids within one [`ChatContext`] are unique.
fn lcs(old: &[&str], new: &[&str]) -> Vec<String> {
    let (n, m) = (old.len(), new.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = if old[i - 1] == new[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }

    let (mut i, mut j) = (n, m);
    let mut out = Vec::new();
    while i > 0 && j > 0 {
        if old[i - 1] == new[j - 1] {
            out.push(old[i - 1].to_owned());
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    out.reverse();
    out
}

/// `compute_chat_ctx_diff` (`llm/utils.py:120`) over the host's items, as
/// `(to_remove, to_create, to_update)`.
#[crate::host(export(name = "compute_chat_ctx_diff"))]
pub fn compute_chat_ctx_diff_of(old: Vec<crate::ChatItem>, new: Vec<crate::ChatItem>) -> (Vec<String>, Vec<Anchored>, Vec<Anchored>) {
    let ops = compute_chat_ctx_diff(&ChatContext::new(old), &ChatContext::new(new));
    (ops.to_remove, ops.to_create, ops.to_update)
}

pub fn compute_chat_ctx_diff(old: &ChatContext, new: &ChatContext) -> DiffOps {
    let old_ids: Vec<&str> = old.items.iter().map(|i| i.id()).collect();
    let new_ids: Vec<&str> = new.items.iter().map(|i| i.id()).collect();
    let kept: HashSet<String> = lcs(&old_ids, &new_ids).into_iter().collect();
    let old_by_id: HashMap<&str, &crate::ChatItem> =
        old.items.iter().map(|i| (i.id(), i)).collect();

    let to_remove =
        old.items.iter().filter(|i| !kept.contains(i.id())).map(|i| i.id().to_owned()).collect();

    let (mut to_create, mut to_update) = (Vec::new(), Vec::new());
    let mut prev: Option<String> = None;

    for item in &new.items {
        if !kept.contains(item.id()) {
            to_create.push((prev.clone(), item.id().to_owned()));
        } else if let (Some(old_item), ItemKind::Message, ItemKind::Message) =
            (old_by_id.get(item.id()), item.kind(), old_by_id[item.id()].kind())
        {
            // Only the text is compared, as in the reference (`TODO: check other content types`).
            let (a, b) = (item.as_message(), old_item.as_message());
            if let (Some(a), Some(b)) = (a, b)
                && a.raw_text_content() != b.raw_text_content()
            {
                to_update.push((prev.clone(), item.id().to_owned()));
            }
        }
        // Advances for created items too, so a run of new items chains in order.
        prev = Some(item.id().to_owned());
    }

    DiffOps { to_remove, to_create, to_update }
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
