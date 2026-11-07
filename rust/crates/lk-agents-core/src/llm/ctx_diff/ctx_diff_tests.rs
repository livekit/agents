use super::*;
use lk_agents_abi::{ChatMessage, Role};

fn msg(id: &str, role: Role, text: &str, at: u64) -> ChatItem {
    let at = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(at);
    ChatMessage::text(id, role, text, at).into()
}

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
        lk_agents_abi::ChatContent::Text("hello".into()),
        lk_agents_abi::ChatContent::Text("there".into()),
    ];
    assert_eq!(compute_chat_ctx_diff(&before, &after), DiffOps::default());
}

// Catches: an expert's outcome replacing its call's pending output under the same id never
// reaching a realtime server, which kept answering from the pending output.
#[test]
fn a_replaced_call_output_is_an_update() {
    let output = |text: &str| {
        let at = std::time::SystemTime::UNIX_EPOCH;
        ChatItem::from(lk_agents_abi::FunctionCallOutput::new("out", "call", text, false, at))
    };
    let before = ChatContext::new(vec![msg("a", Role::User, "hi", 0), output("pending")]);
    let after = ChatContext::new(vec![msg("a", Role::User, "hi", 0), output("refunded")]);
    assert_eq!(ids(&compute_chat_ctx_diff(&before, &after).to_update), vec![(Some("a"), "out")]);
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
    assert!(compute_chat_ctx_diff(&a, &b).to_update.is_empty(), "reference does not detect this");
}
