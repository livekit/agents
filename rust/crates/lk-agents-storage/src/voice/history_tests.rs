use super::*;
use lk_agents_abi::{ChatMessage, Role};
use std::time::UNIX_EPOCH;

fn item(id: &str, text: &str) -> ChatItem {
    ChatMessage::text(id, Role::User, text, UNIX_EPOCH).into()
}

fn snapshot(items: Vec<ChatItem>, stored: Vec<ChatItem>) -> VoiceHistory {
    VoiceHistory { items, stored, revision: (3, 7), ..Default::default() }
}

#[test]
fn distinct_local_and_foreign_rows_rebase_with_unseen_additions() {
    let base = vec![item("a", "before a"), item("b", "before b")];
    let local = snapshot(vec![base[0].clone(), item("b", "local b")], base);
    let remote = vec![item("a", "foreign a"), item("b", "before b"), item("c", "foreign c")];
    let merged = reconcile(&remote, local, &Default::default()).unwrap();
    assert_eq!(
        merged.items,
        [item("a", "foreign a"), item("b", "local b"), item("c", "foreign c")]
    );
    assert_eq!(merged.rebased, ["a", "c"]);
}

#[test]
fn divergent_same_row_intents_are_refused_instead_of_silently_selected() {
    let local = snapshot(vec![item("a", "local")], vec![item("a", "original")]);
    let remote = vec![item("a", "foreign")];
    assert_eq!(reconcile(&remote, local.clone(), &Default::default()).unwrap_err(), ["a"]);
    assert_eq!(local.items, [item("a", "local")], "the caller retains its own intent and base");
    assert_eq!(local.stored, [item("a", "original")]);
    assert_eq!(remote, [item("a", "foreign")]);
}

#[test]
fn equal_revisions_converge_but_unknown_same_identity_additions_conflict() {
    let local = snapshot(vec![item("a", "same")], vec![]);
    assert_eq!(
        reconcile(&[item("a", "same")], local.clone(), &Default::default()).unwrap().items,
        local.items
    );
    assert_eq!(
        reconcile(&[item("a", "different")], local, &Default::default()).unwrap_err(),
        ["a"]
    );
}

#[test]
fn foreign_deletion_retires_unchanged_reads_but_refuses_a_local_rewrite() {
    let original = item("a", "before");
    assert!(
        reconcile(
            &[],
            snapshot(vec![original.clone()], vec![original.clone()]),
            &Default::default()
        )
        .unwrap()
        .items
        .is_empty()
    );
    assert_eq!(
        reconcile(&[], snapshot(vec![item("a", "local")], vec![original]), &Default::default())
            .unwrap_err(),
        ["a"]
    );
}

#[test]
fn a_newer_own_write_is_kept_when_the_submitted_row_was_unchanged() {
    let original = item("a", "before");
    let local = snapshot(vec![original.clone()], vec![original]);
    let current = item("a", "own checkpoint");
    let owned = BTreeMap::from([("a".into(), 8)]);
    assert_eq!(reconcile(std::slice::from_ref(&current), local, &owned).unwrap().items, [current]);
}

#[test]
fn proven_own_in_flight_writes_allow_the_later_submission_final_words() {
    let original = item("a", "before");
    let local = snapshot(vec![item("a", "final")], vec![original]);
    let current = item("a", "own partial");
    let owned = BTreeMap::from([("a".into(), 8)]);
    assert_eq!(reconcile(&[current], local, &owned).unwrap().items, [item("a", "final")]);
}
