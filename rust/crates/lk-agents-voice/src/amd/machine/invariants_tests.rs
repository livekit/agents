//! Spec §14 invariants over every short sequence of classifier answers. The workspace has no
//! property-testing library, so these enumerate the sequences instead of sampling them.

use super::harness::*;
use super::*;
use crate::amd::AmdAnswer;
use std::collections::{BTreeMap, BTreeSet};

/// Every sequence of `len` categories over `alphabet`.
fn sequences(alphabet: &[Category], len: u32) -> Vec<Vec<Category>> {
    let n = alphabet.len();
    (0..n.pow(len))
        .map(|mut i| {
            (0..len)
                .map(|_| {
                    let category = alphabet[i % n];
                    i /= n;
                    category
                })
                .collect()
        })
        .collect()
}

/// The stage a stage-naming category enters.
fn stage_of(category: Category) -> Option<Stage> {
    match category {
        Category::MachineScreening => Some(Stage::MachineScreening),
        Category::MachineVm => Some(Stage::MachineVm),
        Category::MachineIvr => Some(Stage::MachineIvr),
        Category::Human => Some(Stage::Human),
        Category::MachineUnavailable => Some(Stage::MachineUnavailable),
        Category::Uncertain | Category::Wait => None,
    }
}

#[test]
fn test_every_three_turn_sequence_keeps_the_prediction_and_decision_rules() {
    for sequence in sequences(&CATEGORIES, 3) {
        let mut h = H::started(AmdOptions::default());
        let mut committed = BTreeSet::new();
        for category in &sequence {
            if !h.detaches().is_empty() {
                break;
            }
            let turn = h.commit("hello");
            committed.insert(turn);
            h.classify(turn, *category);
        }
        h.finish();

        // rule 1: one prediction per turn at most
        let mut per_turn = BTreeMap::new();
        for p in h.predictions() {
            *per_turn.entry(p.turn_id).or_insert(0) += 1;
        }
        assert!(per_turn.values().all(|n| *n == 1), "{sequence:?}: {per_turn:?}");

        // rule 4: `uncertain` and `wait` keep the stage; other categories enter theirs
        let mut stage = Stage::Uncertain;
        for p in h.predictions() {
            let expected = stage_of(p.category).unwrap_or(stage);
            assert_eq!(p.stage, expected, "{sequence:?}: {p:?}");
            stage = p.stage;
        }

        // every answer names this run; every decision names a turn it committed and predicted
        let predicted: BTreeMap<u64, Category> =
            h.predictions().iter().map(|p| (p.turn_id, p.category)).collect();
        for answer in h.answers() {
            match answer {
                AmdAnswer::ReplyDecided { run, turn, decision } => {
                    assert_eq!(run, RUN, "{sequence:?}");
                    assert!(committed.contains(&turn), "{sequence:?}: {turn:?}");
                    // rule 3: `wait` never replies (AMD turn ids follow the session's from 1 here)
                    let category = predicted.get(&turn.0);
                    assert!(
                        category.is_some(),
                        "{sequence:?}: {turn:?} decided without a prediction"
                    );
                    if category == Some(&Category::Wait) {
                        assert!(!decision.allow, "{sequence:?}: {turn:?}");
                    }
                }
                AmdAnswer::Activated { run, .. }
                | AmdAnswer::InsertUserItem { run, .. }
                | AmdAnswer::Detach { run, .. } => assert_eq!(run, RUN, "{sequence:?}"),
            }
        }
        assert!(h.detaches().len() <= 1, "{sequence:?}");
        assert!(h.completions().len() <= 1, "{sequence:?}");
        assert!(!h.predictions().is_empty(), "{sequence:?}");
    }
}

#[test]
fn test_every_four_turn_sequence_plays_one_voicemail_message_at_most() {
    // rule 6, with every allowed reply played to its end
    let alphabet = [Category::MachineVm, Category::Uncertain, Category::Wait, Category::MachineIvr];
    for sequence in sequences(&alphabet, 4) {
        let mut h = H::started(AmdOptions::default());
        let mut messages = 0;
        for category in &sequence {
            if !h.detaches().is_empty() {
                break;
            }
            let turn = h.turn("hello", *category);
            let decisions = h.decisions(turn);
            assert!(decisions.len() <= 1, "{sequence:?}: {decisions:?}");
            let Some(decision) = decisions.first() else { continue };
            if decision.instructions.as_deref() == Some(DEFAULT_VOICEMAIL_INSTRUCTIONS) {
                messages += 1;
            }
            if decision.allow {
                h.reply_played(turn, "Hi Sam, this is Alex.");
            }
        }
        assert!(messages <= 1, "{sequence:?}: {messages} voicemail messages");
        // every attempt plays to its end, so the one allowed message is the one played
        h.step(In::Close);
        assert_eq!(h.outcome().voicemail_message_played, messages == 1, "{sequence:?}");
    }
}
