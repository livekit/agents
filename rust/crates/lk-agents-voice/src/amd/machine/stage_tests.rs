//! Ports of the `_fsm.py` tests in Python's `tests/test_amd.py`.

use super::*;
use std::collections::BTreeSet;

use crate::amd::machine::harness::{CATEGORIES, spec_recommended};

const STAGES: [Stage; 6] = [
    Stage::Uncertain,
    Stage::MachineScreening,
    Stage::MachineVm,
    Stage::MachineIvr,
    Stage::Human,
    Stage::MachineUnavailable,
];

fn transition(next_stage: Stage, effect: Option<StageEffect>) -> Transition {
    Transition { next_stage, effect }
}

/// The stage a category names; `wait` names none.
fn named(category: Category) -> Option<Stage> {
    match category {
        Category::Uncertain => Some(Stage::Uncertain),
        Category::Wait => None,
        Category::MachineScreening => Some(Stage::MachineScreening),
        Category::MachineVm => Some(Stage::MachineVm),
        Category::MachineIvr => Some(Stage::MachineIvr),
        Category::Human => Some(Stage::Human),
        Category::MachineUnavailable => Some(Stage::MachineUnavailable),
    }
}

#[test]
fn test_transition_is_repeatable_and_does_not_mutate_input() {
    let first = next(Stage::Uncertain, Category::MachineVm);
    assert_eq!(next(Stage::Uncertain, Category::MachineVm), first);
    assert_eq!(first, Some(transition(Stage::MachineVm, None)));
}

#[test]
fn test_stage_transitions() {
    for current in STAGES {
        for category in CATEGORIES {
            let result = next(current, category);
            if spec_recommended(current).is_empty() {
                // a terminal stage has completed AMD
                assert_eq!(result, None, "{current:?} -> {category:?}");
                continue;
            }
            let expected_stage = match category {
                Category::Uncertain | Category::Wait => current,
                other => named(other).unwrap(),
            };
            let expected_effect = match category {
                Category::Human | Category::MachineUnavailable => Some(StageEffect::Complete),
                Category::MachineIvr => Some(StageEffect::ExtractMenu),
                _ => None,
            };
            assert_eq!(
                result,
                Some(transition(expected_stage, expected_effect)),
                "{current:?} -> {category:?}"
            );
        }
    }
}

#[test]
fn test_recommended_next_categories_follow_the_spec() {
    for stage in STAGES {
        let listed: BTreeSet<_> = recommended(stage).iter().copied().collect();
        assert_eq!(listed, spec_recommended(stage), "{stage:?}");
        assert_eq!(listed.len(), recommended(stage).len(), "{stage:?} lists a category twice");
    }
}

#[test]
fn test_wait_and_uncertain_keep_the_stage() {
    let pairs = [
        (Stage::MachineScreening, Category::MachineIvr, Stage::MachineIvr),
        (Stage::MachineVm, Category::MachineScreening, Stage::MachineScreening),
        (Stage::MachineIvr, Category::MachineScreening, Stage::MachineScreening),
    ];
    for (initial, corrected, corrected_stage) in pairs {
        for bridge in [Category::Uncertain, Category::Wait] {
            let intermediate = next(initial, bridge).unwrap();
            assert_eq!(intermediate, transition(initial, None));
            assert_eq!(
                next(intermediate.next_stage, corrected).unwrap().next_stage,
                corrected_stage
            );
        }
    }
}

#[test]
fn test_same_ivr_state_extracts_each_menu() {
    assert_eq!(
        next(Stage::MachineIvr, Category::MachineIvr),
        Some(transition(Stage::MachineIvr, Some(StageEffect::ExtractMenu)))
    );
}

#[test]
fn test_leaving_the_recommendations_corrects_the_stage() {
    let pairs = [
        (Stage::MachineScreening, Category::MachineIvr, Stage::MachineIvr),
        (Stage::MachineVm, Category::MachineScreening, Stage::MachineScreening),
        (Stage::MachineIvr, Category::MachineScreening, Stage::MachineScreening),
    ];
    for (initial, corrected, corrected_stage) in pairs {
        assert!(spec_recommended(initial).contains(&Category::Human));
        assert!(!recommended(initial).contains(&corrected), "{initial:?} recommends {corrected:?}");
        let effect = (corrected == Category::MachineIvr).then_some(StageEffect::ExtractMenu);
        assert_eq!(next(initial, corrected), Some(transition(corrected_stage, effect)));
    }
}

#[test]
fn test_terminal_stages_cannot_reopen_detection() {
    for stage in [Stage::Human, Stage::MachineUnavailable] {
        assert_eq!(next(stage, Category::MachineScreening), None);
    }
}
