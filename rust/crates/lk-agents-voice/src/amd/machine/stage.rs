//! Stage transitions (`_fsm.py`). [`Amd`](super::Amd) runs the effects.
//!
//! `uncertain` and `wait` are per-turn predictions, not stages: they keep the current stage.
//! [`recommended`] lists the usual next predictions for each stage. It guides the classifier but
//! does not restrict it: another category corrects an earlier stage.

use crate::amd::events::{Category, Stage};

/// `RECOMMENDED`: the usual next predictions from `stage`, sorted by value as a request sends
/// them. Empty for a terminal stage, which completes AMD.
pub fn recommended(stage: Stage) -> Vec<Category> {
    use Category::*;
    let mut list = match stage {
        Stage::Uncertain => Category::ALL.to_vec(),
        Stage::MachineScreening => {
            vec![MachineScreening, Human, MachineVm, MachineUnavailable, Uncertain, Wait]
        }
        Stage::MachineVm => vec![MachineVm, Human, MachineIvr, MachineUnavailable, Uncertain, Wait],
        Stage::MachineIvr => {
            vec![MachineIvr, Human, MachineVm, MachineUnavailable, Uncertain, Wait]
        }
        Stage::Human | Stage::MachineUnavailable => Vec::new(),
    };
    list.sort_by_key(|category| category.as_str());
    list
}

/// `LIKELY_HUMAN_STAGES`. After a menu or a screener, the next speaker can be a person or a
/// voicemail greeting, and a short greeting fits both. In these stages a first human prediction
/// is likely human: it counts as uncertain, and the next turn decides.
pub const LIKELY_HUMAN_STAGES: [Stage; 2] = [Stage::MachineScreening, Stage::MachineIvr];

pub fn is_likely_human_stage(stage: Stage) -> bool {
    LIKELY_HUMAN_STAGES.contains(&stage)
}

/// What entering a stage asks of AMD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageEffect {
    ExtractMenu,
    Complete,
}

/// The stage after a prediction and its effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transition {
    pub next_stage: Stage,
    pub effects: Vec<StageEffect>,
}

/// `transition`: the stage after `prediction`. `None` from a terminal stage, where Python raises:
/// AMD has completed, so nothing follows it.
pub fn next(stage: Stage, prediction: Category) -> Option<Transition> {
    if recommended(stage).is_empty() {
        return None;
    }
    let (next_stage, effects) = match prediction {
        Category::Human => (Stage::Human, vec![StageEffect::Complete]),
        Category::MachineUnavailable => (Stage::MachineUnavailable, vec![StageEffect::Complete]),
        Category::MachineIvr => (Stage::MachineIvr, vec![StageEffect::ExtractMenu]),
        Category::Uncertain | Category::Wait => (stage, Vec::new()),
        Category::MachineScreening => (Stage::MachineScreening, Vec::new()),
        Category::MachineVm => (Stage::MachineVm, Vec::new()),
    };
    Some(Transition { next_stage, effects })
}

#[cfg(test)]
#[path = "stage_tests.rs"]
mod tests;
