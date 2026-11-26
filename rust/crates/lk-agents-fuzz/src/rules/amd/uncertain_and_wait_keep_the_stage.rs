use crate::{
    authored::{Authored, amd_predicted, amd_prediction},
    check::{Check, Contract, Rule, Subject},
    record::Recording,
};
use lk_agents_voice::amd::events::{AmdEvent, AmdReason, Category, Stage};

// Spec §14 invariant 4: `uncertain` and `wait` keep the stage. Judged on what the host heard:
// each `amd_prediction` whose category is `uncertain` or `wait` carries the stage of the run's
// previous prediction (`uncertain` before the first). The executor's `Predicted` steps must say
// the same stages in the same order.

pub struct UncertainAndWaitKeepTheStage;

impl Rule for UncertainAndWaitKeepTheStage {
    const CONTRACT: Contract = Contract {
        id: "uncertain_and_wait_keep_the_stage",
        guarantee: "A turn predicted uncertain or wait leaves AMD in the stage it was in.",
        examples: &[a_screener_then_a_hold],
        counterexamples: &[a_hold_resets_the_stage, the_trace_disagrees],
        probe: Some(a_screener_puts_the_caller_on_hold),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let view = r.amd_run();
        for (_, run) in view.entered() {
            let events = view.predictions(run);
            let mut stage = Stage::Uncertain;
            for (at, p) in &events {
                if matches!(p.category, Category::Uncertain | Category::Wait) {
                    check.expect_with(Subject::Session, p.stage == stage, || {
                        format!(
                            "AMD turn {} was predicted {:?} at {} ms and moved the stage from \
                             {stage:?} to {:?}",
                            p.turn_id, p.category, at.at_ms, p.stage
                        )
                    });
                }
                stage = p.stage;
            }
            let traced: Vec<_> = (view.predicted().into_iter())
                .filter(|p| p.run == run)
                .map(|p| (p.turn_id, p.category, p.stage))
                .collect();
            let told: Vec<_> = events.iter().map(|(_, p)| (p.turn_id, p.category, p.stage)).collect();
            if !told.is_empty() || !traced.is_empty() {
                check.expect_with(Subject::Session, traced == told, || {
                    format!("the executor journalled {traced:?} and told the host {told:?}")
                });
            }
        }
    }
}

fn predicted(run: Authored, turn_id: u64, category: Category, stage: Stage) -> Authored {
    let step = amd_predicted(1, turn_id + 3, turn_id, Some(turn_id), Some(category), category, stage, AmdReason::Prediction, false);
    let event = AmdEvent::Prediction(amd_prediction(turn_id, category, stage, AmdReason::Prediction));
    run.amd_step(step).amd_event(1, event)
}

fn screened() -> Authored {
    predicted(Authored::new().amd_entered(1), 1, Category::MachineScreening, Stage::MachineScreening)
}

fn a_screener_then_a_hold() -> Recording {
    let held = predicted(screened(), 2, Category::Wait, Stage::MachineScreening);
    predicted(held, 3, Category::Uncertain, Stage::MachineScreening).done()
}

fn a_hold_resets_the_stage() -> Recording {
    predicted(screened(), 2, Category::Wait, Stage::Uncertain).done()
}

fn the_trace_disagrees() -> Recording {
    let step = amd_predicted(1, 5, 2, Some(2), Some(Category::Wait), Category::Wait, Stage::Uncertain, AmdReason::Prediction, false);
    let event = AmdEvent::Prediction(amd_prediction(2, Category::Wait, Stage::MachineScreening, AmdReason::Prediction));
    screened().amd_step(step).amd_event(1, event).done()
}

/// A screener, then a hold.
fn a_screener_puts_the_caller_on_hold() -> crate::drive::Case {
    use crate::fakes::amd::{executor_probe::{call, turn}, probe};
    use crate::input::{AmdSays, Input, Ref};
    let mut inputs = turn(0, "hi this is call assist please state your name", 0, AmdSays::Screening, 0);
    inputs.extend(probe::says(1, "please hold while i check"));
    inputs.push(Input::AmdReply { ask: Ref::new(1), says: AmdSays::Wait });
    inputs.push(probe::advance(3_000));
    call(Default::default(), inputs)
}
