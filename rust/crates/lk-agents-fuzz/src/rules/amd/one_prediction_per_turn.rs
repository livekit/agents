use crate::{
    authored::{Authored, amd_bound, amd_predicted, amd_prediction},
    check::{Check, Contract, Rule, Subject},
    record::Recording,
};
use lk_agents_voice::amd::events::{AmdEvent, AmdReason, Category, Stage};
use lk_agents_voice::trace::AmdStep;

// Spec §14 invariant 1: one prediction per turn at most. Each turn AMD bound gets at most one
// `Predicted` step, and the host gets exactly as many `amd_prediction` events for its turn id
// as the executor journalled: the trace and the outcome must agree. Every event names a bound
// turn.

pub struct OnePredictionPerTurn;

impl Rule for OnePredictionPerTurn {
    const CONTRACT: Contract = Contract {
        id: "one_prediction_per_turn",
        guarantee: "Each turn AMD classifies gets at most one prediction, and the host hears of \
                    each prediction once.",
        examples: &[one_each],
        counterexamples: &[predicted_twice, predicted_silently, an_event_for_no_turn],
        probe: Some(a_person_answers),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let view = r.amd_run();
        let steps = view.steps();
        let predicted = view.predicted();
        for (_, run) in view.entered() {
            let events = view.predictions(run);
            for (_, step) in &steps {
                let AmdStep::TurnBound { run: bound, turn, turn_id, .. } = step else { continue };
                if *bound != run {
                    continue;
                }
                let traced = predicted.iter().filter(|p| p.run == run && p.turn == *turn).count();
                let told = events.iter().filter(|(_, p)| p.turn_id == *turn_id).count();
                check.expect_with(Subject::Session, traced <= 1 && told == traced, || {
                    format!(
                        "run {} turn {} (AMD turn {turn_id}) was predicted {traced} times and \
                         the host told {told} times",
                        run.0, turn.0
                    )
                });
            }
            for (at, prediction) in &events {
                let bound = steps.iter().any(|(_, s)| {
                    matches!(s, AmdStep::TurnBound { run: r, turn_id, .. }
                        if *r == run && *turn_id == prediction.turn_id)
                });
                check.expect_with(Subject::Session, bound, || {
                    format!(
                        "run {} told the host of AMD turn {} at {} ms, which it never bound",
                        run.0, prediction.turn_id, at.at_ms
                    )
                });
            }
        }
    }
}

fn bound() -> Authored {
    Authored::new().amd_entered(1).amd_step(amd_bound(1, 4, 1))
}

fn human() -> AmdStep {
    let human = Some(Category::Human);
    amd_predicted(1, 4, 1, Some(1), human, Category::Human, Stage::Human, AmdReason::Prediction, false)
}

fn event() -> AmdEvent {
    AmdEvent::Prediction(amd_prediction(1, Category::Human, Stage::Human, AmdReason::Prediction))
}

fn one_each() -> Recording {
    bound().amd_step(human()).amd_event(1, event()).done()
}

fn predicted_twice() -> Recording {
    bound().amd_step(human()).amd_event(1, event()).amd_step(human()).amd_event(1, event()).done()
}

fn predicted_silently() -> Recording {
    bound().amd_step(human()).done()
}

fn an_event_for_no_turn() -> Recording {
    let stray = amd_prediction(2, Category::Human, Stage::Human, AmdReason::Reused);
    bound().amd_step(human()).amd_event(1, event()).amd_event(1, AmdEvent::Prediction(stray)).done()
}

/// A person answers the first turn.
fn a_person_answers() -> crate::drive::Case {
    use crate::fakes::amd::executor_probe::{call, turn};
    use crate::input::AmdSays;
    call(Default::default(), turn(0, "hello who is this", 0, AmdSays::Human, 0))
}
