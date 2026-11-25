use crate::{
    authored::{Authored, amd_predicted, amd_prediction},
    check::{Check, Contract, Rule, Subject},
    record::{Recording, amd_run::AskKind},
};
use lk_agents_voice::amd::effects::LIKELY_HUMAN;
use lk_agents_voice::amd::events::{AmdEvent, AmdReason, Category, Stage};
use lk_agents_voice::amd::machine::stage::is_likely_human_stage;

// Spec §14 invariant 5: the likely-human mark. In a menu or screener stage, the classifier's
// first `human` answer is held: the prediction is `uncertain` in the same stage and the mark is
// set. With the mark set, `human` is accepted. Any other category but `uncertain` clears the
// mark. Read from the classifier's raw answer, the prediction and the mark the executor
// journalled; the outcome side is the host's event for the turn and the next request, which
// shows the previous category as `uncertain (likely human)`.

pub struct LikelyHumanIsHeldOnce;

impl Rule for LikelyHumanIsHeldOnce {
    const CONTRACT: Contract = Contract {
        id: "likely_human_is_held_once",
        guarantee: "After a screener or a menu, the first human answer is held as uncertain and \
                    the next request says so; a second one is accepted.",
        examples: &[held_then_accepted],
        counterexamples: &[accepted_at_once, held_twice, the_next_request_is_not_told],
        probe: Some(a_person_after_a_screener),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let view = r.amd_run();
        let asks = view.asks_of(AskKind::Classify);
        let choices = view.asks_of(AskKind::Choice);
        for (_, run) in view.entered() {
            let (mut stage, mut mark) = (Stage::Uncertain, false);
            let events = view.predictions(run);
            for p in view.predicted().into_iter().filter(|p| p.run == run) {
                if p.reason == AmdReason::Prediction && p.raw == Some(Category::Human) {
                    let hold = is_likely_human_stage(stage) && !mark;
                    let (category, kept) = if hold { (Category::Uncertain, stage) } else { (Category::Human, Stage::Human) };
                    let told = events.iter().any(|(_, e)| e.turn_id == p.turn_id && e.category == category);
                    check.expect_with(
                        Subject::Session,
                        p.category == category && p.likely_human == hold && p.stage == kept && told,
                        || {
                            format!(
                                "AMD turn {}'s human answer in {stage:?} (mark {mark}) was \
                                 predicted {:?} in {:?} with mark {}; the host was told: {told}",
                                p.turn_id, p.category, p.stage, p.likely_human
                            )
                        },
                    );
                    if hold {
                        // The next request after the held prediction tells the model.
                        let next = (asks.iter().chain(&choices))
                            .filter(|(_, opened, _)| *opened > p.at)
                            .min_by_key(|(_, opened, _)| *opened);
                        if let Some((nth, _, asked)) = next {
                            let state = asked.state().unwrap_or_default();
                            check.expect_with(Subject::Session, state.contains(LIKELY_HUMAN), || {
                                format!("request {nth} after a held human shows {state}")
                            });
                        }
                    }
                } else if p.raw.is_some_and(|raw| raw != Category::Uncertain) {
                    check.expect_with(Subject::Session, !p.likely_human, || {
                        format!("AMD turn {}'s {:?} answer kept the likely-human mark", p.turn_id, p.raw)
                    });
                }
                stage = p.stage;
                mark = p.likely_human;
            }
        }
    }
}

const SCREENING: &str = r#"{"stage": "machine-screening", "previous_prediction": {"category": "machine-screening"}}"#;
const HELD: &str = r#"{"stage": "machine-screening", "previous_prediction": {"category": "uncertain (likely human)"}}"#;

fn predicted(run: Authored, turn_id: u64, raw: Category, category: Category, stage: Stage, mark: bool) -> Authored {
    let step = amd_predicted(1, turn_id + 3, turn_id, Some(turn_id), Some(raw), category, stage, AmdReason::Prediction, mark);
    let event = AmdEvent::Prediction(amd_prediction(turn_id, category, stage, AmdReason::Prediction));
    run.amd_step(step).amd_event(1, event)
}

fn screened() -> Authored {
    let run = Authored::new().amd_entered(1);
    predicted(run, 1, Category::MachineScreening, Category::MachineScreening, Stage::MachineScreening, false)
}

fn held() -> Authored {
    predicted(screened(), 2, Category::Human, Category::Uncertain, Stage::MachineScreening, true)
}

fn held_then_accepted() -> Recording {
    let asked = held().amd_asked(AskKind::Classify, HELD, "hello who is this");
    predicted(asked, 3, Category::Human, Category::Human, Stage::Human, false).done()
}

fn accepted_at_once() -> Recording {
    predicted(screened(), 2, Category::Human, Category::Human, Stage::Human, false).done()
}

fn held_twice() -> Recording {
    let asked = held().amd_asked(AskKind::Classify, HELD, "hello who is this");
    predicted(asked, 3, Category::Human, Category::Uncertain, Stage::MachineScreening, true).done()
}

fn the_next_request_is_not_told() -> Recording {
    held().amd_asked(AskKind::Classify, SCREENING, "hello who is this").done()
}

/// A screener, then a person's first words.
fn a_person_after_a_screener() -> crate::drive::Case {
    use crate::fakes::amd::executor_probe::{call, turn};
    use crate::input::AmdSays;
    let mut inputs = turn(0, "hi this is call assist please state your name", 0, AmdSays::Screening, 0);
    inputs.extend(turn(1, "hello", 1, AmdSays::Human, 1));
    call(Default::default(), inputs)
}
