use crate::{
    authored::{Authored, amd_outcome},
    check::{Check, Contract, Rule, Subject},
    record::Recording,
};
use lk_agents_voice::amd::events::{AmdEvent, AmdReason, Stage};
use lk_agents_voice::amd::{AmdObservation, AmdRunId};
use lk_agents_voice::trace::AmdStep;

// Spec §14 invariant 12: `execute()` returns once. Each run entered through the executor
// completes once: one `Completed` step, after the session's `Detached` for the run, one
// `amd_completed` event, and an `execute()` that returns that same outcome. A run that ended
// with the session must have completed: the session's close completes it with `cancelled`.

pub struct AmdCompletesOnce;

impl Rule for AmdCompletesOnce {
    const CONTRACT: Contract = Contract {
        id: "amd_completes_once",
        guarantee: "Every AMD run completes once, after the session detached it, and every \
                    caller of execute() gets that completion.",
        examples: &[completed_after_detached],
        counterexamples: &[completed_twice, completed_before_detached, never_completed, returned_another],
        probe: Some(a_person_answers),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let view = r.amd_run();
        for (_, run) in view.entered() {
            let steps: Vec<_> = (view.steps().into_iter())
                .filter_map(|(at, s)| match s {
                    AmdStep::Completed { run: r, category, reason } if r == run => {
                        Some((at, category, reason))
                    }
                    _ => None,
                })
                .collect();
            let events = view.completed(run);
            let returned = view.outcomes(run);
            let ended = r.driver_finished();
            if !ended && steps.is_empty() && events.is_empty() && returned.is_empty() {
                continue;
            }
            let once = |n: usize| if ended { n == 1 } else { n <= 1 };
            check.expect_with(
                Subject::Session,
                once(steps.len()) && once(events.len()) && once(returned.len()),
                || {
                    format!(
                        "run {} completed {} times, told the host {} times and returned from \
                         execute() {} times",
                        run.0,
                        steps.len(),
                        events.len(),
                        returned.len()
                    )
                },
            );
            let detached = view.detached(run);
            for (at, category, reason) in &steps {
                check.expect_with(Subject::Session, detached.is_some_and(|d| d < *at), || {
                    format!(
                        "run {} completed at {} ms, before the session detached it ({detached:?})",
                        run.0, at.at_ms
                    )
                });
                let agrees = (events.iter().chain(&returned))
                    .all(|(_, o)| o.category == *category && o.reason == *reason);
                let same = returned.iter().all(|(_, o)| events.iter().all(|(_, e)| e == o));
                check.expect_with(Subject::Session, agrees && same, || {
                    format!(
                        "run {} completed as {category:?}/{reason:?}, told the host {events:?} and \
                         returned {returned:?}",
                        run.0
                    )
                });
            }
        }
    }
}

fn detached() -> AmdObservation {
    AmdObservation::Detached { run: AmdRunId(1) }
}

fn completed() -> AmdStep {
    AmdStep::Completed { run: AmdRunId(1), category: Stage::Human, reason: AmdReason::Finished }
}

fn outcome() -> lk_agents_voice::amd::events::Outcome {
    amd_outcome(Stage::Human, AmdReason::Finished, false)
}

fn entered() -> Authored {
    Authored::new().amd_entered(1)
}

fn completed_after_detached() -> Recording {
    (entered().amd(1, detached()).amd_step(completed()))
        .amd_event(1, AmdEvent::Completed(outcome()))
        .amd_returned(1, outcome())
        .driver_finished()
        .done()
}

fn completed_twice() -> Recording {
    (entered().amd(1, detached()).amd_step(completed()).amd_step(completed()))
        .amd_event(1, AmdEvent::Completed(outcome()))
        .amd_event(1, AmdEvent::Completed(outcome()))
        .amd_returned(1, outcome())
        .driver_finished()
        .done()
}

fn completed_before_detached() -> Recording {
    (entered().amd_step(completed()).amd(1, detached()))
        .amd_event(1, AmdEvent::Completed(outcome()))
        .amd_returned(1, outcome())
        .driver_finished()
        .done()
}

fn never_completed() -> Recording {
    entered().amd(1, detached()).driver_finished().done()
}

fn returned_another() -> Recording {
    let cancelled = amd_outcome(Stage::Uncertain, AmdReason::Cancelled, false);
    (entered().amd(1, detached()).amd_step(completed()))
        .amd_event(1, AmdEvent::Completed(outcome()))
        .amd_returned(1, cancelled)
        .driver_finished()
        .done()
}

/// A person answers the first turn, and the run completes.
fn a_person_answers() -> crate::drive::Case {
    use crate::fakes::amd::executor_probe::{call, turn};
    use crate::input::AmdSays;
    call(Default::default(), turn(0, "hello who is this", 0, AmdSays::Human, 0))
}
