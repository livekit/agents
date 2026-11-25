use lk_agents_voice::trace::{GateStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdVerdict, Input, Ref},
    record::{Kind, Recording, amd::Answer},
};

// Every answer is keyed: the session applies a decision only to the latest
// committed turn of the current run; every other answer is dropped, and journalled as
// `AnswerDropped`. The outcome is judged first: a stale decision never answers its turn.

pub struct GateAnswersAreKeyed;

impl Rule for GateAnswersAreKeyed {
    const CONTRACT: Contract = Contract {
        id: "gate_answers_are_keyed",
        guarantee: "A decision naming a turn other than the latest committed one of the current \
                    AMD run answers nothing and is dropped.",
        examples: &[stale_dropped],
        counterexamples: &[stale_applied, stale_not_journalled],
        probe: Some(a_stale_decision),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        let turns = amd.gated_turns();
        for answer in amd.answers() {
            let Answer::Decide { at, run, turn, .. } = answer else { continue };
            let current = amd.attached_at(at).map(|a| a.run);
            // A turn counts from the gate's own report of its commit, which AMD answers, or from
            // the host's, whichever came first: the host's can come after AMD's answer.
            let reported = |turn: u64| {
                amd.observed(run).into_iter().any(|(seen, o)| {
                    seen < at
                        && matches!(o, lk_agents_voice::amd::AmdObservation::TurnCommitted { turn: t, .. }
                            if t.0 == turn)
                })
            };
            let latest = (turns.iter())
                .filter(|t| t.run == run && (t.committed < at || reported(t.turn)))
                .map(|t| t.turn)
                .next_back();
            if current == Some(run) && latest == Some(turn) {
                continue;
            }
            let named = turns.iter().find(|t| t.run == run && t.turn == turn);
            if let Some(named) = named {
                let answered_after = amd
                    .requests_answering(named)
                    .into_iter()
                    .filter(|q| !amd.preemptive(q))
                    // Asked before any answer of its own allowed it: a decision for a turn the
                    // session had not committed is no allow for it once it commits.
                    .find(|q| {
                        q.opened > at && amd.allowed_at(named).is_none_or(|a| q.opened < a.at())
                    });
                check.expect_with(
                    Subject::Resource(Kind::AmdTurn, turn as u32),
                    answered_after.is_none(),
                    || {
                        format!(
                            "a decision for run {run} turn {turn} at {} ms, when the latest turn \
                             was {latest:?} of run {current:?}, answered it",
                            at.at_ms
                        )
                    },
                );
            }
            let dropped = r.journal().iter().any(|(seen, step)| {
                *seen > at
                    && matches!(step, Step::Gate(GateStep::AnswerDropped { run: r, turn: t, .. })
                        if r.0 == run && *t == Some(lk_agents_voice::audio_recognition::TurnId(turn)))
            });
            check.expect_with(Subject::Session, dropped, || {
                format!("the stale decision for run {run} turn {turn} at {} ms was not dropped", at.at_ms)
            });
        }
    }
}

fn two_turns() -> Authored {
    Authored::new()
        .sent(Input::AmdAttach {})
        .user_turn(Some(1), 1, "hello")
        .user_turn(Some(1), 2, "is anyone there")
}

fn stale() -> Input {
    Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Allow }
}

fn dropped() -> Step {
    Step::Gate(GateStep::AnswerDropped {
        run: lk_agents_voice::amd::AmdRunId(1),
        turn: Some(lk_agents_voice::audio_recognition::TurnId(1)),
        why: lk_agents_voice::trace::AnswerDrop::NotLatest,
    })
}

fn stale_dropped() -> Recording {
    two_turns().sent(stale()).journal(dropped()).done()
}

fn stale_applied() -> Recording {
    two_turns().sent(stale()).journal(dropped()).request(10, 1, "hello", &[], &[]).done()
}

fn stale_not_journalled() -> Recording {
    two_turns().sent(stale()).done()
}

/// AMD decides an older turn after a newer one committed, then the newer one.
fn a_stale_decision() -> crate::drive::Case {
    use crate::fakes::amd::probe::*;
    let mut inputs = attached(None);
    inputs.extend(says(0, "hello"));
    inputs.extend(says(1, "is anyone there"));
    inputs.push(decide(0, AmdVerdict::Allow));
    inputs.push(decide(1, AmdVerdict::Allow));
    inputs.extend(answered(3));
    inputs.push(advance(3_000));
    call(inputs)
}
