use lk_agents_voice::amd::{AmdObservation, TurnEnd};
use lk_agents_voice::audio_recognition::TurnId;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdVerdict, Input, Ref},
    record::{Observed, Recording},
};

// the AMD proposal, "DST": every turn committed under the gate reaches a decision, a
// supersession or an exit within a bound, while AMD makes progress. The scripted answerer is
// that progress: it can answer only what the session tells it, so the session must report each
// commit at once; once AMD answered (or a newer turn superseded it), the turn must end and show
// its outcome: a request answering it when allowed, its saved words when denied. A session that
// does nothing fails here.

pub struct GateSettles;

/// How long an answered turn may take to end, past its customer hook.
const BOUND_MS: u64 = 2_000;

impl Rule for GateSettles {
    const CONTRACT: Contract = Contract {
        id: "gate_settles",
        guarantee: "Every turn committed under the AMD gate is reported to AMD at its commit, \
                    and ends with its outcome within a bound once AMD answered it or a newer \
                    turn superseded it.",
        examples: &[answered_and_ended, superseded_and_ended],
        counterexamples: &[never_reported, answered_never_ended, allowed_with_no_reply],
        probe: Some(a_denied_turn),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        for turn in amd.gated_turns() {
            let observed = amd.observed(turn.run);
            let reported = observed.iter().any(|(at, o)| {
                matches!(o, AmdObservation::TurnCommitted { turn: t, .. } if *t == TurnId(turn.turn))
                    && *at <= next_input(r, turn.committed)
            });
            check.expect_with(Subject::Session, reported, || {
                format!(
                    "turn {} ({:?}) committed under run {} at {} ms and AMD was never told",
                    turn.turn, turn.words, turn.run, turn.committed.at_ms
                )
            });
            let answer = amd.answers_for(&turn).into_iter().next();
            let from = match (answer, turn.superseded) {
                (Some(a), Some(s)) => a.at().min(s),
                (Some(a), None) => a.at(),
                (None, Some(s)) => s,
                (None, None) => continue,
            };
            // Past the turn's own customer hook, which is issued only once every older turn's
            // hook returned: no bound on the session covers customer code.
            let Some(hook) = amd.hook_returned(&turn) else { continue };
            let from = from.max(hook);
            let due = from.at_ms + BOUND_MS;
            if r.now().at_ms < due || r.close_begun().is_some_and(|c| c.at_ms <= due) {
                continue;
            }
            let ended = observed.iter().find_map(|(at, o)| match o {
                AmdObservation::TurnEnded { turn: t, how } if *t == TurnId(turn.turn) => {
                    Some((*at, *how))
                }
                _ => None,
            });
            // `Detached` is the run's last observation (proposal "Detach", step 5): a turn its
            // detach settled ends after it, judged by its outcome below.
            let detached = (amd.attachments().into_iter())
                .find(|a| a.run == turn.run)
                .and_then(|a| a.detached)
                .is_some_and(|d| d.at_ms <= due && ended.is_none_or(|(at, _)| at > d));
            let ended_in_time = ended.is_some_and(|(at, _)| at.at_ms <= due);
            check.expect_with(Subject::Session, ended_in_time || detached, || {
                format!(
                    "turn {} was answered or superseded at {} ms and had not ended by {due} ms \
                     ({ended:?})",
                    turn.turn, from.at_ms
                )
            });
            let replied = !amd.requests_answering(&turn).is_empty();
            let kept = turn.item_id.is_some();
            // `StopResponse` ignores the turn, as Python does (`agent_activity.py`).
            let stopped = amd.stopped(&turn);
            // Dropped behind a speech it could not interrupt, or while scheduling was paused: an
            // exit with no reply (proposal "Turn exits").
            let dropped = amd.dropped_unanswered(&turn);
            check.expect_with(Subject::Session, replied || kept || stopped || dropped, || {
                format!(
                    "turn {} ended ({ended:?}) with neither a reply nor its words saved",
                    turn.turn
                )
            });
            // A stored allow can be replaced by `Detach.pending`, so the turn's own end says
            // whether it was answered.
            if matches!(ended, Some((_, TurnEnd::Replied))) {
                check.expect_with(Subject::Session, replied, || {
                    format!("turn {} ended as replied ({answer:?}) and nothing answered it", turn.turn)
                });
            }
        }
    }
}

/// The first input the run applied after `at`: a report made in the commit's own transition
/// is recorded before it.
fn next_input(r: &Recording, at: Observed) -> Observed {
    r.applied().iter().map(|a| a.at).find(|seen| *seen > at).unwrap_or(Observed::END)
}

fn turn() -> Authored {
    Authored::new().sent(Input::AmdAttach {}).user_turn(Some(1), 1, "hello")
}

fn ended(how: TurnEnd) -> AmdObservation {
    AmdObservation::TurnEnded { turn: TurnId(1), how }
}

fn answered_and_ended() -> Recording {
    turn()
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Allow })
        .request(10, 1, "hello", &[], &[])
        .amd(1, ended(TurnEnd::Replied))
        .later(3_000)
        .done()
}

fn superseded_and_ended() -> Recording {
    turn()
        .user_turn(Some(1), 2, "hello again")
        .amd(1, ended(TurnEnd::Superseded))
        .later(3_000)
        .done()
}

fn never_reported() -> Recording {
    Authored::new()
        .sent(Input::AmdAttach {})
        .user_turn(None, 1, "hello")
        .sent(Input::Advance { ms: crate::input::Millis(1) })
        .later(3_000)
        .done()
}

fn answered_never_ended() -> Recording {
    turn()
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Deny })
        .later(3_000)
        .done()
}

fn allowed_with_no_reply() -> Recording {
    turn()
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Allow })
        .amd(1, ended(TurnEnd::Replied))
        .later(3_000)
        .done()
}

/// A turn AMD denies is reported and ends with its words saved.
fn a_denied_turn() -> crate::drive::Case {
    crate::fakes::amd::probe::one_turn("please hold", AmdVerdict::Wait)
}
