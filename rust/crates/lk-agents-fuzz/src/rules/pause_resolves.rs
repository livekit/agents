//! A speech paused for the person's speech does not stay paused once they stopped.
//!
//! The pause is the false-interruption machine's: the person's voice over the agent pauses the
//! speech (`PlayoutGate { Pause }`) until it is clear whether they meant to interrupt. Its cause is
//! their speech; once that has ended, the speech resumes (a false interruption), is interrupted,
//! ends, or the session closes, within the false-interruption timeout and as much again. A pause
//! nothing ends is an agent gone silent mid-sentence: what the realtime server's onset did before
//! c6881abb9, when no false-interruption timer was armed for its pause.
use lk_agents_abi::voice::events::UserState;
use lk_agents_voice::{
    agent_activity::scheduling::SpeechId,
    agent_session::effects::{GateCause, GateOp},
    events::SessionEvent,
    trace::{CloseStep, CoreStep, InterruptStep, SpeechStep, Step},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Observed, Recording},
};

/// How long a pause may outlast the person's speech, beyond the turn decision it waits on (the
/// endpointing delay in effect): the default false-interruption timeout (2 s) and as much again.
pub const BOUND_MS: u64 = 4_000;

pub struct PauseResolves;

impl Rule for PauseResolves {
    const CONTRACT: Contract = Contract {
        id: "pause_resolves",
        guarantee: "A speech paused for the person's speech resumes, is interrupted, ends or the session closes within a bound of that speech ending.",
        examples: &[resumed_after_the_person_stopped],
        counterexamples: &[paused_for_good],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let journal = r.journal();
        let end = r.now().at_ms;
        for (i, (at, step)) in journal.iter().enumerate() {
            let Step::Interrupt(InterruptStep::PlayoutGate { speech, op: GateOp::Pause, cause }) =
                step
            else {
                continue;
            };
            // The launch hold waits for the person by design, and its release is the person's.
            if *cause == GateCause::Launch {
                continue;
            }
            // A pause the same speech already has is the earlier pause's to judge.
            if journal[..i].iter().rev().find_map(|(_, s)| match s {
                Step::Interrupt(InterruptStep::PlayoutGate { speech: sp, op, .. })
                    if sp == speech =>
                {
                    Some(*op == GateOp::Pause)
                }
                _ => None,
            }) == Some(true)
            {
                continue;
            }
            let resolved =
                journal[i + 1..].iter().find(|(_, s)| resolves(s, speech.0)).map(|(a, _)| *a);
            let until = resolved.map_or(end, |r| r.at_ms);
            // The endpointing in effect while the turn decision owns the pause: a change made
            // after the pause (a larger maximum delay) is waited on too.
            let bound = allowance_ms(journal, at.at_ms).max(allowance_ms(journal, until));
            // The first stretch, from the pause to its resolution, in which the person stayed
            // quiet for the whole bound: speaking again cancels the resume, as it should.
            let Some(quiet) = quiet_stretch(journal, *at, until, bound) else {
                // Resolved before the person was ever quiet for the bound: in time.
                if resolved.is_some() {
                    check.expect_with(Subject::Speech(speech.0), true, String::new);
                }
                continue;
            };
            let deadline = quiet + bound;
            check.expect_with(
                Subject::Speech(speech.0),
                resolved.is_some_and(|r| r.at_ms <= deadline),
                || match resolved {
                    Some(r) => format!(
                        "paused at {} ms, the person quiet from {quiet} ms, resolved only at {} ms",
                        at.at_ms, r.at_ms
                    ),
                    None => format!(
                        "paused at {} ms, the person quiet from {quiet} ms, never resolved",
                        at.at_ms
                    ),
                },
            );
        }
    }
}

/// What ends a pause of `speech`.
fn resolves(step: &Step, speech: u64) -> bool {
    match step {
        Step::Interrupt(InterruptStep::PlayoutGate { speech: s, op, .. }) => {
            s.0 == speech && *op != GateOp::Pause
        }
        Step::Interrupt(InterruptStep::Interrupted { speech: s, .. }) => s.0 == speech,
        Step::Speech(SpeechStep::ReplyFinished { speech: s, .. }) => s.0 == speech,
        Step::Emitted(SessionEvent::SpeechDone { speech: s, .. }) => s.0 == speech,
        Step::Close(CloseStep::Requested { .. }) => true,
        _ => false,
    }
}

/// How long after the person goes quiet a pause at `at_ms` may hold: [`BOUND_MS`] beyond the
/// ceiling of the endpointing delay in effect, the turn decision the pause waits on: the larger
/// of its bounds, since a fixed delay serves `min_delay` even above `max_delay`.
pub(crate) fn allowance_ms(journal: &[(Observed, Step)], at_ms: u64) -> u64 {
    let endpointing = journal
        .iter()
        .take_while(|(a, _)| a.at_ms <= at_ms)
        .filter_map(|(_, s)| match s {
            // The larger bound: a minimum set above the maximum is the delay waited (seed 62945).
            Step::Core(CoreStep::ConfigApplied { effective, .. }) => {
                let e = &effective.turn.endpointing;
                e.max_delay.max(e.min_delay)
            }
            _ => None,
        })
        .last()
        .map_or(3_000, |d| d.as_millis() as u64);
    BOUND_MS + endpointing
}

/// The start of the first stretch, between `from` and `until`, in which the person was quiet for
/// `bound` without a break; `None` when they never were.
fn quiet_stretch(
    journal: &[(Observed, Step)],
    from: Observed,
    until: u64,
    bound: u64,
) -> Option<u64> {
    let mut quiet = quiet_from(journal, from);
    for (a, s) in journal.iter().skip_while(|(a, _)| a.at_ms <= from.at_ms) {
        if a.at_ms > until {
            break;
        }
        if let Some(q) = quiet
            && a.at_ms > q + bound
        {
            return Some(q);
        }
        if let Step::Emitted(SessionEvent::UserStateChanged { new, .. }) = s {
            quiet = match (*new == UserState::Speaking, quiet) {
                (true, _) => None,
                (false, Some(q)) => Some(q),
                (false, None) => Some(a.at_ms),
            };
        }
        // Their voice over the agent again (a recogniser's interim with no onset of its own, say):
        // they spoke just now, so a quiet stretch starts over from here.
        if let Step::Interrupt(InterruptStep::ByAudioActivity(_)) = s
            && quiet.is_some_and(|q| q <= a.at_ms)
        {
            quiet = Some(a.at_ms);
        }
    }
    quiet.filter(|q| until > q + bound)
}

/// The instant, at or after `at`, from which the person is not speaking: `at` itself when they
/// were not speaking then.
fn quiet_from(journal: &[(Observed, Step)], at: Observed) -> Option<u64> {
    // The person's state as of the pause's instant: what that onset said, even journalled after
    // the pause it caused.
    let state_at = journal
        .iter()
        .take_while(|(a, _)| a.at_ms <= at.at_ms)
        .filter_map(|(_, s)| match s {
            Step::Emitted(SessionEvent::UserStateChanged { new, .. }) => Some(*new),
            _ => None,
        })
        .last()
        .unwrap_or(UserState::Listening);
    if state_at != UserState::Speaking {
        return Some(at.at_ms);
    }
    journal.iter().skip_while(|(a, _)| a.at_ms <= at.at_ms).find_map(|(a, s)| match s {
        Step::Emitted(SessionEvent::UserStateChanged { new, .. })
            if *new != UserState::Speaking =>
        {
            Some(a.at_ms)
        }
        _ => None,
    })
}

fn gate(op: GateOp) -> Step {
    Step::Interrupt(InterruptStep::PlayoutGate {
        speech: SpeechId(4),
        op,
        cause: GateCause::ByAudioActivity,
    })
}

/// Paused, then resumed a moment later: a false interruption.
fn resumed_after_the_person_stopped() -> Recording {
    Authored::new()
        .journal(gate(GateOp::Pause))
        .later(1_000)
        .journal(gate(GateOp::Resume))
        .later(BOUND_MS * 2)
        .done()
}

/// Paused, and nothing ever ends it.
fn paused_for_good() -> Recording {
    Authored::new().journal(gate(GateOp::Pause)).later(BOUND_MS * 2).done()
}
