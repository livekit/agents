//! Every speech is done once the test has answered everything the driver was waiting on.
//!
//! While the test is still talking, a speech may legitimately wait on it. The test then answers
//! every request, synthesis, playout, hook and hold, lets the driver's timers fire, and keeps
//! going until nothing new appears. At that instant the driver waits on nobody but itself, so a
//! speech created before it and not yet done was waiting on nothing: it was stuck, and only the
//! close will end it.
use lk_agents_voice::trace::{Step, TurnStep};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    record::{Handle, Observed, Recording, SpeechView},
};

pub struct SpeechSettles;

impl Rule for SpeechSettles {
    const CONTRACT: Contract = Contract {
        id: "speech_settles",
        guarantee: "Every speech is done once the test has answered everything the driver was waiting on.",
        examples: &[done_in_time],
        counterexamples: &[still_running],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        // A run that never got there -- still producing work when the runner gave up -- cannot
        // judge a speech; `session_settles` reports it.
        let Some(answered) = r.inputs_view().everything_answered_at() else { return };
        for speech in r.host_view().speeches().into_iter().filter(SpeechView::is_known) {
            // A speech the driver created afterwards, on its own, is judged by nobody here.
            if speech.created_at().is_some_and(|created| created > answered) {
                continue;
            }
            // A preemptive reply parked for a turn that never ended waits on the person's final
            // words, which the case never sent: the reference parks it the same way.
            if speech.is_speculative() && !speech.is_scheduled() && !turn_ended(r, &speech, answered) {
                continue;
            }
            let done = speech.done_at().is_some_and(|done| done <= answered);
            check.expect(&speech, done, format!("still running after everything was answered ({speech:?})"));
        }
    }
}

/// A turn ended after the speech was created and before everything was answered.
fn turn_ended(r: &Recording, speech: &SpeechView, answered: Observed) -> bool {
    let Some(created) = speech.created_at() else { return false };
    r.internal_view().steps().iter().any(|(at, step)| {
        *at > created && *at <= answered && matches!(step, Step::Turn(TurnStep::EndOfTurn { .. }))
    })
}

/// The speech finished, then the test had answered everything.
fn done_in_time() -> Recording {
    Authored::new().speech(7, Handle::finished()).everything_answered().done()
}

/// The test had answered everything and the speech was still running.
fn still_running() -> Recording {
    Authored::new().speech(7, Handle::live()).everything_answered().done()
}
