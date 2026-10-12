//! The host hears exactly one `SpeechDone` per speech, after its `SpeechCreated`, with the
//! outcome the handle reports; and a handle that is done is announced.
use lk_agents_voice::{
    agent_activity::scheduling::SpeechKind,
    events::SpeechOutcome,
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    record::{Handle, Recording},
};

pub struct SpeechDoneOnce;

impl Rule for SpeechDoneOnce {
    const CONTRACT: Contract = Contract {
        id: "speech_done_once",
        guarantee: "Each speech gets one SpeechDone, after SpeechCreated, matching its handle; a done handle is announced.",
        examples: &[announced_once],
        counterexamples: &[twice, before_created, wrong_outcome, never_announced, unknown_speech],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for speech in r.host_view().speeches() {
            let announced = speech.done_announcements();

            // A completion names a speech the host holds a handle for; one for a stranger is a
            // completion attributed to nobody.
            if !announced.is_empty() {
                check.expect(&speech, speech.is_known(), "SpeechDone for a speech the test never saw");
            }
            if !speech.is_known() {
                continue;
            }

            // At most one completion per speech.
            check.expect(&speech, announced.len() <= 1, format!("{} SpeechDone events", announced.len()));

            match announced.first() {
                Some((done, outcome)) => {
                    // The host heard the speech exist before it heard it finish.
                    let created = speech.created_at();
                    check.expect(&speech, created.is_some_and(|created| created < *done), "SpeechDone before SpeechCreated");

                    // The announced outcome is the one the handle itself reports.
                    if let Some(expected) = speech.outcome() {
                        check.expect(&speech, *outcome == expected, format!("announced {outcome:?}, handle says {expected:?} ({speech:?})"));
                    }
                }
                None => {
                    // A done handle must reach the host; the driver gets one input of grace.
                    if let Some(done) = speech.done_at() {
                        let told = !r.inputs_view().driver_ran_after(done);
                        check.expect(&speech, told, format!("handle done since {done:?}, host never told"));
                    }
                }
            }
        }
    }
}

/// Created, finished, announced once with the matching outcome.
fn announced_once() -> Recording {
    Authored::new()
        .speech_created(1, SpeechKind::Say)
        .speech(1, Handle::finished())
        .speech_done(1, SpeechOutcome::Completed)
        .done()
}

/// Two completions for one speech.
fn twice() -> Recording {
    Authored::new()
        .speech_created(1, SpeechKind::Say)
        .speech(1, Handle::finished())
        .speech_done(1, SpeechOutcome::Completed)
        .speech_done(1, SpeechOutcome::Completed)
        .done()
}

/// The completion arrives before the creation.
fn before_created() -> Recording {
    Authored::new()
        .speech_done(1, SpeechOutcome::Completed)
        .speech_created(1, SpeechKind::Say)
        .speech(1, Handle::finished())
        .done()
}

/// The handle completed normally, the host was told it was interrupted.
fn wrong_outcome() -> Recording {
    Authored::new()
        .speech_created(1, SpeechKind::Say)
        .speech(1, Handle::finished())
        .speech_done(1, SpeechOutcome::Interrupted)
        .done()
}

/// The handle is done, the runner moved on, the host heard nothing.
fn never_announced() -> Recording {
    Authored::new()
        .speech_created(1, SpeechKind::Say)
        .speech(1, Handle::finished())
        .then_act()
        .done()
}

/// A completion for a speech the test never saw created.
fn unknown_speech() -> Recording {
    Authored::new().speech_done(9, SpeechOutcome::Completed).done()
}
