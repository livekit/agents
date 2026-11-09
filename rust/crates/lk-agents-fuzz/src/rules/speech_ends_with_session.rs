use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Handle, Recording, SpeechView},
};

// Every wait on a speech resolves once it is done or interrupted, so the session's end must leave
// no handle short of done: one left running would hang its waiters forever.

pub struct SpeechEndsWithSession;

impl Rule for SpeechEndsWithSession {
    const CONTRACT: Contract = Contract {
        id: "speech_ends_with_session",
        guarantee: "Once the session has closed, every speech handle is done.",
        examples: &[ended],
        counterexamples: &[left_running],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        if !r.driver_finished() {
            return;
        }
        for speech in r.host_view().speeches().into_iter().filter(SpeechView::is_known) {
            check.expect(
                Subject::Speech(speech.id),
                speech.is_done(),
                "the session closed with this speech's handle still running".to_owned(),
            );
        }
    }
}

/// Scheduled, then done before the session closed.
fn ended() -> Recording {
    Authored::new().speech(7, Handle::live()).speech(7, Handle::finished()).driver_finished().done()
}

/// The session closed with the speech still live.
fn left_running() -> Recording {
    Authored::new().speech(7, Handle::live()).driver_finished().done()
}
