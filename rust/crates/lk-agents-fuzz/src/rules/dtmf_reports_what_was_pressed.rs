//! A DTMF collection the driver runs answers with exactly the keys the caller pressed.
//!
//! Evidence is the keys the linked caller pressed, as the session journaled them, after the
//! collection's `Start` (and before its call, the next `Start` or a `Stop`), up to the stop key
//! `#`, dropping what is no DTMF key. A collection without a read-back (`confirm: false`) holds no guard across a reply, so
//! each of those keys is buffered. The call is then decided by count alone: `Complete` carries
//! them, space-separated, when there are `num_digits` of them; `Incomplete` when there are not;
//! `Stopped` only once the collection was ended -- a `Stop`, a newer `Start`, or a close (the
//! host's, or the session's own).
use lk_agents_voice::trace::{RoomLinkStep, RoomStep, Step};
use lk_agents_voice::workflows::dtmf::{DtmfEvent, DtmfOutcome, GetDtmfResult};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    drive::{Case, RoomSetup, Setup},
    input::{Count, DtmfInput, DtmfKey, Input, Millis, RoomIn},
    record::{Answer, Detail, Kind, Recording},
};

pub struct DtmfReportsWhatWasPressed;

impl Rule for DtmfReportsWhatWasPressed {
    const CONTRACT: Contract = Contract {
        id: "dtmf_reports_what_was_pressed",
        guarantee: "A DTMF collection answers with the keys pressed: complete at num_digits, incomplete otherwise, stopped only once ended.",
        examples: &[complete_as_pressed, stopped_after_stop],
        counterexamples: &[wrong_digits, complete_too_short, stopped_while_running],
        probe: Some(two_collections),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let host = r.host_view();
        let internal = r.internal_view();
        for call in host.api_calls() {
            let Input::Dtmf { input: DtmfInput::Start, digits, confirm, .. } = call.input() else {
                continue;
            };
            let Some((at, Answer::Dtmf(outcome))) = call.returned() else { continue };
            let started = call.called_at();

            // What the case did between the start and the call: the keys, and whether it ended the
            // collection (a stop, a newer start) or the session.
            let mut ended_at = None;
            for applied in inputs.sent().iter().filter(|a| started < a.at && a.at < at) {
                if let Input::Dtmf { input: DtmfInput::Start | DtmfInput::Stop, .. }
                | Input::Close { .. } = applied.input
                {
                    ended_at = Some(applied.at);
                    break;
                }
            }
            let mut ended = ended_at.is_some();
            let mut pressed: Vec<&str> = Vec::new();
            let mut stop_key = false;
            for key in r.journal_where(|step| match step {
                Step::Room(RoomStep::Link(RoomLinkStep::KeyPressed { key })) => Some(*key),
                _ => None,
            }) {
                let (when, key) = key;
                if when <= started || when >= at || ended_at.is_some_and(|e| when >= e) || stop_key {
                    continue;
                }
                match key {
                    Some(DtmfEvent::Pound) => stop_key = true,
                    Some(key) => pressed.push(key.as_str()),
                    None => {}
                }
            }

            // A close ends it too -- the host's, or the session's own (the error policy, the room
            // going away) -- including one begun before the `Start`, which never starts.
            ended |= inputs.close_asked_at().is_some_and(|close| close < at)
                || internal.close_requested().is_some_and(|(when, _)| when < at);
            match outcome {
                DtmfOutcome::Stopped => check.expect(
                    &call,
                    ended,
                    format!("collection answered Stopped at seq {} while still running", at.seq),
                ),
                // The read-back arm holds a guard across its reply; its keys are not this rule's.
                DtmfOutcome::Complete(_) | DtmfOutcome::Incomplete { .. } if *confirm => {}
                DtmfOutcome::Complete(GetDtmfResult { user_input }) => {
                    let expected = pressed.join(" ");
                    let whole = pressed.len() == digits.0 as usize + 1;
                    check.expect(
                        &call,
                        *user_input == expected && whole,
                        format!(
                            "completed with {user_input:?}; pressed {expected:?} of {} keys",
                            digits.0 + 1
                        ),
                    );
                }
                DtmfOutcome::Incomplete { .. } => {
                    check.expect(
                        &call,
                        pressed.len() != digits.0 as usize + 1,
                        format!(
                            "answered incomplete with all {} keys pressed: {pressed:?}",
                            digits.0 + 1
                        ),
                    );
                }
            }
        }
    }
}

fn start(digits: u32) -> Input {
    Input::Dtmf {
        input: DtmfInput::Start,
        key: DtmfKey::Pound,
        digits: Count(digits),
        confirm: false,
        timeout: Millis(500),
    }
}

fn key(key: DtmfKey) -> Input {
    Input::Dtmf {
        input: DtmfInput::Digit,
        key,
        digits: Count(0),
        confirm: false,
        timeout: Millis(0),
    }
}

fn stop() -> Input {
    Input::Dtmf {
        input: DtmfInput::Stop,
        key: DtmfKey::Pound,
        digits: Count(0),
        confirm: false,
        timeout: Millis(0),
    }
}

/// A key sent, and the session's journal of the caller pressing it.
trait Press {
    fn press(self, key: DtmfKey) -> Self;
}

impl Press for Authored {
    fn press(self, k: DtmfKey) -> Self {
        let pressed = DtmfEvent::parse(k.digit());
        self.sent(key(k)).journal(Step::Room(RoomStep::Link(RoomLinkStep::KeyPressed { key: pressed })))
    }
}

fn collecting(digits: u32) -> Authored {
    Authored::new().sent_opening(start(digits), &[(Kind::Answer, Detail::None)])
}

fn complete(user_input: &str) -> Answer {
    Answer::Dtmf(DtmfOutcome::Complete(GetDtmfResult { user_input: user_input.into() }))
}

/// Two keys asked for, two pressed (and a non-key dropped), answered with both.
fn complete_as_pressed() -> Recording {
    collecting(1)
        .press(DtmfKey::One)
        .press(DtmfKey::Unknown)
        .press(DtmfKey::Nine)
        .result(0, complete("1 9"))
        .done()
}

/// Stopped after the task's exit.
fn stopped_after_stop() -> Recording {
    collecting(1)
        .press(DtmfKey::One)
        .sent(stop())
        .result(0, Answer::Dtmf(DtmfOutcome::Stopped))
        .done()
}

/// The call names keys nobody pressed.
fn wrong_digits() -> Recording {
    collecting(1).press(DtmfKey::One).press(DtmfKey::Two).result(0, complete("1 9")).done()
}

/// Complete with one key of two, the stop key cutting the sequence short.
fn complete_too_short() -> Recording {
    collecting(1).press(DtmfKey::One).press(DtmfKey::Pound).result(0, complete("1")).done()
}

/// Stopped with nothing having ended the collection.
fn stopped_while_running() -> Recording {
    collecting(1).press(DtmfKey::One).result(0, Answer::Dtmf(DtmfOutcome::Stopped)).done()
}

/// Prerequisite: two collections without a read-back. The first gets both of its keys and
/// completes on the debounce; the second is cut short by the stop key and answers incomplete.
fn two_collections() -> Case {
    Case {
        setup: Setup { room: Some(RoomSetup { close_on_disconnect: false, delete_room_on_close: false }), ..Setup::default() },
        inputs: vec![
            Input::Room { event: RoomIn::Linked },
            Input::Advance { ms: Millis(20) },
            start(1),
            key(DtmfKey::One),
            key(DtmfKey::Two),
            Input::Advance { ms: Millis(1_000) },
            start(2),
            key(DtmfKey::Nine),
            key(DtmfKey::Pound),
            Input::Advance { ms: Millis(100) },
        ],
        ..Case::default()
    }
}
