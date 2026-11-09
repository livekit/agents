use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::{CloseStep, RoomLinkStep, RoomStep, Step};
use lk_agents_voice::workflows::machine::dtmf::{DtmfEvent, DtmfOutcome, GetDtmfResult};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, RoomSetup, Setup},
    input::{Count, DtmfInput, DtmfKey, Input, Millis, RoomIn},
    record::{Answer, Detail, Kind, Observed, Recording},
};

// The journalled keys must first be the person's own presses, in order, before the close, never a
// guest's. A collection's evidence is the keys after its `Start`, up to the stop key `#`. Without a
// read-back (`confirm: false`) no guard spans a reply, so every key is buffered and the answer is
// decided by count alone.

pub struct DtmfReportsWhatWasPressed;

impl Rule for DtmfReportsWhatWasPressed {
    const CONTRACT: Contract = Contract {
        id: "dtmf_reports_what_was_pressed",
        guarantee: "A DTMF collection answers with the keys pressed: complete at num_digits, incomplete otherwise, stopped only once ended.",
        examples: &[
            complete_as_pressed,
            stopped_after_stop,
            a_guest_is_not_heard,
            a_press_racing_the_close,
            a_press_left_with_its_participant,
        ],
        counterexamples: &[
            wrong_digits,
            complete_too_short,
            stopped_while_running,
            a_press_dropped,
            a_guests_press_heard,
            a_press_dropped_before_the_close,
            a_press_dropped_before_a_draining_close,
            a_press_dropped_before_a_rejoin,
        ],
        probe: Some(two_collections),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let host = r.host_view();
        let internal = r.internal_view();
        let journaled: Vec<_> = r
            .journal_where(|step| match step {
                Step::Room(RoomStep::Link(RoomLinkStep::KeyPressed { key })) => Some(*key),
                _ => None,
            })
            .collect();
        // The person's presses the room took, and how many came before the session began closing.
        let closing = inputs
            .close_asked_at()
            .into_iter()
            .chain(internal.close_requested().map(|(at, _)| at))
            .min();
        // When the session finished closing: room I/O is torn down with it.
        let closed = r
            .journal_where(|step| match step {
                Step::Emitted(SessionEvent::Close { .. }) => Some(()),
                _ => None,
            })
            .map(|(at, ())| at)
            .next();
        let presses: Vec<_> = inputs
            .sent()
            .iter()
            .filter(|a| {
                a.accepted && matches!(a.input, Input::Dtmf { input: DtmfInput::Digit, .. })
            })
            .collect();
        // The person rejoining or leaving ends the room's participant that pressed. The room link
        // drops the events of a participant replaced before it took them, a press with them: a
        // press is owed only when the session was listening to the person as they left.
        let links: Vec<_> = r
            .journal_where(|step| match step {
                Step::Room(RoomStep::Link(RoomLinkStep::Linked)) => Some(true),
                Step::Room(RoomStep::Link(RoomLinkStep::Unlinked { .. })) => Some(false),
                _ => None,
            })
            .collect();
        let listened_as_they_left = |pressed: Observed| {
            let left = inputs.sent().iter().find(|a| {
                a.accepted
                    && pressed < a.at
                    && matches!(a.input, Input::Room { event: RoomIn::Linked | RoomIn::Unlinked })
            });
            left.is_none_or(|left| {
                links.iter().take_while(|(at, _)| *at < left.at).last().is_some_and(|(_, l)| *l)
            })
        };
        // The room link forwards a press on its own task: a close that tore room I/O down in the
        // driver's first turn after the press may beat it, as the room's events and a failing
        // provider race in a real call. A press the driver had a whole turn to take before the
        // session finished closing is owed, a close that is still draining included.
        let before_the_close = |pressed: Observed| {
            closing.is_none_or(|c| pressed < c)
                && closed.is_none_or(|d| {
                    inputs.driver_turn_after(pressed).is_some_and(|t| t.ended.is_some_and(|e| e < d))
                })
        };
        let heard: Vec<_> = journaled.iter().map(|(_, key)| *key).collect();
        let pressed_keys: Vec<_> = presses
            .iter()
            .map(|a| match a.input {
                Input::Dtmf { key, .. } => DtmfEvent::from_wire(key.digit()),
                _ => unreachable!(),
            })
            .collect();
        // The keys heard are the person's in order, skipping only presses dropped with their
        // participant, and no owed press is missing.
        let mut unheard = heard.iter().peekable();
        let mut stopped = false;
        let mut owed_missed = false;
        for (a, key) in presses.iter().zip(&pressed_keys) {
            let listened = listened_as_they_left(a.at);
            if !stopped && unheard.peek() == Some(&key) {
                unheard.next();
            } else if stopped || listened {
                stopped = true;
                owed_missed |= listened && before_the_close(a.at);
            }
        }
        let owed = presses.iter().filter(|a| listened_as_they_left(a.at) && before_the_close(a.at));
        let owed = owed.count();
        check.expect(
            Subject::Session,
            !owed_missed && unheard.next().is_none(),
            format!(
                "the session heard keys {heard:?}; the person pressed {pressed_keys:?} ({owed} owed \
                 before the close)"
            ),
        );
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
            for &(when, key) in &journaled {
                if when <= started || when >= at || ended_at.is_some_and(|e| when >= e) || stop_key
                {
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
        let pressed = DtmfEvent::from_wire(k.digit());
        self.sent(key(k))
            .journal(Step::Room(RoomStep::Link(RoomLinkStep::KeyPressed { key: pressed })))
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
        setup: Setup {
            room: Some(RoomSetup { close_on_disconnect: false, delete_room_on_close: false }),
            ..Setup::default()
        },
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

fn guest_key(key: DtmfKey) -> Input {
    Input::Dtmf {
        input: DtmfInput::GuestDigit,
        key,
        digits: Count(0),
        confirm: false,
        timeout: Millis(0),
    }
}

/// A guest's press is not the session's to hear.
fn a_guest_is_not_heard() -> Recording {
    collecting(0).sent(guest_key(DtmfKey::Two)).press(DtmfKey::One).result(0, complete("1")).done()
}

/// The person pressed a key the session never heard.
fn a_press_dropped() -> Recording {
    collecting(0).sent(key(DtmfKey::One)).done()
}

/// The session closed, room I/O with it, in the driver's turn that would have taken the press.
fn a_press_racing_the_close() -> Recording {
    collecting(0)
        .sent(key(DtmfKey::One))
        .sent(Input::Advance { ms: Millis(20) })
        .journal(error_close())
        .journal(closed())
        .sent(Input::Advance { ms: Millis(20) })
        .done()
}

/// The driver had a whole turn to take the press before its close, and never heard it.
fn a_press_dropped_before_the_close() -> Recording {
    collecting(0)
        .sent(key(DtmfKey::One))
        .sent(Input::Advance { ms: Millis(20) })
        .sent(Input::Advance { ms: Millis(20) })
        .journal(error_close())
        .journal(closed())
        .done()
}

/// A draining close keeps room I/O up until it is done: a press right before it is still owed.
fn a_press_dropped_before_a_draining_close() -> Recording {
    let reason = lk_agents_abi::voice::events::CloseReason::UserInitiated;
    collecting(0)
        .sent(key(DtmfKey::One))
        .sent(Input::Advance { ms: Millis(20) })
        .journal(Step::Close(CloseStep::Requested { reason, drain: true }))
        .sent(Input::Advance { ms: Millis(20) })
        .done()
}

fn linked() -> Step {
    Step::Room(RoomStep::Link(RoomLinkStep::Linked))
}

/// The person rejoined before the room link took their first join: that participant's press left
/// with it.
fn a_press_left_with_its_participant() -> Recording {
    collecting(0)
        .sent(key(DtmfKey::One))
        .sent(Input::Room { event: RoomIn::Linked })
        .journal(linked())
        .done()
}

/// The session was listening to the person when they pressed and rejoined, and never heard it.
fn a_press_dropped_before_a_rejoin() -> Recording {
    collecting(0)
        .journal(linked())
        .sent(key(DtmfKey::One))
        .sent(Input::Room { event: RoomIn::Linked })
        .journal(linked())
        .done()
}

fn closed() -> Step {
    let reason = lk_agents_abi::voice::events::CloseReason::Error;
    Step::Emitted(SessionEvent::Close { reason, error: None })
}

fn error_close() -> Step {
    let reason = lk_agents_abi::voice::events::CloseReason::Error;
    Step::Close(CloseStep::Requested { reason, drain: false })
}

/// A guest's press reached the session.
fn a_guests_press_heard() -> Recording {
    let heard = Step::Room(RoomStep::Link(RoomLinkStep::KeyPressed { key: Some(DtmfEvent::Two) }));
    collecting(0).sent(guest_key(DtmfKey::Two)).journal(heard).done()
}
