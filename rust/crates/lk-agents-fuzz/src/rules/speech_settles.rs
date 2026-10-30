//! Every speech is done once the test has answered everything the driver was waiting on.
//!
//! While the test is still talking, a speech may legitimately wait on it. The test then answers
//! every request, synthesis, playout, hook and hold, lets the driver's timers fire, and keeps
//! going until nothing new appears. At that instant the driver waits on nobody but itself, so a
//! speech created before it and not yet done was waiting on nothing: it was stuck, and only the
//! close will end it.
use lk_agents_voice::agent_session::effects::{GateCause, GateOp};
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::{InterruptStep, SpeechStep, Step, ToolStep, TurnStep};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    input::Input,
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
            if speech.is_speculative()
                && !speech.is_scheduled()
                && !turn_ended(r, &speech, answered)
            {
                continue;
            }
            // A speech held at its launch waits for the person to stop, as the reference's reply
            // waits for the user's silence. When the driver no longer takes the recogniser's end
            // (the turn detection left `stt` mid-utterance) and hears them speaking still, no
            // answer of the case's ends that wait.
            if held_at_launch(r, &speech, answered)
                && (end_unheard(r, answered) || never_stopped(r, answered))
            {
                continue;
            }
            // Queued behind such a speech, never let go: it waits on the same person.
            if let Some(holder) = floor_holder(r, answered)
                && holder != speech.id
                && !authorized(r, speech.id, answered)
                && r.host_view()
                    .speeches()
                    .iter()
                    .any(|h| h.id == holder && held_at_launch(r, h, answered))
                && (end_unheard(r, answered) || never_stopped(r, answered))
            {
                continue;
            }
            // A speech whose tool is still running once every tool was answered is in
            // `ctx.foreground()`, waiting for idle, which the same unheard end holds off.
            if tool_running(r, speech.id, answered) && end_unheard(r, answered) {
                continue;
            }
            let done = speech.done_at().is_some_and(|done| done <= answered);
            check.expect(
                speech,
                done,
                format!("still running after everything was answered ({speech:?})"),
            );
        }
    }
}

/// Who held the floor at `answered`, by the last change at or before it.
fn floor_holder(r: &Recording, answered: Observed) -> Option<u64> {
    r.journal_where(|step| match step {
        Step::Emitted(SessionEvent::FloorChanged { holder }) => Some(holder.map(|h| h.0)),
        _ => None,
    })
    .filter(|(at, _)| *at <= answered)
    .last()
    .and_then(|(_, holder)| holder)
}

/// Whether the scheduler let `speech`'s current step go by `answered`: a tool step started after
/// its last authorization waits for the floor again (seed 17597).
fn authorized(r: &Recording, speech: u64, answered: Observed) -> bool {
    let mut go = false;
    for (_, step) in r.internal_view().steps().iter().filter(|(at, _)| *at <= answered) {
        match step {
            Step::Speech(SpeechStep::Authorized { speech: s }) if s.0 == speech => go = true,
            Step::Tool(ToolStep::StepStarted { speech: s, .. }) if s.0 == speech => go = false,
            _ => {}
        }
    }
    go
}

/// The speech's launch hold was taken and not released by `answered`.
fn held_at_launch(r: &Recording, speech: &SpeechView, answered: Observed) -> bool {
    let id = speech.id;
    r.internal_view()
        .steps()
        .iter()
        .filter(|(at, _)| *at <= answered)
        .filter_map(|(_, step)| match step {
            Step::Interrupt(InterruptStep::PlayoutGate {
                speech,
                op,
                cause: GateCause::Launch,
            }) if speech.0 == id => Some(*op == GateOp::Pause),
            _ => None,
        })
        .next_back()
        .unwrap_or(false)
}

/// The person the driver last heard start, on a detector's word (the VAD's, a realtime server's),
/// is speaking still at `answered`: the case never sent that detector's end (`VadEnd`, the
/// server's `speech_stopped`), here because it took the microphone away mid-utterance. A speech
/// held at its launch waits on that person, as the reference's does.
fn never_stopped(r: &Recording, answered: Observed) -> bool {
    let Some((began, true)) = r
        .journal_where(|step| match step {
            Step::Emitted(SessionEvent::StartOfSpeech(ev)) => Some(ev.from_vad),
            Step::Emitted(SessionEvent::EndOfSpeech(_)) => Some(false),
            _ => None,
        })
        .filter(|(at, _)| *at <= answered)
        .last()
    else {
        return false;
    };
    !r.applied().iter().any(|a| {
        a.accepted
            && a.at > began
            && a.at <= answered
            && matches!(
                a.input,
                Input::VadEnd { .. }
                    | Input::RealtimeEvent { event: crate::input::RtEvent::SpeechStopped, .. }
            )
    })
}

/// The person the driver last heard start, on the recogniser's word, has not ended by
/// `answered`, though a whole utterance was started and ended since: the driver does not take the
/// recogniser's end of speech any more.
fn end_unheard(r: &Recording, answered: Observed) -> bool {
    let Some((began, true)) = r
        .journal_where(|step| match step {
            Step::Emitted(SessionEvent::StartOfSpeech(ev)) => Some(!ev.from_vad),
            Step::Emitted(SessionEvent::EndOfSpeech(_)) => Some(false),
            _ => None,
        })
        .filter(|(at, _)| *at <= answered)
        .last()
    else {
        return false;
    };
    let accepted = r.applied().iter().filter(|a| a.accepted && a.at > began && a.at <= answered);
    let restarted = accepted.clone().find(|a| matches!(a.input, Input::SttStart { .. }));
    restarted.is_some_and(|start| {
        accepted.clone().any(|a| a.at > start.at && matches!(a.input, Input::SttEnd { .. }))
    })
}

/// One of the speech's tool calls started by `answered` and had not ended.
fn tool_running(r: &Recording, speech: u64, answered: Observed) -> bool {
    let mut running = std::collections::BTreeSet::new();
    for (at, step) in r.internal_view().steps() {
        if *at > answered {
            break;
        }
        match step {
            Step::Tool(ToolStep::CallStarted { speech: s, call_id, .. }) if s.0 == speech => {
                running.insert(call_id.clone());
            }
            Step::Tool(ToolStep::CallEnded { speech: s, call_id, .. }) if s.0 == speech => {
                running.remove(call_id);
            }
            _ => {}
        }
    }
    !running.is_empty()
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
