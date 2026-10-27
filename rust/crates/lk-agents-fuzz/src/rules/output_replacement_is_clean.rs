//! A replaced output instance receives nothing more, but what a step already running owes it.
//!
//! Attaching an output creates a new instance; the case keeps the old one's identity, so a call
//! that lands on it after the replacement is buffered work escaping onto a sink the host has
//! detached. Each kind is tracked separately: replacing the audio output says nothing about the
//! transcription one.
//!
//! Two calls are the reference's own, so they are allowed. A step running when the output was
//! replaced read the output at its start and finishes on it (`_pipeline_reply_task_impl`,
//! `_tts_task`, `_realtime_generation_task_impl` take `self._session.output` once). And a text
//! output gets a closing flush from the synchroniser that wrapped it each time that synchroniser
//! closes a segment's writer on it (`_capture_task`'s `finally`): once when the swap rotates it,
//! once when the session closes it. A swap made while the session closes owes both at the close.
use lk_agents_voice::{
    events::SessionEvent,
    agent_session::effects::GateOp,
    trace::{InterruptStep, PlayoutStep, SpeechStep, Step, ToolStep},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    input::Input,
    record::{Observed, Recording, Sink, Tag},
};

pub struct OutputReplacementIsClean;

impl Rule for OutputReplacementIsClean {
    const CONTRACT: Contract = Contract {
        id: "output_replacement_is_clean",
        guarantee: "Once an output of a kind is replaced, the replaced instance receives no further calls but those of steps already running and a synchroniser's closing flushes.",
        examples: &[calls_stop_at_replacement, replaced_while_closing],
        counterexamples: &[captured_after_replacement, flushed_after_replacement],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        let steps = step_windows(r);
        let close = r.inputs_view().close_asked_at().unwrap_or(Observed::END);
        for output in host.outputs() {
            // The next instance of the same kind; until one exists this output is still current.
            let Some(replaced_at) = output.replaced_at() else { continue };
            // When the driver swapped the new one in: until then this one was still current. Its
            // own command's swap, not an earlier attach's handled in the same burst (seed 49224).
            let asked = r
                .inputs_view()
                .attach_received(output.is_audio(), replaced_at)
                .unwrap_or(replaced_at);
            let swapped = r
                .internal_view()
                .steps()
                .iter()
                .find(|(at, step)| {
                    *at >= asked
                        && matches!(step, Step::Playout(PlayoutStep::OutputReplaced { .. }))
                })
                .map_or(replaced_at, |(at, _)| *at);
            // The steps that had begun and not ended at the swap, and when the last of them ended:
            // until then the old instance is theirs.
            let running = steps
                .iter()
                .filter(|(from, to)| *from < swapped && to.is_none_or(|to| to > swapped))
                .map(|(_, to)| to.unwrap_or(Observed::END))
                .max();
            // A pause the old instance held is released on it as the swap happens: the speech it
            // was paused under keeps that instance.
            let released = r.internal_view().steps().iter().any(|(at, step)| {
                *at >= replaced_at
                    && *at <= swapped
                    && matches!(
                        step,
                        Step::Interrupt(InterruptStep::PlayoutGate { op: GateOp::Resume, .. })
                    )
            });
            let mut release_owed = released && output.is_audio();
            // One closing flush for the swap's rotation and one for the session's close, text
            // outputs only: before the close only the swap's is owed, after it either may be.
            let text = !output.is_audio();
            let (mut swap_flush, mut close_flush) = (text, text);
            for (at, sink) in output.received() {
                let owed = at < swapped
                    // The synchroniser releases a step's heard words as its playback finishes,
                    // reaching the text output in that same instant (seed 88522).
                    || running.is_some_and(|end| {
                        at <= end || (!output.is_audio() && at.at_ms == end.at_ms)
                    })
                    || (matches!(sink, Sink::Resumed { .. }) && std::mem::take(&mut release_owed))
                    || (matches!(sink, Sink::TextFlush { .. })
                        && if at < close {
                            std::mem::take(&mut swap_flush)
                        } else {
                            std::mem::take(&mut close_flush) || std::mem::take(&mut swap_flush)
                        });
                // Anything else reaching this instance after its replacement was attached is
                // output the host had already taken away.
                check.expect(
                    &output,
                    owed,
                    format!(
                        "output {} (audio: {}) received {sink:?} at seq {} after being replaced at seq {}",
                        output.nth(),
                        output.is_audio(),
                        at.seq,
                        replaced_at.seq
                    ),
                );
            }
        }
    }
}

/// Every step the journal shows, from the reply task that began it (the speech's creation, or a
/// tool step's start), which is where the reference reads the outputs, to the step's or the
/// speech's end (`None` while it had not ended).
fn step_windows(r: &Recording) -> Vec<(Observed, Option<Observed>)> {
    let mut open: std::collections::BTreeMap<u64, Observed> = Default::default();
    let mut windows = Vec::new();
    for (at, step) in r.internal_view().steps() {
        match step {
            Step::Speech(SpeechStep::ReplyCreated { speech })
            | Step::Tool(ToolStep::StepStarted { speech, .. }) => {
                open.entry(speech.0).or_insert(*at);
            }
            Step::Speech(SpeechStep::StepSpoken { speech, .. })
            | Step::Emitted(SessionEvent::SpeechDone { speech, .. }) => {
                if let Some(from) = open.remove(&speech.0) {
                    windows.push((from, Some(*at)));
                }
            }
            _ => {}
        }
    }
    windows.extend(open.into_values().map(|from| (from, None)));
    windows
}

/// The first instance is written to, then replaced, then only the new one is written to.
fn calls_stop_at_replacement() -> Recording {
    Authored::new()
        .output(true)
        .sink(Sink::Flush { output: 0, segment: 0 })
        .output(true)
        .sink(Sink::Flush { output: 1, segment: 1 })
        .done()
}

/// The transcription output replaced after the close was asked for: the swap's rotation and the
/// close each flush the old instance once. Catches: the rotation's flush counted as the close's
/// (survey seed 4496).
fn replaced_while_closing() -> Recording {
    Authored::new()
        .output(false)
        .sent(Input::Close { drain: true })
        .output(false)
        .sink(Sink::TextFlush { output: 0 })
        .sink(Sink::TextFlush { output: 0 })
        .done()
}

/// Audio reaches the old instance after the new one was attached.
fn captured_after_replacement() -> Recording {
    Authored::new()
        .output(true)
        .output(true)
        .sink(Sink::Audio { output: 0, tags: vec![Tag::Synthesized { tts: 0, chunk: 0 }] })
        .done()
}

/// A segment is flushed on the old instance after the new one was attached, twice before any
/// close: one closing flush is the synchroniser's rotation, a second is not.
fn flushed_after_replacement() -> Recording {
    Authored::new()
        .output(false)
        .output(false)
        .sink(Sink::TextFlush { output: 0 })
        .sink(Sink::TextFlush { output: 0 })
        .done()
}
