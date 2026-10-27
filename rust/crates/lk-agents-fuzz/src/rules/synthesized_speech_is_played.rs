//! A speech that asked for synthesis and finished normally had audio played.
//!
//! A liveness rule, and deliberately cause-agnostic. It does not say what went wrong, only that
//! the thing the speech existed to do never happened. A dropped chunk, a stalled pipe, a race that
//! loses the playback report, an output swapped at the wrong instant -- every one of those shows
//! up here as the same absence, which is what lets this catch failures nobody wrote a rule for.
use lk_agents_voice::{
    agent_activity::scheduling::SpeechId,
    events::{ErrorSource, SessionEvent},
    trace::{GenerationStep, PlayoutStep, Step},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    record::{Handle, Recording},
};

pub struct SynthesizedSpeechIsPlayed;

impl Rule for SynthesizedSpeechIsPlayed {
    const CONTRACT: Contract = Contract {
        id: "synthesized_speech_is_played",
        guarantee: "A speech that requested synthesis of real text and completed normally had audio played.",
        examples: &[requested_and_played, interrupted_before_playing],
        counterexamples: &[requested_never_played],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let host = r.host_view();
        let internal = r.internal_view();
        // With no audio output attached nothing can be played, so the guarantee does not apply.
        if inputs.audio_output().ever_off() {
            return;
        }
        for speech in host.speeches() {
            // Only a speech that ran to its natural end owes us audio; an interrupted or failed one
            // is somebody else's rule.
            if !speech.completed_normally() {
                continue;
            }
            let trace = internal.speech(speech.id);
            let Some(asked) = trace.synthesis_requested_at() else { continue };
            // A voice that failed for good was surfaced as the speech's unrecoverable error: the
            // speech had nothing left to play with.
            let voice_lost = internal.steps().iter().any(|(_, step)| {
                matches!(step, Step::Emitted(SessionEvent::Error(ev))
                    if ev.source == ErrorSource::Tts && !ev.recoverable
                        && ev.speech.is_some_and(|s| s.0 == speech.id))
            });
            if voice_lost {
                continue;
            }
            // The output threw its playback away while the speech's audio waited in it: nothing
            // of it played, the speech none the wiser (a cut reports where playback stood).
            let done = speech.done_at().unwrap_or(r.now());
            let reset = inputs.accepted().any(|a| {
                a.at > asked && a.at <= done && matches!(a.input, crate::input::Input::SinkReset { .. })
            });
            // The same from the output's own side: it reset, or reported the speech's playback
            // cut that the driver never cut, as an avatar whose worker never answered a clear
            // does once its timeout fires (`_clear_buffer_task`'s fallback; seeds 1468, 18097).
            let cut_by_output = internal.steps().iter().any(|(at, step)| {
                *at > asked
                    && *at <= done
                    && matches!(step,
                        Step::Playout(PlayoutStep::SinkReset { speech: s })
                        | Step::Playout(PlayoutStep::PlaybackFinished { speech: s, interrupted: true, .. })
                            if s.0 == speech.id)
            });
            if reset || cut_by_output {
                continue;
            }
            check.expect(
                &speech,
                trace.audio_played(),
                format!(
                    "speech {} requested synthesis at seq {} and finished, but no audio was ever played",
                    speech.id, asked.seq
                ),
            );
        }
    }
}

fn requested(speech: u64) -> Step {
    Step::Generation(GenerationStep::TtsRequested {
        speech: SpeechId(speech),
        segment: 0,
        text: "Sure, I can help with that.".into(),
    })
}

fn played(speech: u64) -> Step {
    Step::Playout(PlayoutStep::AudioPlayed { speech: SpeechId(speech), frames: 12 })
}

/// Synthesis was requested, audio was played, the speech finished.
fn requested_and_played() -> Recording {
    Authored::new().journal(requested(1)).journal(played(1)).speech(1, Handle::finished()).done()
}

/// Interrupted before anything played: nothing is owed, so nothing is checked.
fn interrupted_before_playing() -> Recording {
    Authored::new()
        .journal(requested(1))
        .speech(1, Handle::finished().interrupted())
        .journal(requested(2))
        .journal(played(2))
        .speech(2, Handle::finished())
        .done()
}

/// Synthesis was requested and the speech finished normally, yet no audio was ever played.
fn requested_never_played() -> Recording {
    Authored::new().journal(requested(1)).speech(1, Handle::finished()).done()
}
