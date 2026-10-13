//! A speech that asked for synthesis and finished normally had audio played.
//!
//! A liveness rule, and deliberately cause-agnostic. It does not say what went wrong, only that
//! the thing the speech existed to do never happened. A dropped chunk, a stalled pipe, a race that
//! loses the playback report, an output swapped at the wrong instant -- every one of those shows
//! up here as the same absence, which is what lets this catch failures nobody wrote a rule for.
use lk_agents_voice::{
    agent_activity::scheduling::SpeechId,
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
