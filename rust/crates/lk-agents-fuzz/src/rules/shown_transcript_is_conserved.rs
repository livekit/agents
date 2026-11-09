use lk_agents_voice::{
    agent_session::machine::activity::scheduling::SpeechId,
    trace::{PlayoutStep, Step},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    record::{Handle, Recording, TranscriptSetup},
};

// Pacing changes when a word appears, never which. Whitespace is ignored, as the tokeniser may
// regroup it. Avatar outputs are skipped (their late reports land on the next segment), as are
// failed voices, whose words go out unpaced.

pub struct ShownTranscriptIsConserved;

impl Rule for ShownTranscriptIsConserved {
    const CONTRACT: Contract = Contract {
        id: "shown_transcript_is_conserved",
        guarantee: "A speech that completed with its audio played through a synchroniser showed exactly the text forwarded for it, in order.",
        examples: &[all_shown],
        counterexamples: &[a_word_lost, a_word_repeated],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        if !r.transcript().synchronized || r.transcript().late_reports {
            return;
        }
        let (inputs, host, internal) = (r.inputs_view(), r.host_view(), r.internal_view());
        if inputs.voice_failed() {
            return;
        }
        for speech in host.speeches() {
            let trace = internal.speech(speech.id);
            let until = speech.done_at().unwrap_or(r.now());
            if !speech.completed_normally()
                || !trace.audio_played()
                || inputs.outputs_touched_by(until)
            {
                continue;
            }
            // A step the playout cut (the handle may still complete, a realtime step with it).
            let cut = trace.steps().any(|(_, s)| {
                matches!(s, Step::Playout(PlayoutStep::PlaybackFinished { interrupted: true, .. }))
            });
            if cut {
                continue;
            }
            let forwarded = squash(&trace.transcript_forwarded().concat());
            // A word shown before the speech forwarded any is an earlier speech's: a cut segment
            // the sink reported played out once the next one was playing.
            let first = trace.steps().find_map(|(at, s)| {
                matches!(s, Step::Playout(PlayoutStep::TranscriptForwarded { .. })).then_some(at)
            });
            let shown: String = trace
                .transcript_shown()
                .iter()
                .filter(|w| first.is_some_and(|f| w.at > f))
                .map(|w| w.text)
                .collect();
            let shown = squash(&shown);
            check.expect(
                speech,
                shown == forwarded,
                format!("showed {shown:?}, was forwarded {forwarded:?}"),
            );
        }
    }
}

fn squash(text: &str) -> String {
    text.split_whitespace().collect()
}

fn forwarded(text: &str) -> Step {
    Step::Playout(PlayoutStep::TranscriptForwarded {
        speech: SpeechId(1),
        text: text.into(),
        start_time: None,
    })
}

fn shown(text: &str) -> Step {
    Step::Playout(PlayoutStep::TranscriptShown {
        speech: SpeechId(1),
        text: text.into(),
        start: 0.0,
        end: 0.1,
        played: std::time::Duration::from_millis(50),
    })
}

fn played() -> Step {
    Step::Playout(PlayoutStep::AudioPlayed { speech: SpeechId(1), frames: 1 })
}

fn synchronized() -> Authored {
    Authored::new()
        .transcript(TranscriptSetup { synchronized: true, ..TranscriptSetup::default() })
        .journal(forwarded("one two "))
        .journal(forwarded("three."))
        .journal(played())
}

fn all_shown() -> Recording {
    synchronized()
        .journal(shown("one"))
        .journal(shown(" two "))
        .journal(shown("three."))
        .speech(1, Handle::finished())
        .done()
}

fn a_word_lost() -> Recording {
    synchronized()
        .journal(shown("one"))
        .journal(shown("three."))
        .speech(1, Handle::finished())
        .done()
}

fn a_word_repeated() -> Recording {
    synchronized()
        .journal(shown("one"))
        .journal(shown(" two"))
        .journal(shown(" two"))
        .journal(shown(" three."))
        .speech(1, Handle::finished())
        .done()
}
