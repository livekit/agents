//! A cut speech is remembered as the words the person heard: those shown whose audio had ended
//! where the playout stopped, no more and no fewer.
//!
//! With a synchroniser, the text an interrupted reply commits is its synchronised transcript (the
//! reference's `forwarded_text` out of `playback_finished`), not the text the model produced. For
//! each segment the synchroniser released words into, the heard part is every word of a segment
//! the sink reported played out whole, and of a cut one the words whose audio ended by the cut's
//! position (a word's audio ends where the next one's starts, if earlier): a word still playing
//! was shown, not heard. The text the reply's last step committed
//! (`ReplyFinished`) must say exactly that of the step's segments (whitespace aside).
//!
//! Judged for synthesised replies cut and committed with the synchroniser between the outputs
//! throughout (not an avatar's, whose late reports land on the next segment), the voice never
//! failed, every segment's end reported by the sink, and not cut by
//! the session's close (which closes the synchroniser first).
use lk_agents_voice::{
    agent_activity::scheduling::SpeechId,
    trace::{
        GenerationStep, InterruptCause, InterruptStep, PlayoutStep, SpeechStep, Step, ToolStep,
    },
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    record::{Handle, Recording, TranscriptSetup},
};

/// What two clocks' rounding may leave between a word's end and the cut, in seconds.
const EPSILON: f64 = 1e-6;

pub struct CutKeepsHeardWords;

impl Rule for CutKeepsHeardWords {
    const CONTRACT: Contract = Contract {
        id: "cut_keeps_heard_words",
        guarantee: "A synthesised speech cut through a synchroniser is remembered as exactly the words whose audio ended before the cut.",
        examples: &[remembered_as_heard],
        counterexamples: &[remembered_whole, remembered_with_the_playing_word],
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
            // Cut by the session's close: the synchroniser closed first, its transcript gone.
            let closing = trace.steps().any(|(_, s)| {
                matches!(
                    s,
                    Step::Interrupt(InterruptStep::Interrupted {
                        cause: InterruptCause::Closing,
                        ..
                    })
                )
            });
            if !speech.is_interrupted()
                || closing
                || trace.synthesis_requested_at().is_none()
                || inputs.outputs_touched_by(until)
            {
                continue;
            }
            let (mut heard, mut remembered) = (String::new(), None::<String>);
            let mut segment: Vec<(&str, f64)> = Vec::new();
            // A segment started and not yet reported finished; whether any started at all.
            let (mut open, mut played) = (false, false);
            for (_, step) in trace.steps() {
                match step {
                    Step::Playout(PlayoutStep::PlaybackStarted { .. }) => {
                        (open, played) = (true, true)
                    }
                    Step::Playout(PlayoutStep::TranscriptShown { text, start, end, .. }) => {
                        (open, played) = (true, true);
                        // One voice: the word before was over when this one began.
                        if let Some((_, before)) = segment.last_mut() {
                            *before = before.min(*start);
                        }
                        segment.push((text, *end))
                    }
                    Step::Playout(PlayoutStep::PlaybackFinished {
                        position, interrupted, ..
                    }) => {
                        open = false;
                        let cut = position.as_secs_f64() + EPSILON;
                        heard.extend(
                            segment
                                .drain(..)
                                .filter(|(_, end)| !interrupted || *end <= cut)
                                .map(|(text, _)| text),
                        );
                    }
                    // A tool follow-up: the steps before it committed their own text.
                    Step::Tool(ToolStep::StepStarted { .. }) => heard.clear(),
                    Step::Speech(SpeechStep::ReplyFinished { text, .. }) => {
                        remembered.get_or_insert_default().push_str(text)
                    }
                    _ => {}
                }
            }
            // A reply ended by the interruption watchdog commits nothing of its own; a segment
            // whose sink never reported its end is read at the playout's own count.
            // A reply cut before any of its audio started playing is no synchroniser's to tell.
            let Some(remembered) = remembered.filter(|_| !open && played) else { continue };
            let (remembered, heard) = (squash(&remembered), squash(&heard));
            check.expect(
                &speech,
                remembered == heard,
                format!("remembered {remembered:?}, heard {heard:?}"),
            );
        }
    }
}

fn squash(text: &str) -> String {
    text.split_whitespace().collect()
}

fn shown(text: &str, start: f64) -> Step {
    Step::Playout(PlayoutStep::TranscriptShown {
        speech: SpeechId(1),
        text: text.into(),
        start,
        end: start + 0.1,
        played: std::time::Duration::from_secs_f64(start + 0.05),
    })
}

/// "one" and "two" shown, the cut at 0.15 s: "one" ended at 0.1, "two" was still playing.
fn cut_after_two_words(remembered: &str) -> Recording {
    Authored::new()
        .transcript(TranscriptSetup { synchronized: true, ..TranscriptSetup::default() })
        .journal(Step::Generation(GenerationStep::TtsRequested {
            speech: SpeechId(1),
            segment: 0,
            text: "one two three".into(),
        }))
        .journal(shown("one", 0.0))
        .journal(shown(" two", 0.1))
        .journal(Step::Playout(PlayoutStep::PlaybackFinished {
            speech: SpeechId(1),
            position: std::time::Duration::from_millis(150),
            interrupted: true,
        }))
        .journal(Step::Speech(SpeechStep::ReplyFinished {
            speech: SpeechId(1),
            text: remembered.into(),
            interrupted: true,
        }))
        .speech(1, Handle::finished().interrupted())
        .done()
}

fn remembered_as_heard() -> Recording {
    cut_after_two_words("one")
}

fn remembered_whole() -> Recording {
    cut_after_two_words("one two three")
}

fn remembered_with_the_playing_word() -> Recording {
    cut_after_two_words("one two")
}
