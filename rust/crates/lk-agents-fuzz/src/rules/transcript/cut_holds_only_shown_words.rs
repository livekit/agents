use lk_agents_voice::{
    agent_session::machine::activity::scheduling::SpeechId,
    trace::{
        GenerationStep, InterruptCause, InterruptStep, PlayoutStep, SpeechStep, Step, ToolStep,
    },
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    record::{Handle, Recording, TranscriptSetup},
};

// `cut_keeps_heard_words` judges a cut the sink reported; this one also holds when the sink never
// reported it (a lost or late clear report), where the synchroniser's words at the clear are the
// cut's text. A word still playing at the cut was shown, not heard, so the message is a prefix of
// the words shown, not all of them. Avatar outputs are skipped: their late reports land on the
// next segment.

pub struct CutHoldsOnlyShownWords;

impl Rule for CutHoldsOnlyShownWords {
    const CONTRACT: Contract = Contract {
        id: "cut_holds_only_shown_words",
        guarantee: "A synthesised speech cut through a synchroniser after its transcript started is remembered as no more than the words the transcript showed, reported or not.",
        examples: &[remembered_as_shown_unreported, remembered_as_heard_unreported],
        counterexamples: &[remembered_whole_unreported, remembered_past_the_shown_words],
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
            let (mut shown, mut remembered) = (String::new(), None::<String>);
            for (_, step) in trace.steps() {
                match step {
                    Step::Playout(PlayoutStep::TranscriptShown { text, .. }) => {
                        shown.push_str(text)
                    }
                    // A tool follow-up: the steps before it committed their own text.
                    Step::Tool(ToolStep::StepStarted { .. }) => shown.clear(),
                    Step::Speech(SpeechStep::ReplyFinished { text, .. }) => {
                        remembered.get_or_insert_default().push_str(text)
                    }
                    _ => {}
                }
            }
            // Cut before the transcript showed a word: the speech's own text stands, as when its
            // audio carried no words.
            let Some(remembered) = remembered.filter(|_| !shown.is_empty()) else { continue };
            // history keeps the transcript plain
            let shown = lk_agents_voice::transcription::plain(&shown);
            let (remembered, shown) = (words(&remembered), words(&shown));
            check.expect(
                speech,
                shown.starts_with(&remembered),
                format!("remembered {remembered:?}, shown {shown:?}"),
            );
        }
    }
}

fn words(text: &str) -> Vec<&str> {
    text.split_whitespace().collect()
}

fn shown(text: &str, start: f64) -> Step {
    Step::Playout(PlayoutStep::TranscriptShown {
        speech: SpeechId(1),
        text: text.into(),
        start,
        end: start + 0.1,
        played: std::time::Duration::from_secs_f64(start + 0.05),
        spoken: true,
    })
}

/// "one" and "two" shown, then cut with no report from the sink.
fn cut_unreported(remembered: &str) -> Recording {
    Authored::new()
        .transcript(TranscriptSetup { synchronized: true, ..TranscriptSetup::default() })
        .journal(Step::Generation(GenerationStep::TtsRequested {
            speech: SpeechId(1),
            segment: 0,
            text: "one two three".into(),
        }))
        .journal(shown("one", 0.0))
        .journal(shown(" two", 0.1))
        .journal(Step::Speech(SpeechStep::ReplyFinished {
            speech: SpeechId(1),
            text: remembered.into(),
            interrupted: true,
        }))
        .speech(1, Handle::finished().interrupted())
        .done()
}

fn remembered_as_shown_unreported() -> Recording {
    cut_unreported("one two")
}

fn remembered_as_heard_unreported() -> Recording {
    cut_unreported("one")
}

fn remembered_whole_unreported() -> Recording {
    cut_unreported("one two three")
}

fn remembered_past_the_shown_words() -> Recording {
    cut_unreported("two three")
}
