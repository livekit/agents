use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    record::{Recording, Sink},
};

// What the model writes that is not for the person -- its reasoning, the voice's markup -- is read
// once and never shown. Under a dialect (`Setup::tts_markup`) the fake model writes both into every
// reply, so a leg that skipped the reading shows them here.

pub struct NoMarkupReachesASink;

/// What only unread model text carries.
const FRAGMENTS: [&str; 5] = ["<expr", "</expr", "<think", "</think", "label=\""];

impl Rule for NoMarkupReachesASink {
    const CONTRACT: Contract = Contract {
        id: "no_markup_reaches_a_sink",
        guarantee: "No transcription output is shown the model's reasoning or a markup tag: every leg shows the text as read once.",
        examples: &[plain_text_is_shown],
        counterexamples: &[a_marker_is_shown, reasoning_is_shown],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for output in r.host_view().outputs() {
            if output.is_audio() {
                continue;
            }
            for (_, sink) in output.received() {
                let Sink::Text { text, .. } = sink else { continue };
                let lower = text.to_lowercase();
                let found = FRAGMENTS.iter().find(|f| lower.contains(*f));
                check.expect_with(&output, found.is_none(), || {
                    format!("transcription output {} was shown {text:?}", output.nth())
                });
            }
        }
    }
}

fn shown(text: &str) -> Recording {
    Authored::new()
        .output(false)
        .output_attachment(0, true)
        .sink(Sink::Text { output: 0, text: text.into() })
        .sink(Sink::TextFlush { output: 0 })
        .done()
}

fn plain_text_is_shown() -> Recording {
    shown("Sure, **3 < 5** and a<b, 🎉")
}

fn a_marker_is_shown() -> Recording {
    shown("<expr type=\"expression\" label=\"warm\"/> Sure")
}

fn reasoning_is_shown() -> Recording {
    shown("<think>the user wants X</think>Sure")
}
