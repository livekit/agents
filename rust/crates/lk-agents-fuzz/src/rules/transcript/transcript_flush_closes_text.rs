use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    record::{Recording, Sink},
};

// A flush ends one transcript segment for the text output (a room's final transcription, a
// host output's message): one with nothing in it is a segment the user sees and that never said
// anything.

pub struct TranscriptFlushClosesText;

impl Rule for TranscriptFlushClosesText {
    const CONTRACT: Contract = Contract {
        id: "transcript_flush_closes_text",
        guarantee: "A transcription output is flushed only to close a segment that had text: every flush follows text it received since its last flush.",
        examples: &[each_flush_closes_its_text],
        counterexamples: &[flushed_on_attach, flushed_twice],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for output in r.host_view().outputs() {
            if output.is_audio() {
                continue;
            }
            let mut open = false;
            for (at, sink) in output.received() {
                match sink {
                    Sink::Text { .. } => open = true,
                    Sink::TextFlush { .. } => {
                        check.expect_with(&output, std::mem::take(&mut open), || {
                            format!(
                                "transcription output {} was flushed at seq {} with no text since \
                                 its last flush",
                                output.nth(),
                                at.seq
                            )
                        });
                    }
                    _ => {}
                }
            }
        }
    }
}

fn each_flush_closes_its_text() -> Recording {
    Authored::new()
        .output(false)
        .output_attachment(0, true)
        .sink(Sink::Text { output: 0, text: "Hello".into() })
        .sink(Sink::TextFlush { output: 0 })
        .sink(Sink::Text { output: 0, text: "Bye".into() })
        .sink(Sink::TextFlush { output: 0 })
        .done()
}

/// Flushed as it was attached, before anything was said.
fn flushed_on_attach() -> Recording {
    Authored::new()
        .output(false)
        .output_attachment(0, true)
        .sink(Sink::TextFlush { output: 0 })
        .sink(Sink::Text { output: 0, text: "Hello".into() })
        .sink(Sink::TextFlush { output: 0 })
        .done()
}

/// Flushed again at the close with nothing more said.
fn flushed_twice() -> Recording {
    Authored::new()
        .output(false)
        .output_attachment(0, true)
        .sink(Sink::Text { output: 0, text: "Hello".into() })
        .sink(Sink::TextFlush { output: 0 })
        .sink(Sink::TextFlush { output: 0 })
        .done()
}
