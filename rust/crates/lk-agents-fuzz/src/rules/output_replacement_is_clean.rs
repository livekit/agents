//! A retained attachment may finish its old segment; retirement ends that permission.
//!
//! Physical attachment callbacks are the boundary facts. A host object can have overlapping
//! attachments, so one detach cannot retire a different attachment of that same object.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    input::Input,
    record::{Recording, Sink, Tag},
};

pub struct OutputReplacementIsClean;

impl Rule for OutputReplacementIsClean {
    const CONTRACT: Contract = Contract {
        id: "output_replacement_is_clean",
        guarantee: "A replaced output receives calls only while a physical attachment still retains it; no call reaches it after its last attachment retires.",
        examples: &[calls_stop_at_replacement, replaced_while_closing, overlapping_attachments],
        counterexamples: &[
            captured_after_replacement,
            flushed_after_replacement,
            called_after_retirement,
        ],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for output in r.host_view().outputs() {
            let Some(replaced_at) = output.replaced_at() else { continue };
            for (at, sink) in output.received() {
                let active = output
                    .attachments()
                    .take_while(|(event, _)| *event < at)
                    .fold(0i32, |count, (_, attached)| count + if attached { 1 } else { -1 });
                check.expect(
                    &output,
                    at < replaced_at || active > 0,
                    format!(
                        "output {} (audio: {}) received {sink:?} at seq {} after replacement at seq {} with {active} retained attachments; callbacks {:?}",
                        output.nth(), output.is_audio(), at.seq, replaced_at.seq, output.attachments().collect::<Vec<_>>()
                    ),
                );
            }
        }
    }
}

fn calls_stop_at_replacement() -> Recording {
    Authored::new()
        .output(true)
        .sink(Sink::Flush { output: 0, segment: 0, ms: 10 })
        .output(true)
        .sink(Sink::Flush { output: 1, segment: 1, ms: 10 })
        .done()
}

/// Closing drains an admitted old destination before its physical attachment retires.
fn replaced_while_closing() -> Recording {
    Authored::new()
        .output(false)
        .output_attachment(0, true)
        .sent(Input::Close { drain: true })
        .output(false)
        .sink(Sink::TextFlush { output: 0 })
        .sink(Sink::TextFlush { output: 0 })
        .output_attachment(0, false)
        .done()
}

fn overlapping_attachments() -> Recording {
    Authored::new()
        .output(false)
        .output_attachment(0, true)
        .output_attachment(0, true)
        .output(false)
        .output_attachment(0, false)
        .sink(Sink::TextFlush { output: 0 })
        .output_attachment(0, false)
        .done()
}

fn captured_after_replacement() -> Recording {
    Authored::new()
        .output(true)
        .output(true)
        .sink(Sink::Audio { output: 0, tags: vec![Tag::Synthesized { tts: 0, chunk: 0 }] })
        .done()
}

fn flushed_after_replacement() -> Recording {
    Authored::new()
        .output(false)
        .output(false)
        .sink(Sink::TextFlush { output: 0 })
        .sink(Sink::TextFlush { output: 0 })
        .done()
}

fn called_after_retirement() -> Recording {
    Authored::new()
        .output(false)
        .output_attachment(0, true)
        .output(false)
        .output_attachment(0, false)
        .sink(Sink::TextFlush { output: 0 })
        .done()
}
