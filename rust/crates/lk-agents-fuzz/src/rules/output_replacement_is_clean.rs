//! A retained attachment may finish its old segment; retirement ends that permission.
//!
//! Physical attachment callbacks are the boundary facts. A host object can have overlapping
//! attachments, so one detach cannot retire a different attachment of that same object.
use lk_agents_voice::agent_activity::scheduling::SpeechKind;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    input::Input,
    record::{Handle, Recording, Sink, Tag},
};

pub struct OutputReplacementIsClean;

impl Rule for OutputReplacementIsClean {
    const CONTRACT: Contract = Contract {
        id: "output_replacement_is_clean",
        guarantee: "A replaced output receives calls only while a physical attachment still retains it; no call reaches it after its last attachment retires.",
        examples: &[
            calls_stop_at_replacement,
            replaced_while_closing,
            overlapping_attachments,
            retained_speech_resumes_its_output,
        ],
        counterexamples: &[
            resumed_after_its_speech_ended,
            captured_after_replacement,
            flushed_after_replacement,
            called_after_retirement,
        ],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        let speeches = host.speeches();
        for output in host.outputs() {
            let Some(replaced_at) = output.replaced_at() else { continue };
            // Work under way at the replacement keeps its binding: pause and resume target it
            // after another output is selected (voice-session.md, "Output handover policy").
            let retained = |at| {
                speeches.iter().any(|s| {
                    s.created_at().is_some_and(|c| c < replaced_at)
                        && s.done_at().is_none_or(|done| done >= at)
                })
            };
            for (at, sink) in output.received() {
                let active = output
                    .attachments()
                    .take_while(|(event, _)| *event < at)
                    .fold(0i32, |count, (_, attached)| count + if attached { 1 } else { -1 });
                let gate = matches!(sink, Sink::Paused { .. } | Sink::Resumed { .. });
                check.expect(
                    &output,
                    at < replaced_at || active > 0 || (gate && retained(at)),
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

/// A speech under way when the output was replaced is resumed on the output it was bound to.
fn retained_speech_resumes_its_output() -> Recording {
    Authored::new()
        .output(true)
        .speech_created(1, SpeechKind::Say)
        .speech(1, Handle::live())
        .output(true)
        .sink(Sink::Resumed { output: 0 })
        .done()
}

/// The old output resumed when nothing under way at the replacement remains.
fn resumed_after_its_speech_ended() -> Recording {
    Authored::new()
        .output(true)
        .speech_created(1, SpeechKind::Say)
        .speech(1, Handle::finished())
        .output(true)
        .sink(Sink::Resumed { output: 0 })
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
