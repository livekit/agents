//! A replaced output instance receives nothing more.
//!
//! Attaching an output creates a new instance; the case keeps the old one's identity, so a call
//! that lands on it after the replacement is buffered work escaping onto a sink the host has
//! detached. Each kind is tracked separately: replacing the audio output says nothing about the
//! transcription one.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    record::{Recording, Sink, Tag},
};

pub struct OutputReplacementIsClean;

impl Rule for OutputReplacementIsClean {
    const CONTRACT: Contract = Contract {
        id: "output_replacement_is_clean",
        guarantee: "Once an output of a kind is replaced, the replaced instance receives no further calls.",
        examples: &[calls_stop_at_replacement],
        counterexamples: &[captured_after_replacement, flushed_after_replacement],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        for output in host.outputs() {
            // The next instance of the same kind; until one exists this output is still current.
            let Some(replaced_at) = output.replaced_at() else { continue };
            for (at, sink) in output.received() {
                // Anything reaching this instance after its replacement was attached is output the
                // host had already taken away.
                check.expect(
                    &output,
                    at < replaced_at,
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

/// The first instance is written to, then replaced, then only the new one is written to.
fn calls_stop_at_replacement() -> Recording {
    Authored::new()
        .output(true)
        .sink(Sink::Flush { output: 0, segment: 0 })
        .output(true)
        .sink(Sink::Flush { output: 1, segment: 1 })
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

/// A segment is flushed on the old instance after the new one was attached.
fn flushed_after_replacement() -> Recording {
    Authored::new().output(false).output(false).sink(Sink::TextFlush { output: 0 }).done()
}
