//! Audio reaching the sink is exactly what was produced for it: every sample carries a known
//! source, and each source's chunks arrive in order without gaps or repeats.
//!
//! The two sources are synthesis streams and the audio a case feeds into a `say`. A truncated
//! tail is legitimate, because that is what an interruption looks like. A gap, a repeat, a
//! reorder, or a chunk nobody produced is not.
use std::collections::BTreeMap;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Kind, Recording, Sink, Tag},
};

pub struct AudioConservation;

impl Rule for AudioConservation {
    const CONTRACT: Contract = Contract {
        id: "audio_conservation",
        guarantee: "Sink audio is a per-source prefix of produced audio, in order, with no unknown samples.",
        examples: &[in_order],
        counterexamples: &[unknown_sample, reordered_chunks, unproduced_chunk],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        let inference = r.inference_view();

        // The chunk each source is expected to deliver next.
        let mut expected: BTreeMap<(Kind, u32), u16> = BTreeMap::new();
        for pushed in host.audio_pushed() {
            let at = pushed.at;
            for tag in pushed.tags {
                let (source, chunk) = match *tag {
                    Tag::Synthesized { tts, chunk } => ((Kind::Tts, tts), chunk),
                    Tag::Fed { stream, chunk } => ((Kind::Stream, stream), chunk),
                    // The person's own voice coming back out of the speaker is an echo, never a chunk
                    // any source produced for playback.
                    Tag::Mic { mic, .. } => {
                        check.expect(
                            Subject::Resource(Kind::Mic, mic),
                            false,
                            format!("microphone {mic} audio reached the output (seq {})", at.seq),
                        );
                        continue;
                    }
                    // Every sample was tagged where it was produced, so an untagged one came from
                    // somewhere this run does not know about.
                    Tag::Unknown(sample) => {
                        check.expect(
                            Subject::Session,
                            false,
                            format!("sink received sample {sample} nobody produced (seq {})", at.seq),
                        );
                        continue;
                    }
                };
                let subject = Subject::Resource(source.0, source.1);

                // One source's chunks reach the sink in production order: no gap, repeat or reorder.
                let next = expected.entry(source).or_default();
                check.expect_with(subject.clone(), chunk == *next, || {
                    format!("{tag:?} arrived, expected chunk {next} of that source (seq {})", at.seq)
                });

                // And a synthesized chunk was really produced by that stream, not invented downstream.
                if let Tag::Synthesized { tts, .. } = *tag {
                    check.expect_with(subject, inference.synthesized(tts, chunk), || {
                        format!("{tag:?} was never produced by the synthesis fake")
                    });
                }
                *next = chunk + 1;
            }
        }
    }
}

fn heard(tag: Tag) -> Sink {
    Sink::Audio { output: 0, tags: vec![tag] }
}

fn synthesized(chunk: u16) -> Tag {
    Tag::Synthesized { tts: 0, chunk }
}

/// Two chunks produced, the same two heard, in order.
fn in_order() -> Recording {
    Authored::new()
        .tts()
        .produced(0, 0, "w0")
        .produced(0, 1, "w1")
        .sink(heard(synthesized(0)))
        .sink(heard(synthesized(1)))
        .done()
}

/// A sample with no source reached the sink.
fn unknown_sample() -> Recording {
    Authored::new().sink(heard(Tag::Unknown(77))).done()
}

/// The second chunk of a stream was heard before the first.
fn reordered_chunks() -> Recording {
    Authored::new()
        .tts()
        .produced(0, 0, "w0")
        .produced(0, 1, "w1")
        .sink(heard(synthesized(1)))
        .sink(heard(synthesized(0)))
        .done()
}

/// A chunk attributed to a stream that never produced it.
fn unproduced_chunk() -> Recording {
    Authored::new().tts().sink(heard(synthesized(0))).done()
}
