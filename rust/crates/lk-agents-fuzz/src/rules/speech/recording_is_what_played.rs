use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{AgentText, Input, Millis, Ref, SayInput},
    record::{RecorderCall, Recording, Sink},
};

// The recorder fake writes the real timeline as PCM. Room-only runs compare identity, timing
// and silence against the simulated source's own audible PCM, independently of recorder
// callbacks. Other outputs retain the callback-based hole and late-placement checks.

pub struct RecordingIsWhatPlayed;

impl Rule for RecordingIsWhatPlayed {
    const CONTRACT: Contract = Contract {
        id: "recording_is_what_played",
        guarantee: "The recording matches the room speaker's audible PCM, including silence and \
                    interrupted tails; other outputs retain every reported played stretch, and \
                    no played audio arrives after its place was written.",
        examples: &[checked],
        counterexamples: &[a_hole, dropped_late, unplayed_audio],
        probe: Some(a_recorded_reply),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for (at, sink) in r.sink() {
            let Sink::Recorder(call) = sink else { continue };
            match call {
                RecorderCall::Flush => check.expect(Subject::Session, true, String::new()),
                RecorderCall::Unheard { at_ms } => check.expect(
                    Subject::Session,
                    false,
                    format!(
                        "the recording is silent {at_ms} ms in, where audio played (seq {})",
                        at.seq
                    ),
                ),
                RecorderCall::Late { samples } => check.expect(
                    Subject::Session,
                    false,
                    format!("{samples} played samples reached the recording late (seq {})", at.seq),
                ),
                RecorderCall::Mismatch { at_ms, expected, actual } => check.expect(
                    Subject::Session,
                    false,
                    format!("recording differs from the room speaker at {at_ms} ms: expected {expected}, recorded {actual} (seq {})", at.seq),
                ),
                _ => {}
            }
        }
    }
}

fn checked() -> Recording {
    Authored::new().sink(Sink::Recorder(RecorderCall::Flush)).done()
}

fn a_hole() -> Recording {
    Authored::new()
        .sink(Sink::Recorder(RecorderCall::Flush))
        .sink(Sink::Recorder(RecorderCall::Unheard { at_ms: 980 }))
        .done()
}

fn dropped_late() -> Recording {
    Authored::new()
        .sink(Sink::Recorder(RecorderCall::Flush))
        .sink(Sink::Recorder(RecorderCall::Late { samples: 960 }))
        .done()
}

fn unplayed_audio() -> Recording {
    Authored::new()
        .sink(Sink::Recorder(RecorderCall::Flush))
        .sink(Sink::Recorder(RecorderCall::Mismatch { at_ms: 120, expected: 0, actual: 1000 }))
        .done()
}

/// A recorded reply the speaker plays in reported stretches, then a quiet stretch the recording
/// writes through.
fn a_recorded_reply() -> Case {
    Case {
        setup: Setup::default(),
        inputs: vec![
            Input::Recorder { attach: true },
            Input::Say {
                text: AgentText("Hello there, how can I help you today?".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(100) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(700) },
            Input::Advance { ms: Millis(700) },
            Input::PlayoutProgressed { segment: Ref::new(0), ms: Millis(10_000) },
            Input::Advance { ms: Millis(5_000) },
        ],
        ..Case::default()
    }
}
