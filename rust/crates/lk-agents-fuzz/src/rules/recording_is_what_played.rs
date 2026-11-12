use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, Setup},
    input::{AgentText, Input, Millis, Ref, SayInput},
    record::{RecorderCall, Recording, Sink},
};

// The recorder fake writes the session's recording with the real timeline and checks each
// flush's newly written agent channel against what the sink reported played.

pub struct RecordingIsWhatPlayed;

impl Rule for RecordingIsWhatPlayed {
    const CONTRACT: Contract = Contract {
        id: "recording_is_what_played",
        guarantee: "The session recording holds the agent's audio wherever the speaker reported it \
                    played, and no played audio reaches the recording after its place was written.",
        examples: &[checked],
        counterexamples: &[a_hole, dropped_late],
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
