//! Every SDK object the room hands the session is bound to the host's event loop: created on it,
//! and taken in the context of the call that produced it.
//!
//! A real SDK media stream binds the loop it is created on, and a stream writer runs its calls in
//! the context it was taken in. The core reaches the host on its own thread, where no loop runs:
//! a declared `call` runs there, and so does the read of an awaited call's reply. An object bound
//! there fails in a real room ("no running event loop"), or reads and writes on no loop at all. The
//! room fake binds each object where the adapters bind it (`fakes::room::HostLoop`) and records
//! where; a binding on the core's thread is the finding.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    drive::{Case, RoomLegs, RoomSetup, Setup},
    fakes::room::Thread,
    input::{AgentText, Count, Input, Millis, Ref, RoomIn, SayInput},
    record::Recording,
};

pub struct HostObjectsBindOnTheirLoop;

impl Rule for HostObjectsBindOnTheirLoop {
    const CONTRACT: Contract = Contract {
        id: "host_objects_bind_on_their_loop",
        guarantee: "Every SDK object the room hands the session (a media stream, a stream \
                    writer) is bound to the host's event loop, never on the core's thread.",
        examples: &[bound_on_the_loop],
        counterexamples: &[writer_taken_off_the_loop, stream_opened_on_the_core_thread],
        probe: Some(a_microphone_and_an_avatar),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        for (at, object, thread) in r.host_bindings() {
            check.expect_with(Subject::Session, *thread == Thread::Loop, || {
                format!("{object} bound on the core's thread at seq {}", at.seq)
            });
        }
    }
}

fn bound_on_the_loop() -> Recording {
    Authored::new()
        .host_binding("AudioStream", Thread::Loop)
        .host_binding("ByteStreamWriter", Thread::Loop)
        .done()
}

/// The writer an awaited `stream_bytes` returned, taken as its reply was read on the core's
/// thread (`owned` instead of `owned_in_context`).
fn writer_taken_off_the_loop() -> Recording {
    Authored::new().host_binding("ByteStreamWriter", Thread::Core).done()
}

/// The SDK stream created inside the declared `call` that opens it (`_core_open_audio_stream`).
fn stream_opened_on_the_core_thread() -> Recording {
    Authored::new().host_binding("AudioStream", Thread::Core).done()
}

/// Prerequisite: the person's microphone read by the session's input, and a speech played by an
/// avatar, a byte stream per segment.
fn a_microphone_and_an_avatar() -> Case {
    Case {
        setup: Setup {
            room: Some(RoomSetup { close_on_disconnect: false, delete_room_on_close: false }),
            room_legs: Some(RoomLegs { pre_connect: None }),
            avatar: true,
            ..Setup::default()
        },
        inputs: vec![
            Input::Room { event: RoomIn::Linked },
            Input::Advance { ms: Millis(20) },
            Input::MicFrames { mic: Ref::new(0), chunks: Count(20) },
            Input::Say {
                text: AgentText("hello".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: true,
            },
            Input::Advance { ms: Millis(1000) },
        ],
        ..Case::default()
    }
}
