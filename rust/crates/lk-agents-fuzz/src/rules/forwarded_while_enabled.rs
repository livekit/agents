//! Microphone audio the case pushed while input was enabled reached every recognition and
//! voice-activity stream that was open; nothing the case pushed while input was disabled reached
//! any of them.
//!
//! Frames move through the audio pump on the driver's next turn, so each is judged after that
//! turn, and one that raced a disable or a gate change inside the same turn is left alone. The
//! STT leg hears silence while the gate is closed, which the journal publishes; VAD always hears
//! the room.
use std::collections::BTreeMap;

use lk_agents_voice::agent_session::IoKind;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{Count, Input, Ref},
    record::{Detail, InferenceView, InputsView, Kind, Observed, Recording, Tag, Window},
};

pub struct ForwardedWhileEnabled;

impl Rule for ForwardedWhileEnabled {
    const CONTRACT: Contract = Contract {
        id: "forwarded_while_enabled",
        guarantee: "Every microphone frame pushed while audio input was enabled reached every open STT and VAD stream; none reached a stream while input was disabled.",
        examples: &[reaches_both_legs, late_delivery_after_disable],
        counterexamples: &[stt_never_heard, forwarded_while_disabled],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let inference = r.inference_view();
        let audio_input = inputs.audio_input();
        let gate_closed = r.internal_view().stt_gate_closed();

        // Every frame pushed while input was enabled, numbered per microphone as the driver numbers
        // them, and when it was first pushed: what a late delivery may be.
        let mut next_chunk: BTreeMap<u32, u16> = BTreeMap::new();
        let mut pushed_enabled: BTreeMap<(u32, u16), Observed> = BTreeMap::new();
        for sent in inputs.accepted() {
            let Input::MicFrames { mic, chunks: Count(chunks) } = &sent.input else { continue };
            let counter = next_chunk.entry(mic.nth).or_default();
            for _ in 0..*chunks {
                let chunk = *counter % (Tag::MAX_CHUNK + 1);
                *counter = counter.wrapping_add(1);
                // Nothing pushed while input was disabled or the microphone detached is owed to
                // any stream.
                if !audio_input.on_at(sent.at) {
                    continue;
                }
                pushed_enabled.entry((mic.nth, chunk)).or_insert(sent.at);
                expect_reached(&inputs, &inference, &gate_closed, check, (mic.nth, chunk), sent.at);
            }
        }

        // A frame heard while input was disabled can only be one pushed earlier, while it was
        // enabled, still moving through the pump. Anything else got past a closed microphone.
        for heard in inference.frames_heard() {
            if audio_input.on_at(heard.at) {
                continue;
            }
            for tag in heard.tags {
                let Tag::Mic { mic, chunk } = *tag else { continue };
                let late =
                    pushed_enabled.get(&(mic, chunk)).is_some_and(|pushed| *pushed < heard.at);
                check.expect_with(Subject::Resource(Kind::Mic, mic), late, || {
                    format!(
                        "{:?} stream {} heard mic {mic} chunk {chunk} while audio input was disabled (seq {})",
                        heard.kind, heard.stream, heard.at.seq
                    )
                });
            }
        }
    }
}

/// The streams that listen to the microphone.
const LISTENERS: [Kind; 2] = [Kind::Stt, Kind::Vad];

/// One frame pushed at `pushed` while input was enabled reached every listening stream that was
/// open for its whole journey.
fn expect_reached(
    inputs: &InputsView,
    inference: &InferenceView,
    gate_closed: &[Window],
    check: &mut Check,
    (mic, chunk): (u32, u16),
    pushed: Observed,
) {
    // Frames move through the pump on the driver's next turn. Until it has had one, nothing has
    // moved the frame and nothing can be said about where it went.
    let Some(turn) = inputs.driver_turn_after(pushed) else { return };
    let end = turn.ended.unwrap_or(Observed::END);
    // A disable landing in the same turn races the pump for the frame; either outcome is right.
    if inputs.audio_input().off_within(pushed, end) {
        return;
    }
    for kind in LISTENERS {
        // The gate feeds STT silence instead of the room while it is closed; a frame whose turn
        // touched a closed window may rightly have been replaced.
        if kind == Kind::Stt && gate_closed.iter().any(|window| window.overlaps(pushed, end)) {
            continue;
        }
        for stream in inference.streams(kind) {
            if !stream.open_throughout(pushed, end) {
                continue;
            }
            check.expect_with(
                Subject::Resource(Kind::Mic, mic),
                stream.heard_after(mic, chunk, pushed),
                || {
                    format!(
                        "mic {mic} chunk {chunk} pushed at seq {} never reached {kind:?} stream {}",
                        pushed.seq,
                        stream.nth()
                    )
                },
            );
        }
    }
}

/// The two provider legs the pump feeds from the microphone; the Rust-native detectors are not
/// fakes and record nothing.
/// The recording, indexed once per check. This rule runs after every input over the whole
/// recording, and a conversation pushes ten frames a beat to three streams: read by linear scans
/// per chunk it was cubic in the run's length and most of a survey's wall time; read through
/// these three tables it is linear.
/// One frame pushed while input was enabled at `pushed` reached every stream open for its
/// whole journey.
/// The windows in which the gate fed the STT leg silence, read off its publishes.
fn frames(chunks: u32) -> Input {
    Input::MicFrames { mic: Ref::new(0), chunks: Count(chunks) }
}

fn audio_input(enabled: bool) -> Input {
    Input::SetIoEnabled { leg: IoKind::AudioInput, enabled }
}

fn heard(chunk: u16) -> Vec<Tag> {
    vec![Tag::Mic { mic: 0, chunk }]
}

/// A microphone, one recognition stream and one VAD stream, as a run with both legs opens them.
fn mic_stt_vad() -> Authored {
    Authored::new()
        .opens(Kind::Mic, Detail::None)
        .opens(Kind::Stt, Detail::None)
        .opens(Kind::Vad, Detail::None)
}

/// Two frames pushed, the driver ran, and both streams heard both.
fn reaches_both_legs() -> Recording {
    mic_stt_vad()
        .sent(frames(2))
        .then_act()
        .forwarded(Kind::Stt, 0, heard(0))
        .forwarded(Kind::Stt, 0, heard(1))
        .forwarded(Kind::Vad, 0, heard(0))
        .forwarded(Kind::Vad, 0, heard(1))
        .done()
}

/// A frame pushed while enabled, input disabled before the driver ran, and the frame heard in
/// the disabled window: a late delivery, not a leak.
fn late_delivery_after_disable() -> Recording {
    mic_stt_vad()
        .sent(frames(1))
        .sent(audio_input(false))
        .then_act()
        .forwarded(Kind::Stt, 0, heard(0))
        .forwarded(Kind::Vad, 0, heard(0))
        .done()
}

/// The VAD heard the frame and the recognition stream never did.
fn stt_never_heard() -> Recording {
    mic_stt_vad().sent(frames(1)).then_act().forwarded(Kind::Vad, 0, heard(0)).done()
}

/// Input disabled, a frame pushed, and the recognition stream heard it anyway.
fn forwarded_while_disabled() -> Recording {
    mic_stt_vad()
        .sent(audio_input(false))
        .sent(frames(1))
        .then_act()
        .forwarded(Kind::Stt, 0, heard(0))
        .done()
}
