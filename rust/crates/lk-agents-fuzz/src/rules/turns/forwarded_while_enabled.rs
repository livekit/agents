use std::collections::BTreeMap;

use lk_agents_voice::agent_session::IoKind;
use lk_agents_voice::agent_session::effects::DetectorCall;
use lk_agents_voice::trace::{CoreStep, InputStep, RoomEventKind, RoomStep, Step, TurnStep};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{Count, Input, Ref},
    record::{Detail, InferenceView, InputsView, Kind, Observed, Recording, Switch, Tag, Window},
};

// Frames move through the pump on the driver's next turn while an enable or disable rides the
// command queue to that same turn, so a frame that raced a switch or a gate change is left alone.
// The STT leg hears silence while the gate is closed; VAD always hears the room. Under push-to-talk
// no stream is owed a frame outside a user turn, and a turn's release or cancel swaps the
// recogniser's stream.

pub struct ForwardedWhileEnabled;

impl Rule for ForwardedWhileEnabled {
    const CONTRACT: Contract = Contract {
        id: "forwarded_while_enabled",
        guarantee: "Every microphone frame pushed while audio input was enabled reached every open STT and VAD stream; none reached a stream while input was disabled.",
        examples: &[reaches_both_legs, late_delivery_after_disable, crossed_a_recogniser_swap],
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
        // Every push of a frame, enabled or not, in order: a frame heard while disabled may be one
        // pushed in the turn the disable was still on its way.
        let mut pushed_at: BTreeMap<(u32, u16), Vec<Observed>> = BTreeMap::new();
        // When each microphone was replaced or detached: its frames go with it.
        let mic_closed: BTreeMap<u32, Observed> = r
            .closed()
            .iter()
            .filter(|(_, kind, _)| *kind == Kind::Mic)
            .map(|(at, _, nth)| (*nth, *at))
            .collect();
        // When the recogniser or the detector was swapped for a new stream.
        let reopened: Vec<(Observed, Kind)> = r
            .journal_where(|step| match step {
                Step::Input(InputStep::ModelsUpdated { stt_reopened, vad_reopened, .. }) => {
                    Some((*stt_reopened, *vad_reopened))
                }
                Step::Turn(
                    TurnStep::UserTurnStarted { .. }
                    | TurnStep::UserTurnReleased { .. }
                    | TurnStep::UserTurnCancelled { .. },
                ) => Some((true, false)),
                _ => None,
            })
            .flat_map(|(at, (stt, vad))| {
                [(stt, Kind::Stt), (vad, Kind::Vad)]
                    .into_iter()
                    .filter(|(on, _)| *on)
                    .map(move |(_, k)| (at, k))
            })
            .collect();
        // The close's edge detaches the microphone: it asks for the frame census and closes the
        // turn detector, and the legs' close stops the microphone's reader. What had not reached
        // a stream by then never will, whichever microphone it is on. The census is journalled
        // only once the data plane answers, and not at all for a close that never fed a channel
        // a frame; the detector's close is journalled at the edge itself.
        let detached = r
            .journal_where(|step| {
                matches!(
                    step,
                    Step::Input(
                        InputStep::LegFrames { .. }
                            | InputStep::InputShape { .. }
                            | InputStep::DetectorCalled(DetectorCall::Close)
                    )
                )
                .then_some(())
            })
            .next()
            .map(|(at, ())| at);
        // The person's track unpublished (a relink under a duplicate identity republishes it) or
        // unsubscribed: what its stream still carried goes with it (seed 68376).
        let unpublished: Vec<Observed> = r
            .journal_where(|step| {
                matches!(
                    step,
                    Step::Room(RoomStep::Event(
                        RoomEventKind::TrackUnpublished | RoomEventKind::TrackUnsubscribed
                    ))
                )
                .then_some(())
            })
            .map(|(at, ())| at)
            .collect();
        let unheard = push_to_talk_idle(r);
        for sent in inputs.accepted() {
            let Input::MicFrames { mic, chunks: Count(chunks) } = &sent.input else { continue };
            // Ignored outside a push-to-talk turn, or racing a turn's edge in its driver turn; its
            // chunks are still counted.
            let ends =
                inputs.rested_turn_after(sent.at).map(|turn| turn.ended.unwrap_or(Observed::END));
            let ignored = idle_within(&unheard, sent.at, ends.unwrap_or(sent.at));
            let counter = next_chunk.entry(mic.nth).or_default();
            for _ in 0..*chunks {
                let chunk = *counter % (Tag::MAX_CHUNK + 1);
                *counter = counter.wrapping_add(1);
                pushed_at.entry((mic.nth, chunk)).or_default().push(sent.at);
                // Nothing pushed while input was disabled or the microphone detached is owed to
                // any stream.
                if !audio_input.on_at(sent.at) {
                    continue;
                }
                pushed_enabled.entry((mic.nth, chunk)).or_insert(sent.at);
                // Kept as a push-to-talk pre-roll at most: heard later or never.
                if ignored {
                    continue;
                }
                let dropped = unpublished.iter().find(|at| **at > sent.at).copied();
                let replaced = mic_closed
                    .get(&mic.nth)
                    .copied()
                    .into_iter()
                    .chain(detached)
                    .chain(dropped)
                    .min();
                let frame = (mic.nth, chunk);
                expect_reached(
                    &inputs,
                    &audio_input,
                    &inference,
                    &gate_closed,
                    &reopened,
                    replaced,
                    check,
                    frame,
                    sent.at,
                );
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
                // A tag no push made by then is a sample the leg's own processing made (a
                // resampler ringing past the last chunk of a microphone's tag range), not audio
                // that got past a closed input.
                let pushed = pushed_at.get(&(mic, chunk));
                if pushed.is_none_or(|pushes| pushes.iter().all(|pushed| *pushed >= heard.at)) {
                    continue;
                }
                let late =
                    pushed_enabled.get(&(mic, chunk)).is_some_and(|pushed| *pushed < heard.at);
                let raced = pushed_at.get(&(mic, chunk)).is_some_and(|pushes| {
                    pushes.iter().rfind(|pushed| **pushed < heard.at).is_some_and(|pushed| {
                        switching(&inputs, &audio_input, *pushed)
                            || switched_in_its_turn(&inputs, &audio_input, *pushed)
                    })
                });
                check.expect_with(Subject::Resource(Kind::Mic, mic), late || raced, || {
                    format!(
                        "{:?} stream {} heard mic {mic} chunk {chunk} while audio input was disabled (seq {})",
                        heard.kind, heard.stream, heard.at.seq
                    )
                });
            }
        }
    }
}

/// Until when a frame pushed at `pushed` was on its way to the streams: until one heard it, or,
/// never heard, until its microphone forwarded anything after it. That can be past the pump's
/// turn: a room microphone holds its frames while it waits for the pre-connect buffer.
fn on_its_way(inference: &InferenceView, (mic, chunk): (u32, u16), pushed: Observed) -> Observed {
    let streams = || LISTENERS.iter().flat_map(|kind| inference.streams(*kind));
    let after = |at: Option<Observed>| at.filter(|at| *at > pushed);
    let arrived = streams().filter_map(|s| after(s.heard_at(mic, chunk))).min();
    let forwarding = || {
        streams()
            .flat_map(|s| (0..=Tag::MAX_CHUNK).filter_map(move |c| after(s.heard_at(mic, c))))
            .min()
    };
    arrived.or_else(forwarding).unwrap_or(Observed::END)
}

/// Whether the path's last switch before a frame was pushed had not yet reached the driver: the
/// switch waits on the command queue while the frame goes straight to the pump, so the two land
/// in the same turn in either order.
fn switching(inputs: &InputsView, audio_input: &Switch, pushed: Observed) -> bool {
    audio_input
        .last_change(pushed)
        .is_some_and(|(at, _)| inputs.driver_turn_after(at).is_none_or(|turn| turn.began > pushed))
}

/// Whether the path was switched in the driver turn that moved a frame pushed at `pushed`: a
/// command sent before the frame lands in that turn, so the frame races it whichever way the path
/// went (an attach and a disable sent together, say).
fn switched_in_its_turn(inputs: &InputsView, audio_input: &Switch, pushed: Observed) -> bool {
    inputs
        .driver_turn_after(pushed)
        .is_some_and(|turn| audio_input.changed_within(pushed, turn.ended.unwrap_or(Observed::END)))
}

/// When push-to-talk went idle (`true`) or listened again (`false`), in order: idle while it is on
/// and no user turn is open.
fn push_to_talk_idle(r: &Recording) -> Vec<(Observed, bool)> {
    let (mut on, mut open, mut idle) = (false, false, false);
    let mut edges = Vec::new();
    for (at, step) in r.journal() {
        match step {
            Step::Core(CoreStep::ConfigApplied { effective, .. }) => {
                on = effective.turn.push_to_talk.unwrap_or(false);
            }
            Step::Turn(TurnStep::UserTurnStarted { .. }) => open = true,
            Step::Turn(TurnStep::UserTurnReleased { .. } | TurnStep::UserTurnCancelled { .. }) => {
                open = false;
            }
            _ => continue,
        }
        if (on && !open) != idle {
            idle = !idle;
            edges.push((*at, idle));
        }
    }
    edges
}

/// Whether push-to-talk was idle at `from` or went idle by `to`.
fn idle_within(edges: &[(Observed, bool)], from: Observed, to: Observed) -> bool {
    let idle_at_from = edges.iter().rfind(|(at, _)| *at <= from).is_some_and(|(_, idle)| *idle);
    idle_at_from || edges.iter().any(|(at, idle)| *idle && *at > from && *at <= to)
}

/// The streams that listen to the microphone.
const LISTENERS: [Kind; 2] = [Kind::Stt, Kind::Vad];

/// One frame pushed at `pushed` while input was enabled reached every listening stream that was
/// open for its whole journey.
#[allow(clippy::too_many_arguments, reason = "Keep the operation’s existing explicit arguments")]
fn expect_reached(
    inputs: &InputsView,
    audio_input: &Switch,
    inference: &InferenceView,
    gate_closed: &[Window],
    reopened: &[(Observed, Kind)],
    replaced: Option<Observed>,
    check: &mut Check,
    (mic, chunk): (u32, u16),
    pushed: Observed,
) {
    // Frames move through the pump on the driver's next turn that runs the session to rest (an
    // await answered first ends its turn before the pump ran, seed 11116). Until it has had one,
    // nothing has moved the frame and nothing can be said about where it went.
    let Some(turn) = inputs.rested_turn_after(pushed) else { return };
    let end = turn.ended.unwrap_or(Observed::END);
    let way = std::cell::OnceCell::new();
    let on_its_way = || *way.get_or_init(|| on_its_way(inference, (mic, chunk), pushed).max(end));
    // A switch landing in the same turn races the pump for the frame, and so does replacing the
    // microphone it was pushed to while the frame was on its way (the reference cancels the old
    // stream's reader with whatever it had not read); either outcome is right.
    if audio_input.off_within(pushed, end)
        || switching(inputs, audio_input, pushed)
        || replaced.is_some_and(|at| at <= on_its_way())
    {
        return;
    }
    // The pump is its own task: a frame still queued when its turn ends is moved in the next one,
    // and a gate that closed meanwhile (published by that turn) gets it.
    let moved_by = inputs.driver_turn_after(end).and_then(|t| t.ended).unwrap_or(Observed::END);
    for kind in LISTENERS {
        // The gate feeds STT silence instead of the room while it is closed; a frame whose journey
        // touched a closed window may rightly have been replaced.
        if kind == Kind::Stt && gate_closed.iter().any(|window| window.overlaps(pushed, moved_by)) {
            continue;
        }
        // A provider swapped while the frame was on its way, up to the turn that moves it or
        // its arrival, may hear it on either stream (seed 47240).
        let way = on_its_way().max(moved_by);
        if reopened.iter().any(|(at, leg)| *leg == kind && pushed < *at && *at <= way) {
            continue;
        }
        for stream in inference.streams(kind) {
            // AMD's own recogniser hears the microphone only while the detector listens
            // (`AMD.push_audio`), and a plugin's detector drops to current audio past its lag:
            // not this path's to owe. A stream closed before the turn that moves the frame (a
            // recogniser failing) is owed nothing (seed 149497).
            let own_path = matches!(stream.detail(), Detail::AmdStt | Detail::PluginVad);
            if !stream.listening_throughout(pushed, moved_by) || own_path {
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

/// A frame pushed, the recogniser swapped before the driver's turn ended, and the old stream
/// never hearing it: a room's input pump hands the frame over a hop later, to the new stream.
/// Catches: the rule owing it to the old stream (survey seed 1862 on the room's legs).
fn crossed_a_recogniser_swap() -> Recording {
    mic_stt_vad()
        .sent(frames(1))
        .journal(Step::Input(InputStep::ModelsUpdated {
            stt_given: true,
            vad_given: false,
            stt_reopened: true,
            vad_reopened: false,
        }))
        .then_act()
        .forwarded(Kind::Vad, 0, heard(0))
        .done()
}

/// The VAD heard the frame and the recognition stream never did.
fn stt_never_heard() -> Recording {
    mic_stt_vad().sent(frames(1)).then_act().forwarded(Kind::Vad, 0, heard(0)).done()
}

/// Input disabled and applied, a frame pushed after, and the recognition stream heard it anyway.
fn forwarded_while_disabled() -> Recording {
    mic_stt_vad()
        .sent(audio_input(false))
        .then_act()
        .sent(frames(1))
        .then_act()
        .forwarded(Kind::Stt, 0, heard(0))
        .done()
}
