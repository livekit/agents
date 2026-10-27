use std::sync::Arc;

use lk_agents_abi::{MediaRef, RefRegistry, Ts};

use super::*;

struct NoRefs;

impl RefRegistry for NoRefs {
    fn release(&self, _id: u64) {}
}

fn media(id: u64) -> MediaRef {
    let registry: Arc<dyn RefRegistry> = Arc::new(NoRefs);
    MediaRef::new(id, &registry)
}

fn audio(id: u64, duration: f64) -> Event {
    Event::Frame(Ok(Some(AvatarFrame::Audio { frame: media(id), duration })))
}

fn ops(effects: &[Effect]) -> Vec<IoOp> {
    effects
        .iter()
        .filter_map(|e| if let Effect::Io(op) = e { Some(op.clone()) } else { None })
        .collect()
}

/// A started runner whose tracks are up and whose two loops have opened.
fn running() -> RunnerCore {
    let mut core = RunnerCore::new(false);
    let start = core.handle(Event::Start { call: 1, connected: true }, Ts::ZERO);
    assert_eq!(ops(&start), vec![IoOp::Publish]);
    let begun = core.handle(Event::Published(Ok(())), Ts::ZERO);
    assert_eq!(ops(&begun), vec![IoOp::OpenAudio, IoOp::OpenFrames]);
    assert!(begun.contains(&Effect::Answer { call: 1, result: Ok(()) }));
    core.handle(Event::AudioOpened(Ok(())), Ts::ZERO);
    core.handle(Event::FramesOpened(Ok(())), Ts::ZERO);
    core
}

// Catches: frames pushed before the tracks exist, or a publish asked of a room not connected.
#[test]
fn a_lazy_runner_publishes_on_the_first_frame_once_the_room_connects() {
    let mut core = RunnerCore::new(true);
    let start = core.handle(Event::Start { call: 1, connected: false }, Ts::ZERO);
    assert_eq!(ops(&start), vec![IoOp::OpenAudio, IoOp::OpenFrames]);
    core.handle(Event::FramesOpened(Ok(())), Ts::ZERO);
    let first = core.handle(Event::Frame(Ok(Some(AvatarFrame::Video(media(1))))), Ts::ZERO);
    assert!(ops(&first).is_empty(), "held until the room connects: {first:?}");
    let connected = core.handle(Event::Connection { connected: true }, Ts::ZERO);
    assert_eq!(ops(&connected), vec![IoOp::Publish]);
    let published = core.handle(Event::Published(Ok(())), Ts::ZERO);
    assert_eq!(ops(&published), vec![IoOp::Push(media(1))]);
}

// Catches: the started report on every frame, or a finished report for a segment never played.
#[test]
fn a_segment_reports_started_once_and_finished_with_what_played() {
    let mut core = running();
    core.handle(
        Event::Audio(Ok(Some(AvatarFrame::Audio { frame: media(1), duration: 0.1 }))),
        Ts::ZERO,
    );
    assert_eq!(ops(&core.handle(audio(2, 0.04), Ts::ZERO)), vec![IoOp::Push(media(2))]);
    assert_eq!(
        ops(&core.handle(Event::Pushed(Ok(())), Ts::ZERO)),
        vec![IoOp::ReportStarted, IoOp::NextFrame]
    );
    core.handle(audio(3, 0.04), Ts::ZERO);
    assert_eq!(ops(&core.handle(Event::Pushed(Ok(())), Ts::ZERO)), vec![IoOp::NextFrame]);
    let end = core.handle(Event::Frame(Ok(Some(AvatarFrame::SegmentEnd))), Ts::ZERO);
    assert_eq!(
        ops(&end),
        vec![IoOp::ReportFinished { position: 0.08, interrupted: false }, IoOp::NextFrame]
    );
    let again = core.handle(Event::Frame(Ok(Some(AvatarFrame::SegmentEnd))), Ts::ZERO);
    assert_eq!(ops(&again), vec![IoOp::NextFrame], "nothing was playing");
}

// Catches: the interrupted report before the generator cleared, or with the position at the clear
// rather than what played up to the generator's clear.
#[test]
fn a_clear_reports_the_cut_segment_after_the_generator_cleared() {
    let mut core = running();
    core.handle(
        Event::Audio(Ok(Some(AvatarFrame::Audio { frame: media(1), duration: 0.1 }))),
        Ts::ZERO,
    );
    core.handle(audio(2, 0.04), Ts::ZERO);
    core.handle(Event::Pushed(Ok(())), Ts::ZERO);
    let clear = core.handle(Event::ClearBuffer, Ts::ZERO);
    assert_eq!(ops(&clear), vec![IoOp::ClearGenerator { clear: 0 }]);
    core.handle(audio(3, 0.04), Ts::ZERO);
    core.handle(Event::Pushed(Ok(())), Ts::ZERO);
    let cleared = core.handle(Event::GeneratorCleared { clear: 0, result: Ok(()) }, Ts::ZERO);
    assert_eq!(ops(&cleared), vec![IoOp::ReportFinished { position: 0.08, interrupted: true }]);
    // the segment's end, once cleared, is not reported again
    let end = core.handle(Event::Frame(Ok(Some(AvatarFrame::SegmentEnd))), Ts::ZERO);
    assert_eq!(ops(&end), vec![IoOp::NextFrame]);
    // a clear with nothing playing reports nothing
    core.handle(Event::ClearBuffer, Ts::ZERO);
    assert!(
        ops(&core.handle(Event::GeneratorCleared { clear: 1, result: Ok(()) }, Ts::ZERO))
            .is_empty()
    );
}

// Catches: `wait_for_complete` answered before both loops end, or not at the first failure.
#[test]
fn wait_for_complete_is_gather() {
    let mut core = RunnerCore::new(true);
    let early = core.handle(Event::WaitForComplete { call: 9 }, Ts::ZERO);
    assert_eq!(early, vec![Effect::Answer { call: 9, result: Err(RunnerError::NotStarted) }]);
    let mut core = running();
    assert!(core.handle(Event::WaitForComplete { call: 2 }, Ts::ZERO).is_empty());
    assert!(
        core.handle(Event::Audio(Ok(None)), Ts::ZERO)
            .iter()
            .all(|e| !matches!(e, Effect::Answer { .. }))
    );
    let done = core.handle(Event::Frame(Ok(None)), Ts::ZERO);
    assert!(done.contains(&Effect::Answer { call: 2, result: Ok(()) }));

    let mut core = running();
    core.handle(Event::WaitForComplete { call: 3 }, Ts::ZERO);
    let failed = ProviderError::Local(lk_agents_abi::errors::ApiError::new("boom"));
    let out = core.handle(Event::Frame(Err(failed.clone())), Ts::ZERO);
    assert!(out.contains(&Effect::Answer { call: 3, result: Err(RunnerError::Failed(failed)) }));
}

// Catches: the media closed before the loops were cut, a frame held across the close, or an op
// issued once cancelling began.
#[test]
fn close_goes_receiver_then_loops_then_media() {
    let mut core = running();
    core.handle(Event::WaitForComplete { call: 5 }, Ts::ZERO);
    assert_eq!(ops(&core.handle(Event::Close { call: 7 }, Ts::ZERO)), vec![IoOp::CloseReceiver]);
    assert_eq!(core.handle(Event::ReceiverClosed(Ok(())), Ts::ZERO), vec![Effect::CancelIo]);
    assert!(core.handle(audio(4, 0.04), Ts::ZERO).is_empty(), "no op once cancelling");
    let cancelled = core.handle(Event::IoCancelled, Ts::ZERO);
    assert!(cancelled.contains(&Effect::Answer { call: 5, result: Err(RunnerError::Closed) }));
    assert_eq!(ops(&cancelled), vec![IoOp::CloseMedia]);
    assert_eq!(
        core.handle(Event::MediaClosed(Ok(())), Ts::ZERO),
        vec![Effect::Answer { call: 7, result: Ok(()) }]
    );
    assert!(core.held.is_none());
}
