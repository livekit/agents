use lk_agents_abi::utils::video::VideoFrame;
use lk_agents_abi::{AudioFrame, Ts};

use super::*;

/// `duration` seconds of audio at 1 kHz, told apart by `id`.
fn pcm(id: u8, duration: f64) -> AudioFrame {
    AudioFrame::new(vec![id], 1000, 1, (duration * 1000.0).round() as u32)
}

fn video(id: u8) -> VideoFrame {
    VideoFrame { width: 1, height: 1, buffer_type: 0, data: vec![id].into() }
}

fn audio(id: u8, duration: f64) -> Event {
    Event::Frame(Ok(Some(AvatarFrame::Audio(pcm(id, duration)))))
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
    let first = core.handle(Event::Frame(Ok(Some(AvatarFrame::Video(video(1))))), Ts::ZERO);
    assert!(ops(&first).is_empty(), "held until the room connects: {first:?}");
    let connected = core.handle(Event::Connection { connected: true }, Ts::ZERO);
    assert_eq!(ops(&connected), vec![IoOp::Publish]);
    let published = core.handle(Event::Published(Ok(())), Ts::ZERO);
    assert_eq!(ops(&published), vec![IoOp::Push(MediaFrame::Video(video(1)))]);
}

// Catches: the started report on every frame, or a finished report for a segment never played.
#[test]
fn a_segment_reports_started_once_and_finished_with_what_played() {
    let mut core = running();
    core.handle(
        Event::Audio(Ok(Some(AvatarFrame::Audio(pcm(1, 0.1))))),
        Ts::ZERO,
    );
    assert_eq!(ops(&core.handle(audio(2, 0.04), Ts::ZERO)), vec![IoOp::Push(MediaFrame::Audio(pcm(2, 0.04)))]);
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
        Event::Audio(Ok(Some(AvatarFrame::Audio(pcm(1, 0.1))))),
        Ts::ZERO,
    );
    core.handle(audio(2, 0.04), Ts::ZERO);
    core.handle(Event::Pushed(Ok(())), Ts::ZERO);
    core.handle(Event::AudioPushed(Ok(())), Ts::ZERO);
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

// Catches: the cut segment's end, read after a clear, reaching the generator once the next
// segment's audio set the runner playing: the next segment reported finished before it played
// (survey seed 7462; the reference has the race).
#[test]
fn a_cut_segments_end_after_a_clear_is_not_the_next_segments() {
    let mut core = running();
    core.handle(Event::Audio(Ok(Some(AvatarFrame::Audio(pcm(1, 0.1))))), Ts::ZERO);
    core.handle(Event::ClearBuffer, Ts::ZERO);
    // the receiver hands over the cut stream's end: it goes no further
    let stale = core.handle(Event::Audio(Ok(Some(AvatarFrame::SegmentEnd))), Ts::ZERO);
    assert_eq!(ops(&stale), vec![IoOp::NextAudio]);
    // the next segment's audio and its own end go through
    let next = core.handle(Event::Audio(Ok(Some(AvatarFrame::Audio(pcm(2, 0.1))))), Ts::ZERO);
    assert_eq!(ops(&next), vec![IoOp::PushAudio(Some(MediaFrame::Audio(pcm(2, 0.1))))]);
    let end = core.handle(Event::Audio(Ok(Some(AvatarFrame::SegmentEnd))), Ts::ZERO);
    assert_eq!(ops(&end), vec![IoOp::PushAudio(None)]);
}

// Catches: one playing flag for a generator holding two segments: the first segment's end, out
// after the second segment's audio was read, took the flag and the second was never reported
// (survey seed 32394; the reference has the race).
#[test]
fn back_to_back_segments_are_each_reported_finished() {
    let mut core = running();
    let read = |core: &mut RunnerCore, item| core.handle(Event::Audio(Ok(Some(item))), Ts::ZERO);
    read(&mut core, AvatarFrame::Audio(pcm(1, 0.1)));
    read(&mut core, AvatarFrame::SegmentEnd);
    read(&mut core, AvatarFrame::Audio(pcm(2, 0.1)));
    read(&mut core, AvatarFrame::SegmentEnd);
    for _ in 0..2 {
        let end = core.handle(Event::Frame(Ok(Some(AvatarFrame::SegmentEnd))), Ts::ZERO);
        assert!(
            ops(&end).iter().any(|op| matches!(op, IoOp::ReportFinished { interrupted: false, .. })),
            "{end:?}"
        );
    }
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

// Catches: the generator cleared while a frame of the cut audio was still on its way into it: the
// push landed after the clear and the avatar spoke the cut audio on (survey seed 1353: two frames
// 60 ms after the clear, against one frame period).
#[test]
fn a_clear_waits_for_the_audio_on_its_way_into_the_generator() {
    let mut core = running();
    let push = core.handle(Event::Audio(Ok(Some(AvatarFrame::Audio(pcm(1, 0.1))))), Ts::ZERO);
    assert_eq!(ops(&push), vec![IoOp::PushAudio(Some(MediaFrame::Audio(pcm(1, 0.1))))]);
    let clear = core.handle(Event::ClearBuffer, Ts::ZERO);
    assert!(ops(&clear).is_empty(), "cleared with the cut audio still on its way: {clear:?}");
    let pushed = core.handle(Event::AudioPushed(Ok(())), Ts::ZERO);
    assert_eq!(ops(&pushed), vec![IoOp::ClearGenerator { clear: 0 }, IoOp::NextAudio]);
}
