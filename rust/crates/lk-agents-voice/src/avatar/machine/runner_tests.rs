use lk_agents_abi::AudioFrame;
use lk_agents_abi::utils::video::VideoFrame;

use super::*;
use lk_agents_abi::voice::io::PlaybackSegment;

fn token(id: u64) -> AvatarPlaybackSegment {
    AvatarPlaybackSegment {
        playback: PlaybackSegment { attachment: 1, segment: id },
        stream_id: Some(format!("S{id}")),
    }
}

/// `duration` seconds of audio at 1 kHz, told apart by `id`.
fn pcm(id: u8, duration: f64) -> AudioFrame {
    AudioFrame::new(vec![id], 1000, 1, (duration * 1000.0).round() as u32)
}

fn video(id: u8) -> VideoFrame {
    let buffer_type = lk_agents_abi::utils::video::VideoBufferType::Rgba;
    VideoFrame { width: 1, height: 1, buffer_type, data: vec![id].into() }
}

fn audio(id: u8, duration: f64) -> AvatarEvent {
    AvatarEvent::Frame(Ok(Some(AvatarFrame::Audio(pcm(id, duration)))))
}

fn ops(effects: &[AvatarEffect]) -> Vec<AvatarIoOperation> {
    effects
        .iter()
        .filter_map(|e| if let AvatarEffect::Io(op) = e { Some(op.clone()) } else { None })
        .collect()
}

/// A started runner whose tracks are up and whose two loops have opened.
fn running() -> RunnerCore {
    let mut core = RunnerCore::new(false);
    let start = core.handle(AvatarEvent::Start { call: 1, connected: true });
    assert_eq!(ops(&start), vec![AvatarIoOperation::Publish]);
    let begun = core.handle(AvatarEvent::Published(Ok(())));
    assert_eq!(ops(&begun), vec![AvatarIoOperation::OpenAudio, AvatarIoOperation::OpenFrames]);
    assert!(begun.contains(&AvatarEffect::Answer { call: 1, result: Ok(()) }));
    core.handle(AvatarEvent::AudioOpened(Ok(())));
    core.handle(AvatarEvent::FramesOpened(Ok(())));
    core
}

// Catches: frames pushed before the tracks exist, or a publish asked of a room not connected.
#[test]
fn a_lazy_runner_publishes_on_the_first_frame_once_the_room_connects() {
    let mut core = RunnerCore::new(true);
    let start = core.handle(AvatarEvent::Start { call: 1, connected: false });
    assert_eq!(ops(&start), vec![AvatarIoOperation::OpenAudio, AvatarIoOperation::OpenFrames]);
    core.handle(AvatarEvent::FramesOpened(Ok(())));
    let first = core.handle(AvatarEvent::Frame(Ok(Some(AvatarFrame::Video(video(1))))));
    assert!(ops(&first).is_empty(), "held until the room connects: {first:?}");
    let connected = core.handle(AvatarEvent::Connection { connected: true });
    assert_eq!(ops(&connected), vec![AvatarIoOperation::Publish]);
    let published = core.handle(AvatarEvent::Published(Ok(())));
    assert_eq!(ops(&published), vec![AvatarIoOperation::Push(MediaFrame::Video(video(1)))]);
}

// Catches: the started report on every frame, or a finished report for a segment never played.
#[test]
fn a_segment_reports_started_once_and_finished_with_what_played() {
    let mut core = running();
    core.handle(audio_input(&core, Ok(Some(AudioReceiverFrame::Audio(pcm(1, 0.1))))));
    assert_eq!(
        ops(&core.handle(audio(2, 0.04))),
        vec![AvatarIoOperation::Push(MediaFrame::Audio(pcm(2, 0.04)))]
    );
    assert_eq!(
        ops(&core.handle(AvatarEvent::Pushed(Ok(())))),
        vec![AvatarIoOperation::ReportStarted { segment: token(1) }, AvatarIoOperation::NextFrame]
    );
    core.handle(audio(3, 0.04));
    assert_eq!(ops(&core.handle(AvatarEvent::Pushed(Ok(())))), vec![AvatarIoOperation::NextFrame]);
    let end = core.handle(AvatarEvent::Frame(Ok(Some(AvatarFrame::SegmentEnd))));
    assert_eq!(
        ops(&end),
        vec![AvatarIoOperation::ReportFinished {
            segment: token(1),
            position: 0.08,
            interrupted: false
        }]
    );
    core.handle(AvatarEvent::Reported { finished: true, result: Ok(()) });
    let again = core.handle(AvatarEvent::Frame(Ok(Some(AvatarFrame::SegmentEnd))));
    assert_eq!(ops(&again), vec![AvatarIoOperation::NextFrame], "nothing was playing");
}

// Catches: the interrupted report before the generator cleared, or with the position at the clear
// rather than what played up to the generator's clear.
#[test]
fn a_clear_reports_the_cut_segment_after_the_generator_cleared() {
    let mut core = running();
    core.handle(audio_input(&core, Ok(Some(AudioReceiverFrame::Audio(pcm(1, 0.1))))));
    core.handle(audio(2, 0.04));
    core.handle(AvatarEvent::Pushed(Ok(())));
    core.handle(AvatarEvent::AudioPushed(Ok(())));
    let clear = core.handle(AvatarEvent::ClearBuffer { epoch: core.clear_epoch + 1 });
    assert_eq!(ops(&clear), vec![AvatarIoOperation::ClearGenerator { clear: 1 }]);
    core.handle(audio(3, 0.04));
    core.handle(AvatarEvent::Pushed(Ok(())));
    let cleared = core.handle(AvatarEvent::GeneratorCleared { clear: 1, result: Ok(()) });
    assert_eq!(
        ops(&cleared),
        vec![AvatarIoOperation::ReportFinished {
            segment: token(1),
            position: 0.08,
            interrupted: true
        }]
    );
    core.handle(AvatarEvent::Reported { finished: true, result: Ok(()) });
    // the segment's end, once cleared, is not reported again
    let end = core.handle(AvatarEvent::Frame(Ok(Some(AvatarFrame::SegmentEnd))));
    assert_eq!(ops(&end), vec![AvatarIoOperation::NextFrame]);
    // a clear with nothing playing reports nothing
    core.handle(AvatarEvent::ClearBuffer { epoch: core.clear_epoch + 1 });
    assert!(
        ops(&core.handle(AvatarEvent::GeneratorCleared { clear: 2, result: Ok(()) })).is_empty()
    );
}

// Catches: the cut segment's end, read after a clear, reaching the generator once the next
// segment's audio set the runner playing: the next segment reported finished before it played
// (survey seed 7462; the reference has the race).
#[test]
fn a_cut_segments_end_after_a_clear_is_not_the_next_segments() {
    let mut core = running();
    core.handle(audio_input(&core, Ok(Some(AudioReceiverFrame::Audio(pcm(1, 0.1))))));
    core.handle(AvatarEvent::AudioPushed(Ok(())));
    core.handle(AvatarEvent::ClearBuffer { epoch: core.clear_epoch + 1 });
    // the receiver hands over the cut stream's end: it goes no further
    let stale = core.handle(audio_input(&core, Ok(Some(AudioReceiverFrame::SegmentEnd))));
    assert_eq!(ops(&stale), vec![AvatarIoOperation::NextAudio]);
    // the next segment's audio and its own end go through
    let next = core.handle(audio_input(&core, Ok(Some(AudioReceiverFrame::Audio(pcm(2, 0.1))))));
    assert!(ops(&next).is_empty());
    core.handle(AvatarEvent::GeneratorCleared { clear: 1, result: Ok(()) });
    let delivered = core.handle(AvatarEvent::Reported { finished: true, result: Ok(()) });
    assert_eq!(
        ops(&delivered),
        vec![AvatarIoOperation::PushAudio(AudioReceiverFrame::Audio(pcm(2, 0.1)))]
    );
    let end = core.handle(audio_input(&core, Ok(Some(AudioReceiverFrame::SegmentEnd))));
    assert_eq!(ops(&end), vec![AvatarIoOperation::PushAudio(AudioReceiverFrame::SegmentEnd)]);
}

// Catches: one playing flag for a generator holding two segments: the first segment's end, out
// after the second segment's audio was read, took the flag and the second was never reported
// (survey seed 32394; the reference has the race).
#[test]
fn back_to_back_segments_are_each_reported_finished() {
    let mut core = running();
    let read = |core: &mut RunnerCore, item| core.handle(audio_input(core, Ok(Some(item))));
    read(&mut core, AudioReceiverFrame::Audio(pcm(1, 0.1)));
    read(&mut core, AudioReceiverFrame::SegmentEnd);
    read(&mut core, AudioReceiverFrame::Audio(pcm(2, 0.1)));
    read(&mut core, AudioReceiverFrame::SegmentEnd);
    for _ in 0..2 {
        let end = core.handle(AvatarEvent::Frame(Ok(Some(AvatarFrame::SegmentEnd))));
        assert!(
            ops(&end).iter().any(|op| matches!(
                op,
                AvatarIoOperation::ReportFinished { interrupted: false, .. }
            )),
            "{end:?}"
        );
        core.handle(AvatarEvent::Reported { finished: true, result: Ok(()) });
    }
}

// Catches: `wait_for_complete` answered before both loops end, or not at the first failure.
#[test]
fn wait_for_complete_is_gather() {
    let mut core = RunnerCore::new(true);
    let early = core.handle(AvatarEvent::WaitForComplete { call: 9 });
    assert_eq!(early, vec![AvatarEffect::Answer { call: 9, result: Err(RunnerError::NotStarted) }]);
    let mut core = running();
    assert!(core.handle(AvatarEvent::WaitForComplete { call: 2 }).is_empty());
    assert!(
        core.handle(audio_input(&core, Ok(None)))
            .iter()
            .all(|e| !matches!(e, AvatarEffect::Answer { .. }))
    );
    let done = core.handle(AvatarEvent::Frame(Ok(None)));
    assert!(done.contains(&AvatarEffect::Answer { call: 2, result: Ok(()) }));

    let mut core = running();
    core.handle(AvatarEvent::WaitForComplete { call: 3 });
    let failed = ProviderError::Local(lk_agents_abi::errors::ApiError::new("boom"));
    let out = core.handle(AvatarEvent::Frame(Err(failed.clone())));
    assert!(
        out.contains(&AvatarEffect::Answer { call: 3, result: Err(RunnerError::Failed(failed)) })
    );
}

// Catches: the media closed before the loops were cut, a frame held across the close, or an op
// issued once cancelling began.
#[test]
fn close_goes_receiver_then_loops_then_media() {
    let mut core = running();
    core.handle(AvatarEvent::WaitForComplete { call: 5 });
    assert_eq!(
        ops(&core.handle(AvatarEvent::Close { call: 7 })),
        vec![AvatarIoOperation::CloseReceiver]
    );
    assert_eq!(core.handle(AvatarEvent::ReceiverClosed(Ok(()))), vec![AvatarEffect::CancelIo]);
    assert!(core.handle(audio(4, 0.04)).is_empty(), "no op once cancelling");
    let cancelled = core.handle(AvatarEvent::IoCancelled);
    assert!(
        cancelled.contains(&AvatarEffect::Answer { call: 5, result: Err(RunnerError::Closed) })
    );
    assert_eq!(ops(&cancelled), vec![AvatarIoOperation::CloseMedia]);
    assert_eq!(
        core.handle(AvatarEvent::MediaClosed(Ok(()))),
        vec![AvatarEffect::Answer { call: 7, result: Ok(()) }]
    );
    assert!(core.held.is_none());
}

#[test]
fn a_failed_receiver_close_preserves_the_running_loops() {
    let mut core = running();
    core.handle(AvatarEvent::WaitForComplete { call: 2 });
    core.handle(AvatarEvent::Close { call: 3 });
    assert_eq!(
        core.handle(AvatarEvent::Close { call: 4 }),
        [AvatarEffect::Answer { call: 4, result: Ok(()) }]
    );
    assert!(core.handle(audio_input(&core, Ok(None))).is_empty());
    let error: ProviderError = lk_agents_abi::errors::ApiError::new("receiver close failed").into();
    assert_eq!(
        core.handle(AvatarEvent::ReceiverClosed(Err(error.clone()))),
        [AvatarEffect::Answer { call: 3, result: Err(RunnerError::Failed(error)) }]
    );
    assert_eq!(
        core.handle(AvatarEvent::Frame(Ok(None))),
        [AvatarEffect::Answer { call: 2, result: Ok(()) }]
    );
    assert_eq!(
        ops(&core.handle(AvatarEvent::Close { call: 5 })),
        [AvatarIoOperation::CloseReceiver]
    );
}

#[test]
fn close_preserves_a_completed_run_and_closes_an_unstarted_run() {
    for started in [false, true] {
        let mut core = if started { running() } else { RunnerCore::new(true) };
        if started {
            core.handle(audio_input(&core, Ok(None)));
            core.handle(AvatarEvent::Frame(Ok(None)));
        }
        core.handle(AvatarEvent::Close { call: 1 });
        core.handle(AvatarEvent::ReceiverClosed(Ok(())));
        core.handle(AvatarEvent::IoCancelled);
        core.handle(AvatarEvent::MediaClosed(Ok(())));
        assert_eq!(
            core.handle(AvatarEvent::WaitForComplete { call: 2 }),
            [AvatarEffect::Answer {
                call: 2,
                result: if started { Ok(()) } else { Err(RunnerError::Closed) },
            }]
        );
        assert_eq!(
            core.handle(AvatarEvent::Close { call: 3 }),
            [AvatarEffect::Answer { call: 3, result: Ok(()) }]
        );
    }
}

#[test]
fn skipped_clear_epochs_keep_each_segments_debt_until_its_clear_answers() {
    let mut core = running();
    core.handle(audio_input(&core, Ok(Some(AudioReceiverFrame::Audio(pcm(1, 0.1))))));
    core.handle(AvatarEvent::AudioPushed(Ok(())));
    assert_eq!(
        ops(&core.handle(AvatarEvent::ClearBuffer { epoch: 3 })),
        [AvatarIoOperation::ClearGenerator { clear: 3 }]
    );
    assert!(core.handle(AvatarEvent::ClearBuffer { epoch: 2 }).is_empty());
    core.handle(AvatarEvent::ClearBuffer { epoch: 7 });
    assert!(core.handle(AvatarEvent::GeneratorCleared { clear: 7, result: Ok(()) }).is_empty());
    core.handle(AvatarEvent::Audio(Ok((
        1,
        true,
        Some(token(2)),
        Some(AudioReceiverFrame::Audio(pcm(2, 0.1))),
    ))));
    let cleared = core.handle(AvatarEvent::GeneratorCleared { clear: 3, result: Ok(()) });
    assert_eq!(
        ops(&cleared),
        [AvatarIoOperation::ReportFinished { segment: token(1), position: 0.0, interrupted: true }]
    );
    let reported = core.handle(AvatarEvent::Reported { finished: true, result: Ok(()) });
    assert_eq!(
        ops(&reported),
        [AvatarIoOperation::ReportFinished { segment: token(2), position: 0.0, interrupted: true }]
    );
}

// Catches: the generator cleared while a frame of the cut audio was still on its way into it: the
// push landed after the clear and the avatar spoke the cut audio on (survey seed 1353: two frames
// 60 ms after the clear, against one frame period).
#[test]
fn a_clear_waits_for_the_audio_on_its_way_into_the_generator() {
    let mut core = running();
    let push = core.handle(audio_input(&core, Ok(Some(AudioReceiverFrame::Audio(pcm(1, 0.1))))));
    assert_eq!(
        ops(&push),
        vec![AvatarIoOperation::PushAudio(AudioReceiverFrame::Audio(pcm(1, 0.1)))]
    );
    let clear = core.handle(AvatarEvent::ClearBuffer { epoch: core.clear_epoch + 1 });
    assert!(ops(&clear).is_empty(), "cleared with the cut audio still on its way: {clear:?}");
    let pushed = core.handle(AvatarEvent::AudioPushed(Ok(())));
    assert_eq!(
        ops(&pushed),
        vec![AvatarIoOperation::ClearGenerator { clear: 1 }, AvatarIoOperation::NextAudio]
    );
}

#[test]
fn a_clear_reports_each_segment_that_the_runner_actually_read() {
    for count in 0..=2 {
        let mut core = running();
        for segment in 0..count {
            core.handle(audio_input(&core, Ok(Some(AudioReceiverFrame::Audio(pcm(segment, 0.1))))));
            core.handle(AvatarEvent::AudioPushed(Ok(())));
            core.handle(audio_input(&core, Ok(Some(AudioReceiverFrame::SegmentEnd))));
            core.handle(AvatarEvent::AudioPushed(Ok(())));
        }
        // Actual output audio contributes its known position to only the oldest report.
        if count > 0 {
            core.handle(audio(9, 0.04));
            core.handle(AvatarEvent::Pushed(Ok(())));
        }
        core.handle(AvatarEvent::ClearBuffer { epoch: core.clear_epoch + 1 });
        let effects = core.handle(AvatarEvent::GeneratorCleared { clear: 1, result: Ok(()) });
        let expected: Vec<_> = (0..count)
            .map(|index| AvatarIoOperation::ReportFinished {
                segment: token(u64::from(index)),
                position: if index == 0 { 0.04 } else { 0.0 },
                interrupted: true,
            })
            .collect();
        let mut reports = ops(&effects);
        while core.reporting {
            reports.extend(ops(
                &core.handle(AvatarEvent::Reported { finished: true, result: Ok(()) })
            ));
        }
        assert_eq!(reports, expected, "read {count} segments");
    }
}

fn audio_input(
    core: &RunnerCore,
    frame: Result<Option<AudioReceiverFrame>, ProviderError>,
) -> AvatarEvent {
    AvatarEvent::Audio(frame.map(|frame| {
        let segment = match &frame {
            Some(AudioReceiverFrame::Audio(frame)) => Some(token(u64::from(frame.data[0]))),
            Some(_) => core.owed.back().cloned(),
            None => None,
        };
        (core.clear_epoch, false, segment, frame)
    }))
}

#[test]
fn stale_first_audio_is_reported_once_before_or_after_the_generator_clear_result() {
    for generator_answered in [false, true] {
        let mut core = running();
        core.handle(AvatarEvent::ClearBuffer { epoch: 1 });
        if generator_answered {
            core.handle(AvatarEvent::GeneratorCleared { clear: 1, result: Ok(()) });
        }
        let stale = core.handle(AvatarEvent::Audio(Ok((
            0,
            true,
            Some(token(1)),
            Some(AudioReceiverFrame::Audio(pcm(1, 0.1))),
        ))));
        let report = AvatarIoOperation::ReportFinished {
            segment: token(1),
            position: 0.0,
            interrupted: true,
        };
        if generator_answered {
            assert_eq!(ops(&stale), [report.clone(), AvatarIoOperation::NextAudio]);
        } else {
            assert_eq!(ops(&stale), [AvatarIoOperation::NextAudio]);
            let failed = core.handle(AvatarEvent::GeneratorCleared {
                clear: 1,
                result: Err(lk_agents_abi::errors::ApiError::new("clear failed").into()),
            });
            assert_eq!(ops(&failed), [report]);
        }
        assert_eq!(
            ops(&core.handle(AvatarEvent::Audio(Ok((
                0,
                false,
                Some(token(1)),
                Some(AudioReceiverFrame::SegmentEnd)
            ))),)),
            [AvatarIoOperation::NextAudio]
        );
        core.handle(AvatarEvent::Reported { finished: true, result: Ok(()) });
        core.handle(AvatarEvent::Audio(Ok((
            1,
            true,
            Some(token(2)),
            Some(AudioReceiverFrame::Audio(pcm(2, 0.1))),
        ))));
        core.handle(AvatarEvent::AudioPushed(Ok(())));
        assert_eq!(
            ops(&core.handle(AvatarEvent::Frame(Ok(Some(AvatarFrame::SegmentEnd))))),
            [AvatarIoOperation::ReportFinished {
                segment: token(2),
                position: 0.0,
                interrupted: false
            }]
        );
    }
}

#[test]
fn delayed_clear_and_report_delivery_hold_new_audio_until_old_debt_is_delivered() {
    let mut core = running();
    core.handle(AvatarEvent::Audio(Ok((
        0,
        true,
        Some(token(1)),
        Some(AudioReceiverFrame::Audio(pcm(1, 0.1))),
    ))));
    core.handle(AvatarEvent::AudioPushed(Ok(())));
    core.handle(AvatarEvent::ClearBuffer { epoch: 1 });
    let new_audio = core.handle(AvatarEvent::Audio(Ok((
        1,
        true,
        Some(token(2)),
        Some(AudioReceiverFrame::Audio(pcm(2, 0.1))),
    ))));
    assert!(ops(&new_audio).is_empty(), "new audio crossed the pending generator clear");
    let stale_end = core.handle(AvatarEvent::Frame(Ok(Some(AvatarFrame::SegmentEnd))));
    assert!(ops(&stale_end).is_empty(), "the output loop must not outrun clear/report delivery");
    let cut = core.handle(AvatarEvent::GeneratorCleared { clear: 1, result: Ok(()) });
    assert_eq!(
        ops(&cut),
        [AvatarIoOperation::ReportFinished { segment: token(1), position: 0.0, interrupted: true }]
    );
    assert!(core.held_audio.is_some(), "a queued report is not a delivered acknowledgement");
    let delivered = core.handle(AvatarEvent::Reported { finished: true, result: Ok(()) });
    assert!(
        ops(&delivered)
            .contains(&AvatarIoOperation::PushAudio(AudioReceiverFrame::Audio(pcm(2, 0.1))))
    );
    core.handle(AvatarEvent::AudioPushed(Ok(())));
    core.handle(AvatarEvent::Audio(Ok((
        1,
        false,
        Some(token(2)),
        Some(AudioReceiverFrame::SegmentEnd),
    ))));
    let finished = core.handle(AvatarEvent::Frame(Ok(Some(AvatarFrame::SegmentEnd))));
    assert_eq!(
        ops(&finished),
        [AvatarIoOperation::ReportFinished {
            segment: token(2),
            position: 0.0,
            interrupted: false
        }]
    );
}

#[test]
fn failed_finish_delivery_keeps_the_barrier_and_close_releases_its_pending_frames() {
    let mut core = running();
    for segment in 0..2 {
        core.handle(AvatarEvent::Audio(Ok((
            0,
            true,
            Some(token(u64::from(segment))),
            Some(AudioReceiverFrame::Audio(pcm(segment, 0.1))),
        ))));
        core.handle(AvatarEvent::AudioPushed(Ok(())));
        core.handle(AvatarEvent::Audio(Ok((
            0,
            false,
            Some(token(1)),
            Some(AudioReceiverFrame::SegmentEnd),
        ))));
        core.handle(AvatarEvent::AudioPushed(Ok(())));
    }
    core.handle(AvatarEvent::ClearBuffer { epoch: 1 });
    core.handle(AvatarEvent::Audio(Ok((
        1,
        true,
        Some(token(3)),
        Some(AudioReceiverFrame::Audio(pcm(3, 0.1))),
    ))));
    core.handle(AvatarEvent::GeneratorCleared { clear: 1, result: Ok(()) });
    assert_eq!(core.reports.len(), 1);
    core.handle(AvatarEvent::WaitForComplete { call: 42 });
    let error: ProviderError =
        lk_agents_abi::errors::ApiError::new("finish delivery failed").into();
    let failed = core.handle(AvatarEvent::Reported { finished: true, result: Err(error.clone()) });
    assert!(ops(&failed).is_empty(), "a failed delivery must not advance reports or audio");
    assert!(
        failed
            .contains(&AvatarEffect::Answer { call: 42, result: Err(RunnerError::Failed(error)) })
    );
    assert!(core.held_audio.is_some());
    assert_eq!(core.reports.len(), 1);
    core.handle(AvatarEvent::Close { call: 43 });
    assert_eq!(core.handle(AvatarEvent::ReceiverClosed(Ok(()))), [AvatarEffect::CancelIo]);
    core.handle(AvatarEvent::IoCancelled);
    assert!(core.held_audio.is_none());
    assert!(core.reports.is_empty());
    assert!(!core.reporting);
    assert!(!core.frame_pull_paused);
}

#[test]
fn a_room_stream_failure_settles_pending_start_and_completion() {
    let mut core = RunnerCore::new(false);
    core.handle(AvatarEvent::Start { call: 1, connected: false });
    core.handle(AvatarEvent::WaitForComplete { call: 2 });
    let error =
        ProviderError::Local(lk_agents_abi::errors::ApiError::new("room event delivery failed"));
    let failed = RunnerError::Failed(error.clone());
    let effects = core.handle(AvatarEvent::RoomFailed(error));
    assert!(effects.contains(&AvatarEffect::Answer { call: 1, result: Err(failed.clone()) }));
    assert!(effects.contains(&AvatarEffect::Answer { call: 2, result: Err(failed.clone()) }));
    assert_eq!(
        core.handle(AvatarEvent::WaitForComplete { call: 3 }),
        vec![AvatarEffect::Answer { call: 3, result: Err(failed) }]
    );
}

// Catches: a receiver that keys no segments (a custom `AudioReceiver`) failing the read loop, or
// two of its segments, or one cut by a clear and the audio after it, sharing a key.
#[test]
fn unkeyed_audio_is_keyed_per_segment_and_per_clear() {
    let unkeyed = |core: &RunnerCore, frame| {
        AvatarEvent::Audio(Ok((core.clear_epoch, false, None, Some(frame))))
    };
    let owed = |core: &RunnerCore| core.owed.back().map(|segment| segment.playback.segment);
    let mut core = running();
    let first = core.handle(unkeyed(&core, AudioReceiverFrame::Audio(pcm(1, 0.1))));
    assert_eq!(
        ops(&first),
        vec![AvatarIoOperation::PushAudio(AudioReceiverFrame::Audio(pcm(1, 0.1)))]
    );
    core.handle(AvatarEvent::AudioPushed(Ok(())));
    core.handle(unkeyed(&core, AudioReceiverFrame::Audio(pcm(2, 0.1))));
    core.handle(AvatarEvent::AudioPushed(Ok(())));
    assert_eq!((core.owed.len(), owed(&core)), (1, Some(1)), "one segment until its end");
    core.handle(unkeyed(&core, AudioReceiverFrame::SegmentEnd));
    core.handle(AvatarEvent::AudioPushed(Ok(())));
    core.handle(unkeyed(&core, AudioReceiverFrame::Audio(pcm(3, 0.1))));
    core.handle(AvatarEvent::AudioPushed(Ok(())));
    assert_eq!(owed(&core), Some(2));
    core.handle(AvatarEvent::ClearBuffer { epoch: core.clear_epoch + 1 });
    let cleared = core.handle(AvatarEvent::GeneratorCleared { clear: 1, result: Ok(()) });
    let reported = core.handle(AvatarEvent::Reported { finished: true, result: Ok(()) });
    let finished: Vec<_> = [cleared, reported]
        .iter()
        .flat_map(|effects| ops(effects))
        .filter_map(|op| match op {
            AvatarIoOperation::ReportFinished { segment, .. } => Some(segment.playback.segment),
            _ => None,
        })
        .collect();
    assert_eq!(finished, vec![1, 2]);
    core.handle(AvatarEvent::Reported { finished: true, result: Ok(()) });
    core.handle(unkeyed(&core, AudioReceiverFrame::Audio(pcm(4, 0.1))));
    assert_eq!(owed(&core), Some(3), "a clear ends the segment");
}
