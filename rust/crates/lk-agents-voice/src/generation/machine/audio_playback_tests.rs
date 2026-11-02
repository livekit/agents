use super::*;
use lk_agents_abi::AudioFrame;
use lk_agents_abi::voice::io::PlaybackFinished;

fn step(playback: &mut AudioPlayback, event: AudioPlaybackEvent) -> AudioPlaybackEffect {
    playback.handle(event, Ts::ZERO).expect("playback still requests work")
}

fn reading(has_sink: bool, stop_at_boundary: bool) -> AudioPlayback {
    let mut playback = AudioPlayback::new(SpeechId(1), Some(key()), has_sink, stop_at_boundary);
    assert!(matches!(playback.begin(), AudioPlaybackEffect::CheckInterruption));
    assert!(matches!(
        step(&mut playback, AudioPlaybackEvent::InterruptionChecked { interrupted: false }),
        AudioPlaybackEffect::ReadSource
    ));
    playback
}

fn key() -> PlaybackSegment {
    PlaybackSegment { attachment: 1, segment: 1 }
}

fn frame() -> AudioFrame {
    AudioFrame::new(vec![0; 640], 16_000, 1, 320)
}

fn source(next: Next) -> AudioPlaybackEvent {
    AudioPlaybackEvent::SourceRead { next: Some(Ok(next)), interrupted: false }
}

fn begin_cut(frames: bool) -> AudioPlayback {
    let mut playback = reading(true, false);
    if frames {
        playback.captured(Some(key()), Duration::from_millis(20));
    }
    assert!(matches!(
        step(&mut playback, AudioPlaybackEvent::SourceRead { next: None, interrupted: true }),
        AudioPlaybackEffect::ReadSourceTranscript
    ));
    assert!(matches!(
        step(&mut playback, AudioPlaybackEvent::SourceTranscriptRead("hello".into())),
        AudioPlaybackEffect::ClearOutput
    ));
    playback
}

fn complete(effect: AudioPlaybackEffect) -> Heard {
    match effect {
        AudioPlaybackEffect::Complete(Ok(heard)) => heard,
        _ => panic!("expected successful playback completion"),
    }
}

// A cut before the sink accepts audio must not wait for a receipt or credit generated words.
#[test]
fn a_zero_frame_cut_never_polls_and_completes_once() {
    let mut playback = begin_cut(false);
    let heard = match step(&mut playback, AudioPlaybackEvent::OutputCleared) {
        AudioPlaybackEffect::FinishCut { heard, reported: Some(true) } => heard,
        _ => panic!("zero accepted frames should report nothing heard"),
    };
    assert!(heard.text.is_empty());
    assert_eq!(heard.frames, 0);
    assert!(!heard.complete);
    let heard = complete(step(&mut playback, AudioPlaybackEvent::CutFinished(heard)));
    assert_eq!(heard.position, Duration::ZERO);
    assert!(playback.handle(AudioPlaybackEvent::OutputCleared, Ts::ZERO).is_none());
    assert!(playback.on_report(SinkReport::Reset, now()).is_empty());
}

fn now() -> Now {
    Now { at: Ts::ZERO, wall: std::time::UNIX_EPOCH }
}

// Receipt grace is a bounded number of scheduling turns, then a timed transport allowance.
// A closing runtime's stopped clock must never be required to complete the cut.
#[test]
fn cut_receipts_get_sixteen_turns_then_a_bound_only_while_open() {
    for closing in [false, true] {
        let mut playback = begin_cut(true);
        assert!(matches!(
            step(&mut playback, AudioPlaybackEvent::OutputCleared),
            AudioPlaybackEffect::PollCutReceipt
        ));
        for attempt in 0..16 {
            assert!(matches!(
                step(&mut playback, AudioPlaybackEvent::CutReceiptPolled(None)),
                AudioPlaybackEffect::YieldForCutReceipt
            ));
            let effect = step(&mut playback, AudioPlaybackEvent::CutReceiptYielded { closing });
            if attempt < 15 {
                assert!(matches!(effect, AudioPlaybackEffect::PollCutReceipt));
            } else if closing {
                assert!(matches!(effect, AudioPlaybackEffect::ResolveCutReceipt(None)));
            } else {
                match effect {
                    AudioPlaybackEffect::WaitForCutReceiptUntil(at) => {
                        assert_eq!(at, Ts::ZERO.saturating_add(Duration::from_secs(1)));
                    }
                    _ => panic!("open playback must allow the bounded transport wait"),
                }
                assert!(matches!(
                    step(&mut playback, AudioPlaybackEvent::CutReceiptWaited(None)),
                    AudioPlaybackEffect::ResolveCutReceipt(None)
                ));
            }
        }
        let heard = match step(
            &mut playback,
            AudioPlaybackEvent::CutReceiptResolved {
                report: None,
                position: Duration::from_millis(10),
            },
        ) {
            AudioPlaybackEffect::FinishCut { heard, reported: None } => heard,
            _ => panic!("missing receipt uses the tracker lower bound without inventing a finish"),
        };
        assert_eq!(heard.position, Duration::from_millis(10));
        assert_eq!(heard.text, "hello");
        assert!(!heard.complete);
        complete(step(&mut playback, AudioPlaybackEvent::CutFinished(heard)));
    }
}

// A clear can arrive after the sink played everything; preserve the physical receipt flag,
// even though the source forwarding was interrupted and cannot claim complete playback.
#[test]
fn a_cut_keeps_the_sinks_actual_finish_flag() {
    let mut playback = begin_cut(true);
    assert!(matches!(
        step(&mut playback, AudioPlaybackEvent::OutputCleared),
        AudioPlaybackEffect::PollCutReceipt
    ));
    let report = PlaybackFinished {
        playback_position: Duration::from_millis(20),
        interrupted: false,
        ..PlaybackFinished::default()
    };
    let report = match step(&mut playback, AudioPlaybackEvent::CutReceiptPolled(Some(report))) {
        AudioPlaybackEffect::ResolveCutReceipt(Some(report)) => report,
        _ => panic!("receipt must be accounted before heard projection"),
    };
    let heard = match step(
        &mut playback,
        AudioPlaybackEvent::CutReceiptResolved { report: Some(report), position: Duration::ZERO },
    ) {
        AudioPlaybackEffect::FinishCut { heard, reported: Some(false) } => heard,
        _ => panic!("do not change a whole physical finish into a cut receipt"),
    };
    assert_eq!(heard.position, Duration::from_millis(20));
    assert!(!heard.complete);
    complete(step(&mut playback, AudioPlaybackEvent::CutFinished(heard)));
}

// A normal internal boundary flushes the resampler tail before reading the next segment.
// Realtime's message boundary instead enters finishing and takes one final transcript snapshot.
#[test]
fn boundaries_forward_the_tail_then_either_resume_or_finish() {
    for stop_at_boundary in [false, true] {
        let mut playback = reading(true, stop_at_boundary);
        assert!(matches!(
            step(&mut playback, source(Next::Boundary)),
            AudioPlaybackEffect::ForwardResamplerTail
        ));
        let effect = step(&mut playback, AudioPlaybackEvent::ResamplerTailForwarded(Ok(())));
        if stop_at_boundary {
            assert!(matches!(effect, AudioPlaybackEffect::ReadSourceTranscript));
            let heard = complete(step(
                &mut playback,
                AudioPlaybackEvent::SourceTranscriptRead("done".into()),
            ));
            assert_eq!(heard.text, "done");
            assert!(heard.complete);
        } else {
            assert!(matches!(effect, AudioPlaybackEffect::FlushBoundary));
            assert!(matches!(
                step(&mut playback, AudioPlaybackEvent::BoundaryFlushed),
                AudioPlaybackEffect::CheckInterruption
            ));
            assert!(matches!(
                step(&mut playback, AudioPlaybackEvent::InterruptionChecked { interrupted: false }),
                AudioPlaybackEffect::ReadSource
            ));
        }
    }
}

// No-output operation is still a source consumer, but must never request launch or capture.
#[test]
fn an_outputless_source_drains_until_its_own_completion_boundary() {
    let mut playback = reading(false, false);
    assert!(matches!(
        step(&mut playback, source(Next::Frame(frame()))),
        AudioPlaybackEffect::CheckInterruption
    ));
    let _ = step(&mut playback, AudioPlaybackEvent::InterruptionChecked { interrupted: false });
    assert!(matches!(
        step(&mut playback, source(Next::Boundary)),
        AudioPlaybackEffect::CheckInterruption
    ));
    let _ = step(&mut playback, AudioPlaybackEvent::InterruptionChecked { interrupted: false });
    assert!(matches!(
        step(&mut playback, source(Next::End)),
        AudioPlaybackEffect::ReadSourceTranscript
    ));
    let heard = complete(step(
        &mut playback,
        AudioPlaybackEvent::SourceTranscriptRead("visible without audio".into()),
    ));
    assert_eq!(heard.text, "visible without audio");
    assert_eq!(heard.frames, 0);
    assert!(heard.complete);
}

#[test]
fn an_outputless_source_error_is_not_a_successful_silent_playback() {
    let mut playback = reading(false, true);
    assert!(matches!(
        step(
            &mut playback,
            AudioPlaybackEvent::SourceRead {
                next: Some(Err(DriverError::Missing("source"))),
                interrupted: false,
            },
        ),
        AudioPlaybackEffect::ReadSourceTranscript
    ));
    match step(&mut playback, AudioPlaybackEvent::SourceTranscriptRead("unconfirmed".into())) {
        AudioPlaybackEffect::Complete(Err(failure)) => {
            assert!(matches!(failure.error, DriverError::Missing("source")));
            assert!(failure.heard.is_none());
        }
        _ => panic!("source failure must survive the no-output path"),
    }
}

// The first capture waits for the independent launch hold; interruption while parked cannot
// capture that retained first frame or turn it into credited playback.
#[test]
fn a_first_frame_waits_for_launch_permission_and_can_be_interrupted() {
    let mut playback = reading(true, false);
    assert!(matches!(
        step(&mut playback, source(Next::Frame(frame()))),
        AudioPlaybackEffect::WaitForLaunchPermission
    ));
    assert!(matches!(
        step(&mut playback, AudioPlaybackEvent::LaunchPermissionResolved { allowed: false }),
        AudioPlaybackEffect::ReadSourceTranscript
    ));
    assert_eq!(playback.pushed_frames(), 0);
    assert!(matches!(
        step(&mut playback, AudioPlaybackEvent::SourceTranscriptRead("never played".into())),
        AudioPlaybackEffect::ClearOutput
    ));
}

// Receipt transport failures may not replace an earlier provider failure or credit its raw
// text as heard. Accepting a real receipt still preserves its synchronized transcript.
#[test]
fn source_error_precedence_survives_receipt_success_and_failure() {
    for receipt_available in [false, true] {
        let mut playback = reading(true, false);
        playback.captured(Some(key()), Duration::from_millis(20));
        assert!(matches!(
            step(
                &mut playback,
                AudioPlaybackEvent::SourceRead {
                    next: Some(Err(DriverError::Missing("source"))),
                    interrupted: false,
                },
            ),
            AudioPlaybackEffect::ReadSourceTranscript
        ));
        assert!(matches!(
            step(&mut playback, AudioPlaybackEvent::SourceTranscriptRead("raw words".into())),
            AudioPlaybackEffect::WaitForPlaybackReceipt
        ));
        if receipt_available {
            let receipt = PlaybackFinished {
                synchronized_transcript: Some("confirmed".into()),
                playback_position: Duration::from_millis(20),
                ..PlaybackFinished::default()
            };
            let receipt = match step(
                &mut playback,
                AudioPlaybackEvent::PlaybackReceiptReceived(Ok(Some(receipt))),
            ) {
                AudioPlaybackEffect::AcceptPlaybackReceipt(receipt) => receipt,
                _ => panic!("receipt must be accepted before error confirmation"),
            };
            assert!(matches!(
                step(&mut playback, AudioPlaybackEvent::PlaybackReceiptAccepted(receipt)),
                AudioPlaybackEffect::ConfirmPlayed
            ));
        } else {
            assert!(matches!(
                step(
                    &mut playback,
                    AudioPlaybackEvent::PlaybackReceiptReceived(Err(DriverError::Missing(
                        "receipt"
                    ))),
                ),
                AudioPlaybackEffect::ClearOutput
            ));
            // An immediately available cut receipt is enough to reach original-error confirmation.
            let _ = step(&mut playback, AudioPlaybackEvent::OutputCleared);
            let receipt = PlaybackFinished::default();
            let report =
                match step(&mut playback, AudioPlaybackEvent::CutReceiptPolled(Some(receipt))) {
                    AudioPlaybackEffect::ResolveCutReceipt(report) => report,
                    _ => panic!("cut receipt resolves before completion"),
                };
            let heard = match step(
                &mut playback,
                AudioPlaybackEvent::CutReceiptResolved { report, position: Duration::ZERO },
            ) {
                AudioPlaybackEffect::FinishCut { heard, .. } => heard,
                _ => panic!("cut settles before error confirmation"),
            };
            assert!(matches!(
                step(&mut playback, AudioPlaybackEvent::CutFinished(heard)),
                AudioPlaybackEffect::ConfirmPlayed
            ));
        }
        match step(&mut playback, AudioPlaybackEvent::PlayedConfirmed) {
            AudioPlaybackEffect::Complete(Err(failure)) => {
                assert!(matches!(failure.error, DriverError::Missing("source")));
                assert_eq!(
                    failure.synchronized.as_deref(),
                    receipt_available.then_some("confirmed")
                );
                assert_eq!(failure.heard.expect("confirmed lower bound").text, "");
            }
            _ => panic!("original source failure must win"),
        }
    }
}
