use super::*;
use crate::test_support_rtc::FakeRtc;
use lk_agents_abi::TokioSpawner;
use lk_agents_abi::voice::rtc::RtcError;

#[tokio::test]
async fn publication_arriving_after_close_is_retired() {
    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let room = rtc.room("late-publish");
    let local = room.local_participant();
    local.flow_publications(false);
    let output = ParticipantAudioOutput::new(
        room,
        clock,
        Arc::new(TokioSpawner::current()),
        AudioOutputConfig::default(),
    );
    let mut start = Box::pin(output.start());
    crate::room_io::owner::tests::pending(start.as_mut()).await;
    let mut close = Box::pin(output.aclose());
    crate::room_io::owner::tests::pending(close.as_mut()).await;
    local.flow_publications(true);
    close.await;
    let result = start.await;
    assert!(local.publications().is_empty(), "the late SID must be unpublished");
    assert!(result.is_err(), "closed output must refuse adoption");
    assert!(output.shared.lock().source.is_none());
}

#[tokio::test]
async fn publication_failure_releases_capture_waiters() {
    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let room = rtc.room("failed-publish");
    room.local_participant().fail_with(RtcError::host("publish failed"));
    let output = ParticipantAudioOutput::new(
        room,
        clock,
        Arc::new(TokioSpawner::current()),
        AudioOutputConfig::default(),
    );
    assert!(output.start().await.is_err());
    let result = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        Duration::from_millis(50),
        output.capture(&lk_agents_abi::voice::rtc::silence_frame(480, 24000, 1)),
    )
    .await;
    assert!(matches!(result, Ok(Err(_))), "capture must receive the publication failure");
    output.aclose().await;
}
#[tokio::test]
async fn paused_room_output_backpressures_capture() {
    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let room = rtc.room("backpressure");
    let output = ParticipantAudioOutput::new(
        room,
        clock,
        Arc::new(TokioSpawner::current()),
        AudioOutputConfig::default(),
    );
    output.start().await.unwrap();
    output.pause();
    let result = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        Duration::from_millis(50),
        async {
            output
                .capture(&lk_agents_abi::voice::rtc::silence_frame(24000, 24000, 1))
                .await
                .unwrap();
        },
    )
    .await;
    assert!(
        result.is_err(),
        "a paused sink must bound accepted audio instead of buffering a second"
    );
    output.aclose().await;
}

#[tokio::test]
async fn concurrent_starts_share_one_publication() {
    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let room = rtc.room("joined-start");
    let local = room.local_participant();
    let output = ParticipantAudioOutput::new(
        room,
        clock,
        Arc::new(TokioSpawner::current()),
        AudioOutputConfig::default(),
    );
    local.flow_publications(false);
    let mut first = Box::pin(output.start());
    let mut second = Box::pin(output.start());
    crate::room_io::owner::tests::pending(first.as_mut()).await;
    crate::room_io::owner::tests::pending(second.as_mut()).await;
    local.flow_publications(true);
    first.await.unwrap();
    second.await.unwrap();
    assert_eq!(local.publications().len(), 1, "start joins its existing creation");
    output.aclose().await;
    assert!(local.publications().is_empty());
}
#[tokio::test]
async fn close_finishes_a_captured_segment_as_interrupted() {
    use lk_agents_abi::voice::io::AudioOutputExt;
    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let output = ParticipantAudioOutput::new(
        rtc.room("close-segment"),
        clock,
        Arc::new(TokioSpawner::current()),
        AudioOutputConfig::default(),
    );
    output.start().await.unwrap();
    output.pause();
    output.capture_frame(&lk_agents_abi::voice::rtc::silence_frame(480, 24000, 1)).await.unwrap();
    output.aclose().await;
    let finished = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        Duration::from_millis(50),
        output.tracker().wait_for_playout(),
    )
    .await
    .expect("close must settle the captured segment");
    assert!(finished.interrupted);
    assert_eq!(finished.playback_position, Duration::ZERO);
}

#[tokio::test]
async fn cancelled_start_keeps_ownership_of_the_created_track() {
    let (rtc, clock) = FakeRtc::with_virtual_clock();
    let room = rtc.room("cancel-start");
    let local = room.local_participant();
    local.flow_publications(false);
    let output = ParticipantAudioOutput::new(
        room,
        clock,
        Arc::new(TokioSpawner::current()),
        AudioOutputConfig::default(),
    );
    let mut start = Box::pin(output.start());
    crate::room_io::owner::tests::pending(start.as_mut()).await;
    tokio::task::yield_now().await;
    assert_eq!(
        local.publications().len(),
        1,
        "the server has created the track before acknowledging it"
    );
    drop(start);
    let mut close = Box::pin(output.aclose());
    crate::room_io::owner::tests::pending(close.as_mut()).await;
    local.flow_publications(true);
    close.await;
    assert!(local.publications().is_empty(), "cancellation must not orphan a created SID");
}
#[tokio::test]
async fn source_capture_failure_releases_playout_and_refuses_further_audio() {
    use lk_agents_abi::voice::io::AudioOutputExt;

    async fn finished(output: &ParticipantAudioOutput) -> PlaybackFinished {
        lk_agents_abi::timeout(
            &lk_agents_abi::RealClock::new(),
            Duration::from_secs(1),
            output.tracker().wait_for_playout(),
        )
        .await
        .expect("source failure must settle the captured segment")
    }

    let mut outcomes = Vec::new();
    for mode in ["fresh", "queued", "heard", "blocked"] {
        let (rtc, clock) = FakeRtc::with_virtual_clock();
        let room = rtc.room(mode);
        let local = room.local_participant();
        let output = ParticipantAudioOutput::new(
            room,
            clock.clone(),
            Arc::new(TokioSpawner::current()),
            AudioOutputConfig::default(),
        );
        output.start().await.unwrap();
        let frame = lk_agents_abi::voice::rtc::silence_frame(480, 24_000, 1);
        let (refused, playback) = if mode == "blocked" {
            output.pause();
            for _ in 0..10 {
                output.capture_frame(&frame).await.unwrap();
            }
            let mut waiting = Box::pin(output.capture_frame(&frame));
            crate::room_io::owner::tests::pending(waiting.as_mut()).await;
            local.sources()[0].close().await.unwrap();
            output.resume();
            let refused = waiting.await;
            (refused, finished(&output).await)
        } else {
            if mode == "queued" || mode == "heard" {
                output.capture_frame(&frame).await.unwrap();
                for _ in 0..8 {
                    tokio::task::yield_now().await;
                }
                assert_eq!(local.sources()[0].captured().len(), 1);
                if mode == "heard" {
                    clock.advance(Duration::from_millis(10));
                }
            }
            local.sources()[0].close().await.unwrap();
            output.capture_frame(&frame).await.unwrap();
            output.flush();
            for _ in 0..8 {
                tokio::task::yield_now().await;
            }
            let playback = finished(&output).await;
            (output.capture_frame(&frame).await, playback)
        };
        output.flush_segment();
        output.aclose().await;
        let settled = lk_agents_abi::timeout(
            &lk_agents_abi::RealClock::new(),
            Duration::from_secs(1),
            output.tracker().wait_for_playout(),
        )
        .await
        .is_ok();
        outcomes.push((
            mode,
            playback.interrupted,
            playback.playback_position,
            refused.is_err(),
            settled,
        ));
    }
    assert_eq!(
        outcomes,
        [
            ("fresh", true, Duration::ZERO, true, true),
            ("queued", true, Duration::ZERO, true, true),
            ("heard", true, Duration::from_millis(10), true, true),
            ("blocked", true, Duration::ZERO, true, true),
        ],
        "source failure must preserve heard media, refuse parked and later captures, and settle every waiter"
    );
}
