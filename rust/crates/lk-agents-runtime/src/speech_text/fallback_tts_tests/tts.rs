use super::*;

// Catches: the TTS drain never marking the serving entry (`AfterOutput::Stop`, `tts:293-297`).
#[tokio::test]
async fn the_tts_that_emitted_audio_retains_its_failure_and_identity() {
    let (broken, _) = built("broken", 24_000, 1, 0, usize::MAX, false);
    let (dying, _) = built("dying", 24_000, 1, 1, 0, true);
    let fb = tts_chain(vec![broken, dying]);

    assert_eq!(fb.active().label(), "broken", "before any traffic the primary is active");

    let (sink, mut source) = within("stream", fb.stream()).await.expect("opened");
    sink.push("hello".into()).expect("push");
    sink.end_input();

    assert!(within("the one frame", source.next()).await.expect("no error").is_some());
    assert!(within("the failure", source.next()).await.is_err());
    assert_eq!(fb.active().label(), "dying", "the entry that emitted audio owns the turn");
    source.close(NO_DEADLINE).await;
    fb.aclose();
}

// Catches: a `TokenLog` still pointing at a provider the chain has finished with.
#[tokio::test]
async fn the_child_sink_is_released_when_the_stream_that_owns_it_ends() {
    let (good, pushed) = built("good", 24_000, 1, 1, 0, false);
    let fb = tts_chain(vec![good]);

    let (sink, mut source) = within("stream", fb.stream()).await.expect("opened");
    sink.push("hello".into()).expect("push");
    sink.end_input();
    drain_tts(&mut source).await;

    let before = pushed.lock().expect("pushed").len();
    sink.push("late".into()).expect("push");
    assert_eq!(
        pushed.lock().expect("pushed").len(),
        before,
        "the turn is over: a late token must not reach a provider the chain has released"
    );
    source.close(NO_DEADLINE).await;
}

// Catches: a resampled frame whose header and payload disagree (run at two channels so `/` vs
// `*` differ); also pins label, channel count and capabilities.
#[tokio::test]
async fn a_resampled_stereo_frame_carries_exactly_the_samples_its_header_claims() {
    let (slow, _) = built("slow", 16_000, 2, 3, 0, false);
    let (fast, _) = built("fast", 24_000, 2, 3, 0, false);
    let fb = tts_chain(vec![slow, fast]);

    assert_eq!(fb.label(), "tts.FallbackAdapter");
    assert_eq!(fb.sample_rate(), 24_000, "the maximum of the entries");
    assert_eq!(fb.num_channels(), 2, "both entries are stereo; the adapter must not invent mono");
    assert!(fb.capabilities().streaming, "the merged shape, not Default");
    assert!(fb.needs_resampling(0), "the 16 kHz primary");
    assert!(!fb.needs_resampling(1), "the entry already at the chain's rate");

    let mut source = within("synthesize", fb.synthesize("hello")).await.expect("a source");
    let mut samples = 0usize;
    let mut frames = 0usize;
    for a in drain_tts(&mut source).await {
        frames += 1;
        assert_eq!(a.frame.sample_rate, 24_000);
        assert_eq!(a.frame.num_channels, 2);
        assert_eq!(
            a.frame.data.len(),
            a.frame.samples_per_channel as usize * 2 * 2,
            "bytes = samples_per_channel x channels x 2, or the consumer slices the wrong frame"
        );
        samples += a.frame.samples_per_channel as usize;
    }
    assert!(frames > 0, "the primary served");
    // soxr's filter delay makes the exact count implementation-defined; assert the ratio.
    assert!(samples > 600, "expected ~720 upsampled samples per channel, got {samples}");
    source.close(NO_DEADLINE).await;
}
