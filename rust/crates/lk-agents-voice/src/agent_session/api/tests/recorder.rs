//! A reused physical output still has one recorder route per attachment.
use super::*;
use lk_agents_abi::voice::io::{PlaybackFinished, PlaybackSegment};
use lk_agents_test_support::playback::MissingPlaybackReports;
use lk_agents_test_support::recorder::RecorderProbe;

#[tokio::test(start_paused = true)]
async fn reattaching_one_physical_output_does_not_duplicate_recorder_playback() {
    let api = unobserved(fake_room("recorder-output-identity").1, None);
    let Stage::Ready(ready) = std::mem::replace(&mut *api.lock(), Stage::Closed) else {
        panic!("session must be ready");
    };
    let (cores, _) = ready.builder.start(&mut NoLegs).await.unwrap();
    let shared = &cores.shared;
    let output = Arc::new(MissingPlaybackReports::default());
    let original = shared.resources.register_audio_output(output.clone()).id;
    let replacement = shared.resources.register_audio_output(output.clone()).id;
    assert_ne!(original, replacement);
    let recorder = Arc::new(RecorderProbe::default());
    *shared.recorder.write().unwrap() =
        Some(crate::recorder_io::RecorderTap::new(recorder.clone(), shared.time.clock()));
    let first_observer: Arc<dyn lk_agents_abi::voice::io::PlaybackObserver> = Arc::new(
        crate::recorder_io::RecorderPlaybackObserver::new(shared.recorder_tap().unwrap(), original),
    );
    output.0.observe(first_observer.clone());
    let first = PlaybackSegment { attachment: original.0, segment: 1 };
    assert!(output.0.begin_segment(first));
    output.0.note_captured(Duration::from_millis(10));
    output.0.note_flush();
    // The first operation still owes its report when the same host object is reattached.
    let next_observer: Arc<dyn lk_agents_abi::voice::io::PlaybackObserver> =
        Arc::new(crate::recorder_io::RecorderPlaybackObserver::new(
            shared.recorder_tap().unwrap(),
            replacement,
        ));
    output.0.observe(next_observer.clone());
    let now = shared.time.now();
    let completed = PlaybackFinished {
        playback_position: Duration::from_millis(10),
        interrupted: false,
        synchronized_transcript: None,
    };
    assert!(output.0.playback_started(first, now));
    assert!(output.0.playback_progressed(first, now, Duration::ZERO, Duration::from_millis(10)));
    assert!(output.0.playback_finished(first, completed.clone()));
    assert_eq!(
        recorder.playback_counts(),
        [1, 1, 1],
        "the replacement observer replayed the original attachment's physical events"
    );
    let next = PlaybackSegment { attachment: replacement.0, segment: 2 };
    assert!(output.0.begin_segment(next));
    output.0.note_captured(Duration::from_millis(10));
    output.0.note_flush();
    assert!(!output.0.playback_finished(first, completed.clone()), "old completion was reused");
    assert!(output.0.playback_started(next, now));
    assert!(output.0.playback_progressed(next, now, Duration::ZERO, Duration::from_millis(10)));
    assert!(output.0.playback_finished(next, completed));
    assert_eq!(
        recorder.playback_counts(),
        [2, 2, 2],
        "the retained original observer replayed the replacement attachment's events"
    );
    output.0.unobserve(&first_observer);
    output.0.unobserve(&next_observer);
}
