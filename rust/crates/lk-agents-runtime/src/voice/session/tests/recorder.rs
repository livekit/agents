//! A reused physical output still has one recorder route per attachment.
use lk_agents_abi::AudioFrame;
use lk_agents_abi::recorder::{AudioRecorder, RecordingFailure};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::{PlaybackFinished, PlaybackSegment, PlaybackTracker};
use lk_agents_net::lock::Mutex;
use lk_agents_voice::recorder_io::{RecorderPlaybackObserver, RecorderTap};
use std::sync::Arc;
use std::time::Duration;

/// Counts the recorder's playback started, progressed and finished callbacks.
#[derive(Default)]
struct RecorderProbe(Mutex<[usize; 3]>);

impl AudioRecorder for RecorderProbe {
    fn on_input_frame(&self, _: f64, _: &AudioFrame) {}
    fn capture_frame(&self, _: f64, _: &AudioFrame) {}
    fn on_playback_started(&self, _: f64) {
        self.0.lock()[0] += 1;
    }
    fn on_playback_progressed(&self, _: f64, _: f64, _: f64) {
        self.0.lock()[1] += 1;
    }
    fn on_playback_finished(&self, _: f64, _: f64) {
        self.0.lock()[2] += 1;
    }
    fn flush(&self, _: f64) -> Result<(), RecordingFailure> {
        Ok(())
    }
}

#[test]
fn reattaching_one_physical_output_does_not_duplicate_recorder_playback() {
    let output = PlaybackTracker::new();
    let (original, replacement) = (ResourceId(1), ResourceId(2));
    let recorder = Arc::new(RecorderProbe::default());
    let counts = || *recorder.0.lock();
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let tap = RecorderTap::new(recorder.clone(), clock.clone());
    let first_observer: Arc<dyn lk_agents_abi::voice::io::PlaybackObserver> =
        Arc::new(RecorderPlaybackObserver::new(tap.clone(), original));
    output.observe(first_observer.clone());
    let first = PlaybackSegment { attachment: original.0, segment: 1 };
    assert!(output.begin_segment(first));
    output.note_captured(Duration::from_millis(10));
    output.note_flush();
    // The first operation still owes its report when the same host object is reattached.
    let next_observer: Arc<dyn lk_agents_abi::voice::io::PlaybackObserver> =
        Arc::new(RecorderPlaybackObserver::new(tap, replacement));
    output.observe(next_observer.clone());
    let now = lk_agents_abi::Clock::now(clock.as_ref());
    let completed = PlaybackFinished {
        playback_position: Duration::from_millis(10),
        interrupted: false,
        synchronized_transcript: None,
    };
    assert!(output.playback_started(first, now));
    assert!(output.playback_progressed(first, now, Duration::ZERO, Duration::from_millis(10)));
    assert!(output.playback_finished(first, completed.clone()));
    assert_eq!(
        counts(),
        [1, 1, 1],
        "the replacement observer replayed the original attachment's physical events"
    );
    let next = PlaybackSegment { attachment: replacement.0, segment: 2 };
    assert!(output.begin_segment(next));
    output.note_captured(Duration::from_millis(10));
    output.note_flush();
    assert!(!output.playback_finished(first, completed.clone()), "old completion was reused");
    assert!(output.playback_started(next, now));
    assert!(output.playback_progressed(next, now, Duration::ZERO, Duration::from_millis(10)));
    assert!(output.playback_finished(next, completed));
    assert_eq!(
        counts(),
        [2, 2, 2],
        "the retained original observer replayed the replacement attachment's events"
    );
    output.unobserve(&first_observer);
    output.unobserve(&next_observer);
}
