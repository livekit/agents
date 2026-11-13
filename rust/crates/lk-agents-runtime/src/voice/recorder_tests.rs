use crate::voice::recorder::{RecorderFeed, RecorderPlaybackObserver, RecorderTap};
use lk_agents_abi::AudioFrame;
use lk_agents_abi::recorder::{AudioRecorder, RecordingFailure};
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::{PlaybackFinished, PlaybackSegment, PlaybackTracker};
use parking_lot::Mutex;
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
    let output = crate::voice::playback::Tracker::new();
    let (original, replacement) = (ResourceId(1), ResourceId(2));
    let recorder = Arc::new(RecorderProbe::default());
    let counts = || *recorder.0.lock();
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let feed = Arc::new(RecorderFeed::default());
    feed.attach(RecorderTap::new(recorder.clone(), clock.clone()), &[]);
    let first_observer: Arc<dyn lk_agents_abi::voice::io::PlaybackObserver> =
        Arc::new(RecorderPlaybackObserver::new(feed.clone(), original));
    output.observe(first_observer.clone());
    let first = PlaybackSegment { attachment: original.0, segment: 1 };
    assert!(output.begin_segment(first));
    output.note_captured(Duration::from_millis(10));
    output.note_flush();
    // The first operation still owes its report when the same host object is reattached.
    let next_observer: Arc<dyn lk_agents_abi::voice::io::PlaybackObserver> =
        Arc::new(RecorderPlaybackObserver::new(feed, replacement));
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

/// Keeps the first sample of every frame the recorder captures.
#[derive(Default)]
struct CaptureProbe(Mutex<Vec<i16>>);

impl AudioRecorder for CaptureProbe {
    fn on_input_frame(&self, _: f64, _: &AudioFrame) {}
    fn capture_frame(&self, _: f64, frame: &AudioFrame) {
        self.0.lock().push(i16::from_le_bytes([frame.data[0], frame.data[1]]));
    }
    fn on_playback_started(&self, _: f64) {}
    fn on_playback_progressed(&self, _: f64, _: f64, _: f64) {}
    fn on_playback_finished(&self, _: f64, _: f64) {}
    fn flush(&self, _: f64) -> Result<(), RecordingFailure> {
        Ok(())
    }
}

// Catches: a recorder attached on another thread while a speech captured. The capture kept its
// frame and read the recorder in two steps, and the attach replayed the kept audio and published
// the recorder in two more, so a frame landing between them reached the recording twice or never.
#[test]
fn a_recorder_attached_while_frames_are_captured_hears_each_once_in_order() {
    const FRAMES: i16 = 2_000;
    let segment = PlaybackSegment { attachment: 1, segment: 1 };
    let feed = RecorderFeed::default();
    let recorder = Arc::new(CaptureProbe::default());
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for n in 0..FRAMES {
                feed.captured(
                    Some(segment),
                    &AudioFrame::new(n.to_le_bytes().to_vec(), 1_000, 1, 1),
                );
            }
        });
        feed.attach(RecorderTap::new(recorder.clone(), clock), &[]);
    });
    // Nothing played yet: the recording holds the whole segment, each frame once.
    assert_eq!(*recorder.0.lock(), (0..FRAMES).collect::<Vec<_>>());
}
