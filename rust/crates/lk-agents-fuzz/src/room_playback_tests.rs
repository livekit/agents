//! The room audio output's playback notifications across a pause and its resume.

use super::*;
use lk_agents_abi::voice::io::{PlaybackEvent, PlaybackObserver};

/// Each playback event the output reported, in order.
#[derive(Default)]
struct Reported(Mutex<Vec<&'static str>>);

impl PlaybackObserver for Reported {
    fn on_playback_event(&self, event: &PlaybackEvent) {
        let label = match event {
            PlaybackEvent::Started(_) => "started",
            PlaybackEvent::Progressed(_) => return,
            PlaybackEvent::Finished { .. } => "finished",
        };
        self.0.lock().unwrap().push(label);
    }
}

/// A segment paused and resumed mid-playback reports its start once and its end once.
/// https://github.com/livekit/agents/pull/6636 https://github.com/livekit/agents/pull/6938
#[tokio::test]
async fn a_resumed_segment_starts_and_finishes_once() {
    let (room, clock) = fake_room("pause-resume");
    let output = audio_output(room, clock.clone());
    let reported = Arc::new(Reported::default());
    output.tracker().observe(reported.clone());
    output.start().await.unwrap();
    let frame = silence_frame(480, 24_000, 1);
    for _ in 0..10 {
        output.capture_frame(&frame).await.unwrap();
    }
    yields(8).await;
    clock.advance(Duration::from_millis(60));
    yields(8).await;
    output.pause();
    yields(8).await;
    clock.advance(Duration::from_millis(500));
    yields(8).await;
    output.resume();
    yields(8).await;
    for _ in 0..5 {
        output.capture_frame(&frame).await.unwrap();
    }
    output.flush();
    for _ in 0..20 {
        clock.advance(Duration::from_millis(50));
        yields(8).await;
    }
    let finished = within_a_second(output.tracker().wait_for_playout())
        .await
        .expect("the resumed segment plays out");
    assert!(!finished.interrupted);
    assert_eq!(*reported.0.lock().unwrap(), ["started", "finished"]);
    output.aclose().await;
}
