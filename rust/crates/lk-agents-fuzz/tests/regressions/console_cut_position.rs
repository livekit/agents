//! Elapsed time cannot establish that the console played queued audio.
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::voice::console::ConsolePort;
use lk_agents_abi::voice::io::AudioOutputExt;
use lk_agents_abi::{AudioFrame, Clock, SimClock};
use lk_agents_fuzz::fakes::FakeCli;
use lk_agents_runtime::net::task::TokioSpawner;

// No transport has delivered this PCM to the CLI. A clock advance alone must not commit it as
// heard; a late unkeyed acknowledgement must not borrow the next operation's identity either.
#[tokio::test(start_paused = true)]
async fn a_cut_console_segment_reports_only_confirmed_audio() {
    let clock: Arc<dyn Clock> = Arc::new(SimClock::new());
    let (console, _cli) = FakeCli::connect(Arc::new(TokioSpawner::current()), clock);
    let output = console.clone().as_audio_output();
    let second = AudioFrame::new(vec![0; 96_000], 48_000, 1, 48_000);
    output.capture_frame(&second).await.unwrap();
    let first = output.tracker().current_segment().unwrap();
    tokio::time::advance(Duration::from_millis(300)).await;
    output.cut();
    let cut = output.wait_for_playout().await;
    assert!(cut.interrupted);
    assert_eq!(cut.playback_position, Duration::ZERO);
    assert!(!output.tracker().has_pending_segment(first));

    output.capture_frame(&second).await.unwrap();
    let next = output.tracker().current_segment().unwrap();
    assert_ne!(first, next);
    console.as_console().playout_finished();
    assert!(output.tracker().has_pending_segment(next));
    output.cut();
    let cut = output.wait_for_playout().await;
    assert!(cut.interrupted);
    assert_eq!(cut.playback_position, Duration::ZERO);
}
