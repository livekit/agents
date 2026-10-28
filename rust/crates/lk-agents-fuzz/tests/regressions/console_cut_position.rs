//! A clear cutting the console's segment reports the audio played on the session's clock.
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::voice::console::ConsolePort;
use lk_agents_abi::voice::io::AudioOutputExt;
use lk_agents_abi::{AudioFrame, Clock, SimClock, TokioSpawner};
use lk_agents_fuzz::fakes::FakeCli;

// Catches: the console's playback accounting reading the host machine's clock inside the
// simulation: a cut then reported how long the machine took to get there, not how much audio the
// simulated CLI had played, so the committed reply did not match what the caller heard.
#[tokio::test(start_paused = true)]
async fn a_cut_console_segment_reports_the_audio_played() {
    let clock: Arc<dyn Clock> = Arc::new(SimClock::new());
    let (console, _cli) = FakeCli::connect(Arc::new(TokioSpawner::current()), clock);
    let output = console.as_audio_output();
    let second = AudioFrame::new(vec![0; 96_000], 48_000, 1, 48_000);
    output.capture_frame(&second).await.unwrap();
    tokio::time::advance(Duration::from_millis(300)).await;
    output.cut();
    let cut = output.wait_for_playout().await;
    assert!(cut.interrupted);
    // Wall seconds since the epoch carry about a microsecond of float error.
    let played = cut.playback_position.as_secs_f64();
    assert!((played - 0.3).abs() < 1e-3, "played {played}s of the 1s segment, not 0.3s");
}
