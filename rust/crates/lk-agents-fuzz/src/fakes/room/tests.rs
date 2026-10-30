//! Where the fake must play out as the LiveKit SDK's audio source does.
use std::time::Duration;

use lk_agents_abi::AudioFrame;
use lk_agents_abi::voice::rtc::{AudioSource as _, RtcRoom as _};
use lk_agents_abi::voice::rtc::{PlaybackInfo, TrackPublishOptions};
use tokio::time::timeout;

use super::FakeRtc;

/// Paused time auto-advances to it once every task is parked: expiring means nothing can make
/// progress.
const NEVER: Duration = Duration::from_secs(3600);

/// 20 ms of mono 24 kHz, tagged in its first byte.
fn frame(tag: u8) -> AudioFrame {
    let mut pcm = vec![0u8; 480 * 2];
    pcm[0] = tag;
    AudioFrame::new(pcm, 24_000, 1, 480)
}

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

fn tags(frames: &[AudioFrame]) -> Vec<u8> {
    frames.iter().map(|f| f.data[0]).collect()
}

async fn published() -> (std::sync::Arc<FakeRtc>, std::sync::Arc<super::FakeAudioSource>) {
    let (rtc, _) = FakeRtc::with_virtual_clock(lk_agents_runtime::rtc::events::RoomEventBus::port);
    let room = rtc.room("session");
    room.local().publish_audio_track(TrackPublishOptions::default()).await.unwrap();
    let source = room.local_participant().sources()[0].clone();
    (rtc, source)
}

// Catches: a `clear_queue` that resets a counter without dropping queued audio, as the SDK's
// does: every interruption would then pass while the agent talked over the person.
#[tokio::test(start_paused = true)]
async fn clear_queue_discards_the_audio_that_had_not_played() {
    let (rtc, source) = published().await;
    for tag in 1..=5 {
        timeout(NEVER, source.capture(&frame(tag))).await.unwrap().unwrap();
    }
    rtc.advance(ms(30));
    source.clear_queue().unwrap();
    let info = timeout(NEVER, source.wait_for_playout()).await.unwrap().unwrap();
    assert_eq!((info.played, info.discarded), (ms(30), ms(70)));
    assert_eq!(tags(&source.played()), vec![1], "only whole frames reach the speaker");
    assert_eq!(tags(&source.discarded()), vec![2, 3, 4, 5]);
}

// Catches: a `wait_for_playout` that resolves before the clock played the queue, or counters
// that span waits; the SDK's waits for the audio and zeroes them per wait.
#[tokio::test(start_paused = true)]
async fn wait_for_playout_follows_the_clock() {
    let (rtc, source) = published().await;
    timeout(NEVER, source.capture(&frame(1))).await.unwrap().unwrap();
    timeout(NEVER, source.capture(&frame(2))).await.unwrap().unwrap();
    rtc.advance(ms(20));
    assert!(timeout(NEVER, source.wait_for_playout()).await.is_err(), "returned with 20 ms left");
    rtc.advance(ms(20));
    let info = timeout(NEVER, source.wait_for_playout()).await.unwrap().unwrap();
    assert_eq!(info, PlaybackInfo { played: ms(40), discarded: Duration::ZERO });
    timeout(NEVER, source.capture(&frame(3))).await.unwrap().unwrap();
    rtc.advance(ms(20));
    assert_eq!(timeout(NEVER, source.wait_for_playout()).await.unwrap().unwrap().played, ms(20));
}
