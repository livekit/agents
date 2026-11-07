use super::*;
use lk_agents_abi::SpawnerExt;
use tokio::sync::Notify;

#[tokio::test]
async fn a_stop_waits_out_the_fade() {
    let fading = PlaybackDone::new(0.5);
    assert!(!fading.stop());
    assert!(fading.stop_requested() && !fading.done());
    assert!(fading.finish());
    assert!(!fading.finish());
    fading.wait().await;
    assert!(!fading.stop());

    let abrupt = PlaybackDone::new(0.0);
    assert!(abrupt.stop());
    abrupt.wait().await;
}

#[test]
fn every_builtin_clip_is_shipped() {
    let resources = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../livekit-agents-next/livekit/agents/resources");
    use BuiltinAudioClip as C;
    for clip in [
        C::CityAmbience,
        C::ForestAmbience,
        C::OfficeAmbience,
        C::CrowdedRoom,
        C::KeyboardTyping,
        C::KeyboardTyping2,
        C::HoldMusic,
    ] {
        assert!(
            resources.join(clip.file_name()).is_file(),
            "{clip:?}: {} is not shipped",
            clip.file_name()
        );
    }
}

struct Frames {
    frames: tokio::sync::Mutex<AudioReceiver<()>>,
    reading: Notify,
    stopped: CancellationToken,
    closed: AtomicBool,
}

impl Frames {
    fn new() -> (Arc<Self>, AudioSender<()>) {
        let (send, frames) = media_queue::audio(
            Policy::CatchUp(Duration::from_secs(1)),
            telemetry::AudioConsumer::BackgroundAudio,
            Arc::new(lk_agents_abi::VirtualClock::new()),
        );
        (
            Arc::new(Self {
                frames: tokio::sync::Mutex::new(frames),
                reading: Notify::new(),
                stopped: CancellationToken::new(),
                closed: AtomicBool::new(false),
            }),
            send,
        )
    }
}

impl FrameStream for Frames {
    fn next(
        &self,
        _: &telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<AudioFrame>, ProviderError>> {
        Box::pin(async move {
            self.reading.notify_one();
            let mut frames = self.frames.lock().await;
            Ok(tokio::select! {
                biased;
                _ = self.stopped.cancelled() => None,
                frame = frames.recv() => match frame {
                    Some(Item::Media(frame)) => Some(frame),
                    _ => None,
                },
            })
        })
    }

    fn cancel_pending(&self) {
        self.stopped.cancel();
    }

    fn close(&self, _: &telemetry::Cx) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async {
            self.closed.store(true, Ordering::Release);
        })
    }
}

fn player() -> (Arc<BackgroundAudioPlayer>, Arc<lk_agents_abi::VirtualClock>) {
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let player = Arc::new(BackgroundAudioPlayer::new(
        48_000,
        1,
        2,
        1000,
        clock.clone(),
        Arc::new(lk_agents_net::task::TokioSpawner::current()),
    ));
    player.started.store(true, Ordering::Release);
    (player, clock)
}

fn play(player: &BackgroundAudioPlayer, stream: Arc<Frames>) -> Arc<PlaybackDone> {
    player
        .play(
            vec![Sound {
                source: SoundSource::Stream(stream),
                volume: 1.0,
                probability: 1.0,
                fade_in: 0.0,
                fade_out: 0.0,
            }],
            false,
            false,
            false,
            0.0,
        )
        .unwrap()
}

fn frame(samples: &[i16]) -> AudioFrame {
    AudioFrame::new(
        samples.iter().flat_map(|sample| sample.to_le_bytes()).collect::<Vec<u8>>(),
        48_000,
        1,
        samples.len() as u32,
    )
}

// A blocked host read must neither hold the mixer lock nor delay stopping the sounds.
#[tokio::test]
async fn a_pending_source_can_be_stopped_without_advancing_the_clock() {
    let (player, _) = player();
    let (stream, _send) = Frames::new();
    let handle = play(&player, stream.clone());
    let engine = player.clone();
    let next = player.spawner.spawn(async move { engine.next_block().await });
    stream.reading.notified().await;
    assert!(player.state.try_lock().is_some());
    tokio::time::timeout(Duration::from_secs(1), player.stop_sounds()).await.unwrap();
    assert!(handle.done());
    assert!(stream.closed.load(Ordering::Acquire));
    next.cancel();
    let _ = next.await;
}

// Queue readiness wakes the mixer, and cancelled consumers leave pending reads owned.
#[tokio::test]
async fn sources_mix_after_a_cancelled_wait_without_losing_frames() {
    let (player, _) = player();
    let (first, send_first) = Frames::new();
    play(&player, first.clone());
    let engine = player.clone();
    let next = player.spawner.spawn(async move { engine.next_block().await });
    first.reading.notified().await;
    next.cancel();
    let _ = next.await;

    let (second, send_second) = Frames::new();
    play(&player, second);
    send_first.push(frame(&[10, 20, 30, 40]));
    send_second.push(frame(&[1, 2, 3, 4]));
    assert_eq!(player.next_block().await, Some(vec![11, 22]));
    assert_eq!(player.next_block().await, Some(vec![33, 44]));
    player.stop_sounds().await;
}

#[tokio::test]
async fn a_short_final_frame_is_padded_and_closed_before_returning() {
    let (player, _) = player();
    let (stream, send) = Frames::new();
    let handle = play(&player, stream.clone());
    send.push(frame(&[9]));
    drop(send);
    assert_eq!(player.next_block().await, Some(vec![9, 0]));
    assert!(handle.done());
    assert!(stream.closed.load(Ordering::Acquire));
}

#[tokio::test]
async fn a_timeout_keeps_the_pending_read_for_the_next_round() {
    let (player, clock) = player();
    let (slow, send_slow) = Frames::new();
    let (fast, send_fast) = Frames::new();
    play(&player, slow.clone());
    play(&player, fast);
    send_fast.push(frame(&[1, 2]));
    let engine = player.clone();
    let next = player.spawner.spawn(async move { engine.next_block().await });
    slow.reading.notified().await;
    clock.advance(Duration::from_secs(1));
    assert_eq!(next.await.unwrap(), Some(vec![1, 2]));
    assert_eq!(player.state.lock().mixer.counters.stream_timeouts, 1);
    send_slow.push(frame(&[10, 20]));
    send_fast.push(frame(&[3, 4]));
    assert_eq!(player.next_block().await, Some(vec![13, 24]));
    player.stop_sounds().await;
}
