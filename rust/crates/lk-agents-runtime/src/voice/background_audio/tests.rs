use super::*;
use crate::room_io::FixedRoom;

fn file(path: &str, probability: f64) -> Sound {
    Sound {
        source: SoundSource::File(path.to_owned()),
        volume: 1.0,
        probability,
        fade_in: 0.0,
        fade_out: 0.0,
    }
}

/// A host stream of `frames` short frames.
struct Frames(Mutex<u32>);

impl FrameStream for Frames {
    fn next(&self) -> lk_agents_abi::BoxFuture<'_, Result<Option<AudioFrame>, ProviderError>> {
        let mut left = self.0.lock().unwrap();
        let frame = (*left > 0).then(|| {
            *left -= 1;
            AudioFrame::new(1000i16.to_le_bytes().repeat(480), 48_000, 1, 480)
        });
        Box::pin(std::future::ready(Ok(frame)))
    }
    fn cancel_pending(&self) {}
    fn close(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(std::future::ready(()))
    }
}

async fn started() -> (Arc<BackgroundAudioPlayer>, Arc<lk_agents_test_support::rtc::FakeRoom>) {
    let (rtc, clock) = lk_agents_test_support::rtc::FakeRtc::with_virtual_clock(
        crate::rtc::events::RoomEventBus::port,
    );
    let room = rtc.room("background");
    let fixed = FixedRoom {
        rtc: room.clone(),
        time: lk_agents_voice::agent_session::Timebase::from_virtual(clock),
        pool: None,
    };
    let player = Arc::new(BackgroundAudioPlayer::for_host(
        48_000,
        1,
        4_800,
        200,
        fixed.time.clock(),
        Arc::new(lk_agents_net::task::TokioSpawner::current()),
        crate::process::env().media_pool,
    ));
    player.start_for_host(Arc::new(fixed), 0.5, Vec::new(), true, None).await.unwrap();
    (player, room)
}

// Catches: a sound played before `start()` or after `aclose()` queued where nothing mixes it, and
// a looping iterator accepted (it can only play once).
#[tokio::test(start_paused = true)]
async fn play_refuses_what_cannot_play() {
    let idle = BackgroundAudioPlayer::for_host(
        48_000,
        1,
        4_800,
        200,
        Arc::new(lk_agents_abi::SimClock::new()),
        Arc::new(lk_agents_net::task::TokioSpawner::current()),
        crate::process::env().media_pool,
    );
    assert!(matches!(
        idle.play_for_host(vec![file("a.ogg", 1.0)], 0.5, false, false, true),
        Err(PlaybackError::NotStarted)
    ));

    let (player, _room) = started().await;
    let stream =
        Sound { source: SoundSource::Stream(Arc::new(Frames(Mutex::new(1)))), ..file("", 1.0) };
    let looping = player.play_for_host(vec![stream], 0.5, true, false, true);
    assert!(matches!(looping, Err(PlaybackError::LoopingStream(_))));
    assert!(LoopingStreamError.to_string().contains("loop=False"));
    // The draw picked silence: the handle is born done.
    let silent = player.play_for_host(vec![file("a.ogg", 0.0)], 0.5, false, false, true).unwrap();
    assert!(silent.done());

    player.close().await;
    assert!(player.closed());
    assert!(matches!(
        player.play(vec![file("a.ogg", 1.0)], false, false, false, 0.5),
        Err(PlaybackError::Closed)
    ));
}

// Catches: a stop with a fade-out reporting the sound done at once (the generator would be closed
// mid-fade), or a stop on a finished sound reopening it.
#[tokio::test]
async fn a_stop_waits_out_the_fade() {
    let fading = PlaybackDone::for_host(0.5);
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

// Catches: a builtin clip naming a file the package does not ship, which fails only when a call
// first plays it.
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

struct DelayedClose {
    entered: tokio::sync::Notify,
    release: tokio::sync::Notify,
    finished: AtomicBool,
    calls: std::sync::atomic::AtomicUsize,
    cancelled: std::sync::atomic::AtomicUsize,
}

impl FrameStream for DelayedClose {
    fn next(&self) -> lk_agents_abi::BoxFuture<'_, Result<Option<AudioFrame>, ProviderError>> {
        Box::pin(async { Ok(None) })
    }

    fn cancel_pending(&self) {}

    fn close(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move {
            struct Closing<'a>(&'a DelayedClose, bool);
            impl Drop for Closing<'_> {
                fn drop(&mut self) {
                    if !self.1 {
                        self.0.cancelled.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut closing = Closing(self, false);
            self.entered.notify_one();
            self.release.notified().await;
            closing.1 = true;
            self.finished.store(true, Ordering::SeqCst);
        })
    }
}

#[tokio::test(start_paused = true)]
async fn stopping_during_stream_retirement_keeps_cleanup_owned_after_a_waiter_is_cancelled() {
    let (rtc, clock) = lk_agents_test_support::rtc::FakeRtc::with_virtual_clock(
        crate::rtc::events::RoomEventBus::port,
    );
    let room = rtc.room("background-retirement-cancellation");
    let player = Arc::new(BackgroundAudioPlayer::for_host(
        48_000,
        1,
        4_800,
        200,
        lk_agents_voice::agent_session::Timebase::from_virtual(clock.clone()).clock(),
        Arc::new(lk_agents_net::task::TokioSpawner::current()),
        crate::process::env().media_pool,
    ));
    let stream = Arc::new(DelayedClose {
        entered: tokio::sync::Notify::new(),
        release: tokio::sync::Notify::new(),
        finished: AtomicBool::new(false),
        calls: std::sync::atomic::AtomicUsize::new(0),
        cancelled: std::sync::atomic::AtomicUsize::new(0),
    });
    let sound = Sound { source: SoundSource::Stream(stream.clone()), ..file("", 1.0) };
    player
        .start_for_host(
            Arc::new(FixedRoom {
                rtc: room.clone(),
                time: lk_agents_voice::agent_session::Timebase::from_virtual(clock),
                pool: None,
            }),
            0.5,
            vec![sound],
            true,
            None,
        )
        .await
        .unwrap();
    stream.entered.notified().await;
    let mut closing = Box::pin(player.close());
    std::future::poll_fn(|cx| {
        assert!(closing.as_mut().poll(cx).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    drop(closing);
    // Retirement stays owned after the close waiter is cancelled.
    stream.release.notify_one();
    tokio::task::yield_now().await;
    player.close().await;
    let finished = stream.finished.load(Ordering::SeqCst);
    let cancelled = stream.cancelled.load(Ordering::SeqCst);
    let publications = room.local_participant().track_publications().unwrap();
    // Leave no published track behind if the regression assertion fails.
    for publication in &publications {
        room.local_participant().unpublish_track(publication.sid()).await.unwrap();
    }
    assert!(finished, "stopping the pump abandoned an in-flight stream close");
    assert_eq!(cancelled, 0, "retirement cleanup was cancelled by the pump's stop");
    assert_eq!(stream.calls.load(Ordering::SeqCst), 1, "cleanup was restarted");
    assert!(publications.is_empty(), "cancelling a close waiter orphaned the output track");
}
