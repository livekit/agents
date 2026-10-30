//! `BackgroundAudioPlayer` over FakeRoom, in what the DST's player (an ambient stream and the
//! thinking sound) does not reach: refused plays and a close cancelled while a stream retires.
use lk_agents_abi::AudioFrame;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::background_audio::BackgroundAudioPlayer as _;
use lk_agents_runtime::net::task::TokioSpawner;
use lk_agents_runtime::room_io::FixedRoom;
use lk_agents_runtime::rtc::events::RoomEventBus;
use lk_agents_runtime::voice::background_audio::{
    BackgroundAudio, BackgroundAudioPlayer, FrameStream, LoopingStreamError, PlaybackError, Sound,
    SoundSource,
};
use lk_agents_runtime::voice::session::Timebase;
use lk_agents_test_support::rtc::{FakeRoom, FakeRtc, LocalParticipant as _};
use std::future::Future as _;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

fn file(path: &str, probability: f64) -> Sound {
    Sound {
        source: SoundSource::File(path.to_owned()),
        volume: 1.0,
        probability,
        fade_in: 0.0,
        fade_out: 0.0,
    }
}

fn player(time: &Timebase) -> BackgroundAudio {
    BackgroundAudio(Arc::new(BackgroundAudioPlayer::new(
        48_000,
        1,
        4_800,
        200,
        time.clock(),
        Arc::new(TokioSpawner::current()),
        lk_agents_runtime::process::env().media_pool,
    )))
}

/// A player started on a fake room with no ambient sound, and the room.
async fn started(name: &str, ambient: Vec<Sound>) -> (BackgroundAudio, Arc<FakeRoom>) {
    let (rtc, clock) = FakeRtc::with_virtual_clock(RoomEventBus::port);
    let room = rtc.room(name);
    std::mem::forget(rtc);
    let fixed = FixedRoom { rtc: Some(room.clone()), time: Timebase::from_virtual(clock), pool: None };
    let player = player(&fixed.time);
    player.start(Arc::new(fixed), 0.5, ambient, true, None).await.unwrap();
    (player, room)
}

/// A host stream of short frames.
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

// Catches: a sound played before `start()` or after `aclose()` queued where nothing mixes it, and
// a looping iterator accepted (it can only play once).
#[tokio::test(start_paused = true)]
async fn play_refuses_what_cannot_play() {
    let idle = player(&Timebase::from_virtual(Arc::new(lk_agents_abi::VirtualClock::new())));
    assert!(matches!(
        idle.0.play(vec![file("a.ogg", 1.0)], true, false, false, 0.5),
        Err(PlaybackError::NotStarted)
    ));

    let (player, _room) = started("background", Vec::new()).await;
    let stream =
        Sound { source: SoundSource::Stream(Arc::new(Frames(Mutex::new(1)))), ..file("", 1.0) };
    let looping = player.0.play(vec![stream], true, true, false, 0.5);
    assert!(matches!(looping, Err(PlaybackError::LoopingStream(_))));
    assert!(LoopingStreamError.to_string().contains("loop=False"));
    // The draw picked silence: the handle is born done.
    let silent = player.0.play(vec![file("a.ogg", 0.0)], true, false, false, 0.5).unwrap();
    assert!(silent.done());

    player.0.close().await;
    assert!(player.0.closed());
    assert!(matches!(
        player.0.play(vec![file("a.ogg", 1.0)], false, false, false, 0.5),
        Err(PlaybackError::Closed)
    ));
}

/// A stream whose close waits for the test, counting entries and cancellations.
#[derive(Default)]
struct DelayedClose {
    entered: tokio::sync::Notify,
    release: tokio::sync::Notify,
    finished: AtomicBool,
    calls: AtomicUsize,
    cancelled: AtomicUsize,
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
    let stream = Arc::new(DelayedClose::default());
    let sound = Sound { source: SoundSource::Stream(stream.clone()), ..file("", 1.0) };
    let (player, room) = started("background-retirement-cancellation", vec![sound]).await;
    stream.entered.notified().await;
    let mut closing = Box::pin(player.0.close());
    std::future::poll_fn(|cx| {
        assert!(closing.as_mut().poll(cx).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    drop(closing);
    // Retirement stays owned after the close waiter is cancelled.
    stream.release.notify_one();
    tokio::task::yield_now().await;
    player.0.close().await;
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
