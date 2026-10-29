use super::*;
use lk_agents_abi::TokioSpawner;
use lk_agents_abi::voice::rtc::LocalParticipant as _;
use lk_agents_runtime::room_io::FixedRoom;
use lk_agents_voice::agent_session::Timebase;
use std::sync::atomic::AtomicUsize;
use tokio::sync::Notify;

struct DelayedClose {
    entered: Notify,
    release: Notify,
    finished: AtomicBool,
    calls: AtomicUsize,
    cancelled: AtomicUsize,
}

impl FrameStream for DelayedClose {
    fn next(&self) -> BoxFuture<'_, Result<Option<AudioFrame>, ProviderError>> {
        Box::pin(async { Ok(None) })
    }

    fn cancel_pending(&self) {}

    fn close(&self) -> BoxFuture<'_, ()> {
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
        lk_agents_runtime::rtc::events::RoomEventBus::port,
    );
    let room = rtc.room("background-retirement-cancellation");
    let player = Arc::new(FakeBackgroundAudioPlayer::new(Arc::new(TokioSpawner::current())));
    let stream = Arc::new(DelayedClose {
        entered: Notify::new(),
        release: Notify::new(),
        finished: AtomicBool::new(false),
        calls: AtomicUsize::new(0),
        cancelled: AtomicUsize::new(0),
    });
    let sound = Sound::for_host(None, Some(stream.clone()), 1.0, 1.0, 0.0, 0.0);
    player
        .start(
            Arc::new(FixedRoom {
                rtc: room.clone(),
                time: Timebase::from_virtual(clock),
                pool: None,
            }),
            vec![sound],
            0.5,
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
    // Both stop and stream completion are ready: stop wins the pump's biased select.
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
