use super::*;
use lk_agents_abi::voice::rtc::PlaybackInfo;
use lk_agents_abi::voice::rtc::media::AVSynchronizer as _;
use lk_agents_abi::{AudioFrame, BoxFuture, VirtualClock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Default)]
struct Audio {
    closes: AtomicUsize,
    closing: Notify,
    release: Notify,
    hold_close: bool,
    fail_close: bool,
}
impl AudioSource for Audio {
    fn capture<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a AudioFrame,
    ) -> BoxFuture<'a, Outcome> {
        Box::pin(async { Ok(()) })
    }
    fn clear_queue(&self) -> Outcome {
        Ok(())
    }
    fn queued_duration(&self) -> Result<Duration, RtcError> {
        Ok(Duration::ZERO)
    }
    fn wait_for_playout(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> BoxFuture<'_, Result<PlaybackInfo, RtcError>> {
        Box::pin(async { Ok(PlaybackInfo::default()) })
    }
    fn playback_info(&self) -> Result<PlaybackInfo, RtcError> {
        Ok(PlaybackInfo::default())
    }
    fn close(&self, _cx: &lk_agents_abi::telemetry::Cx) -> BoxFuture<'_, Outcome> {
        Box::pin(async move {
            self.closes.fetch_add(1, Ordering::SeqCst);
            self.closing.notify_one();
            if self.hold_close {
                self.release.notified().await;
            }
            if self.fail_close { Err(RtcError::Closed("audio test failure")) } else { Ok(()) }
        })
    }
}

#[derive(Default)]
struct Video {
    captures: AtomicUsize,
    closes: AtomicUsize,
    fail: bool,
}
impl VideoSource for Video {
    fn capture(&self, _: &VideoFrame) -> Outcome {
        self.captures.fetch_add(1, Ordering::SeqCst);
        if self.fail { Err(RtcError::Closed("video test failure")) } else { Ok(()) }
    }
    fn close(&self, _cx: &lk_agents_abi::telemetry::Cx) -> BoxFuture<'_, Outcome> {
        Box::pin(async move {
            self.closes.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}

fn sync(audio: Arc<Audio>, video: Arc<Video>, clock: Arc<VirtualClock>) -> Arc<AVSynchronizer> {
    AVSynchronizer::new(
        audio,
        video,
        FramePacer::new(25.0).unwrap(),
        1,
        clock,
        Arc::new(lk_agents_net::task::TokioSpawner::current()),
    )
}
fn frame() -> MediaFrame {
    MediaFrame::Video(VideoFrame {
        width: 1,
        height: 1,
        buffer_type: 0,
        data: bytes::Bytes::from_static(&[0; 4]),
    })
}
async fn until(mut ready: impl FnMut() -> bool) {
    for _ in 0..1000 {
        if ready() {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("native media operation did not make progress");
}

#[tokio::test]
async fn queue_is_bounded_clear_discards_waiting_frames_and_releases_producers() {
    let (audio, video, clock) =
        (Arc::new(Audio::default()), Arc::new(Video::default()), Arc::new(VirtualClock::new()));
    let sync = sync(audio, video.clone(), clock.clone());
    sync.push(frame(), None).await.unwrap();
    until(|| video.captures.load(Ordering::SeqCst) == 1).await;
    sync.push(frame(), None).await.unwrap();
    until(|| sync.inner.state.lock().in_flight).await;
    sync.push(frame(), None).await.unwrap();
    let mut blocked = Box::pin(sync.push(frame(), None));
    assert!(
        std::future::poll_fn(|cx| std::task::Poll::Ready(blocked.as_mut().poll(cx).is_pending()))
            .await
    );
    sync.clear_queue().await.unwrap();
    blocked.await.unwrap();
    clock.advance(Duration::from_secs(1));
    sync.wait_for_playout().await.unwrap();
    assert_eq!(video.captures.load(Ordering::SeqCst), 1);
    sync.aclose().await.unwrap();
}

#[tokio::test]
async fn close_survives_cancelled_waiter_and_closes_both_sources_after_failure() {
    let audio = Arc::new(Audio { hold_close: true, fail_close: true, ..Default::default() });
    let video = Arc::new(Video::default());
    let sync = sync(audio.clone(), video.clone(), Arc::new(VirtualClock::new()));
    let mut closing = Box::pin(sync.aclose());
    assert!(
        std::future::poll_fn(|cx| std::task::Poll::Ready(closing.as_mut().poll(cx).is_pending()))
            .await
    );
    audio.closing.notified().await;
    drop(closing);
    audio.release.notify_one();
    assert!(sync.aclose().await.is_err());
    assert!(sync.aclose().await.is_err());
    assert_eq!(audio.closes.load(Ordering::SeqCst), 1);
    assert_eq!(video.closes.load(Ordering::SeqCst), 1);
    assert!(sync.push(frame(), None).await.is_err());
}

#[tokio::test]
async fn capture_failure_is_reported_to_push_and_playout_waiters() {
    let video = Arc::new(Video { fail: true, ..Default::default() });
    let sync = sync(Arc::new(Audio::default()), video.clone(), Arc::new(VirtualClock::new()));
    sync.push(frame(), Some(42.0)).await.unwrap();
    until(|| video.captures.load(Ordering::SeqCst) == 1).await;
    assert!(sync.wait_for_playout().await.is_err());
    assert!(sync.push(frame(), None).await.is_err());
    assert_eq!(sync.last_video_time(), 0.0);
    sync.aclose().await.unwrap();
}
