//! Deterministic boundary ordering: clear overtakes an audio result already yielded by the host.
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::video::{AudioReceiverFrame, AvatarFrame, MediaFrame};
use lk_agents_abi::voice::avatar::{AvatarAudioReceiver, AvatarMedia, AvatarVideoGenerator};
use lk_agents_abi::voice::io::{AvatarPlaybackSegment, PlaybackSegment};
use lk_agents_abi::{AudioFrame, TokioSpawner, VirtualClock};
use lk_agents_abi::voice::avatar::AvatarRunner as _;
use lk_agents_runtime::avatar::runner::{AvatarIo, AvatarRunner, Event};
use lk_agents_test_support::rtc::FakeRtc;
use std::sync::{
    Arc, OnceLock, Weak,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{Mutex, Notify, mpsc};

struct Receiver {
    audio: Mutex<
        mpsc::Receiver<(u64, bool, Option<AvatarPlaybackSegment>, Option<AudioReceiverFrame>)>,
    >,
    runner: OnceLock<Weak<AvatarRunner>>,
    clear_once: AtomicBool,
    reports: mpsc::Sender<(AvatarPlaybackSegment, bool)>,
    block_interruption: bool,
    report_started: mpsc::Sender<()>,
    allow_report: Arc<Notify>,
}
impl AvatarAudioReceiver for Receiver {
    fn open_audio(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move { Ok(()) })
    }
    fn next_audio(
        &self,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            (u64, bool, Option<AvatarPlaybackSegment>, Option<AudioReceiverFrame>),
            ProviderError,
        >,
    > {
        Box::pin(async move {
            let frame = self.audio.lock().await.recv().await;
            if !self.clear_once.swap(true, Ordering::AcqRel) {
                self.runner.get().unwrap().upgrade().unwrap().clear_buffer(1);
            }
            Ok(frame.unwrap_or((1, false, None, None)))
        })
    }
    fn notify_playback_started(
        &self,
        _: AvatarPlaybackSegment,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move { Ok(()) })
    }
    fn notify_playback_finished(
        &self,
        segment: AvatarPlaybackSegment,
        _: f64,
        interrupted: bool,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move {
            if interrupted && self.block_interruption {
                self.report_started.send(()).await.unwrap();
                self.allow_report.notified().await;
            }
            self.reports.send((segment, interrupted)).await.unwrap();
            Ok(())
        })
    }
    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move { Ok(()) })
    }
}
struct Generator {
    pushed: mpsc::Sender<MediaFrame>,
    frames_tx: mpsc::Sender<AvatarFrame>,
    frames: Mutex<mpsc::Receiver<AvatarFrame>>,
}
impl AvatarVideoGenerator for Generator {
    fn open_frames(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move { Ok(()) })
    }
    fn next_frame(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<AvatarFrame>, ProviderError>> {
        Box::pin(async move { Ok(self.frames.lock().await.recv().await) })
    }
    fn push_audio(
        &self,
        frame: Option<MediaFrame>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move {
            if let Some(frame) = frame {
                self.pushed.send(frame).await.unwrap();
            } else {
                self.frames_tx.send(AvatarFrame::SegmentEnd).await.unwrap();
            }
            Ok(())
        })
    }
    fn clear_buffer(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move { Ok(()) })
    }
}
struct Media;
impl AvatarMedia for Media {
    fn publish(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move { Ok(()) })
    }
    fn push(&self, _: MediaFrame) -> lk_agents_abi::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move { Ok(()) })
    }
    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move { Ok(()) })
    }
}
async fn settle() {
    for _ in 0..60 {
        tokio::task::yield_now().await;
    }
}
fn token(id: u64) -> AvatarPlaybackSegment {
    AvatarPlaybackSegment {
        playback: PlaybackSegment { attachment: 1, segment: id },
        stream_id: Some(format!("S{id}")),
    }
}

fn pcm(id: u8) -> AudioReceiverFrame {
    AudioReceiverFrame::Audio(AudioFrame::new(vec![id; 20], 1000, 1, 10))
}

#[tokio::test]
async fn a_clear_scheduled_ahead_of_already_yielded_audio_drops_it_and_settles_it_once() {
    scheduled_clear_race(false).await;
}

#[tokio::test]
async fn an_async_cut_report_is_delivered_before_the_next_segments_finish() {
    scheduled_clear_race(true).await;
}

async fn scheduled_clear_race(block_interruption: bool) {
    let (audio_tx, audio) = mpsc::channel(8);
    let (reports, mut report_rx) = mpsc::channel(8);
    let (report_started, mut report_started_rx) = mpsc::channel(1);
    let allow_report = Arc::new(Notify::new());
    let receiver = Arc::new(Receiver {
        audio: Mutex::new(audio),
        runner: OnceLock::new(),
        clear_once: AtomicBool::new(false),
        reports,
        block_interruption,
        report_started,
        allow_report: allow_report.clone(),
    });
    let (pushed, mut pushed_rx) = mpsc::channel(8);
    let (frames_tx, frames) = mpsc::channel(8);
    let generator = Arc::new(Generator { pushed, frames_tx, frames: Mutex::new(frames) });
    let clock = Arc::new(VirtualClock::new());
    let room = FakeRtc::new(clock.clone(), lk_agents_runtime::rtc::events::RoomEventBus::port)
        .room("clear_overtakes_audio");
    let overtook = Arc::new(AtomicBool::new(false));
    let saw_race = Arc::clone(&overtook);
    let runner = Arc::new(AvatarRunner::scheduled(
        room,
        AvatarIo { receiver: receiver.clone(), generator, media: Arc::new(Media) },
        true,
        clock,
        Arc::new(TokioSpawner::current()),
        move |ready| {
            let clear = ready.iter().position(|e| matches!(e, Event::ClearBuffer { .. }));
            if let Some(clear) = clear {
                if ready.iter().any(|e| matches!(e, Event::Audio(_))) {
                    saw_race.store(true, Ordering::Release);
                }
                return clear;
            }
            0
        },
    ));
    receiver.runner.set(Arc::downgrade(&runner)).unwrap();
    audio_tx.send((0, true, Some(token(1)), Some(pcm(1)))).await.unwrap();
    runner.start().await.unwrap();
    settle().await;
    assert!(overtook.load(Ordering::Acquire));
    assert!(pushed_rx.try_recv().is_err(), "cut audio reached the generator after clear");
    if block_interruption {
        report_started_rx.recv().await.unwrap();
        assert!(report_rx.try_recv().is_err());
    } else {
        assert_eq!(report_rx.try_recv(), Ok((token(1), true)));
    }
    assert!(report_rx.try_recv().is_err());
    audio_tx.send((1, true, Some(token(2)), Some(pcm(2)))).await.unwrap();
    audio_tx.send((1, false, Some(token(2)), Some(AudioReceiverFrame::SegmentEnd))).await.unwrap();
    if block_interruption {
        settle().await;
        assert!(pushed_rx.try_recv().is_err(), "new audio crossed the pending old acknowledgement");
        assert!(report_rx.try_recv().is_err());
        allow_report.notify_one();
    }
    settle().await;
    if block_interruption {
        assert_eq!(report_rx.try_recv(), Ok((token(1), true)));
    }
    assert!(matches!(pushed_rx.try_recv(), Ok(MediaFrame::Audio(_))));
    assert_eq!(report_rx.try_recv(), Ok((token(2), false)));
    assert!(report_rx.try_recv().is_err());
    runner.aclose().await.unwrap();
}
