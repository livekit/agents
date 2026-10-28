//! Deterministic boundary ordering: clear overtakes an audio result already yielded by the host.
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::utils::video::{AvatarFrame, MediaFrame};
use lk_agents_abi::voice::avatar::{AvatarAudioReceiver, AvatarMedia, AvatarVideoGenerator};
use lk_agents_abi::{AudioFrame, TokioSpawner, VirtualClock};
use lk_agents_rtc::FakeRtc;
use lk_agents_voice::avatar_runner::{AvatarIo, AvatarRunner, Event};
use std::sync::{
    Arc, OnceLock, Weak,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{Mutex, Notify, mpsc};

struct Receiver {
    audio: Mutex<mpsc::Receiver<(u64, bool, Option<AvatarFrame>)>>,
    runner: OnceLock<Weak<AvatarRunner>>,
    clear_once: AtomicBool,
    reports: mpsc::Sender<bool>,
    block_interruption: bool,
    report_started: mpsc::Sender<()>,
    allow_report: Arc<Notify>,
}
#[async_trait::async_trait]
impl AvatarAudioReceiver for Receiver {
    async fn open_audio(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    async fn next_audio(&self) -> Result<(u64, bool, Option<AvatarFrame>), ProviderError> {
        let frame = self.audio.lock().await.recv().await;
        if !self.clear_once.swap(true, Ordering::AcqRel) {
            self.runner.get().unwrap().upgrade().unwrap().clear_buffer(1);
        }
        Ok(frame.unwrap_or((1, false, None)))
    }
    async fn notify_playback_started(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    async fn notify_playback_finished(
        &self,
        _: f64,
        interrupted: bool,
    ) -> Result<(), ProviderError> {
        if interrupted && self.block_interruption {
            self.report_started.send(()).await.unwrap();
            self.allow_report.notified().await;
        }
        self.reports.send(interrupted).await.unwrap();
        Ok(())
    }
    async fn aclose(&self) -> Result<(), ProviderError> {
        Ok(())
    }
}
struct Generator {
    pushed: mpsc::Sender<MediaFrame>,
    frames_tx: mpsc::Sender<AvatarFrame>,
    frames: Mutex<mpsc::Receiver<AvatarFrame>>,
}
#[async_trait::async_trait]
impl AvatarVideoGenerator for Generator {
    async fn open_frames(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    async fn next_frame(&self) -> Result<Option<AvatarFrame>, ProviderError> {
        Ok(self.frames.lock().await.recv().await)
    }
    async fn push_audio(&self, frame: Option<MediaFrame>) -> Result<(), ProviderError> {
        if let Some(frame) = frame {
            self.pushed.send(frame).await.unwrap();
        } else {
            self.frames_tx.send(AvatarFrame::SegmentEnd).await.unwrap();
        }
        Ok(())
    }
    async fn clear_buffer(&self) -> Result<(), ProviderError> {
        Ok(())
    }
}
struct Media;
#[async_trait::async_trait]
impl AvatarMedia for Media {
    async fn publish(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    async fn push(&self, _: MediaFrame) -> Result<(), ProviderError> {
        Ok(())
    }
    async fn aclose(&self) -> Result<(), ProviderError> {
        Ok(())
    }
}
async fn settle() {
    for _ in 0..60 {
        tokio::task::yield_now().await;
    }
}
fn pcm(id: u8) -> AvatarFrame {
    AvatarFrame::Audio(AudioFrame::new(vec![id; 20], 1000, 1, 10))
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
    let room = FakeRtc::new(clock.clone()).room("clear_overtakes_audio");
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
    audio_tx.send((0, true, Some(pcm(1)))).await.unwrap();
    runner.start().await.unwrap();
    settle().await;
    assert!(overtook.load(Ordering::Acquire));
    assert!(pushed_rx.try_recv().is_err(), "cut audio reached the generator after clear");
    if block_interruption {
        report_started_rx.recv().await.unwrap();
        assert!(report_rx.try_recv().is_err());
    } else {
        assert_eq!(report_rx.try_recv(), Ok(true));
    }
    assert!(report_rx.try_recv().is_err());
    audio_tx.send((1, true, Some(pcm(2)))).await.unwrap();
    audio_tx.send((1, false, Some(AvatarFrame::SegmentEnd))).await.unwrap();
    if block_interruption {
        settle().await;
        assert!(pushed_rx.try_recv().is_err(), "new audio crossed the pending old acknowledgement");
        assert!(report_rx.try_recv().is_err());
        allow_report.notify_one();
    }
    settle().await;
    if block_interruption {
        assert_eq!(report_rx.try_recv(), Ok(true));
    }
    assert!(matches!(pushed_rx.try_recv(), Ok(MediaFrame::Audio(_))));
    assert_eq!(report_rx.try_recv(), Ok(false));
    assert!(report_rx.try_recv().is_err());
    runner.aclose().await.unwrap();
}
