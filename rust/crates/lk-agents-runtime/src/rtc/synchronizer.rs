//! Executes frame pacing over shared RTC capabilities. No SDK-specific avatar host object.

use crate::telemetry;
use lk_agents_abi::utils::video::{MediaFrame, VideoFrame};
use lk_agents_abi::voice::rtc::{AudioSource, RtcError, VideoSource};
use lk_agents_abi::{AudioFrame, Clock, OwnedTask, Spawner, SpawnerExt, Ts};
use lk_agents_net::task::TaskSet;
use lk_agents_voice::av_sync::FramePacer;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{Notify, watch};

type Outcome<T = ()> = Result<T, RtcError>;

struct Queued {
    frame: VideoFrame,
    timestamp: Option<f64>,
    epoch: u64,
}

struct State {
    queue: VecDeque<Queued>,
    pacer: FramePacer,
    epoch: u64,
    in_flight: bool,
    closed: bool,
    failure: Option<RtcError>,
    last_video: f64,
    last_audio: f64,
}

struct Inner {
    audio: Arc<dyn AudioSource>,
    video: Arc<dyn VideoSource>,
    clock: Arc<dyn Clock>,
    capacity: usize,
    state: Mutex<State>,
    changed: Notify,
    closed: watch::Sender<Option<Outcome>>,
}

pub struct AVSynchronizer {
    inner: Arc<Inner>,
    worker: Mutex<Option<OwnedTask>>,
    spawner: Arc<dyn Spawner>,
    close_tasks: Arc<TaskSet>,
}

impl AVSynchronizer {
    pub fn new(
        audio: Arc<dyn AudioSource>,
        video: Arc<dyn VideoSource>,
        pacer: FramePacer,
        capacity: usize,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
    ) -> Arc<Self> {
        let inner = Arc::new(Inner {
            audio,
            video,
            clock,
            capacity,
            state: Mutex::new(State {
                queue: VecDeque::new(),
                pacer,
                epoch: 0,
                in_flight: false,
                closed: false,
                failure: None,
                last_video: 0.0,
                last_audio: 0.0,
            }),
            changed: Notify::new(),
            closed: watch::channel(None).0,
        });
        let run = inner.clone();
        let worker = spawner.spawn(async move { run.run().await });
        Arc::new(Self {
            inner,
            worker: Mutex::new(Some(worker)),
            spawner,
            close_tasks: Arc::default(),
        })
    }
}

impl Inner {
    fn check(state: &State) -> Outcome {
        if let Some(error) = &state.failure {
            return Err(error.clone());
        }
        if state.closed {
            return Err(RtcError::Closed("AVSynchronizer"));
        }
        Ok(())
    }

    async fn run(&self) {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let Some(next) = self.take_next() else { return };
            let Some((queued, deadline)) = next else {
                changed.await;
                continue;
            };
            self.changed.notify_waiters();
            self.present(&queued, deadline).await;
            if self.end_flight() {
                return;
            }
        }
    }

    /// Pops the next frame, in flight from here, with its pacing deadline; `None` once closed.
    fn take_next(&self) -> Option<Option<(Queued, Ts)>> {
        let mut state = self.state.lock();
        if state.closed {
            return None;
        }
        let next = state.queue.pop_front();
        state.in_flight = next.is_some();
        Some(next.map(|frame| (frame, state.pacer.deadline(self.clock.now()))))
    }

    /// Captures the frame at its deadline. A clear/reset wakes pacing; the epoch is re-checked
    /// before entering the SDK.
    async fn present(&self, queued: &Queued, deadline: Ts) {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let discard = {
                let s = self.state.lock();
                s.closed || s.epoch != queued.epoch
            };
            if discard {
                return;
            }
            tokio::select! {
                biased;
                _ = &mut changed => continue,
                _ = self.clock.wait_until(deadline) => {}
            }
            let result = self.video.capture(&queued.frame);
            self.record_capture(result, queued.timestamp);
            return;
        }
    }

    fn record_capture(&self, result: Outcome, timestamp: Option<f64>) {
        let mut state = self.state.lock();
        match result {
            Ok(()) => {
                state.pacer.captured(self.clock.now());
                if let Some(timestamp) = timestamp {
                    state.last_video = timestamp;
                }
            }
            Err(error) => state.failure = Some(error),
        }
    }

    /// Ends the frame's flight and wakes the waiters; whether a capture failed.
    fn end_flight(&self) -> bool {
        let failed = {
            let mut state = self.state.lock();
            state.in_flight = false;
            state.failure.is_some()
        };
        self.changed.notify_waiters();
        failed
    }
}

impl AVSynchronizer {
    /// The current epoch, unless the synchronizer failed or closed.
    fn current_epoch(&self) -> Outcome<u64> {
        let state = self.inner.state.lock();
        Inner::check(&state)?;
        Ok(state.epoch)
    }

    /// Captures the audio frame; its timestamp counts unless a clear came meanwhile.
    async fn push_audio(&self, frame: &AudioFrame, timestamp: Option<f64>) -> Outcome {
        let epoch = self.current_epoch()?;
        self.inner.audio.capture(&telemetry::Cx::NONE, frame).await?;
        let mut state = self.inner.state.lock();
        Inner::check(&state)?;
        if state.epoch == epoch
            && let Some(timestamp) = timestamp
        {
            state.last_audio = timestamp;
        }
        Ok(())
    }

    /// Queues the video frame for pacing once there is room; a clear meanwhile drops it.
    async fn push_video(&self, frame: VideoFrame, timestamp: Option<f64>) -> Outcome {
        let epoch = self.current_epoch()?;
        loop {
            let changed = self.inner.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            {
                let mut state = self.inner.state.lock();
                Inner::check(&state)?;
                if state.epoch != epoch {
                    return Ok(());
                }
                if state.queue.len() < self.inner.capacity {
                    state.queue.push_back(Queued { frame, timestamp, epoch });
                    drop(state);
                    self.inner.changed.notify_waiters();
                    return Ok(());
                }
            }
            changed.await;
        }
    }
}

impl lk_agents_abi::voice::rtc::media::AVSynchronizer for AVSynchronizer {
    async fn push(&self, frame: MediaFrame, timestamp: Option<f64>) -> Outcome {
        match frame {
            MediaFrame::Audio(frame) => self.push_audio(&frame, timestamp).await,
            MediaFrame::Video(frame) => self.push_video(frame, timestamp).await,
        }
    }

    async fn clear_queue(&self) -> Outcome {
        {
            let mut state = self.inner.state.lock();
            Inner::check(&state)?;
            state.epoch += 1;
            state.queue.clear();
        }
        self.inner.changed.notify_waiters();
        self.inner.audio.clear_queue()
    }

    async fn wait_for_playout(&self) -> Outcome {
        loop {
            let changed = self.inner.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            {
                let state = self.inner.state.lock();
                Inner::check(&state)?;
                if state.queue.is_empty() && !state.in_flight {
                    break;
                }
            }
            changed.await;
        }
        self.inner.audio.wait_for_playout(&telemetry::Cx::NONE).await.map(|_| ())
    }

    fn reset(&self) {
        self.inner.state.lock().pacer.reset();
    }
    fn actual_fps(&self) -> f64 {
        self.inner.state.lock().pacer.actual_fps()
    }
    fn last_video_time(&self) -> f64 {
        self.inner.state.lock().last_video
    }
    fn last_audio_time(&self) -> Result<f64, RtcError> {
        let last = self.inner.state.lock().last_audio;
        Ok(last - self.inner.audio.queued_duration()?.as_secs_f64())
    }

    async fn aclose(&self) -> Outcome {
        let mut closed = self.inner.closed.subscribe();
        let first = {
            let mut state = self.inner.state.lock();
            let first = !state.closed;
            state.closed = true;
            state.queue.clear();
            first
        };
        self.inner.changed.notify_waiters();
        if first {
            let worker = self.worker.lock().take();
            let inner = self.inner.clone();
            self.close_tasks.spawn(&*self.spawner, "RTC synchronizer close", async move {
                if let Some(worker) = worker {
                    worker.cancel();
                    let _ = worker.await;
                }
                let audio = inner.audio.close(&telemetry::Cx::NONE).await;
                let video = inner.video.close(&telemetry::Cx::NONE).await;
                inner.closed.send_replace(Some(audio.and(video)));
            });
        }
        loop {
            if let Some(result) = closed.borrow_and_update().clone() {
                return result;
            }
            if closed.changed().await.is_err() {
                return Err(RtcError::Closed("AVSynchronizer"));
            }
        }
    }
}

#[cfg(test)]
#[path = "synchronizer_tests.rs"]
mod tests;
