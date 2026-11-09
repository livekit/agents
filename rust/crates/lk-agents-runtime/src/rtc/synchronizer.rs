use crate::telemetry;
use lk_agents_abi::utils::video::{MediaFrame, VideoFrame};
use lk_agents_abi::voice::rtc::{AudioSource, RtcError, VideoSource};
use lk_agents_abi::{AudioFrame, Clock, OwnedTask, Spawner, SpawnerExt, Ts};
use lk_agents_net::signal;
use lk_agents_net::task::TaskSet;
use lk_agents_voice::av_sync::FramePacer;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;

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
    /// Its terminal value is `closed`, which every wait accepts.
    state: signal::Shared<State>,
    /// The close task's: it ends with that task.
    closed: signal::Cell<Option<Outcome>>,
}

pub struct AVSynchronizer {
    inner: Arc<Inner>,
    close: Mutex<Option<signal::Producer<Option<Outcome>>>>,
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
        let (close, closed) = signal::once();
        let inner = Arc::new(Inner {
            audio,
            video,
            clock,
            capacity,
            state: signal::Shared::new(State {
                queue: VecDeque::new(),
                pacer,
                epoch: 0,
                in_flight: false,
                closed: false,
                failure: None,
                last_video: 0.0,
                last_audio: 0.0,
            }),
            closed,
        });
        let run = inner.clone();
        let worker = spawner.spawn(async move { run.run().await });
        Arc::new(Self {
            inner,
            close: Mutex::new(Some(close)),
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
            let next = self.state.wait_until(|state| self.take_next(state)).await;
            let Some((queued, deadline)) = next else { return };
            // Ends the flight however the presentation ends: done, cancelled or panicked.
            let flight = Flight(self);
            self.present(&queued, deadline).await;
            drop(flight);
            if self.state.with(|state| state.failure.is_some()) {
                return;
            }
        }
    }

    /// The next frame, in flight from here, with its pacing deadline; `Some(None)` once closed.
    fn take_next(&self, state: &mut State) -> Option<Option<(Queued, Ts)>> {
        if state.closed {
            return Some(None);
        }
        let next = state.queue.pop_front()?;
        state.in_flight = true;
        Some(Some((next, state.pacer.deadline(self.clock.now()))))
    }

    /// Captures the frame at its deadline. A clear/reset wakes pacing; the epoch is re-checked
    /// before entering the SDK.
    async fn present(&self, queued: &Queued, deadline: Ts) {
        let discarded =
            self.state.wait_until(|s| (s.closed || s.epoch != queued.epoch).then_some(()));
        tokio::select! {
            biased;
            () = discarded => return,
            _ = self.clock.wait_until(deadline) => {}
        }
        let result = self.video.capture(&queued.frame);
        self.record_capture(result, queued.timestamp);
    }

    fn record_capture(&self, result: Outcome, timestamp: Option<f64>) {
        self.state.with(|state| match result {
            Ok(()) => {
                state.pacer.captured(self.clock.now());
                if let Some(timestamp) = timestamp {
                    state.last_video = timestamp;
                }
            }
            Err(error) => state.failure = Some(error),
        });
    }
}

/// A frame in flight; dropping it lands the flight and wakes the waiters.
struct Flight<'a>(&'a Inner);

impl Drop for Flight<'_> {
    fn drop(&mut self) {
        self.0.state.modify(|state| state.in_flight = false);
    }
}

impl AVSynchronizer {
    /// The current epoch, unless the synchronizer failed or closed.
    fn current_epoch(&self) -> Outcome<u64> {
        self.inner.state.with(|state| {
            Inner::check(state)?;
            Ok(state.epoch)
        })
    }

    /// Captures the audio frame; its timestamp counts unless a clear came meanwhile.
    async fn push_audio(&self, frame: &AudioFrame, timestamp: Option<f64>) -> Outcome {
        let epoch = self.current_epoch()?;
        self.inner.audio.capture(&telemetry::Cx::NONE, frame).await?;
        self.inner.state.with(|state| {
            Inner::check(state)?;
            if state.epoch == epoch
                && let Some(timestamp) = timestamp
            {
                state.last_audio = timestamp;
            }
            Ok(())
        })
    }

    /// Queues the video frame for pacing once there is room; a clear meanwhile drops it.
    async fn push_video(&self, frame: VideoFrame, timestamp: Option<f64>) -> Outcome {
        let epoch = self.current_epoch()?;
        let mut frame = Some(frame);
        let capacity = self.inner.capacity;
        self.inner
            .state
            .wait_until(|state| {
                if let Err(error) = Inner::check(state) {
                    return Some(Err(error));
                }
                if state.epoch != epoch {
                    return Some(Ok(()));
                }
                if state.queue.len() < capacity {
                    let frame = frame.take().expect("a frame is queued once");
                    state.queue.push_back(Queued { frame, timestamp, epoch });
                    return Some(Ok(()));
                }
                None
            })
            .await
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
        self.inner.state.modify(|state| {
            Inner::check(state)?;
            state.epoch += 1;
            state.queue.clear();
            Outcome::Ok(())
        })?;
        self.inner.audio.clear_queue()
    }

    async fn wait_for_playout(&self) -> Outcome {
        self.inner
            .state
            .wait_until(|state| match Inner::check(state) {
                Err(error) => Some(Err(error)),
                Ok(()) => (state.queue.is_empty() && !state.in_flight).then_some(Ok(())),
            })
            .await?;
        self.inner.audio.wait_for_playout(&telemetry::Cx::NONE).await.map(|_| ())
    }

    fn reset(&self) {
        self.inner.state.with(|state| state.pacer.reset());
    }
    fn actual_fps(&self) -> f64 {
        self.inner.state.with(|state| state.pacer.actual_fps())
    }
    fn last_video_time(&self) -> f64 {
        self.inner.state.with(|state| state.last_video)
    }
    fn last_audio_time(&self) -> Result<f64, RtcError> {
        let last = self.inner.state.with(|state| state.last_audio);
        Ok(last - self.inner.audio.queued_seconds()?)
    }

    /// The first call closes, bounded by the host teardown timeout; every call waits for it.
    async fn aclose(&self) -> Outcome {
        self.inner.state.modify(|state| {
            state.closed = true;
            state.queue.clear();
        });
        let close = self.close.lock().take();
        if let Some(closed) = close {
            let worker = self.worker.lock().take();
            let inner = self.inner.clone();
            self.close_tasks.spawn(&*self.spawner, "RTC synchronizer close", async move {
                let close = async {
                    if let Some(worker) = worker {
                        worker.cancel();
                        let _ = worker.await;
                    }
                    let audio = inner.audio.close(&telemetry::Cx::NONE).await;
                    let video = inner.video.close(&telemetry::Cx::NONE).await;
                    audio.and(video)
                };
                let bound = lk_agents_abi::HOST_TEARDOWN_TIMEOUT;
                let outcome = lk_agents_abi::timeout(&*inner.clock, bound, close).await;
                closed.resolve(
                    outcome
                        .unwrap_or_else(|_| Err(RtcError::host("AVSynchronizer close timed out"))),
                );
            });
        }
        let closed = self.inner.closed.wait().await;
        closed.unwrap_or(Err(RtcError::Closed("AVSynchronizer")))
    }
}

#[cfg(test)]
#[path = "synchronizer_tests.rs"]
mod tests;
