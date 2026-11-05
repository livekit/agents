//! Ordered VAD streams on admitted native workers. Construction, resampling, prediction and
//! recurrent resets stay on the same worker; callers only enqueue audio on a `Now` media queue and
//! pull events. A gap in the queue resets the detector, as a flush does.

use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{AudioFrame, Clock, Closed, StreamInput, StreamOutput, Ts};
use lk_agents_media::media_queue::{self, AudioReceiver, AudioSender, Item, Policy};
use lk_agents_media::pool::{Job, JobClass, MediaPool};
use lk_agents_net::audio_drops::AudioDrops;
use lk_agents_net::telemetry;
use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

use super::machine::{VadMachine, VadOptions, VadOutput, VadPredict, VadResult};
use super::{VadModel, VadModelFactory};

/// A decision on old audio is a wrong one: past 500 ms of lag the detector restarts on the newest.
const INPUT_POLICY: Policy = Policy::Now(Duration::from_millis(500));
const OUTPUT_CAPACITY: usize = 64;

fn failed(message: impl Into<String>) -> ProviderError {
    ProviderError::Local(ApiError::new(message.into()))
}

/// The `_FlushSentinel`, in order with the audio.
pub(super) struct Flush;

/// A detection's audio input, the `Now` queue's sending half.
pub(super) struct Input {
    /// `None` once the input ended.
    audio: Mutex<Option<AudioSender<Flush>>>,
    pub(super) drops: Arc<AudioDrops>,
    cancelled: CancellationToken,
}

impl Input {
    /// Refuses audio once `cancelled`; drops are measured under `cx`'s job.
    pub(super) fn open(
        cancelled: CancellationToken,
        cx: &telemetry::Cx,
    ) -> (Self, AudioReceiver<Flush>) {
        let (audio, rx) = media_queue::audio(INPUT_POLICY);
        let drops = Arc::new(AudioDrops::new(telemetry::AudioConsumer::Vad, INPUT_POLICY));
        drops.observe(cx.clone());
        (Input { audio: Mutex::new(Some(audio)), drops, cancelled }, rx)
    }

    fn send<R>(&self, op: impl FnOnce(&AudioSender<Flush>) -> R) -> Result<R, ProviderError> {
        if self.cancelled.is_cancelled() {
            return Err(failed("the VAD stream is closed"));
        }
        let audio = self.audio.lock();
        let audio = audio.as_ref().ok_or_else(|| failed("the VAD input is closed"))?;
        if audio.is_closed() {
            return Err(failed("the VAD stream is closed"));
        }
        Ok(op(audio))
    }
}

impl StreamInput<AudioFrame> for Input {
    fn push(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        let dropped = self.send(|audio| audio.push(frame))?;
        self.drops.record(dropped);
        Ok(())
    }

    fn head(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        self.send(|audio| audio.head(frame))
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.send(|audio| audio.event(Flush))?.map_err(|stuck| failed(stuck.to_string()))
    }

    fn end_input(&self) {
        drop(self.audio.lock().take());
    }
}

struct Output {
    events: mpsc::Receiver<VadOutput>,
    worker: Option<Job<Result<(), ProviderError>>>,
    cancelled: CancellationToken,
    clock: Arc<dyn Clock>,
}

impl Drop for Output {
    fn drop(&mut self) {
        self.cancelled.cancel();
        self.events.close();
    }
}

impl StreamOutput<VadOutput> for Output {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<VadOutput>, ProviderError>> {
        Box::pin(async move {
            if let Some(event) = self.events.recv().await {
                return Ok(Some(event));
            }
            if let Some(worker) = self.worker.as_mut() {
                let result = worker.await;
                self.worker.take();
                result.map_err(|error| failed(format!("VAD worker failed: {error}")))??;
            }
            Ok(None)
        })
    }

    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.cancelled.cancel();
            self.events.close();
            let Some(mut worker) = self.worker.take() else { return Closed::Clean };
            // A timed-out receipt transfers to Job's supervisor; its permit remains held
            // until a native model that ignored cancellation actually returns.
            let deadline =
                deadline.min(self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT));
            Closed::bounded(self.clock.as_ref(), deadline, async {
                let _ = (&mut worker).await;
            })
            .await
        })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<VadOutput> for Output {}

/// Breaks once the stream is cancelled or its reader is gone.
async fn emit(
    stop: &CancellationToken,
    output: &mpsc::Sender<VadOutput>,
    event: VadOutput,
) -> ControlFlow<()> {
    let sent = tokio::select! {
        biased;
        () = stop.cancelled() => return ControlFlow::Break(()),
        result = output.send(event) => result,
    };
    if sent.is_err() { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
}

/// One detection on its admitted worker: the model and the machine's recurrent state, from
/// construction to settlement.
struct Detection {
    machine: VadMachine,
    model: Box<dyn VadModel>,
    options: watch::Receiver<VadOptions>,
    input: AudioReceiver<Flush>,
    output: mpsc::Sender<VadOutput>,
    stop: CancellationToken,
    catch_up: Arc<AudioDrops>,
    clock: Arc<dyn Clock>,
}

impl Detection {
    async fn run(mut self) -> Result<(), ProviderError> {
        loop {
            self.catch_up.catching_up(self.input.head(), self.clock.now());
            let item = tokio::select! {
                biased;
                () = self.stop.cancelled() => return Ok(()),
                item = self.input.recv() => item,
            };
            let Some(item) = item else { return Ok(()) };
            let ControlFlow::Continue(next) = self.admit(item).await else { return Ok(()) };
            if self.predict_windows(next).await.is_break() {
                return Ok(());
            }
        }
    }

    /// One input item through the machine, under the latest options: audio may complete a
    /// window, and a gap or flush resets the detector. An invalid frame is dropped.
    async fn admit(
        &mut self,
        item: Item<AudioFrame, Flush>,
    ) -> ControlFlow<(), Option<VadPredict>> {
        if self.options.has_changed().unwrap_or(false) {
            self.machine.update_options(*self.options.borrow_and_update());
        }
        match item {
            Item::Media(frame) => match self.machine.push(frame) {
                Ok(next) => ControlFlow::Continue(next),
                Err(error) => {
                    log::warn!(target: "livekit.agents", "dropping invalid VAD frame: {error}");
                    ControlFlow::Continue(None)
                }
            },
            Item::Gap(_) | Item::Event(Flush) => {
                let ended = self.machine.reset();
                self.model.reset();
                if let Some(event) = ended
                    && emit(&self.stop, &self.output, event).await.is_break()
                {
                    return ControlFlow::Break(());
                }
                ControlFlow::Continue(None)
            }
        }
    }

    /// Predicts every window the machine asks for, emitting its events; a cancel is checked
    /// before each prediction.
    async fn predict_windows(&mut self, mut next: Option<VadPredict>) -> ControlFlow<()> {
        while let Some(VadPredict { window, samples }) = next {
            if self.stop.is_cancelled() {
                return ControlFlow::Break(());
            }
            let (probability, inference_duration) = self.model.predict(&samples);
            let (events, following) =
                self.machine.predicted(VadResult { window, probability, inference_duration });
            for event in events {
                if emit(&self.stop, &self.output, event).await.is_break() {
                    return ControlFlow::Break(());
                }
            }
            next = following;
        }
        ControlFlow::Continue(())
    }
}

type DetectionStream =
    lk_agents_abi::DuplexStream<Box<dyn StreamInput<AudioFrame>>, Box<dyn StreamOutput<VadOutput>>>;

/// One detection: input queued for, and events produced by, an admitted worker that owns the
/// model from construction to settlement. `runtime` runs the worker's channel waits; drops are
/// measured under `cx`'s job.
pub(super) fn open(
    mut options: watch::Receiver<VadOptions>,
    models: Arc<dyn VadModelFactory>,
    clock: Arc<dyn Clock>,
    media_pool: MediaPool,
    runtime: tokio::runtime::Handle,
    cx: &telemetry::Cx,
) -> Result<DetectionStream, ProviderError> {
    let cancelled = CancellationToken::new();
    let (sink, input) = Input::open(cancelled.clone(), cx);
    let (output, events) = mpsc::channel(OUTPUT_CAPACITY);
    let (catch_up, worker_clock) = (sink.drops.clone(), clock.clone());
    let stop = cancelled.clone();
    // Only channel descriptors precede admission; refusal constructs no model and retains no
    // queued audio. The same worker owns the recurrent state until settlement.
    let worker = media_pool
        .try_spawn(JobClass::VadModel, move || {
            let machine = VadMachine::new(*options.borrow_and_update());
            let model = models.create();
            let detection = Detection {
                machine,
                model,
                options,
                input,
                output,
                stop,
                catch_up,
                clock: worker_clock,
            };
            runtime.block_on(detection.run())
        })
        .map_err(|error| failed(error.to_string()))?;
    Ok(lk_agents_abi::DuplexStream::new(
        Box::new(sink),
        Box::new(Output { events, worker: Some(worker), cancelled, clock }),
    ))
}
