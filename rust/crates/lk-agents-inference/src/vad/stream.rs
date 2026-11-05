//! Ordered VAD streams on admitted native workers. Construction, resampling, prediction and
//! recurrent resets stay on the same worker; callers only enqueue audio on a `Now` media queue and
//! pull events. A gap in the queue resets the detector, as a flush does.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{AudioFrame, Clock, Closed, StreamInput, StreamOutput, Ts};
use lk_agents_media::media_queue::{self, AudioSender, Item, Policy};
use lk_agents_media::pool::{Job, JobClass, MediaPool};
use lk_agents_net::audio_drops::AudioDrops;
use lk_agents_net::telemetry;
use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

use super::VadModelFactory;
use super::machine::{VadMachine, VadOptions, VadOutput, VadPredict, VadResult};

/// A decision on old audio is a wrong one: past 500 ms of lag the detector restarts on the newest.
const INPUT_POLICY: Policy = Policy::Now(Duration::from_millis(500));
const OUTPUT_CAPACITY: usize = 64;

fn failed(message: impl Into<String>) -> ProviderError {
    ProviderError::Local(ApiError::new(message.into()))
}

/// The `_FlushSentinel`, in order with the audio.
struct Flush;

struct Input {
    /// `None` once the input ended.
    audio: Mutex<Option<AudioSender<Flush>>>,
    drops: AudioDrops,
    cancelled: CancellationToken,
}

impl Input {
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
    let (audio, mut input) = media_queue::audio(INPUT_POLICY);
    let (output, events) = mpsc::channel(OUTPUT_CAPACITY);
    let drops = AudioDrops::new(telemetry::AudioConsumer::Vad, INPUT_POLICY);
    drops.observe(cx.clone());
    let cancelled = CancellationToken::new();
    let stop = cancelled.clone();
    // Only channel descriptors precede admission; refusal constructs no model and retains no
    // queued audio. The same worker owns the recurrent state until settlement.
    let worker = media_pool
        .try_spawn(JobClass::VadModel, move || {
            let mut machine = VadMachine::new(*options.borrow_and_update());
            let mut model = models.create();
            runtime.block_on(async move {
                loop {
                    let item = tokio::select! {
                        biased;
                        () = stop.cancelled() => return Ok(()),
                        item = input.recv() => match item {
                            Some(item) => item,
                            None => return Ok(()),
                        },
                    };
                    if options.has_changed().unwrap_or(false) {
                        machine.update_options(*options.borrow_and_update());
                    }
                    let mut next = match item {
                        Item::Media(frame) => match machine.push(frame) {
                            Ok(next) => next,
                            Err(error) => {
                                log::warn!(
                                    target: "livekit.agents",
                                    "dropping invalid VAD frame: {error}"
                                );
                                continue;
                            }
                        },
                        Item::Gap(_) | Item::Event(Flush) => {
                            machine.reset();
                            model.reset();
                            None
                        }
                    };
                    while let Some(VadPredict { window, samples }) = next {
                        if stop.is_cancelled() {
                            return Ok(());
                        }
                        let (probability, inference_duration) = model.predict(&samples);
                        let (events, following) = machine.predicted(VadResult {
                            window,
                            probability,
                            inference_duration,
                        });
                        for event in events {
                            tokio::select! {
                                biased;
                                () = stop.cancelled() => return Ok(()),
                                result = output.send(event) => {
                                    if result.is_err() { return Ok(()); }
                                },
                            }
                        }
                        next = following;
                    }
                }
            })
        })
        .map_err(|error| failed(error.to_string()))?;
    Ok(lk_agents_abi::DuplexStream::new(
        Box::new(Input { audio: Mutex::new(Some(audio)), drops, cancelled: cancelled.clone() }),
        Box::new(Output { events, worker: Some(worker), cancelled, clock }),
    ))
}
