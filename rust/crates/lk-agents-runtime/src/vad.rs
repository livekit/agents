//! Ordered VAD streams on admitted native workers. Construction, resampling, prediction and
//! recurrent resets stay on the same worker; callers only enqueue bounded audio and pull events.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{AudioFrame, Clock, Closed, StreamSink, StreamSource, Ts};
use lk_agents_inference::vad::machine::{VadMachine, VadOptions, VadOutput, VadPredict, VadResult};
use lk_agents_inference::vad::{VadModelFactory, VadRuntime, VadRuntimeStream};
use tokio::sync::{Notify, mpsc, watch};
use tokio_util::sync::CancellationToken;

use lk_agents_media::workers::{MediaWorker, MediaWorkerClass};

// At a 10 ms input cadence the item ceiling admits 2.56 s of pending audio. The byte ceiling
// also bounds oversized producers; one full second of 48 kHz stereo fits a single push.
const INPUT_CAPACITY: usize = 256;
const MAX_FRAME_BYTES: usize = 192_000;
const INPUT_BYTES: usize = 2 * 1024 * 1024;
const OUTPUT_CAPACITY: usize = 64;

fn failed(message: impl Into<String>) -> ProviderError {
    ProviderError::Local(ApiError::new(message.into()))
}

enum Command {
    Frame(AudioFrame),
    Flush,
}

struct InputState {
    ended: AtomicBool,
    ending: Notify,
    queued_bytes: AtomicUsize,
}

struct Input {
    sender: mpsc::Sender<Command>,
    state: Arc<InputState>,
    cancelled: CancellationToken,
}

impl Input {
    fn send(&self, command: Command) -> Result<(), ProviderError> {
        if self.cancelled.is_cancelled() {
            return Err(failed("the VAD stream is closed"));
        }
        if self.state.ended.load(Ordering::Acquire) {
            return Err(failed("the VAD input is closed"));
        }
        let bytes = match &command {
            Command::Frame(frame) => frame.data.len(),
            Command::Flush => 0,
        };
        self.state
            .queued_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                queued.checked_add(bytes).filter(|&queued| queued <= INPUT_BYTES)
            })
            .map_err(|_| failed("VAD input byte capacity exhausted"))?;
        if let Err(error) = self.sender.try_send(command) {
            self.state.queued_bytes.fetch_sub(bytes, Ordering::AcqRel);
            return Err(match error {
                mpsc::error::TrySendError::Full(_) => failed("VAD input capacity exhausted"),
                mpsc::error::TrySendError::Closed(_) => failed("the VAD stream is closed"),
            });
        }
        Ok(())
    }
}

impl StreamSink<AudioFrame> for Input {
    fn push(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        if frame.data.len() > MAX_FRAME_BYTES {
            return Err(failed("VAD input frame exceeds the 192000-byte limit"));
        }
        self.send(Command::Frame(frame))
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.send(Command::Flush)
    }

    fn end_input(&self) {
        self.state.ended.store(true, Ordering::Release);
        self.state.ending.notify_one();
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        self.end_input();
    }
}

struct Output {
    events: mpsc::Receiver<VadOutput>,
    worker: Option<MediaWorker<Result<(), ProviderError>>>,
    cancelled: CancellationToken,
    clock: Arc<dyn Clock>,
}

impl Drop for Output {
    fn drop(&mut self) {
        self.cancelled.cancel();
        self.events.close();
    }
}

impl StreamSource<VadOutput> for Output {
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
            // A timed-out receipt transfers to MediaWorker's supervisor; its permit remains held
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

fn open(
    mut options: watch::Receiver<VadOptions>,
    models: Arc<dyn VadModelFactory>,
    clock: Arc<dyn Clock>,
) -> Result<VadRuntimeStream, ProviderError> {
    let workers = crate::process::workers().map_err(|error| failed(error.to_string()))?;
    let (sender, mut input) = mpsc::channel(INPUT_CAPACITY);
    let (output, events) = mpsc::channel(OUTPUT_CAPACITY);
    let state = Arc::new(InputState {
        ended: AtomicBool::new(false),
        ending: Notify::new(),
        queued_bytes: AtomicUsize::new(0),
    });
    let input_state = state.clone();
    let cancelled = CancellationToken::new();
    let stop = cancelled.clone();
    let runtime = crate::process::caller_handle().map_err(|error| failed(error.to_string()))?;
    // Only bounded channel descriptors precede admission; refusal constructs no model and
    // retains no queued audio. The same worker owns the recurrent state until settlement.
    let worker = workers
        .try_spawn(MediaWorkerClass::VadModel, move || {
            let mut machine = VadMachine::new(*options.borrow_and_update());
            let mut model = models.create();
            runtime.block_on(async move {
                loop {
                    if input_state.ended.load(Ordering::Acquire) {
                        input.close();
                    }
                    let item = tokio::select! {
                        biased;
                        () = stop.cancelled() => return Ok(()),
                        () = input_state.ending.notified() => { input.close(); continue; },
                        item = input.recv() => match item {
                            Some(item) => item,
                            None => return Ok(()),
                        },
                    };
                    if options.has_changed().unwrap_or(false) {
                        machine.update_options(*options.borrow_and_update());
                    }
                    let mut next = match item {
                        Command::Frame(frame) => {
                            input_state.queued_bytes.fetch_sub(frame.data.len(), Ordering::AcqRel);
                            match machine.push(frame) {
                                Ok(next) => next,
                                Err(error) => {
                                    log::warn!(
                                        target: "livekit.agents",
                                        "dropping invalid VAD frame: {error}"
                                    );
                                    continue;
                                }
                            }
                        }
                        Command::Flush => {
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
    Ok((
        Box::new(Input { sender, state, cancelled: cancelled.clone() }),
        Box::new(Output { events, worker: Some(worker), cancelled, clock }),
    ))
}

lk_agents_abi::inventory::submit! { VadRuntime { open } }
