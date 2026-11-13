use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{AudioFrame, Clock, Closed, StreamInput, StreamOutput, Ts};
use lk_agents_media::media_queue::{self, AudioReceiver, AudioSender, Item, Policy};
use lk_agents_media::pool::{Job, JobClass, MediaPool};
use lk_agents_net::signal::Cell;
use lk_agents_net::source::{Gated, InputGate};
use lk_agents_telemetry as telemetry;
use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::machine::{VadMachine, VadOptions, VadOutput, VadPredict, VadResult};
use super::{VadModel, VadModelFactory};

// Construction, resampling, prediction and recurrent resets stay on the same worker. A gap in the
// queue resets the detector, as a flush does.

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
    /// What a refused call answers: the detection's failure, else its end.
    gate: InputGate,
}

impl Input {
    /// Refuses audio once `gate` is no longer open; the queue is measured under `cx`'s job.
    pub(super) fn open(
        gate: InputGate,
        cx: &telemetry::Cx,
        clock: Arc<dyn Clock>,
    ) -> (Self, AudioReceiver<Flush>) {
        let (audio, rx) = media_queue::audio(INPUT_POLICY, telemetry::AudioConsumer::Vad, clock);
        audio.observe(cx.clone());
        (Input { audio: Mutex::new(Some(audio)), gate }, rx)
    }

    fn send<R>(&self, op: impl FnOnce(&AudioSender<Flush>) -> R) -> Result<R, ProviderError> {
        self.gate.admit()?;
        let audio = self.audio.lock();
        if let Some(audio) = audio.as_ref().filter(|audio| !audio.is_closed()) {
            return Ok(op(audio));
        }
        drop(audio);
        Err(self.gate.refused())
    }
}

impl StreamInput<AudioFrame> for Input {
    fn push(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        self.send(|audio| audio.push(frame))?;
        Ok(())
    }

    fn head(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        self.send(|audio| audio.head(frame))
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.send(|audio| audio.event(Flush))?.map_err(|stuck| failed(stuck.to_string()))
    }

    fn end_input(&self) {
        self.gate.end();
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
    options: Cell<VadOptions>,
    /// The options the machine runs under.
    applied: VadOptions,
    input: AudioReceiver<Flush>,
    output: mpsc::Sender<VadOutput>,
    stop: CancellationToken,
}

impl Detection {
    async fn run(mut self) -> Result<(), ProviderError> {
        loop {
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
        let latest = self.options.get();
        if latest != self.applied {
            self.applied = latest;
            self.machine.update_options(latest);
        }
        match item {
            Item::Media(frame) => match self.machine.push(frame) {
                Ok(next) => ControlFlow::Continue(next),
                Err(error) => {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, "dropping invalid VAD frame: {error}");
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
/// model from construction to settlement and drives its channel waits itself; drops are measured
/// under `cx`'s job.
pub(super) fn open(
    options: Cell<VadOptions>,
    models: Arc<dyn VadModelFactory>,
    clock: Arc<dyn Clock>,
    media_pool: MediaPool,
    cx: &telemetry::Cx,
) -> Result<DetectionStream, ProviderError> {
    let cancelled = CancellationToken::new();
    let gate = InputGate::default();
    let (sink, input) = Input::open(gate.clone(), cx, clock.clone());
    let (output, events) = mpsc::channel(OUTPUT_CAPACITY);
    let stop = cancelled.clone();
    // Only channel descriptors precede admission; refusal constructs no model and retains no
    // queued audio. The same worker owns the recurrent state until settlement.
    let worker = media_pool
        .try_spawn(JobClass::VadModel, move || {
            let applied = options.get();
            let machine = VadMachine::new(applied);
            let model = models.create();
            let detection = Detection { machine, model, options, applied, input, output, stop };
            park_on(detection.run())
        })
        .map_err(|error| failed(error.to_string()))?;
    Ok(lk_agents_abi::DuplexStream::new(
        Box::new(sink),
        Box::new(Gated::new(Output { events, worker: Some(worker), cancelled, clock }, gate)),
    ))
}

/// Drives `work` on this worker thread, parked between wakes: a detection waits only on its
/// channels and its cancellation, never on time or I/O.
fn park_on<F: std::future::Future>(work: F) -> F::Output {
    struct Unpark(std::thread::Thread);
    impl std::task::Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let mut work = std::pin::pin!(work);
    let waker = std::task::Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = std::task::Context::from_waker(&waker);
    loop {
        if let std::task::Poll::Ready(output) = work.as_mut().poll(&mut cx) {
            return output;
        }
        std::thread::park();
    }
}
