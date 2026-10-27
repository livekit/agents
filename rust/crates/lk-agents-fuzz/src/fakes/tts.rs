//! Synthesis: each opened stream is an invocation the case controls. Text the product pushes
//! becomes one identifiable 10 ms chunk per push, tagged with the stream's `RequestId` and the
//! push index, and the pushed text is recorded beside it. The case can `Hold` a stream (pushed
//! text queues, nothing is produced), `Release` it (the queue drains), or `Fail` it (the product
//! reads a provider error). This fake cannot synthesize arbitrary audio.
use std::sync::{Arc, Mutex, OnceLock};

use lk_agents_abi::{
    AudioFrame, SynthesizedAudio, Ts, Tts, TtsCapabilities, TtsSink, TtsSource,
    errors::{ApiError, ProviderError},
    utils::stream::{Closed, StreamSink, StreamSource},
};
use tokio::sync::mpsc;

use super::{Control, Registry, SAMPLE_RATE, SAMPLES_PER_CHUNK, Slot};
use crate::{
    input::Input,
    record::{Detail, Kind, Tag},
};

/// The recorded voice the latency is drawn from.
const TTS_MODEL: &str = "cartesia/sonic-3";

pub struct FakeTts {
    registry: Arc<Registry>,
    /// The voice's time to first audio, drawn per stream from the corpus (`tts/first_audio`),
    /// when the setup gives the voice latency; `None` answers at once.
    latency: Option<Mutex<fastrand::Rng>>,
}

impl FakeTts {
    pub fn new(registry: Arc<Registry>) -> Arc<Self> {
        Arc::new(FakeTts { registry, latency: None })
    }

    /// A voice that takes as long to start as the recorded one did.
    pub fn with_latency(registry: Arc<Registry>, seed: u64) -> Arc<Self> {
        Arc::new(FakeTts { registry, latency: Some(Mutex::new(fastrand::Rng::with_seed(seed))) })
    }
}

/// One stream, shared by the product's input side and the case's control.
pub struct Synthesis {
    slot: OnceLock<Slot>,
    registry: Arc<Registry>,
    state: Mutex<SynthesisState>,
}

struct SynthesisState {
    sender: Option<mpsc::UnboundedSender<Result<SynthesizedAudio, ProviderError>>>,
    held: bool,
    /// The product said its input is complete; the output ends once the queue is drained.
    ended: bool,
    queued: Vec<String>,
    chunks: u16,
}

impl Synthesis {
    fn nth(&self) -> u32 {
        self.slot.get().expect("registered before use").nth
    }

    fn emit(&self, state: &mut SynthesisState, text: String) {
        let Some(sender) = &state.sender else { return };
        let chunk = state.chunks;
        let sample = Tag::synthesized(self.nth(), chunk);
        let frame = AudioFrame::new(
            sample.to_le_bytes().repeat(SAMPLES_PER_CHUNK),
            SAMPLE_RATE,
            1,
            SAMPLES_PER_CHUNK as u32,
        );
        if sender.send(Ok(SynthesizedAudio::new(frame, "synthesis"))).is_ok() {
            self.registry.record_produced(self.nth(), chunk, text);
            state.chunks += 1;
        }
    }

    fn end_if_done(state: &mut SynthesisState) {
        if state.ended && !state.held && state.queued.is_empty() {
            state.sender.take();
        }
    }

    /// One synthesis delivery from the case. True when the product still reads the stream.
    pub(super) fn deliver(&self, what: &Input) -> bool {
        let mut state = self.state.lock().unwrap();
        let live = state.sender.as_ref().is_some_and(|s| !s.is_closed());
        match what {
            Input::TtsHold { .. } => state.held = true,
            Input::TtsRelease { .. } => {
                state.held = false;
                for text in std::mem::take(&mut state.queued) {
                    self.emit(&mut state, text);
                }
                Self::end_if_done(&mut state);
            }
            Input::TtsFail { .. } => {
                if let Some(sender) = state.sender.take() {
                    let _ = sender.send(Err(ProviderError::Local(ApiError::new(
                        "injected synthesis failure",
                    ))));
                }
            }
            _ => return false,
        }
        live
    }

    /// Whether the case still has a stream to control.
    pub(super) fn open(&self) -> bool {
        self.state.lock().unwrap().sender.is_some()
    }
}

struct Output {
    slot: Slot,
    registry: Arc<Registry>,
    receiver: mpsc::UnboundedReceiver<Result<SynthesizedAudio, ProviderError>>,
    /// When the first audio may leave: the stream's open plus the voice's time to first audio.
    first_audio: Option<tokio::time::Instant>,
    /// The first audio was handed over.
    delivered: bool,
}

#[async_trait::async_trait]
impl Tts for FakeTts {
    fn label(&self) -> String {
        "fake::tts".into()
    }

    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities { streaming: true, aligned_transcript: false }
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn num_channels(&self) -> u16 {
        1
    }

    async fn synthesize(&self, text: &str) -> Result<TtsSource, ProviderError> {
        let (sink, source) = self.stream().await?;
        sink.push(text.into())?;
        sink.end_input();
        Ok(source)
    }

    async fn stream(&self) -> Result<(TtsSink, TtsSource), ProviderError> {
        if self.registry.voice_down() {
            return Err(ProviderError::Local(ApiError::new("voice provider is down")));
        }
        let (sender, receiver) = mpsc::unbounded_channel();
        let synthesis = Arc::new(Synthesis {
            slot: OnceLock::new(),
            registry: self.registry.clone(),
            state: Mutex::new(SynthesisState {
                sender: Some(sender),
                held: false,
                ended: false,
                queued: vec![],
                chunks: 0,
            }),
        });
        let slot = self.registry.register(Kind::Tts, Detail::None, Control::Tts(synthesis.clone()));
        let _ = synthesis.slot.set(slot);
        let first_audio = self.latency.as_ref().map(|rng| {
            let ms = crate::corpus::draw_ms(&mut rng.lock().unwrap(), &["tts", TTS_MODEL, "first_audio"]);
            tokio::time::Instant::now() + std::time::Duration::from_millis(ms)
        });
        Ok((
            Box::new(InputSide(synthesis)),
            Box::new(Output {
                slot,
                registry: self.registry.clone(),
                receiver,
                first_audio,
                delivered: false,
            }),
        ))
    }
}

/// The product's side of a stream: what it pushes goes through the shared state.
struct InputSide(Arc<Synthesis>);

impl StreamSink<String> for InputSide {
    fn push(&self, text: String) -> Result<(), ProviderError> {
        if text.trim().is_empty() {
            return Ok(());
        }
        let mut state = self.0.state.lock().unwrap();
        if state.held {
            state.queued.push(text);
        } else {
            self.0.emit(&mut state, text);
        }
        Ok(())
    }

    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn end_input(&self) {
        let mut state = self.0.state.lock().unwrap();
        state.ended = true;
        Synthesis::end_if_done(&mut state);
    }
}

#[async_trait::async_trait]
impl StreamSource<SynthesizedAudio> for Output {
    async fn next(&mut self) -> Result<Option<SynthesizedAudio>, ProviderError> {
        let next = self.receiver.recv().await.transpose();
        // The first audio leaves no earlier than the voice can produce it; a failure or the end
        // is reported as soon as it is known.
        if let (Ok(Some(_)), Some(at)) = (&next, self.first_audio.take()) {
            tokio::time::sleep_until(at).await;
        }
        if matches!(next, Ok(Some(_))) && !std::mem::replace(&mut self.delivered, true) {
            self.registry.record_delivered(self.slot);
        }
        next
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    async fn close(self: Box<Self>, _deadline: Ts) -> Closed {
        self.registry.mark_closed(self.slot);
        Closed::Clean
    }
}
