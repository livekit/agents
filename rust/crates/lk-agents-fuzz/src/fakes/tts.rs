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

pub struct FakeTts {
    registry: Arc<Registry>,
}

impl FakeTts {
    pub fn new(registry: Arc<Registry>) -> Arc<Self> {
        Arc::new(FakeTts { registry })
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
        Ok((
            Box::new(InputSide(synthesis)),
            Box::new(Output { slot, registry: self.registry.clone(), receiver }),
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
        self.receiver.recv().await.transpose()
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    async fn close(self: Box<Self>, _deadline: Ts) -> Closed {
        self.registry.mark_closed(self.slot);
        Closed::Clean
    }
}
