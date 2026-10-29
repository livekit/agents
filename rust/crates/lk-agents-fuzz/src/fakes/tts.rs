//! Synthesis: each opened stream is an invocation the case controls. Text the product pushes
//! becomes one identifiable 10 ms chunk per push, tagged with the stream's `RequestId` and the
//! push index, and the pushed text is recorded beside it. The case can `Hold` a stream (pushed
//! text queues, nothing is produced), `Release` it (the queue drains), or `Fail` it (the product
//! reads a provider error). This fake cannot synthesize arbitrary audio.
//!
//! A voice that aligns stamps each chunk's words as Cartesia's `word_timestamps` do through the
//! gateway: one token per word, the word and a trailing space, with its start and end in seconds
//! from the stream's first audio. A chunk's words share its audio evenly, in order.
use std::sync::{Arc, Mutex, OnceLock};

use lk_agents_abi::{
    AudioFrame, SynthesizedAudio, TimedString, Ts, Tts, TtsCapabilities, TtsSink, TtsSource,
    errors::{ApiError, ProviderError},
    utils::stream::{Closed, StreamSink, StreamSource},
};
use tokio::sync::mpsc;

use super::{Control, Registry, SAMPLE_RATE, SAMPLES_PER_CHUNK, Slot};
use crate::{
    drive::TtsAlignment,
    input::Input,
    record::{Detail, Kind, Tag},
};

/// The recorded voice the latency and the pace are drawn from.
const TTS_MODEL: &str = "cartesia/sonic-3";

/// The longest frame the voice sends, as a streaming voice sends audio in short frames.
const FRAME_MS: usize = 100;
const FRAME_SAMPLES: usize = FRAME_MS * SAMPLE_RATE as usize / 1000;

/// Characters a word of the agent's lines takes, its space included (`input::AGENT_TURNS`).
const CHARS_PER_WORD: f64 = 5.3;

pub struct FakeTts {
    registry: Arc<Registry>,
    /// The voice's time to first audio, drawn per stream from the corpus (`tts/first_audio`),
    /// when the setup gives the voice latency; `None` answers at once.
    latency: Option<Mutex<fastrand::Rng>>,
    /// Off, the voice only synthesizes whole texts: the session adapts it sentence by sentence.
    streaming: bool,
    alignment: TtsAlignment,
}

impl FakeTts {
    /// `latency`: a voice that takes as long to start as the recorded one did, drawn from this
    /// seed. `streaming` off, the voice only synthesizes whole texts: the session adapts it
    /// sentence by sentence.
    pub fn new(
        registry: Arc<Registry>,
        latency: Option<u64>,
        streaming: bool,
        alignment: TtsAlignment,
    ) -> Arc<Self> {
        let latency = latency.map(|seed| Mutex::new(fastrand::Rng::with_seed(seed)));
        Arc::new(FakeTts { registry, latency, streaming, alignment })
    }
}

/// One stream, shared by the product's input side and the case's control.
pub struct Synthesis {
    slot: OnceLock<Slot>,
    registry: Arc<Registry>,
    state: Mutex<SynthesisState>,
    stamps: bool,
}

struct SynthesisState {
    sender: Option<mpsc::UnboundedSender<Result<SynthesizedAudio, ProviderError>>>,
    held: bool,
    /// The product said its input is complete; the output ends once the queue is drained.
    ended: bool,
    queued: Vec<String>,
    chunks: u16,
    /// Seconds of audio produced so far: where the next chunk's words start.
    spoken: f64,
    /// How long this voice takes over a character of text, in samples.
    samples_per_char: f64,
}

impl Synthesis {
    fn nth(&self) -> u32 {
        self.slot.get().expect("registered before use").nth
    }

    /// The audio for one push of text, as long as the text takes to say at the voice's pace, in
    /// frames of at most [`FRAME_MS`]: each frame a chunk of its own, the text and its word
    /// timings on the first.
    fn emit(&self, state: &mut SynthesisState, text: String) {
        let total = ((text.chars().count() as f64 * state.samples_per_char) as usize)
            .max(SAMPLES_PER_CHUNK);
        let span = total as f64 / f64::from(SAMPLE_RATE);
        let mut timed_transcript =
            if self.stamps { stamps(&text, state.spoken, span) } else { Vec::new() };
        let mut text = Some(text);
        let mut left = total;
        while left > 0 {
            let Some(sender) = &state.sender else { return };
            let samples = left.min(FRAME_SAMPLES);
            let chunk = state.chunks;
            let sample = Tag::synthesized(self.nth(), chunk);
            let frame = AudioFrame::new(
                sample.to_le_bytes().repeat(samples),
                SAMPLE_RATE,
                1,
                samples as u32,
            );
            let words = std::mem::take(&mut timed_transcript);
            let stamped =
                words.iter().map(|w| (w.text.clone(), w.start_time.unwrap_or_default())).collect();
            let audio = SynthesizedAudio {
                timed_transcript: words,
                ..SynthesizedAudio::new(frame, "synthesis")
            };
            if sender.send(Ok(audio)).is_err() {
                return;
            }
            state.spoken += samples as f64 / f64::from(SAMPLE_RATE);
            self.registry.record_produced(
                self.nth(),
                chunk,
                text.take().unwrap_or_default(),
                stamped,
            );
            state.chunks += 1;
            left -= samples;
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

impl Tts for FakeTts {
    fn label(&self) -> String {
        "fake::tts".into()
    }

    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities {
            streaming: self.streaming,
            aligned_transcript: self.alignment != TtsAlignment::None,
        }
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn num_channels(&self) -> u16 {
        1
    }

    fn synthesize<'a>(
        &'a self,
        text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<TtsSource, ProviderError>> {
        Box::pin(async move {
            let (sink, source) = self.stream().await?;
            sink.push(text.into())?;
            sink.end_input();
            Ok(source)
        })
    }

    fn stream(&self) -> lk_agents_abi::BoxFuture<'_, Result<(TtsSink, TtsSource), ProviderError>> {
        Box::pin(async move {
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
                    spoken: 0.0,
                    samples_per_char: 0.0,
                }),
                stamps: self.alignment == TtsAlignment::Words,
            });
            let slot =
                self.registry.register(Kind::Tts, Detail::None, Control::Tts(synthesis.clone()));
            let _ = synthesis.slot.set(slot);
            // Each stream speaks at a pace the recorded voice had: its milliseconds a word, over the
            // characters a word of the agent's lines takes.
            let mut pace = fastrand::Rng::with_seed(u64::from(slot.nth) ^ 0x7061_6365);
            let ms_per_word =
                lk_agents_test_support::corpus::draw(&mut pace, &["tts", TTS_MODEL, "ms_per_word"]);
            synthesis.state.lock().unwrap().samples_per_char =
                ms_per_word / CHARS_PER_WORD * f64::from(SAMPLE_RATE) / 1000.0;
            let first_audio = self.latency.as_ref().map(|rng| {
                let ms = lk_agents_test_support::corpus::draw_ms(
                    &mut rng.lock().unwrap(),
                    &["tts", TTS_MODEL, "first_audio"],
                );
                tokio::time::Instant::now() + std::time::Duration::from_millis(ms)
            });
            if first_audio.is_some() {
                self.registry.voice_due(slot.nth, first_audio);
            }
            Ok((
                Box::new(InputSide(synthesis)) as TtsSink,
                Box::new(Output {
                    slot,
                    registry: self.registry.clone(),
                    receiver,
                    first_audio,
                    delivered: false,
                }) as TtsSource,
            ))
        })
    }
}

/// A chunk's words, stamped across its `span` seconds from `at`, in order.
fn stamps(text: &str, at: f64, span: f64) -> Vec<TimedString> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let each = span / words.len().max(1) as f64;
    words
        .iter()
        .enumerate()
        .map(|(i, word)| {
            let start = at + i as f64 * each;
            TimedString {
                start_time: Some(start),
                end_time: Some(start + each),
                ..TimedString::new(format!("{word} "))
            }
        })
        .collect()
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

impl StreamSource<SynthesizedAudio> for Output {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(async move {
            let next = self.receiver.recv().await.transpose();
            // The first audio leaves no earlier than the voice can produce it; a failure or the end
            // is reported as soon as it is known.
            if let (Ok(Some(_)), Some(at)) = (&next, self.first_audio.take()) {
                tokio::time::sleep_until(at).await;
                self.registry.voice_due(self.slot.nth, None);
            }
            if matches!(next, Ok(Some(_))) && !std::mem::replace(&mut self.delivered, true) {
                self.registry.record_delivered(self.slot);
            }
            next
        })
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    fn close(self: Box<Self>, _deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.registry.voice_due(self.slot.nth, None);
            self.registry.mark_closed(self.slot);
            Closed::Clean
        })
    }
}
