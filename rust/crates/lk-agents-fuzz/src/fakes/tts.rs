use std::sync::{Arc, Mutex, OnceLock};

use lk_agents_abi::{
    AudioFrame, SpeechChunk, SynthesizedAudio, TranscriptChunk, Ts, Tts, TtsCapabilities, TtsSink,
    TtsSource,
    errors::{ApiError, ProviderError},
    utils::stream::{Closed, Entry, StreamInput, StreamOutput},
};
use lk_agents_core::markup::TranscriptMarkupStripper;
use tokio::sync::{Notify, mpsc};

use super::{
    Control, Registry, SAMPLE_RATE, SAMPLES_PER_CHUNK, Slot, Timing, delay,
    hazard::{Site, StreamHazard, strike},
    measured,
};
use crate::{
    drive::TtsAlignment,
    input::Input,
    record::{Detail, Kind, Tag},
};

use crate::calibration::tts::FIRST_AUDIO;
pub use crate::calibration::tts::MS_PER_WORD;

/// The longest frame the voice sends.
const FRAME_MS: usize = 100;
const FRAME_SAMPLES: usize = FRAME_MS * SAMPLE_RATE as usize / 1000;

/// Characters a word of the agent's lines takes, its space included (`input::AGENT_TURNS`).
const CHARS_PER_WORD: f64 = 5.3;

/// Text a held voice's input takes before refusing a push, as a host TTS's input refuses once
/// its loop stops taking it (`py_streams::SINK_QUEUE`, smaller so a case reaches it). A `send`
/// waits for the voice instead.
pub const HELD_INPUT: usize = 4;

pub struct FakeTts {
    registry: Arc<Registry>,
    /// Draws each stream's time to first audio; `None` answers at once.
    latency: Option<Mutex<fastrand::Rng>>,
    timing: Timing,
    /// Off, the session adapts the voice sentence by sentence.
    streaming: bool,
    alignment: TtsAlignment,
    /// The markup dialect the voice speaks (`""` for none).
    markup: &'static str,
    /// Whether its stage paces the chunks it is sent.
    pacing: bool,
}

pub struct TtsOptions {
    pub latency: Option<u64>,
    pub timing: Timing,
    pub streaming: bool,
    pub alignment: TtsAlignment,
    pub markup: &'static str,
    pub pacing: bool,
}

impl FakeTts {
    pub fn new(registry: Arc<Registry>, options: TtsOptions) -> Arc<Self> {
        let TtsOptions { latency, timing, streaming, alignment, markup, pacing } = options;
        let latency = latency.map(|seed| Mutex::new(fastrand::Rng::with_seed(seed)));
        Arc::new(FakeTts { registry, latency, timing, streaming, alignment, markup, pacing })
    }
}

/// One stream, shared by the product's input side and the case's control.
pub struct Synthesis {
    slot: OnceLock<Slot>,
    registry: Arc<Registry>,
    state: Mutex<SynthesisState>,
    stamps: bool,
    /// The voice stopped holding its input: released, or failed.
    unheld: Notify,
}

struct SynthesisState {
    sender: Option<mpsc::UnboundedSender<Result<SynthesizedAudio, ProviderError>>>,
    held: bool,
    /// Input ended; the output ends once the queue is drained.
    ended: bool,
    queued: Vec<String>,
    chunks: u16,
    /// Seconds of audio produced so far: where the next chunk's words start.
    spoken: f64,
    samples_per_char: f64,
    /// The words the voice says, its markup taken as instructions: a real voice never stamps a
    /// tag, even one split across pushes.
    said: TranscriptMarkupStripper,
}

impl Synthesis {
    fn nth(&self) -> u32 {
        self.slot.get().expect("registered before use").nth
    }

    /// One push of text at the voice's pace, in frames of at most [`FRAME_MS`], the text and its
    /// word timings on the first.
    fn emit(&self, state: &mut SynthesisState, text: String) {
        let total = ((text.chars().count() as f64 * state.samples_per_char) as usize)
            .max(SAMPLES_PER_CHUNK);
        let span = total as f64 / f64::from(SAMPLE_RATE);
        let mut timed_transcript = if self.stamps {
            stamps(&state.said.push(&text), state.spoken, span)
        } else {
            Vec::new()
        };
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

    fn full(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.held && state.sender.is_some() && state.queued.len() >= HELD_INPUT
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
                self.unheld.notify_waiters();
            }
            Input::TtsFail { .. } => {
                self.unheld.notify_waiters();
                if let Some(sender) = state.sender.take() {
                    return sender
                        .send(Err(ProviderError::Local(ApiError::new(
                            "injected synthesis failure",
                        ))))
                        .is_ok();
                }
            }
            _ => return false,
        }
        live
    }

    pub(super) fn open(&self) -> bool {
        self.state.lock().unwrap().sender.is_some()
    }
}

struct Output {
    slot: Slot,
    registry: Arc<Registry>,
    receiver: mpsc::UnboundedReceiver<Result<SynthesizedAudio, ProviderError>>,
    /// When the first audio may leave.
    first_audio: Option<tokio::time::Instant>,
    delivered: bool,
    hazard: StreamHazard,
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

    fn markup_provider(&self) -> String {
        self.markup.to_owned()
    }

    fn options(&self) -> lk_agents_abi::tts::types::TtsOptions {
        lk_agents_abi::tts::types::TtsOptions {
            pacing: self
                .pacing
                .then(lk_agents_abi::tts::stream_pacer::SentenceStreamPacer::default),
            ..lk_agents_abi::tts::types::TtsOptions::default()
        }
    }

    fn synthesize<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        chunk: &'a SpeechChunk,
    ) -> lk_agents_abi::BoxFuture<'a, Result<TtsSource, ProviderError>> {
        Box::pin(async move {
            let (sink, source) = self.stream(&lk_agents_abi::telemetry::Cx::NONE).await?.split();
            sink.push(chunk.clone())?;
            sink.end_input();
            Ok(source)
        })
    }

    fn stream(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::DuplexStream<TtsSink, TtsSource>, ProviderError>,
    > {
        Box::pin(async move {
            if self.registry.voice_down() {
                return Err(ProviderError::Local(ApiError::new("voice provider is down")));
            }
            if let Some(fault) = self.registry.hazard(Site::TtsOpen) {
                return strike(fault, Site::TtsOpen).await;
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
                    said: TranscriptMarkupStripper::new(),
                }),
                stamps: self.alignment == TtsAlignment::Words,
                unheld: Notify::new(),
            });
            let slot =
                self.registry.register(Kind::Tts, Detail::None, Control::Tts(synthesis.clone()));
            assert!(synthesis.slot.set(slot).is_ok(), "synthesis registered twice");
            // A real voice's pace: measured milliseconds per word over characters per word.
            let mut pace = fastrand::Rng::with_seed(u64::from(slot.nth) ^ 0x7061_6365);
            let ms_per_word = measured(&mut pace, &MS_PER_WORD) as f64;
            synthesis.state.lock().unwrap().samples_per_char =
                ms_per_word / CHARS_PER_WORD * f64::from(SAMPLE_RATE) / 1000.0;
            let first_audio = self.latency.as_ref().map(|rng| {
                let ms = delay(&mut rng.lock().unwrap(), &FIRST_AUDIO, &self.timing);
                self.registry.record_first_audio_due(slot, ms);
                tokio::time::Instant::now() + std::time::Duration::from_millis(ms)
            });
            if first_audio.is_some() {
                self.registry.voice_due(slot.nth, first_audio);
            }
            Ok(lk_agents_abi::DuplexStream::new(
                Box::new(InputSide(synthesis)) as TtsSink,
                Box::new(Output {
                    slot,
                    registry: self.registry.clone(),
                    receiver,
                    first_audio,
                    delivered: false,
                    hazard: self.registry.stream_hazard(Site::TtsNext, Site::TtsClose),
                }) as TtsSource,
            ))
        })
    }
}

/// A chunk's words, sharing its `span` seconds from `at` evenly, as Cartesia's `word_timestamps`
/// arrive through the gateway: one token per word, with a trailing space.
fn stamps(text: &str, at: f64, span: f64) -> Vec<TranscriptChunk> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let each = span / words.len().max(1) as f64;
    words
        .iter()
        .enumerate()
        .map(|(i, word)| {
            let start = at + i as f64 * each;
            TranscriptChunk {
                start_time: Some(start),
                end_time: Some(start + each),
                ..TranscriptChunk::new(format!("{word} "))
            }
        })
        .collect()
}

struct InputSide(Arc<Synthesis>);

/// The provider's input: it speaks each chunk's text, and stamps what it hears itself say of it
/// (markup taken as instructions, a cue it echoes stamped as a word, as a real voice does).
impl StreamInput<SpeechChunk> for InputSide {
    fn push(&self, chunk: SpeechChunk) -> Result<(), ProviderError> {
        let text = chunk.text;
        if text.trim().is_empty() {
            return Ok(());
        }
        let mut state = self.0.state.lock().unwrap();
        if state.held {
            if state.queued.len() >= HELD_INPUT {
                return Err(ProviderError::Local(ApiError::new(
                    "the voice is not taking its input",
                )));
            }
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

    /// Waits while the held voice's input is full, then hands `entry` over.
    fn send(
        &self,
        entry: Entry<SpeechChunk>,
    ) -> lk_agents_abi::BoxFuture<'static, Result<(), ProviderError>> {
        let input = InputSide(self.0.clone());
        Box::pin(async move {
            loop {
                let unheld = input.0.unheld.notified();
                tokio::pin!(unheld);
                unheld.as_mut().enable();
                if !input.0.full() {
                    break;
                }
                unheld.await;
            }
            match entry {
                Entry::Item(text) => input.push(text),
                Entry::Flush => input.flush(),
                Entry::EndInput => {
                    input.end_input();
                    Ok(())
                }
            }
        })
    }
}

impl StreamOutput<SynthesizedAudio> for Output {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        let fault = self.registry.stream_fault(&mut self.hazard, Site::TtsNext, self.slot);
        Box::pin(async move {
            if let Some(fault) = fault {
                return strike(fault, Site::TtsNext).await;
            }
            let next = self.receiver.recv().await.transpose();
            // Only audio waits for the first-audio latency; a failure or the end reports at once.
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

    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        let fault = self.registry.stream_fault(&mut self.hazard, Site::TtsClose, self.slot);
        Box::pin(async move {
            self.registry.voice_due(self.slot.nth, None);
            if let Some(fault) = fault {
                return self.registry.strike_close(fault, Site::TtsClose, deadline).await;
            }
            self.registry.mark_closed(self.slot);
            Closed::Clean
        })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<SynthesizedAudio> for Output {}
