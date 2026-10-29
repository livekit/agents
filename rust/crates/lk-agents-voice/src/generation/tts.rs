//! Streaming synthesis decisions: feed ordering, completion and aligned captions.

use crate::transcription::Visible;
use lk_agents_abi::TimedString;
use lk_agents_abi::errors::ProviderError;

/// About five seconds of speculative 20 ms frames. Runtime backpressures the provider at this bound.
pub const MAX_BUFFERED_TTS_FRAMES: usize = 256;

/// The ordered operations caused by one text input.
pub enum TextEffect {
    Capture(Visible),
    Push(String),
    EndInput,
    FlushText,
}

/// Whether a subsequent source read is still needed.
pub enum Read {
    Provider,
    End,
    Failed(ProviderError),
}

pub struct StreamingTts {
    pub words: Visible,
    pub input_done: bool,
    ended: bool,
    send_error: Option<ProviderError>,
}

impl StreamingTts {
    pub fn new(words: Visible, input_done: bool) -> Self {
        Self { words, input_done, ended: false, send_error: None }
    }

    pub fn before_read(&mut self) -> Read {
        if self.ended {
            Read::End
        } else if let Some(error) = self.send_error.take() {
            Read::Failed(error)
        } else {
            Read::Provider
        }
    }

    /// The transcript delta precedes the same delta's synthesis input.
    pub fn text(&mut self, chunk: Option<Visible>) -> Vec<TextEffect> {
        match chunk {
            Some(chunk) => vec![TextEffect::Capture(chunk)],
            None => {
                self.input_done = true;
                vec![TextEffect::EndInput, TextEffect::FlushText]
            }
        }
    }

    /// Capturing a transcript delta completed; only now does the feed include its words.
    pub fn captured(&mut self, chunk: Visible) -> TextEffect {
        self.words.raw.push_str(&chunk.raw);
        self.words.text.push_str(&chunk.text);
        self.words.tags.extend(chunk.tags);
        TextEffect::Push(chunk.raw)
    }

    /// A rejected text push ends feeding. Its error follows the source read already in flight.
    pub fn push_failed(&mut self, error: ProviderError) {
        self.send_error = Some(error);
        self.input_done = true;
    }

    /// A rejected opening feed is reported before the first source read.
    pub fn opening_pushed(&mut self, error: Option<ProviderError>) -> bool {
        let accepted = error.is_none();
        self.send_error = error;
        self.input_done && accepted
    }

    pub fn source_ended(&mut self) {
        self.ended = true;
    }

    /// A pooled source can stay open after its final frame. End audio once input is complete.
    pub fn frame(&mut self, is_final: bool) -> bool {
        if is_final && self.input_done {
            self.ended = true;
            true
        } else {
            false
        }
    }
}

pub enum TranscriptEffect {
    CaptureTimed(TimedString),
    Capture(String),
    WarnUnaligned,
    Flush,
}

/// Stamped words own an aligned transcript after the first stamp. An unstamped segment falls back
/// to generated text. Closing flushes the transcript exactly once.
#[derive(Default)]
pub struct AlignedTranscript {
    pub words: String,
    pub stamped: bool,
    pub closed: bool,
}

impl AlignedTranscript {
    pub fn word(&mut self, word: &TimedString) -> TranscriptEffect {
        self.stamped = true;
        self.words.push_str(&word.text);
        TranscriptEffect::CaptureTimed(word.clone())
    }

    pub fn end(&mut self, fallback: &str) -> Vec<TranscriptEffect> {
        if std::mem::replace(&mut self.closed, true) {
            return Vec::new();
        }
        let mut effects = Vec::new();
        if !self.stamped {
            self.words.push_str(fallback);
            effects.push(TranscriptEffect::WarnUnaligned);
            effects.push(TranscriptEffect::Capture(fallback.to_owned()));
        }
        effects.push(TranscriptEffect::Flush);
        effects
    }
}

/// One synthesis request's observations. All timestamps are supplied by the executor.
pub struct SynthesisMetrics {
    provider: lk_agents_abi::ProviderId,
    pub segment: usize,
    synthesis_started_at: f64,
    started_at: f64,
    metrics: lk_agents_abi::telemetry::provider::TtsMetricState,
    metered_calls: bool,
}

impl SynthesisMetrics {
    pub fn new(
        provider: lk_agents_abi::ProviderId,
        segment: usize,
        synthesis_started_at: f64,
        started_at: f64,
    ) -> Self {
        Self {
            provider,
            segment,
            synthesis_started_at,
            started_at,
            metrics: lk_agents_abi::telemetry::provider::TtsMetricState::for_host(true),
            metered_calls: false,
        }
    }

    pub fn produced(&self) -> bool {
        self.metrics.ttfb().is_some()
    }

    pub fn note(
        &mut self,
        chunk: &lk_agents_abi::SynthesizedAudio,
        now: f64,
    ) -> Option<super::TtsStats> {
        let first = (!self.produced()).then(|| super::TtsStats {
            ttfb: Some(now - chunk.started_time.unwrap_or(self.synthesis_started_at)),
            synthesis_started_at: chunk.started_time,
        });
        self.metrics.observe(chunk, || now - self.started_at);
        first
    }

    pub fn calls(
        &mut self,
        records: Vec<lk_agents_abi::metrics::base::AgentMetrics>,
        speech_id: &str,
    ) -> Vec<lk_agents_abi::metrics::base::AgentMetrics> {
        self.metered_calls = true;
        records
            .into_iter()
            .map(|mut record| {
                if let lk_agents_abi::metrics::base::AgentMetrics::Tts(tts) = &mut record {
                    tts.speech_id = Some(speech_id.to_owned());
                }
                record
            })
            .collect()
    }

    pub fn finish(
        mut self,
        text: &str,
        now: f64,
        interrupted: bool,
        speech_id: &str,
    ) -> Option<lk_agents_abi::metrics::base::AgentMetrics> {
        if self.metered_calls {
            return None;
        }
        let context = lk_agents_abi::telemetry::provider::Context {
            timestamp: now,
            label: self.provider.label.clone(),
            cancelled: interrupted,
            metadata: Some(self.provider.metadata()),
        };
        let mut record =
            self.metrics.finish(now, self.started_at, text, 0, 0, 0.0, false, context)?;
        record.speech_id = Some(speech_id.to_owned());
        Some(lk_agents_abi::metrics::base::AgentMetrics::Tts(record))
    }
}
