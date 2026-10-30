//! Streaming synthesis decisions: feed ordering, completion and aligned captions.

use crate::transcription::{Heard, Visible};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{TimedString, Ts};

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
#[derive(Clone, Default)]
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
    metrics: lk_agents_core::telemetry::provider::TtsMetricState,
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
            metrics: lk_agents_core::telemetry::provider::TtsMetricState::new(true),
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
        let context = lk_agents_core::telemetry::provider::Context {
            timestamp: now,
            label: self.provider.label.clone(),
            cancelled: interrupted,
            metadata: Some(self.provider.metadata()),
        };
        let mut record = self.metrics.finish(now, self.started_at, text, context)?;
        record.speech_id = Some(speech_id.to_owned());
        Some(lk_agents_abi::metrics::base::AgentMetrics::Tts(record))
    }
}

/// The outcome of synthesis that did not produce a completed segment.
pub struct Lost {
    pub failure: Option<crate::agent_session::DriverError>,
    pub lost: bool,
    pub opening: Visible,
    pub synchronized: Option<String>,
    pub played: bool,
    /// What the sink confirmed of the failed attempt's audio: its frames and playhead.
    pub heard: Heard,
}

/// The accumulated input of a synthesis attempt, preserved for a retry.
pub struct SynthesisAttempt {
    /// The reply's outer retry of its synthesis, on the session's TTS connection options.
    pub retry: lk_agents_core::attempts::Attempts,
    pub opening: Visible,
    pub input_done: bool,
    synchronized: Option<String>,
    heard: Heard,
    produced: bool,
}

pub enum AttemptFailure {
    Retry(ProviderError),
    Stop(Lost),
}

pub enum RetryOutcome {
    Again,
    Interrupted,
    Exhausted,
}

impl SynthesisAttempt {
    pub fn new(options: &lk_agents_abi::ApiConnectOptions, opening: Visible, now: Ts) -> Self {
        Self {
            retry: lk_agents_core::attempts::Attempts::retrying(options, now),
            opening,
            input_done: false,
            synchronized: None,
            heard: Heard::default(),
            produced: false,
        }
    }

    pub fn before_open(&self, interrupted: bool, closing: bool) -> Option<Lost> {
        (interrupted || closing).then(|| self.stopped())
    }

    pub fn stream_failed(
        &mut self,
        opening: Visible,
        input_done: bool,
        synchronized: Option<String>,
        heard: Option<Heard>,
    ) {
        self.opening = opening;
        self.input_done = input_done;
        self.synchronized = synchronized;
        self.heard = heard.unwrap_or_default();
    }

    pub fn failed(
        &mut self,
        error: crate::agent_session::DriverError,
        produced: bool,
    ) -> AttemptFailure {
        self.produced = produced;
        match error {
            crate::agent_session::DriverError::Provider(error) => {
                if produced {
                    self.retry.delivered();
                }
                AttemptFailure::Retry(error)
            }
            error => AttemptFailure::Stop(Lost {
                failure: Some(error),
                lost: false,
                opening: self.opening.clone(),
                synchronized: self.synchronized.clone(),
                played: produced,
                heard: self.heard.clone(),
            }),
        }
    }

    pub fn after_retry(&self, outcome: RetryOutcome) -> Option<Lost> {
        match outcome {
            RetryOutcome::Again => None,
            RetryOutcome::Interrupted => Some(self.stopped()),
            RetryOutcome::Exhausted => Some(Lost {
                failure: None,
                lost: true,
                opening: self.opening.clone(),
                synchronized: self.synchronized.clone(),
                played: self.produced,
                heard: self.heard.clone(),
            }),
        }
    }

    fn stopped(&self) -> Lost {
        Lost {
            failure: None,
            lost: false,
            opening: self.opening.clone(),
            synchronized: None,
            played: false,
            heard: Heard::default(),
        }
    }
}

/// How a lost synthesis settles the rest of its segment.
#[derive(Default)]
pub struct SynthesisLoss {
    pending: Option<LossText>,
    /// The sink's account of the audio that did play: the segment is voiced if any reached it.
    sink: Heard,
}

enum LossText {
    Unheard,
    Stamped(String),
    Generated { played: bool, synchronized: Option<String> },
}

pub enum LossEvent {
    Failed { lost: Box<Lost>, aligned: Option<AlignedTranscript> },
    Drained(Visible),
}

pub enum LossEffect {
    Fail(crate::agent_session::DriverError),
    ReleaseVoice,
    CutShort,
    Capture(String),
    Drain { opening: Visible, forward_text: bool },
    Heard(Heard),
}

impl SynthesisLoss {
    pub fn handle(&mut self, event: LossEvent) -> Vec<LossEffect> {
        match event {
            LossEvent::Failed { lost, aligned } => {
                let lost = *lost;
                if let Some(error) = lost.failure {
                    return vec![LossEffect::Fail(error)];
                }
                if !lost.lost {
                    self.pending = Some(LossText::Unheard);
                    return vec![LossEffect::Drain { opening: lost.opening, forward_text: false }];
                }
                self.sink = lost.heard;
                let mut effects = vec![LossEffect::ReleaseVoice, LossEffect::CutShort];
                match aligned {
                    Some(aligned) if aligned.stamped => {
                        // Once stamps were accepted, model text can never become their fallback.
                        self.pending = Some(LossText::Stamped(aligned.words));
                        effects
                            .push(LossEffect::Drain { opening: lost.opening, forward_text: false });
                    }
                    aligned => {
                        if aligned.is_some_and(|a| !a.closed) {
                            effects.push(LossEffect::Capture(lost.opening.text.clone()));
                        }
                        self.pending = Some(LossText::Generated {
                            played: lost.played,
                            synchronized: lost.synchronized,
                        });
                        effects
                            .push(LossEffect::Drain { opening: lost.opening, forward_text: true });
                    }
                }
                effects
            }
            LossEvent::Drained(words) => {
                let Some(pending) = self.pending.take() else {
                    return Vec::new();
                };
                let text = match pending {
                    LossText::Unheard => return vec![LossEffect::Heard(Heard::default())],
                    LossText::Stamped(text) => text,
                    LossText::Generated { played: true, synchronized } => {
                        synchronized.unwrap_or(words.text)
                    }
                    LossText::Generated { played: false, .. } => String::new(),
                };
                let sink = std::mem::take(&mut self.sink);
                vec![LossEffect::Heard(Heard { text, complete: true, ..sink })]
            }
        }
    }
}
