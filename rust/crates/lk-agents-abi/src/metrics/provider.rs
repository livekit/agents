//! The provider metric folds the host's streams drive; the implementation is `lk-agents-core`'s
//! `metrics::provider`.

use crate::metrics::base::{LlmMetrics, SttMetrics, TtsMetrics};
use crate::{LlmChunk, SpeechEvent, SynthesizedAudio};

/// A `RECOGNITION_USAGE` event without its usage.
#[crate::host(error(kind = "AssertionError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("recognition_usage must be provided for RECOGNITION_USAGE event")]
pub struct RecognitionUsageMissing;

#[crate::host(functions(pid = false))]
pub trait ProviderMetrics {
    /// `STT.recognize`'s metrics for one batch recognition, stamped at `timestamp` (unix seconds)
    /// with the provider's label, model and provider name.
    #[host(function)]
    #[allow(clippy::too_many_arguments, reason = "the host passes each stamp field by name")]
    fn stt_batch_metrics(
        event: SpeechEvent,
        start: f64,
        now: f64,
        audio_durations: Vec<f64>,
        timestamp: f64,
        label: String,
        model: String,
        provider: String,
    ) -> SttMetrics;

    /// `RecognizeStream`'s metrics for one event, and whether it resets the retry count.
    #[host(function)]
    fn stt_stream_metrics(
        event: SpeechEvent,
        timestamp: f64,
        label: String,
        model: String,
        provider: String,
    ) -> Result<(Option<SttMetrics>, bool), RecognitionUsageMissing>;
}

/// `LLMStream`'s metric fold as the host drives it.
#[crate::host(class(name = "LlmMetricState", pid = false))]
pub trait LlmMetricState: Send + Sized + 'static {
    /// `label`, `model` and `provider`: the provider's, stamped on the record.
    #[host(new)]
    fn new(start: f64, label: String, model: String, provider: String) -> Self;

    /// Whether `chunk` is the first with a response.
    #[host(call)]
    fn observe(&mut self, now: f64, chunk: LlmChunk) -> bool;

    /// `None` for a failed attempt or one that yielded no chunk; `timestamp`: unix seconds.
    #[host(call)]
    fn finish(&self, now: f64, failed: bool, timestamp: f64) -> Option<LlmMetrics>;
}

/// The TTS streams' metric fold as the host drives it.
#[crate::host(class(name = "TtsMetricState", pid = false))]
pub trait TtsMetricState: Send + Sized + 'static {
    /// `label`, `model` and `provider`: the provider's, stamped on each record.
    #[host(new)]
    fn new(streamed: bool, label: String, model: String, provider: String) -> Self;

    /// A streamed segment's text, in the order its audio is finished.
    #[host(call)]
    fn queue_text(&mut self, text: String);

    /// Whether `audio` ends its segment.
    #[host(call)]
    fn observe(&mut self, now: f64, start: f64, audio: SynthesizedAudio) -> bool;

    /// The record of a finished request or segment, then the figures reset; `timestamp`: unix
    /// seconds.
    #[host(call)]
    fn finish(
        &mut self,
        now: f64,
        start: f64,
        failed: bool,
        batch_text: String,
        timestamp: f64,
    ) -> Option<TtsMetrics>;
}
