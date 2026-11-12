use crate::errors::{HostError, ProviderError};
use crate::media::emitter::EmitterError;

/// A host `tokenize.SentenceStream`: text in, sentences out. The native pacer and sentence
/// adapter read it; its calls run on the host loop, in order.
#[crate::host(proxy(host = "livekit.agents.tokenize.SentenceStream"))]
pub trait HostSentenceStream: Send + Sync {
    #[host(notify, name = "push_text")]
    fn push_text(&self, text: String);
    #[host(notify, name = "flush")]
    fn flush(&self);
    #[host(notify, name = "end_input")]
    fn end_input(&self);
    /// The next sentence, `None` once the stream ended.
    #[host(await, iter)]
    fn next(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Option<crate::tokenize::TokenData>, ProviderError>>;
    #[host(await, name = "aclose")]
    fn aclose(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, Result<(), HostError>>;
}

/// A `tts.AudioEmitter` as native code pushes a synthesis into it and reads how much it pushed.
#[crate::host(native(refused = "expected a tts.AudioEmitter"))]
pub trait PushedAudio: Send + Sync {
    /// Seconds of audio pushed, every segment included.
    fn pushed_duration(&self) -> f64;
    fn start_segment(&self, segment_id: String) -> Result<(), EmitterError>;
    fn end_segment(&self) -> Result<(), EmitterError>;
    fn push_frame(&self, frame: crate::AudioFrame) -> Result<(), EmitterError>;
    /// The words the next frames cover; none is a no-op.
    fn push_timed_transcript(&self, words: Vec<crate::TranscriptChunk>)
    -> Result<(), EmitterError>;
    fn report_usage(
        &self,
        usage: Option<crate::usage::TokenUsage>,
        characters_count: Option<u64>,
    ) -> Result<(), EmitterError>;
}

/// Paces the text sent to a TTS: the first chunk goes at once; each later batch once the audio
/// stopped growing and at most ``min_remaining_audio`` seconds of it are left to play, of whole
/// chunks joined by a space up to ``max_text_length`` characters (never past the provider's
/// request limit). It wastes less on interruptions and gives the TTS more context.
#[crate::host(record(
    module = "stream_pacer",
    path = "livekit.agents.tts",
    dataclass,
    extract,
    emit
))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SentenceStreamPacer {
    /// The audio left, in seconds, below which the next batch goes.
    #[host(default = 5.0)]
    pub min_remaining_audio: f64,
    /// The longest batch sent at once, in characters.
    #[host(default = 300)]
    pub max_text_length: usize,
}

/// `SentenceStreamPacer()`'s defaults: what `TTSOptions` paces with.
impl Default for SentenceStreamPacer {
    fn default() -> Self {
        SentenceStreamPacer { min_remaining_audio: 5.0, max_text_length: 300 }
    }
}
