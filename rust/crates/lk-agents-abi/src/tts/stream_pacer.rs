//! `tts.SentenceStreamPacer`: sentences released as the audio already synthesized plays out, and
//! the host sentence streams the pacer and the sentence adapter read.

use std::sync::Arc;

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

/// `tts.SentenceStreamPacer`: the first sentence goes at once; the next batch once the audio
/// stopped growing and less than `min_remaining_audio` seconds of it is left to play, at most
/// `max_text_length` characters of whole sentences joined by a space.
#[crate::host(record(
    module = "stream_pacer",
    extract,
    emit,
    host_type = "livekit.agents.tts.SentenceStreamPacer"
))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SentenceStreamPacer {
    pub min_remaining_audio: f64,
    pub max_text_length: usize,
}

/// `SentenceStreamPacer()`'s defaults, the pacer `text_pacing=True` asks for.
impl Default for SentenceStreamPacer {
    fn default() -> Self {
        SentenceStreamPacer { min_remaining_audio: 5.0, max_text_length: 300 }
    }
}

/// `SentenceStreamPacer.wrap()`: a host sentence stream's sentences, paced by the audio its
/// emitter pushed.
#[crate::host(class(name = "_core_PacedSentences"))]
pub trait PacedSentences: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        sent_stream: Arc<dyn HostSentenceStream>,
        audio_emitter: Arc<dyn PushedAudio>,
        #[host(kw)] pacer: SentenceStreamPacer,
    ) -> Self;

    /// The next batch, `None` once the input ended and every sentence went; the host stream's
    /// failure is raised as it was.
    #[host(await)]
    async fn next(&self, cx: crate::telemetry::Cx) -> Result<Option<String>, ProviderError>;

    /// Stop reading and close the host stream.
    #[host(await)]
    async fn aclose(&self, cx: crate::telemetry::Cx);
}

/// `tts.StreamPacerWrapper`: a sentence stream paced by the audio its emitter pushed.
#[crate::host(facade(path = "livekit.agents.tts", base = "tokenize.SentenceStream",
                     native = PacedSentences))]
pub trait StreamPacerWrapper {
    /// Text pushed to the wrapper goes here.
    #[host(delegate)]
    type SentStream: HostSentenceStream;

    #[host(new)]
    fn new(
        sent_stream: Self::SentStream,
        audio_emitter: Arc<dyn PushedAudio>,
        #[host(kw)] pacer: SentenceStreamPacer,
    ) -> Self;

    #[host(via = SentStream)]
    fn push_text(&self, text: String);
    #[host(via = SentStream)]
    fn flush(&self);
    #[host(via = SentStream)]
    fn end_input(&self);

    #[host(none_ends_iteration, wrap = crate::tokenize::TokenData::token)]
    async fn __anext__(&self) -> Option<crate::tokenize::TokenData>;
    async fn aclose(&self);
}
