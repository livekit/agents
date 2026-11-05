//! `tts.SentenceStreamPacer`: sentences released as the audio already synthesized plays out, and
//! the host sentence streams the pacer and the sentence adapter read.

use std::sync::Arc;

use crate::errors::{HostError, ProviderError};

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
    #[host(await, name = "_core_next")]
    fn next(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Option<String>, ProviderError>>;
    #[host(await, name = "aclose")]
    fn aclose(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, Result<(), HostError>>;
}

/// How much audio a synthesis has pushed so far: a `tts.AudioEmitter`.
#[crate::host(native(refused = "expected a tts.AudioEmitter"))]
pub trait PushedAudio: Send + Sync {
    /// Seconds of audio pushed, every segment included.
    fn pushed_duration(&self) -> f64;
}

/// `SentenceStreamPacer.wrap()`: a host sentence stream's sentences, paced by the audio its
/// emitter pushed. The first sentence goes at once; the next batch once the audio stopped
/// growing and less than `min_remaining_audio` seconds of it is left to play, at most
/// `max_text_length` characters of whole sentences joined by a space.
#[crate::host(class(name = "_core_PacedSentences"))]
pub trait PacedSentences: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        sentences: Arc<dyn HostSentenceStream>,
        audio: Arc<dyn PushedAudio>,
        min_remaining_audio: f64,
        max_text_length: usize,
    ) -> Self;

    /// The next batch, `None` once the input ended and every sentence went; the host stream's
    /// failure is raised as it was.
    #[host(await)]
    async fn next(&self, cx: crate::telemetry::Cx) -> Result<Option<String>, ProviderError>;

    /// Stop reading and close the host stream.
    #[host(await)]
    async fn aclose(&self, cx: crate::telemetry::Cx);
}
