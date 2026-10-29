//! `tts.StreamPacerWrapper`'s pacing: sentences batched by the synthesis duration observed so far,
//! on the caller's clock.

/// Paces a synthesis's sentences: the first at once, the rest in batches once the audio already
/// synthesized is nearly played.
#[crate::host(class(name = "SentencePacer"))]
pub trait SentencePacer: Send + Sized + 'static {
    #[host(new)]
    fn new(minimum_audio: f64, maximum_text: usize) -> Self;

    #[host(call)]
    fn push(&mut self, text: String);

    #[host(call)]
    fn end(&mut self);

    /// Text to emit, remaining audio for diagnostics, and next wakeup (None means done).
    #[host(call)]
    fn poll(&mut self, now: f64, audio: f64) -> (Option<String>, f64, Option<f64>);
}
