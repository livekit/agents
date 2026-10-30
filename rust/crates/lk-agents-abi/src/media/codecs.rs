//! Codec and byte-pipe host contracts; native execution is supplied by the runtime.

use crate::AudioFrame;

#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct CodecFailure(pub String);

/// Container bytes decoded on one admitted native worker: input is queued without waiting for
/// the decoder, and decoded frames are awaited.
#[crate::host(class(name = "AudioDecoder"))]
pub trait AudioDecoder: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        mime: String,
        #[host(kw, default = 48000)] sample_rate: Option<u32>,
        #[host(default = 1)] num_channels: u16,
    ) -> Result<Self, CodecFailure>;

    /// Refused after `end_input`, `aclose` or a decode failure.
    #[host(call)]
    fn push(&self, data: &[u8]) -> Result<(), CodecFailure>;

    /// No more input: `next` returns `None` once everything pushed is decoded.
    #[host(call)]
    fn end_input(&self);

    /// The next decoded frame, at the requested format; `None` at the end of input.
    #[host(await)]
    async fn next(&self) -> Result<Option<AudioFrame>, CodecFailure>;

    /// Cancel decoding and observe release of the native decoder's worker admission.
    #[host(await)]
    async fn aclose(&self) -> Result<(), CodecFailure>;
}

/// A byte pipe: normal EOF drains buffered bytes; close discards them immediately.
#[crate::host(class(name = "StreamBuffer"))]
pub trait StreamBuffer: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new() -> Self;
    #[host(call)]
    fn write(&self, data: &[u8]);
    #[host(call)]
    fn end_input(&self);
    #[host(call)]
    fn close(&self);
    #[host(call, name = "__len__")]
    fn len(&self) -> usize;
    #[host(call)]
    fn read(&self, #[host(default = -1)] size: isize) -> Vec<u8>;
}

/// Native sentence tokenization shared by buffered TTS token streams.
#[crate::host(functions(pid = false))]
pub trait MediaTokenization: Send + Sync + Sized + 'static {
    #[host(function)]
    fn split_sentences(
        text: &str,
        #[host(default = 20)] min_sentence_len: usize,
        #[host(kw, default = false)] retain_format: bool,
    ) -> Result<Vec<(String, usize, usize)>, CodecFailure>;
}
