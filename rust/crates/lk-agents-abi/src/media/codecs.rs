//! Codec and byte-pipe host contracts; native execution is supplied by the runtime.

use std::sync::Arc;

#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct CodecFailure(pub String);

/// A decoder has PCM or a failure ready for its loop to drain.
#[crate::host(proxy(host = "typing.Callable"))]
pub trait DecoderOutput: Send + Sync {
    #[host(notify, name = "__call__")]
    fn ready(&self);
}

/// Incremental container bytes to little-endian interleaved PCM.
#[crate::host(class(name = "AudioDecoder"))]
pub trait AudioDecoder: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(
        mime: String,
        #[host(kw, default = 48000)] sample_rate: Option<u32>,
        #[host(default = 1)] num_channels: u16,
    ) -> Result<Self, CodecFailure>;

    #[host(getter)]
    fn output_format(&self) -> Option<(u32, u16)>;

    #[host(call, detach)]
    fn push(&self, data: &[u8]) -> Result<Vec<u8>, CodecFailure>;

    /// Blocks only the invoking native thread while the admitted decoder drains.
    #[host(call, detach)]
    fn flush(&self) -> Result<Vec<u8>, CodecFailure>;

    #[host(call)]
    fn set_on_output(&self, #[host(default = None)] on_output: Option<Arc<dyn DecoderOutput>>);

    /// Cancel pending input and observe release of the native decoder's worker admission.
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
    #[host(call, detach)]
    fn read(&self, #[host(default = -1)] size: isize) -> Vec<u8>;
}

/// Native sentence tokenization shared by buffered TTS token streams.
#[crate::host(functions(pid = false))]
pub trait MediaTokenization: Send + Sync + Sized + 'static {
    #[host(function, detach)]
    fn split_sentences(
        text: &str,
        #[host(default = 20)] min_sentence_len: usize,
        #[host(kw, default = false)] retain_format: bool,
    ) -> Result<Vec<(String, usize, usize)>, CodecFailure>;
}
