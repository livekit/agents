use crate::AudioFrame;

/// An audio container a decoder reads, named as FFmpeg's demuxer is.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    Mp3,
    Aac,
    Flac,
    Wav,
    Ogg,
    Webm,
    Mp4,
}

/// `utils.media.AudioStreamDecoder`: container bytes decoded on one admitted native worker into
/// PCM frames, which come out as soon as they are decoded. Input is queued without waiting for
/// the decoder. One decoder decodes one stream. A failure (an unknown format, no worker admitted,
/// invalid audio) is logged and ends the stream.
#[crate::host(class(name = "AudioStreamDecoder", module = "livekit.agents.utils.media"))]
pub trait AudioDecoder: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// `sample_rate` is the output rate (the source's when `None`), `format` the input's media
    /// type (`"audio/mpeg"`), probed when `None`.
    #[host(new)]
    fn new(
        env: &Self::Env,
        #[host(kw, default = 48000)] sample_rate: Option<u32>,
        #[host(default = 1)] num_channels: u16,
        #[host(default = None)] format: Option<String>,
    ) -> Self;

    /// Queue `chunk`; ignored after `end_input`, `aclose` or the end of the stream.
    #[host(call)]
    fn push(&self, chunk: &[u8]);

    /// No more input: the stream ends once everything pushed is decoded.
    #[host(call)]
    fn end_input(&self);

    /// The next decoded frame, at the requested format; `None` ends the stream.
    #[host(next)]
    async fn next(&self) -> Option<AudioFrame>;

    /// Cancel decoding and observe release of the native decoder's worker admission.
    #[host(await)]
    async fn aclose(&self);
}
