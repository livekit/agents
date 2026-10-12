//! One error type for both halves, `Clone + PartialEq` so tests assert on values.

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CodecError {
    /// `decoder.py:_consume_riff`'s `ValueError`.
    #[error("invalid WAV file: missing RIFF/WAVE: {0:x?}")]
    MissingRiffWave(Vec<u8>),

    #[error("unsupported WAV bits per sample: {0} (only 16-bit PCM supported)")]
    UnsupportedWavBits(u16),

    /// `fmt `'s `wFormatTag` was not 1 (`WAVE_FORMAT_PCM`). 3 is float, 0xFFFE is EXTENSIBLE.
    #[error("unsupported WAV audio format: {0}")]
    UnsupportedWavFormat(u16),

    #[error("invalid WAV file: data chunk before fmt chunk")]
    WavDataBeforeFmt,

    #[error("truncated WAV fmt chunk: {0} bytes, need at least 16")]
    ShortWavFmt(u32),

    /// Without the `ffmpeg` feature that is everything except WAV.
    #[error(
        "unsupported mime type {mime:?}: {reason}. Enable the `ffmpeg` feature of `lk-agents-media` \
         for container formats other than WAV."
    )]
    UnsupportedMime { mime: String, reason: &'static str },

    /// Only a rate pair libsoxr itself rejects.
    #[error("cannot resample {from} Hz to {to} Hz: {detail}")]
    Resample { from: u32, to: u32, detail: String },

    /// Carries `av_strerror`'s text where there is one.
    #[error("ffmpeg: {op} failed: {detail}")]
    Ffmpeg { op: &'static str, detail: String },
}

pub type Result<T> = std::result::Result<T, CodecError>;
