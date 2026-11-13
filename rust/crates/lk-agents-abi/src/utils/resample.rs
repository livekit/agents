/// The discriminants are the protobuf ordinals passed to libsoxr as the recipe.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Quality {
    Quick = 0,
    Low = 1,
    /// `rtc.AudioResampler`'s default.
    #[default]
    Medium = 2,
    High = 3,
    VeryHigh = 4,
}

#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, thiserror::Error)]
pub enum ResampleError {
    #[error("failed to create the resampler: {0}")]
    Create(String),
    #[error("failed to resample: {0}")]
    Process(String),
    /// The input interleaves by channel; a partial frame would silently shear the channels apart.
    #[error("input of {len} samples is not a whole number of {num_channels}-channel frames")]
    PartialFrame { len: usize, num_channels: u32 },
}

/// `utils.media.AudioConverter`: frames of any rate and channel count, converted to one format
/// (resampled, and averaged to mono or copied across channels). A frame already in that format
/// passes through untouched; a change of input format ends the previous stream first.
#[crate::host(class(name = "AudioConverter", module = "livekit.agents.utils.media", pid = false))]
pub trait AudioConverter: Send + Sized + 'static {
    #[host(new)]
    fn new(sample_rate: u32, num_channels: u16) -> Self;

    /// The converted audio `frame` completes, after the tail of a stream it ends.
    #[host(call)]
    fn push(&mut self, frame: crate::AudioFrame) -> Result<Vec<crate::AudioFrame>, ResampleError>;

    /// The audio still held by the resampler, at the end of a segment or stream.
    #[host(call)]
    fn flush(&mut self) -> Result<Vec<crate::AudioFrame>, ResampleError>;
}
