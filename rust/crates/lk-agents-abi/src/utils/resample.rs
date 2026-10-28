//! Resampler contracts. Implementations and fallback selection live in `lk-agents-media`.

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

/// `flush` drains the tail and resets, as `rtc.AudioResampler.flush` does.
pub trait Resampler: Send {
    fn input_rate(&self) -> u32;
    fn output_rate(&self) -> u32;
    fn num_channels(&self) -> u32;
    fn push(&mut self, input: &[i16]) -> Result<&[i16], ResampleError>;
    fn flush(&mut self) -> Result<&[i16], ResampleError>;
}
