//! Sample-rate conversion through libsoxr, with PCM-stream and audio-frame adapters.

mod frame;
mod stream;

pub use crate::soxr::Resampler;
pub use frame::{FrameResampler, OnChange};
pub(crate) use stream::PcmResampler;
