//! Every media format the agent speaks: container decode to PCM, ogg/opus encode, image encode,
//! and resampling, kept below the purity line (nothing in tiers 0/1 may depend on this crate).
//! Pure Rust by default; `blingfire`, `images`, `ffmpeg` and `resample` add the C/C++ parts.
//! On the decode path a drift flush must not tear the [`Decoder`] down: the watchdog fires
//! mid-container (`tts/tts.py:1300-1310`).

#[cfg(feature = "python")]
pub mod audio_emitter;
#[cfg(feature = "blingfire")]
pub mod blingfire;
pub mod decoder;
pub mod error;
pub mod frame_resampler;
#[cfg(feature = "images")]
pub mod images;
#[cfg(feature = "ffmpeg")]
pub mod ffmpeg;
pub mod mime;
pub mod mixer;
pub mod recorder;
#[cfg(feature = "resample")]
pub mod soxr;
pub mod stream_buffer;
pub mod wav;

pub use decoder::Decoder;
pub use error::{CodecError, Result};
pub use frame_resampler::{FrameResampler, OnChange};
pub use lk_agents_abi::{LinearResampler, Quality, ResampleError, Resampler};
pub use mime::{DecodePlan, UnknownMime, decode_plan, mime_to_av_format};
pub use mixer::{Mixer, MixerCounters, StreamId};
#[cfg(feature = "resample")]
pub use soxr::SoxrResampler;
pub use stream_buffer::StreamBuffer;
pub use wav::{WavDecoder, WavSegment};

/// The one place a resampler is chosen: libsoxr under `resample`, [`LinearResampler`] otherwise.
pub fn new_resampler(
    input_rate: u32,
    output_rate: u32,
    num_channels: u32,
    quality: Quality,
) -> std::result::Result<Box<dyn Resampler>, ResampleError> {
    #[cfg(feature = "resample")]
    {
        SoxrResampler::new(input_rate, output_rate, num_channels, quality).map(|r| Box::new(r) as Box<dyn Resampler>)
    }
    #[cfg(not(feature = "resample"))]
    {
        let _ = quality;
        LinearResampler::new(input_rate, output_rate, num_channels).map(|r| Box::new(r) as Box<dyn Resampler>)
    }
}

#[cfg(feature = "ffmpeg")]
pub use ffmpeg::OggOpusEncoder;

/// Interleaved `i16` as little-endian bytes, as `rtc.AudioFrame` and the emitter's framer take.
pub fn pcm_bytes(pcm: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub num_channels: u16,
}

impl AudioFormat {
    pub fn new(sample_rate: u32, num_channels: u16) -> Self {
        AudioFormat { sample_rate, num_channels }
    }
}
