//! Media formats for the agent: container decode to PCM, ogg/opus encode, image encode, filters
//! and libsoxr resampling. `images` and `ffmpeg` are optional features.

pub mod audio_byte_stream;
#[cfg(test)]
mod audio_byte_stream_tests;
pub mod audio_decoder;
pub mod emitter;
pub mod error;
#[cfg(feature = "ffmpeg")]
pub mod ffmpeg;
pub mod filters;
#[cfg(feature = "images")]
pub mod images;
pub mod media_queue;
pub mod mime;
pub mod mixer;
pub mod pauses;
pub mod pool;
pub mod recorder;
pub mod resample;
pub mod source;
mod soxr;
pub mod wav;

pub use audio_byte_stream::{AudioByteStream, Frame};
pub use audio_decoder::AudioDecoder;
pub use error::{CodecError, Result};
pub use filters::{ExpFilter, MovingAverage};
pub use lk_agents_abi::media::codecs::Container;
pub use lk_agents_abi::{Quality, ResampleError};
pub use mime::{AudioDecoding, UnknownMime, audio_decoding, mime_container};
pub use resample::{AudioResampler, OnChange, Resampler};
pub use wav::{WavDecoder, WavSegment};

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

/// Little-endian bytes as interleaved `i16`, the inverse of [`pcm_bytes`]; an odd trailing byte
/// is dropped.
pub fn pcm_samples(bytes: &[u8]) -> Vec<i16> {
    bytes.as_chunks::<2>().0.iter().copied().map(i16::from_le_bytes).collect()
}

/// Interleaved PCM averaged to mono, truncating toward zero like the recorder's sample conversion.
/// A zero channel header is treated as mono; an incomplete channel group is discarded.
pub fn downmix_mono(samples: &[i16], num_channels: u16) -> Vec<i16> {
    let channels = usize::from(num_channels.max(1));
    if channels == 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(channels)
        .map(|frame| {
            (frame.iter().map(|&sample| i64::from(sample)).sum::<i64>() / channels as i64) as i16
        })
        .collect()
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
pub use mixer::{Mixer, MixerCounters, StreamId};
