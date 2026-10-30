//! Owned PCM, not a [`MediaRef`](crate::MediaRef): native VAD and TTS read the samples and frames
//! live milliseconds, so the binding copies; the [`Arc`] makes the STT/VAD fan-out free.

use std::sync::Arc;

/// Interleaved signed 16-bit little-endian PCM, with a one-dimensional contiguous payload.
/// `samples_per_channel` is carried, not derived: a plugin frame may disagree with its data length.
/// The field names are `rtc.AudioFrame(..)`'s keyword arguments.
#[crate::host(handwritten(extract, emit, host = "livekit.rtc.AudioFrame", py))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFrame {
    /// `bytes`, not a list of ints (which `rtc.AudioFrame` would read as garbage).
    pub data: Arc<[u8]>,
    pub sample_rate: u32,
    pub num_channels: u16,
    pub samples_per_channel: u32,
}

impl AudioFrame {
    pub fn new(
        data: impl Into<Arc<[u8]>>,
        sample_rate: u32,
        num_channels: u16,
        samples_per_channel: u32,
    ) -> Self {
        AudioFrame { data: data.into(), sample_rate, num_channels, samples_per_channel }
    }

    /// Zero when `sample_rate` is zero.
    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        f64::from(self.samples_per_channel) / f64::from(self.sample_rate)
    }
}

/// `utils.AudioBuffer`: a slice covers one frame or a list.
pub type AudioBuffer<'a> = &'a [AudioFrame];

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: computing duration from `data.len()` instead of the declared sample count, or a
    // divide-by-zero panic on a host-supplied zero sample rate.
    #[test]
    fn duration_uses_the_declared_sample_count() {
        let f = AudioFrame::new(vec![0u8; 4], 48_000, 1, 480);
        assert!((f.duration() - 0.01).abs() < 1e-12);
        assert_eq!(AudioFrame::new(vec![0u8; 4], 0, 1, 480).duration(), 0.0);
    }
}
