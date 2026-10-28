//! PCM sample arithmetic and the fixed-size mono window behind `utils.audio`.
//! Resampling is supplied by the runtime before a frame enters this window.

use lk_agents_abi::AudioFrame;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AudioError {
    #[error("buffer size must be positive")]
    EmptyBuffer,
    #[error("audio buffer allocation failed")]
    Allocation,
    #[error("sample rate and channel count must be positive")]
    InvalidFormat,
    #[error("duration must be finite and nonnegative")]
    InvalidDuration,
    #[error("audio sample count exceeds the configured limit ({limit})")]
    SampleLimit { limit: usize },
    #[error("PCM data length does not match the declared sample count")]
    InvalidPcm,
    #[error("frame samples ({samples}) are greater than the buffer size ({capacity})")]
    FrameTooLarge { samples: usize, capacity: usize },
}

/// A fixed allocation: advancing the oldest sample never moves retained PCM.
pub struct AudioArrayBuffer {
    samples: Vec<i16>,
    head: usize,
    len: usize,
}

impl AudioArrayBuffer {
    /// The owner admits the requested window before construction. Allocation failure is an error.
    pub fn new(buffer_size: usize, max_samples: usize) -> Result<Self, AudioError> {
        if buffer_size == 0 {
            return Err(AudioError::EmptyBuffer);
        }
        if buffer_size > max_samples {
            return Err(AudioError::SampleLimit { limit: max_samples });
        }
        let mut samples = Vec::new();
        samples.try_reserve_exact(buffer_size).map_err(|_| AudioError::Allocation)?;
        samples.resize(buffer_size, 0);
        Ok(Self { samples, head: 0, len: 0 })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append one frame after rate normalization; downmix with floor division, as NumPy does.
    pub fn push_frame(&mut self, frame: &AudioFrame) -> Result<usize, AudioError> {
        if frame.sample_rate == 0 || frame.num_channels == 0 {
            return Err(AudioError::InvalidFormat);
        }
        let channels = usize::from(frame.num_channels);
        let count = frame.samples_per_channel as usize;
        let frame_bytes = channels * 2;
        if count.checked_mul(frame_bytes) != Some(frame.data.len()) {
            return Err(AudioError::InvalidPcm);
        }
        self.admit_frame(count)?;
        let samples = frame.data.chunks_exact(frame_bytes).map(|pcm| {
            if channels == 1 {
                return i16::from_le_bytes([pcm[0], pcm[1]]);
            }
            let sum: i64 =
                pcm.chunks_exact(2).map(|s| i64::from(i16::from_le_bytes([s[0], s[1]]))).sum();
            (sum.div_euclid(channels as i64)) as i16
        });
        let tail = (self.head + self.len) % self.samples.len();
        let first = count.min(self.samples.len() - tail);
        let (left, right) = self.samples.split_at_mut(tail);
        for (output, sample) in
            right[..first].iter_mut().chain(&mut left[..count - first]).zip(samples)
        {
            *output = sample;
        }
        self.len += count;
        Ok(count)
    }

    /// Append one normalized mono chunk, preserving the previous window if it is oversized.
    pub fn push(&mut self, samples: &[i16]) -> Result<usize, AudioError> {
        self.admit_frame(samples.len())?;
        let tail = (self.head + self.len) % self.samples.len();
        let first = samples.len().min(self.samples.len() - tail);
        self.samples[tail..tail + first].copy_from_slice(&samples[..first]);
        self.samples[..samples.len() - first].copy_from_slice(&samples[first..]);
        self.len += samples.len();
        Ok(samples.len())
    }

    fn admit_frame(&mut self, count: usize) -> Result<(), AudioError> {
        let capacity = self.samples.len();
        if count > capacity {
            return Err(AudioError::FrameTooLarge { samples: count, capacity });
        }
        self.shift((self.len + count).saturating_sub(capacity));
        Ok(())
    }

    pub fn shift(&mut self, size: usize) {
        let size = size.min(self.len);
        self.head = (self.head + size) % self.samples.len();
        self.len -= size;
    }

    /// An independent snapshot in timeline order; changing it cannot mutate the window.
    pub fn read(&self) -> Vec<i16> {
        let mut output = Vec::with_capacity(self.len);
        let first = self.len.min(self.samples.len() - self.head);
        output.extend_from_slice(&self.samples[self.head..self.head + first]);
        output.extend_from_slice(&self.samples[..self.len - first]);
        output
    }

    pub fn read_bytes(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(self.len * 2);
        let first = self.len.min(self.samples.len() - self.head);
        for sample in self.samples[self.head..self.head + first]
            .iter()
            .chain(&self.samples[..self.len - first])
        {
            output.extend_from_slice(&sample.to_le_bytes());
        }
        output
    }

    pub fn reset(&mut self) {
        self.head = 0;
        self.len = 0;
    }
}

/// Bound the allocation before computing PCM bytes. Duration truncates to whole sample frames.
pub fn silence_frame(
    duration: f64,
    sample_rate: u32,
    num_channels: u16,
    max_samples: usize,
) -> Result<AudioFrame, AudioError> {
    if !duration.is_finite() || duration < 0.0 {
        return Err(AudioError::InvalidDuration);
    }
    if sample_rate == 0 || num_channels == 0 {
        return Err(AudioError::InvalidFormat);
    }
    let frames = (duration * f64::from(sample_rate)).trunc();
    let limit = (max_samples / usize::from(num_channels)).min(u32::MAX as usize);
    if frames > limit as f64 {
        return Err(AudioError::SampleLimit { limit: max_samples });
    }
    let frames = frames as u32;
    silent_samples(frames, sample_rate, num_channels, max_samples)
}

pub fn silence_frame_like(
    frame: &AudioFrame,
    max_samples: usize,
) -> Result<AudioFrame, AudioError> {
    silent_samples(frame.samples_per_channel, frame.sample_rate, frame.num_channels, max_samples)
}

fn silent_samples(
    frames: u32,
    sample_rate: u32,
    num_channels: u16,
    max_samples: usize,
) -> Result<AudioFrame, AudioError> {
    if sample_rate == 0 || num_channels == 0 {
        return Err(AudioError::InvalidFormat);
    }
    let count = (frames as usize)
        .checked_mul(usize::from(num_channels))
        .filter(|count| *count <= max_samples)
        .ok_or(AudioError::SampleLimit { limit: max_samples })?;
    let bytes = count.checked_mul(2).ok_or(AudioError::SampleLimit { limit: max_samples })?;
    let mut data = Vec::new();
    data.try_reserve_exact(bytes).map_err(|_| AudioError::Allocation)?;
    data.resize(bytes, 0);
    Ok(AudioFrame::new(data, sample_rate, num_channels, frames))
}

pub fn calculate_audio_duration(frames: &[AudioFrame]) -> f64 {
    frames.iter().map(AudioFrame::duration).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_window_retains_order_across_wrap_shift_and_reset() {
        let mut buffer = AudioArrayBuffer::new(5, 5).unwrap();
        buffer.push(&[1, 2, 3, 4]).unwrap();
        buffer.push(&[5, 6, 7]).unwrap();
        assert_eq!(buffer.read(), [3, 4, 5, 6, 7]);
        buffer.shift(3);
        buffer.push(&[8, 9, 10]).unwrap();
        assert_eq!(buffer.read(), [6, 7, 8, 9, 10]);
        buffer.shift(usize::MAX);
        assert!(buffer.read().is_empty());
        buffer.push(&[-32768, 32767]).unwrap();
        assert_eq!(buffer.read_bytes(), [0, 128, 255, 127]);
        buffer.reset();
        buffer.push(&[11]).unwrap();
        assert_eq!(buffer.read(), [11]);
    }

    #[test]
    fn rejected_audio_does_not_discard_the_previous_window() {
        let mut buffer = AudioArrayBuffer::new(2, 2).unwrap();
        buffer.push(&[1, 2]).unwrap();
        assert_eq!(
            buffer.push(&[3, 4, 5]),
            Err(AudioError::FrameTooLarge { samples: 3, capacity: 2 })
        );
        assert_eq!(buffer.read(), [1, 2]);
        let malformed = AudioFrame::new(vec![0; 3], 16_000, 2, 1);
        assert_eq!(buffer.push_frame(&malformed), Err(AudioError::InvalidPcm));
        assert_eq!(buffer.read(), [1, 2]);
        assert!(matches!(AudioArrayBuffer::new(3, 2), Err(AudioError::SampleLimit { limit: 2 })));
    }

    #[test]
    fn downmix_uses_floor_for_negative_averages_without_intermediate_pcm() {
        let mut buffer = AudioArrayBuffer::new(3, 3).unwrap();
        let pcm: Vec<u8> = [-1_i16, 0, -32768, 32767, 32767, 32767]
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        buffer.push_frame(&AudioFrame::new(pcm, 16_000, 2, 3)).unwrap();
        assert_eq!(buffer.read(), [-1, -1, 32767]);
    }

    #[test]
    fn silence_and_duration_preserve_declared_frame_accounting() {
        let frame = silence_frame(0.01001, 16_000, 2, 320).unwrap();
        assert_eq!(frame.samples_per_channel, 160);
        assert_eq!(frame.data.len(), 640);
        let header = AudioFrame::new(Vec::new(), 16_000, 2, 160);
        assert_eq!(silence_frame_like(&header, 320).unwrap(), frame);
        assert_eq!(calculate_audio_duration(&[frame, header]), 0.02);
        assert_eq!(silence_frame(f64::NAN, 16_000, 1, 320), Err(AudioError::InvalidDuration));
        assert_eq!(silence_frame(1.0, 48_000, 2, 320), Err(AudioError::SampleLimit { limit: 320 }));
    }
}
