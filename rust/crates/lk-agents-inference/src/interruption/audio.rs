//! `AudioArrayBuffer` (`utils/audio.py:90`). No resampler: the leg is fed the session's 16 kHz
//! conversion, so an off-rate frame is refused. Multi-channel input averages with floor division.

use thiserror::Error;

pub use lk_agents_abi::AudioFrame;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AudioError {
    /// The caller is on the wrong side of the 16 kHz gate.
    #[error("frame is {frame}Hz, the window is {window}Hz: the interruption leg is fed converted audio")]
    UnexpectedFormat { frame: u32, window: u32 },
    /// `ValueError: frame samples (...) are greater than the buffer size (...)`.
    #[error("frame samples ({samples}) are greater than the buffer size ({buffer_size})")]
    FrameLargerThanBuffer { samples: usize, buffer_size: usize },
    /// `np.frombuffer(..., count=...)` raises when the buffer is short; a torn read, not padded.
    #[error("frame carries {actual} bytes, expected {expected}")]
    ShortFrame { actual: usize, expected: usize },
    #[error("frame declares zero channels")]
    NoChannels,
}

#[derive(Debug)]
pub struct AudioWindow {
    buffer: Vec<i16>,
    start_idx: usize,
    sample_rate: u32,
}

impl AudioWindow {
    /// `buffer_size` is in samples: `int(max_audio_duration * sample_rate)`, truncating.
    pub fn new(buffer_size: usize, sample_rate: u32) -> Self {
        AudioWindow { buffer: vec![0; buffer_size], start_idx: 0, sample_rate }
    }

    /// Samples held, not the capacity.
    pub fn len(&self) -> usize {
        self.start_idx
    }

    pub fn is_empty(&self) -> bool {
        self.start_idx == 0
    }

    /// Drops just enough from the front to make room; returns the samples written.
    pub fn push_frame(&mut self, frame: &AudioFrame) -> Result<usize, AudioError> {
        if frame.num_channels == 0 {
            return Err(AudioError::NoChannels);
        }
        if frame.sample_rate != self.sample_rate {
            return Err(AudioError::UnexpectedFormat {
                frame: frame.sample_rate,
                window: self.sample_rate,
            });
        }
        let channels = frame.num_channels as usize;
        let samples = frame.samples_per_channel as usize;
        let expected = samples * channels * 2;
        if frame.data.len() < expected {
            return Err(AudioError::ShortFrame { actual: frame.data.len(), expected });
        }
        if samples == 0 {
            return Ok(0);
        }
        if samples > self.buffer.len() {
            return Err(AudioError::FrameLargerThanBuffer {
                samples,
                buffer_size: self.buffer.len(),
            });
        }

        // `if (shift_size := start + n - size) > 0`; `checked_sub` covers it (`shift(0)` is a no-op).
        if let Some(overflow) = (self.start_idx + samples).checked_sub(self.buffer.len()) {
            self.shift(overflow);
        }

        let interleaved: Vec<i16> = frame.data[..expected]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        for i in 0..samples {
            // Floor division, like numpy's `//` (Rust's `/` truncates toward zero).
            let mut sum: i32 = 0;
            for c in 0..channels {
                sum += i32::from(interleaved[i * channels + c]);
            }
            self.buffer[self.start_idx + i] = sum.div_euclid(channels as i32) as i16;
        }
        self.start_idx += samples;
        Ok(samples)
    }

    /// Drop `size` samples from the front, clamped to what is held (`min(size, self._start_idx)`).
    pub fn shift(&mut self, size: usize) {
        let size = size.min(self.start_idx);
        self.buffer.copy_within(size..self.start_idx, 0);
        self.start_idx -= size;
    }

    pub fn read(&self) -> Vec<i16> {
        self.buffer[..self.start_idx].to_vec()
    }

    /// The reference's zero-fill is unobservable and omitted.
    pub fn reset(&mut self) {
        self.start_idx = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mono(samples: &[i16]) -> AudioFrame {
        AudioFrame {
            data: samples.iter().flat_map(|s| s.to_le_bytes()).collect(),
            sample_rate: 16_000,
            num_channels: 1,
            samples_per_channel: samples.len() as u32,
        }
    }

    fn stereo(pairs: &[(i16, i16)]) -> AudioFrame {
        AudioFrame {
            data: pairs
                .iter()
                .flat_map(|(l, r)| [l.to_le_bytes(), r.to_le_bytes()])
                .flatten()
                .collect(),
            sample_rate: 16_000,
            num_channels: 2,
            samples_per_channel: pairs.len() as u32,
        }
    }

    // Catches: the window growing without bound, or dropping from the wrong end.
    #[test]
    fn a_full_window_slides_and_keeps_the_newest_audio() {
        let mut w = AudioWindow::new(4, 16_000);
        w.push_frame(&mono(&[1, 2, 3])).unwrap();
        assert_eq!(w.read(), vec![1, 2, 3]);
        w.push_frame(&mono(&[4, 5])).unwrap();
        assert_eq!(w.read(), vec![2, 3, 4, 5], "oldest sample dropped, capacity respected");
        assert_eq!(w.len(), 4);
    }

    // Catches: `sum / channels` instead of floor division.
    #[test]
    fn stereo_downmix_floors_negative_sums_the_way_numpy_does() {
        let mut w = AudioWindow::new(8, 16_000);
        w.push_frame(&stereo(&[(-1, -2), (1, 2), (-1, 0), (0, 1)])).unwrap();
        assert_eq!(w.read(), vec![-2, 1, -1, 0]);
    }

    // Catches: a resampler growing back (DR-9), or an off-rate frame reinterpreted at the window's
    // rate; it is refused and the window untouched.
    #[test]
    fn an_off_rate_frame_is_refused_rather_than_resampled_or_reinterpreted() {
        let mut w = AudioWindow::new(16_000, 16_000);
        let mut off = mono(&[1, 2, 3, 4, 5, 6]);
        off.sample_rate = 48_000;
        assert_eq!(
            w.push_frame(&off),
            Err(AudioError::UnexpectedFormat { frame: 48_000, window: 16_000 })
        );
        assert!(w.is_empty(), "a refused frame writes nothing");
        assert_eq!(w.push_frame(&mono(&[7, 8])), Ok(2));
        assert_eq!(w.read(), vec![7, 8]);
    }

    // Catches: a frame bigger than the window writing out of bounds, or rejecting one exactly the
    // window's size.
    #[test]
    fn a_frame_larger_than_the_window_is_an_error_but_an_exactly_sized_one_is_not() {
        let mut w = AudioWindow::new(2, 16_000);
        assert_eq!(
            w.push_frame(&mono(&[1, 2, 3])),
            Err(AudioError::FrameLargerThanBuffer { samples: 3, buffer_size: 2 })
        );
        assert!(w.is_empty());
        w.push_frame(&mono(&[8, 9])).expect("a window-sized frame must be accepted");
        assert!(!w.is_empty());
        assert_eq!(w.read(), vec![8, 9]);
    }

    // Catches: trusting `samples_per_channel` over the byte count.
    #[test]
    fn a_truncated_frame_reports_short_instead_of_reading_past_the_end() {
        let mut w = AudioWindow::new(8, 16_000);
        let mut f = mono(&[1, 2, 3]);
        f.data = f.data[..5].into();
        assert_eq!(w.push_frame(&f), Err(AudioError::ShortFrame { actual: 5, expected: 6 }));

        // The expected size scales with the channel count.
        let mut f = stereo(&[(1, 2), (3, 4)]);
        f.data = f.data[..4].into();
        assert_eq!(w.push_frame(&f), Err(AudioError::ShortFrame { actual: 4, expected: 8 }));
    }

    // Catches: `shift` panicking or underflowing when asked for more than is held.
    #[test]
    fn shifting_past_the_end_empties_the_window_instead_of_underflowing() {
        let mut w = AudioWindow::new(8, 16_000);
        w.push_frame(&mono(&[1, 2, 3])).unwrap();
        w.shift(99);
        assert!(w.is_empty());
        w.push_frame(&mono(&[7])).unwrap();
        assert_eq!(w.read(), vec![7], "and the window is still usable afterwards");
    }
}
