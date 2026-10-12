//! The resampler seam and a pure-Rust fallback. Production uses libsoxr (what the models were
//! trained on); `lk_agents_media::new_resampler` is the one place that chooses.

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

/// No anti-aliasing, for builds without libsoxr. `n` input frames at ratio `r` produce
/// `floor(n * r)` output frames over the stream, the remainder released by `flush`.
pub struct LinearResampler {
    input_rate: u32,
    output_rate: u32,
    channels: usize,
    /// The first is the frame before the read position.
    buf: Vec<i16>,
    /// In input frames (fractional).
    pos: f64,
    step: f64,
    out: Vec<i16>,
}

impl LinearResampler {
    pub fn new(input_rate: u32, output_rate: u32, num_channels: u32) -> Result<Self, ResampleError> {
        if input_rate == 0 || output_rate == 0 {
            return Err(ResampleError::Create(format!("rates must be non-zero: {input_rate} -> {output_rate}")));
        }
        if num_channels == 0 {
            return Err(ResampleError::Create("at least one channel is required".to_owned()));
        }
        Ok(LinearResampler {
            input_rate,
            output_rate,
            channels: num_channels as usize,
            buf: Vec::new(),
            pos: 0.0,
            step: f64::from(input_rate) / f64::from(output_rate),
            out: Vec::new(),
        })
    }

    fn frames(&self) -> usize {
        self.buf.len() / self.channels
    }

    /// Produce every output frame whose two neighbours are in `buf`, then drop the consumed head.
    fn produce(&mut self) {
        self.out.clear();
        let ch = self.channels;
        let frames = self.frames();
        while (self.pos.floor() as usize) + 1 < frames {
            let i = self.pos.floor() as usize;
            let frac = self.pos - i as f64;
            for c in 0..ch {
                let a = f64::from(self.buf[i * ch + c]);
                let b = f64::from(self.buf[(i + 1) * ch + c]);
                let v = a + (b - a) * frac;
                self.out.push(v.round().clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16);
            }
            self.pos += self.step;
        }
        let consumed = (self.pos.floor() as usize).min(frames);
        self.buf.drain(..consumed * ch);
        self.pos -= consumed as f64;
    }
}

impl Resampler for LinearResampler {
    fn input_rate(&self) -> u32 {
        self.input_rate
    }

    fn output_rate(&self) -> u32 {
        self.output_rate
    }

    fn num_channels(&self) -> u32 {
        self.channels as u32
    }

    fn push(&mut self, input: &[i16]) -> Result<&[i16], ResampleError> {
        if !input.len().is_multiple_of(self.channels) {
            return Err(ResampleError::PartialFrame { len: input.len(), num_channels: self.channels as u32 });
        }
        self.buf.extend_from_slice(input);
        self.produce();
        Ok(&self.out)
    }

    fn flush(&mut self) -> Result<&[i16], ResampleError> {
        // Hold the last frame so the interpolation can reach the end of the input, then reset.
        if let Some(last) = self.buf.len().checked_sub(self.channels) {
            let tail: Vec<i16> = self.buf[last..].to_vec();
            self.buf.extend_from_slice(&tail);
        }
        self.produce();
        self.buf.clear();
        self.pos = 0.0;
        Ok(&self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(from: u32, to: u32, ch: u32, chunks: &[Vec<i16>]) -> Vec<i16> {
        let mut r = LinearResampler::new(from, to, ch).unwrap();
        let mut out = Vec::new();
        for c in chunks {
            out.extend_from_slice(r.push(c).unwrap());
        }
        out.extend_from_slice(r.flush().unwrap());
        out
    }

    // Catches: the frame count drifting with the chunking.
    #[test]
    fn the_output_length_is_the_ratio_whatever_the_chunking() {
        let input: Vec<i16> = (0..480).map(|i| (i * 10) as i16).collect();
        let whole = run(48_000, 16_000, 1, &[input.clone()]);
        let chunked: Vec<Vec<i16>> = input.chunks(20).map(<[i16]>::to_vec).collect();
        let pieces = run(48_000, 16_000, 1, &chunked);
        assert_eq!(whole.len(), 160);
        assert_eq!(whole, pieces);
    }

    // Catches: a ramp not staying a ramp (channel shear, off-by-one read position).
    #[test]
    fn a_ramp_upsampled_stays_monotone_per_channel() {
        let input: Vec<i16> = (0..100).flat_map(|i| [(i * 100) as i16, -(i * 100) as i16]).collect();
        let out = run(16_000, 48_000, 2, &[input]);
        assert!(out.len() >= 2 * 290, "too few frames: {}", out.len() / 2);
        for pair in out.chunks_exact(2).collect::<Vec<_>>().windows(2) {
            assert!(pair[1][0] >= pair[0][0], "left channel fell: {pair:?}");
            assert!(pair[1][1] <= pair[0][1], "right channel rose: {pair:?}");
        }
    }

    // Catches: `flush` leaving history behind, so the next utterance starts on the last one's tail.
    #[test]
    fn flush_resets_the_state() {
        let mut r = LinearResampler::new(48_000, 16_000, 1).unwrap();
        let _ = r.push(&[1000; 47]).unwrap();
        let _ = r.flush().unwrap();
        assert_eq!(r.push(&[0; 3]).unwrap(), &[0]);
    }

    #[test]
    fn a_partial_frame_is_refused() {
        let mut r = LinearResampler::new(48_000, 16_000, 2).unwrap();
        assert!(matches!(r.push(&[1, 2, 3]), Err(ResampleError::PartialFrame { len: 3, num_channels: 2 })));
        assert!(LinearResampler::new(0, 16_000, 1).is_err());
    }
}
