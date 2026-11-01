//! Rate conversion for decoded PCM. Format changes drain the preceding stream first.

use super::Resampler;
use crate::{AudioFormat, CodecError, Quality, Result};

pub(crate) struct PcmResampler {
    target_rate: Option<u32>,
    stream: Option<(AudioFormat, Resampler)>,
    output_format: Option<AudioFormat>,
}

impl PcmResampler {
    pub fn new(target_rate: Option<u32>) -> Self {
        Self { target_rate, stream: None, output_format: None }
    }

    pub fn output_format(&self) -> Option<AudioFormat> {
        self.output_format
    }

    /// Drain and reset without ending the decoded stream (also used for drift flushes).
    pub fn flush(&mut self) -> Result<Vec<i16>> {
        match self.stream.as_mut() {
            Some((format, resampler)) => {
                flush_tail(*format, self.target_rate.unwrap_or(format.sample_rate), resampler)
            }
            None => Ok(Vec::new()),
        }
    }

    /// A format change drains the previous resampler's tail before tearing it down.
    pub fn push(&mut self, src: AudioFormat, samples: &[i16]) -> Result<Vec<i16>> {
        let resampler = &mut self.stream;
        let output_format = &mut self.output_format;
        let target_rate = self.target_rate;
        let Some(target_rate) = target_rate.filter(|r| *r != src.sample_rate) else {
            let mut tail = Vec::new();
            if let Some((fmt, mut r)) = resampler.take() {
                tail = flush_tail(fmt, src.sample_rate, &mut r)?;
            }
            *output_format = Some(src);
            if tail.is_empty() {
                return Ok(samples.to_vec());
            }
            tail.extend_from_slice(samples);
            return Ok(tail);
        };

        let mut prefix = Vec::new();
        if resampler.as_ref().is_some_and(|(f, _)| *f != src)
            && let Some((fmt, mut r)) = resampler.take()
        {
            prefix = flush_tail(fmt, target_rate, &mut r)?;
        }
        if resampler.is_none() {
            // Match the default quality used by frame conversion.
            let r = Resampler::new(
                src.sample_rate,
                target_rate,
                u32::from(src.num_channels),
                Quality::Medium,
            )
            .map_err(|e| CodecError::Resample {
                from: src.sample_rate,
                to: target_rate,
                detail: e.to_string(),
            })?;
            *resampler = Some((src, r));
        }
        let (_, r) = resampler.as_mut().expect("just inserted");
        prefix.extend_from_slice(r.push(samples).map_err(|e| CodecError::Resample {
            from: src.sample_rate,
            to: target_rate,
            detail: e.to_string(),
        })?);
        *output_format = Some(AudioFormat { sample_rate: target_rate, ..src });
        Ok(prefix)
    }
}

/// The tail belongs to the run that produced it, so call this before dropping the resampler.
fn flush_tail(src: AudioFormat, target_rate: u32, r: &mut Resampler) -> Result<Vec<i16>> {
    r.flush().map(<[i16]>::to_vec).map_err(|e| CodecError::Resample {
        from: src.sample_rate,
        to: target_rate,
        detail: e.to_string(),
    })
}
