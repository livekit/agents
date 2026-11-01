//! The WAV half of `AudioStreamDecoder` (`utils/codecs/decoder.py:337-515`) as one synchronous
//! calculator: `push` returns the PCM its bytes produced; the channel, executor and framing are the
//! runtime's.

use crate::AudioFormat;
use crate::error::{CodecError, Result};
use crate::resample::AudioResampler;
use crate::wav::WavDecoder;

/// Native transform bounds, separate from the runtime's retained input/output queues.
#[derive(Debug, Clone, Copy)]
pub struct DecodeLimits {
    pub min_sample_rate: u32,
    pub max_sample_rate: u32,
    pub max_channels: u16,
    pub max_streams: usize,
    pub max_expansion_ratio: u32,
    pub max_samples_per_frame: usize,
}

impl DecodeLimits {
    pub fn validate(
        &self,
        source: AudioFormat,
        target_rate: Option<u32>,
        samples: usize,
    ) -> Result<()> {
        if !self.is_valid() {
            return Err(CodecError::DecodeLimit("invalid limits"));
        }
        let rate = target_rate.unwrap_or(source.sample_rate);
        if source.sample_rate < self.min_sample_rate
            || source.sample_rate > self.max_sample_rate
            || rate < self.min_sample_rate
            || rate > self.max_sample_rate
        {
            return Err(CodecError::DecodeLimit("sample rate"));
        }
        if source.num_channels == 0 || source.num_channels > self.max_channels {
            return Err(CodecError::DecodeLimit("channels"));
        }
        if u64::from(rate) > u64::from(source.sample_rate) * u64::from(self.max_expansion_ratio) {
            return Err(CodecError::DecodeLimit("resampling expansion"));
        }
        let output = (samples as u128 * u128::from(rate)).div_ceil(u128::from(source.sample_rate));
        if samples > self.max_samples_per_frame || output > self.max_samples_per_frame as u128 {
            return Err(CodecError::DecodeLimit("decoded frame samples"));
        }
        Ok(())
    }

    /// For a WAV calculator fed whole pushes: the input that carried a push bounds its size,
    /// and a provider's concatenated files restart only at a push boundary.
    pub fn whole_pushes(self) -> Self {
        Self { max_samples_per_frame: i32::MAX as usize, ..self }
    }

    pub fn is_valid(&self) -> bool {
        self.min_sample_rate > 0
            && self.max_sample_rate >= self.min_sample_rate
            && self.max_sample_rate <= i32::MAX as u32
            && self.max_channels > 0
            && self.max_streams > 0
            && self.max_streams <= i32::MAX as usize
            && self.max_expansion_ratio > 0
            && self.max_expansion_ratio <= i32::MAX as u32
            && self.max_samples_per_frame > 0
            && self.max_samples_per_frame <= i32::MAX as usize
    }
}

/// One decoder decodes one stream, as in the reference.
pub struct AudioDecoder {
    wav: WavDecoder,
    /// `None` keeps the source's rate.
    target_rate: Option<u32>,
    /// The file's channels are remixed to these, as the container path's resampler does.
    num_channels: u16,
    resampler: AudioResampler,
    limits: DecodeLimits,
    partial: Vec<i16>,
    partial_format: Option<AudioFormat>,
}

impl std::fmt::Debug for AudioDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The resampler owns a native handle.
        f.debug_struct("AudioDecoder")
            .field("target_rate", &self.target_rate)
            .field("output_format", &self.output_format())
            .finish()
    }
}

/// Incremental WAV-to-PCM calculator; compressed containers decode on the runtime's workers.
impl AudioDecoder {
    /// `sample_rate=None` keeps the source rate.
    pub fn wav(sample_rate: Option<u32>, num_channels: u16, limits: DecodeLimits) -> Result<Self> {
        if !limits.is_valid() {
            return Err(CodecError::DecodeLimit("invalid limits"));
        }
        Ok(Self {
            wav: WavDecoder::new(),
            target_rate: sample_rate,
            num_channels,
            resampler: AudioResampler::decoded(sample_rate),
            limits,
            partial: Vec::new(),
            partial_format: None,
        })
    }

    /// `None` until the first header is parsed.
    pub fn output_format(&self) -> Option<AudioFormat> {
        self.resampler.output_format()
    }

    /// Routinely empty: a push that only carried a header decodes to nothing.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<i16>> {
        let segments = self.wav.push(bytes)?;
        let mut out = Vec::new();
        for seg in segments {
            self.limits.validate(
                seg.format,
                self.target_rate,
                seg.samples.len().saturating_add(self.partial.len()),
            )?;
            let channels = usize::from(seg.format.num_channels);
            if channels == 0 {
                return Err(CodecError::DecodeLimit("channels"));
            }
            if self.partial_format.is_some_and(|format| format != seg.format)
                && !self.partial.is_empty()
            {
                return Err(CodecError::TruncatedWav);
            }
            self.partial_format = Some(seg.format);
            if self.partial.is_empty() && seg.samples.len() % channels == 0 {
                out.extend(self.convert(seg.format, &seg.samples)?);
                continue;
            }
            self.partial.extend(seg.samples);
            let complete = self.partial.len() / channels * channels;
            let samples =
                remix(&self.partial[..complete], seg.format.num_channels, self.num_channels)?;
            out.extend(self.resampler.push_samples(
                AudioFormat { num_channels: self.num_channels, ..seg.format },
                &samples,
            )?);
            self.partial.drain(..complete);
        }
        Ok(out)
    }

    /// The resampler's tail; the stream stays open.
    pub fn flush(&mut self) -> Result<Vec<i16>> {
        self.wav.flush();
        self.resampler.flush_samples()
    }

    /// Validate terminal input before draining. `flush` remains available for mid-stream drift.
    pub fn finish(&mut self) -> Result<Vec<i16>> {
        self.wav.end_input()?;
        if !self.partial.is_empty() {
            return Err(CodecError::TruncatedWav);
        }
        self.flush()
    }

    fn convert(&mut self, format: AudioFormat, samples: &[i16]) -> Result<Vec<i16>> {
        let samples = remix(samples, format.num_channels, self.num_channels)?;
        self.resampler
            .push_samples(AudioFormat { num_channels: self.num_channels, ..format }, &samples)
    }
}

/// Interleaved `samples` of `from` channels as `to`: averaged down to mono, or mono repeated.
pub(crate) fn remix(samples: &[i16], from: u16, to: u16) -> Result<Vec<i16>> {
    if from == to {
        return Ok(samples.to_vec());
    }
    if to == 1 {
        return Ok(crate::downmix_mono(samples, from));
    }
    if from == 1 {
        return Ok(samples.iter().flat_map(|s| std::iter::repeat_n(*s, usize::from(to))).collect());
    }
    Err(CodecError::DecodeLimit("channel layout"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::tests::wav_file;

    fn limits() -> DecodeLimits {
        DecodeLimits {
            min_sample_rate: 8_000,
            max_sample_rate: 192_000,
            max_channels: 8,
            max_streams: 8,
            max_expansion_ratio: 24,
            max_samples_per_frame: 1 << 20,
        }
    }

    /// A deterministic ramp, long enough that soxr's filter is holding history.
    fn ramp(n: usize) -> Vec<i16> {
        (0..n).map(|i| ((i % 200) as i16 - 100) * 100).collect()
    }

    // Catches: routing `audio/wav` through the ffmpeg path (slow, and unavailable in the default
    // build) or failing it outright. This is the case that unblocks `AudioEmitter` today, so it
    // has to work with no features enabled at all.
    #[test]
    fn a_wav_mime_type_decodes_with_no_features_enabled() {
        let want: Vec<i16> = vec![3, -3, 300, -300];
        let mut d = AudioDecoder::wav(Some(24_000), 1, limits()).unwrap();
        let got = d.push(&wav_file(24_000, 1, &want)).unwrap();
        assert_eq!(got, want);
        assert_eq!(d.output_format(), Some(AudioFormat::new(24_000, 1)));
        assert!(d.flush().unwrap().is_empty());
    }

    // Catches: a rate mismatch refused (before PLAN.md F1 it was `ResampleRequired`) or emitted
    // at the source rate -- a pitch shift nothing downstream can detect; the resampler built with
    // its rates swapped (upsampling must lengthen); and the channel count not reaching soxr
    // (interleaved stereo resampled as mono runs both channels through one filter).
    #[test]
    fn a_rate_mismatch_is_resampled_to_the_requested_rate() {
        // (requested rate, file rate, channels, input samples, expected output frames)
        for (rate, src, ch, n, frames) in [
            (24_000, 48_000, 1, 4800, 2350..=2450),
            (48_000, 24_000, 1, 2400, 4700..=4900),
            (24_000, 48_000, 2, 4800, 1150..=1250),
        ] {
            let mut d = AudioDecoder::wav(Some(rate), ch, limits()).unwrap();
            let total =
                d.push(&wav_file(src, ch, &ramp(n))).unwrap().len() + d.flush().unwrap().len();
            assert_eq!(total % usize::from(ch), 0, "whole frames");
            let got = total / usize::from(ch);
            assert!(frames.contains(&got), "{src} -> {rate} x{ch}: {got} frames, want {frames:?}");
            assert_eq!(d.output_format(), Some(AudioFormat::new(rate, ch)));
        }
    }

    // Catches: `sample_rate=None` read as 48 kHz. The reference keeps the source's rate then.
    #[test]
    fn no_requested_rate_keeps_the_sources_rate() {
        let mut d = AudioDecoder::wav(None, 1, limits()).unwrap();
        assert_eq!(d.push(&wav_file(16_000, 1, &ramp(160))).unwrap(), ramp(160));
        assert_eq!(d.output_format(), Some(AudioFormat::new(16_000, 1)));
    }

    // Catches: dropping the resampler's tail at end of stream. soxr always holds a filter delay;
    // losing it clips the last milliseconds of every utterance, which gets reported as "the agent
    // cuts itself off".
    #[test]
    fn flush_releases_the_resamplers_tail() {
        let mut d = AudioDecoder::wav(Some(48_000), 1, limits()).unwrap();
        let pushed = d.push(&wav_file(24_000, 1, &ramp(2400))).unwrap();
        let tail = d.flush().unwrap();
        assert!(!tail.is_empty(), "the held interpolation/filter tail must be released");
        assert!(
            (4700..=4900).contains(&(pushed.len() + tail.len())),
            "the complete 100 ms input must survive final draining"
        );
    }

    // Catches: reusing the first file's resampler for a second, differently-rated file, and
    // losing the first one's tail when the format changes. Concatenated WAVs from one provider
    // can change rate mid-connection; a stale resampler plays the rest at the wrong speed.
    //
    // Duration conservation is the observation, because with a real resampler there is no call
    // count to check: 4800 samples at 48 kHz plus 1600 at 16 kHz is 0.2 s, and 0.2 s at 24 kHz is
    // 4800 samples. A dropped tail or a stale resampler both make that number wrong.
    #[test]
    fn a_format_change_drains_the_old_resampler_before_building_a_new_one() {
        let mut d = AudioDecoder::wav(Some(24_000), 1, limits()).unwrap();
        let mut total = d.push(&wav_file(48_000, 1, &ramp(4800))).unwrap().len();
        total += d.push(&wav_file(16_000, 1, &ramp(1600))).unwrap().len();
        total += d.flush().unwrap().len();
        assert!(
            (4700..=4900).contains(&total),
            "0.1 s + 0.1 s at 24 kHz is 4800 samples; got {total}"
        );
    }

    // Catches: throwing away the previous resampler's tail when the NEXT file happens to match
    // the target rate and so takes the straight-through path. That branch is easy to write as an
    // early return, and then the join between the two files is a click.
    #[test]
    fn a_change_to_a_matching_rate_still_drains_the_old_resampler_first() {
        let mut d = AudioDecoder::wav(Some(48_000), 1, limits()).unwrap();
        let first = d.push(&wav_file(24_000, 1, &ramp(2400))).unwrap();
        let second = d.push(&wav_file(48_000, 1, &ramp(240))).unwrap();
        assert!(
            second.len() > 240,
            "the second push must carry the first run's tail ahead of its own {} samples; got {}",
            240,
            second.len()
        );
        assert_eq!(
            &second[second.len() - 240..],
            &ramp(240)[..],
            "and the straight-through samples must be untouched behind it"
        );
        assert!((5000..=5100).contains(&(first.len() + second.len())));
    }

    // Catches: a stereo WAV framed as the emitter's mono (every other sample a channel: twice the
    // length, half the pitch) or a mono one as stereo. The WAV path converts the file's channels
    // to the requested count, as the container path does.
    #[test]
    fn the_wav_path_converts_the_files_channels_to_the_requested_count() {
        let mut d = AudioDecoder::wav(Some(24_000), 1, limits()).unwrap();
        let stereo: Vec<i16> = vec![10, 30, -10, -30, 200, 0];
        assert_eq!(d.push(&wav_file(24_000, 2, &stereo)).unwrap(), vec![20, -20, 100]);
        assert_eq!(d.output_format(), Some(AudioFormat::new(24_000, 1)));

        let mut d = AudioDecoder::wav(Some(24_000), 2, limits()).unwrap();
        assert_eq!(d.push(&wav_file(24_000, 1, &[5, -7])).unwrap(), vec![5, 5, -7, -7]);
        assert_eq!(d.output_format(), Some(AudioFormat::new(24_000, 2)));
    }

    // Catches: a header-only push being reported as an error or as silence-with-a-format. It is
    // neither: it is simply no audio yet, and the caller must be able to push again.
    #[test]
    fn a_push_that_only_carries_a_header_decodes_to_nothing_without_failing() {
        let file = wav_file(24_000, 1, &[9, 8, 7]);
        let mut d = AudioDecoder::wav(Some(24_000), 1, limits()).unwrap();
        assert!(d.push(&file[..44]).unwrap().is_empty());
        assert_eq!(d.output_format(), None);
        assert_eq!(d.push(&file[44..]).unwrap(), vec![9, 8, 7]);
    }
}
