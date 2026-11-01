//! Audio-frame and decoded-PCM conversion over one owned libsoxr stream.

pub use crate::soxr::Resampler;
use crate::{AudioFormat, CodecError, Quality, ResampleError};
use lk_agents_abi::AudioFrame;

/// What a new frame format does to the stream before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnChange {
    /// Drain the old tail first. Passthrough and `flush` also end the old stream.
    Drain,
    /// Replace without draining after a successful build. Passthrough and `flush` keep it.
    Discard,
}

#[derive(Clone, Copy)]
enum Mode {
    Follow(OnChange),
    /// Headers never override the declared input, including after a reset.
    Fixed(AudioFormat),
    /// Decoded format changes drain; flushing resets the filter without ending the stream.
    Decoded,
}

struct Stream {
    resampler: Resampler,
    input: AudioFormat,
    output: AudioFormat,
}

/// One conversion failure, retaining the format of the operation that failed (which may be an
/// old stream's drain). Frame callers receive the engine error; decoders retain codec context.
struct Failure {
    from: u32,
    to: u32,
    error: ResampleError,
}

impl Failure {
    fn codec(self) -> CodecError {
        CodecError::Resample { from: self.from, to: self.to, detail: self.error.to_string() }
    }
}

impl Stream {
    fn push(&mut self, samples: &[i16]) -> Result<&[i16], Failure> {
        let from = self.input.sample_rate;
        let to = self.output.sample_rate;
        self.resampler.push(samples).map_err(|error| Failure { from, to, error })
    }

    fn flush(&mut self) -> Result<Vec<i16>, Failure> {
        let from = self.input.sample_rate;
        let to = self.output.sample_rate;
        self.resampler.flush().map(<[i16]>::to_vec).map_err(|error| Failure { from, to, error })
    }
}

/// Owns one resampling stream, independent of whether its input arrives as frames or decoded
/// samples. Frame headers and decoded format records describe the same input; only boundary
/// representation and the selected lifecycle policy differ. soxr may buffer an entire push.
pub struct AudioResampler {
    target_rate: Option<u32>,
    quality: Quality,
    mode: Mode,
    stream: Option<Stream>,
    output_format: Option<AudioFormat>,
}

impl AudioResampler {
    /// Follow frame headers, building lazily for off-rate input. A zero-channel frame is treated
    /// as mono for conversion while retaining its original channel header.
    pub fn new(output_rate: u32, quality: Quality, on_change: OnChange) -> Self {
        Self {
            target_rate: Some(output_rate),
            quality,
            mode: Mode::Follow(on_change),
            stream: None,
            output_format: None,
        }
    }

    /// Use the declared format regardless of frame headers, even for a 1:1 ratio.
    pub fn fixed(
        input: AudioFormat,
        output_rate: u32,
        quality: Quality,
    ) -> Result<Self, ResampleError> {
        let stream =
            Self::build(input, output_rate, quality, false).map_err(|failure| failure.error)?;
        Ok(Self {
            target_rate: Some(output_rate),
            quality,
            mode: Mode::Fixed(input),
            stream: Some(stream),
            output_format: None,
        })
    }

    /// Follow decoded PCM formats at medium quality. No target rate means passthrough; format
    /// changes drain the old stream before its replacement, and flush keeps the stream.
    pub fn decoded(target_rate: Option<u32>) -> Self {
        Self {
            target_rate,
            quality: Quality::Medium,
            mode: Mode::Decoded,
            stream: None,
            output_format: None,
        }
    }

    /// The configured output rate, or the latest source rate when no target was configured.
    pub fn output_rate(&self) -> Option<u32> {
        self.target_rate.or_else(|| self.output_format.map(|format| format.sample_rate))
    }

    pub fn input_format(&self) -> Option<AudioFormat> {
        self.stream.as_ref().map(|stream| stream.input)
    }

    /// The most recently accepted input's output format, including passthrough input.
    pub fn output_format(&self) -> Option<AudioFormat> {
        self.output_format
    }

    /// Emit the preceding tail first, then converted output or the untouched passthrough frame.
    pub fn push_frame(&mut self, frame: &AudioFrame) -> Result<Vec<AudioFrame>, ResampleError> {
        let input = AudioFormat::new(frame.sample_rate, frame.num_channels);
        self.frame(input, frame).map_err(|failure| failure.error)
    }

    fn frame(
        &mut self,
        input: AudioFormat,
        frame: &AudioFrame,
    ) -> Result<Vec<AudioFrame>, Failure> {
        let mut out = Vec::new();
        if let Some((format, tail)) = self.prepare(input)? {
            out.extend(to_frame(&tail, format));
        }
        let format = if self.bypasses(input) {
            out.push(frame.clone());
            input
        } else {
            let samples = crate::pcm_samples(&frame.data);
            let (format, converted) = self.resample(input, &samples)?;
            out.extend(to_frame(converted, format));
            format
        };
        self.output_format = Some(format);
        Ok(out)
    }

    /// Feed decoded samples directly, without encoding them into a frame and decoding them back.
    pub fn push_samples(&mut self, input: AudioFormat, samples: &[i16]) -> crate::Result<Vec<i16>> {
        self.samples(input, samples).map_err(Failure::codec)
    }

    fn samples(&mut self, input: AudioFormat, samples: &[i16]) -> Result<Vec<i16>, Failure> {
        let mut out = self.prepare(input)?.map_or_else(Vec::new, |(_, tail)| tail);
        let format = if self.bypasses(input) {
            out.extend_from_slice(samples);
            input
        } else {
            let (format, converted) = self.resample(input, samples)?;
            out.extend_from_slice(converted);
            format
        };
        self.output_format = Some(format);
        Ok(out)
    }

    /// Flush the current tail as a frame. Follow/Drain ends the stream; the other modes retain
    /// the declared input and reset the filter, ready for more samples.
    pub fn flush(&mut self) -> Result<Option<AudioFrame>, ResampleError> {
        self.flush_raw()
            .map(|tail| tail.and_then(|(format, samples)| to_frame(&samples, format)))
            .map_err(|failure| failure.error)
    }

    /// The same flush for decoded PCM, preserving codec error context.
    pub fn flush_samples(&mut self) -> crate::Result<Vec<i16>> {
        self.flush_raw()
            .map(|tail| tail.map_or_else(Vec::new, |(_, samples)| samples))
            .map_err(Failure::codec)
    }

    pub fn reset(&mut self) {
        self.stream = None;
        self.output_format = None;
    }

    fn bypasses(&self, input: AudioFormat) -> bool {
        !matches!(self.mode, Mode::Fixed(_))
            && self.target_rate.is_none_or(|rate| rate == input.sample_rate)
    }

    /// Apply the stream-boundary policy once for either input representation. A drained tail
    /// carries its own output format; a following channel change must not relabel it.
    fn prepare(&mut self, input: AudioFormat) -> Result<Option<(AudioFormat, Vec<i16>)>, Failure> {
        if matches!(self.mode, Mode::Fixed(_)) {
            return Ok(None);
        }
        let bypass = self.bypasses(input);
        if matches!(self.mode, Mode::Follow(OnChange::Discard)) {
            if !bypass && self.input_format() != Some(input) {
                let stream = self.build_input(input)?;
                self.stream = Some(stream);
            }
            return Ok(None);
        }
        if bypass || self.input_format().is_some_and(|format| format != input) {
            return self.drain();
        }
        Ok(None)
    }

    fn resample(
        &mut self,
        input: AudioFormat,
        samples: &[i16],
    ) -> Result<(AudioFormat, &[i16]), Failure> {
        let input = match self.mode {
            Mode::Fixed(input) => input,
            _ => input,
        };
        let output_rate = self.target_rate.unwrap_or(input.sample_rate);
        let quality = self.quality;
        let normalize_channels = matches!(self.mode, Mode::Follow(_));
        let stream = match &mut self.stream {
            Some(stream) => stream,
            slot => slot.insert(Self::build(input, output_rate, quality, normalize_channels)?),
        };
        let format = stream.output;
        Ok((format, stream.push(samples)?))
    }

    fn build_input(&self, input: AudioFormat) -> Result<Stream, Failure> {
        Self::build(
            input,
            self.target_rate.unwrap_or(input.sample_rate),
            self.quality,
            matches!(self.mode, Mode::Follow(_)),
        )
    }

    fn build(
        input: AudioFormat,
        output_rate: u32,
        quality: Quality,
        normalize_channels: bool,
    ) -> Result<Stream, Failure> {
        let channels =
            if normalize_channels { input.num_channels.max(1) } else { input.num_channels };
        let resampler =
            Resampler::new(input.sample_rate, output_rate, u32::from(channels), quality)
                .map_err(|error| Failure { from: input.sample_rate, to: output_rate, error })?;
        Ok(Stream { resampler, input, output: AudioFormat { sample_rate: output_rate, ..input } })
    }

    fn drain(&mut self) -> Result<Option<(AudioFormat, Vec<i16>)>, Failure> {
        let Some(mut stream) = self.stream.take() else { return Ok(None) };
        Ok(Some((stream.output, stream.flush()?)))
    }

    fn flush_raw(&mut self) -> Result<Option<(AudioFormat, Vec<i16>)>, Failure> {
        if matches!(self.mode, Mode::Follow(OnChange::Drain)) {
            return self.drain();
        }
        match &mut self.stream {
            Some(stream) => Ok(Some((stream.output, stream.flush()?))),
            None => Ok(None),
        }
    }
}

fn to_frame(samples: &[i16], format: AudioFormat) -> Option<AudioFrame> {
    if samples.is_empty() {
        return None;
    }
    let spc = (samples.len() / usize::from(format.num_channels.max(1))) as u32;
    Some(AudioFrame::new(crate::pcm_bytes(samples), format.sample_rate, format.num_channels, spc))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(rate: u32, channels: u16, samples_per_channel: usize) -> AudioFrame {
        let pcm: Vec<i16> = (0..samples_per_channel * usize::from(channels))
            .map(|i| ((i * 37) % 2000) as i16 - 1000)
            .collect();
        AudioFrame::new(crate::pcm_bytes(&pcm), rate, channels, samples_per_channel as u32)
    }

    /// Output lengths of a bare resampler over `pushes`, then its flush.
    fn lengths(from: u32, to: u32, channels: u32, pushes: &[&AudioFrame]) -> (Vec<usize>, usize) {
        let mut r = Resampler::new(from, to, channels, Quality::Medium).unwrap();
        let out =
            pushes.iter().map(|f| r.push(&crate::pcm_samples(&f.data)).unwrap().len()).collect();
        (out, r.flush().unwrap().len())
    }

    // Catches: a rate change that drops the old stream's tail (the end of a sentence goes missing
    // at every TTS rate switch), or a passthrough frame emitted ahead of that tail.
    #[test]
    fn drain_puts_the_old_streams_tail_ahead_of_a_frame_at_the_output_rate() {
        let mut r = AudioResampler::new(16_000, Quality::Medium, OnChange::Drain);
        let off = ramp(48_000, 1, 4_801);
        let on = ramp(16_000, 1, 320);
        r.push_frame(&off).unwrap();
        let out = r.push_frame(&on).unwrap();
        let (_, tail) = lengths(48_000, 16_000, 1, &[&off]);
        assert_eq!(out.len(), 2, "the tail, then the frame");
        assert_eq!((out[0].sample_rate, out[0].data.len() / 2), (16_000, tail));
        assert_eq!(out[1], on, "a frame already at the output rate is not touched");
        assert_eq!(r.input_format(), None, "and the drained stream is over");
    }

    // Catches: `Discard` flushing or rebuilding across a frame at the output rate -- the
    // converted input leg would restart soxr's filter (and its delay) at every such frame.
    #[test]
    fn discard_keeps_the_stream_across_a_frame_at_the_output_rate() {
        let mut r = AudioResampler::new(16_000, Quality::Medium, OnChange::Discard);
        // Sized so soxr emits a different count for `b` depending on its filter history.
        let (a, b) = (ramp(44_100, 1, 4_411), ramp(44_100, 1, 1_000));
        let on = ramp(16_000, 1, 160);
        r.push_frame(&a).unwrap();
        assert_eq!(r.push_frame(&on).unwrap(), vec![on]);
        let second: usize = r.push_frame(&b).unwrap().iter().map(|f| f.data.len() / 2).sum();
        let (continuous, _) = lengths(44_100, 16_000, 1, &[&a, &b]);
        let (fresh, _) = lengths(44_100, 16_000, 1, &[&b]);
        assert_ne!(continuous[1], fresh[0], "the lengths tell a kept stream from a new one");
        assert_eq!(second, continuous[1], "the stream from before the passthrough carried on");
    }

    // Catches: `Discard` draining the old stream into the new rate's output on a rate change.
    #[test]
    fn discard_drops_the_old_stream_unflushed_on_a_rate_change() {
        let mut r = AudioResampler::new(16_000, Quality::Medium, OnChange::Discard);
        let (a, b) = (ramp(48_000, 1, 4_801), ramp(24_000, 1, 2_400));
        r.push_frame(&a).unwrap();
        let out = r.push_frame(&b).unwrap();
        let (fresh, _) = lengths(24_000, 16_000, 1, &[&b]);
        assert!(out.len() <= 1, "no tail frame");
        assert_eq!(out.iter().map(|f| f.data.len() / 2).sum::<usize>(), fresh[0]);
        assert_eq!(r.input_format(), Some(AudioFormat::new(24_000, 1)));
    }

    // Catches: a failed rebuild that leaves the stream half-replaced (the converted input leg
    // skips the frame and must carry on with the stream it had).
    #[test]
    fn a_failed_build_keeps_the_stream_it_would_have_replaced() {
        let mut r = AudioResampler::new(16_000, Quality::Medium, OnChange::Discard);
        r.push_frame(&ramp(48_000, 1, 480)).unwrap();
        assert!(matches!(r.push_frame(&ramp(0, 1, 480)), Err(ResampleError::Create(_))));
        assert_eq!(r.input_format(), Some(AudioFormat::new(48_000, 1)));
    }

    // Catches: a fixed resampler that follows the header -- the VAD reads every frame as mono
    // and the fallback TTS as its entry's declared rate, whatever the frame claims.
    #[test]
    fn fixed_reads_every_frame_as_the_declared_format() {
        let mut r =
            AudioResampler::fixed(AudioFormat::new(48_000, 1), 16_000, Quality::Medium).unwrap();
        let liar = ramp(24_000, 2, 2_400);
        let out = r.push_frame(&liar).unwrap();
        let (want, _) = lengths(48_000, 16_000, 1, &[&liar]);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].sample_rate, out[0].num_channels), (16_000, 1));
        assert_eq!(
            (out[0].data.len() / 2, out[0].samples_per_channel as usize),
            (want[0], want[0])
        );
        assert_eq!(r.input_format(), Some(AudioFormat::new(48_000, 1)));
    }

    // Catches: building with the header's 0 channels (soxr would refuse every push), or rewriting
    // the header the frame came with.
    #[test]
    fn a_zero_channel_header_is_resampled_as_mono_and_keeps_its_header() {
        let mut r = AudioResampler::new(16_000, Quality::Medium, OnChange::Drain);
        let mono = ramp(48_000, 1, 4_800);
        let mut out = r.push_frame(&AudioFrame { num_channels: 0, ..mono }).unwrap();
        out.extend(r.flush().unwrap());
        let total: usize = out.iter().map(|f| f.data.len() / 2).sum();
        assert!(total > 1_500, "resampled as one channel: {total} samples");
        assert!(
            out.iter()
                .all(|f| f.num_channels == 0 && f.samples_per_channel as usize * 2 == f.data.len())
        );
    }

    #[test]
    fn frame_and_sample_entrypoints_share_the_same_conversion() {
        for rate in [16_000, 44_100, 48_000] {
            let mut frames = AudioResampler::new(16_000, Quality::Medium, OnChange::Drain);
            let mut samples = AudioResampler::new(16_000, Quality::Medium, OnChange::Drain);
            let mut from_frames = Vec::new();
            let mut from_samples = Vec::new();
            for size in [4_801, 1_000, 373] {
                let frame = ramp(rate, 2, size);
                for converted in frames.push_frame(&frame).unwrap() {
                    from_frames.extend_from_slice(&crate::pcm_samples(&converted.data));
                }
                from_samples.extend(
                    samples
                        .push_samples(AudioFormat::new(rate, 2), &crate::pcm_samples(&frame.data))
                        .unwrap(),
                );
            }
            if let Some(tail) = frames.flush().unwrap() {
                from_frames.extend_from_slice(&crate::pcm_samples(&tail.data));
            }
            from_samples.extend(samples.flush_samples().unwrap());
            assert_eq!(from_frames, from_samples, "input rate {rate}");
            assert!(frames.input_format().is_none());
            assert!(samples.input_format().is_none());
        }
    }

    #[test]
    fn decoded_flush_keeps_the_stream_and_frame_drain_ends_it() {
        let frame = ramp(48_000, 1, 4_801);
        let input = AudioFormat::new(48_000, 1);
        let mut decoded = AudioResampler::decoded(Some(16_000));
        decoded.push_samples(input, &crate::pcm_samples(&frame.data)).unwrap();
        decoded.flush_samples().unwrap();
        assert_eq!(decoded.input_format(), Some(input));
        assert_eq!(decoded.output_format(), Some(AudioFormat::new(16_000, 1)));
        decoded.push_frame(&frame).unwrap();
        assert_eq!(decoded.input_format(), Some(input));
        let mut framed = AudioResampler::new(16_000, Quality::Medium, OnChange::Drain);
        framed.push_frame(&frame).unwrap();
        framed.flush().unwrap();
        assert_eq!(framed.input_format(), None);
    }

    #[test]
    fn a_frame_format_change_preserves_the_old_tails_channel_header() {
        let mut r = AudioResampler::new(16_000, Quality::Medium, OnChange::Drain);
        r.push_frame(&ramp(48_000, 2, 4_801)).unwrap();
        let out = r.push_frame(&ramp(24_000, 1, 2_401)).unwrap();
        assert!(!out.is_empty());
        assert_eq!(out[0].num_channels, 2, "the stereo tail precedes the new mono stream");
        assert!(out.iter().skip(1).all(|frame| frame.num_channels == 1));
    }

    #[test]
    fn fixed_reset_rebuilds_the_declared_stream_without_following_headers() {
        let input = AudioFormat::new(48_000, 1);
        let mut r = AudioResampler::fixed(input, 16_000, Quality::Medium).unwrap();
        r.reset();
        assert!(r.input_format().is_none());
        r.push_frame(&ramp(24_000, 2, 2_400)).unwrap();
        assert_eq!(r.input_format(), Some(input));
    }

    #[test]
    fn decoded_without_a_target_passes_samples_through_without_a_stream() {
        let mut r = AudioResampler::decoded(None);
        assert_eq!(r.output_rate(), None);
        let input = AudioFormat::new(44_100, 2);
        let samples = [1, -2, 3, -4];
        assert_eq!(r.push_samples(input, &samples).unwrap(), samples);
        assert_eq!(r.output_format(), Some(input));
        assert_eq!(r.output_rate(), Some(44_100));
        assert_eq!(r.input_format(), None);
        assert!(r.flush_samples().unwrap().is_empty());
    }
}
