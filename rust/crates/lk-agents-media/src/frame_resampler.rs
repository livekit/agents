//! Frames in, frames out, over [`new_resampler`](crate::new_resampler): the one frame-level
//! resampler behind the playout chain, the converted input leg, the pre-connect buffer, the
//! fallback TTS, the VAD and the recorder. What the sites disagree on is explicit here, in
//! [`OnChange`] and in the choice between [`FrameResampler::new`] and [`FrameResampler::fixed`].

use crate::{AudioFormat, Quality, ResampleError, Resampler};
use lk_agents_abi::AudioFrame;

/// What a frame in a new format (rate or channel count) does to the stream before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnChange {
    /// The old stream's tail comes out first and the stream ends. A frame at the output rate
    /// passes through and ends the stream too, and so does [`FrameResampler::flush`]: the next
    /// off-rate frame starts a fresh one. The playout chain in front of a sink.
    Drain,
    /// The old stream is dropped unflushed once its replacement is built (a failed build keeps
    /// it). A frame at the output rate passes through and leaves the stream as it was, and
    /// [`FrameResampler::flush`] keeps it. The converted input leg and the recorder.
    Discard,
}

enum Mode {
    /// Built at the first off-rate frame, from the frame's own header.
    Follow(OnChange),
    /// Built up front; every frame is read as that format, whatever its header says.
    Fixed,
}

struct Stream {
    resampler: Box<dyn Resampler>,
    input: AudioFormat,
}

/// A resampler that owns its framing: every frame it emits is at the output rate, carries the
/// input's channel count, and has `samples_per_channel` equal to its data. soxr buffers, so a
/// push may emit nothing.
pub struct FrameResampler {
    output_rate: u32,
    quality: Quality,
    mode: Mode,
    stream: Option<Stream>,
}

impl FrameResampler {
    /// Follows the frames' headers: nothing is built until the first frame off `output_rate`, and
    /// a header 0 channels wide is resampled as mono.
    pub fn new(output_rate: u32, quality: Quality, on_change: OnChange) -> Self {
        FrameResampler { output_rate, quality, mode: Mode::Follow(on_change), stream: None }
    }

    /// For a stream whose format is declared rather than read off each frame, as the reference
    /// builds `rtc.AudioResampler` once: built now, even for a 1:1 ratio, and with exactly
    /// `input.num_channels` channels.
    pub fn fixed(input: AudioFormat, output_rate: u32, quality: Quality) -> Result<Self, ResampleError> {
        let resampler =
            crate::new_resampler(input.sample_rate, output_rate, u32::from(input.num_channels), quality)?;
        Ok(FrameResampler { output_rate, quality, mode: Mode::Fixed, stream: Some(Stream { resampler, input }) })
    }

    pub fn output_rate(&self) -> u32 {
        self.output_rate
    }

    /// The format the current stream was built for; `None` before the first off-rate frame, or
    /// after a drained one.
    pub fn input_format(&self) -> Option<AudioFormat> {
        self.stream.as_ref().map(|s| s.input)
    }

    /// The frames now ready: zero or one resampled frame, a drained tail ahead of it, or the frame
    /// itself where it passes through. An error from building a stream leaves the frame unread.
    pub fn push(&mut self, frame: &AudioFrame) -> Result<Vec<AudioFrame>, ResampleError> {
        let mut out = Vec::new();
        if let Mode::Follow(on_change) = self.mode {
            let format = AudioFormat::new(frame.sample_rate, frame.num_channels);
            match on_change {
                OnChange::Drain => {
                    if self.input_format() != Some(format) {
                        out.extend(self.flush()?);
                        if frame.sample_rate == self.output_rate {
                            out.push(frame.clone());
                            return Ok(out);
                        }
                        self.stream = Some(self.build(format)?);
                    }
                }
                OnChange::Discard => {
                    if frame.sample_rate == self.output_rate {
                        out.push(frame.clone());
                        return Ok(out);
                    }
                    if self.input_format() != Some(format) {
                        self.stream = Some(self.build(format)?);
                    }
                }
            }
        }
        let stream = self.stream.as_mut().expect("built above, or at construction");
        let pcm: Vec<i16> =
            frame.data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
        let resampled = stream.resampler.push(&pcm)?;
        out.extend(to_frame(resampled, self.output_rate, stream.input.num_channels));
        Ok(out)
    }

    /// The current stream's tail, if any. The resampler resets either way; under
    /// [`OnChange::Drain`] the stream also ends.
    pub fn flush(&mut self) -> Result<Option<AudioFrame>, ResampleError> {
        let output_rate = self.output_rate;
        if matches!(self.mode, Mode::Follow(OnChange::Drain)) {
            let Some(mut stream) = self.stream.take() else { return Ok(None) };
            let tail = stream.resampler.flush()?;
            return Ok(to_frame(tail, output_rate, stream.input.num_channels));
        }
        let Some(stream) = self.stream.as_mut() else { return Ok(None) };
        let tail = stream.resampler.flush()?;
        Ok(to_frame(tail, output_rate, stream.input.num_channels))
    }

    /// Drops the stream unflushed; the next off-rate frame builds a fresh one.
    pub fn reset(&mut self) {
        self.stream = None;
    }

    fn build(&self, input: AudioFormat) -> Result<Stream, ResampleError> {
        let channels = u32::from(input.num_channels.max(1));
        let resampler = crate::new_resampler(input.sample_rate, self.output_rate, channels, self.quality)?;
        Ok(Stream { resampler, input })
    }
}

fn to_frame(samples: &[i16], rate: u32, channels: u16) -> Option<AudioFrame> {
    if samples.is_empty() {
        return None;
    }
    let spc = (samples.len() / usize::from(channels.max(1))) as u32;
    Some(AudioFrame::new(crate::pcm_bytes(samples), rate, channels, spc))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Sample values are compared only where they pass through: libsoxr dithers every int16
    // sample with a per-instance seed, so two resamplers fed the same calls agree on lengths and
    // shapes, not on the last bit.

    fn ramp(rate: u32, channels: u16, samples_per_channel: usize) -> AudioFrame {
        let pcm: Vec<i16> =
            (0..samples_per_channel * usize::from(channels)).map(|i| ((i * 37) % 2000) as i16 - 1000).collect();
        AudioFrame::new(crate::pcm_bytes(&pcm), rate, channels, samples_per_channel as u32)
    }

    fn samples(frame: &AudioFrame) -> Vec<i16> {
        frame.data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect()
    }

    /// Output lengths of a bare resampler over `pushes`, then its flush.
    fn lengths(from: u32, to: u32, channels: u32, pushes: &[&AudioFrame]) -> (Vec<usize>, usize) {
        let mut r = crate::new_resampler(from, to, channels, Quality::Medium).unwrap();
        let out = pushes.iter().map(|f| r.push(&samples(f)).unwrap().len()).collect();
        (out, r.flush().unwrap().len())
    }

    // Catches: a rate change that drops the old stream's tail (the end of a sentence goes missing
    // at every TTS rate switch), or a passthrough frame emitted ahead of that tail.
    #[test]
    fn drain_puts_the_old_streams_tail_ahead_of_a_frame_at_the_output_rate() {
        let mut r = FrameResampler::new(16_000, Quality::Medium, OnChange::Drain);
        // Not a whole number of output periods, so the linear fallback holds a tail too.
        let off = ramp(48_000, 1, 4_801);
        let on = ramp(16_000, 1, 320);
        r.push(&off).unwrap();
        let out = r.push(&on).unwrap();
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
        let mut r = FrameResampler::new(16_000, Quality::Medium, OnChange::Discard);
        // Sized so both soxr (its delay) and the linear fallback (its carried phase) emit a
        // different count for `b` depending on what came before it.
        let (a, b) = (ramp(44_100, 1, 4_411), ramp(44_100, 1, 1_000));
        let on = ramp(16_000, 1, 160);
        r.push(&a).unwrap();
        assert_eq!(r.push(&on).unwrap(), vec![on]);
        let second: usize = r.push(&b).unwrap().iter().map(|f| f.data.len() / 2).sum();
        let (continuous, _) = lengths(44_100, 16_000, 1, &[&a, &b]);
        let (fresh, _) = lengths(44_100, 16_000, 1, &[&b]);
        assert_ne!(continuous[1], fresh[0], "the lengths tell a kept stream from a new one");
        assert_eq!(second, continuous[1], "the stream from before the passthrough carried on");
    }

    // Catches: `Discard` draining the old stream into the new rate's output on a rate change.
    #[test]
    fn discard_drops_the_old_stream_unflushed_on_a_rate_change() {
        let mut r = FrameResampler::new(16_000, Quality::Medium, OnChange::Discard);
        let (a, b) = (ramp(48_000, 1, 4_801), ramp(24_000, 1, 2_400));
        r.push(&a).unwrap();
        let out = r.push(&b).unwrap();
        let (fresh, _) = lengths(24_000, 16_000, 1, &[&b]);
        assert!(out.len() <= 1, "no tail frame");
        assert_eq!(out.iter().map(|f| f.data.len() / 2).sum::<usize>(), fresh[0]);
        assert_eq!(r.input_format(), Some(AudioFormat::new(24_000, 1)));
    }

    // Catches: a failed rebuild that leaves the stream half-replaced (the converted input leg
    // skips the frame and must carry on with the stream it had).
    #[test]
    fn a_failed_build_keeps_the_stream_it_would_have_replaced() {
        let mut r = FrameResampler::new(16_000, Quality::Medium, OnChange::Discard);
        r.push(&ramp(48_000, 1, 480)).unwrap();
        assert!(matches!(r.push(&ramp(0, 1, 480)), Err(ResampleError::Create(_))));
        assert_eq!(r.input_format(), Some(AudioFormat::new(48_000, 1)));
    }

    // Catches: a fixed resampler that follows the header -- the VAD reads every frame as mono
    // and the fallback TTS as its entry's declared rate, whatever the frame claims.
    #[test]
    fn fixed_reads_every_frame_as_the_declared_format() {
        let mut r = FrameResampler::fixed(AudioFormat::new(48_000, 1), 16_000, Quality::Medium).unwrap();
        let liar = ramp(24_000, 2, 2_400);
        let out = r.push(&liar).unwrap();
        let (want, _) = lengths(48_000, 16_000, 1, &[&liar]);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].sample_rate, out[0].num_channels), (16_000, 1));
        assert_eq!((out[0].data.len() / 2, out[0].samples_per_channel as usize), (want[0], want[0]));
        assert_eq!(r.input_format(), Some(AudioFormat::new(48_000, 1)));
    }

    // Catches: building with the header's 0 channels (soxr would refuse every push), or rewriting
    // the header the frame came with.
    #[test]
    fn a_zero_channel_header_is_resampled_as_mono_and_keeps_its_header() {
        let mut r = FrameResampler::new(16_000, Quality::Medium, OnChange::Drain);
        let mono = ramp(48_000, 1, 4_800);
        let mut out = r.push(&AudioFrame { num_channels: 0, ..mono }).unwrap();
        out.extend(r.flush().unwrap());
        let total: usize = out.iter().map(|f| f.data.len() / 2).sum();
        assert!(total > 1_500, "resampled as one channel: {total} samples");
        assert!(out.iter().all(|f| f.num_channels == 0 && f.samples_per_channel as usize * 2 == f.data.len()));
    }
}
