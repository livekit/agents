//! `AudioStreamDecoder` (`utils/codecs/decoder.py:337-515`) as one synchronous calculator: `push`
//! returns the PCM its bytes produced; the channel, executor and framing are the driver's.

use crate::AudioFormat;
use crate::error::{CodecError, Result};
use crate::mime::mime_to_av_format;
use crate::wav::WavDecoder;
use crate::{Quality, Resampler};

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

enum Inner {
    Wav(WavDecoder),
    #[cfg(feature = "ffmpeg")]
    Ffmpeg(crate::ffmpeg::FfmpegDecoder),
}

/// Called from the decoding thread when output (or an error) is ready to drain with an empty
/// [`Decoder::push`]; must not block.
pub type OnOutput = std::sync::Arc<dyn Fn() + Send + Sync>;

/// One decoder decodes one stream, as in the reference.
pub struct Decoder {
    inner: Inner,
    /// `None` keeps the source's rate. `num_channels` is advisory on the WAV path, as the
    /// reference's `_WavInlineDecoder` only resamples the rate; the ffmpeg path honours it.
    target_rate: Option<u32>,
    #[allow(dead_code)]
    target_channels: u16,
    /// Per `(source format -> target rate)`: a provider may concatenate WAV files of different formats.
    resampler: Option<(AudioFormat, Box<dyn Resampler>)>,
    output_format: Option<AudioFormat>,
    limits: Option<DecodeLimits>,
    partial: Vec<i16>,
    partial_format: Option<AudioFormat>,
}

impl std::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // No `derive`: `SoxrResampler` wraps a raw `soxr_t` and is not `Debug`.
        f.debug_struct("Decoder")
            .field(
                "kind",
                &match self.inner {
                    Inner::Wav(_) => "wav",
                    #[cfg(feature = "ffmpeg")]
                    Inner::Ffmpeg(_) => "ffmpeg",
                },
            )
            .field("target_rate", &self.target_rate)
            .field("target_channels", &self.target_channels)
            .field("output_format", &self.output_format)
            .field("resampling", &self.resampler.is_some())
            .finish()
    }
}

/// Incremental bytes-to-PCM calculator.
impl Decoder {
    /// `mime` is a media type, not a container name -- `"audio/mpeg"`, not `"mp3"`. An unknown
    /// or absent type asks ffmpeg to probe; without the `ffmpeg` feature it is an error, because
    /// the alternative is framing an ID3 tag as audio. `sample_rate=None` keeps the source's rate.
    pub fn new(mime: &str, sample_rate: Option<u32>, num_channels: u16) -> Result<Self> {
        Self::with_optional_workers(None, mime, sample_rate, num_channels)
    }

    /// Compressed containers execute on the caller's admitted native worker pool.
    pub fn with_workers(
        workers: &crate::workers::MediaWorkers,
        mime: &str,
        sample_rate: Option<u32>,
        num_channels: u16,
    ) -> Result<Self> {
        Self::with_optional_workers(Some(workers), mime, sample_rate, num_channels)
    }

    fn with_optional_workers(
        workers: Option<&crate::workers::MediaWorkers>,
        mime: &str,
        sample_rate: Option<u32>,
        num_channels: u16,
    ) -> Result<Self> {
        #[cfg(not(feature = "ffmpeg"))]
        let _ = workers;
        let av_format = mime_to_av_format(Some(mime));
        let inner = match av_format {
            Some("wav") => Inner::Wav(WavDecoder::new()),
            #[cfg(feature = "ffmpeg")]
            other => Inner::Ffmpeg(crate::ffmpeg::FfmpegDecoder::new(
                workers.ok_or(CodecError::DecodeLimit(
                    "container decoding requires an injected worker pool",
                ))?,
                other,
                sample_rate.unwrap_or(0),
                num_channels,
            )?),
            #[cfg(not(feature = "ffmpeg"))]
            Some(other) => {
                return Err(CodecError::UnsupportedMime {
                    mime: mime.to_owned(),
                    reason: match other {
                        "mp3" => "mp3 needs a demuxer",
                        "aac" => "aac needs a demuxer",
                        "flac" => "flac needs a demuxer",
                        "ogg" => "ogg/opus needs a demuxer",
                        "webm" => "matroska/webm needs a demuxer",
                        _ => "this container needs a demuxer",
                    },
                });
            }
            #[cfg(not(feature = "ffmpeg"))]
            None => {
                return Err(CodecError::UnsupportedMime {
                    mime: mime.to_owned(),
                    reason: "unrecognised media type, and probing needs a demuxer",
                });
            }
        };
        Ok(Decoder {
            inner,
            target_rate: sample_rate,
            target_channels: num_channels,
            resampler: None,
            output_format: None,
            limits: None,
            partial: Vec::new(),
            partial_format: None,
        })
    }

    /// Construct the supplied-chunk WAV calculator with runtime-selected transform bounds.
    pub fn new_bounded(
        mime: &str,
        sample_rate: Option<u32>,
        num_channels: u16,
        limits: DecodeLimits,
    ) -> Result<Self> {
        if !limits.is_valid() {
            return Err(CodecError::DecodeLimit("invalid limits"));
        }
        if mime_to_av_format(Some(mime)) != Some("wav") {
            return Err(CodecError::DecodeLimit("container decoding requires a runtime worker"));
        }
        let mut decoder = Self::new(mime, sample_rate, num_channels)?;
        decoder.limits = Some(limits);
        Ok(decoder)
    }

    /// `None` until the first header is parsed.
    pub fn output_format(&self) -> Option<AudioFormat> {
        self.output_format
    }

    /// `(sample_rate, num_channels)` of what `push` emits, or `None` before the first header. For
    /// WAV the file's channel count wins over the requested one, as in the reference.
    pub fn output_format_pair(&self) -> Option<(u32, u16)> {
        self.output_format.map(|f| (f.sample_rate, f.num_channels))
    }

    /// [`Decoder::push`] as little-endian PCM bytes; often empty, which is not an error.
    pub fn push_bytes(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        self.push(data).map(|pcm| crate::pcm_bytes(&pcm))
    }

    /// [`Decoder::flush`] as little-endian PCM bytes: the demuxer's and the resampler's tails.
    // The GIL is released while the decode thread finishes: its `on_output` may be waiting on it.
    pub fn flush_bytes(&mut self) -> Result<Vec<u8>> {
        self.flush().map(|pcm| crate::pcm_bytes(&pcm))
    }

    /// A container decodes on its own thread, so its PCM is ready between pushes: `on_output`
    /// says when. A WAV decodes inside `push` and never calls it.
    pub fn set_on_output(&self, on_output: Option<OnOutput>) {
        match &self.inner {
            Inner::Wav(_) => drop(on_output),
            #[cfg(feature = "ffmpeg")]
            Inner::Ffmpeg(f) => f.set_on_output(on_output),
        }
    }

    /// Routinely empty: a push that only carried a header decodes to nothing.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<i16>> {
        match &mut self.inner {
            Inner::Wav(w) => {
                let segments = w.push(bytes)?;
                let mut out = Vec::new();
                for seg in segments {
                    if let Some(limits) = self.limits {
                        limits.validate(
                            seg.format,
                            self.target_rate,
                            seg.samples.len().saturating_add(self.partial.len()),
                        )?;
                    }
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
                        out.extend(Self::convert(
                            &mut self.resampler,
                            &mut self.output_format,
                            self.target_rate,
                            seg.format,
                            &seg.samples,
                        )?);
                        continue;
                    }
                    self.partial.extend(seg.samples);
                    let complete = self.partial.len() / channels * channels;
                    out.extend(Self::convert(
                        &mut self.resampler,
                        &mut self.output_format,
                        self.target_rate,
                        seg.format,
                        &self.partial[..complete],
                    )?);
                    self.partial.drain(..complete);
                }
                Ok(out)
            }
            #[cfg(feature = "ffmpeg")]
            Inner::Ffmpeg(f) => {
                let pcm = f.push(bytes)?;
                self.output_format = f.output_format();
                Ok(pcm)
            }
        }
    }

    pub fn flush(&mut self) -> Result<Vec<i16>> {
        match &mut self.inner {
            Inner::Wav(w) => {
                w.flush();
                Ok(match self.resampler.as_mut() {
                    Some((fmt, r)) => {
                        flush_tail(*fmt, self.target_rate.unwrap_or(fmt.sample_rate), r.as_mut())?
                    }
                    None => Vec::new(),
                })
            }
            #[cfg(feature = "ffmpeg")]
            Inner::Ffmpeg(f) => {
                let pcm = f.flush()?;
                self.output_format = f.output_format();
                Ok(pcm)
            }
        }
    }

    /// Cancel a live container and hand native settlement back to its owner.
    pub fn close(&mut self) -> Option<crate::workers::MediaWorker<()>> {
        match &mut self.inner {
            Inner::Wav(_) => None,
            #[cfg(feature = "ffmpeg")]
            Inner::Ffmpeg(decoder) => decoder.close(),
        }
    }

    /// Validate terminal input before draining. `flush` remains available for mid-stream drift.
    pub fn finish(&mut self) -> Result<Vec<i16>> {
        match &self.inner {
            Inner::Wav(wav) => wav.end_input()?,
            #[cfg(feature = "ffmpeg")]
            Inner::Ffmpeg(_) => {}
        }
        if !self.partial.is_empty() {
            return Err(CodecError::TruncatedWav);
        }
        self.flush()
    }

    /// A format change drains the previous resampler's tail before tearing it down.
    fn convert(
        resampler: &mut Option<(AudioFormat, Box<dyn Resampler>)>,
        output_format: &mut Option<AudioFormat>,
        target_rate: Option<u32>,
        src: AudioFormat,
        samples: &[i16],
    ) -> Result<Vec<i16>> {
        // `if self._sample_rate is not None and self._sample_rate != self._wave_rate`.
        let Some(target_rate) = target_rate.filter(|r| *r != src.sample_rate) else {
            let mut tail = Vec::new();
            if let Some((fmt, mut r)) = resampler.take() {
                tail = flush_tail(fmt, src.sample_rate, r.as_mut())?;
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
            prefix = flush_tail(fmt, target_rate, r.as_mut())?;
        }
        if resampler.is_none() {
            // `rtc.AudioResampler`'s default, as the reference builds it.
            let r = crate::new_resampler(
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
fn flush_tail(src: AudioFormat, target_rate: u32, r: &mut dyn Resampler) -> Result<Vec<i16>> {
    r.flush().map(<[i16]>::to_vec).map_err(|e| CodecError::Resample {
        from: src.sample_rate,
        to: target_rate,
        detail: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::tests::wav_file;

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
        let mut d = Decoder::new("audio/wav", Some(24_000), 1).unwrap();
        let got = d.push(&wav_file(24_000, 1, &want)).unwrap();
        assert_eq!(got, want);
        assert_eq!(d.output_format(), Some(AudioFormat::new(24_000, 1)));
        assert!(d.flush().unwrap().is_empty());
    }

    // Catches: accepting a container this build cannot decode and returning empty PCM, which
    // reads downstream as "the TTS sent silence" rather than "this binary cannot decode mp3".
    // `AudioEmitter` raises `NotImplementedError` for exactly this reason and the replacement
    // must be no quieter.
    #[cfg(not(feature = "ffmpeg"))]
    #[test]
    fn a_container_mime_type_is_refused_when_ffmpeg_is_not_compiled_in() {
        for mime in ["audio/mpeg", "audio/aac", "audio/flac", "audio/ogg", "audio/webm;codecs=opus"]
        {
            let err = Decoder::new(mime, Some(48_000), 1).unwrap_err();
            assert!(matches!(err, CodecError::UnsupportedMime { .. }), "{mime}: {err:?}");
            // The message has to name the feature, or the next person greps the codebase for it.
            assert!(err.to_string().contains("ffmpeg"), "{mime}: {err}");
        }
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
            let mut d = Decoder::new("audio/wav", Some(rate), ch).unwrap();
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
        let mut d = Decoder::new("audio/wav", None, 1).unwrap();
        assert_eq!(d.push(&wav_file(16_000, 1, &ramp(160))).unwrap(), ramp(160));
        assert_eq!(d.output_format(), Some(AudioFormat::new(16_000, 1)));
    }

    // Catches: dropping the resampler's tail at end of stream. soxr always holds a filter delay;
    // losing it clips the last milliseconds of every utterance, which gets reported as "the agent
    // cuts itself off".
    #[test]
    fn flush_releases_the_resamplers_tail() {
        let mut d = Decoder::new("audio/wav", Some(48_000), 1).unwrap();
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
        let mut d = Decoder::new("audio/wav", Some(24_000), 1).unwrap();
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
        let mut d = Decoder::new("audio/wav", Some(48_000), 1).unwrap();
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

    // Catches: the WAV path quietly downmixing to the requested channel count. The reference's
    // fast path does NOT -- `_WavInlineDecoder` builds its `AudioByteStream` from the file's own
    // channel count and only resamples the rate -- so a port that downmixed would disagree with
    // the reference on every stereo WAV. Pinned so the divergence is a decision, not a drift.
    #[test]
    fn the_wav_path_keeps_the_files_channel_count_rather_than_the_requested_one() {
        let mut d = Decoder::new("audio/wav", Some(24_000), 1).unwrap();
        let interleaved: Vec<i16> = vec![1, -1, 2, -2, 3, -3];
        assert_eq!(d.push(&wav_file(24_000, 2, &interleaved)).unwrap(), interleaved);
        assert_eq!(d.output_format(), Some(AudioFormat::new(24_000, 2)));
    }

    // Catches: a header-only push being reported as an error or as silence-with-a-format. It is
    // neither: it is simply no audio yet, and the caller must be able to push again.
    #[test]
    fn a_push_that_only_carries_a_header_decodes_to_nothing_without_failing() {
        let file = wav_file(24_000, 1, &[9, 8, 7]);
        let mut d = Decoder::new("audio/wav", Some(24_000), 1).unwrap();
        assert!(d.push(&file[..44]).unwrap().is_empty());
        assert_eq!(d.output_format(), None);
        assert_eq!(d.push(&file[44..]).unwrap(), vec![9, 8, 7]);
    }
}
