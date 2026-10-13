//! `AudioStreamDecoder` (`utils/codecs/decoder.py:337-515`) as one synchronous calculator: `push`
//! returns the PCM its bytes produced; the channel, executor and framing are the driver's.

use crate::AudioFormat;
use crate::error::{CodecError, Result};
use crate::mime::mime_to_av_format;
use crate::{Quality, Resampler};
use crate::wav::WavDecoder;

enum Inner {
    Wav(WavDecoder),
    #[cfg(feature = "ffmpeg")]
    Ffmpeg(crate::ffmpeg::FfmpegDecoder),
}

/// One decoder decodes one stream, as in the reference.
pub struct Decoder {
    inner: Inner,
    /// `num_channels` is advisory on the WAV path, as the reference's `_WavInlineDecoder` only
    /// resamples the rate; the ffmpeg path honours it.
    target_rate: u32,
    #[allow(dead_code)]
    target_channels: u16,
    /// Per `(source format -> target rate)`: a provider may concatenate WAV files of different formats.
    resampler: Option<(AudioFormat, Box<dyn Resampler>)>,
    output_format: Option<AudioFormat>,
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

/// `AudioStreamDecoder`'s engine: bytes in, interleaved 16-bit PCM out, one decoder per stream.
#[lk_agents_abi::host(object(name = "AudioDecoder", cell = "mutex"))]
impl Decoder {
    /// `mime` is a media type, not a container name -- `"audio/mpeg"`, not `"mp3"`. An unknown
    /// or absent type asks ffmpeg to probe; without the `ffmpeg` feature it is an error, because
    /// the alternative is framing an ID3 tag as audio.
    #[host(new(kw_only = "sample_rate", defaults(sample_rate = 48000, num_channels = 1)))]
    pub fn new(mime: &str, sample_rate: u32, num_channels: u16) -> Result<Self> {
        let av_format = mime_to_av_format(Some(mime));
        let inner = match av_format {
            Some("wav") => Inner::Wav(WavDecoder::new()),
            #[cfg(feature = "ffmpeg")]
            other => {
                Inner::Ffmpeg(crate::ffmpeg::FfmpegDecoder::new(other, sample_rate, num_channels)?)
            }
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
        })
    }

    /// `None` until the first header is parsed.
    pub fn output_format(&self) -> Option<AudioFormat> {
        self.output_format
    }

    /// `(sample_rate, num_channels)` of what `push` emits, or `None` before the first header. For
    /// WAV the file's channel count wins over the requested one, as in the reference.
    #[host(getter(name = "output_format"))]
    pub fn output_format_pair(&self) -> Option<(u32, u16)> {
        self.output_format.map(|f| (f.sample_rate, f.num_channels))
    }

    /// [`Decoder::push`] as little-endian PCM bytes; often empty, which is not an error.
    #[host(method(name = "push", err = "RuntimeError"))]
    pub fn push_bytes(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        self.push(data).map(|pcm| crate::pcm_bytes(&pcm))
    }

    /// [`Decoder::flush`] as little-endian PCM bytes: the demuxer's and the resampler's tails.
    #[host(method(name = "flush", err = "RuntimeError"))]
    pub fn flush_bytes(&mut self) -> Result<Vec<u8>> {
        self.flush().map(|pcm| crate::pcm_bytes(&pcm))
    }

    /// Routinely empty: a push that only carried a header decodes to nothing.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<i16>> {
        match &mut self.inner {
            Inner::Wav(w) => {
                let segments = w.push(bytes)?;
                let mut out = Vec::new();
                for seg in segments {
                    out.extend(Self::convert(
                        &mut self.resampler,
                        &mut self.output_format,
                        self.target_rate,
                        seg.format,
                        &seg.samples,
                    )?);
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
                    Some((fmt, r)) => flush_tail(*fmt, self.target_rate, r.as_mut())?,
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

    /// A format change drains the previous resampler's tail before tearing it down.
    fn convert(
        resampler: &mut Option<(AudioFormat, Box<dyn Resampler>)>,
        output_format: &mut Option<AudioFormat>,
        target_rate: u32,
        src: AudioFormat,
        samples: &[i16],
    ) -> Result<Vec<i16>> {
        // `if self._sample_rate is not None and self._sample_rate != self._wave_rate`.
        if src.sample_rate == target_rate {
            let mut tail = Vec::new();
            if let Some((fmt, mut r)) = resampler.take() {
                tail = flush_tail(fmt, target_rate, r.as_mut())?;
            }
            *output_format = Some(src);
            if tail.is_empty() {
                return Ok(samples.to_vec());
            }
            tail.extend_from_slice(samples);
            return Ok(tail);
        }

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
        let mut d = Decoder::new("audio/wav", 24_000, 1).unwrap();
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
            let err = Decoder::new(mime, 48_000, 1).unwrap_err();
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
            let mut d = Decoder::new("audio/wav", rate, ch).unwrap();
            let total = d.push(&wav_file(src, ch, &ramp(n))).unwrap().len() + d.flush().unwrap().len();
            assert_eq!(total % usize::from(ch), 0, "whole frames");
            let got = total / usize::from(ch);
            assert!(frames.contains(&got), "{src} -> {rate} x{ch}: {got} frames, want {frames:?}");
            assert_eq!(d.output_format(), Some(AudioFormat::new(rate, ch)));
        }
    }

    // Catches: dropping the resampler's tail at end of stream. soxr always holds a filter delay;
    // losing it clips the last milliseconds of every utterance, which gets reported as "the agent
    // cuts itself off".
    #[test]
    fn flush_releases_the_resamplers_tail() {
        let mut d = Decoder::new("audio/wav", 24_000, 1).unwrap();
        let pushed = d.push(&wav_file(48_000, 1, &ramp(4800))).unwrap();
        let tail = d.flush().unwrap();
        assert!(!tail.is_empty(), "soxr holds a filter delay and flush must release it");
        assert!(pushed.len() < 2400, "the held tail is what is missing from the push");
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
        let mut d = Decoder::new("audio/wav", 24_000, 1).unwrap();
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
        let mut d = Decoder::new("audio/wav", 24_000, 1).unwrap();
        let first = d.push(&wav_file(48_000, 1, &ramp(4800))).unwrap();
        let second = d.push(&wav_file(24_000, 1, &ramp(240))).unwrap();
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
        assert!(first.len() + second.len() >= 2600);
    }

    // Catches: the WAV path quietly downmixing to the requested channel count. The reference's
    // fast path does NOT -- `_WavInlineDecoder` builds its `AudioByteStream` from the file's own
    // channel count and only resamples the rate -- so a port that downmixed would disagree with
    // the reference on every stereo WAV. Pinned so the divergence is a decision, not a drift.
    #[test]
    fn the_wav_path_keeps_the_files_channel_count_rather_than_the_requested_one() {
        let mut d = Decoder::new("audio/wav", 24_000, 1).unwrap();
        let interleaved: Vec<i16> = vec![1, -1, 2, -2, 3, -3];
        assert_eq!(d.push(&wav_file(24_000, 2, &interleaved)).unwrap(), interleaved);
        assert_eq!(d.output_format(), Some(AudioFormat::new(24_000, 2)));
    }

    // Catches: a header-only push being reported as an error or as silence-with-a-format. It is
    // neither: it is simply no audio yet, and the caller must be able to push again.
    #[test]
    fn a_push_that_only_carries_a_header_decodes_to_nothing_without_failing() {
        let file = wav_file(24_000, 1, &[9, 8, 7]);
        let mut d = Decoder::new("audio/wav", 24_000, 1).unwrap();
        assert!(d.push(&file[..44]).unwrap().is_empty());
        assert_eq!(d.output_format(), None);
        assert_eq!(d.push(&file[44..]).unwrap(), vec![9, 8, 7]);
    }
}
