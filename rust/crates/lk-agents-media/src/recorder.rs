//! `voice/recorder_io/`: both sides of a call in one stereo ogg/opus file. Both channels sit on one
//! absolute timeline and silence is whatever nothing was written over (#6921).

use std::collections::VecDeque;

use lk_agents_abi::AudioFrame;

use crate::{FrameResampler, OnChange, Quality, ResampleError};

mod file;
pub use file::{BetweenBlocks, FileLimits, RecordingCommand, RecordingFile, RecordingLimits};

/// Python's `round()` is half-to-even; a one-sample disagreement at `.5` is an overlap or gap.
#[inline]
fn round_ties_even_like_python(v: f64) -> f64 {
    v.round_ties_even()
}

pub const WRITE_INTERVAL: f64 = 2.5;

/// How long the writer waits on a stalled source before taking the silence as real.
pub const INPUT_STALL_TIMEOUT: f64 = 1.0;

/// Re-anchoring beyond this keeps a drifting capture clock from sliding the channel.
pub const RESYNC_TOLERANCE: f64 = 0.1;

pub const INPUT_CHANNEL: usize = 0;
pub const OUTPUT_CHANNEL: usize = 1;

/// Interleaved PCM (`rtc.AudioFrame` on the host side).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcmFrame {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
    pub num_channels: u16,
}

impl PcmFrame {
    pub fn new(samples: Vec<i16>, sample_rate: u32, num_channels: u16) -> Self {
        PcmFrame { samples, sample_rate, num_channels }
    }

    pub fn samples_per_channel(&self) -> usize {
        self.samples.len() / self.num_channels.max(1) as usize
    }

    pub fn duration(&self) -> f64 {
        self.samples_per_channel() as f64 / f64::from(self.sample_rate)
    }
}

#[cfg(feature = "ffmpeg")]
impl OggOpusEncoder for crate::ffmpeg::OggOpusEncoder {
    fn push(&mut self, pcm: &[i16]) -> Result<Vec<u8>, EncodeError> {
        crate::ffmpeg::OggOpusEncoder::push(self, pcm).map_err(|e| EncodeError(e.to_string()))
    }

    fn finish(self) -> Result<Vec<u8>, EncodeError> {
        crate::ffmpeg::OggOpusEncoder::finish(self).map_err(|e| EncodeError(e.to_string()))
    }
}

/// Still a trait because the real encoder is behind the non-default `ffmpeg` feature.
pub trait OggOpusEncoder {
    fn push(&mut self, pcm: &[i16]) -> Result<Vec<u8>, EncodeError>;
    fn finish(self) -> Result<Vec<u8>, EncodeError>
    where
        Self: Sized;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeError(pub String);

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EncodeError {}

/// Keeps the PCM instead of encoding it; public because the driver's tests need it too.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PcmRecordingEncoder {
    pub pcm: Vec<i16>,
}

/// Non-empty on purpose, so a recorder that never calls `finish` is distinguishable.
pub const PCM_ENCODER_TRAILER: &[u8] = b"EOS";

impl OggOpusEncoder for PcmRecordingEncoder {
    fn push(&mut self, pcm: &[i16]) -> Result<Vec<u8>, EncodeError> {
        self.pcm.extend_from_slice(pcm);
        Ok(Vec::new())
    }

    fn finish(self) -> Result<Vec<u8>, EncodeError> {
        Ok(PCM_ENCODER_TRAILER.to_vec())
    }
}

/// Samples are `f32` because overlapping placements are summed, not replaced.
pub struct Track {
    sample_rate: u32,
    t0: f64,
    placed: Vec<(i64, VecDeque<f32>)>,
    /// Mono, rebuilt unflushed when the source rate changes.
    resampler: FrameResampler,
    run_start: Option<f64>,
    run_samples: i64,
    received_until: Option<f64>,
    dropped_samples: u64,
}

impl Track {
    pub fn new(sample_rate: u32, t0: f64) -> Self {
        Track {
            sample_rate,
            t0,
            placed: Vec::new(),
            resampler: FrameResampler::new(sample_rate, Quality::Medium, OnChange::Discard),
            run_start: None,
            run_samples: 0,
            received_until: None,
            dropped_samples: 0,
        }
    }

    pub fn retained_bytes(&self) -> usize {
        self.placed.iter().map(|(_, samples)| samples.capacity() * std::mem::size_of::<f32>()).sum()
    }

    /// Late audio; the driver warns on a non-zero count at close.
    pub fn dropped_samples(&self) -> u64 {
        self.dropped_samples
    }

    fn resample(&mut self, frame: &PcmFrame) -> Vec<i16> {
        let mono = crate::downmix_mono(&frame.samples, frame.num_channels);

        if frame.sample_rate == self.sample_rate {
            return mono;
        }
        let frame =
            AudioFrame::new(crate::pcm_bytes(&mono), frame.sample_rate, 1, mono.len() as u32);
        match self.resampler.push(&frame) {
            Ok(ready) => {
                ready.first().map(|f| PcmFrame::from_audio_frame(f).samples).unwrap_or_default()
            }
            // soxr refused the rate pair: wrong-rate audio is recoverable, silence is not. The
            // next frame asks again.
            Err(ResampleError::Create(_)) => {
                self.resampler.reset();
                mono
            }
            // Swallowed to an empty buffer: the reference has no per-frame error path, and
            // dropping one frame beats losing the rest of the recording.
            Err(_) => Vec::new(),
        }
    }

    /// While timestamps stay within [`RESYNC_TOLERANCE`] of the run's own clock the run keeps
    /// counting samples, so drift does not slide the channel and a real gap re-anchors.
    pub fn push(&mut self, started_at: f64, frame: &PcmFrame) {
        let expected = self
            .run_start
            .map(|start| start + self.run_samples as f64 / f64::from(self.sample_rate));
        let resync = match expected {
            None => true,
            Some(expected) => (started_at - expected).abs() > RESYNC_TOLERANCE,
        };
        if resync {
            // The resampler's tail belongs to the run that just ended: place it before re-anchoring.
            if let Ok(Some(tail)) = self.resampler.flush() {
                self.place(&PcmFrame::from_audio_frame(&tail).samples);
            }
            self.run_start = Some(started_at);
            self.run_samples = 0;
        }
        let samples = self.resample(frame);
        self.place(&samples);
        self.received_until = Some(started_at + frame.duration());
    }

    fn place(&mut self, samples: &[i16]) {
        if samples.is_empty() {
            return;
        }
        let run_start = self.run_start.expect("a run is open whenever samples are placed");
        let start = round_ties_even_like_python((run_start - self.t0) * f64::from(self.sample_rate))
            as i64
            + self.run_samples;
        let normalized = samples.iter().map(|&s| f32::from(s) / 32768.0);
        if let Some((position, previous)) = self.placed.last_mut()
            && *position + previous.len() as i64 == start
        {
            previous.extend(normalized);
        } else {
            self.placed.push((start, normalized.collect()));
        }
        self.run_samples += samples.len() as i64;
    }

    fn emitted_until(&self) -> Option<f64> {
        self.run_start.map(|start| start + self.run_samples as f64 / f64::from(self.sample_rate))
    }

    fn pending_resampler_until(&self, now: f64) -> Option<f64> {
        let received = self.received_until?;
        let emitted = self.emitted_until()?;
        (self.resampler.input_format().is_some()
            && emitted < received
            && now - received < INPUT_STALL_TIMEOUT)
            .then_some(emitted)
    }

    fn flush_resampler(&mut self) -> Result<(), ResampleError> {
        if let Some(tail) = self.resampler.flush()? {
            self.place(&PcmFrame::from_audio_frame(&tail).samples);
        }
        Ok(())
    }

    /// Placements ending before `start` are dropped and counted; ones past `end` are re-anchored
    /// at `end`, so the tail appears exactly once in the next block.
    pub fn take(&mut self, start: i64, end: i64) -> Vec<f32> {
        let mut block = vec![0.0_f32; (end - start).max(0) as usize];
        let mut keep: Vec<(i64, VecDeque<f32>)> = Vec::new();
        for (pos, mut samples) in std::mem::take(&mut self.placed) {
            let stop = pos + samples.len() as i64;
            if stop <= start {
                self.dropped_samples += samples.len() as u64;
                continue;
            }
            if pos >= end {
                keep.push((pos, samples));
                continue;
            }

            let lo = pos.max(start);
            let hi = stop.min(end);
            for i in lo..hi {
                // `+=`: overlapping runs are mixed, not overwritten.
                block[(i - start) as usize] += samples[(i - pos) as usize];
            }
            if stop > end {
                samples.drain(..(end - pos) as usize);
                keep.push((end, samples));
            }
        }
        self.placed = keep;
        block
    }
}

pub struct Mixer {
    tracks: [Track; 2],
    sample_rate: u32,
    t0: f64,
    cursor: i64,
}

impl Mixer {
    pub fn new(sample_rate: u32, t0: f64) -> Self {
        Mixer {
            tracks: [Track::new(sample_rate, t0), Track::new(sample_rate, t0)],
            sample_rate,
            t0,
            cursor: 0,
        }
    }

    pub fn retained_bytes(&self) -> usize {
        self.tracks.iter().map(Track::retained_bytes).sum()
    }

    pub fn push(&mut self, channel: usize, started_at: f64, frame: &PcmFrame) {
        self.tracks[channel].push(started_at, frame);
    }

    fn flush_resamplers(&mut self) -> Result<(), ResampleError> {
        for track in &mut self.tracks {
            track.flush_resampler()?;
        }
        Ok(())
    }

    pub fn dropped_samples(&self) -> [u64; 2] {
        [self.tracks[0].dropped_samples(), self.tracks[1].dropped_samples()]
    }

    /// The cursor only moves forward, so a flush for a moment already written produces nothing.
    pub fn flush_until(&mut self, until: f64) -> Option<Vec<i16>> {
        let end =
            round_ties_even_like_python((until - self.t0) * f64::from(self.sample_rate)) as i64;
        if end <= self.cursor {
            return None;
        }
        let left = self.tracks[0].take(self.cursor, end);
        let right = self.tracks[1].take(self.cursor, end);
        self.cursor = end;

        // The encoder takes interleaved i16; only samples summed past full scale change here.
        let mut out = Vec::with_capacity(left.len() * 2);
        for (l, r) in left.iter().zip(right.iter()) {
            out.push(to_i16(*l));
            out.push(to_i16(*r));
        }
        Some(out)
    }
}

fn to_i16(v: f32) -> i16 {
    (v * 32768.0).clamp(-32768.0, 32767.0) as i16
}

#[derive(Debug, Clone, PartialEq)]
pub struct Placed {
    pub started_at: f64,
    pub frame: PcmFrame,
}

/// `RecorderAudioOutput`'s state: the segment's frames and what the sink said about where they went.
#[derive(Debug, Default, Clone)]
pub struct SegmentPlacer {
    pcm: VecDeque<i16>,
    /// Absolute interleaved offset of the first retained sample in this segment.
    offset: usize,
    format: Option<(u32, u16)>,
    segment_since: Option<f64>,
    started_at: Option<f64>,
    reported: bool,
}

impl SegmentPlacer {
    pub fn new() -> Self {
        SegmentPlacer::default()
    }

    pub fn pending_since(&self) -> Option<f64> {
        self.segment_since
    }

    pub fn retained_bytes(&self) -> usize {
        self.pcm.capacity() * std::mem::size_of::<i16>()
    }

    pub fn pending_bytes(&self) -> usize {
        self.pcm.len() * std::mem::size_of::<i16>()
    }

    pub fn capture_frame(&mut self, now: f64, frame: &PcmFrame) {
        if self.pcm.is_empty() {
            self.segment_since = Some(now);
        }
        if self.format.is_none() {
            self.format = Some((frame.sample_rate, frame.num_channels));
        }
        self.pcm.reserve(frame.samples.len());
        self.pcm.extend(frame.samples.iter().copied());
    }

    /// The first `created_at` wins, so a paused and resumed segment keeps its anchor.
    pub fn on_playback_started(&mut self, created_at: f64) {
        if self.started_at.is_none() {
            self.started_at = Some(created_at);
        }
    }

    /// A jump in `offset` is audio the sink discarded, so it is honoured (#6728).
    pub fn on_playback_progressed(
        &mut self,
        started_at: f64,
        offset: f64,
        duration: f64,
    ) -> Option<Placed> {
        self.progress(started_at, offset, duration, None)
    }

    /// Place at most one bounded interval; advance from the returned timestamp and duration.
    pub fn on_playback_progressed_chunk(
        &mut self,
        started_at: f64,
        offset: f64,
        duration: f64,
        max_duration: f64,
    ) -> Option<Placed> {
        self.progress(started_at, offset, duration, Some(max_duration))
    }

    fn progress(
        &mut self,
        started_at: f64,
        offset: f64,
        duration: f64,
        max_duration: Option<f64>,
    ) -> Option<Placed> {
        self.reported = true;
        let placed = self.place(started_at, offset, duration, max_duration)?;
        self.segment_since = if self.pcm.is_empty() {
            None
        } else {
            Some(placed.started_at + placed.frame.duration())
        };
        Some(placed)
    }

    /// Fallback for a sink that reported nothing: its endpoints place the segment.
    pub fn on_playback_finished(&mut self, now: f64, playback_position: f64) -> Option<Placed> {
        self.finish(now, playback_position, None)
    }

    /// Drain fallback placement in bounded intervals, then retire the segment on exhaustion.
    pub fn on_playback_finished_chunk(
        &mut self,
        now: f64,
        playback_position: f64,
        max_duration: f64,
    ) -> Option<Placed> {
        self.finish(now, playback_position, Some(max_duration))
    }

    fn finish(
        &mut self,
        now: f64,
        playback_position: f64,
        max_duration: Option<f64>,
    ) -> Option<Placed> {
        let placed = if !self.reported && !self.pcm.is_empty() {
            let started_at = self.started_at.unwrap_or(now - playback_position);
            self.place(started_at, 0.0, playback_position, max_duration)
        } else {
            None
        };
        if placed.is_none() || max_duration.is_none() {
            *self = SegmentPlacer::new();
        }
        placed
    }

    fn place(
        &mut self,
        started_at: f64,
        offset: f64,
        duration: f64,
        max_duration: Option<f64>,
    ) -> Option<Placed> {
        // A report covering no time places nothing (#5279).
        if self.pcm.is_empty() || duration <= 0.0 {
            return None;
        }
        let (rate, channels) = self.format?;
        let ch = i64::from(channels.max(1));
        let requested_lo =
            (round_ties_even_like_python(offset * f64::from(rate)) as i64).saturating_mul(ch);
        let lo = requested_lo.max(self.offset as i64).max(0) as usize;
        let hi = ((round_ties_even_like_python((offset + duration) * f64::from(rate)) as i64)
            .saturating_mul(ch))
        .max(0) as usize;
        let limit = max_duration.map_or(usize::MAX, |duration| {
            let chunk = (round_ties_even_like_python(duration * f64::from(rate)) as usize)
                .max(1)
                .saturating_mul(ch as usize);
            lo.saturating_add(chunk)
        });
        let hi = hi.min(self.offset + self.pcm.len()).min(limit);
        if hi <= lo {
            return None;
        }
        let chunk = self.pcm.range(lo - self.offset..hi - self.offset).copied().collect();
        self.pcm.drain(..hi - self.offset);
        self.offset = hi;
        let started_at =
            started_at + (lo as i64 - requested_lo) as f64 / (f64::from(rate) * ch as f64);
        Some(Placed { started_at, frame: PcmFrame::new(chunk, rate, channels) })
    }
}

/// `RecorderIO` minus its thread, its queue and its file handle.
pub struct Recorder<E: OggOpusEncoder> {
    mixer: Mixer,
    encoder: E,
    input_settled: f64,
}

impl<E: OggOpusEncoder> Recorder<E> {
    /// Every sample index in the file is relative to `t0`.
    pub fn new(encoder: E, sample_rate: u32, t0: f64) -> Self {
        Recorder { mixer: Mixer::new(sample_rate, t0), encoder, input_settled: t0 }
    }

    /// Frames carry no capture timestamp: the frame is taken to have begun a frame-duration ago.
    pub fn on_input_frame(&mut self, now: f64, frame: &PcmFrame) {
        let started_at = now - frame.duration();
        self.mixer.push(INPUT_CHANNEL, started_at, frame);
        self.input_settled = self.mixer.tracks[INPUT_CHANNEL].emitted_until().unwrap_or(started_at);
    }

    pub fn on_output_placed(&mut self, placed: &Placed) {
        self.mixer.push(OUTPUT_CHANNEL, placed.started_at, &placed.frame);
    }

    /// A stalled source is waited on only for [`INPUT_STALL_TIMEOUT`]; nothing past an in-flight
    /// segment's start may be written, or the agent channel would be silent there permanently.
    pub fn settle_until(&self, now: f64, pending_since: Option<f64>) -> f64 {
        let settled = self
            .mixer
            .tracks
            .iter()
            .filter_map(|track| track.pending_resampler_until(now))
            .fold(self.input_settled.max(now - INPUT_STALL_TIMEOUT), f64::min);
        match pending_since {
            Some(pending) => settled.min(pending),
            None => settled,
        }
    }

    /// Drain conversion history before writing the final timeline block; otherwise the last
    /// samples retained by a resampler would be replaced with silence in that block.
    pub fn flush_resamplers(&mut self) -> Result<(), EncodeError> {
        self.mixer.flush_resamplers().map_err(|error| EncodeError(error.to_string()))
    }

    /// Empty output is normal: opus buffers.
    pub fn flush_until(&mut self, until: f64) -> Result<Vec<u8>, EncodeError> {
        for track in &mut self.mixer.tracks {
            if let (Some(received), Some(emitted)) = (track.received_until, track.emitted_until())
                && until >= received
                && until > emitted
            {
                track.flush_resampler().map_err(|error| EncodeError(error.to_string()))?;
            }
        }
        match self.mixer.flush_until(until) {
            Some(pcm) => self.encoder.push(&pcm),
            None => Ok(Vec::new()),
        }
    }

    pub fn dropped_samples(&self) -> [u64; 2] {
        self.mixer.dropped_samples()
    }

    pub fn retained_bytes(&self) -> usize {
        self.mixer.retained_bytes()
    }

    pub fn encoder(&self) -> &E {
        &self.encoder
    }

    pub fn finish(self) -> Result<Vec<u8>, EncodeError> {
        self.encoder.finish()
    }
}

impl PcmFrame {
    pub fn from_audio_frame(frame: &AudioFrame) -> PcmFrame {
        PcmFrame::new(crate::pcm_samples(&frame.data), frame.sample_rate, frame.num_channels)
    }
}

#[cfg(test)]
mod tests;
