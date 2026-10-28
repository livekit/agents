//! `voice/recorder_io/`: both sides of a call in one stereo ogg/opus file. Both channels sit on one
//! absolute timeline and silence is whatever nothing was written over (#6921).

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::Mutex;

use lk_agents_abi::AudioFrame;

use crate::{FrameResampler, OnChange, Quality, ResampleError};

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
        let ch = frame.num_channels.max(1) as usize;
        let mono: Vec<i16> = if ch > 1 {
            // `data.mean(axis=1).astype(np.int16)`: narrowed by truncation, not rounding.
            frame
                .samples
                .chunks_exact(ch)
                .map(|c| (c.iter().map(|&s| f64::from(s)).sum::<f64>() / ch as f64) as i16)
                .collect()
        } else {
            frame.samples.clone()
        };

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
        self.placed.push((start, samples.iter().map(|&s| f32::from(s) / 32768.0).collect()));
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
        self.pcm.reserve_exact(frame.samples.len());
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
        self.reported = true;
        let placed = self.place(started_at, offset, duration)?;
        self.segment_since = if self.pcm.is_empty() {
            None
        } else {
            Some(placed.started_at + placed.frame.duration())
        };
        Some(placed)
    }

    /// Fallback for a sink that reported nothing: its endpoints place the segment.
    pub fn on_playback_finished(&mut self, now: f64, playback_position: f64) -> Option<Placed> {
        let placed = if !self.reported && !self.pcm.is_empty() {
            let started_at = self.started_at.unwrap_or(now - playback_position);
            self.place(started_at, 0.0, playback_position)
        } else {
            None
        };
        *self = SegmentPlacer::new();
        placed
    }

    fn place(&mut self, started_at: f64, offset: f64, duration: f64) -> Option<Placed> {
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
        let hi = hi.min(self.offset + self.pcm.len());
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

/// Lets the room audio path feed the recorder without being generic over the encoder.
pub trait AudioRecorder: Send + Sync {
    fn on_input_frame(&self, now: f64, frame: &AudioFrame);
    /// Held until the sink says where it went.
    fn capture_frame(&self, now: f64, frame: &AudioFrame);
    fn on_playback_started(&self, created_at: f64);
    fn on_playback_progressed(&self, started_at: f64, offset: f64, duration: f64);
    fn on_playback_finished(&self, now: f64, playback_position: f64);
    /// Called on the driver's flush timer (`RecorderIO._write_task`'s 2.5 s), on the driver's
    /// step: a recorder that encodes or writes a file does it on a thread of its own.
    fn flush(&self, now: f64) -> Result<(), RecorderError>;
}

impl PcmFrame {
    pub fn from_audio_frame(frame: &AudioFrame) -> PcmFrame {
        let samples =
            frame.data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
        PcmFrame::new(samples, frame.sample_rate, frame.num_channels)
    }
}

enum Item {
    /// The clock reading at which the frame *arrived*.
    Input {
        now: f64,
        frame: PcmFrame,
    },
    Output(Placed),
}

struct Staging {
    accepting: bool,
    items: Vec<Item>,
    placer: SegmentPlacer,
}

/// The encoder exists before the recorder has an origin, and is consumed by `finish`'s trailer.
enum Core<E: OggOpusEncoder> {
    Waiting(E),
    Running(Recorder<E>),
    Done,
}

/// Encode failures lose the recording; write failures (full disk) do not, and may be retried.
#[derive(Debug)]
pub enum RecorderError {
    Encode(EncodeError),
    Io(std::io::Error),
}

impl std::fmt::Display for RecorderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecorderError::Encode(e) => write!(f, "{e}"),
            RecorderError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RecorderError {}

impl From<EncodeError> for RecorderError {
    fn from(e: EncodeError) -> Self {
        RecorderError::Encode(e)
    }
}

impl From<std::io::Error> for RecorderError {
    fn from(e: std::io::Error) -> Self {
        RecorderError::Io(e)
    }
}

/// Two locks: the audio path only touches the O(1) staging half, so a tens-of-ms encode never
/// stalls it. Blocks are written and flushed to the OS as encoded, so a killed worker leaves a file.
pub struct SessionRecorder<E: OggOpusEncoder> {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    staging: Mutex<Staging>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    core: Mutex<Core<E>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// Kept outside the recorder because `finish` consumes it and hosts read totals after closing.
    dropped: Mutex<[u64; 2]>,
    sample_rate: u32,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    t0: Mutex<Option<f64>>,
    path: PathBuf,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    file: Mutex<Option<std::fs::File>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    written: Mutex<u64>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// The `now` of the last drain: where a host that has no session clock finishes.
    last: Mutex<Option<f64>>,
}

impl<E: OggOpusEncoder> SessionRecorder<E> {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// Creates `path` and its parents before returning.
    pub fn new(
        encoder: E,
        sample_rate: u32,
        t0: Option<f64>,
        path: impl AsRef<Path>,
    ) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(&path)?;
        Ok(SessionRecorder {
            staging: Mutex::new(Staging {
                accepting: true,
                items: Vec::new(),
                placer: SegmentPlacer::new(),
            }),
            core: Mutex::new(Core::Waiting(encoder)),
            dropped: Mutex::new([0, 0]),
            sample_rate,
            t0: Mutex::new(t0),
            path,
            file: Mutex::new(Some(file)),
            written: Mutex::new(0),
            last: Mutex::new(None),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Zero at first is normal: opus and the muxer buffer.
    pub fn bytes_written(&self) -> u64 {
        *self.written.lock().expect("recorder written")
    }

    /// A no-op after `finish`: the encoder is consumed, so the block is empty.
    fn write(&self, block: &[u8]) -> Result<(), RecorderError> {
        if block.is_empty() {
            return Ok(());
        }
        let mut slot = self.file.lock().expect("recorder file");
        let Some(file) = slot.as_mut() else { return Ok(()) };
        file.write_all(block)?;
        file.flush()?;
        *self.written.lock().expect("recorder written") += block.len() as u64;
        Ok(())
    }

    /// The `now` the last flush encoded up to, on the clock the session stamps with.
    pub fn last_flushed(&self) -> Option<f64> {
        *self.last.lock().expect("recorder last")
    }

    pub fn started_at(&self) -> Option<f64> {
        *self.t0.lock().expect("recorder t0")
    }

    /// Fix the origin at `now` if nothing has yet.
    fn anchor(&self, now: f64) {
        self.t0.lock().expect("recorder t0").get_or_insert(now);
    }

    pub fn pending_since(&self) -> Option<f64> {
        self.staging.lock().expect("recorder staging").placer.pending_since()
    }

    pub fn dropped_samples(&self) -> [u64; 2] {
        *self.dropped.lock().expect("recorder dropped")
    }

    pub fn flush(&self, now: f64) -> Result<(), RecorderError> {
        self.drain_and_encode(now, false)
    }

    /// Writes everything staged, settled or not, plus the trailer in the same write.
    pub fn finish(&self, now: f64) -> Result<(), RecorderError> {
        {
            let mut staging = self.staging.lock().expect("recorder staging");
            staging.accepting = false;
        }
        let out = self.drain_and_encode(now, true);
        self.staging.lock().expect("recorder staging").placer = SegmentPlacer::new();
        // Dropped regardless of errors above, so no descriptor outlives `finish`.
        *self.file.lock().expect("recorder file") = None;
        out
    }

    /// `pending_since` is read before the drain: a segment starting mid-encode belongs to the next block.
    /// The encoder's lock is taken first, so two drains (a background flush, the finish) encode
    /// their items in the order they were staged.
    fn drain_and_encode(&self, now: f64, final_block: bool) -> Result<(), RecorderError> {
        let mut core = self.core.lock().expect("recorder");
        let (items, pending) = {
            let mut s = self.staging.lock().expect("recorder staging");
            (std::mem::take(&mut s.items), s.placer.pending_since())
        };

        self.anchor(now);
        let t0 = self.started_at().expect("anchored just above");
        *self.last.lock().expect("recorder last") = Some(now);

        if let Core::Waiting(_) = &*core {
            let Core::Waiting(encoder) = std::mem::replace(&mut *core, Core::Done) else {
                unreachable!("checked immediately above")
            };
            *core = Core::Running(Recorder::new(encoder, self.sample_rate, t0));
        }
        let Core::Running(rec) = &mut *core else {
            return Ok(());
        };

        for item in items {
            match item {
                Item::Input { now, frame } => rec.on_input_frame(now, &frame),
                Item::Output(placed) => rec.on_output_placed(&placed),
            }
        }

        // The final block flushes to *now*; the settle clamps would truncate the tail.
        let until = if final_block {
            rec.flush_resamplers()?;
            now
        } else {
            rec.settle_until(now, pending)
        };
        let mut out = rec.flush_until(until)?;

        *self.dropped.lock().expect("recorder dropped") = rec.dropped_samples();

        if final_block {
            let Core::Running(rec) = std::mem::replace(&mut *core, Core::Done) else {
                unreachable!("matched immediately above")
            };
            out.extend_from_slice(&rec.finish()?);
        }
        // Keep encode and write in one serial operation: a later flush or trailer must never
        // overtake bytes already encoded for this timeline.
        self.write(&out)
    }
}

impl<E: OggOpusEncoder> SessionRecorder<E> {
    pub fn on_input_pcm(&self, now: f64, frame: PcmFrame) {
        let mut staging = self.staging.lock().expect("recorder staging");
        if staging.accepting {
            self.anchor(now);
            staging.items.push(Item::Input { now, frame });
        }
    }

    pub fn capture_pcm(&self, now: f64, frame: &PcmFrame) {
        let mut staging = self.staging.lock().expect("recorder staging");
        if staging.accepting {
            self.anchor(now);
            staging.placer.capture_frame(now, frame);
        }
    }

    pub fn playback_started(&self, created_at: f64) {
        let mut staging = self.staging.lock().expect("recorder staging");
        if staging.accepting {
            self.anchor(created_at);
            staging.placer.on_playback_started(created_at);
        }
    }

    pub fn playback_progressed(&self, started_at: f64, offset: f64, duration: f64) {
        let mut staging = self.staging.lock().expect("recorder staging");
        if staging.accepting {
            self.anchor(started_at);
            if let Some(placed) =
                staging.placer.on_playback_progressed(started_at, offset, duration)
            {
                staging.items.push(Item::Output(placed));
            }
        }
    }

    pub fn playback_finished(&self, now: f64, playback_position: f64) {
        let mut staging = self.staging.lock().expect("recorder staging");
        if staging.accepting {
            self.anchor(now);
            if let Some(placed) = staging.placer.on_playback_finished(now, playback_position) {
                staging.items.push(Item::Output(placed));
            }
        }
    }
}

impl<E: OggOpusEncoder + Send> AudioRecorder for SessionRecorder<E> {
    fn on_input_frame(&self, now: f64, frame: &AudioFrame) {
        self.on_input_pcm(now, PcmFrame::from_audio_frame(frame));
    }

    fn capture_frame(&self, now: f64, frame: &AudioFrame) {
        self.capture_pcm(now, &PcmFrame::from_audio_frame(frame));
    }

    fn on_playback_started(&self, created_at: f64) {
        self.playback_started(created_at);
    }

    fn on_playback_progressed(&self, started_at: f64, offset: f64, duration: f64) {
        self.playback_progressed(started_at, offset, duration);
    }

    fn on_playback_finished(&self, now: f64, playback_position: f64) {
        self.playback_finished(now, playback_position);
    }

    fn flush(&self, now: f64) -> Result<(), RecorderError> {
        SessionRecorder::flush(self, now)
    }
}

#[cfg(test)]
mod tests;

// `_core.SessionRecorder`. On the room path it is fed by `AgentSession::attach_recorder`; the
// frame hooks are for hosts that own their audio.

/// Only the native `SessionRecorder` crosses as the recording tap: there is no host protocol.
#[cfg(feature = "python")]
impl lk_agents_abi::python::HostProxy for dyn AudioRecorder {
    fn from_host(
        obj: &pyo3::Bound<'_, pyo3::PyAny>,
        _clock: Option<std::sync::Arc<dyn lk_agents_abi::Clock>>,
    ) -> pyo3::PyResult<std::sync::Arc<Self>> {
        #[cfg(feature = "ffmpeg")]
        {
            use pyo3::types::PyAnyMethods as _;
            Ok(obj.extract::<pyo3::PyRef<'_, python::PySessionRecorder>>()?.tap())
        }
        #[cfg(not(feature = "ffmpeg"))]
        {
            let _ = obj;
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "recording needs the extension's `ffmpeg` build",
            ))
        }
    }

    fn view(
        _handle: lk_agents_abi::python::HostHandle,
        _clock: Option<std::sync::Arc<dyn lk_agents_abi::Clock>>,
    ) -> pyo3::PyResult<Box<Self>> {
        Err(pyo3::exceptions::PyTypeError::new_err("a recorder is native only"))
    }
}

#[cfg(all(feature = "python", feature = "ffmpeg"))]
pub use python::PySessionRecorder;

#[cfg(all(feature = "python", feature = "ffmpeg"))]
mod python {
    use std::path::PathBuf;
    use std::sync::Arc;

    use lk_agents_abi::python::check_pid;
    use pyo3::exceptions::{PyOSError, PyRuntimeError};
    use pyo3::prelude::*;

    use super::{AudioRecorder, PcmFrame, RecorderError, SessionRecorder};

    /// A write failure is `OSError`, an encoder failure `RuntimeError`.
    fn recorder_err(e: RecorderError) -> PyErr {
        match e {
            RecorderError::Io(e) => PyOSError::new_err(format!("recorder could not write: {e}")),
            RecorderError::Encode(e) => {
                PyRuntimeError::new_err(format!("recorder encode failed: {e}"))
            }
        }
    }

    /// `frozen` with an `Arc` interior: the room's audio path holds it while `flush` runs on another
    /// thread.
    #[pyclass(frozen, name = "SessionRecorder", module = "livekit.agents._core")]
    pub struct PySessionRecorder {
        inner: Arc<SessionRecorder<crate::OggOpusEncoder>>,
        /// Whether the tap's flush is writing; `finish` waits for it.
        busy: Arc<Busy>,
    }

    #[derive(Default)]
    struct Busy {
        #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
        on: std::sync::Mutex<bool>,
        idle: std::sync::Condvar,
    }

    impl Busy {
        fn set(&self, on: bool) {
            *self.on.lock().unwrap_or_else(|p| p.into_inner()) = on;
            self.idle.notify_all();
        }

        fn wait_idle(&self) {
            let on = self.on.lock().unwrap_or_else(|p| p.into_inner());
            drop(self.idle.wait_while(on, |on| *on).unwrap_or_else(|p| p.into_inner()));
        }
    }

    impl PySessionRecorder {
        /// The recorder as the room's audio path sees it.
        pub fn tap(&self) -> Arc<dyn AudioRecorder> {
            Arc::new(OffStep { recorder: self.inner.clone(), busy: self.busy.clone() })
        }
    }

    /// The session's tap: the driver's flush timer starts an encode and its file write on a
    /// thread of its own, never on the driver's step; one at a time, a failure logged.
    struct OffStep {
        recorder: Arc<SessionRecorder<crate::OggOpusEncoder>>,
        busy: Arc<Busy>,
    }

    impl AudioRecorder for OffStep {
        fn on_input_frame(&self, now: f64, frame: &lk_agents_abi::AudioFrame) {
            AudioRecorder::on_input_frame(&*self.recorder, now, frame);
        }

        fn capture_frame(&self, now: f64, frame: &lk_agents_abi::AudioFrame) {
            AudioRecorder::capture_frame(&*self.recorder, now, frame);
        }

        fn on_playback_started(&self, created_at: f64) {
            self.recorder.playback_started(created_at);
        }

        fn on_playback_progressed(&self, started_at: f64, offset: f64, duration: f64) {
            self.recorder.playback_progressed(started_at, offset, duration);
        }

        fn on_playback_finished(&self, now: f64, playback_position: f64) {
            self.recorder.playback_finished(now, playback_position);
        }

        #[allow(clippy::disallowed_methods, reason = "S0: runtime migration")]
        fn flush(&self, now: f64) -> Result<(), RecorderError> {
            {
                let mut on = self.busy.on.lock().unwrap_or_else(|p| p.into_inner());
                if *on {
                    // The last one is still writing; what settled since goes with the next.
                    return Ok(());
                }
                *on = true;
            }
            let (recorder, busy) = (self.recorder.clone(), self.busy.clone());
            std::thread::spawn(move || {
                if let Err(e) = recorder.flush(now) {
                    lk_agents_abi::python::log::warn!(error = e.to_string().as_str(); "failed to write the session recording");
                }
                busy.set(false);
            });
            Ok(())
        }
    }

    fn pcm_from_py(frame: &Bound<'_, PyAny>) -> PyResult<PcmFrame> {
        Ok(PcmFrame::from_audio_frame(&lk_agents_abi::python::extract_generated::<
            lk_agents_abi::AudioFrame,
        >(frame)?))
    }

    #[pymethods]
    impl PySessionRecorder {
        /// `t0` is the timeline origin. `None` (the room path) anchors at the first frame's core
        /// timestamp; a host with its own audio passes its clock's reading. `bitrate` is stereo total.
        #[new]
        #[pyo3(signature = (path, *, sample_rate = 48_000, t0 = None, bitrate = 64_000))]
        fn new(path: PathBuf, sample_rate: u32, t0: Option<f64>, bitrate: u32) -> PyResult<Self> {
            check_pid()?;
            let encoder = crate::OggOpusEncoder::new(sample_rate, 2, bitrate).map_err(|e| {
                PyRuntimeError::new_err(format!("could not open the opus encoder: {e}"))
            })?;
            let inner = SessionRecorder::new(encoder, sample_rate, t0, path)
                .map_err(|e| PyOSError::new_err(format!("could not open the recording: {e}")))?;
            Ok(PySessionRecorder { inner: Arc::new(inner), busy: Arc::default() })
        }

        /// Bytes on disk so far; zero at first is normal (opus and the muxer buffer).
        #[getter]
        fn bytes_written(&self) -> PyResult<u64> {
            check_pid()?;
            Ok(self.inner.bytes_written())
        }

        /// The timeline origin, or `None` before the first frame.
        #[getter]
        fn started_at(&self) -> PyResult<Option<f64>> {
            check_pid()?;
            Ok(self.inner.started_at())
        }

        /// A frame arrived on the user's input. Frames carry no capture time, so it is taken to have
        /// begun one frame-duration before `now`.
        fn on_input_frame(&self, now: f64, frame: &Bound<'_, PyAny>) -> PyResult<()> {
            check_pid()?;
            let frame = pcm_from_py(frame)?;
            self.inner.on_input_pcm(now, frame);
            Ok(())
        }

        /// A frame was forwarded down the output chain; held until the sink reports where it went.
        fn capture_frame(&self, now: f64, frame: &Bound<'_, PyAny>) -> PyResult<()> {
            check_pid()?;
            let frame = pcm_from_py(frame)?;
            self.inner.capture_pcm(now, &frame);
            Ok(())
        }

        /// `on_playback_started`. The first `created_at` wins, so a resumed segment keeps its anchor.
        fn on_playback_started(&self, created_at: f64) -> PyResult<()> {
            check_pid()?;
            self.inner.playback_started(created_at);
            Ok(())
        }

        /// `on_playback_progressed`: the sink says where a run of its audio actually went.
        fn on_playback_progressed(
            &self,
            started_at: f64,
            offset: f64,
            duration: f64,
        ) -> PyResult<()> {
            check_pid()?;
            self.inner.playback_progressed(started_at, offset, duration);
            Ok(())
        }

        /// `on_playback_finished`: end of segment, and the fallback placement for a sink that reported
        /// no progress.
        fn on_playback_finished(&self, now: f64, playback_position: f64) -> PyResult<()> {
            check_pid()?;
            self.inner.playback_finished(now, playback_position);
            Ok(())
        }

        /// Encode everything settled as of `now` and append it to the file, with the GIL released.
        fn flush(&self, py: Python<'_>, now: f64) -> PyResult<()> {
            check_pid()?;
            py.detach(|| self.inner.flush(now)).map_err(recorder_err)
        }

        /// Write everything staged plus the trailer, and close the file. `now` defaults to where
        /// the session's last flush ended (its close), on the session's own clock; a flush still
        /// writing is waited for.
        #[pyo3(signature = (now = None))]
        fn finish(&self, py: Python<'_>, now: Option<f64>) -> PyResult<()> {
            check_pid()?;
            py.detach(|| {
                self.busy.wait_idle();
                let now = now.or_else(|| self.inner.last_flushed()).or(self.inner.started_at());
                self.inner.finish(now.unwrap_or(0.0))
            })
            .map_err(recorder_err)
        }

        /// `(input, output)` samples that arrived after their place had been written.
        #[getter]
        fn dropped_samples(&self) -> PyResult<(u64, u64)> {
            check_pid()?;
            let [a, b] = self.inner.dropped_samples();
            Ok((a, b))
        }
    }

    lk_agents_abi::register_host!(class PySessionRecorder);
}
