//! `voice/recorder_io/`: both sides of a call in one stereo ogg/opus file. Both channels sit on one
//! absolute timeline and silence is whatever nothing was written over (#6921).

use std::io::Write;
use std::path::{Path, PathBuf};
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
    placed: Vec<(i64, Vec<f32>)>,
    /// Mono, rebuilt unflushed when the source rate changes.
    resampler: FrameResampler,
    run_start: Option<f64>,
    run_samples: i64,
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
            dropped_samples: 0,
        }
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
        let frame = AudioFrame::new(crate::pcm_bytes(&mono), frame.sample_rate, 1, mono.len() as u32);
        match self.resampler.push(&frame) {
            Ok(ready) => ready.first().map(|f| PcmFrame::from_audio_frame(f).samples).unwrap_or_default(),
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
    }

    fn place(&mut self, samples: &[i16]) {
        if samples.is_empty() {
            return;
        }
        let run_start = self.run_start.expect("a run is open whenever samples are placed");
        let start =
            round_ties_even_like_python((run_start - self.t0) * f64::from(self.sample_rate)) as i64 + self.run_samples;
        self.placed.push((start, samples.iter().map(|&s| f32::from(s) / 32768.0).collect()));
        self.run_samples += samples.len() as i64;
    }

    /// Placements ending before `start` are dropped and counted; ones past `end` are re-anchored
    /// at `end`, so the tail appears exactly once in the next block.
    pub fn take(&mut self, start: i64, end: i64) -> Vec<f32> {
        let mut block = vec![0.0_f32; (end - start).max(0) as usize];
        let mut keep: Vec<(i64, Vec<f32>)> = Vec::new();
        for (pos, samples) in std::mem::take(&mut self.placed) {
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
                keep.push((end, samples[(end - pos) as usize..].to_vec()));
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

    pub fn push(&mut self, channel: usize, started_at: f64, frame: &PcmFrame) {
        self.tracks[channel].push(started_at, frame);
    }

    pub fn dropped_samples(&self) -> [u64; 2] {
        [self.tracks[0].dropped_samples(), self.tracks[1].dropped_samples()]
    }

    /// The cursor only moves forward, so a flush for a moment already written produces nothing.
    pub fn flush_until(&mut self, until: f64) -> Option<Vec<i16>> {
        let end = round_ties_even_like_python((until - self.t0) * f64::from(self.sample_rate)) as i64;
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
    pcm: Vec<i16>,
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

    pub fn capture_frame(&mut self, now: f64, frame: &PcmFrame) {
        if self.pcm.is_empty() {
            self.segment_since = Some(now);
            self.format = Some((frame.sample_rate, frame.num_channels));
        }
        self.pcm.extend_from_slice(&frame.samples);
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
        self.place(started_at, offset, duration)
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

    fn place(&self, started_at: f64, offset: f64, duration: f64) -> Option<Placed> {
        // A report covering no time places nothing (#5279).
        if self.pcm.is_empty() || duration <= 0.0 {
            return None;
        }
        let (rate, channels) = self.format?;
        let ch = i64::from(channels.max(1));
        let lo = round_ties_even_like_python(offset * f64::from(rate)) as i64 * ch;
        let hi = (round_ties_even_like_python((offset + duration) * f64::from(rate)) as i64 * ch)
            .min(self.pcm.len() as i64);
        if hi <= lo {
            return None;
        }
        let chunk = self.pcm[lo.max(0) as usize..hi as usize].to_vec();
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
        self.input_settled = started_at + frame.duration();
        self.mixer.push(INPUT_CHANNEL, started_at, frame);
    }

    pub fn on_output_placed(&mut self, placed: &Placed) {
        self.mixer.push(OUTPUT_CHANNEL, placed.started_at, &placed.frame);
    }

    /// A stalled source is waited on only for [`INPUT_STALL_TIMEOUT`]; nothing past an in-flight
    /// segment's start may be written, or the agent channel would be silent there permanently.
    pub fn settle_until(&self, now: f64, pending_since: Option<f64>) -> f64 {
        let settled = self.input_settled.max(now - INPUT_STALL_TIMEOUT);
        match pending_since {
            Some(pending) => settled.min(pending),
            None => settled,
        }
    }

    /// Empty output is normal: opus buffers.
    pub fn flush_until(&mut self, until: f64) -> Result<Vec<u8>, EncodeError> {
        match self.mixer.flush_until(until) {
            Some(pcm) => self.encoder.push(&pcm),
            None => Ok(Vec::new()),
        }
    }

    pub fn dropped_samples(&self) -> [u64; 2] {
        self.mixer.dropped_samples()
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
    /// Called on the driver's flush timer (`RecorderIO._write_task`'s 2.5 s).
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
    Input { now: f64, frame: PcmFrame },
    Output(Placed),
}

struct Staging {
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
    staging: Mutex<Staging>,
    core: Mutex<Core<E>>,
    /// Kept outside the recorder because `finish` consumes it and hosts read totals after closing.
    dropped: Mutex<[u64; 2]>,
    sample_rate: u32,
    t0: Mutex<Option<f64>>,
    path: PathBuf,
    file: Mutex<Option<std::fs::File>>,
    written: Mutex<u64>,
}

impl<E: OggOpusEncoder> SessionRecorder<E> {
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
            staging: Mutex::new(Staging { items: Vec::new(), placer: SegmentPlacer::new() }),
            core: Mutex::new(Core::Waiting(encoder)),
            dropped: Mutex::new([0, 0]),
            sample_rate,
            t0: Mutex::new(t0),
            path,
            file: Mutex::new(Some(file)),
            written: Mutex::new(0),
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
        let out = self.drain_and_encode(now, true);
        // Dropped regardless of errors above, so no descriptor outlives `finish`.
        *self.file.lock().expect("recorder file") = None;
        out
    }

    /// `pending_since` is read before the drain: a segment starting mid-encode belongs to the next block.
    fn drain_and_encode(&self, now: f64, final_block: bool) -> Result<(), RecorderError> {
        let (items, pending) = {
            let mut s = self.staging.lock().expect("recorder staging");
            (std::mem::take(&mut s.items), s.placer.pending_since())
        };

        self.anchor(now);
        let t0 = self.started_at().expect("anchored just above");

        let mut core = self.core.lock().expect("recorder");
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
        let until = if final_block { now } else { rec.settle_until(now, pending) };
        let mut out = rec.flush_until(until)?;

        *self.dropped.lock().expect("recorder dropped") = rec.dropped_samples();

        if final_block {
            let Core::Running(rec) = std::mem::replace(&mut *core, Core::Done) else {
                unreachable!("matched immediately above")
            };
            out.extend_from_slice(&rec.finish()?);
        }
        drop(core);
        self.write(&out)
    }
}

impl<E: OggOpusEncoder> SessionRecorder<E> {
    pub fn on_input_pcm(&self, now: f64, frame: PcmFrame) {
        self.anchor(now);
        self.staging.lock().expect("recorder staging").items.push(Item::Input { now, frame });
    }

    pub fn capture_pcm(&self, now: f64, frame: &PcmFrame) {
        self.anchor(now);
        self.staging.lock().expect("recorder staging").placer.capture_frame(now, frame);
    }

    pub fn playback_started(&self, created_at: f64) {
        self.anchor(created_at);
        self.staging.lock().expect("recorder staging").placer.on_playback_started(created_at);
    }

    pub fn playback_progressed(&self, started_at: f64, offset: f64, duration: f64) {
        self.anchor(started_at);
        let mut s = self.staging.lock().expect("recorder staging");
        if let Some(placed) = s.placer.on_playback_progressed(started_at, offset, duration) {
            s.items.push(Item::Output(placed));
        }
    }

    pub fn playback_finished(&self, now: f64, playback_position: f64) {
        self.anchor(now);
        let mut s = self.staging.lock().expect("recorder staging");
        if let Some(placed) = s.placer.on_playback_finished(now, playback_position) {
            s.items.push(Item::Output(placed));
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
mod tests {
    use super::*;

    // `HalvingResampler`, `Factory` and `Counting` are gone with the trait pair (PLAN.md F1).
    // Most of the tests below never touched them: they run at the recording rate, where
    // `Track::resample` short-circuits before the resampler exists. The two that DID are
    // re-derived over real soxr below, and the properties they pin are unchanged -- the tail
    // belongs to the run that ended, and a rate change starts a new resampler.

    fn mono(samples: &[i16], rate: u32) -> PcmFrame {
        PcmFrame::new(samples.to_vec(), rate, 1)
    }

    /// The whole point of the timeline: a gap between two placements is *silence*, not the second
    /// placement moved earlier. Concatenating instead is what #6921 replaced, and the symptom was
    /// a recording where the agent answers before the user finishes.
    #[test]
    fn a_gap_between_runs_is_written_as_silence() {
        let mut t = Track::new(10, 100.0);
        t.push(100.0, &mono(&[32767, 32767], 10));
        t.push(100.5, &mono(&[-32768], 10));

        let block = t.take(0, 8);
        assert_eq!(block.len(), 8);
        assert!(block[0] > 0.9 && block[1] > 0.9);
        assert_eq!(&block[2..5], &[0.0, 0.0, 0.0], "the 0.3 s gap must be silent");
        assert_eq!(block[5], -1.0, "the second run lands at its own timestamp");
    }

    /// Within tolerance the run keeps counting its own samples, so per-frame jitter cannot
    /// accumulate into a slide. Re-anchoring on every frame would drift the channel by whatever
    /// the capture clock is doing.
    #[test]
    fn small_jitter_does_not_re_anchor_the_run() {
        let mut t = Track::new(10, 0.0);
        t.push(0.0, &mono(&[1000], 10));
        // 0.1 s per frame; this one claims to start 0.09 s late, inside RESYNC_TOLERANCE
        t.push(0.19, &mono(&[2000], 10));
        let block = t.take(0, 4);
        assert!(block[1] > 0.0, "the second sample stayed contiguous at index 1");
        assert_eq!(block[2], 0.0);

        let mut t = Track::new(10, 0.0);
        t.push(0.0, &mono(&[1000], 10));
        t.push(0.21, &mono(&[2000], 10)); // 0.11 s late: past tolerance, re-anchor
        let block = t.take(0, 4);
        assert_eq!(block[1], 0.0);
        assert!(block[2] > 0.0, "re-anchored to its own timestamp");

        // exactly at the tolerance still counts as continuous: the comparison is `>`, and a
        // capture clock that drifts by precisely the tolerance every frame is the case where an
        // inclusive bound would re-anchor forever and quantise the channel to frame boundaries
        let mut t = Track::new(1000, 0.0);
        t.push(0.0, &mono(&[1000; 100], 1000)); // 0.1 s, so the run expects 0.1 next
        t.push(0.2, &mono(&[2000; 100], 1000)); // 0.1 s late, exactly RESYNC_TOLERANCE
        let block = t.take(0, 300);
        assert!(block[100] > 0.0, "extended the run at sample 100");
        assert_eq!(block[200], 0.0, "and did not re-anchor to sample 200");
    }

    /// A resampler's tail belongs to the run that just ended, not to the one about to start.
    /// Flushing it after the re-anchor would place the old audio at the new timestamp -- a
    /// millisecond of the previous utterance teleported into the next one.
    ///
    /// Real soxr now, at rates it will actually accept. The double emitted one sample per pair
    /// and held the odd one, so the old assertions counted its samples; what is asserted here is
    /// the *property* -- the second run starts five seconds later, so nothing the first run owed
    /// may appear there.
    #[test]
    fn the_resampler_tail_is_flushed_into_the_run_that_ended() {
        let mut t = Track::new(16_000, 0.0);
        let loud: Vec<i16> = (0..1600).map(|i| if i % 2 == 0 { 20_000 } else { -20_000 }).collect();
        t.push(0.0, &mono(&loud, 48_000));
        // Five seconds later: a new run, and the first one's tail must already be behind it.
        t.push(5.0, &mono(&loud, 48_000));

        // 1600 samples at 48 kHz is 1/30 s, which is ~533 samples at 16 kHz.
        let first = t.take(0, 1600);
        assert!(first.iter().take(533).any(|s| *s != 0.0), "the first run is at its own start");
        assert!(
            first[600..].iter().all(|s| *s == 0.0),
            "the gap after the first run has to be silence, tail included"
        );
    }

    /// Audio that arrives after its place has been written is dropped and *counted*. Mixing it
    /// into the current block instead would put the user's words several seconds late in the
    /// recording, with nothing to say it happened.
    #[test]
    fn late_audio_is_dropped_and_counted() {
        let mut t = Track::new(10, 0.0);
        t.push(0.0, &mono(&[1000, 1000], 10));
        assert_eq!(t.take(5, 10), vec![0.0; 5]);
        assert_eq!(t.dropped_samples(), 2);
    }

    /// A placement straddling the end of a block is kept, re-anchored at the boundary, so its tail
    /// appears in the next block exactly once. Dropping the remainder truncates every utterance at
    /// a 2.5 s boundary; keeping it un-anchored duplicates it.
    #[test]
    fn a_placement_straddling_the_boundary_continues_in_the_next_block() {
        let mut t = Track::new(10, 0.0);
        // starts at sample 2, not 0, and the boundary is not a multiple of it, so an index
        // computed from the wrong end lands somewhere visible
        t.push(0.2, &mono(&[4096, 4096, 4096, 4096, 8192, 16384], 10));
        let first = t.take(0, 6);
        assert_eq!(&first[..2], &[0.0, 0.0], "silence before the run");
        assert!(first[2..].iter().all(|s| (*s - 0.125).abs() < 1e-6), "{first:?}");
        let second = t.take(6, 10);
        assert!((second[0] - 0.25).abs() < 1e-6, "the tail continues, in order: {second:?}");
        assert!((second[1] - 0.5).abs() < 1e-6, "{second:?}");
        assert_eq!(&second[2..], &[0.0, 0.0], "and does not repeat");
        assert_eq!(t.dropped_samples(), 0);
    }

    /// Overlapping placements are summed, not overwritten. A sink reporting the tail of one
    /// segment while the next has started must not erase either.
    #[test]
    fn overlapping_placements_are_mixed() {
        let mut t = Track::new(10, 0.0);
        t.push(0.0, &mono(&[8192, 8192], 10));
        t.push(5.0, &mono(&[8192], 10));
        t.push(0.0, &mono(&[8192, 8192], 10));
        let block = t.take(0, 2);
        assert!((block[0] - 0.5).abs() < 1e-6, "two quarter-scale runs summed: {block:?}");
    }

    /// Multi-channel input is averaged down, not left-channel-only: recording only the left
    /// channel of a stereo microphone loses half the room.
    #[test]
    fn stereo_input_is_downmixed_by_averaging() {
        let mut t = Track::new(10, 0.0);
        t.push(0.0, &PcmFrame::new(vec![16384, 0, -16384, 0], 10, 2));
        let block = t.take(0, 2);
        assert!((block[0] - 0.25).abs() < 1e-6, "{block:?}");
        assert!((block[1] + 0.25).abs() < 1e-6, "{block:?}");
    }

    /// The write cursor never rewinds. A `settled` estimate that goes backwards -- and it does,
    /// the moment a segment starts mid-interval -- must produce nothing rather than a second copy
    /// of audio already in the file.
    #[test]
    fn the_cursor_never_rewinds() {
        // a non-zero t0, because every sample index is relative to it
        let mut m = Mixer::new(10, 100.0);
        m.push(INPUT_CHANNEL, 100.0, &mono(&[32767; 10], 10));
        let first = m.flush_until(101.0).expect("first block");
        assert_eq!(first.len(), 20, "10 samples, stereo interleaved");
        assert!(m.flush_until(100.5).is_none());
        assert!(m.flush_until(101.0).is_none());
    }

    /// The user is the left channel and the agent the right. Swapping them is invisible to every
    /// test that looks at one channel, and obvious to anyone who plays the file.
    #[test]
    fn the_user_is_left_and_the_agent_is_right() {
        let mut m = Mixer::new(10, 100.0);
        m.push(INPUT_CHANNEL, 100.0, &mono(&[16384], 10));
        m.push(OUTPUT_CHANNEL, 100.0, &mono(&[-16384], 10));
        assert_eq!(m.flush_until(100.1), Some(vec![16384, -16384]));
    }

    /// A stalled microphone must not hold the writer, and a segment in flight must hold it back.
    /// Losing the first clamp freezes the recording; losing the second writes silence over audio
    /// the sink has not yet reported.
    #[test]
    fn the_writer_waits_for_a_stalled_input_but_not_forever() {
        let mut r = Recorder::new(PcmRecordingEncoder::default(), 10, 100.0);
        r.on_input_frame(101.0, &mono(&[0; 4], 10)); // 0.4 s, arriving at 101.0
        assert_eq!(r.settle_until(101.5, None), 101.0, "settled follows the input");
        assert_eq!(r.settle_until(103.0, None), 102.0, "and then the stall timeout");
        assert_eq!(r.settle_until(103.0, Some(101.4)), 101.4, "a segment in flight holds it back");
        assert_eq!(r.settle_until(101.2, None), 101.0, "never past what has arrived");
    }

    /// A frame carries no capture timestamp, so it is taken to have *begun* a frame-duration
    /// before it arrived. Anchoring it at arrival instead puts the whole user channel one frame
    /// late, which is exactly the misalignment the timeline exists to avoid.
    #[test]
    fn an_input_frame_is_backdated_by_its_own_duration() {
        let mut r = Recorder::new(PcmRecordingEncoder::default(), 10, 100.0);
        // 0.5 s of *stereo* audio arriving at 100.7: it began at 100.2, i.e. sample 2. Stereo
        // because the duration is per channel -- reading it off the interleaved length instead
        // backdates the frame by twice as long.
        r.on_input_frame(100.7, &PcmFrame::new(vec![32767; 10], 10, 2));
        r.flush_until(100.9).expect("encoded");
        let pcm = &r.encoder().pcm;
        assert_eq!(pcm.len(), 18, "9 stereo samples");
        assert_eq!(&pcm[..4], &[0, 0, 0, 0], "silence before the frame began");
        assert!(pcm[4] > 32000, "the audio starts at sample 2: {pcm:?}");
    }

    /// Both channels have to reach the encoder. Each of the three hops -- input frame, output
    /// report, flush -- can be dropped on its own and leave a file that plays, with one side of
    /// the conversation missing.
    #[test]
    fn both_channels_reach_the_encoder() {
        let mut r = Recorder::new(PcmRecordingEncoder::default(), 10, 100.0);
        r.on_input_frame(100.1, &mono(&[16384], 10));
        r.on_output_placed(&Placed {
            started_at: 100.0,
            frame: PcmFrame::new(vec![-16384], 10, 1),
        });
        assert!(r.flush_until(100.1).expect("encoded").is_empty(), "the double buffers");
        assert_eq!(r.encoder().pcm, vec![16384, -16384]);
        assert_eq!(r.dropped_samples(), [0, 0]);
        assert_eq!(r.finish().expect("closed"), PCM_ENCODER_TRAILER, "the trailer must be written");
    }

    /// The dropped-sample count has to survive the two hops up to the recorder, because the
    /// warning the driver logs at close is the only signal that a recording is missing audio. A
    /// rollup stuck at zero turns a broken recording into a silent one.
    #[test]
    fn dropped_audio_is_reported_up_to_the_recorder() {
        let mut r = Recorder::new(PcmRecordingEncoder::default(), 10, 100.0);
        r.on_input_frame(101.0, &mono(&[16384; 10], 10));
        r.flush_until(102.0).expect("encoded");
        // an output report for a moment already written: three samples with nowhere to go
        r.on_output_placed(&Placed {
            started_at: 100.2,
            frame: PcmFrame::new(vec![16384; 3], 10, 1),
        });
        r.flush_until(103.0).expect("encoded");
        assert_eq!(r.dropped_samples(), [0, 3]);
    }

    /// A source that changes sample rate mid-call gets a new resampler. Keeping the old one feeds
    /// it audio at a rate it was not built for, and the channel drifts for the rest of the call.
    ///
    /// The old version counted calls to a `Counting` factory. There is no factory (PLAN.md F1),
    /// so the observation is the state the count stood in for: the format the resampler was built
    /// for, which is what `resample` compares to decide whether to rebuild. That is strictly closer to the thing that goes wrong -- a factory called the
    /// right number of times with the wrong rates would have passed the old test.
    #[test]
    fn a_source_rate_change_rebuilds_the_resampler() {
        let mut t = Track::new(16_000, 0.0);
        let frame: Vec<i16> = (0..480).map(|i| (i % 100) as i16 * 100).collect();

        let built_for = |t: &Track| t.resampler.input_format().map(|f| f.sample_rate);
        t.push(0.0, &mono(&frame, 48_000));
        assert_eq!(built_for(&t), Some(48_000));
        t.push(0.01, &mono(&frame, 48_000));
        assert_eq!(built_for(&t), Some(48_000), "same rate, same resampler");

        t.push(0.02, &mono(&frame, 24_000));
        assert_eq!(
            built_for(&t),
            Some(24_000),
            "built for the NEW rate -- a rebuild with the old rates is the bug this catches"
        );
    }

    /// A sink that reports its own progress must not *also* get the endpoint fallback at
    /// `playback_finished`, or every segment is written into the recording twice.
    #[test]
    fn a_reporting_sink_does_not_also_get_the_endpoint_fallback() {
        let mut p = SegmentPlacer::new();
        p.capture_frame(10.0, &mono(&[1, 2, 3, 4], 4));
        p.on_playback_started(10.0);
        let placed = p.on_playback_progressed(10.0, 0.0, 1.0).expect("reported run");
        assert_eq!(placed.frame.samples, vec![1, 2, 3, 4]);
        assert!(p.on_playback_finished(12.0, 1.0).is_none());
    }

    /// A sink that reports nothing -- a remote avatar worker -- has its segment placed from its
    /// endpoints, anchored at `playback_started` when there was one.
    #[test]
    fn a_silent_sink_falls_back_to_its_segment_endpoints() {
        let mut p = SegmentPlacer::new();
        p.capture_frame(10.0, &mono(&[1, 2, 3, 4], 4));
        p.on_playback_started(9.5);
        let placed = p.on_playback_finished(12.0, 1.0).expect("endpoint fallback");
        assert_eq!(placed.started_at, 9.5);
        assert_eq!(placed.frame.samples, vec![1, 2, 3, 4]);

        // and with no `playback_started`, the finish time less the position
        let mut p = SegmentPlacer::new();
        p.capture_frame(10.0, &mono(&[1, 2], 2));
        let placed = p.on_playback_finished(12.0, 1.0).expect("endpoint fallback");
        assert_eq!(placed.started_at, 11.0);
    }

    /// The slice bounds are in *interleaved* samples, so they scale by the channel count. Dropping
    /// the factor cuts a stereo segment to a quarter and pans what is left.
    #[test]
    fn the_slice_bounds_scale_by_the_channel_count() {
        let mut p = SegmentPlacer::new();
        p.capture_frame(10.0, &PcmFrame::new(vec![1, -1, 2, -2, 3, -3, 4, -4], 4, 2));
        let placed = p.on_playback_progressed(10.0, 0.5, 0.5).expect("second half");
        assert_eq!(placed.frame.samples, vec![3, -3, 4, -4]);
        assert_eq!(placed.frame.num_channels, 2);
    }

    /// The offset is honoured, because a jump in it is audio the sink *discarded* on a pause
    /// (#6728). Assuming the reports are contiguous writes discarded audio into the file.
    #[test]
    fn a_jump_in_the_offset_skips_the_audio_the_sink_discarded() {
        let mut p = SegmentPlacer::new();
        p.capture_frame(10.0, &mono(&[1, 2, 3, 4, 5, 6, 7, 8], 8));
        let placed = p.on_playback_progressed(10.0, 0.5, 0.25).expect("second half");
        assert_eq!(placed.frame.samples, vec![5, 6]);
    }

    /// Three ways a report describes no audio, all of which must place nothing rather than an
    /// empty or reversed slice: a zero duration, a report before any frame was captured (#5279),
    /// and a report starting past the end of what was captured.
    #[test]
    fn a_report_covering_no_audio_places_nothing() {
        let mut p = SegmentPlacer::new();
        assert!(p.on_playback_progressed(10.0, 0.0, 1.0).is_none(), "nothing captured yet");
        p.capture_frame(10.0, &mono(&[1, 2], 2));
        assert!(p.on_playback_progressed(10.0, 0.0, 0.0).is_none(), "zero duration");
        assert!(p.on_playback_progressed(10.0, 5.0, 1.0).is_none(), "past the end");
    }

    /// `pending_since` is set by the first frame of a segment and cleared when it finishes -- it
    /// is what holds the writer back, so a stuck `Some` freezes the recording and a premature
    /// `None` writes silence over the agent.
    #[test]
    fn pending_since_spans_exactly_the_segment() {
        let mut p = SegmentPlacer::new();
        assert_eq!(p.pending_since(), None);
        p.capture_frame(10.0, &mono(&[1, 2], 2));
        p.capture_frame(10.5, &mono(&[3, 4], 2));
        assert_eq!(p.pending_since(), Some(10.0), "the first frame of the segment, not the last");
        p.on_playback_finished(12.0, 1.0);
        assert_eq!(p.pending_since(), None);
    }

    /// Catches a port that reached for `f64::round`: `round(0.5)` is `0` in Python and `1` in Rust,
    /// and the recorder turns these straight into sample offsets.
    #[test]
    fn ties_go_to_even_like_python() {
        use super::round_ties_even_like_python as py_round;
        assert_eq!(py_round(0.5), 0.0);
        assert_eq!(py_round(1.5), 2.0);
        assert_eq!(py_round(2.5), 2.0);
        assert_eq!(py_round(-0.5), -0.0);
    }
}

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
            Err(pyo3::exceptions::PyRuntimeError::new_err("recording needs the extension's `ffmpeg` build"))
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
            RecorderError::Encode(e) => PyRuntimeError::new_err(format!("recorder encode failed: {e}")),
        }
    }

    /// `frozen` with an `Arc` interior: the room's audio path holds it while `flush` runs on another
    /// thread.
    #[pyclass(frozen, name = "SessionRecorder", module = "livekit.agents._core")]
    pub struct PySessionRecorder {
        inner: Arc<SessionRecorder<crate::OggOpusEncoder>>,
    }

    impl PySessionRecorder {
        /// The recorder as the room's audio path sees it.
        pub fn tap(&self) -> Arc<dyn AudioRecorder> {
            self.inner.clone()
        }
    }

    fn pcm_from_py(frame: &Bound<'_, PyAny>) -> PyResult<PcmFrame> {
        Ok(PcmFrame::from_audio_frame(&lk_agents_abi::python::extract_generated::<lk_agents_abi::AudioFrame>(frame)?))
    }

    #[pymethods]
    impl PySessionRecorder {
        /// `t0` is the timeline origin. `None` (the room path) anchors at the first frame's core
        /// timestamp; a host with its own audio passes its clock's reading. `bitrate` is stereo total.
        #[new]
        #[pyo3(signature = (path, *, sample_rate = 48_000, t0 = None, bitrate = 64_000))]
        fn new(path: PathBuf, sample_rate: u32, t0: Option<f64>, bitrate: u32) -> PyResult<Self> {
            check_pid()?;
            let encoder = crate::OggOpusEncoder::new(sample_rate, 2, bitrate)
                .map_err(|e| PyRuntimeError::new_err(format!("could not open the opus encoder: {e}")))?;
            let inner = SessionRecorder::new(encoder, sample_rate, t0, path)
                .map_err(|e| PyOSError::new_err(format!("could not open the recording: {e}")))?;
            Ok(PySessionRecorder { inner: Arc::new(inner) })
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
        fn on_playback_progressed(&self, started_at: f64, offset: f64, duration: f64) -> PyResult<()> {
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

        /// Write everything staged plus the trailer, and close the file.
        fn finish(&self, py: Python<'_>, now: f64) -> PyResult<()> {
            check_pid()?;
            py.detach(|| self.inner.finish(now)).map_err(recorder_err)
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
