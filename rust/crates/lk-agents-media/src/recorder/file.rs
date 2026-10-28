//! One recording file: a call's commands placed on its timeline, encoded and written in bounded
//! blocks to the supplied writer. The caller owns the queue, admission and the writer's lifetime.

use std::io::Write;

use lk_agents_abi::AudioFrame;
use lk_agents_abi::recorder::{RecordingFailure, RecordingSummary};

use super::{OggOpusEncoder, PcmFrame, Recorder, SegmentPlacer};

/// One ordered recording command, as the live session's tap reports it.
pub enum RecordingCommand {
    Input(f64, AudioFrame),
    Capture(f64, AudioFrame),
    Started(f64),
    Progress(f64, f64, f64),
    Finished(f64, f64),
    Flush(f64),
    Finish(Option<f64>),
}

impl RecordingCommand {
    /// Refuses non-finite times, negative playback positions and malformed PCM before the command
    /// is queued.
    pub fn validate(&self) -> Result<(), RecordingFailure> {
        let times = |values: &[f64]| {
            if values.iter().all(|value| value.is_finite()) {
                Ok(())
            } else {
                Err(RecordingFailure::Invalid("non-finite timestamp"))
            }
        };
        match self {
            RecordingCommand::Input(now, frame) | RecordingCommand::Capture(now, frame) => {
                times(&[*now])?;
                if frame.sample_rate == 0
                    || frame.num_channels == 0
                    || frame.data.len()
                        != frame.samples_per_channel as usize * frame.num_channels as usize * 2
                {
                    return Err(RecordingFailure::Invalid("PCM frame shape"));
                }
                Ok(())
            }
            RecordingCommand::Started(at) | RecordingCommand::Flush(at) => times(&[*at]),
            RecordingCommand::Progress(started_at, offset, duration) => {
                times(&[*started_at, *offset, *duration])?;
                if *offset < 0.0 || *duration < 0.0 {
                    return Err(RecordingFailure::Invalid("negative playback position"));
                }
                Ok(())
            }
            RecordingCommand::Finished(now, position) => {
                times(&[*now, *position])?;
                if *position < 0.0 {
                    return Err(RecordingFailure::Invalid("negative playback position"));
                }
                Ok(())
            }
            RecordingCommand::Finish(now) => now.map_or(Ok(()), |now| times(&[now])),
        }
    }

    /// The PCM bytes this command retains while queued.
    pub fn bytes(&self) -> usize {
        match self {
            RecordingCommand::Input(_, frame) | RecordingCommand::Capture(_, frame) => {
                frame.data.len()
            }
            _ => 0,
        }
    }
}

/// The per-file work bounds: retained PCM, one segment's PCM and the block every step works in.
#[derive(Debug, Clone, Copy)]
pub struct FileLimits {
    pub retained_bytes: usize,
    pub segment_bytes: usize,
    pub block_seconds: f64,
}

/// Between two blocks the caller may stop the file (a failure elsewhere) and yield its thread.
pub type BetweenBlocks<'a> = &'a mut dyn FnMut() -> Result<(), RecordingFailure>;

pub struct RecordingFile<E: OggOpusEncoder, W: Write> {
    writer: W,
    encoder: Option<E>,
    recording: Option<Recorder<E>>,
    placer: SegmentPlacer,
    requested_flush: Option<f64>,
    summary: RecordingSummary,
    sample_rate: u32,
    limits: FileLimits,
}

#[cfg(feature = "ffmpeg")]
impl RecordingFile<crate::ffmpeg::OggOpusEncoder, std::fs::File> {
    /// A stereo Ogg/Opus file at `path`, its directory created.
    pub fn create(
        path: &std::path::Path,
        sample_rate: u32,
        bitrate: u32,
        t0: Option<f64>,
        limits: FileLimits,
    ) -> Result<Self, RecordingFailure> {
        let io = |error: std::io::Error| RecordingFailure::Io(error.to_string());
        if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(io)?;
        }
        let file = std::fs::File::create(path).map_err(io)?;
        let encoder = crate::ffmpeg::OggOpusEncoder::new(sample_rate, 2, bitrate)
            .map_err(|error| RecordingFailure::Encode(error.to_string()))?;
        Ok(Self::new(encoder, file, sample_rate, t0, limits))
    }
}

impl<E: OggOpusEncoder, W: Write> RecordingFile<E, W> {
    pub fn new(
        encoder: E,
        writer: W,
        sample_rate: u32,
        t0: Option<f64>,
        limits: FileLimits,
    ) -> Self {
        Self {
            writer,
            encoder: Some(encoder),
            recording: None,
            placer: SegmentPlacer::new(),
            requested_flush: None,
            summary: RecordingSummary { started_at: t0, ..Default::default() },
            sample_rate,
            limits,
        }
    }

    pub fn summary(&self) -> &RecordingSummary {
        &self.summary
    }

    pub fn retained_bytes(&self) -> usize {
        self.recording.as_ref().map_or(0, Recorder::retained_bytes) + self.placer.retained_bytes()
    }

    /// Applies one command; `Finish` writes the file's tail and returns `true`.
    pub fn apply(
        &mut self,
        command: RecordingCommand,
        between: BetweenBlocks<'_>,
    ) -> Result<bool, RecordingFailure> {
        let now = match &command {
            RecordingCommand::Input(now, _)
            | RecordingCommand::Capture(now, _)
            | RecordingCommand::Started(now)
            | RecordingCommand::Progress(now, _, _)
            | RecordingCommand::Finished(now, _)
            | RecordingCommand::Flush(now) => *now,
            RecordingCommand::Finish(now) => now
                .or(self.requested_flush)
                .or(self.summary.last_flushed)
                .or(self.summary.started_at)
                .unwrap_or(0.0),
        };
        let first_sample = match &command {
            RecordingCommand::Input(now, frame) => now - frame.duration(),
            _ => now,
        };
        let origin = *self.summary.started_at.get_or_insert(first_sample);
        let (sample_rate, block) = (self.sample_rate, self.limits.block_seconds);
        let encoder = &mut self.encoder;
        let recorder = self.recording.get_or_insert_with(|| {
            Recorder::new(encoder.take().expect("one encoder"), sample_rate, origin)
        });
        let placer = &mut self.placer;
        match command {
            RecordingCommand::Input(now, frame) => {
                let output_samples = (frame.samples_per_channel as u64 * sample_rate as u64)
                    .div_ceil(frame.sample_rate as u64);
                if output_samples.saturating_mul(4) > self.limits.retained_bytes as u64 {
                    return Err(RecordingFailure::Overflow("resampled frame PCM"));
                }
                let mut end = now - frame.duration();
                for chunk in chunks(&frame, block) {
                    between()?;
                    end += chunk.duration();
                    recorder.on_input_frame(end, &chunk);
                }
            }
            RecordingCommand::Capture(now, frame) => {
                if frame.data.len()
                    > self.limits.segment_bytes.saturating_sub(placer.pending_bytes())
                {
                    return Err(RecordingFailure::Overflow("segment PCM"));
                }
                let mut start = now;
                for chunk in chunks(&frame, block) {
                    between()?;
                    placer.capture_frame(start, &chunk);
                    start += chunk.duration();
                }
            }
            RecordingCommand::Started(created_at) => placer.on_playback_started(created_at),
            RecordingCommand::Progress(started_at, offset, duration) => {
                let (mut start, mut offset, mut remaining) = (started_at, offset, duration);
                loop {
                    between()?;
                    let Some(placed) =
                        placer.on_playback_progressed_chunk(start, offset, remaining, block)
                    else {
                        break;
                    };
                    let advanced = placed.started_at - start + placed.frame.duration();
                    recorder.on_output_placed(&placed);
                    start += advanced;
                    offset += advanced;
                    remaining -= advanced;
                }
            }
            RecordingCommand::Finished(now, position) => loop {
                between()?;
                let Some(placed) = placer.on_playback_finished_chunk(now, position, block) else {
                    break;
                };
                recorder.on_output_placed(&placed);
            },
            RecordingCommand::Flush(now) => {
                self.requested_flush =
                    Some(self.requested_flush.map_or(now, |previous| previous.max(now)));
                let until = recorder.settle_until(now, placer.pending_since());
                flush(recorder, &mut self.writer, until, &mut self.summary, block, between)?;
            }
            RecordingCommand::Finish(_) => {
                recorder.flush_resamplers().map_err(encode_failure)?;
                flush(recorder, &mut self.writer, now, &mut self.summary, block, between)?;
                let tail = self
                    .recording
                    .take()
                    .expect("recording initialized")
                    .finish()
                    .map_err(encode_failure)?;
                write(&mut self.writer, &tail, &mut self.summary)?;
                return Ok(true);
            }
        }
        Ok(false)
    }
}

fn encode_failure(error: impl std::fmt::Display) -> RecordingFailure {
    RecordingFailure::Encode(error.to_string())
}

/// The frame in blocks of `block` seconds, as PCM.
fn chunks(frame: &AudioFrame, block: f64) -> impl Iterator<Item = PcmFrame> + '_ {
    let samples = (f64::from(frame.sample_rate) * block).round().max(1.) as usize;
    let bytes = samples.saturating_mul(usize::from(frame.num_channels)).saturating_mul(2);
    frame.data.chunks(bytes).map(move |chunk| {
        PcmFrame::new(
            chunk
                .chunks_exact(2)
                .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
                .collect(),
            frame.sample_rate,
            frame.num_channels,
        )
    })
}

fn flush<E: OggOpusEncoder, W: Write>(
    recorder: &mut Recorder<E>,
    writer: &mut W,
    until: f64,
    summary: &mut RecordingSummary,
    block: f64,
    between: BetweenBlocks<'_>,
) -> Result<(), RecordingFailure> {
    let mut cursor = summary.last_flushed.unwrap_or_else(|| summary.started_at.unwrap_or(until));
    while cursor < until {
        between()?;
        let next = (cursor + block).min(until);
        if next <= cursor {
            return Err(RecordingFailure::Invalid(
                "timestamp precision exceeds recording block size",
            ));
        }
        cursor = next;
        let encoded = recorder.flush_until(cursor).map_err(encode_failure)?;
        write(writer, &encoded, summary)?;
        summary.last_flushed = Some(cursor);
    }
    [summary.dropped_input_samples, summary.dropped_output_samples] = recorder.dropped_samples();
    Ok(())
}

fn write<W: Write>(
    writer: &mut W,
    bytes: &[u8],
    summary: &mut RecordingSummary,
) -> Result<(), RecordingFailure> {
    if bytes.is_empty() {
        return Ok(());
    }
    let failed = |error: std::io::Error| RecordingFailure::Io(error.to_string());
    writer.write_all(bytes).map_err(failed)?;
    writer.flush().map_err(failed)?;
    summary.bytes_written += bytes.len() as u64;
    Ok(())
}
