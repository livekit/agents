//! One admitted worker owns a recording's PCM, encoder and ordered writes. Ingress only retains
//! bounded frame references; terminal status is published after the worker and file have settled.

use std::collections::BTreeMap;
use std::io::Write;
#[cfg(feature = "media-ffmpeg")]
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use lk_agents_abi::AudioFrame;
use lk_agents_media::recorder::{
    AudioRecorder, OggOpusEncoder, PcmFrame, Recorder, RecorderError, SegmentPlacer,
};
use tokio::sync::{RwLock, mpsc, oneshot, watch};

use crate::media_workers::{MediaWorkerClass, MediaWorkers};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RecordingSegment {
    pub generation: u64,
    pub segment: u64,
}

const LEGACY_SEGMENT: RecordingSegment = RecordingSegment { generation: 0, segment: 0 };

/// Limits are supplied by the runtime's media budget, rather than selected by an audio callback.
#[derive(Debug, Clone, Copy)]
pub struct RecordingLimits {
    pub queued_commands: usize,
    pub queued_bytes: usize,
    pub frame_bytes: usize,
    pub retained_bytes: usize,
    pub segment_bytes: usize,
    pub segments: usize,
    /// Bound silence generation and encoding work per worker iteration.
    pub block_duration: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecordingFailure {
    #[error("recording capacity exceeded: {0}")]
    Overflow(&'static str),
    #[error("recording has stopped accepting audio")]
    Closed,
    #[error("invalid recording data: {0}")]
    Invalid(&'static str),
    #[error("recording encoding failed: {0}")]
    Encode(String),
    #[error("recording write failed: {0}")]
    Io(String),
    #[error("recording worker failed: {0}")]
    Worker(String),
    #[error("recording cleanup deadline elapsed; worker remains owned")]
    IncompleteCleanup,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecordingSummary {
    pub bytes_written: u64,
    pub started_at: Option<f64>,
    pub last_flushed: Option<f64>,
    pub dropped_samples: [u64; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecordingStatus {
    Open,
    Finishing,
    /// The failure is observable while the owned worker is still settling.
    Failing(RecordingFailure),
    Finished(RecordingSummary),
    Failed(RecordingFailure),
}

type Outcome = Result<RecordingSummary, RecordingFailure>;

enum Command {
    Input(f64, AudioFrame),
    Capture(RecordingSegment, f64, AudioFrame),
    Started(RecordingSegment, f64),
    Progress(RecordingSegment, f64, f64, f64),
    Finished(RecordingSegment, f64, f64),
    Flush(f64, Option<oneshot::Sender<Outcome>>),
    Finish(f64),
    Abandon,
}

struct Queued {
    command: Command,
    bytes: usize,
    /// The terminal command uses the reserved slot and is excluded from data admission.
    data: bool,
}

struct Shared {
    admission: RwLock<()>,
    finishing: AtomicBool,
    commands: AtomicUsize,
    queued_bytes: AtomicUsize,
    retained_bytes: AtomicUsize,
    failure: OnceLock<RecordingFailure>,
    status: watch::Sender<RecordingStatus>,
    limits: RecordingLimits,
}

/// The tap is cheap to share; dropping the final tap requests abandonment. The worker's supervisor
/// retains its actual task until exit even when a close deadline or a calling future is cancelled.
pub struct SessionRecorder {
    tx: mpsc::Sender<Queued>,
    shared: Arc<Shared>,
    runtime: tokio::runtime::Handle,
}

impl Shared {
    fn fail(&self, error: RecordingFailure) -> RecordingFailure {
        let _ = self.failure.set(error.clone());
        let error = self.failure.get().cloned().unwrap_or(error);
        self.status.send_if_modified(|status| {
            if matches!(status, RecordingStatus::Open | RecordingStatus::Finishing) {
                *status = RecordingStatus::Failing(error.clone());
                true
            } else {
                false
            }
        });
        error
    }
}

impl SessionRecorder {
    #[cfg(feature = "media-ffmpeg")]
    pub async fn open(
        workers: &MediaWorkers,
        path: PathBuf,
        sample_rate: u32,
        t0: Option<f64>,
        bitrate: u32,
        limits: RecordingLimits,
    ) -> Result<Arc<Self>, RecordingFailure> {
        Self::open_with(workers, sample_rate, t0, limits, move || {
            if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent).map_err(io_failure)?;
            }
            let file = std::fs::File::create(path).map_err(io_failure)?;
            let encoder = lk_agents_media::OggOpusEncoder::new(sample_rate, 2, bitrate)
                .map_err(|error| RecordingFailure::Encode(error.to_string()))?;
            Ok((encoder, file))
        })
        .await
    }

    async fn open_with<E, W>(
        workers: &MediaWorkers,
        sample_rate: u32,
        t0: Option<f64>,
        limits: RecordingLimits,
        open: impl FnOnce() -> Result<(E, W), RecordingFailure> + Send + 'static,
    ) -> Result<Arc<Self>, RecordingFailure>
    where
        E: OggOpusEncoder + Send + 'static,
        W: Write + Send + 'static,
    {
        if sample_rate == 0
            || t0.is_some_and(|time| !time.is_finite())
            || limits.queued_commands == 0
            || limits.queued_commands >= tokio::sync::Semaphore::MAX_PERMITS
            || limits.queued_bytes == 0
            || limits.frame_bytes == 0
            || limits.retained_bytes == 0
            || limits.segment_bytes == 0
            || limits.segments == 0
            || limits.block_duration.is_zero()
            || limits.block_duration.as_secs_f64() * f64::from(sample_rate) < 1.0
            || limits.block_duration.as_secs_f64() * f64::from(sample_rate) * 4.0
                > limits.retained_bytes as f64
        {
            return Err(RecordingFailure::Invalid("recording format or limits"));
        }
        let (status, _) = watch::channel(RecordingStatus::Open);
        let shared = Arc::new(Shared {
            admission: RwLock::new(()),
            finishing: AtomicBool::new(false),
            commands: AtomicUsize::new(0),
            queued_bytes: AtomicUsize::new(0),
            retained_bytes: AtomicUsize::new(0),
            failure: OnceLock::new(),
            status,
            limits,
        });
        let (tx, rx) = mpsc::channel(limits.queued_commands + 1);
        let runtime = workers.runtime_handle();
        let recorder = Arc::new(Self { tx, shared: shared.clone(), runtime: runtime.clone() });
        let (ready, opened) = oneshot::channel();
        let worker_shared = shared.clone();
        let worker = workers
            .try_spawn(MediaWorkerClass::Recording, move || {
                let (encoder, writer) = match open() {
                    Ok(opened) => opened,
                    Err(error) => {
                        let _ = ready.send(Err(error.clone()));
                        return Err(error);
                    }
                };
                if ready.send(Ok(())).is_err() {
                    return Err(RecordingFailure::Closed);
                }
                run(encoder, writer, sample_rate, t0, rx, worker_shared)
            })
            .map_err(|error| RecordingFailure::Worker(error.to_string()))?;
        runtime.spawn(async move {
            let outcome = worker
                .await
                .unwrap_or_else(|error| Err(RecordingFailure::Worker(error.to_string())));
            if let Err(error) = &outcome {
                shared.fail(error.clone());
            }
            shared.finishing.store(true, Ordering::Release);
            // Let any producer that already claimed ingress capacity finish before clearing
            // accounting; its failed send must not subtract from an already-zeroed counter.
            let _admission = shared.admission.write().await;
            shared.retained_bytes.store(0, Ordering::Release);
            shared.queued_bytes.store(0, Ordering::Release);
            shared.commands.store(0, Ordering::Release);
            shared.status.send_replace(match outcome {
                Ok(summary) => RecordingStatus::Finished(summary),
                Err(error) => {
                    let _ = shared.failure.set(error.clone());
                    RecordingStatus::Failed(error)
                }
            });
        });
        opened.await.map_err(|_| RecordingFailure::Worker("open worker disappeared".into()))??;
        Ok(recorder)
    }

    pub fn status(&self) -> RecordingStatus {
        self.shared.status.borrow().clone()
    }

    pub fn subscribe(&self) -> watch::Receiver<RecordingStatus> {
        self.shared.status.subscribe()
    }

    pub fn retained_bytes(&self) -> usize {
        self.shared
            .queued_bytes
            .load(Ordering::Acquire)
            .saturating_add(self.shared.retained_bytes.load(Ordering::Acquire))
    }

    fn fail(&self, error: RecordingFailure) -> RecordingFailure {
        let error = self.shared.fail(error);
        // The control slot wakes an idle worker; a full queue already guarantees a wake-up.
        let _ = self.tx.try_send(Queued { command: Command::Abandon, bytes: 0, data: false });
        error
    }

    fn submit(&self, command: Command, bytes: usize) -> Result<(), RecordingFailure> {
        if let Some(error) = self.shared.failure.get() {
            return Err(error.clone());
        }
        if self.shared.finishing.load(Ordering::Acquire) {
            return Err(RecordingFailure::Closed);
        }
        let Ok(_admission) = self.shared.admission.try_read() else {
            return Err(RecordingFailure::Closed);
        };
        if self.shared.finishing.load(Ordering::Acquire) {
            return Err(RecordingFailure::Closed);
        }
        if bytes > self.shared.limits.frame_bytes {
            return Err(self.fail(RecordingFailure::Overflow("frame bytes")));
        }
        if self
            .shared
            .commands
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < self.shared.limits.queued_commands).then_some(count + 1)
            })
            .is_err()
        {
            return Err(self.fail(RecordingFailure::Overflow("staging commands")));
        }
        if self
            .shared
            .queued_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                queued.checked_add(bytes).filter(|total| *total <= self.shared.limits.queued_bytes)
            })
            .is_err()
        {
            self.shared.commands.fetch_sub(1, Ordering::AcqRel);
            return Err(self.fail(RecordingFailure::Overflow("staging bytes")));
        }
        if self.tx.try_send(Queued { command, bytes, data: true }).is_err() {
            self.shared.commands.fetch_sub(1, Ordering::AcqRel);
            self.shared.queued_bytes.fetch_sub(bytes, Ordering::AcqRel);
            return Err(
                self.fail(RecordingFailure::Worker("worker stopped accepting commands".into()))
            );
        }
        Ok(())
    }

    pub fn try_on_input_frame(&self, now: f64, frame: AudioFrame) -> Result<(), RecordingFailure> {
        validate_frame(now, &frame).map_err(|error| self.fail(error))?;
        let bytes = frame.data.len();
        self.submit(Command::Input(now, frame), bytes)
    }

    pub fn try_capture_frame(
        &self,
        segment: RecordingSegment,
        now: f64,
        frame: AudioFrame,
    ) -> Result<(), RecordingFailure> {
        validate_frame(now, &frame).map_err(|error| self.fail(error))?;
        let bytes = frame.data.len();
        self.submit(Command::Capture(segment, now, frame), bytes)
    }

    pub fn try_playback_started(
        &self,
        segment: RecordingSegment,
        created_at: f64,
    ) -> Result<(), RecordingFailure> {
        validate_times(&[created_at]).map_err(|error| self.fail(error))?;
        self.submit(Command::Started(segment, created_at), 0)
    }

    pub fn try_playback_progressed(
        &self,
        segment: RecordingSegment,
        started_at: f64,
        offset: f64,
        duration: f64,
    ) -> Result<(), RecordingFailure> {
        validate_times(&[started_at, offset, duration]).map_err(|error| self.fail(error))?;
        if offset < 0.0 || duration < 0.0 {
            return Err(self.fail(RecordingFailure::Invalid("negative playback position")));
        }
        self.submit(Command::Progress(segment, started_at, offset, duration), 0)
    }

    pub fn try_playback_finished(
        &self,
        segment: RecordingSegment,
        now: f64,
        playback_position: f64,
    ) -> Result<(), RecordingFailure> {
        validate_times(&[now, playback_position]).map_err(|error| self.fail(error))?;
        if playback_position < 0.0 {
            return Err(self.fail(RecordingFailure::Invalid("negative playback position")));
        }
        self.submit(Command::Finished(segment, now, playback_position), 0)
    }

    pub async fn flush(&self, now: f64) -> Outcome {
        validate_times(&[now]).map_err(|error| self.fail(error))?;
        let (reply, outcome) = oneshot::channel();
        self.submit(Command::Flush(now, Some(reply)), 0)?;
        match outcome.await {
            Ok(outcome) => outcome,
            Err(_) => self.terminal().await,
        }
    }

    fn request_finish(&self, now: f64) -> Result<(), RecordingFailure> {
        validate_times(&[now]).map_err(|error| self.fail(error))?;
        if !self.shared.finishing.swap(true, Ordering::AcqRel) {
            let shared = self.shared.clone();
            let tx = self.tx.clone();
            // Finalization is owned by the recording, so cancellation of this waiter cannot
            // strand admission closed without delivering the reserved finish barrier.
            self.runtime.spawn(async move {
                let _admission = shared.admission.write().await;
                if matches!(
                    *shared.status.borrow(),
                    RecordingStatus::Finished(_) | RecordingStatus::Failed(_)
                ) {
                    return;
                }
                shared.status.send_if_modified(|status| {
                    if matches!(status, RecordingStatus::Open) {
                        *status = RecordingStatus::Finishing;
                        true
                    } else {
                        false
                    }
                });
                if tx
                    .try_send(Queued { command: Command::Finish(now), bytes: 0, data: false })
                    .is_err()
                {
                    let _ = shared.failure.set(RecordingFailure::Worker(
                        "finish barrier could not be delivered".into(),
                    ));
                }
            });
        }
        Ok(())
    }

    pub async fn finish(&self, now: f64) -> Outcome {
        self.request_finish(now)?;
        self.terminal().await
    }

    pub async fn close(&self, now: f64, deadline: Duration) -> Outcome {
        self.request_finish(now)?;
        match tokio::time::timeout(deadline, self.terminal()).await {
            Ok(outcome) => outcome,
            Err(_) => Err(RecordingFailure::IncompleteCleanup),
        }
    }

    async fn terminal(&self) -> Outcome {
        let mut status = self.subscribe();
        loop {
            match status.borrow_and_update().clone() {
                RecordingStatus::Finished(summary) => return Ok(summary),
                RecordingStatus::Failed(error) => return Err(error),
                RecordingStatus::Open
                | RecordingStatus::Finishing
                | RecordingStatus::Failing(_) => {}
            }
            status
                .changed()
                .await
                .map_err(|_| RecordingFailure::Worker("recording status closed".into()))?;
        }
    }
}

impl Drop for SessionRecorder {
    fn drop(&mut self) {
        if !self.shared.finishing.load(Ordering::Acquire) {
            let _ = self.shared.failure.set(RecordingFailure::Closed);
            let _ = self.tx.try_send(Queued { command: Command::Abandon, bytes: 0, data: false });
        }
    }
}

impl AudioRecorder for SessionRecorder {
    fn on_input_frame(&self, now: f64, frame: &AudioFrame) {
        let _ = self.try_on_input_frame(now, frame.clone());
    }
    fn capture_frame(&self, now: f64, frame: &AudioFrame) {
        let _ = self.try_capture_frame(LEGACY_SEGMENT, now, frame.clone());
    }
    fn on_playback_started(&self, created_at: f64) {
        let _ = self.try_playback_started(LEGACY_SEGMENT, created_at);
    }
    fn on_playback_progressed(&self, started_at: f64, offset: f64, duration: f64) {
        let _ = self.try_playback_progressed(LEGACY_SEGMENT, started_at, offset, duration);
    }
    fn on_playback_finished(&self, now: f64, playback_position: f64) {
        let _ = self.try_playback_finished(LEGACY_SEGMENT, now, playback_position);
    }
    fn flush(&self, now: f64) -> Result<(), RecorderError> {
        // Driver compatibility only submits; the runtime observes terminal failures via subscribe.
        self.submit(Command::Flush(now, None), 0).map_err(|error| {
            RecorderError::Encode(lk_agents_media::recorder::EncodeError(error.to_string()))
        })
    }
}

fn validate_times(values: &[f64]) -> Result<(), RecordingFailure> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(RecordingFailure::Invalid("non-finite timestamp"))
    } else {
        Ok(())
    }
}

fn validate_frame(now: f64, frame: &AudioFrame) -> Result<(), RecordingFailure> {
    validate_times(&[now])?;
    if frame.sample_rate == 0
        || frame.num_channels == 0
        || frame.data.len() != frame.samples_per_channel as usize * frame.num_channels as usize * 2
    {
        return Err(RecordingFailure::Invalid("PCM frame shape"));
    }
    Ok(())
}

fn io_failure(error: std::io::Error) -> RecordingFailure {
    RecordingFailure::Io(error.to_string())
}

fn run<E: OggOpusEncoder, W: Write>(
    encoder: E,
    mut writer: W,
    sample_rate: u32,
    t0: Option<f64>,
    rx: mpsc::Receiver<Queued>,
    shared: Arc<Shared>,
) -> Outcome {
    let outcome = process(encoder, &mut writer, sample_rate, t0, rx, shared.clone());
    if let Err(error) = &outcome {
        shared.fail(error.clone());
    }
    // A filesystem close may block too. Publish Failing before dropping the file and only
    // publish Failed from the supervisor after this function actually exits.
    drop(writer);
    outcome
}

fn process<E: OggOpusEncoder, W: Write>(
    encoder: E,
    writer: &mut W,
    sample_rate: u32,
    t0: Option<f64>,
    mut rx: mpsc::Receiver<Queued>,
    shared: Arc<Shared>,
) -> Outcome {
    let mut summary = RecordingSummary { started_at: t0, ..Default::default() };
    let mut encoder = Some(encoder);
    let mut recording: Option<Recorder<E>> = None;
    let mut segments = BTreeMap::<RecordingSegment, SegmentPlacer>::new();
    while let Some(item) = rx.blocking_recv() {
        if item.data {
            shared.commands.fetch_sub(1, Ordering::AcqRel);
            shared.queued_bytes.fetch_sub(item.bytes, Ordering::AcqRel);
        }
        if let Some(error) = shared.failure.get() {
            return Err(error.clone());
        }
        let now = match &item.command {
            Command::Input(now, _)
            | Command::Capture(_, now, _)
            | Command::Started(_, now)
            | Command::Progress(_, now, _, _)
            | Command::Finished(_, now, _)
            | Command::Flush(now, _)
            | Command::Finish(now) => *now,
            Command::Abandon => return Err(RecordingFailure::Closed),
        };
        let first_sample = match &item.command {
            Command::Input(now, frame) => now - frame.duration(),
            _ => now,
        };
        let origin = *summary.started_at.get_or_insert(first_sample);
        let recorder = recording.get_or_insert_with(|| {
            Recorder::new(encoder.take().expect("one encoder"), sample_rate, origin)
        });
        match item.command {
            Command::Input(now, frame) => {
                let output_samples = (frame.samples_per_channel as u64 * sample_rate as u64)
                    .div_ceil(frame.sample_rate as u64);
                if output_samples.saturating_mul(4) > shared.limits.retained_bytes as u64 {
                    return Err(RecordingFailure::Overflow("resampled frame PCM"));
                }
                recorder.on_input_frame(now, &PcmFrame::from_audio_frame(&frame))
            }
            Command::Capture(segment, now, frame) => {
                if !segments.contains_key(&segment) && segments.len() >= shared.limits.segments {
                    return Err(RecordingFailure::Overflow("segments"));
                }
                let placer = segments.entry(segment).or_default();
                if frame.data.len()
                    > shared.limits.segment_bytes.saturating_sub(placer.pending_bytes())
                {
                    return Err(RecordingFailure::Overflow("segment PCM"));
                }
                placer.capture_frame(now, &PcmFrame::from_audio_frame(&frame));
            }
            Command::Started(segment, created_at) => {
                if !segments.contains_key(&segment) && segments.len() >= shared.limits.segments {
                    return Err(RecordingFailure::Overflow("segments"));
                }
                segments.entry(segment).or_default().on_playback_started(created_at);
            }
            Command::Progress(segment, started_at, offset, duration) => {
                if let Some(placed) = segments
                    .get_mut(&segment)
                    .and_then(|placer| placer.on_playback_progressed(started_at, offset, duration))
                {
                    recorder.on_output_placed(&placed);
                }
            }
            Command::Finished(segment, now, position) => {
                if let Some(mut placer) = segments.remove(&segment)
                    && let Some(placed) = placer.on_playback_finished(now, position)
                {
                    recorder.on_output_placed(&placed);
                }
            }
            Command::Flush(now, reply) => {
                let pending = segments
                    .values()
                    .filter_map(SegmentPlacer::pending_since)
                    .min_by(f64::total_cmp);
                let until = recorder.settle_until(now, pending);
                let outcome = flush(recorder, writer, until, &mut summary, &shared);
                if let Some(reply) = reply {
                    let _ = reply.send(outcome.clone().map(|()| summary.clone()));
                }
                outcome?;
            }
            Command::Finish(now) => {
                recorder
                    .flush_resamplers()
                    .map_err(|error| RecordingFailure::Encode(error.to_string()))?;
                flush(recorder, writer, now, &mut summary, &shared)?;
                let tail = recording
                    .take()
                    .expect("recording initialized")
                    .finish()
                    .map_err(|error| RecordingFailure::Encode(error.to_string()))?;
                write(writer, &tail, &mut summary)?;
                return Ok(summary);
            }
            Command::Abandon => return Err(RecordingFailure::Closed),
        }
        let retained = recorder.retained_bytes()
            + segments.values().map(SegmentPlacer::retained_bytes).sum::<usize>();
        shared.retained_bytes.store(retained, Ordering::Release);
        if retained > shared.limits.retained_bytes {
            return Err(RecordingFailure::Overflow("retained PCM"));
        }
    }
    Err(RecordingFailure::Closed)
}

fn flush<E: OggOpusEncoder, W: Write>(
    recorder: &mut Recorder<E>,
    writer: &mut W,
    until: f64,
    summary: &mut RecordingSummary,
    shared: &Shared,
) -> Result<(), RecordingFailure> {
    let mut cursor = summary.last_flushed.unwrap_or_else(|| summary.started_at.unwrap_or(until));
    while cursor < until {
        if let Some(error) = shared.failure.get() {
            return Err(error.clone());
        }
        let next = (cursor + shared.limits.block_duration.as_secs_f64()).min(until);
        if next <= cursor {
            return Err(RecordingFailure::Invalid(
                "timestamp precision exceeds recording block size",
            ));
        }
        cursor = next;
        let encoded = recorder
            .flush_until(cursor)
            .map_err(|error| RecordingFailure::Encode(error.to_string()))?;
        write(writer, &encoded, summary)?;
        summary.last_flushed = Some(cursor);
    }
    summary.dropped_samples = recorder.dropped_samples();
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
    writer.write_all(bytes).map_err(io_failure)?;
    writer.flush().map_err(io_failure)?;
    summary.bytes_written += bytes.len() as u64;
    Ok(())
}

#[cfg(test)]
mod tests;
