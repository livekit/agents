//! One admitted worker owns a recording's PCM, encoder and ordered writes. Ingress only retains
//! bounded frame references; terminal status is published after the worker and file have settled.

use std::io::Write;
#[cfg(feature = "media-ffmpeg")]
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use lk_agents_abi::AudioFrame;
use lk_agents_abi::recorder::AudioRecorder;
use lk_agents_media::recorder::{FileLimits, OggOpusEncoder, RecordingCommand, RecordingFile};
use tokio::sync::{RwLock, mpsc, oneshot, watch};

use lk_agents_media::workers::{MediaWorkerClass, MediaWorkers};

/// Limits are supplied by the runtime's media budget, rather than selected by an audio callback.
#[derive(Debug, Clone, Copy)]
struct RecordingLimits {
    pub queued_commands: usize,
    pub queued_bytes: usize,
    pub frame_bytes: usize,
    pub retained_bytes: usize,
    pub segment_bytes: usize,
    /// Bound silence generation and encoding work per worker iteration.
    pub block_duration: Duration,
}

pub use lk_agents_abi::recorder::{RecordingFailure, RecordingStatus, RecordingSummary};

type Outcome = Result<RecordingSummary, RecordingFailure>;

enum Command {
    Record(RecordingCommand),
    Flush(f64, Option<oneshot::Sender<Outcome>>),
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
    cleanup_reported: AtomicBool,
    commands: AtomicUsize,
    queued_bytes: AtomicUsize,
    retained_bytes: AtomicUsize,
    failure: OnceLock<RecordingFailure>,
    status: watch::Sender<RecordingStatus>,
    ready: watch::Sender<Option<Result<(), RecordingFailure>>>,
    summary: watch::Sender<RecordingSummary>,
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
    fn check_failure(&self) -> Result<(), RecordingFailure> {
        self.failure.get().cloned().map_or(Ok(()), Err)
    }

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
    fn start(
        workers: &MediaWorkers,
        path: PathBuf,
        sample_rate: u32,
        t0: Option<f64>,
        bitrate: u32,
        limits: RecordingLimits,
    ) -> Result<Self, RecordingFailure> {
        Self::start_with(workers, sample_rate, t0, limits, move || {
            if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent).map_err(io_failure)?;
            }
            let file = std::fs::File::create(path).map_err(io_failure)?;
            let encoder = lk_agents_media::OggOpusEncoder::new(sample_rate, 2, bitrate)
                .map_err(|error| RecordingFailure::Encode(error.to_string()))?;
            Ok((encoder, file))
        })
    }

    #[cfg(test)]
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
        let recorder = Arc::new(Self::start_with(workers, sample_rate, t0, limits, open)?);
        recorder.ready().await?;
        Ok(recorder)
    }

    fn start_with<E, W>(
        workers: &MediaWorkers,
        sample_rate: u32,
        t0: Option<f64>,
        limits: RecordingLimits,
        open: impl FnOnce() -> Result<(E, W), RecordingFailure> + Send + 'static,
    ) -> Result<Self, RecordingFailure>
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
            || limits.block_duration.is_zero()
            || limits.block_duration.as_secs_f64() * f64::from(sample_rate) < 1.0
            || limits.block_duration.as_secs_f64() * f64::from(sample_rate) * 4.0
                > limits.retained_bytes as f64
        {
            return Err(RecordingFailure::Invalid("recording format or limits"));
        }
        let (status, _) = watch::channel(RecordingStatus::Open);
        let (ready, _) = watch::channel(None);
        let (summary, _) =
            watch::channel(RecordingSummary { started_at: t0, ..Default::default() });
        let shared = Arc::new(Shared {
            admission: RwLock::new(()),
            finishing: AtomicBool::new(false),
            cleanup_reported: AtomicBool::new(false),
            commands: AtomicUsize::new(0),
            queued_bytes: AtomicUsize::new(0),
            retained_bytes: AtomicUsize::new(0),
            failure: OnceLock::new(),
            status,
            ready,
            summary,
            limits,
        });
        let (tx, rx) = mpsc::channel(limits.queued_commands + 1);
        let runtime = crate::process::caller_handle()
            .map_err(|error| RecordingFailure::Worker(error.to_string()))?;
        let recorder = Self { tx, shared: shared.clone(), runtime: runtime.clone() };
        let worker_shared = shared.clone();
        let worker = workers
            .try_spawn(MediaWorkerClass::Recording, move || {
                let (encoder, writer) = match open() {
                    Ok(opened) => opened,
                    Err(error) => {
                        worker_shared.ready.send_replace(Some(Err(error.clone())));
                        return Err(error);
                    }
                };
                worker_shared.ready.send_replace(Some(Ok(())));
                run(encoder, writer, sample_rate, t0, rx, worker_shared)
            })
            .map_err(|error| RecordingFailure::Worker(error.to_string()))?;
        runtime.spawn(async move {
            let outcome = worker
                .await
                .unwrap_or_else(|error| Err(RecordingFailure::Worker(error.to_string())));
            if let Err(error) = &outcome {
                shared.fail(error.clone());
                shared.ready.send_if_modified(|ready| {
                    if ready.is_none() {
                        *ready = Some(Err(error.clone()));
                        true
                    } else {
                        false
                    }
                });
            } else if let Ok(summary) = &outcome {
                shared.summary.send_replace(summary.clone());
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
        Ok(recorder)
    }

    pub async fn ready(&self) -> Result<(), RecordingFailure> {
        let mut ready = self.shared.ready.subscribe();
        loop {
            if let Some(result) = ready.borrow_and_update().clone() {
                return result;
            }
            ready
                .changed()
                .await
                .map_err(|_| RecordingFailure::Worker("open worker disappeared".into()))?;
        }
    }

    #[cfg(test)]
    fn status(&self) -> RecordingStatus {
        self.shared.status.borrow().clone()
    }

    fn subscribe(&self) -> watch::Receiver<RecordingStatus> {
        self.shared.status.subscribe()
    }

    #[cfg(test)]
    fn retained_bytes(&self) -> usize {
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

    fn try_on_input_frame(&self, now: f64, frame: AudioFrame) -> Result<(), RecordingFailure> {
        validate_frame(now, &frame).map_err(|error| self.fail(error))?;
        let bytes = frame.data.len();
        self.submit(Command::Record(RecordingCommand::Input(now, frame)), bytes)
    }

    fn try_capture_frame(&self, now: f64, frame: AudioFrame) -> Result<(), RecordingFailure> {
        validate_frame(now, &frame).map_err(|error| self.fail(error))?;
        let bytes = frame.data.len();
        self.submit(Command::Record(RecordingCommand::Capture(now, frame)), bytes)
    }

    fn try_playback_started(&self, created_at: f64) -> Result<(), RecordingFailure> {
        validate_times(&[created_at]).map_err(|error| self.fail(error))?;
        self.submit(Command::Record(RecordingCommand::Started(created_at)), 0)
    }

    fn try_playback_progressed(
        &self,
        started_at: f64,
        offset: f64,
        duration: f64,
    ) -> Result<(), RecordingFailure> {
        validate_times(&[started_at, offset, duration]).map_err(|error| self.fail(error))?;
        if offset < 0.0 || duration < 0.0 {
            return Err(self.fail(RecordingFailure::Invalid("negative playback position")));
        }
        self.submit(Command::Record(RecordingCommand::Progress(started_at, offset, duration)), 0)
    }

    fn try_playback_finished(
        &self,
        now: f64,
        playback_position: f64,
    ) -> Result<(), RecordingFailure> {
        validate_times(&[now, playback_position]).map_err(|error| self.fail(error))?;
        if playback_position < 0.0 {
            return Err(self.fail(RecordingFailure::Invalid("negative playback position")));
        }
        self.submit(Command::Record(RecordingCommand::Finished(now, playback_position)), 0)
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

    fn request_finish(&self, now: Option<f64>) -> Result<(), RecordingFailure> {
        if let Some(now) = now {
            validate_times(&[now]).map_err(|error| self.fail(error))?;
        }
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
                    .try_send(Queued {
                        command: Command::Record(RecordingCommand::Finish(now)),
                        bytes: 0,
                        data: false,
                    })
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
        self.request_finish(Some(now))?;
        self.terminal().await
    }

    async fn close(&self, now: Option<f64>, deadline: Duration) -> Outcome {
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
        let _ = self.try_capture_frame(now, frame.clone());
    }
    fn on_playback_started(&self, created_at: f64) {
        let _ = self.try_playback_started(created_at);
    }
    fn on_playback_progressed(&self, started_at: f64, offset: f64, duration: f64) {
        let _ = self.try_playback_progressed(started_at, offset, duration);
    }
    fn on_playback_finished(&self, now: f64, playback_position: f64) {
        let _ = self.try_playback_finished(now, playback_position);
    }
    fn flush(&self, now: f64) -> Result<(), RecordingFailure> {
        self.submit(Command::Flush(now, None), 0)
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
    writer: W,
    sample_rate: u32,
    t0: Option<f64>,
    rx: mpsc::Receiver<Queued>,
    shared: Arc<Shared>,
) -> Outcome {
    let limits = FileLimits {
        retained_bytes: shared.limits.retained_bytes,
        segment_bytes: shared.limits.segment_bytes,
        block_seconds: shared.limits.block_duration.as_secs_f64(),
    };
    let mut file = RecordingFile::new(encoder, writer, sample_rate, t0, limits);
    let outcome = process(&mut file, rx, &shared);
    if let Err(error) = &outcome {
        shared.fail(error.clone());
    }
    // A filesystem close may block too. Publish Failing before dropping the file and only
    // publish Failed from the supervisor after this function actually exits.
    drop(file);
    outcome
}

fn process<E: OggOpusEncoder, W: Write>(
    file: &mut RecordingFile<E, W>,
    mut rx: mpsc::Receiver<Queued>,
    shared: &Shared,
) -> Outcome {
    // Between blocks: stop on a failure reported elsewhere, and let other workers run.
    let mut between = || {
        shared.check_failure()?;
        std::thread::yield_now();
        Ok(())
    };
    while let Some(item) = rx.blocking_recv() {
        if item.data {
            shared.commands.fetch_sub(1, Ordering::AcqRel);
            shared.queued_bytes.fetch_sub(item.bytes, Ordering::AcqRel);
        }
        shared.check_failure()?;
        match item.command {
            Command::Record(command) => {
                if file.apply(command, &mut between)? {
                    return Ok(file.summary().clone());
                }
            }
            Command::Flush(now, reply) => {
                let outcome = file.apply(RecordingCommand::Flush(now), &mut between).map(drop);
                if let Some(reply) = reply {
                    let _ = reply.send(outcome.clone().map(|()| file.summary().clone()));
                }
                outcome?;
            }
            Command::Abandon => return Err(RecordingFailure::Closed),
        }
        let retained = file.retained_bytes();
        shared.summary.send_replace(file.summary().clone());
        shared.retained_bytes.store(retained, Ordering::Release);
        if retained > shared.limits.retained_bytes {
            return Err(RecordingFailure::Overflow("retained PCM"));
        }
    }
    Err(RecordingFailure::Closed)
}

#[cfg(test)]
mod tests;

impl lk_agents_abi::recorder::SessionRecorder for SessionRecorder {
    fn new(
        path: String,
        sample_rate: u32,
        t0: Option<f64>,
        bitrate: u32,
    ) -> Result<Self, RecordingFailure> {
        #[cfg(feature = "media-ffmpeg")]
        {
            let workers =
                crate::process::workers().map_err(|e| RecordingFailure::Worker(e.to_string()))?;
            Self::start(
                &workers,
                path.into(),
                sample_rate,
                t0,
                bitrate,
                RecordingLimits {
                    queued_commands: 256,
                    queued_bytes: 8 * 1024 * 1024,
                    frame_bytes: 2 * 1024 * 1024,
                    retained_bytes: 16 * 1024 * 1024,
                    segment_bytes: 8 * 1024 * 1024,
                    block_duration: Duration::from_millis(20),
                },
            )
        }
        #[cfg(not(feature = "media-ffmpeg"))]
        {
            let _ = (path, sample_rate, t0, bitrate);
            Err(RecordingFailure::Worker("recording requires the ffmpeg build".into()))
        }
    }

    async fn ready(&self) -> Result<(), RecordingFailure> {
        Self::ready(self).await
    }
    fn bytes_written(&self) -> u64 {
        self.shared.summary.borrow().bytes_written
    }
    fn started_at(&self) -> Option<f64> {
        self.shared.summary.borrow().started_at
    }
    fn dropped_samples(&self) -> (u64, u64) {
        let summary = self.shared.summary.borrow();
        (summary.dropped_input_samples, summary.dropped_output_samples)
    }
    async fn flush(&self, now: f64) -> Result<RecordingSummary, RecordingFailure> {
        Self::flush(self, now).await
    }
    async fn finish(&self, now: Option<f64>) -> Result<(), RecordingFailure> {
        match self.close(now, lk_agents_abi::HOST_TEARDOWN_TIMEOUT).await {
            Ok(_) => Ok(()),
            Err(RecordingFailure::IncompleteCleanup) => {
                if !self.shared.cleanup_reported.swap(true, Ordering::AcqRel) {
                    log::error!("recording cleanup incomplete; runtime retains the native worker");
                }
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}
