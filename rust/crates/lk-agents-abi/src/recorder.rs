//! Recording observations and the native tap consumed by the live session.

use crate::AudioFrame;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
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
    #[host(kind = "OSError")]
    Io(String),
    #[error("recording worker failed: {0}")]
    Worker(String),
    #[error("recording cleanup deadline elapsed; worker remains owned")]
    IncompleteCleanup,
}

#[derive(Debug, Clone, Default, PartialEq)]
#[crate::host(record(module = "recorder", emit))]
pub struct RecordingSummary {
    pub bytes_written: u64,
    pub started_at: Option<f64>,
    pub last_flushed: Option<f64>,
    pub dropped_input_samples: u64,
    pub dropped_output_samples: u64,
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

/// A nonblocking recording tap. Flush admits ordered work; completion belongs to the recorder.
#[crate::host(native(refused = "expected a native SessionRecorder"))]
pub trait AudioRecorder: Send + Sync {
    fn on_input_frame(&self, now: f64, frame: &AudioFrame);
    fn capture_frame(&self, now: f64, frame: &AudioFrame);
    fn on_playback_started(&self, created_at: f64);
    fn on_playback_progressed(&self, started_at: f64, offset: f64, duration: f64);
    fn on_playback_finished(&self, now: f64, playback_position: f64);
    fn flush(&self, now: f64) -> Result<(), RecordingFailure>;
}

/// One admitted runtime worker owns opening, encoding, writes and finalization.
#[crate::host(class(name = "SessionRecorder", native = AudioRecorder))]
pub trait SessionRecorder: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(
        path: String,
        #[host(kw, default = 48_000)] sample_rate: u32,
        #[host(default = None)] t0: Option<f64>,
        #[host(default = 64_000)] bitrate: u32,
    ) -> Result<Self, RecordingFailure>;

    /// Wait until the native file and encoder are open before attaching the tap.
    #[host(await)]
    async fn ready(&self) -> Result<(), RecordingFailure>;

    #[host(getter)]
    fn bytes_written(&self) -> u64;
    #[host(getter)]
    fn started_at(&self) -> Option<f64>;
    #[host(getter)]
    fn dropped_samples(&self) -> (u64, u64);

    #[host(await)]
    async fn flush(&self, now: f64) -> Result<RecordingSummary, RecordingFailure>;
    #[host(await)]
    async fn finish(
        &self,
        #[host(default = None)] now: Option<f64>,
    ) -> Result<(), RecordingFailure>;
}
