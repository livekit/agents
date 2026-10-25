//! A host `io.AudioInput` / `VideoInput` / `AudioOutput` / `TextOutput` is its trait through the
//! trait's bare `#[host(proxy)]`. A host sink reports its playback into the [`PlaybackTracker`] its
//! `_core_tracker()` hands over, stamped on the session's clock by [`PlaybackTracker::anchor_to`].

use crate::voice::rtc::TrackSid;
use crate::{AudioFrame, TimedString, Ts, WallAnchor};
use async_trait::async_trait;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio::sync::Notify as IoChainNotify;

pub use crate::voice::rtc::AudioOutputCapabilities;

/// `PlaybackFinishedEvent` (the ABI's).
pub use crate::voice::room_io::PlaybackFinished;

/// `PlaybackStartedEvent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaybackStarted {
    pub created_at: Ts,
}

/// `PlaybackProgressedEvent`: a stretch of the current segment that played, never revised
/// (`RecorderIO` needs when it played).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaybackProgressed {
    pub started_at: Ts,
    /// Where it starts within the audio captured for the current segment.
    pub offset: Duration,
    pub duration: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaybackEvent {
    Started(PlaybackStarted),
    Progressed(PlaybackProgressed),
    Finished(PlaybackFinished),
}

/// Something that wants the playback events of one [`AudioOutput`]. Called synchronously from the
/// raising task and must not block: the parent's counter must have moved when `emit` returns.
pub trait PlaybackObserver: Send + Sync {
    fn on_playback_event(&self, ev: &PlaybackEvent);

    /// A synchroniser below released `text` as of the played position. Ignored by default.
    fn on_transcript_shown(&self, text: &str) {
        let _ = text;
    }

    /// The sink's playback count was reset (`_reset_playback_count`); parked waits are released
    /// as interrupted.
    fn on_playback_reset(&self) {}

    /// Nobody listens any more; the tracker drops this observer after the next event.
    fn is_closed(&self) -> bool {
        false
    }
}

#[derive(Debug)]
struct TrackerState {
    /// Between the first `capture_frame` of a segment and its `flush`.
    capturing: bool,
    segments: u64,
    finished: u64,
    last: PlaybackFinished,
    /// Bumped by [`PlaybackTracker::reset`]; a waiter latches it so a reset releases it.
    epoch: u64,
    progressed: Duration,
}

impl TrackerState {
    fn open_progress(&self) -> Duration {
        if self.finished < self.segments { self.progressed } else { Duration::ZERO }
    }
}

/// The segment bookkeeping every [`AudioOutput`] shares, plus the fan-out to observers. Also the
/// observer a child registers with, so a terminal's event counts at every level.
pub struct PlaybackTracker {
    state: Mutex<TrackerState>,
    wake: IoChainNotify,
    observers: Mutex<Vec<Arc<dyn PlaybackObserver>>>,
    /// The session's clock as wall seconds, for a host sink's `time.time()` stamps. Set once.
    anchor: OnceLock<WallAnchor>,
}

impl std::fmt::Debug for PlaybackTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlaybackTracker").field("state", &*self.state.lock().unwrap()).finish()
    }
}

impl Default for PlaybackTracker {
    fn default() -> Self {
        PlaybackTracker::new()
    }
}

/// A host sink's side (`io.AudioOutput._core_tracker()`): it reports its playback events here, in
/// its own `time.time()` stamps and float seconds.
#[crate::host(object(cell = "arc"))]
impl PlaybackTracker {
    #[host(new)]
    pub fn new() -> Self {
        PlaybackTracker {
            state: Mutex::new(TrackerState {
                capturing: false,
                segments: 0,
                finished: 0,
                last: PlaybackFinished::default(),
                epoch: 0,
                progressed: Duration::ZERO,
            }),
            wake: IoChainNotify::new(),
            observers: Mutex::new(Vec::new()),
            anchor: OnceLock::new(),
        }
    }

    /// Stamp a host sink's reports on the session's clock. The first anchor wins: one host sink
    /// reports to one session. Unanchored, reports are stamped on a fresh wall clock.
    pub fn anchor_to(&self, anchor: WallAnchor) {
        let _ = self.anchor.set(anchor);
    }

    fn anchor(&self) -> WallAnchor {
        *self.anchor.get_or_init(|| WallAnchor::of(&crate::RealClock::new()))
    }

    /// `on_playback_started(created_at=)`.
    #[host(method(name = "playback_started"))]
    pub fn report_started(&self, created_at: f64) {
        self.playback_started(self.anchor().ts(created_at));
    }

    /// `on_playback_progressed(started_at=, offset=, duration=)`. A negative or non-finite figure
    /// (a buggy sink) is zero.
    #[host(method(name = "playback_progressed"))]
    pub fn report_progressed(&self, started_at: f64, offset: f64, duration: f64) {
        self.playback_progressed(self.anchor().ts(started_at), seconds(offset), seconds(duration));
    }

    /// `on_playback_finished(..)`. `False` when no segment was outstanding (nothing is emitted).
    #[host(method(name = "playback_finished"))]
    pub fn report_finished(
        &self,
        playback_position: f64,
        interrupted: bool,
        #[host(required)] synchronized_transcript: Option<String>,
    ) -> bool {
        self.playback_finished(PlaybackFinished {
            playback_position: seconds(playback_position),
            interrupted,
            synchronized_transcript,
        })
    }

    /// Forward this node's events to `observer` (`next.on("playback_finished", ...)`). Pass the
    /// parent's tracker, not the node, so the chain isn't a reference cycle.
    pub fn observe(&self, observer: Arc<dyn PlaybackObserver>) {
        self.observers.lock().unwrap().push(observer);
    }

    /// Stop forwarding to `observer` (`next.off(...)`), by pointer.
    pub fn unobserve(&self, observer: &Arc<dyn PlaybackObserver>) {
        self.observers.lock().unwrap().retain(|o| !Arc::ptr_eq(o, observer));
    }

    /// Cloned out: an observer is often a parent tracker that re-enters, and `Mutex` isn't reentrant.
    fn observers(&self) -> Vec<Arc<dyn PlaybackObserver>> {
        self.observers.lock().unwrap().clone()
    }

    fn emit(&self, ev: PlaybackEvent) {
        for o in self.observers() {
            o.on_playback_event(&ev);
        }
        // Whoever stopped listening is forgotten here, not at `aclose`.
        self.observers.lock().unwrap().retain(|o| !o.is_closed());
    }

    /// A synchroniser's release of `text`, fanned out like a playback event.
    pub fn transcript_shown(&self, text: &str) {
        for o in self.observers() {
            o.on_transcript_shown(text);
        }
    }

    /// `AudioOutput.capture_frame`'s `super()` half: the first frame after a flush opens a segment.
    pub fn note_capture(&self) {
        let mut s = self.state.lock().unwrap();
        if !s.capturing {
            s.capturing = true;
            s.segments += 1;
            // A new segment: what was heard of the previous one is not this one's position.
            s.progressed = Duration::ZERO;
        }
    }

    /// `AudioOutput.flush`'s `super()` half: the segment is closed; its playout is still owed.
    pub fn note_flush(&self) {
        self.state.lock().unwrap().capturing = false;
    }

    /// `_reset_playback_count`, plus: a parked `wait_for_playout` is released with
    /// `Finished { interrupted: true }` at what the open segment played (never the previous
    /// segment's finish), rather than left waiting forever.
    pub fn reset(&self) {
        let mut s = self.state.lock().unwrap();
        let played = s.open_progress();
        s.last = PlaybackFinished {
            playback_position: played,
            interrupted: true,
            synchronized_transcript: None,
        };
        s.segments = 0;
        s.finished = 0;
        s.capturing = false;
        s.epoch += 1;
        s.progressed = Duration::ZERO;
        drop(s);
        self.wake.notify_waiters();
        self.on_playback_reset();
    }

    /// The playhead per segment: what the sink reported played of the open segment, zero when
    /// none is open. Never the previous segment's finish (a cut speech would truncate at it).
    pub fn position(&self) -> Duration {
        self.state.lock().unwrap().open_progress()
    }

    pub fn playback_started(&self, created_at: Ts) {
        self.emit(PlaybackEvent::Started(PlaybackStarted { created_at }));
    }

    pub fn playback_progressed(&self, started_at: Ts, offset: Duration, duration: Duration) {
        self.state.lock().unwrap().progressed += duration;
        self.emit(PlaybackEvent::Progressed(PlaybackProgressed { started_at, offset, duration }));
    }

    /// A segment finished playing. With no outstanding segment it returns `false` and emits
    /// nothing: a spurious `finished` would release the next segment's wait.
    pub fn playback_finished(&self, ev: PlaybackFinished) -> bool {
        {
            let mut s = self.state.lock().unwrap();
            if s.finished >= s.segments {
                return false;
            }
            s.finished += 1;
            s.last = ev.clone();
            s.progressed = Duration::ZERO;
        }
        self.wake.notify_waiters();
        self.emit(PlaybackEvent::Finished(ev));
        true
    }

    /// Wait for every segment captured so far; returns the last event. The target is latched at
    /// entry, as the reference does, or this never returns while the agent talks.
    pub async fn wait_for_playout(&self) -> PlaybackFinished {
        let (target, epoch) = {
            let s = self.state.lock().unwrap();
            (s.segments, s.epoch)
        };
        loop {
            // Registered before the check, or a `finished` between the two is missed.
            let parked = self.wake.notified();
            tokio::pin!(parked);
            parked.as_mut().enable();
            {
                let s = self.state.lock().unwrap();
                if s.epoch != epoch {
                    // Reset while parked: `last` is the release the reset wrote.
                    return s.last.clone();
                }
                if s.finished >= target {
                    return s.last.clone();
                }
            }
            parked.await;
        }
    }
}

impl PlaybackObserver for PlaybackTracker {
    /// Forward a child's event as this node's own (`_forward_next_playback_finished`).
    fn on_playback_event(&self, ev: &PlaybackEvent) {
        match ev {
            PlaybackEvent::Started(e) => self.playback_started(e.created_at),
            PlaybackEvent::Progressed(e) => {
                self.playback_progressed(e.started_at, e.offset, e.duration)
            }
            PlaybackEvent::Finished(e) => {
                self.playback_finished(e.clone());
            }
        }
    }

    fn on_transcript_shown(&self, text: &str) {
        self.transcript_shown(text);
    }

    /// A reset below is told above, not repeated: each node's count is its own.
    fn on_playback_reset(&self) {
        for o in self.observers() {
            o.on_playback_reset();
        }
    }
}

/// Host seconds as a [`Duration`]; a negative, NaN or infinite figure is zero.
fn seconds(s: f64) -> Duration {
    if s.is_finite() && s > 0.0 { Duration::from_secs_f64(s) } else { Duration::ZERO }
}

/// What a host wrapper over a core leg hears of its playback (`_ParticipantAudioOutput`, re-emitting
/// the leg's events as its own): `io.AudioOutput`'s `on_playback_*`, through its `_core_*` surface.
#[crate::host(proxy(host = "livekit.agents.voice.io.AudioOutput"))]
pub trait PlaybackListener: Send + Sync {
    fn on_playback_started(&self, created_at: f64);
    fn on_playback_progressed(&self, started_at: f64, offset: f64, duration: f64);
    fn on_playback_finished(
        &self,
        playback_position: f64,
        interrupted: bool,
        synchronized_transcript: Option<String>,
    );
}

/// A [`PlaybackListener`] as a tracker's observer, stamps in wall seconds against `anchor`.
pub struct ListenerObserver {
    pub listener: Arc<dyn PlaybackListener>,
    pub anchor: WallAnchor,
}

impl PlaybackObserver for ListenerObserver {
    fn on_playback_event(&self, ev: &PlaybackEvent) {
        match ev {
            PlaybackEvent::Started(e) => {
                self.listener.on_playback_started(self.anchor.wall(e.created_at))
            }
            PlaybackEvent::Progressed(e) => self.listener.on_playback_progressed(
                self.anchor.wall(e.started_at),
                e.offset.as_secs_f64(),
                e.duration.as_secs_f64(),
            ),
            PlaybackEvent::Finished(e) => self.listener.on_playback_finished(
                e.playback_position.as_secs_f64(),
                e.interrupted,
                e.synchronized_transcript.clone(),
            ),
        }
    }
}

/// `io.AudioInput`: the source end of the audio-in chain. Exactly one reader: two tasks pulling
/// one host-backed async iterator lose data.
#[crate::host(proxy(host = "livekit.agents.voice.io.AudioInput"))]
#[async_trait]
pub trait AudioInput: Send + Sync {
    fn label(&self) -> &str;

    /// The next frame, or `None` at end of stream.
    async fn next_frame(&self) -> Option<AudioFrame>;

    /// The chain was enabled; frames are dropped while detached, so a re-attach resumes mid-stream.
    fn on_attached(&self) {}
    fn on_detached(&self) {}
}

/// `io.VideoInput`: the source end of the video-in chain; one reader. Frames are side-table keys.
#[crate::host(proxy(host = "livekit.agents.voice.io.VideoInput"))]
#[async_trait]
pub trait VideoInput: Send + Sync {
    fn label(&self) -> &str;

    /// The next frame, or `None` at end of stream.
    async fn next_frame(&self) -> Option<crate::voice::rtc::VideoFrameRef>;

    /// `set_video_enabled`: frames are dropped while detached.
    fn on_attached(&self) {}
    fn on_detached(&self) {}
}

/// `io.AudioOutput`: one link in the audio-out chain. Implement [`capture`](Self::capture) and
/// [`flush_segment`](Self::flush_segment); call [`AudioOutputExt::capture_frame`] and [`AudioOutputExt::flush`].
#[crate::host(proxy(host = "livekit.agents.voice.io.AudioOutput"))]
#[async_trait]
pub trait AudioOutput: Send + Sync {
    fn label(&self) -> &str;

    /// This node's own bookkeeping and event fan-out. One per node, never shared.
    fn tracker(&self) -> &Arc<PlaybackTracker>;

    fn capabilities(&self) -> AudioOutputCapabilities;

    /// The rate this sink requires, or `None` if it takes any.
    fn sample_rate(&self) -> Option<u32> {
        None
    }

    fn next_in_chain(&self) -> Option<Arc<dyn AudioOutput>> {
        None
    }

    /// Push one frame downstream. Callers use [`AudioOutputExt::capture_frame`]. Awaited on a host
    /// sink, where backpressure lives.
    async fn capture(&self, frame: &AudioFrame) -> Result<()>;

    /// End the current segment. **Not the caller's entry point** -- use [`AudioOutputExt::flush`].
    fn flush_segment(&self);

    /// Throw away buffered audio and stop playback immediately. Synchronous: this is interruption.
    fn clear_buffer(&self);

    fn on_attached(&self) {
        if let Some(next) = self.next_in_chain() {
            next.on_attached();
        }
    }

    fn on_detached(&self) {
        if let Some(next) = self.next_in_chain() {
            next.on_detached();
        }
    }

    fn pause(&self) {
        if let Some(next) = self.next_in_chain() {
            next.pause();
        }
    }

    fn resume(&self) {
        if let Some(next) = self.next_in_chain() {
            next.resume();
        }
    }
}

/// The caller-facing half of [`AudioOutput`]. Blanket-implemented; never implement it directly.
#[async_trait]
pub trait AudioOutputExt: AudioOutput {
    async fn capture_frame(&self, frame: &AudioFrame) -> Result<()> {
        self.tracker().note_capture();
        self.capture(frame).await
    }

    fn flush(&self) {
        self.tracker().note_flush();
        self.flush_segment();
    }

    /// Cut the current segment: `flush()` then `clear_buffer()`, as the reference's cancelled
    /// forwarding does. The only way to cut: a bare `clear_buffer()` leaves the segment open.
    fn cut(&self) {
        self.flush();
        self.clear_buffer();
    }

    async fn wait_for_playout(&self) -> PlaybackFinished {
        self.tracker().wait_for_playout().await
    }

    /// Only if every link can pause, or wrapper and terminal desynchronise.
    fn can_pause(&self) -> bool {
        self.capabilities().pause && self.next_in_chain().map(|n| n.can_pause()).unwrap_or(true)
    }
}

impl<T: AudioOutput + ?Sized> AudioOutputExt for T {}

/// `io.TextOutput`: one link in the transcript chain; `flush` is the whole protocol.
#[crate::host(proxy(host = "livekit.agents.voice.io.TextOutput"))]
#[async_trait]
pub trait TextOutput: Send + Sync {
    fn label(&self) -> &str;

    fn next_in_chain(&self) -> Option<Arc<dyn TextOutput>> {
        None
    }

    /// Capture a delta with word timings (Python's `TimedString` is a `str`; here it's a separate
    /// method whose default drops the timings).
    async fn capture_timed(&self, text: &TimedString) -> Result<()> {
        self.capture_text(&text.text).await
    }

    /// Capture a text delta. Must not drop: a lost token mangles the transcript and the speech.
    async fn capture_text(&self, text: &str) -> Result<()>;

    /// The generation is complete; close the segment.
    fn flush(&self);

    fn on_attached(&self) {
        if let Some(next) = self.next_in_chain() {
            next.on_attached();
        }
    }

    fn on_detached(&self) {
        if let Some(next) = self.next_in_chain() {
            next.on_detached();
        }
    }
}

pub type Result<T> = std::result::Result<T, IoError>;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[crate::host(error(kind = "RuntimeError"))]
pub enum IoError {
    /// A room failure other than the host's (`lk-agents-rtc`'s `RtcError`).
    #[error("{0}")]
    Rtc(String),

    /// Pre-connect audio didn't arrive in time; the caller warns and proceeds with the live stream.
    #[error("timed out waiting for the pre-connect audio buffer for track {0}")]
    PreConnectTimeout(TrackSid),

    /// A malformed pre-connect stream header.
    #[error("pre-connect audio stream is malformed: {0}")]
    PreConnectMalformed(String),

    /// soxr failed (pre-connect buffer path only).
    #[error("resample failed: {0}")]
    Resample(String),

    /// A host-backed chain node failed; the host's message verbatim.
    #[error("{0}")]
    Host(#[from] crate::errors::HostError),
}

/// A host failure keeps the original exception: re-raising it raises what the host raised.
#[cfg(feature = "python")]
impl crate::python::FromHostErr for IoError {
    fn from_host(py: pyo3::Python<'_>, err: &pyo3::PyErr) -> Self {
        IoError::Host(crate::python::capture_host(py, err))
    }
}

impl From<crate::ResampleError> for IoError {
    fn from(e: crate::ResampleError) -> Self {
        IoError::Resample(e.to_string())
    }
}
