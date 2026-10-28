//! A host `io.AudioInput` / `VideoInput` / `AudioOutput` / `TextOutput` is its trait through the
//! generated proxy, which calls the host object's `_core_view` view. A host sink reports its playback into the [`PlaybackTracker`] its
//! `_core_tracker()` hands over, stamped on the session's clock by [`PlaybackTracker::anchor_to`].

use crate::voice::rtc::TrackSid;
use crate::{AudioFrame, TimedString, Ts, WallAnchor};
use async_trait::async_trait;
use std::collections::BTreeMap;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Poll, Waker};
use std::time::Duration;

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

    /// A synchroniser below released `word` (where its audio starts and ends in the segment) as
    /// of the played position. Ignored by default.
    fn on_transcript_shown(&self, word: &TimedString) {
        let _ = word;
    }

    /// The sink's playback count was reset (`_reset_playback_count`); parked waits are released
    /// as interrupted.
    fn on_playback_reset(&self) {}

    /// The tracker's `segment`th segment (counted from 1 over the tracker's life) finished, so a
    /// listener tells its own segments' reports from a late one of an earlier segment.
    fn on_segment_finished(&self, segment: u64, ev: &PlaybackFinished) {
        let _ = segment;
        self.on_playback_event(&PlaybackEvent::Finished(ev.clone()));
    }

    /// Nobody listens any more; the tracker drops this observer after the next event.
    fn is_closed(&self) -> bool {
        false
    }
}

#[derive(Debug)]
struct TrackerState {
    /// Between the first `capture_frame` of a segment and its `flush`.
    capturing: bool,
    /// Segments opened and finished over the tracker's life: the `n`th finish closes the `n`th
    /// segment, and a reset finishes every one outstanding.
    segments: u64,
    finished: u64,
    last: PlaybackFinished,
    /// Bumped by [`PlaybackTracker::reset`]; a waiter latches it so a reset releases it.
    epoch: u64,
    progressed: Duration,
    /// Audio captured into the open segment that no clear has dropped yet.
    uncleared: Duration,
    next_waiter: u64,
    waiters: BTreeMap<u64, Waker>,
}

impl TrackerState {
    fn open_progress(&self) -> Duration {
        if self.finished < self.segments { self.progressed } else { Duration::ZERO }
    }
}

/// The segment bookkeeping every [`AudioOutput`] shares, plus the fan-out to observers. Also the
/// observer a child registers with, so a terminal's event counts at every level.
pub struct PlaybackTracker {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    state: Mutex<TrackerState>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
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
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
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
                uncleared: Duration::ZERO,
                next_waiter: 0,
                waiters: BTreeMap::new(),
            }),
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

    /// `on_playback_progressed(started_at=, offset=, duration=)`. Invalid durations are refused
    /// before updating playback or notifying observers.
    #[host(method(name = "playback_progressed"))]
    pub fn report_progressed(
        &self,
        started_at: f64,
        offset: f64,
        duration: f64,
    ) -> std::result::Result<(), InvalidPlaybackTime> {
        let offset = seconds(offset, "offset")?;
        let duration = seconds(duration, "duration")?;
        self.playback_progressed(self.anchor().ts(started_at), offset, duration);
        Ok(())
    }

    /// `on_playback_finished(..)`. `False` when no segment was outstanding (nothing is emitted).
    #[host(method(name = "playback_finished"))]
    pub fn report_finished(
        &self,
        playback_position: f64,
        interrupted: bool,
        #[host(required)] synchronized_transcript: Option<String>,
    ) -> std::result::Result<bool, InvalidPlaybackTime> {
        let playback_position = seconds(playback_position, "playback_position")?;
        Ok(self.playback_finished(PlaybackFinished {
            playback_position,
            interrupted,
            synchronized_transcript,
        }))
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

    fn emit(&self, notify: impl Fn(&dyn PlaybackObserver)) {
        for o in self.observers() {
            notify(&*o);
        }
        // Whoever stopped listening is forgotten here, not at `aclose`.
        self.observers.lock().unwrap().retain(|o| !o.is_closed());
    }

    /// A synchroniser's release of `word`, fanned out like a playback event.
    pub fn transcript_shown(&self, word: &TimedString) {
        for o in self.observers() {
            o.on_transcript_shown(word);
        }
    }

    /// `AudioOutput.capture_frame`'s `super()` half: the first frame after a flush opens a segment.
    #[host(method)]
    pub fn note_capture(&self) {
        self.note_captured(Duration::ZERO);
    }

    /// [`note_capture`](Self::note_capture) of `duration` of audio.
    pub fn note_captured(&self, duration: Duration) {
        let mut s = self.state.lock().unwrap();
        if !s.capturing {
            s.capturing = true;
            s.segments += 1;
            // A new segment: what was heard of the previous one is not this one's position.
            s.progressed = Duration::ZERO;
            s.uncleared = Duration::ZERO;
        }
        s.uncleared += duration;
    }

    /// The audio captured into the current segment since it opened or since the last take: what
    /// a sink's clear still drops. Zero when a clear already dropped everything captured.
    pub fn take_uncleared(&self) -> Duration {
        std::mem::take(&mut self.state.lock().unwrap().uncleared)
    }

    /// `AudioOutput.flush`'s `super()` half: the segment is closed; its playout is still owed.
    #[host(method)]
    pub fn note_flush(&self) {
        self.state.lock().unwrap().capturing = false;
    }

    /// `_reset_playback_count`, plus: a parked `wait_for_playout` is released with
    /// `Finished { interrupted: true }` at what the open segment played (never the previous
    /// segment's finish), rather than left waiting forever.
    #[host(method)]
    pub fn reset(&self) {
        let mut s = self.state.lock().unwrap();
        let played = s.open_progress();
        s.last = PlaybackFinished {
            playback_position: played,
            interrupted: true,
            synchronized_transcript: None,
        };
        s.finished = s.segments;
        s.capturing = false;
        s.epoch += 1;
        s.progressed = Duration::ZERO;
        let waiters = std::mem::take(&mut s.waiters);
        drop(s);
        for waiter in waiters.into_values() {
            waiter.wake();
        }
        self.on_playback_reset();
    }

    /// How many segments were opened so far: the number of the one open or last closed.
    pub fn segments(&self) -> u64 {
        self.state.lock().unwrap().segments
    }

    /// The playhead per segment: what the sink reported played of the open segment, zero when
    /// none is open. Never the previous segment's finish (a cut speech would truncate at it).
    pub fn position(&self) -> Duration {
        self.state.lock().unwrap().open_progress()
    }

    pub fn playback_started(&self, created_at: Ts) {
        let ev = PlaybackEvent::Started(PlaybackStarted { created_at });
        self.emit(|o| o.on_playback_event(&ev));
    }

    pub fn playback_progressed(&self, started_at: Ts, offset: Duration, duration: Duration) {
        self.state.lock().unwrap().progressed += duration;
        let ev = PlaybackEvent::Progressed(PlaybackProgressed { started_at, offset, duration });
        self.emit(|o| o.on_playback_event(&ev));
    }

    /// A segment finished playing. With no outstanding segment it returns `false` and emits
    /// nothing: a spurious `finished` would release the next segment's wait.
    pub fn playback_finished(&self, ev: PlaybackFinished) -> bool {
        let (segment, waiters) = {
            let mut s = self.state.lock().unwrap();
            if s.finished >= s.segments {
                return false;
            }
            s.finished += 1;
            s.last = ev.clone();
            s.progressed = Duration::ZERO;
            (s.finished, std::mem::take(&mut s.waiters))
        };
        for waiter in waiters.into_values() {
            waiter.wake();
        }
        self.emit(|o| o.on_segment_finished(segment, &ev));
        true
    }

    /// `AudioOutput.wait_for_playout()`: [`wait_for_playout`](Self::wait_for_playout)'s event as
    /// `(playback_position, interrupted, synchronized_transcript)`.
    #[host(method(name = "wait_for_playout"))]
    pub async fn wait_for_playout_for_host(&self) -> (f64, bool, Option<String>) {
        let ev = self.wait_for_playout().await;
        (ev.playback_position.as_secs_f64(), ev.interrupted, ev.synchronized_transcript)
    }

    /// Wait for every segment captured so far; returns the last event. The target is latched at
    /// entry, as the reference does, or this never returns while the agent talks.
    pub async fn wait_for_playout(&self) -> PlaybackFinished {
        let (target, epoch) = {
            let s = self.state.lock().unwrap();
            (s.segments, s.epoch)
        };
        let mut waiting = PlaybackWait { tracker: self, id: None };
        std::future::poll_fn(|cx| {
            let mut s = self.state.lock().unwrap();
            if s.epoch != epoch || s.finished >= target {
                return Poll::Ready(s.last.clone());
            }
            let id = *waiting.id.get_or_insert_with(|| {
                s.next_waiter += 1;
                s.next_waiter
            });
            s.waiters.insert(id, cx.waker().clone());
            Poll::Pending
        })
        .await
    }
}

// Cancellation releases the captured task even if playback never completes.
struct PlaybackWait<'a> {
    tracker: &'a PlaybackTracker,
    id: Option<u64>,
}

impl Drop for PlaybackWait<'_> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            self.tracker.state.lock().unwrap().waiters.remove(&id);
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

    fn on_transcript_shown(&self, word: &TimedString) {
        self.transcript_shown(word);
    }

    /// A reset below is told above, not repeated: each node's count is its own. The segment this
    /// node had open is over, though, once the sink reported it cut: the next frame opens a
    /// segment of its own, and its wait waits for it.
    fn on_playback_reset(&self) {
        {
            let mut s = self.state.lock().unwrap();
            if s.finished == s.segments {
                s.capturing = false;
            }
        }
        for o in self.observers() {
            o.on_playback_reset();
        }
    }
}

/// Invalid host sink input, raised before any segment is consumed.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, thiserror::Error)]
#[error("{path}={value} is not a usable duration in seconds")]
pub struct InvalidPlaybackTime {
    path: &'static str,
    value: f64,
}

fn seconds(value: f64, path: &'static str) -> std::result::Result<Duration, InvalidPlaybackTime> {
    Duration::try_from_secs_f64(value).map_err(|_| InvalidPlaybackTime { path, value })
}

/// What a host wrapper over a core leg hears of its playback (`_ParticipantAudioOutput`, re-emitting
/// the leg's events as its own): `io.AudioOutput`'s `on_playback_*`, through its `_core_*` surface.
#[crate::host(proxy(
    host = "livekit.agents.voice.io.AudioOutput",
    view = "livekit.agents._core_view._AudioOutput"
))]
pub trait PlaybackListener: Send + Sync {
    #[host(notify, name = "_core_on_playback_started")]
    fn on_playback_started(&self, created_at: f64);
    #[host(notify, name = "_core_on_playback_progressed")]
    fn on_playback_progressed(&self, started_at: f64, offset: f64, duration: f64);
    #[host(notify, name = "_core_on_playback_finished")]
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
#[crate::host(proxy(
    host = "livekit.agents.voice.io.AudioInput",
    view = "livekit.agents._core_view._AudioInput"
))]
pub trait AudioInput: Send + Sync {
    #[host(read_once, name = "_core_label")]
    fn label(&self) -> &str;

    /// The next frame, or `None` at end of stream.
    #[host(await, name = "_core_next_frame")]
    fn next_frame(&self) -> crate::BoxFuture<'_, Result<Option<AudioFrame>>>;

    /// The chain was enabled; frames are dropped while detached, so a re-attach resumes mid-stream.
    #[host(notify, name = "_core_on_attached")]
    fn on_attached(&self) {}
    #[host(notify, name = "_core_on_detached")]
    fn on_detached(&self) {}
}

/// `io.VideoInput`: the source end of the video-in chain; one reader. Frames are side-table keys.
#[crate::host(proxy(
    host = "livekit.agents.voice.io.VideoInput",
    view = "livekit.agents._core_view._VideoInput"
))]
pub trait VideoInput: Send + Sync {
    #[host(read_once, name = "_core_label")]
    fn label(&self) -> &str;

    /// The next frame, or `None` at end of stream.
    #[host(await, name = "_core_next_frame")]
    fn next_frame(&self) -> crate::BoxFuture<'_, Result<Option<crate::voice::rtc::VideoFrameRef>>>;

    /// `set_video_enabled`: frames are dropped while detached.
    #[host(notify, name = "_core_on_attached")]
    fn on_attached(&self) {}
    #[host(notify, name = "_core_on_detached")]
    fn on_detached(&self) {}
}

/// `io.AudioOutput`: one link in the audio-out chain. Implement [`capture`](Self::capture) and
/// [`flush_segment`](Self::flush_segment); call [`AudioOutputExt::capture_frame`] and [`AudioOutputExt::flush`].
#[crate::host(proxy(
    host = "livekit.agents.voice.io.AudioOutput",
    view = "livekit.agents._core_view._AudioOutput"
))]
#[async_trait]
pub trait AudioOutput: Send + Sync {
    #[host(read_once, name = "_core_label")]
    fn label(&self) -> &str;

    /// This node's own bookkeeping and event fan-out. One per node, never shared.
    #[host(read_once, name = "_core_tracker")]
    fn tracker(&self) -> &Arc<PlaybackTracker>;

    #[host(read_once, name = "_core_capabilities")]
    fn capabilities(&self) -> AudioOutputCapabilities;

    /// The rate this sink requires, or `None` if it takes any.
    #[host(read_once, name = "_core_sample_rate")]
    fn sample_rate(&self) -> Option<u32> {
        None
    }

    fn next_in_chain(&self) -> Option<Arc<dyn AudioOutput>> {
        None
    }

    /// Push one frame downstream. Callers use [`AudioOutputExt::capture_frame`]. Awaited on a host
    /// sink, where backpressure lives.
    #[host(await, name = "_core_capture")]
    async fn capture(&self, frame: &AudioFrame) -> Result<()>;

    /// End the current segment. **Not the caller's entry point** -- use [`AudioOutputExt::flush`].
    #[host(notify, name = "_core_flush_segment")]
    fn flush_segment(&self);

    /// Throw away buffered audio and stop playback immediately. Synchronous: this is interruption.
    #[host(notify, name = "_core_clear_buffer")]
    fn clear_buffer(&self);

    #[host(notify, name = "_core_on_attached")]
    fn on_attached(&self) {
        if let Some(next) = self.next_in_chain() {
            next.on_attached();
        }
    }

    #[host(notify, name = "_core_on_detached")]
    fn on_detached(&self) {
        if let Some(next) = self.next_in_chain() {
            next.on_detached();
        }
    }

    #[host(notify, name = "_core_pause")]
    fn pause(&self) {
        if let Some(next) = self.next_in_chain() {
            next.pause();
        }
    }

    #[host(notify, name = "_core_resume")]
    fn resume(&self) {
        if let Some(next) = self.next_in_chain() {
            next.resume();
        }
    }
}

/// The caller-facing half of [`AudioOutput`]. Blanket-implemented; never implement it directly.
pub trait AudioOutputExt: AudioOutput {
    fn capture_frame<'a>(&'a self, frame: &'a AudioFrame) -> crate::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.tracker().note_captured(Duration::from_secs_f64(frame.duration()));
            self.capture(frame).await
        })
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

    fn wait_for_playout(&self) -> crate::BoxFuture<'_, PlaybackFinished> {
        Box::pin(async move { self.tracker().wait_for_playout().await })
    }

    /// Only if every link can pause, or wrapper and terminal desynchronise.
    fn can_pause(&self) -> bool {
        self.capabilities().pause && self.next_in_chain().map(|n| n.can_pause()).unwrap_or(true)
    }
}

impl<T: AudioOutput + ?Sized> AudioOutputExt for T {}

/// `io.TextOutput`: one link in the transcript chain; `flush` is the whole protocol.
#[crate::host(proxy(
    host = "livekit.agents.voice.io.TextOutput",
    view = "livekit.agents._core_view._TextOutput"
))]
#[async_trait]
pub trait TextOutput: Send + Sync {
    #[host(read_once, name = "_core_label")]
    fn label(&self) -> &str;

    fn next_in_chain(&self) -> Option<Arc<dyn TextOutput>> {
        None
    }

    /// Capture a delta with word timings (Python's `TimedString` is a `str`; here it's a separate
    /// method whose default drops the timings).
    #[host(await, name = "_core_capture_timed")]
    async fn capture_timed(&self, text: &TimedString) -> Result<()> {
        self.capture_text(&text.text).await
    }

    /// Capture a text delta. Must not drop: a lost token mangles the transcript and the speech.
    #[host(await, name = "_core_capture_text")]
    async fn capture_text(&self, text: &str) -> Result<()>;

    /// The generation is complete; close the segment.
    #[host(notify, name = "_core_flush")]
    fn flush(&self);

    #[host(notify, name = "_core_on_attached")]
    fn on_attached(&self) {
        if let Some(next) = self.next_in_chain() {
            next.on_attached();
        }
    }

    #[host(notify, name = "_core_on_detached")]
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

#[cfg(test)]
mod playback_wait_tests {
    use super::*;
    use std::future::Future;
    use std::task::{Context, Wake, Waker};

    struct Observer(Arc<OnceLock<()>>);

    impl Wake for Observer {
        fn wake(self: Arc<Self>) {
            let _ = self.0.set(());
        }
    }

    #[test]
    fn cancellation_releases_a_playback_waiter_and_finish_releases_every_remaining_waiter() {
        let tracker = PlaybackTracker::new();
        tracker.note_capture();
        let task = Arc::new(OnceLock::new());
        let waker = Waker::from(Arc::new(Observer(task.clone())));
        let mut cx = Context::from_waker(&waker);
        let mut cancelled = Box::pin(tracker.wait_for_playout());
        assert!(cancelled.as_mut().poll(&mut cx).is_pending());
        drop(cancelled);
        drop(waker);
        assert_eq!(Arc::strong_count(&task), 1, "a cancelled wait must release its task");

        let mut first = Box::pin(tracker.wait_for_playout());
        let mut second = Box::pin(tracker.wait_for_playout());
        let first_woken = Arc::new(OnceLock::new());
        let second_woken = Arc::new(OnceLock::new());
        let first_waker = Waker::from(Arc::new(Observer(first_woken.clone())));
        let second_waker = Waker::from(Arc::new(Observer(second_woken.clone())));
        assert!(first.as_mut().poll(&mut Context::from_waker(&first_waker)).is_pending());
        assert!(second.as_mut().poll(&mut Context::from_waker(&second_waker)).is_pending());
        let finished =
            PlaybackFinished { playback_position: Duration::from_millis(20), ..Default::default() };
        assert!(tracker.playback_finished(finished.clone()));
        assert!(first_woken.get().is_some());
        assert!(second_woken.get().is_some());
        let mut cx = Context::from_waker(Waker::noop());
        assert_eq!(first.as_mut().poll(&mut cx), Poll::Ready(finished.clone()));
        assert_eq!(second.as_mut().poll(&mut cx), Poll::Ready(finished));
    }

    #[test]
    fn reset_releases_a_waiter_with_the_open_segments_played_position() {
        let tracker = PlaybackTracker::new();
        tracker.note_capture();
        tracker.playback_progressed(Ts::ZERO, Duration::ZERO, Duration::from_millis(20));
        let mut waiting = Box::pin(tracker.wait_for_playout());
        let mut cx = Context::from_waker(Waker::noop());
        assert!(waiting.as_mut().poll(&mut cx).is_pending());
        tracker.reset();
        let Poll::Ready(finished) = waiting.as_mut().poll(&mut cx) else {
            panic!("reset must release a parked waiter");
        };
        assert!(finished.interrupted);
        assert_eq!(finished.playback_position, Duration::from_millis(20));
    }
}
