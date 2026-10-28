//! A host `io.AudioInput` / `VideoInput` / `AudioOutput` / `TextOutput` is its trait through the
//! generated proxy, which calls the host object's `_core_view` view. A host sink reports its playback into the [`PlaybackTracker`] its
//! `_core_tracker()` hands over, stamped on the session's clock by [`PlaybackTracker::anchor_to`].

use crate::voice::rtc::TrackSid;
use crate::{AudioFrame, TimedString, Ts, WallAnchor};
use std::collections::{BTreeMap, VecDeque};
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Poll, Waker};
use std::time::Duration;

pub use crate::voice::rtc::AudioOutputCapabilities;

/// `PlaybackFinishedEvent` (the ABI's).
pub use crate::voice::room_io::PlaybackFinished;

/// Correlation supplied at segment admission and retained by every transport operation.
/// A delayed report from an earlier attachment must not settle its replacement.
#[crate::host(record(module = "io_events", extract, emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PlaybackSegment {
    pub attachment: u64,
    pub segment: u64,
}

/// An avatar receiver retains the source key and, for RTC, its existing byte-stream id.
#[crate::host(record(module = "io_events", extract, emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvatarPlaybackSegment {
    pub playback: PlaybackSegment,
    pub stream_id: Option<String>,
}

/// `PlaybackStartedEvent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaybackStarted {
    pub segment: PlaybackSegment,
    pub created_at: Ts,
}

/// `PlaybackProgressedEvent`: a stretch of the current segment that played, never revised
/// (`RecorderIO` needs when it played).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaybackProgressed {
    pub segment: PlaybackSegment,
    pub started_at: Ts,
    /// Where it starts within the audio captured for the current segment.
    pub offset: Duration,
    pub duration: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaybackEvent {
    Started(PlaybackStarted),
    Progressed(PlaybackProgressed),
    Finished { segment: PlaybackSegment, event: PlaybackFinished },
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

    /// The admitted segment finished. The key survives wrapper forwarding and delayed reports.
    fn on_segment_finished(&self, segment: PlaybackSegment, ev: &PlaybackFinished) {
        self.on_playback_event(&PlaybackEvent::Finished { segment, event: ev.clone() });
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
    selected: Option<PlaybackSegment>,
    pending: VecDeque<PlaybackSegment>,
    highest_segment: u64,
    /// Counts for waits latched at entry; keyed reports additionally enforce FIFO completion.
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
                selected: None,
                pending: VecDeque::new(),
                highest_segment: 0,
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
    pub fn report_started(&self, segment: PlaybackSegment, created_at: f64) -> bool {
        self.playback_started(segment, self.anchor().ts(created_at))
    }

    /// `on_playback_progressed(started_at=, offset=, duration=)`. Invalid durations are refused
    /// before updating playback or notifying observers.
    pub fn report_progressed(
        &self,
        segment: PlaybackSegment,
        started_at: f64,
        offset: f64,
        duration: f64,
    ) -> std::result::Result<bool, InvalidPlaybackTime> {
        let offset = seconds(offset, "offset")?;
        let duration = seconds(duration, "duration")?;
        Ok(self.playback_progressed(segment, self.anchor().ts(started_at), offset, duration))
    }

    /// `on_playback_finished(..)`. False for a stale or out-of-order key; nothing is emitted.
    pub fn report_finished(
        &self,
        segment: PlaybackSegment,
        playback_position: f64,
        interrupted: bool,
        synchronized_transcript: Option<String>,
    ) -> std::result::Result<bool, InvalidPlaybackTime> {
        let playback_position = seconds(playback_position, "playback_position")?;
        Ok(self.playback_finished(
            segment,
            PlaybackFinished { playback_position, interrupted, synchronized_transcript },
        ))
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

    /// Select the supplied correlation for the next captured segment. Already admitted or
    /// still-capturing work cannot be replaced by a second begin.
    pub fn begin_segment(&self, segment: PlaybackSegment) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.selected == Some(segment) {
            return true;
        }
        if state.capturing || segment.segment <= state.highest_segment {
            return false;
        }
        state.highest_segment = segment.segment;
        state.selected = Some(segment);
        true
    }

    pub fn has_pending_segment(&self, segment: PlaybackSegment) -> bool {
        self.state.lock().unwrap().pending.contains(&segment)
    }

    pub fn accepts_report(&self, segment: PlaybackSegment) -> bool {
        self.state.lock().unwrap().pending.front() == Some(&segment)
    }

    /// The key to latch before starting an external operation; never look it up on completion.
    pub fn current_segment(&self) -> Option<PlaybackSegment> {
        self.state.lock().unwrap().selected
    }

    /// `AudioOutput.capture_frame`'s `super()` half: the first frame after a flush opens a segment.
    pub fn note_capture(&self) -> PlaybackSegment {
        self.note_captured(Duration::ZERO)
    }

    /// [`note_capture`](Self::note_capture) of `duration` of audio.
    pub fn note_captured(&self, duration: Duration) -> PlaybackSegment {
        let mut s = self.state.lock().unwrap();
        if !s.capturing {
            s.capturing = true;
            s.segments += 1;
            let segment = s.selected.unwrap_or(PlaybackSegment {
                attachment: 0,
                segment: s.highest_segment.saturating_add(1),
            });
            s.highest_segment = s.highest_segment.max(segment.segment);
            s.selected = Some(segment);
            s.pending.push_back(segment);
            // A new segment: what was heard of the previous one is not this one's position.
            s.progressed = Duration::ZERO;
            s.uncleared = Duration::ZERO;
        }
        s.uncleared += duration;
        s.selected.expect("capture opens a segment")
    }

    /// The audio captured into the current segment since it opened or since the last take: what
    /// a sink's clear still drops. Zero when a clear already dropped everything captured.
    pub fn take_uncleared(&self) -> Duration {
        std::mem::take(&mut self.state.lock().unwrap().uncleared)
    }

    /// `AudioOutput.flush`'s `super()` half: the segment is closed; its playout is still owed.
    #[host(method)]
    pub fn note_flush(&self) {
        let mut state = self.state.lock().unwrap();
        state.capturing = false;
        state.selected = None;
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
        s.selected = None;
        s.pending.clear();
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

    pub fn playback_started(&self, segment: PlaybackSegment, created_at: Ts) -> bool {
        if self.state.lock().unwrap().pending.front() != Some(&segment) {
            return false;
        }
        let ev = PlaybackEvent::Started(PlaybackStarted { segment, created_at });
        self.emit(|o| o.on_playback_event(&ev));
        true
    }

    pub fn playback_progressed(
        &self,
        segment: PlaybackSegment,
        started_at: Ts,
        offset: Duration,
        duration: Duration,
    ) -> bool {
        {
            let mut state = self.state.lock().unwrap();
            if state.pending.front() != Some(&segment) {
                return false;
            }
            state.progressed += duration;
        }
        let ev =
            PlaybackEvent::Progressed(PlaybackProgressed { segment, started_at, offset, duration });
        self.emit(|o| o.on_playback_event(&ev));
        true
    }

    /// Only the admitted oldest segment can finish; duplicate, reset and replaced attachment
    /// reports are refused before counters, waiters or observers change.
    pub fn playback_finished(&self, segment: PlaybackSegment, ev: PlaybackFinished) -> bool {
        let waiters = {
            let mut state = self.state.lock().unwrap();
            if state.pending.front() != Some(&segment) {
                return false;
            }
            state.pending.pop_front();
            state.finished += 1;
            state.last = ev.clone();
            state.progressed = Duration::ZERO;
            std::mem::take(&mut state.waiters)
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

/// The keyed report boundary. Its generated functions share the tracker used by every output.
#[crate::host(functions)]
pub trait PlaybackReports: Send + Sync + Sized + 'static {
    #[host(function, name = "_core_playback_begin")]
    fn begin(tracker: Arc<PlaybackTracker>, segment: PlaybackSegment) -> bool;
    #[host(function, name = "_core_playback_current_segment")]
    fn current_segment(tracker: Arc<PlaybackTracker>) -> Option<PlaybackSegment>;
    #[host(function, name = "_core_playback_note_capture")]
    fn note_capture(tracker: Arc<PlaybackTracker>) -> PlaybackSegment;
    #[host(function, name = "_core_playback_started")]
    fn started(tracker: Arc<PlaybackTracker>, segment: PlaybackSegment, created_at: f64) -> bool;
    #[host(function, name = "_core_playback_progressed")]
    fn progressed(
        tracker: Arc<PlaybackTracker>,
        segment: PlaybackSegment,
        started_at: f64,
        offset: f64,
        duration: f64,
    ) -> std::result::Result<bool, InvalidPlaybackTime>;
    #[host(function, name = "_core_playback_finished")]
    fn finished(
        tracker: Arc<PlaybackTracker>,
        segment: PlaybackSegment,
        playback_position: f64,
        interrupted: bool,
        synchronized_transcript: Option<String>,
    ) -> std::result::Result<bool, InvalidPlaybackTime>;
}

impl PlaybackReports for PlaybackTracker {
    fn begin(tracker: Arc<PlaybackTracker>, segment: PlaybackSegment) -> bool {
        tracker.begin_segment(segment)
    }
    fn current_segment(tracker: Arc<PlaybackTracker>) -> Option<PlaybackSegment> {
        tracker.current_segment()
    }
    fn note_capture(tracker: Arc<PlaybackTracker>) -> PlaybackSegment {
        tracker.note_capture()
    }
    fn started(tracker: Arc<PlaybackTracker>, segment: PlaybackSegment, created_at: f64) -> bool {
        tracker.report_started(segment, created_at)
    }
    fn progressed(
        tracker: Arc<PlaybackTracker>,
        segment: PlaybackSegment,
        started_at: f64,
        offset: f64,
        duration: f64,
    ) -> std::result::Result<bool, InvalidPlaybackTime> {
        tracker.report_progressed(segment, started_at, offset, duration)
    }
    fn finished(
        tracker: Arc<PlaybackTracker>,
        segment: PlaybackSegment,
        playback_position: f64,
        interrupted: bool,
        synchronized_transcript: Option<String>,
    ) -> std::result::Result<bool, InvalidPlaybackTime> {
        tracker.report_finished(segment, playback_position, interrupted, synchronized_transcript)
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
            PlaybackEvent::Started(e) => {
                self.playback_started(e.segment, e.created_at);
            }
            PlaybackEvent::Progressed(e) => {
                self.playback_progressed(e.segment, e.started_at, e.offset, e.duration);
            }
            PlaybackEvent::Finished { segment, event } => {
                self.playback_finished(*segment, event.clone());
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
                s.selected = None;
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
    fn on_playback_started(&self, segment: PlaybackSegment, created_at: f64);
    #[host(notify, name = "_core_on_playback_progressed")]
    fn on_playback_progressed(
        &self,
        segment: PlaybackSegment,
        started_at: f64,
        offset: f64,
        duration: f64,
    );
    #[host(notify, name = "_core_on_playback_finished")]
    fn on_playback_finished(
        &self,
        segment: PlaybackSegment,
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
                self.listener.on_playback_started(e.segment, self.anchor.wall(e.created_at))
            }
            PlaybackEvent::Progressed(e) => self.listener.on_playback_progressed(
                e.segment,
                self.anchor.wall(e.started_at),
                e.offset.as_secs_f64(),
                e.duration.as_secs_f64(),
            ),
            PlaybackEvent::Finished { segment, event } => self.listener.on_playback_finished(
                *segment,
                event.playback_position.as_secs_f64(),
                event.interrupted,
                event.synchronized_transcript.clone(),
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
pub trait AudioOutput: Send + Sync {
    #[host(read_once, name = "_core_label")]
    fn label(&self) -> &str;

    /// This node's own bookkeeping and event fan-out. One per node, never shared.
    #[host(read_once, name = "_core_tracker")]
    fn tracker(&self) -> &Arc<PlaybackTracker>;

    #[host(read_once, name = "_core_capabilities")]
    fn capabilities(&self) -> AudioOutputCapabilities;

    /// Admit the key before capturing; wrappers forward that same key to their leaves.
    #[host(call, name = "_core_begin_segment")]
    fn begin_segment(&self, segment: PlaybackSegment) -> Result<()> {
        if !self.tracker().begin_segment(segment) {
            return Err(IoError::PlaybackSegmentUnavailable);
        }
        if let Some(next) = self.next_in_chain() {
            next.begin_segment(segment)?;
        }
        Ok(())
    }

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
    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> crate::BoxFuture<'a, Result<()>>;

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
pub trait TextOutput: Send + Sync {
    #[host(read_once, name = "_core_label")]
    fn label(&self) -> &str;

    fn next_in_chain(&self) -> Option<Arc<dyn TextOutput>> {
        None
    }

    /// Capture a delta with word timings (Python's `TimedString` is a `str`; here it's a separate
    /// method whose default drops the timings).
    #[host(await, name = "_core_capture_timed")]
    fn capture_timed<'a>(&'a self, text: &'a TimedString) -> crate::BoxFuture<'a, Result<()>> {
        Box::pin(async move { self.capture_text(&text.text).await })
    }

    /// Capture a text delta. Must not drop: a lost token mangles the transcript and the speech.
    #[host(await, name = "_core_capture_text")]
    fn capture_text<'a>(&'a self, text: &'a str) -> crate::BoxFuture<'a, Result<()>>;

    /// Keep an admitted binding alive until this segment's queued processing has settled.
    fn retain_until_flushed(&self, owner: Box<dyn Send>) {
        if let Some(next) = self.next_in_chain() {
            next.retain_until_flushed(owner);
        }
    }

    /// Cancel processing belonging to the flushed segment.
    fn cancel_flush(&self) {
        if let Some(next) = self.next_in_chain() {
            next.cancel_flush();
        }
    }

    /// Wait for processing queued before the segment's flush to settle.
    fn wait_for_flush(&self) -> crate::BoxFuture<'_, Result<()>> {
        Box::pin(async { Ok(()) })
    }

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
    #[error("the playback key was reused or another output segment is still being captured")]
    PlaybackSegmentUnavailable,

    /// A bounded transcription writer refused retained work.
    #[error("transcription {0} capacity exceeded")]
    TranscriptionCapacity(String),

    /// A room failure reported by the RTC capability.
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

impl From<crate::errors::ProviderError> for IoError {
    fn from(error: crate::errors::ProviderError) -> Self {
        match error {
            crate::errors::ProviderError::Host { error, .. } => Self::Host(error),
            crate::errors::ProviderError::Local(error) => Self::Rtc(error.to_string()),
        }
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
        let segment = tracker.note_capture();
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
        assert!(tracker.playback_finished(segment, finished.clone()));
        assert!(first_woken.get().is_some());
        assert!(second_woken.get().is_some());
        let mut cx = Context::from_waker(Waker::noop());
        assert_eq!(first.as_mut().poll(&mut cx), Poll::Ready(finished.clone()));
        assert_eq!(second.as_mut().poll(&mut cx), Poll::Ready(finished));
    }

    #[test]
    fn reset_releases_a_waiter_with_the_open_segments_played_position() {
        let tracker = PlaybackTracker::new();
        let segment = tracker.note_capture();
        tracker.playback_progressed(segment, Ts::ZERO, Duration::ZERO, Duration::from_millis(20));
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

#[cfg(test)]
mod playback_identity_regression {
    use super::{
        PlaybackEvent, PlaybackFinished, PlaybackObserver, PlaybackSegment, PlaybackTracker,
    };
    use lk_agents_test_support::playback::PlaybackReports;
    use std::sync::Arc;
    use std::time::Duration;

    #[derive(Default)]
    struct Reports(PlaybackReports<PlaybackEvent>);

    impl PlaybackObserver for Reports {
        fn on_playback_event(&self, event: &PlaybackEvent) {
            self.0.record(event.clone());
        }
    }

    #[test]
    fn a_delayed_old_finish_cannot_complete_a_new_segment_on_the_same_output() {
        let tracker = PlaybackTracker::new();
        let reports = Arc::new(Reports::default());
        tracker.observe(reports.clone());
        let old = PlaybackSegment { attachment: 4, segment: 1 };
        let new = PlaybackSegment { attachment: 5, segment: 2 };
        let finished =
            PlaybackFinished { playback_position: Duration::from_millis(20), ..Default::default() };
        assert!(tracker.begin_segment(old));
        tracker.note_captured(Duration::from_millis(20));
        tracker.note_flush();
        assert!(tracker.playback_finished(old, finished.clone()));
        assert!(!tracker.begin_segment(old), "a completed key cannot be reused");
        assert!(tracker.begin_segment(new));
        tracker.note_captured(Duration::from_millis(20));
        tracker.note_flush();
        assert!(!tracker.playback_finished(old, finished.clone()));
        assert!(!tracker.playback_started(old, crate::Ts::ZERO));
        assert!(!tracker.playback_progressed(
            old,
            crate::Ts::ZERO,
            Duration::ZERO,
            Duration::from_millis(20)
        ));
        assert_eq!(reports.0.event_count(), 1);
        assert!(tracker.playback_finished(new, finished));
        assert_eq!(reports.0.event_count(), 2);
    }

    #[test]
    fn reset_rejects_reports_from_the_previous_attachment_without_advancing_new_work() {
        let tracker = PlaybackTracker::new();
        let old = PlaybackSegment { attachment: 4, segment: 1 };
        let new = PlaybackSegment { attachment: 5, segment: 2 };
        tracker.begin_segment(old);
        tracker.note_capture();
        tracker.reset();
        tracker.begin_segment(new);
        tracker.note_capture();
        let finished = PlaybackFinished::default();
        assert!(!tracker.playback_finished(old, finished.clone()));
        assert!(tracker.playback_finished(new, finished));
    }
}
