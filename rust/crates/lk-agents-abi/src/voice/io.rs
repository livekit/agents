//! A host `io.AudioInput` / `VideoInput` / `AudioOutput` / `TextOutput` is its trait through the
//! generated proxy, which calls the io class's own methods and reads its properties. A host sink
//! reports its playback into the [`PlaybackTracker`] its `_tracker` holds, stamped on the
//! session's clock by [`PlaybackTracker::anchor_to`].

use crate::voice::rtc::TrackSid;
use crate::{AudioFrame, TranscriptChunk, Ts, WallAnchor};
use std::sync::Arc;
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
    /// of the played position of `segment`, the playback that released it; `None` when no
    /// playback did (words with no audio). Ignored by default.
    fn on_transcript_shown(&self, segment: Option<PlaybackSegment>, word: &TranscriptChunk) {
        let _ = (segment, word);
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

/// `_core.PlaybackTracker` as the core uses it: one output node's segment bookkeeping and the
/// fan-out of its playback events. One implementation, the runtime's; ABI code reaches it only
/// through this trait.
#[crate::host(native(refused = "expected a `_core.PlaybackTracker`"))]
pub trait PlaybackTracker: PlaybackObserver + Send + Sync {
    /// Stamp a host sink's reports on the session's clock. The first anchor wins: one host sink
    /// reports to one session. A host sink reporting before `set_audio_output` anchors it is
    /// stamped on the real wall clock. Runtime-built outputs report session-clock stamps and need
    /// no anchor.
    fn anchor_to(&self, anchor: WallAnchor);

    /// The anchor a host sink's `time.time()` stamps are read against.
    fn anchor(&self) -> WallAnchor;

    /// Select the supplied correlation for the next captured segment. Already admitted or
    /// still-capturing work cannot be replaced by a second begin.
    fn begin_segment(&self, segment: PlaybackSegment) -> bool;

    fn has_pending_segment(&self, segment: PlaybackSegment) -> bool;

    /// Only the oldest segment owing a report may report.
    fn accepts_report(&self, segment: PlaybackSegment) -> bool;

    /// The key to latch before starting an external operation; never look it up on completion.
    fn current_segment(&self) -> Option<PlaybackSegment>;

    /// `AudioOutput.capture_frame`'s `super()` half for `duration` of audio: the first frame after
    /// a flush opens a segment.
    fn note_captured(&self, duration: Duration) -> PlaybackSegment;

    /// The audio captured into the current segment since it opened or since the last take: what
    /// a sink's clear still drops.
    fn take_uncleared(&self) -> Duration;

    /// `AudioOutput.flush`'s `super()` half: the segment is closed; its playout is still owed.
    fn note_flush(&self);

    /// `_reset_playback_count`: a parked `wait_for_playout` is released as interrupted at what
    /// the open segment played.
    fn reset(&self);

    /// What the sink reported played of the open segment, zero when none is open.
    fn position(&self) -> Duration;

    fn playback_started(&self, segment: PlaybackSegment, created_at: Ts) -> bool;

    fn playback_progressed(
        &self,
        segment: PlaybackSegment,
        started_at: Ts,
        offset: Duration,
        duration: Duration,
    ) -> bool;

    /// False for a stale or out-of-order key; nothing changes and nothing is emitted.
    fn playback_finished(&self, segment: PlaybackSegment, ev: PlaybackFinished) -> bool;

    /// A synchroniser's release of `word` by `segment`'s playback, fanned out like a playback
    /// event.
    fn transcript_shown(&self, segment: Option<PlaybackSegment>, word: &TranscriptChunk);

    /// Forward this node's events to `observer` (`next.on("playback_finished", ...)`). Pass the
    /// parent's tracker, not the node, so the chain isn't a reference cycle.
    fn observe(&self, observer: Arc<dyn PlaybackObserver>);

    /// Stop forwarding to `observer` (`next.off(...)`), by pointer.
    fn unobserve(&self, observer: &Arc<dyn PlaybackObserver>);

    /// Wait for every segment captured so far; returns the last event. Cancellation releases it.
    fn wait_for_playout(&self) -> crate::BoxFuture<'_, PlaybackFinished>;
}

/// `_core.PlaybackTracker`: the tracker a host sink reports to (`io.AudioOutput._tracker`),
/// shared with the session that reads it.
#[crate::host(class(name = "PlaybackTracker", native = PlaybackTracker))]
pub trait Playback: PlaybackTracker + Sized + 'static {
    #[host(new)]
    fn new() -> Self;
    #[host(call)]
    fn note_flush(&self) {
        PlaybackTracker::note_flush(self)
    }
    #[host(call)]
    fn reset(&self) {
        PlaybackTracker::reset(self)
    }
    /// `AudioOutput.wait_for_playout()`: the last event as `(playback_position, interrupted,
    /// synchronized_transcript)`.
    #[host(await, name = "wait_for_playout")]
    async fn wait_for_played(&self) -> (f64, bool, Option<String>) {
        let ev = PlaybackTracker::wait_for_playout(self).await;
        (ev.playback_position.as_secs_f64(), ev.interrupted, ev.synchronized_transcript)
    }
}

/// The keyed report boundary, in the host sink's `time.time()` stamps and float seconds.
#[crate::host(functions)]
pub trait PlaybackReports {
    #[host(function, name = "_core_playback_begin")]
    fn begin(tracker: Arc<dyn PlaybackTracker>, segment: PlaybackSegment) -> bool {
        tracker.begin_segment(segment)
    }
    #[host(function, name = "_core_playback_current_segment")]
    fn current_segment(tracker: Arc<dyn PlaybackTracker>) -> Option<PlaybackSegment> {
        tracker.current_segment()
    }
    #[host(function, name = "_core_playback_note_capture")]
    fn note_capture(tracker: Arc<dyn PlaybackTracker>) -> PlaybackSegment {
        tracker.note_captured(Duration::ZERO)
    }
    #[host(function, name = "_core_playback_started")]
    fn started(
        tracker: Arc<dyn PlaybackTracker>,
        segment: PlaybackSegment,
        created_at: f64,
    ) -> bool {
        tracker.playback_started(segment, tracker.anchor().ts(created_at))
    }
    /// Invalid durations are refused before updating playback or notifying observers.
    #[host(function, name = "_core_playback_progressed")]
    fn progressed(
        tracker: Arc<dyn PlaybackTracker>,
        segment: PlaybackSegment,
        started_at: f64,
        offset: f64,
        duration: f64,
    ) -> std::result::Result<bool, InvalidPlaybackTime> {
        let offset = seconds(offset, "offset")?;
        let duration = seconds(duration, "duration")?;
        let started_at = tracker.anchor().ts(started_at);
        Ok(tracker.playback_progressed(segment, started_at, offset, duration))
    }
    #[host(function, name = "_core_playback_finished")]
    fn finished(
        tracker: Arc<dyn PlaybackTracker>,
        segment: PlaybackSegment,
        playback_position: f64,
        interrupted: bool,
        synchronized_transcript: Option<String>,
    ) -> std::result::Result<bool, InvalidPlaybackTime> {
        let playback_position = seconds(playback_position, "playback_position")?;
        Ok(tracker.playback_finished(
            segment,
            PlaybackFinished { playback_position, interrupted, synchronized_transcript },
        ))
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
/// the leg's events as its own): `io.AudioOutput`'s `on_playback_*`, called by keyword.
#[crate::host(proxy(host = "livekit.agents.voice.io.AudioOutput"))]
pub trait PlaybackListener: Send + Sync {
    #[host(notify)]
    fn on_playback_started(&self, #[host(kw)] segment: PlaybackSegment, created_at: f64);
    #[host(notify)]
    fn on_playback_progressed(
        &self,
        #[host(kw)] segment: PlaybackSegment,
        started_at: f64,
        offset: f64,
        duration: f64,
    );
    #[host(notify)]
    fn on_playback_finished(
        &self,
        #[host(kw)] segment: PlaybackSegment,
        playback_position: f64,
        interrupted: bool,
        synchronized_transcript: Option<String>,
    );
}

/// A frame read from an [`AudioInput`].
#[derive(Debug, Clone)]
pub struct InputFrame {
    pub frame: AudioFrame,
    /// Pre-connect audio: every consumer hears it in full, however far behind it falls.
    pub head: bool,
}

/// `io.AudioInput`: the source end of the audio-in chain. Exactly one reader: two tasks pulling
/// one host-backed async iterator lose data.
#[crate::host(proxy(host = "livekit.agents.voice.io.AudioInput"))]
pub trait AudioInput: Send + Sync {
    #[host(read_once, getter = "label")]
    fn label(&self) -> &str;

    /// The next frame, or `None` at end of stream (its `__anext__` raised `StopAsyncIteration`).
    #[host(await, function = "livekit.agents._bindings.next_audio_frame")]
    fn next_frame(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Option<AudioFrame>>>;

    /// [`next_frame`](Self::next_frame), marking the pre-connect audio a room input replays
    /// ahead of its live frames.
    fn next_input(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Option<InputFrame>>> {
        let next = self.next_frame(cx);
        Box::pin(async move { Ok(next.await?.map(|frame| InputFrame { frame, head: false })) })
    }

    /// The chain was enabled; frames are dropped while detached, so a re-attach resumes mid-stream.
    #[host(notify)]
    fn on_attached(&self) {}
    #[host(notify)]
    fn on_detached(&self) {}
}

/// `io.VideoInput`: the source end of the video-in chain; one reader. Frames are side-table keys.
#[crate::host(proxy(host = "livekit.agents.voice.io.VideoInput"))]
pub trait VideoInput: Send + Sync {
    #[host(read_once, getter = "label")]
    fn label(&self) -> &str;

    /// The next frame as its side-table key, or `None` at end of stream.
    #[host(await, function = "livekit.agents._bindings.next_video_frame")]
    fn next_frame(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Option<crate::voice::rtc::VideoFrameRef>>>;

    /// `set_video_enabled`: frames are dropped while detached.
    #[host(notify)]
    fn on_attached(&self) {}
    #[host(notify)]
    fn on_detached(&self) {}
}

/// `io.AudioOutput`: one link in the audio-out chain. Implement [`capture`](Self::capture) and
/// [`flush_segment`](Self::flush_segment); call [`AudioOutputExt::capture_frame`] and [`AudioOutputExt::flush`].
#[crate::host(proxy(host = "livekit.agents.voice.io.AudioOutput"))]
pub trait AudioOutput: Send + Sync {
    #[host(read_once, getter = "label")]
    fn label(&self) -> &str;

    /// This node's own bookkeeping and event fan-out. One per node, never shared.
    #[host(read_once, getter = "_tracker")]
    fn tracker(&self) -> &Arc<dyn PlaybackTracker>;

    /// A host output's `can_pause`, its whole chain's.
    #[host(read_once, function = "livekit.agents._bindings.audio_output_capabilities")]
    fn capabilities(&self) -> AudioOutputCapabilities;

    /// Admit the key before capturing; wrappers forward that same key to their leaves.
    #[host(call)]
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
    #[host(read_once, getter = "sample_rate")]
    fn sample_rate(&self) -> Option<u32> {
        None
    }

    fn next_in_chain(&self) -> Option<Arc<dyn AudioOutput>> {
        None
    }

    /// Push one frame downstream. Callers use [`AudioOutputExt::capture_frame`]. Awaited on a host
    /// sink, where backpressure lives.
    #[host(await, name = "capture_frame")]
    fn capture<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        frame: &'a AudioFrame,
    ) -> crate::BoxFuture<'a, Result<()>>;

    /// End the current segment. **Not the caller's entry point** -- use [`AudioOutputExt::flush`].
    #[host(notify, name = "flush")]
    fn flush_segment(&self);

    /// Throw away buffered audio and stop playback immediately. Synchronous: this is interruption.
    #[host(notify)]
    fn clear_buffer(&self);

    #[host(notify)]
    fn on_attached(&self) {
        if let Some(next) = self.next_in_chain() {
            next.on_attached();
        }
    }

    #[host(notify)]
    fn on_detached(&self) {
        if let Some(next) = self.next_in_chain() {
            next.on_detached();
        }
    }

    #[host(notify)]
    fn pause(&self) {
        if let Some(next) = self.next_in_chain() {
            next.pause();
        }
    }

    #[host(notify)]
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
            self.capture(&crate::telemetry::Cx::NONE, frame).await
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
#[crate::host(proxy(host = "livekit.agents.voice.io.TextOutput"))]
pub trait TextOutput: Send + Sync {
    #[host(read_once, getter = "label")]
    fn label(&self) -> &str;

    fn next_in_chain(&self) -> Option<Arc<dyn TextOutput>> {
        None
    }

    /// Capture text with alignment and speaker metadata. Plain-text outputs can use the default,
    /// which forwards only the explicit text field.
    #[host(await, name = "capture_transcript")]
    fn capture_timed<'a>(
        &'a self,
        _cx: &crate::telemetry::Cx,
        text: &'a TranscriptChunk,
    ) -> crate::BoxFuture<'a, Result<()>> {
        Box::pin(async move { self.capture_text(&crate::telemetry::Cx::NONE, &text.text).await })
    }

    /// Capture a text delta. Must not drop: a lost token mangles the transcript and the speech.
    #[host(await)]
    fn capture_text<'a>(
        &'a self,
        cx: &crate::telemetry::Cx,
        text: &'a str,
    ) -> crate::BoxFuture<'a, Result<()>>;

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
    #[host(notify)]
    fn flush(&self);

    #[host(notify)]
    fn on_attached(&self) {
        if let Some(next) = self.next_in_chain() {
            next.on_attached();
        }
    }

    #[host(notify)]
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
    #[host(delegate, capture)]
    Host(#[from] crate::errors::HostError),
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

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(module = "playback_events", path = "livekit.agents.voice.io", dataclass))]
pub struct PlaybackFinishedEvent {
    /// How much of the audio was played back
    pub playback_position: f64,
    /// Interrupted is True if playback was interrupted (clear_buffer() was called)
    pub interrupted: bool,
    /// Transcript synced with playback; may be partial if the audio was interrupted
    #[host(default = None)]
    pub synchronized_transcript: Option<String>,
    /// The captured segment key on emitted playback observations.
    #[host(default = None)]
    pub segment: Option<PlaybackSegment>,
}

#[crate::host(record(module = "playback_events", path = "livekit.agents.voice.io", dataclass))]
pub struct PlaybackStartedEvent {
    pub created_at: f64,
    /// The timestamp (time.time()) when the playback started
    pub segment: PlaybackSegment,
}

/// A stretch of the current segment that has played; never revised.
#[crate::host(record(module = "playback_events", path = "livekit.agents.voice.io", dataclass))]
pub struct PlaybackProgressedEvent {
    pub started_at: f64,
    pub offset: f64,
    pub duration: f64,
    pub segment: PlaybackSegment,
}
