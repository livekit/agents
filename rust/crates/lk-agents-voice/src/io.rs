//! A host `io.AudioInput` / `VideoInput` / `AudioOutput` / `TextOutput` is its trait through the
//! trait's bare `#[host(proxy)]`. A host sink reports its playback into the [`PlaybackTracker`] its
//! `_core_tracker()` hands over, stamped on the session's clock by [`PlaybackTracker::anchor_to`].

use async_trait::async_trait;
use lk_agents_abi::{AudioFrame, TimedString, Ts, WallAnchor};
use lk_agents_rtc::{RtcError, TrackSid};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio::sync::Notify as IoChainNotify;

pub use lk_agents_abi::voice::rtc::AudioOutputCapabilities;

/// `PlaybackFinishedEvent` (the ABI's).
pub use lk_agents_abi::voice::room_io::PlaybackFinished;

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
#[lk_agents_abi::host(object(cell = "arc"))]
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
        *self.anchor.get_or_init(|| WallAnchor::of(&lk_agents_abi::RealClock::new()))
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
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.io.AudioOutput"))]
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
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.io.AudioInput"))]
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
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.io.VideoInput"))]
#[async_trait]
pub trait VideoInput: Send + Sync {
    fn label(&self) -> &str;

    /// The next frame, or `None` at end of stream.
    async fn next_frame(&self) -> Option<lk_agents_rtc::VideoFrameRef>;

    /// `set_video_enabled`: frames are dropped while detached.
    fn on_attached(&self) {}
    fn on_detached(&self) {}
}

/// `io.AudioOutput`: one link in the audio-out chain. Implement [`capture`](Self::capture) and
/// [`flush_segment`](Self::flush_segment); call [`AudioOutputExt::capture_frame`] and [`AudioOutputExt::flush`].
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.io.AudioOutput"))]
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
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.io.TextOutput"))]
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

/// A stable node beneath an audio wrapper chain, with its own segment tracker.
pub struct AudioSinkProxy {
    tracker: Arc<PlaybackTracker>,
    next: Arc<dyn AudioOutput>,
}

impl AudioSinkProxy {
    pub fn new(next: Arc<dyn AudioOutput>) -> Arc<Self> {
        let tracker = Arc::new(PlaybackTracker::new());
        next.tracker().observe(tracker.clone());
        Arc::new(AudioSinkProxy { tracker, next })
    }
}

#[async_trait]
impl AudioOutput for AudioSinkProxy {
    fn label(&self) -> &str {
        "AudioSinkProxy"
    }

    fn tracker(&self) -> &Arc<PlaybackTracker> {
        &self.tracker
    }

    /// The proxy itself imposes nothing; the answer is whatever is below it.
    fn capabilities(&self) -> AudioOutputCapabilities {
        AudioOutputCapabilities { pause: true }
    }

    fn sample_rate(&self) -> Option<u32> {
        self.next.sample_rate()
    }

    fn next_in_chain(&self) -> Option<Arc<dyn AudioOutput>> {
        Some(self.next.clone())
    }

    async fn capture(&self, frame: &AudioFrame) -> Result<()> {
        self.next.capture_frame(frame).await
    }

    fn flush_segment(&self) {
        self.next.flush();
    }

    fn clear_buffer(&self) {
        self.next.clear_buffer();
    }

    fn on_attached(&self) {
        self.next.on_attached();
    }

    fn on_detached(&self) {
        self.next.on_detached();
    }
}

// Text is handed to the node on its loop and never awaited: a `transcription_node` may hold text
// back arbitrarily long, and awaiting it per delta would pace the LLM leg by the host.

/// A host transcription node (`TranscriptionNodeSink`): pushes are fire-and-forget on its loop.
/// Seen through `voice/io.py`'s `_TextNode` surface, which a node need not subclass.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.io._TextNode"))]
pub trait TextNode: Send + Sync {
    fn capture_text(&self, text: &str);
    fn capture_timed(&self, text: &TimedString);
    fn flush(&self);
}

/// A [`TextNode`] as a link in the chain, with the core leg it emits into below it.
pub struct NodeTextOutput {
    node: Arc<dyn TextNode>,
    next: Arc<dyn TextOutput>,
}

impl NodeTextOutput {
    pub fn new(node: Arc<dyn TextNode>, next: Arc<dyn TextOutput>) -> Self {
        NodeTextOutput { node, next }
    }
}

#[async_trait]
impl TextOutput for NodeTextOutput {
    fn label(&self) -> &str {
        "Agent.transcription_node"
    }

    fn next_in_chain(&self) -> Option<Arc<dyn TextOutput>> {
        Some(Arc::clone(&self.next))
    }

    async fn capture_timed(&self, text: &TimedString) -> Result<()> {
        self.node.capture_timed(text);
        Ok(())
    }

    async fn capture_text(&self, text: &str) -> Result<()> {
        self.node.capture_text(text);
        Ok(())
    }

    fn flush(&self) {
        self.node.flush();
    }
}

/// A core text leg as a host object: what a host node emits into (`transcription_node`'s output,
/// a synchroniser's text node). The node's pump emits while the driver pushes above, on two threads.
pub struct CoreTextOutput(Arc<dyn TextOutput>);

#[lk_agents_abi::host(object(cell = "arc"))]
impl CoreTextOutput {
    pub fn new(inner: Arc<dyn TextOutput>) -> Self {
        CoreTextOutput(inner)
    }

    /// One transcript delta, downstream.
    #[host(method)]
    pub async fn capture_text(&self, text: String) -> std::result::Result<(), IoError> {
        self.0.capture_text(&text).await
    }

    /// The same, with word timings -- what `json_format` publishes on the wire.
    #[host(method(kw_only = "start_time"))]
    pub async fn capture_timed(
        &self,
        text: String,
        start_time: Option<f64>,
        end_time: Option<f64>,
    ) -> std::result::Result<(), IoError> {
        self.0
            .capture_timed(&TimedString { text, start_time, end_time, ..TimedString::default() })
            .await
    }

    /// The segment is complete. Synchronous, as on the trait.
    #[host(method)]
    pub fn flush(&self) {
        self.0.flush();
    }
}

/// `Agent.realtime_audio_output_node`: the realtime model's audio, on its way to the speaker.
///
/// One invocation per generation (a node's per-turn state must not carry over); frames are
/// pushed with `call_soon_threadsafe`, not a round trip.
#[lk_agents_abi::host(proxy(name = HostRealtimeAudioNode))]
#[async_trait::async_trait]
pub trait RealtimeAudioNode: Send + Sync {
    /// One frame in; returns as soon as it is queued.
    #[host(soon = "push")]
    async fn push(&self, frame: &lk_agents_abi::AudioFrame);

    /// The generation is over; the next `push` starts a fresh invocation.
    #[host(soon = "end")]
    fn end(&self);
}

/// The audio leg below a host node -- what a `realtime_audio_output_node` emits into.
pub struct CoreAudioSink(Arc<dyn AudioOutput>);

#[lk_agents_abi::host(object(cell = "arc"))]
impl CoreAudioSink {
    pub fn new(inner: Arc<dyn AudioOutput>) -> Self {
        CoreAudioSink(inner)
    }

    /// One frame, downstream. Awaited: the leg's backpressure paces playout.
    #[host(method)]
    pub async fn capture_frame(&self, frame: AudioFrame) -> std::result::Result<(), IoError> {
        self.0.capture_frame(&frame).await
    }

    /// The segment is complete -- `MessageDone`, in realtime terms.
    #[host(method)]
    pub fn flush(&self) {
        self.0.flush_segment();
    }

    /// InterruptByAudioActivity-in: drop whatever has not played.
    #[host(method)]
    pub fn clear_buffer(&self) {
        self.0.clear_buffer();
    }
}

pub type Result<T> = std::result::Result<T, IoError>;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
pub enum IoError {
    #[error(transparent)]
    Rtc(#[from] RtcError),

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
    Host(#[from] lk_agents_abi::errors::HostError),
}

/// A host failure keeps the original exception: re-raising it raises what the host raised.
#[cfg(feature = "python")]
impl lk_agents_abi::python::FromHostErr for IoError {
    fn from_host(py: pyo3::Python<'_>, err: &pyo3::PyErr) -> Self {
        IoError::Host(lk_agents_abi::python::capture_host(py, err))
    }
}

/// `voice/io.py`'s `AgentInput` as the driver holds it: the microphone and camera sources with the
/// tasks pumping them into the leg table, the gate that silences the substituted legs, and the STT
/// and VAD streams being reopened for a new agent.
pub(crate) struct AgentInput {
    /// So `SetAudioInput` can stop the old source first.
    audio_forwarder_parts: crate::audio::AudioForwarderParts,
    audio_forwarder: Option<lk_agents_abi::AbortHandle>,
    /// Held until the performer runs.
    pending_audio_in: Option<Option<Arc<dyn AudioInput>>>,
    audio_in: Option<Arc<dyn AudioInput>>,
    video_forwarder_parts: crate::audio::VideoForwarderParts,
    video_forwarder: Option<lk_agents_abi::AbortHandle>,
    pending_video_in: Option<Option<Arc<dyn VideoInput>>>,
    /// So the same leg handed back is a no-op.
    video_in: Option<Arc<dyn VideoInput>>,
    /// Shared: its rates are public attributes a caller may change mid-call.
    video_sampler: Option<Arc<crate::room_io::_input::VideoSampler>>,
    video_sink_warned: bool,
    gate: Arc<crate::agent_activity::GateSnapshot>,
    next_open: u64,
    /// A `StreamOpened` for any other open is discarded (0: none).
    stt_open: u64,
    vad_open: u64,
    stt_open_same: bool,
    speaker_filter: Option<crate::audio::SpeakerFilterConfig>,
}

impl AgentInput {
    /// The legs as the session opened them: each source with its pump (if one was attached), nothing
    /// pending.
    pub(crate) fn new(
        (audio_forwarder_parts, audio_forwarder, audio_in): (
            crate::audio::AudioForwarderParts,
            Option<lk_agents_abi::AbortHandle>,
            Option<Arc<dyn AudioInput>>,
        ),
        (video_forwarder_parts, video_forwarder, video_in): (
            crate::audio::VideoForwarderParts,
            Option<lk_agents_abi::AbortHandle>,
            Option<Arc<dyn VideoInput>>,
        ),
        video_sampler: Option<Arc<crate::room_io::_input::VideoSampler>>,
        gate: Arc<crate::agent_activity::GateSnapshot>,
        speaker_filter: Option<crate::audio::SpeakerFilterConfig>,
    ) -> Self {
        AgentInput {
            audio_forwarder_parts,
            audio_forwarder,
            pending_audio_in: None,
            audio_in,
            video_forwarder_parts,
            video_forwarder,
            pending_video_in: None,
            video_in,
            video_sampler,
            video_sink_warned: false,
            gate,
            next_open: 0,
            stt_open: 0,
            vad_open: 0,
            stt_open_same: false,
            speaker_filter,
        }
    }

    /// Read rather than borrowed: a handoff can replace it.
    pub(crate) fn leg(
        &self,
        leg: crate::audio::AudioChannel,
    ) -> Option<Arc<dyn lk_agents_abi::StreamSink<AudioFrame>>> {
        self.audio_forwarder_parts.legs.get(leg)
    }

    /// The one leg table every source feeds.
    pub(crate) fn legs(&self) -> &Arc<crate::audio::AudioChannels> {
        &self.audio_forwarder_parts.legs
    }

    /// Run the gate: whether the substituted legs now get silence, **only if that changed**.
    pub(crate) fn publish_gate(&self, inputs: &crate::audio::GateInputs) -> Option<bool> {
        self.gate.publish(inputs)
    }

    /// Where a leg's drain reports its end.
    pub(crate) fn leg_ended(&self) -> &crate::agent_session::Tx<crate::audio::ChannelEnd> {
        &self.audio_forwarder_parts.ended
    }

    /// The speaker filter rides the STT row: a fresh detector per stream opened from now on, and
    /// the installed stream's replaced.
    pub(crate) fn set_speaker_filter(&mut self, config: Option<crate::audio::SpeakerFilterConfig>) {
        self.audio_forwarder_parts
            .legs
            .set_speaker_filter(config.as_ref().map(crate::audio::speaker_filter));
        self.speaker_filter = config;
    }

    /// A fresh detector for a stream about to open, when a filter is configured.
    pub(crate) fn new_speaker_filter(&self) -> Option<Arc<crate::audio::SpeakerFilter>> {
        self.speaker_filter.as_ref().map(crate::audio::speaker_filter)
    }

    /// Whether there is a local VAD (`self.vad is None` decides whether an interim is the onset).
    pub(crate) fn has_vad(&self) -> bool {
        self.audio_forwarder_parts.legs.present(crate::audio::AudioChannel::Vad)
    }

    /// The one door every leg mutation goes through. A new source is parked until the performer
    /// swaps the pump (`AudioOp::SetInput`).
    pub(crate) fn apply_leg_op(
        &mut self,
        op: crate::audio::ChannelOp,
    ) -> Option<crate::audio::Ticket> {
        use crate::audio::ChannelOp;
        match op {
            ChannelOp::Install { leg, sink } => {
                Some(self.audio_forwarder_parts.legs.install(leg, sink))
            }
            ChannelOp::InstallStt { sink, speaker } => {
                Some(self.audio_forwarder_parts.legs.install_stt(sink, speaker))
            }
            ChannelOp::Clear(leg) => {
                self.audio_forwarder_parts.legs.clear(leg);
                None
            }
            // Abort the old pump first, so two sources never feed the same table.
            ChannelOp::SetSource(input) => {
                self.pending_audio_in = Some(input);
                None
            }
        }
    }

    /// The parked microphone replaces the pumped one: the old pump is aborted first (two sources
    /// on one table look like the user talking over themselves).
    pub(crate) fn swap_audio_forwarder(&mut self, spawner: &dyn lk_agents_abi::Spawner) {
        let Some(input) = self.pending_audio_in.take() else { return };
        if let Some(old) = self.audio_forwarder.take() {
            old.abort();
        }
        self.audio_forwarder = input.map(|input| {
            lk_agents_abi::SpawnerExt::spawn_unboxed(
                spawner,
                self.audio_forwarder_parts.clone().pump(input),
            )
        });
    }

    /// The parked camera replaces the pumped one, the old pump aborted first.
    pub(crate) fn swap_video_forwarder(&mut self, spawner: &dyn lk_agents_abi::Spawner) {
        let Some(input) = self.pending_video_in.take() else { return };
        if let Some(old) = self.video_forwarder.take() {
            old.abort();
        }
        self.video_forwarder = input.map(|input| {
            lk_agents_abi::SpawnerExt::spawn_unboxed(
                spawner,
                self.video_forwarder_parts.clone().pump(input),
            )
        });
    }

    /// The reference's setter tells both streams (`AgentInput.audio`): the old one detached, so it
    /// drops frames, the new one attached if enabled.
    pub(crate) fn replace_audio_in(&mut self, input: Option<Arc<dyn AudioInput>>) {
        if let Some(old) = self.audio_in.take() {
            old.on_detached();
        }
        if let Some(new) = &input
            && self.audio_forwarder_parts.input_enabled.load(std::sync::atomic::Ordering::Relaxed)
        {
            new.on_attached();
        }
        self.audio_in = input;
    }

    /// A re-attached source forgets when audio first flowed.
    pub(crate) fn forget_first_input(&self) {
        self.audio_forwarder_parts
            .first_input_at
            .store(crate::audio::NO_INPUT_YET, std::sync::atomic::Ordering::Relaxed);
    }

    /// When audio started flowing from the current source, if it has.
    pub(crate) fn first_input_at(&self) -> Option<Ts> {
        let first =
            self.audio_forwarder_parts.first_input_at.load(std::sync::atomic::Ordering::Relaxed);
        (first != crate::audio::NO_INPUT_YET).then(|| Ts::from_nanos(first))
    }

    /// `input.set_audio_enabled(..)`: whether it changed; the host object is told. The leg stays
    /// installed.
    pub(crate) fn set_audio_enabled(&self, enabled: bool) -> bool {
        let was = self
            .audio_forwarder_parts
            .input_enabled
            .swap(enabled, std::sync::atomic::Ordering::Relaxed);
        if was != enabled
            && let Some(input) = &self.audio_in
        {
            if enabled { input.on_attached() } else { input.on_detached() }
        }
        was != enabled
    }

    /// The camera handed over, parked for the performer. `false` when it is the leg already read
    /// (nothing to swap).
    pub(crate) fn replace_video_in(&mut self, input: Option<Arc<dyn VideoInput>>) -> bool {
        let same = match (&self.video_in, &input) {
            (Some(old), Some(new)) => Arc::ptr_eq(old, new),
            (None, None) => true,
            (Some(_), None) | (None, Some(_)) => false,
        };
        if same {
            return false;
        }
        if let Some(old) = self.video_in.take() {
            old.on_detached();
        }
        if let Some(new) = &input {
            new.on_attached();
        }
        self.video_in = input.clone();
        self.pending_video_in = Some(input);
        true
    }

    /// `VoiceActivityVideoSampler`'s rate gate (faster while the user speaks); `None` samples
    /// every frame.
    pub(crate) fn samples_video(&self, user_speaking: bool, now: Ts) -> bool {
        self.video_sampler.as_ref().is_none_or(|s| s.should_sample_at(user_speaking, now))
    }

    /// A pipeline session has nowhere to push a frame: `true` the first time only.
    pub(crate) fn warn_video_sink_once(&mut self) -> bool {
        !std::mem::replace(&mut self.video_sink_warned, true)
    }

    pub(crate) fn mint_open(&mut self) -> u64 {
        self.next_open += 1;
        self.next_open
    }

    /// The open a leg's `StreamOpened` must carry to land (0: none), and for the STT whether it
    /// reopens the same recogniser.
    pub(crate) fn expect_open(
        &mut self,
        leg: crate::agent_session::effects::StreamKind,
        open: u64,
    ) {
        use crate::agent_session::effects::StreamKind;
        match leg {
            StreamKind::Stt => self.stt_open = open,
            StreamKind::Vad => self.vad_open = open,
        }
    }

    pub(crate) fn expected_open(&self, leg: crate::agent_session::effects::StreamKind) -> u64 {
        use crate::agent_session::effects::StreamKind;
        match leg {
            StreamKind::Stt => self.stt_open,
            StreamKind::Vad => self.vad_open,
        }
    }

    pub(crate) fn set_stt_open_same(&mut self, same: bool) {
        self.stt_open_same = same;
    }

    pub(crate) fn stt_open_same(&self) -> bool {
        self.stt_open_same
    }

    /// One `_detector` per wrapper stream: the event to forward and a changed speaker verdict.
    pub(crate) fn filter_speaker(
        &self,
        open: u32,
        ev: lk_agents_abi::SpeechEvent,
    ) -> crate::audio::SpeakerDecision {
        match self.audio_forwarder_parts.legs.speaker_filter_for(open) {
            Some(filter) => filter.decide(ev),
            None => crate::audio::SpeakerDecision { kept: Some(ev), changed: None },
        }
    }

    /// `_commit_user_turn`'s "flush the stt by pushing silence", `ceil(duration / 0.2)` frames:
    /// the frames pushed, `None` before any audio or with no STT leg.
    pub(crate) fn push_stt_silence(&self, seconds: f64) -> Option<u32> {
        let packed =
            self.audio_forwarder_parts.last_input.load(std::sync::atomic::Ordering::Relaxed);
        if packed == 0 || !(seconds > 0.0) {
            return None;
        }
        let rate = (packed >> 32) as u32;
        let channels = (packed & 0xffff_ffff) as u32;
        let stt = self.leg(crate::audio::AudioChannel::Stt)?;
        // `ceil`, so at least the asked duration is flushed.
        let per_frame = (f64::from(rate) * 0.2) as u32;
        if per_frame == 0 {
            return None;
        }
        let count = (seconds / 0.2).ceil() as u32;
        for _ in 0..count {
            let _ = stt.push(lk_agents_rtc::silence_frame(per_frame, rate, channels as u16));
        }
        Some(count)
    }
}
