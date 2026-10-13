//! What leaves the session, and the inputs that stay silent. Every output instance the case
//! attaches carries its own index, so sink calls on a replaced instance are distinguishable from
//! the new one's. Audio segments are invocations the case reports on the way a speaker does:
//! started, and -- once flushed -- finished. Also the microphone, the always-pending video input,
//! and the recorder tap.
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use lk_agents_abi::{AudioFrame, MediaRef, Ts, errors::RefRegistry};
use lk_agents_media::recorder::{AudioRecorder, RecorderError};
use lk_agents_voice::io::{
    self, AudioInput, AudioOutput, AudioOutputCapabilities, PlaybackFinished, PlaybackTracker,
    TextOutput, VideoInput,
};

use super::{Control, Registry};
use crate::{
    input::Input,
    record::{Detail, Kind, RecorderCall, Sink, Tag},
};

/// Keeps the microphone leg open without supplying frames.
/// The microphone: whatever frames the case pushes, in order.
///
/// Every frame is tagged so it can be traced to the provider streams it reaches. Whether it was
/// speech is not the frame's to say -- the case injects VAD and STT's verdicts separately, which
/// is what lets a rule test how the driver reconciles them.
pub struct FakeAudioInput {
    frames: tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<AudioFrame>>,
}

impl FakeAudioInput {
    pub fn new() -> (Arc<Self>, tokio::sync::mpsc::UnboundedSender<AudioFrame>) {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        (Arc::new(FakeAudioInput { frames: tokio::sync::Mutex::new(receiver) }), sender)
    }
}

#[async_trait::async_trait]
impl AudioInput for FakeAudioInput {
    fn label(&self) -> &str {
        "fake::audio_input"
    }
    async fn next_frame(&self) -> Option<AudioFrame> {
        self.frames.lock().await.recv().await
    }
}

/// The camera: whatever frames the case pushes, in order. Each attached camera is a resource the
/// case feeds by `VideoFrames`; a camera nobody feeds keeps the leg open and yields nothing.
pub struct FakeVideoInput {
    frames: tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<u64>>,
    refs: Arc<dyn RefRegistry>,
}

impl FakeVideoInput {
    pub fn new(registry: &Registry, refs: Arc<dyn RefRegistry>) -> Arc<Self> {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        registry.register(Kind::Video, Detail::None, Control::Video(sender));
        Arc::new(FakeVideoInput { frames: tokio::sync::Mutex::new(receiver), refs })
    }
}

#[async_trait::async_trait]
impl VideoInput for FakeVideoInput {
    fn label(&self) -> &str {
        "fake::video_input"
    }
    async fn next_frame(&self) -> Option<lk_agents_rtc::VideoFrameRef> {
        let ordinal = self.frames.lock().await.recv().await?;
        Some(lk_agents_rtc::VideoFrameRef {
            media: MediaRef::new(ordinal, &self.refs),
            width: 1280,
            height: 720,
            timestamp_us: (ordinal & 0xffff_ffff) as i64 * 33_000,
        })
    }
}

/// So many frames into a camera, each named by the camera's index and its ordinal.
pub(super) fn feed_video(
    sender: &tokio::sync::mpsc::UnboundedSender<u64>,
    video: u32,
    what: &Input,
) -> bool {
    let Input::VideoFrames { frames, .. } = what else { return false };
    static ORDINAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for _ in 0..frames.0 {
        let nth = ORDINAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed) & 0xffff_ffff;
        if sender.send(u64::from(video) << 32 | nth).is_err() {
            return false;
        }
    }
    true
}

/// The playout side of one audio output: the segment taking frames, the flushed segments still
/// owing a report, and the tracker every report goes through.
///
/// A segment opens at its first frame, exactly as the driver's own tracker counts them, and a
/// flush closes it to new frames. The case then reports on it the way a speaker does: `started`
/// when the first of its audio actually leaves for the listener -- the edge the driver marks the
/// agent as speaking on, and so the only thing that makes an overlap possible -- and `finished`
/// once it has all played, which is only meaningful after the flush.
///
/// **One owner for both ways a segment ends.** A playout report the case delivers and a clear the
/// product performs are the same question asked from two sides -- which segments are still
/// outstanding -- and an answer kept in two places drifts: the output would re-report a segment
/// the case had already released. So the set lives here, with the output that flushed them, and
/// the harness keeps only the routing table. The lock holds plain numbers and is always released
/// before the tracker is called, so a tracker that notifies a recorder cannot re-enter it.
pub struct Playout {
    tracker: Arc<PlaybackTracker>,
    state: Mutex<PlayoutState>,
}

#[derive(Default)]
struct PlayoutState {
    /// Audio captured since the last flush, as the position the open segment will report.
    duration: Duration,
    /// The segment taking frames, once its first frame gave it an identity.
    open: Option<u32>,
    /// Segments closed to frames and not yet reported or cleared, with the position each reports.
    flushed: BTreeMap<u32, Duration>,
    /// Segments whose start the speaker has reported.
    started: BTreeSet<u32>,
    /// How much of each started segment the speaker has reported played so far.
    played: BTreeMap<u32, Duration>,
}

impl Playout {
    /// One more frame. True when it is the first of a new segment, which then needs an identity.
    fn captured(&self, duration: Duration) -> bool {
        let mut state = self.state.lock().unwrap();
        state.duration += duration;
        state.open.is_none()
    }

    /// The segment the frames since the last flush belong to.
    fn opened(&self, segment: u32) {
        self.state.lock().unwrap().open = Some(segment);
    }

    /// The open segment is closed to new frames and carries what was captured for it. `None`
    /// when nothing was captured since the last flush: the tracker opened no segment for that
    /// flush either, so there is nothing to report on.
    fn flushed(&self) -> Option<u32> {
        let mut state = self.state.lock().unwrap();
        let segment = state.open.take()?;
        let position = std::mem::take(&mut state.duration);
        state.flushed.insert(segment, position);
        Some(segment)
    }

    /// The speaker began emitting this segment. True when the segment is outstanding and had not
    /// started yet; a second start, or one for a segment already gone, is refused.
    pub(super) fn started(&self, segment: u32, at: Ts) -> bool {
        {
            let mut state = self.state.lock().unwrap();
            let outstanding = state.open == Some(segment) || state.flushed.contains_key(&segment);
            if !outstanding || !state.started.insert(segment) {
                return false;
            }
        }
        self.tracker.playback_started(at);
        true
    }

    /// A stretch of this segment played. True when the segment is outstanding and had started;
    /// the stretch begins where the last one ended, as a real output's runs do.
    pub(super) fn progressed(&self, segment: u32, played: Duration, at: Ts) -> bool {
        let offset = {
            let mut state = self.state.lock().unwrap();
            if !state.started.contains(&segment) {
                return false;
            }
            let so_far = state.played.entry(segment).or_default();
            let offset = *so_far;
            *so_far += played;
            offset
        };
        self.tracker.playback_progressed(at, offset, played);
        true
    }

    /// The output threw its playback count away: every segment still outstanding is forgotten
    /// here and the tracker is reset, which is what releases a driver waiting on any of them.
    /// Returns the segments that were cut.
    pub(super) fn reset(&self) -> Vec<u32> {
        let cut: Vec<u32> = {
            let mut state = self.state.lock().unwrap();
            let mut cut: Vec<u32> = std::mem::take(&mut state.flushed).into_keys().collect();
            cut.extend(state.open.take());
            state.duration = Duration::ZERO;
            state.started.clear();
            state.played.clear();
            cut
        };
        self.tracker.reset();
        cut
    }

    /// This segment finished playing. True when it was flushed and still owed; a segment still
    /// taking frames cannot have finished, and one already reported is not reported twice.
    pub(super) fn report(&self, segment: u32) -> bool {
        let position = {
            let mut state = self.state.lock().unwrap();
            state.started.remove(&segment);
            state.played.remove(&segment);
            match state.flushed.remove(&segment) {
                Some(position) => position,
                None => return false,
            }
        };
        self.tracker.playback_finished(PlaybackFinished {
            playback_position: position,
            ..Default::default()
        })
    }

    /// The product threw away the buffer: every segment still outstanding, open or flushed, is
    /// reported interrupted at the position captured for it, the way a real output does.
    fn cleared(&self) -> Vec<u32> {
        let cut: Vec<(u32, Duration)> = {
            let mut state = self.state.lock().unwrap();
            let mut cut: Vec<(u32, Duration)> =
                std::mem::take(&mut state.flushed).into_iter().collect();
            if let Some(open) = state.open.take() {
                cut.push((open, std::mem::take(&mut state.duration)));
            }
            state.started.clear();
            state.played.clear();
            cut
        };
        for (_, position) in &cut {
            self.tracker.playback_finished(PlaybackFinished {
                interrupted: true,
                playback_position: *position,
                ..Default::default()
            });
        }
        cut.into_iter().map(|(segment, _)| segment).collect()
    }
}

pub struct FakeAudioOutput {
    id: u32,
    registry: Arc<Registry>,
    playout: Arc<Playout>,
    /// Whether this output says it can pause. A pause is recorded, nothing more: the playout is
    /// the case's to report on either way, which is what a paused speaker looks like from here.
    pausable: bool,
}

impl FakeAudioOutput {
    pub fn new(registry: Arc<Registry>, pausable: bool) -> Arc<Self> {
        let playout = Arc::new(Playout { tracker: Arc::default(), state: Mutex::default() });
        let id = registry.new_audio_output(playout.clone());
        Arc::new(FakeAudioOutput { id, registry, playout, pausable })
    }
}

#[async_trait::async_trait]
impl AudioOutput for FakeAudioOutput {
    fn label(&self) -> &str {
        "fake::audio_output"
    }

    fn tracker(&self) -> &Arc<PlaybackTracker> {
        &self.playout.tracker
    }

    fn capabilities(&self) -> AudioOutputCapabilities {
        AudioOutputCapabilities { pause: self.pausable }
    }

    fn pause(&self) {
        self.registry.record_sink(Sink::Paused { output: self.id });
    }

    fn resume(&self) {
        self.registry.record_sink(Sink::Resumed { output: self.id });
    }

    async fn capture(&self, frame: &AudioFrame) -> io::Result<()> {
        // The first frame after a flush opens a segment, as the driver's tracker counts them;
        // the case reports on it from here on, so it needs its identity now, not at the flush.
        if self.playout.captured(Duration::from_secs_f64(frame.duration())) {
            let control = Control::Segment(self.playout.clone());
            let segment = self.registry.register(Kind::Segment, Detail::None, control).nth;
            self.playout.opened(segment);
            self.registry.attach_segment(segment, self.id);
        }
        let mut tags: Vec<Tag> = Vec::new();
        for sample in frame.data.as_chunks::<2>().0 {
            let tag = Tag::decode(i16::from_le_bytes(*sample));
            if tags.last() != Some(&tag) {
                tags.push(tag);
            }
        }
        self.registry.record_sink(Sink::Audio { output: self.id, tags });
        Ok(())
    }

    fn flush_segment(&self) {
        // A flush with nothing captured owes nothing: the tracker opened no segment for it.
        if let Some(segment) = self.playout.flushed() {
            self.registry.record_sink(Sink::Flush { output: self.id, segment });
        }
    }

    fn clear_buffer(&self) {
        let segments = self.playout.cleared();
        self.registry.record_sink(Sink::Clear { output: self.id, segments });
    }
}

pub struct FakeTextOutput {
    id: u32,
    registry: Arc<Registry>,
}

impl FakeTextOutput {
    pub fn new(registry: Arc<Registry>) -> Arc<Self> {
        let id = registry.new_output(false);
        Arc::new(FakeTextOutput { id, registry })
    }
}

#[async_trait::async_trait]
impl TextOutput for FakeTextOutput {
    fn label(&self) -> &str {
        "fake::text_output"
    }

    async fn capture_text(&self, text: &str) -> io::Result<()> {
        self.registry.record_sink(Sink::Text { output: self.id, text: text.to_string() });
        Ok(())
    }

    fn flush(&self) {
        self.registry.record_sink(Sink::TextFlush { output: self.id });
    }
}

/// The recorder tap: records that it was called, nothing else.
pub struct FakeRecorder {
    registry: Arc<Registry>,
}

impl FakeRecorder {
    pub fn new(registry: Arc<Registry>) -> Arc<Self> {
        Arc::new(FakeRecorder { registry })
    }
}

impl AudioRecorder for FakeRecorder {
    fn on_input_frame(&self, _now: f64, _frame: &AudioFrame) {
        self.registry.record_sink(Sink::Recorder(RecorderCall::InputFrame));
    }
    fn capture_frame(&self, _now: f64, _frame: &AudioFrame) {
        self.registry.record_sink(Sink::Recorder(RecorderCall::CaptureFrame));
    }
    fn on_playback_started(&self, _created_at: f64) {
        self.registry.record_sink(Sink::Recorder(RecorderCall::PlaybackStarted));
    }
    fn on_playback_progressed(&self, _started_at: f64, _offset: f64, _duration: f64) {
        self.registry.record_sink(Sink::Recorder(RecorderCall::PlaybackProgressed));
    }
    fn on_playback_finished(&self, _now: f64, _playback_position: f64) {
        self.registry.record_sink(Sink::Recorder(RecorderCall::PlaybackFinished));
    }
    fn flush(&self, _now: f64) -> Result<(), RecorderError> {
        self.registry.record_sink(Sink::Recorder(RecorderCall::Flush));
        Ok(())
    }
}
