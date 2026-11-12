use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use lk_agents_abi::SpawnerExt as _;
use lk_agents_abi::utils::video::FakeFrame;
use lk_agents_abi::voice::io;
use lk_agents_abi::voice::io::{
    AudioInput, AudioOutput, AudioOutputCapabilities, PlaybackFinished, PlaybackTracker,
    TranscriptionOutput, VideoInput,
};
use lk_agents_abi::{AudioFrame, Ts};

use super::{Control, Registry};
use crate::{
    input::Input,
    record::{Detail, Kind, Sink, Tag},
};

// Each attached output carries its own index, so sink calls on a replaced instance are
// distinguishable from the new one's.

/// The microphone: the case's frames, in order. Frames are tagged, never judged as speech: the
/// case injects VAD and STT verdicts separately, so a rule can test how the driver reconciles them.
pub struct FakeAudioInput {
    frames: tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<AudioFrame>>,
}

impl FakeAudioInput {
    pub fn new() -> (Arc<Self>, tokio::sync::mpsc::UnboundedSender<AudioFrame>) {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        (Arc::new(FakeAudioInput { frames: tokio::sync::Mutex::new(receiver) }), sender)
    }
}

impl AudioInput for FakeAudioInput {
    fn label(&self) -> &str {
        "fake::audio_input"
    }
    fn next_frame(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'_, lk_agents_abi::voice::io::Result<Option<AudioFrame>>> {
        Box::pin(async move { Ok(self.frames.lock().await.recv().await) })
    }
}

/// A camera the case feeds by `VideoFrames`; unfed, it keeps the leg open and yields nothing.
pub struct FakeVideoInput {
    frames: tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<u64>>,
}

impl FakeVideoInput {
    pub fn new(registry: &Registry) -> Arc<Self> {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        registry.register(Kind::Video, Detail::None, Control::Video(sender));
        Arc::new(FakeVideoInput { frames: tokio::sync::Mutex::new(receiver) })
    }
}

impl VideoInput for FakeVideoInput {
    fn label(&self) -> &str {
        "fake::video_input"
    }
    fn next_frame(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        lk_agents_abi::voice::io::Result<Option<lk_agents_abi::voice::rtc::VideoFrameRef>>,
    > {
        Box::pin(async move {
            let Some(ordinal) = self.frames.lock().await.recv().await else { return Ok(None) };
            Ok(Some(lk_agents_abi::voice::rtc::VideoFrameRef {
                frame: FakeFrame::shared(ordinal),
                width: 1280,
                height: 720,
                timestamp_us: (ordinal & 0xffff_ffff) as i64 * 33_000,
            }))
        })
    }
}

/// The person's camera track as a resource the case feeds by `VideoFrames`.
pub fn room_camera(registry: &Registry, camera: Arc<crate::fakes::room::FakeTrackPublication>) {
    registry.register(Kind::Video, Detail::RoomCamera, Control::RoomVideo(camera));
}

/// Frames onto the person's camera track. True when a reader took them.
pub(super) fn feed_room_video(
    camera: &crate::fakes::room::FakeTrackPublication,
    video: u32,
    what: &Input,
) -> bool {
    let (sender, mut frames) = tokio::sync::mpsc::unbounded_channel();
    if !feed_video(&sender, video, what) {
        return false;
    }
    drop(sender);
    let mut read = false;
    while let Ok(ordinal) = frames.try_recv() {
        let frame = lk_agents_abi::voice::rtc::VideoFrameRef {
            frame: FakeFrame::shared(ordinal),
            width: 1280,
            height: 720,
            timestamp_us: (ordinal & 0xffff_ffff) as i64 * 33_000,
        };
        read |= camera.push_video_frame(frame) > 0;
    }
    read
}

/// Frames into a camera, each named by the camera's index and its ordinal.
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

/// The playout side of one audio output. A segment opens at its first frame, as the driver's
/// tracker counts them, and a flush closes it. The case reports `started` when its first audio
/// leaves (the edge the driver marks speaking on, so the only one that makes an overlap possible)
/// and `finished` once it all played.
///
/// The outstanding set has one owner for both ways a segment ends, a case report and a product
/// clear, or the output would re-report a segment the case already released. The lock is
/// released before the tracker is called, so a tracker that notifies a recorder cannot re-enter.
pub struct Playout {
    tracker: Arc<dyn PlaybackTracker>,
    state: Mutex<PlayoutState>,
    /// A host source: a cut is reported a few scheduler hops after the clear, as the report
    /// crosses back from the host's own loop.
    lagging: Option<Arc<dyn lk_agents_abi::Spawner>>,
    registry: Option<Arc<Registry>>,
}

/// Hops a lagging output takes to report a cut: more than the driver yields for it.
const LAGGING_REPORT_HOPS: usize = 64;

#[derive(Default)]
struct PlayoutState {
    /// Audio captured since the last flush.
    duration: Duration,
    open: Option<u32>,
    keys: BTreeMap<u32, lk_agents_abi::voice::io::PlaybackSegment>,
    /// Segments flushed and not yet reported or cleared, with the position each reports.
    flushed: BTreeMap<u32, Duration>,
    started: BTreeSet<u32>,
    played: BTreeMap<u32, Duration>,
    /// The open segment's frames, by realtime generation (none for other audio).
    frames: Vec<(Option<u32>, Duration)>,
    flushed_frames: BTreeMap<u32, Vec<(Option<u32>, Duration)>>,
    /// Per realtime generation: the audio captured, and how much of it a segment's end settled.
    captured_by: BTreeMap<u32, Duration>,
    settled_by: BTreeMap<u32, Duration>,
}

impl Playout {
    /// True when the frame is the first of a new segment, which then needs an identity.
    fn captured(&self, duration: Duration, generation: Option<u32>) -> bool {
        let mut state = self.state.lock().unwrap();
        state.duration += duration;
        state.frames.push((generation, duration));
        if let Some(g) = generation {
            *state.captured_by.entry(g).or_default() += duration;
        }
        state.open.is_none()
    }

    /// Records what was heard of each realtime generation in `frames` (the first `played`), and
    /// whether all of it captured so far has now played or been cut.
    fn heard(&self, frames: Vec<(Option<u32>, Duration)>, mut played: Duration) {
        let mut by: BTreeMap<u32, Duration> = BTreeMap::new();
        let mut settled: BTreeMap<u32, Duration> = BTreeMap::new();
        for (generation, len) in frames {
            let heard = len.min(played);
            played -= heard;
            if let Some(g) = generation {
                *by.entry(g).or_default() += heard;
                *settled.entry(g).or_default() += len;
            }
        }
        let whole: BTreeMap<u32, bool> = {
            let mut state = self.state.lock().unwrap();
            settled
                .into_iter()
                .map(|(g, len)| {
                    let done = state.settled_by.entry(g).or_default();
                    *done += len;
                    let done = *done;
                    (g, done >= state.captured_by.get(&g).copied().unwrap_or_default())
                })
                .collect()
        };
        if let Some(registry) = &self.registry {
            for (g, d) in by {
                registry.record_heard(g, d.as_millis() as u64, whole[&g]);
            }
        }
    }

    fn opened(&self, segment: u32) {
        let key = self.tracker.current_segment().expect("a captured frame has a segment key");
        let mut state = self.state.lock().unwrap();
        state.open = Some(segment);
        state.keys.insert(segment, key);
    }

    /// Closes the open segment. `None` when nothing was captured since the last flush: the
    /// tracker opened no segment either.
    fn flushed(&self) -> Option<(u32, Duration)> {
        let mut state = self.state.lock().unwrap();
        let segment = state.open.take()?;
        let position = std::mem::take(&mut state.duration);
        state.flushed.insert(segment, position);
        let frames = std::mem::take(&mut state.frames);
        state.flushed_frames.insert(segment, frames);
        Some((segment, position))
    }

    /// The speaker began emitting this segment. Refused unless it is the oldest outstanding one
    /// and nothing else is playing: a real sink plays its queue in order, one at a time.
    pub(super) fn started(&self, segment: u32, at: Ts) -> bool {
        let key = {
            let mut state = self.state.lock().unwrap();
            let oldest = state.flushed.keys().copied().chain(state.open).min();
            if oldest != Some(segment) || !state.started.is_empty() {
                return false;
            }
            state.started.insert(segment);
            state.keys[&segment]
        };
        self.tracker.playback_started(key, at);
        true
    }

    /// A stretch of a started segment played, beginning where the last one ended.
    pub(super) fn progressed(&self, segment: u32, played: Duration, at: Ts) -> bool {
        let offset = {
            let mut state = self.state.lock().unwrap();
            if !state.started.contains(&segment) {
                return false;
            }
            // A speaker plays no more than the segment holds.
            let captured = match state.flushed.get(&segment) {
                Some(position) => *position,
                None => state.duration,
            };
            let so_far = state.played.entry(segment).or_default();
            let offset = *so_far;
            let played = played.min(captured.saturating_sub(offset));
            if played.is_zero() {
                return false;
            }
            *so_far += played;
            (state.keys[&segment], offset, played)
        };
        let (key, offset, played) = offset;
        self.tracker.playback_progressed(key, at, offset, played);
        true
    }

    /// Reports every outstanding segment interrupted, then resets the tracker, as the reference's
    /// only resetting sink does (`DataStreamAudioOutput._clear_buffer_task._on_timeout`): a reset
    /// alone leaves a wrapper waiting on the segment forever. Returns the segments cut.
    pub(super) fn reset(&self) -> Vec<u32> {
        let cut = self.cleared();
        self.tracker.reset();
        cut
    }

    /// This segment finished playing. True when it was flushed and still owed. A finish the
    /// product already settled (its lost-finish bound) is still reported; the tracker drops it
    /// (seed 11256).
    pub(super) fn report(&self, segment: u32) -> bool {
        let Some(key) = self.state.lock().unwrap().keys.get(&segment).copied() else {
            return false;
        };
        let Some(position) = self.finished(segment) else { return false };
        self.tracker.playback_finished(
            key,
            PlaybackFinished { playback_position: position, ..Default::default() },
        );
        true
    }

    /// This segment played to the end but its finish report was lost. True when it was flushed
    /// and still owed.
    pub(super) fn lose(&self, segment: u32, at: Ts) -> bool {
        let (start, rest) = {
            let state = self.state.lock().unwrap();
            let Some(position) = state.flushed.get(&segment).copied() else { return false };
            let played = state.played.get(&segment).copied().unwrap_or_default();
            (!state.started.contains(&segment), position.saturating_sub(played))
        };
        if start && !self.started(segment, at) {
            return false;
        }
        if !rest.is_zero() {
            self.progressed(segment, rest, at);
        }
        self.finished(segment).is_some()
    }

    /// Retires a flushed segment, heard whole; its length, or `None` when it was not owed.
    fn finished(&self, segment: u32) -> Option<Duration> {
        let (position, frames) = {
            let mut state = self.state.lock().unwrap();
            let position = state.flushed.remove(&segment)?;
            state.started.remove(&segment);
            state.played.remove(&segment);
            state.keys.remove(&segment);
            let frames = state.flushed_frames.remove(&segment).unwrap_or_default();
            (position, frames)
        };
        self.heard(frames, position);
        Some(position)
    }

    /// Reports every outstanding segment, open or flushed, interrupted where the speaker had
    /// played it to, as a real output does on a clear.
    fn cleared(&self) -> Vec<u32> {
        let cut: Vec<(u32, lk_agents_abi::voice::io::PlaybackSegment, Duration)> = {
            let mut state = self.state.lock().unwrap();
            type OutstandingGeneration = (u32, Vec<(Option<u32>, Duration)>);
            let mut outstanding: Vec<OutstandingGeneration> = std::mem::take(&mut state.flushed)
                .into_keys()
                .map(|s| (s, state.flushed_frames.remove(&s).unwrap_or_default()))
                .collect();
            if let Some(open) = state.open.take() {
                state.duration = Duration::ZERO;
                outstanding.push((open, std::mem::take(&mut state.frames)));
            }
            let played = std::mem::take(&mut state.played);
            let keys = std::mem::take(&mut state.keys);
            state.started.clear();
            drop(state);
            outstanding
                .into_iter()
                .map(|(segment, frames)| {
                    let position = played.get(&segment).copied().unwrap_or_default();
                    self.heard(frames, position);
                    (segment, keys[&segment], position)
                })
                .collect()
        };
        let finished: Vec<_> = cut
            .iter()
            .map(|(_, key, position)| {
                (
                    *key,
                    PlaybackFinished {
                        interrupted: true,
                        playback_position: *position,
                        ..Default::default()
                    },
                )
            })
            .collect();
        if let Some(world) = &self.lagging {
            let tracker = self.tracker.clone();
            // The world's own work: the run's executor ends it with the run.
            world.spawn_detached(async move {
                for _ in 0..LAGGING_REPORT_HOPS {
                    tokio::task::yield_now().await;
                }
                for (key, ev) in finished {
                    tracker.playback_finished(key, ev);
                }
            });
        } else {
            for (key, ev) in finished {
                self.tracker.playback_finished(key, ev);
            }
        }
        cut.into_iter().map(|(segment, _, _)| segment).collect()
    }
}

pub struct FakeAudioOutput {
    id: u32,
    registry: Arc<Registry>,
    playout: Arc<Playout>,
    /// A pause is only recorded: playout stays the case's to report, as a paused speaker looks.
    pausable: bool,
}

impl FakeAudioOutput {
    pub fn new(registry: Arc<Registry>, pausable: bool, lagging: bool) -> Arc<Self> {
        let playout = Arc::new(Playout {
            tracker: Arc::new(lk_agents_runtime::voice::playback::Tracker::new()),
            state: Mutex::default(),
            lagging: lagging.then(|| registry.world.clone()),
            registry: Some(registry.clone()),
        });
        let id = registry.new_audio_output(playout.clone());
        Arc::new(FakeAudioOutput { id, registry, playout, pausable })
    }
}

impl Drop for FakeAudioOutput {
    fn drop(&mut self) {
        self.registry.mark_closed(super::Slot { kind: Kind::Output, nth: self.id });
    }
}

impl AudioOutput for FakeAudioOutput {
    fn label(&self) -> &str {
        "fake::audio_output"
    }

    fn tracker(&self) -> &Arc<dyn PlaybackTracker> {
        &self.playout.tracker
    }

    fn on_attached(&self) {
        self.registry.output_attachment(self.id, true);
    }

    fn on_detached(&self) {
        self.registry.output_attachment(self.id, false);
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

    fn capture<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        frame: &'a AudioFrame,
    ) -> lk_agents_abi::BoxFuture<'a, io::Result<()>> {
        Box::pin(async move {
            // The case reports on a segment from its first frame, so it gets its identity now.
            let generation = frame.data.as_chunks::<2>().0.iter().find_map(|s| {
                match Tag::decode(i16::from_le_bytes(*s)) {
                    Tag::Generated { generation, .. } => Some(generation),
                    _ => None,
                }
            });
            if self.playout.captured(Duration::from_secs_f64(frame.duration()), generation) {
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
        })
    }

    fn flush_segment(&self) {
        if let Some((segment, captured)) = self.playout.flushed() {
            let ms = captured.as_millis() as u64;
            self.registry.record_sink(Sink::Flush { output: self.id, segment, ms });
        }
    }

    fn clear_buffer(&self) {
        let segments = self.playout.cleared();
        self.registry.record_sink(Sink::Clear { output: self.id, segments });
    }
}

pub struct FakeTranscriptionOutput {
    id: u32,
    registry: Arc<Registry>,
    raises: bool,
}

impl FakeTranscriptionOutput {
    pub fn new(registry: Arc<Registry>, raises: bool) -> Arc<Self> {
        let id = registry.new_output(false);
        Arc::new(FakeTranscriptionOutput { id, registry, raises })
    }
}

impl Drop for FakeTranscriptionOutput {
    fn drop(&mut self) {
        self.registry.mark_closed(super::Slot { kind: Kind::Output, nth: self.id });
    }
}

impl TranscriptionOutput for FakeTranscriptionOutput {
    fn label(&self) -> &str {
        "fake::transcription_output"
    }

    fn on_attached(&self) {
        self.registry.output_attachment(self.id, true);
    }

    fn on_detached(&self) {
        self.registry.output_attachment(self.id, false);
    }

    fn capture_text<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, io::Result<()>> {
        Box::pin(async move {
            if self.raises {
                return Err(io::IoError::Rtc("the host transcription output raised".into()));
            }
            self.registry.record_sink(Sink::Text { output: self.id, text: text.to_string() });
            Ok(())
        })
    }

    fn flush(&self) {
        self.registry.record_sink(Sink::TextFlush { output: self.id });
    }
}

/// A background sound from a host stream (`AudioConfig(source=async_iterator)`): `frames` 20 ms
/// frames of tone, then its end.
pub struct FakeSoundStream {
    left: Mutex<u32>,
}

impl FakeSoundStream {
    pub fn sound(frames: u32) -> lk_agents_abi::voice::background_audio::Sound {
        let stream = Arc::new(FakeSoundStream { left: Mutex::new(frames) });
        lk_agents_abi::voice::background_audio::Sound {
            source: lk_agents_abi::voice::background_audio::SoundSource::Stream(stream),
            volume: 0.8,
            probability: 1.0,
            fade_in: 0.1,
            fade_out: 0.1,
        }
    }
}

impl lk_agents_abi::voice::background_audio::FrameStream for FakeSoundStream {
    fn next(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<AudioFrame>, lk_agents_abi::errors::ProviderError>,
    > {
        let mut left = self.left.lock().unwrap();
        let samples = super::SAMPLE_RATE / 50;
        let frame = (*left > 0).then(|| {
            *left -= 1;
            AudioFrame::new(
                1000i16.to_le_bytes().repeat(samples as usize),
                super::SAMPLE_RATE,
                1,
                samples,
            )
        });
        Box::pin(std::future::ready(Ok(frame)))
    }
}

/// `random.random()` for the background player: always the middle, so a seed picks one sound.
pub const HALF: f64 = 0.5;
