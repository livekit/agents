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

use lk_agents_abi::voice::io;
use lk_agents_abi::voice::io::{
    AudioInput, AudioOutput, AudioOutputCapabilities, PlaybackFinished, PlaybackTracker,
    TextOutput, VideoInput,
};
use lk_agents_abi::{AudioFrame, MediaRef, Ts, errors::RefRegistry};
use lk_agents_abi::errors::HostError;
use lk_agents_media::recorder::{AudioRecorder, RecorderError};
use lk_agents_voice::remote_session::{self, FrameDecoder, console::ConsoleIo, proto};
use prost::Message as _;
use proto::agent_session_message::{Message, console_io};

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

/// The person's camera track as a resource the case feeds by `VideoFrames`.
pub fn room_camera(
    registry: &Registry,
    camera: Arc<lk_agents_rtc::FakeTrackPublication>,
    refs: Arc<dyn RefRegistry>,
) {
    registry.register(Kind::Video, Detail::RoomCamera, Control::RoomVideo(camera, refs));
}

/// So many frames onto the person's camera track, minted as the host camera's are. Accepted
/// when a reader took them.
pub(super) fn feed_room_video(
    camera: &lk_agents_rtc::FakeTrackPublication,
    refs: &Arc<dyn RefRegistry>,
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
        let frame = lk_agents_rtc::VideoFrameRef {
            media: MediaRef::new(ordinal, refs),
            width: 1280,
            height: 720,
            timestamp_us: (ordinal & 0xffff_ffff) as i64 * 33_000,
        };
        read |= camera.push_video_frame(frame) > 0;
    }
    read
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
    /// A host source: a cut is reported a few scheduler hops after the clear, as the report of
    /// one crosses back from the host's own loop.
    lagging: bool,
}

/// Hops a lagging output takes to report a cut: more than the driver yields for it.
const LAGGING_REPORT_HOPS: usize = 64;

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

    /// The output threw its playback count away: every segment still outstanding is reported
    /// interrupted, then the tracker is reset. Finished first, as the reference's only resetting
    /// sink does (`DataStreamAudioOutput._clear_buffer_task._on_timeout`): a reset alone moves no
    /// count above this output, and a wrapper waiting on the segment would wait forever.
    /// Returns the segments that were cut.
    pub(super) fn reset(&self) -> Vec<u32> {
        let cut = self.cleared();
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
        let finished: Vec<PlaybackFinished> = cut
            .iter()
            .map(|(_, position)| PlaybackFinished {
                interrupted: true,
                playback_position: *position,
                ..Default::default()
            })
            .collect();
        if self.lagging {
            let tracker = self.tracker.clone();
            tokio::spawn(async move {
                for _ in 0..LAGGING_REPORT_HOPS {
                    tokio::task::yield_now().await;
                }
                for ev in finished {
                    tracker.playback_finished(ev);
                }
            });
        } else {
            for ev in finished {
                self.tracker.playback_finished(ev);
            }
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
    pub fn new(registry: Arc<Registry>, pausable: bool, lagging: bool) -> Arc<Self> {
        let playout =
            Arc::new(Playout { tracker: Arc::default(), state: Mutex::default(), lagging });
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

/// `lk agent console`'s end of the socket, in process: what it sends the agent (the microphone,
/// requests, playout acks) goes out in order, and every segment the agent flushes is acked once
/// its audio has played on the clock, unless a clear cut it first. Its tasks are the CLI's, not
/// the session's, so nothing judges them.
#[derive(Clone)]
pub struct FakeCli {
    to_agent: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
    /// When the last flushed segment finishes playing, on the clock: its ack is owed until then.
    playing_until: Arc<std::sync::Mutex<Option<tokio::time::Instant>>>,
}

impl FakeCli {
    /// The session's console over an in-process stream, and the CLI on its other end.
    pub fn connect(spawner: Arc<dyn lk_agents_abi::Spawner>) -> (Arc<ConsoleIo>, FakeCli) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (agent_end, cli_end) = tokio::io::duplex(1 << 20);
        let (mut read, mut write) = tokio::io::split(cli_end);
        let (to_agent, mut outgoing) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();
        tokio::spawn(async move {
            while let Some(bytes) = outgoing.recv().await {
                if write.write_all(&bytes).await.is_err() {
                    return;
                }
            }
        });
        let cli = FakeCli { to_agent, playing_until: Arc::default() };
        let speaker = cli.clone();
        tokio::spawn(async move {
            let (mut decoder, mut chunk) = (FrameDecoder::new(), vec![0u8; 16 * 1024]);
            // The audio of the segment being received, and which segment it is: a clear moves
            // on, so an ack still waiting for a cut segment's audio is dropped.
            let (mut pending, mut segment) = (Duration::ZERO, 0u64);
            let played = Arc::new(std::sync::atomic::AtomicU64::new(0));
            loop {
                while let Ok(Some(body)) = decoder.next_frame() {
                    match proto::AgentSessionMessage::decode(body.as_slice()).ok().and_then(|m| m.message) {
                        Some(Message::AudioOutput(f)) => {
                            pending += Duration::from_secs_f64(
                                f64::from(f.samples_per_channel) / f64::from(f.sample_rate.max(1)),
                            );
                        }
                        Some(Message::AudioPlaybackFlush(_)) => {
                            let (speaker, played, this) = (speaker.clone(), played.clone(), segment);
                            let audio = std::mem::take(&mut pending);
                            let until = tokio::time::Instant::now() + audio;
                            let mut playing = speaker.playing_until.lock().unwrap();
                            *playing = Some(playing.map_or(until, |at| at.max(until)));
                            drop(playing);
                            tokio::spawn(async move {
                                tokio::time::sleep(audio).await;
                                if played.load(std::sync::atomic::Ordering::Relaxed) == this {
                                    speaker.send(Message::AudioPlaybackFinished(
                                        console_io::AudioPlaybackFinished {},
                                    ));
                                }
                            });
                        }
                        Some(Message::AudioPlaybackClear(_)) => {
                            pending = Duration::ZERO;
                            segment += 1;
                            played.store(segment, std::sync::atomic::Ordering::Relaxed);
                        }
                        _ => {}
                    }
                }
                match read.read(&mut chunk).await {
                    Ok(0) | Err(_) => return,
                    Ok(n) => decoder.push(&chunk[..n]),
                }
            }
        });
        (Arc::new(ConsoleIo::over(agent_end, spawner)), cli)
    }

    /// The agent's voice the speaker has yet to play: an ack still owed on the clock.
    pub fn queued(&self) -> Duration {
        let until = *self.playing_until.lock().unwrap();
        until.map_or(Duration::ZERO, |at| at.saturating_duration_since(tokio::time::Instant::now()))
    }

    /// One message to the agent; `false` once the CLI's socket is gone.
    pub fn send(&self, message: Message) -> bool {
        let message = proto::AgentSessionMessage { message: Some(message) };
        self.to_agent.send(remote_session::frame(&message.encode_to_vec())).is_ok()
    }
}

/// What the CLI asks of the session (`ConsoleCall`'s `_core_*` port), answered as the host would
/// from its own state. Toggling the legs stays with the case's own inputs, so `update_io` is
/// acknowledged and changes nothing.
pub struct FakeConsolePort;

#[async_trait::async_trait]
impl remote_session::SessionPort for FakeConsolePort {
    async fn chat_history(&self) -> Result<Vec<lk_agents_abi::ChatItem>, HostError> {
        Ok(Vec::new())
    }
    async fn agent_info(&self) -> Result<remote_session::AgentInfo, HostError> {
        Ok(("agent".into(), Some("You are a test agent.".into()), Vec::new(), Vec::new()))
    }
    async fn run_input(&self, text: &str) -> Result<remote_session::RunOutcome, HostError> {
        Ok(if text.is_empty() { (None, Some("empty run_input text".into())) } else { (Some(Vec::new()), None) })
    }
    async fn session_state(&self) -> Result<remote_session::SessionState, HostError> {
        let options = remote_session::STATE_OPTION_FIELDS
            .iter()
            .map(|(key, _)| ((*key).to_owned(), "None".to_owned()))
            .collect();
        Ok(("listening".into(), "listening".into(), "agent".into(), options, Some(0.0)))
    }
    async fn rtc_stats(&self) -> Result<Option<remote_session::RtcStats>, HostError> {
        Ok(None)
    }
    async fn session_usage(
        &self,
    ) -> Result<Vec<lk_agents_abi::metrics::usage::ModelUsage>, HostError> {
        Ok(Vec::new())
    }
    fn framework_info(&self) -> (String, String) {
        ("rust".into(), "fuzz".into())
    }
    async fn update_io(&self, _update: remote_session::IoUpdate) -> Result<(), HostError> {
        Ok(())
    }
    async fn finalize_simulation(
        &self,
        _provisional: remote_session::SimulationVerdict,
    ) -> Result<remote_session::VerdictOutcome, HostError> {
        Ok((None, None))
    }
}

/// A background sound from a host stream (`AudioConfig(source=async_iterator)`): so many 20 ms
/// frames of tone, then its end.
pub struct FakeSoundStream {
    left: Mutex<u32>,
}

impl FakeSoundStream {
    pub fn sound(frames: u32) -> lk_agents_voice::background_audio::Sound {
        let stream = Arc::new(FakeSoundStream { left: Mutex::new(frames) });
        lk_agents_voice::background_audio::Sound::for_host(None, Some(stream), 0.8, 1.0, 0.1, 0.1)
    }
}

#[async_trait::async_trait]
impl lk_agents_voice::background_audio::FrameStream for FakeSoundStream {
    async fn next(&self) -> Result<Option<AudioFrame>, lk_agents_abi::errors::ProviderError> {
        let mut left = self.left.lock().unwrap();
        if *left == 0 {
            return Ok(None);
        }
        *left -= 1;
        let samples = super::SAMPLE_RATE / 50;
        Ok(Some(AudioFrame::new(1000i16.to_le_bytes().repeat(samples as usize), super::SAMPLE_RATE, 1, samples)))
    }
    fn cancel_pending(&self) {}
    async fn close(&self) {
        *self.left.lock().unwrap() = 0;
    }
}

/// `random.random` for the background player: always the middle, so a seed picks one sound.
pub struct Half;

impl lk_agents_voice::background_audio::RandomDraw for Half {
    fn draw(&self) -> f64 {
        0.5
    }
}
