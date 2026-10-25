//! The voice core's own IO nodes over the host traits in `lk_agents_abi::voice::io`: the proxy
//! beneath a wrapper chain, the text and audio nodes a host hands in, and the driver's input legs.

use async_trait::async_trait;
use lk_agents_abi::voice::io::{
    AudioInput, AudioOutput, AudioOutputCapabilities, AudioOutputExt, IoError, PlaybackTracker,
    Result, TextOutput, VideoInput,
};
use lk_agents_abi::{AudioFrame, TimedString, Ts};
use std::sync::Arc;

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
