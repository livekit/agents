//! The voice core's own IO nodes over the host traits in `lk_agents_abi::voice::io`: the proxy
//! beneath a wrapper chain, the text and audio nodes a host hands in, and the driver's input legs.

use async_trait::async_trait;
use lk_agents_abi::resources::ResourceId;
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputCapabilities, AudioOutputExt, IoError, PlaybackTracker, Result,
    TextOutput, VideoInput,
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

/// `voice/io.py`'s `AgentInput` as the decision core holds it: the sink installed on each audio
/// channel, the gate's last verdict and the input switch (all three told to the data plane), the
/// camera with the task pumping it, and the STT and VAD streams being reopened for a new agent.
pub(crate) struct AgentInput {
    /// What this core installed on each channel; the data plane holds the same.
    pub(crate) sinks: std::collections::BTreeMap<crate::audio::AudioChannel, ResourceId>,
    /// The STT stream whose items the loop consumes: retired by a replacement or a removal, not
    /// by the stream's own end.
    pub(crate) live_stt: Option<ResourceId>,
    /// The verdict last sent to the data plane; `None` before the first, so it is an edge.
    pub(crate) gate_discard: Option<bool>,
    /// `input.set_audio_enabled(..)`.
    pub(crate) audio_enabled: bool,
    /// When audio started flowing from the current source, as the data plane reported it.
    pub(crate) first_input_at: Option<Ts>,
    /// Where a leg's drain reports its end.
    pub(crate) ended: crate::agent_session::Tx<crate::audio::ChannelEnd>,
    video_forwarder_parts: crate::audio::VideoForwarderParts,
    video_forwarder: Option<lk_agents_abi::AbortHandle>,
    pending_video_in: Option<Option<Arc<dyn VideoInput>>>,
    /// So the same leg handed back is a no-op.
    video_in: Option<Arc<dyn VideoInput>>,
    /// Shared: its rates are public attributes a caller may change mid-call.
    video_sampler: Option<Arc<crate::room_io::_input::VideoSampler>>,
    video_sink_warned: bool,
    next_open: u64,
    /// A `StreamOpened` for any other open is discarded (0: none).
    stt_open: u64,
    vad_open: u64,
    stt_open_same: bool,
}

impl AgentInput {
    /// The legs as the session opened them: the installed sinks, the camera with its pump (if one
    /// was attached), nothing pending.
    pub(crate) fn new(
        sinks: std::collections::BTreeMap<crate::audio::AudioChannel, ResourceId>,
        ended: crate::agent_session::Tx<crate::audio::ChannelEnd>,
        (video_forwarder_parts, video_forwarder, video_in): (
            crate::audio::VideoForwarderParts,
            Option<lk_agents_abi::AbortHandle>,
            Option<Arc<dyn VideoInput>>,
        ),
        video_sampler: Option<Arc<crate::room_io::_input::VideoSampler>>,
    ) -> Self {
        AgentInput {
            live_stt: sinks.get(&crate::audio::AudioChannel::Stt).copied(),
            sinks,
            gate_discard: None,
            audio_enabled: true,
            first_input_at: None,
            ended,
            video_forwarder_parts,
            video_forwarder,
            pending_video_in: None,
            video_in,
            video_sampler,
            video_sink_warned: false,
            next_open: 0,
            stt_open: 0,
            vad_open: 0,
            stt_open_same: false,
        }
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
}
