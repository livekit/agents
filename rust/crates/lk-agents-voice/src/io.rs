//! The voice core's own IO nodes over the host traits in `lk_agents_abi::voice::io`: the proxy
//! beneath a wrapper chain, and the text and audio nodes a host hands in.

use async_trait::async_trait;
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputCapabilities, AudioOutputExt, IoError, PlaybackTracker, Result,
    TextOutput,
};
use lk_agents_abi::{AudioFrame, TimedString};
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
#[lk_agents_abi::host(proxy)]
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

/// `Agent.realtime_audio_output_node`: one realtime generation's audio, on its way to the speaker
/// through the leg the node was opened over (a node per generation, so its state never carries
/// over and it plays into the output of its own step). Frames are pushed with
/// `call_soon_threadsafe`, not a round trip.
#[lk_agents_abi::host(proxy(name = HostRealtimeAudioNode))]
#[async_trait::async_trait]
pub trait RealtimeAudioNode: Send + Sync {
    /// One frame in; returns as soon as it is queued.
    #[host(soon = "push")]
    async fn push(&self, frame: &lk_agents_abi::AudioFrame);

    /// Interrupted: what the node still yields is dropped (the leg below was cleared).
    #[host(soon = "clear")]
    fn clear(&self);

    /// The generation is over: no more frames come in.
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
