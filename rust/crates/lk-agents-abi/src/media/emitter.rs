//! The TTS emitter's host class, its errors and its output port; framing (lk-agents-media),
//! decoding and the drift timer (lk-agents-inference) stay native.

use std::sync::Arc;

use crate::errors::HostError;
use crate::usage::TokenUsage;
use crate::{AudioFrame, TranscriptChunk};

/// Where a synthesis' audio goes: `tts.py`'s `_EmitterOutput` over the host channel. Its calls
/// queue in order and run on the host loop, in the context the emitter was built in.
#[crate::host(proxy(
    host = "livekit.agents.utils.aio.Chan[livekit.agents.tts.SynthesizedAudio]",
    view = "livekit.agents.tts._EmitterOutput"
))]
pub trait EmitterOutput: Send + Sync {
    /// One `SynthesizedAudio`; its frame's userdata carries the words it covers.
    #[host(notify)]
    #[allow(clippy::too_many_arguments, reason = "Preserve the host notification signature")]
    fn send(
        &self,
        frame: AudioFrame,
        request_id: String,
        segment_id: String,
        is_final: bool,
        timed_transcript: Vec<TranscriptChunk>,
        usage: Option<TokenUsage>,
        characters_count: Option<u64>,
        provider_request_ids: Vec<String>,
    );
    #[host(call)]
    fn closed(&self) -> Result<bool, HostError>;
}

#[crate::host(class(name = "AudioEmitter", module = "livekit.agents.tts", subclass))]
pub trait AudioEmitter: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        #[host(kw)] label: String,
        dst_ch: Arc<dyn EmitterOutput>,
    ) -> Result<Self, EmitterError>;
    #[host(call)]
    fn initialize(
        &self,
        #[host(kw)] request_id: String,
        sample_rate: u32,
        num_channels: u16,
        mime_type: String,
        #[host(default = 200)] frame_size_ms: u32,
        #[host(default = false)] stream: bool,
    ) -> Result<(), EmitterError>;
    #[host(call)]
    fn start_segment(&self, #[host(kw)] segment_id: String) -> Result<(), EmitterError>;
    #[host(call)]
    fn end_segment(&self) -> Result<(), EmitterError>;
    #[host(call)]
    fn push(&self, data: &[u8]) -> Result<(), EmitterError>;
    #[host(call)]
    fn push_frame(&self, frame: AudioFrame) -> Result<(), EmitterError>;
    #[host(call)]
    fn push_timed_transcript(
        &self,
        #[host(one_or_many)] delta_text: Vec<TranscriptChunk>,
    ) -> Result<(), EmitterError>;
    /// What the provider reported it billed for the open segment, or the next one when none is
    /// open, carried on the segment's final frame. Reports add up, one per provider request;
    /// providers that report nothing keep the framework's own measure (characters sent, audio
    /// received).
    #[host(call)]
    fn report_usage(
        &self,
        #[host(default = None)] usage: Option<TokenUsage>,
        #[host(kw, default = None)] characters_count: Option<u64>,
    ) -> Result<(), EmitterError>;
    #[host(call)]
    fn flush(&self) -> Result<(), EmitterError>;
    #[host(call)]
    fn end_input(&self) -> Result<(), EmitterError>;
    /// Waits for compressed input to finish decoding; reports a decode failure.
    #[host(await)]
    async fn join(&self) -> Result<(), EmitterError>;
    /// Discards pending audio and observes native settlement; never reports a synthesis failure.
    #[host(await)]
    async fn aclose(&self) -> Result<(), EmitterError>;
    #[host(call)]
    fn pushed_duration(&self, #[host(default = -1)] idx: i64) -> f64;
    #[host(getter)]
    fn num_segments(&self) -> usize;
    #[host(call)]
    fn _note_provider_request_id(&self, context_id: String) -> Result<(), EmitterError>;
    #[host(getter)]
    fn output_closed(&self) -> Result<bool, EmitterError>;
}

/// Every misuse the reference raises `RuntimeError` for. Messages are verbatim: plugin tests
/// assert on them.
#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EmitterError {
    /// A mime type nothing in this build decodes, refused before any byte is framed as PCM.
    #[error("AudioEmitter: {0}")]
    #[host(kind = "NotImplementedError")]
    UnsupportedMime(String),
    #[error("AudioEmitter: decode failed: {0}")]
    Decode(String),
    /// The runtime or the host output failed underneath the emitter.
    #[error("AudioEmitter: {0}")]
    Unavailable(String),
    #[error("AudioEmitter already started")]
    AlreadyStarted,
    #[error("AudioEmitter isn't started")]
    NotStarted,
    #[error(
        "start_segment() can only be called when SynthesizeStream is initialized with stream=True"
    )]
    StartSegmentNotStreaming,
    #[error(
        "end_segment() can only be called when SynthesizeStream is initialized with stream=True"
    )]
    EndSegmentNotStreaming,
    #[error("start_segment() called before the previous segment was ended")]
    SegmentAlreadyOpen,
    #[error("start_segment() must be called before pushing audio data")]
    NoSegment,
}
