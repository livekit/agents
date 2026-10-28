//! The TTS emitter's host class and its output port; framing, decoding and the drift timer stay
//! native.

use std::sync::Arc;

use crate::errors::HostError;
use crate::tts::emitter::EmitterError;
use crate::{AudioFrame, TimedString};

/// Where a synthesis' audio goes: `tts.py`'s `_EmitterOutput` over the host channel. Its calls
/// queue in order and run on the host loop, in the context the emitter was built in.
#[crate::host(proxy(
    host = "livekit.agents.utils.aio.Chan",
    view = "livekit.agents.tts._EmitterOutput"
))]
pub trait EmitterOutput: Send + Sync {
    /// One `SynthesizedAudio`; its frame's userdata carries the words it covers.
    #[host(notify)]
    fn send(
        &self,
        frame: AudioFrame,
        request_id: String,
        segment_id: String,
        is_final: bool,
        timed_transcript: Vec<TimedString>,
    );
    /// An attribute of the synthesis' current span, when it is recording.
    #[host(notify)]
    fn span_attribute(&self, key: String, values: Vec<String>);
    /// `LK_DUMP_TTS`: one segment written to `lk_dump/<name>_<local time>.wav`.
    #[host(notify)]
    fn dump(&self, name: String, wav: Vec<u8>);
    #[host(call)]
    fn closed(&self) -> Result<bool, HostError>;
}

#[crate::host(class(name = "AudioEmitter", module = "livekit.agents.tts", subclass))]
pub trait AudioEmitter: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(#[host(kw)] label: String, dst_ch: Arc<dyn EmitterOutput>)
    -> Result<Self, EmitterError>;
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
        #[host(one_or_many)] delta_text: Vec<TimedString>,
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
