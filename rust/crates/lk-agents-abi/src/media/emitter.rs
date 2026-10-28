//! The TTS emitter's host class; PCM framing and segment state stay in the native machine.

use crate::{AudioFrame, HostRef, errors::ProviderError};

#[crate::host(class(name = "AudioEmitter", module = "livekit.agents.tts", subclass))]
pub trait AudioEmitter: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(#[host(kw)] label: String, dst_ch: HostRef) -> Result<Self, ProviderError>;
    #[host(call)]
    fn initialize(
        &self,
        #[host(kw)] request_id: String,
        sample_rate: u32,
        num_channels: u16,
        mime_type: String,
        #[host(default = 200)] frame_size_ms: u32,
        #[host(default = false)] stream: bool,
    ) -> Result<(), ProviderError>;
    #[host(call)]
    fn start_segment(&self, #[host(kw)] segment_id: String) -> Result<(), ProviderError>;
    #[host(call)]
    fn end_segment(&self) -> Result<(), ProviderError>;
    #[host(call)]
    fn push(&self, data: HostRef) -> Result<(), ProviderError>;
    #[host(call)]
    fn push_frame(&self, frame: AudioFrame) -> Result<(), ProviderError>;
    #[host(call)]
    fn push_timed_transcript(&self, delta_text: HostRef) -> Result<(), ProviderError>;
    #[host(call)]
    fn flush(&self) -> Result<(), ProviderError>;
    #[host(call)]
    fn end_input(&self) -> Result<(), ProviderError>;
    #[host(await)]
    async fn join(
        &self,
        #[host(context)] context: crate::invoke::CallContext,
    ) -> Result<(), ProviderError>;
    #[host(await)]
    async fn aclose(
        &self,
        #[host(context)] context: crate::invoke::CallContext,
    ) -> Result<(), ProviderError>;
    #[host(call)]
    fn pushed_duration(&self, #[host(default = -1)] idx: i64) -> f64;
    #[host(getter)]
    fn num_segments(&self) -> usize;
    #[host(call)]
    fn _note_provider_request_id(&self, context_id: String) -> Result<(), ProviderError>;
    #[host(getter)]
    fn output_closed(&self) -> Result<bool, ProviderError>;
}
