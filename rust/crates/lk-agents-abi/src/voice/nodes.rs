//! An agent's overridden nodes, as the core opens them. Only the binding's own agent nodes
//! (`_core.AgentNodes`) implement them: a host object is never a node factory itself.

use std::sync::Arc;

use crate::errors::{HostError, ProviderError};
use crate::voice::io::TextOutput;
use crate::{
    AudioFrame, ChatRequest, LlmStream, SttSink, SttSource, TimedString, TtsSink, TtsSource,
};

/// An agent's overridden nodes, as the core calls them. `transcription_node` and
/// `realtime_audio_output_node` are opened per segment and per realtime generation over the core
/// leg below (what one raises comes back as the exception it raised). `llm_node`, `tts_node` and
/// `stt_node` are opened per request in place of the provider: one completion, one segment's
/// synthesis, one recognition stream.
#[crate::host(native(refused = "expected an agent's nodes (`_core.AgentNodes`)"))]
pub trait NodeFactory: Send + Sync {
    fn text_node(&self, below: Arc<dyn TextOutput>) -> Result<Arc<dyn TextNode>, HostError>;
    fn audio_node(
        &self,
        below: Arc<dyn crate::voice::io::AudioOutput>,
    ) -> Result<Arc<dyn RealtimeAudioNode>, HostError>;
    /// `llm_node(chat_ctx, tools, model_settings)`: its chunks, strings and flushes as chunks.
    fn llm_node<'a>(
        &'a self,
        req: &'a ChatRequest,
    ) -> crate::BoxFuture<'a, Result<LlmStream, ProviderError>>;
    /// `tts_node(text, model_settings)`: text pushed in, its frames out.
    fn tts_node(&self) -> crate::BoxFuture<'_, Result<(TtsSink, TtsSource), ProviderError>>;
    /// `stt_node(audio, model_settings)`: frames pushed in, its speech events out.
    fn stt_node(&self) -> crate::BoxFuture<'_, Result<(SttSink, SttSource), ProviderError>>;
}

// Text is handed to the node on its loop and never awaited: a `transcription_node` may hold text
// back arbitrarily long, and awaiting it per delta would pace the LLM leg by the node.

/// A transcription node: pushes are fire-and-forget.
pub trait TextNode: Send + Sync {
    fn capture_text(&self, text: &str);
    fn capture_timed(&self, text: &TimedString);
    fn flush(&self);
}

/// `Agent.realtime_audio_output_node`: one realtime generation's audio, on its way to the speaker
/// through the leg the node was opened over (a node per generation, so its state never carries
/// over and it plays into the output of its own step).
pub trait RealtimeAudioNode: Send + Sync {
    /// One frame in; returns as soon as it is queued.
    fn push<'a>(&'a self, frame: &'a AudioFrame) -> crate::BoxFuture<'a, ()>;

    /// Interrupted: what the node still yields is dropped (the leg below was cleared).
    fn clear(&self);

    /// The generation is over: no more frames come in.
    fn end(&self);
}
