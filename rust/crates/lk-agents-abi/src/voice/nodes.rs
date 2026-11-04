//! The session's registered nodes, as the core opens them over the host's `_SessionNodes`
//! (`_core.AgentNodes`), and the default `llm_node`, `tts_node` and `stt_node` the host's nodes
//! may call.

use std::any::Any;
use std::sync::Arc;

use crate::errors::{HostError, ProviderError};
use crate::invoke::CallContext;
use crate::types::connect::ApiConnectOptionsError;
use crate::utils::stream::StreamOutput;
use crate::voice::io::TextOutput;
use crate::{
    ApiConnectOptions, AudioFrame, ChatItem, ChatRequest, HostRef, Llm, LlmChunk, LlmStream,
    SpeechEvent, Stt, SttSink, SttSource, SynthesizedAudio, ToolChoice, ToolSpec, TranscriptChunk,
    Tts, TtsSink, TtsSource, Vad,
};

/// The host's `_SessionNodes`: the `llm_node`, `tts_node`, `stt_node` and
/// `transcription_node` registered on the session, each call in a copy of the binding of
/// the agent holding the call. Whatever a node returns (an async iterable, an awaitable of one, one value or `None`)
/// reaches the core as an async iterable.
#[crate::host(proxy(host = "livekit.agents.voice.agent_session._SessionNodes"))]
pub trait SessionNodes: Send + Sync {
    /// `trace`: the step's `llm_node` span current, so application spans nest under it.
    /// `calls` carries the step's trace to the default `llm_node` the node runs (`ProviderCalls`).
    #[host(await)]
    fn llm<'a>(
        &'a self,
        req: &'a ChatRequest,
        calls: Arc<dyn ProviderCalls>,
        #[host(context)] trace: crate::telemetry::record::TraceContext,
    ) -> crate::BoxFuture<'a, Result<Box<dyn crate::ProviderStream<NodeChunk>>, ProviderError>>;

    /// `calls` meters the default `tts_node` the node runs (`ProviderCalls`).
    #[host(await)]
    fn tts<'a>(
        &'a self,
        #[host(source(class = "NodeText"))] text: Box<dyn StreamOutput<String>>,
        calls: Arc<dyn ProviderCalls>,
        #[host(context)] trace: crate::telemetry::record::TraceContext,
    ) -> crate::BoxFuture<'a, Result<Box<dyn crate::ProviderStream<SynthesizedAudio>>, ProviderError>>;

    /// `calls` hears the connections of the default `stt_node` the node runs (`ProviderCalls`).
    #[host(await)]
    fn stt<'a>(
        &'a self,
        #[host(source(class = "NodeAudio"))] audio: Box<dyn StreamOutput<AudioFrame>>,
        calls: Arc<dyn ProviderCalls>,
        #[host(context)] trace: crate::telemetry::record::TraceContext,
    ) -> crate::BoxFuture<'a, Result<Box<dyn crate::ProviderStream<SpeechEvent>>, ProviderError>>;

    #[host(await)]
    fn transcription<'a>(
        &'a self,
        #[host(source(class = "NodeTranscript"))] text: Box<dyn StreamOutput<TranscriptChunk>>,
    ) -> crate::BoxFuture<'a, Result<Box<dyn StreamOutput<TranscriptChunk>>, ProviderError>>;
}

/// What an `llm_node` yields: text it answered itself (spoken, never metered as a model call), a
/// `FlushSentinel` (the spoken segment ends here), or a model's chunk.
#[crate::host(record(module = "agent_nodes", extract, transparent))]
#[derive(Debug, Clone)]
pub enum NodeChunk {
    Text(String),
    #[host(host_type = "livekit.agents.types.FlushSentinel")]
    Flush,
    Chunk(LlmChunk),
}

/// `_core.AgentNodes(host)`: the session's registered nodes over the host's `_SessionNodes`, taken
/// in the binding of the agent they run for.
#[crate::host(class(name = "AgentNodes", native = NodeFactory))]
pub trait AgentNodes: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(env: &Self::Env, host: Arc<dyn SessionNodes>) -> Self;
}

/// The provider calls of the default node a registered node runs: metered for the speech in place
/// of the node's output, as the reference meters the TTS's own streams, the connections they
/// acquired, which the node's output does not carry, and the step's trace (its job and
/// conversation), which the default node's request records for.
#[crate::host(native(refused = "expected the calls of a registered tts_node"))]
pub trait ProviderCalls: Any + Send + Sync {}

#[crate::host(class(name = "_core_NodeCalls", native = ProviderCalls))]
pub trait NodeCalls: Send + Sync + Sized + 'static {}

/// A frame the default `tts_node` yields, with what the reference stamps on its `userdata`: the
/// words it covers and when its synthesis started (wall-clock seconds).
#[crate::host(record(module = "agent_nodes", emit))]
#[derive(Debug, Clone)]
pub struct NodeFrame {
    pub frame: AudioFrame,
    pub timed_transcript: Vec<TranscriptChunk>,
    pub started_time: Option<f64>,
}

/// The default `llm_node`'s chunks.
#[crate::host(class(name = "LlmNodeChunks"))]
pub trait LlmNodeChunks: Send + Sync + Sized + 'static {
    #[host(next)]
    async fn next(&self) -> Result<Option<LlmChunk>, ProviderError>;
}

/// The default `tts_node`'s frames.
#[crate::host(class(name = "TtsNodeFrames"))]
pub trait TtsNodeFrames: Send + Sync + Sized + 'static {
    #[host(next)]
    async fn next(&self) -> Result<Option<NodeFrame>, ProviderError>;
}

/// The default `stt_node`'s speech events.
#[crate::host(class(name = "SttNodeEvents"))]
pub trait SttNodeEvents: Send + Sync + Sized + 'static {
    #[host(next)]
    async fn next(&self) -> Result<Option<SpeechEvent>, ProviderError>;
}

/// `session.default.llm_node`, `tts_node` and `stt_node`: the session's provider over the node's
/// input (an LLM retried per `conn_options`, a TTS that does not stream fed sentence by sentence,
/// an STT that does not stream over the VAD). The LLM and TTS nodes open their `llm_request` or
/// `tts_request` under the span current at the call.
#[crate::host(functions)]
pub trait DefaultNodes {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(class(name = "LlmNodeChunks"))]
    type Chunks;
    #[host(class(name = "TtsNodeFrames"))]
    type Frames;
    #[host(class(name = "SttNodeEvents"))]
    type Events;

    /// A completion of `chat_ctx` (its items, and the host's `llm.ChatContext` a host LLM is
    /// handed) offering `tools` (their specs, and the host's `list[Tool]`). `conn_options`: the
    /// session's, else the defaults. `calls`: the running agent node's, when it runs this one.
    #[host(function)]
    #[allow(clippy::too_many_arguments, reason = "Preserve the public host operation signature")]
    fn llm_node(
        env: &Self::Env,
        llm: Arc<dyn Llm>,
        chat_ctx: Vec<ChatItem>,
        chat_ctx_host: HostRef,
        #[host(kw)] tools: Vec<ToolSpec>,
        #[host(default = None)] tools_host: Option<HostRef>,
        #[host(default = None)] tool_choice: Option<ToolChoice>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
        #[host(default = None)] calls: Option<Arc<dyn ProviderCalls>>,
        #[host(context)] context: CallContext,
    ) -> Result<Self::Chunks, ApiConnectOptionsError>;

    /// `conn_options`: the node's, else the provider's own policy (models.md), else the
    /// defaults. `calls`: the running agent node's, when it runs this one.
    #[host(function)]
    fn tts_node(
        env: &Self::Env,
        tts: Arc<dyn Tts>,
        text: Box<dyn StreamOutput<String>>,
        #[host(kw, default = None)] conn_options: Option<ApiConnectOptions>,
        #[host(default = None)] calls: Option<Arc<dyn ProviderCalls>>,
        #[host(context)] context: CallContext,
    ) -> Result<Self::Frames, ApiConnectOptionsError>;

    #[host(function)]
    fn stt_node(
        env: &Self::Env,
        stt: Arc<dyn Stt>,
        audio: Box<dyn StreamOutput<AudioFrame>>,
        #[host(kw, default = None)] conn_options: Option<ApiConnectOptions>,
        #[host(default = None)] vad: Option<Arc<dyn Vad>>,
        #[host(default = None)] calls: Option<Arc<dyn ProviderCalls>>,
    ) -> Result<Self::Events, ApiConnectOptionsError>;
}

/// The session's registered nodes, as the core calls them. `transcription_node` is
/// opened per segment over the core leg below (what one raises comes back as the exception
/// it raised). `llm_node`, `tts_node` and
/// `stt_node` are opened per request in place of the provider: one completion, one segment's
/// synthesis, one recognition stream.
#[crate::host(native(
    refused = "expected the session's nodes (`_core.AgentNodes`)",
    class = "AgentNodes"
))]
pub trait NodeFactory: Send + Sync {
    fn text_node(&self, below: Arc<dyn TextOutput>) -> Result<Arc<dyn TextNode>, HostError>;
    /// `llm_node(chat_ctx, tools, model_settings)`: its chunks, strings and flushes as chunks; it
    /// runs under `req.trace`, the step's `llm_node` span.
    fn llm_node<'a>(
        &'a self,
        req: &'a ChatRequest,
    ) -> crate::BoxFuture<'a, Result<LlmStream, ProviderError>>;
    /// `tts_node(text, model_settings)`: text pushed in, its frames out. `expressive`: the
    /// synthesis carries expressive markup. It runs under `trace`, the step's `tts_node` span.
    fn tts_node(
        &self,
        expressive: bool,
        trace: crate::telemetry::record::TraceContext,
    ) -> crate::BoxFuture<'_, Result<crate::DuplexStream<TtsSink, TtsSource>, ProviderError>>;
    /// `stt_node(audio, model_settings)`: frames pushed in, its speech events out. It runs under
    /// `trace`'s span.
    fn stt_node(
        &self,
        trace: crate::telemetry::record::TraceContext,
    ) -> crate::BoxFuture<'_, Result<crate::DuplexStream<SttSink, SttSource>, ProviderError>>;
}

// Text is handed to the node on its loop and never awaited: a `transcription_node` may hold text
// back arbitrarily long, and awaiting it per delta would pace the LLM leg by the node.

/// A transcription node: pushes queue input; draining settles its flushed segment.
pub trait TextNode: Send + Sync {
    fn capture_text(&self, text: &str);
    fn capture_timed(&self, text: &TranscriptChunk);
    fn flush(&self);
    /// Retain the binding's owner until the same drain receipt is complete.
    fn retain_until_drained(&self, owner: Box<dyn Send>);
    /// Stop this segment's processing and settle its drain after owned cleanup.
    fn cancel(&self);
    /// The flushed segment's output has been delivered, or processing failed.
    fn drain(&self) -> crate::BoxFuture<'_, crate::voice::io::Result<()>>;
}
