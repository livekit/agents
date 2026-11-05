//! `llm.DuplexRealtimeAdapter`'s engine: a duplex model run as a realtime model, its output gated
//! and segmented in `lk-agents-inference`; the plugin's model and session stay host objects.

use std::any::Any;
use std::sync::Arc;

use crate::errors::{ApiError, HostError};
use crate::llm::inference::InvalidPayload;
use crate::llm::realtime::Event;
use crate::{AudioFrame, ChatItem, ToolChoice};

/// Decides which frames of the model's output carry speech worth playing.
#[crate::host(proxy(host = "livekit.agents.llm.AudioGate"))]
pub trait AudioGate: Send + Sync {
    /// True while the frame belongs to an open burst of output.
    #[host(call)]
    fn update(&self, frame: AudioFrame) -> Result<bool, HostError>;

    /// End the open burst, keeping whatever has been learned about the model.
    #[host(call)]
    fn deactivate(&self) -> Result<(), HostError>;
}

/// Opens on output that stands out from a silence the plugin already knows.
#[crate::host(class(name = "FixedGate", native = AudioGate))]
pub trait FixedGate: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(
        silence: f64,
        #[host(kw, default = 3.0)] activation_ratio: f64,
        #[host(default = 1.8)] deactivation_ratio: f64,
        #[host(default = 0.5)] min_silence_duration: f64,
    ) -> Self;

    /// True while the frame belongs to an open burst of output.
    #[host(call)]
    fn update(&self, frame: AudioFrame) -> Result<bool, InvalidPayload>;

    /// End the open burst, keeping whatever has been learned about the model.
    #[host(call)]
    fn deactivate(&self);
}

/// Opens on output that stands out from the model's own silence: the floor is the quietest
/// `min_silence_duration` stretch it produced while not speaking, within `window`.
#[crate::host(class(name = "AdaptiveNoiseGate", native = AudioGate))]
pub trait AdaptiveNoiseGate: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(
        #[host(kw, default = 3.0)] activation_ratio: f64,
        #[host(default = 1.8)] deactivation_ratio: f64,
        #[host(default = 0.5)] min_silence_duration: f64,
        #[host(default = 10.0)] window: f64,
    ) -> Self;

    /// True while the frame belongs to an open burst of output.
    #[host(call)]
    fn update(&self, frame: AudioFrame) -> Result<bool, InvalidPayload>;

    /// End the open burst, keeping whatever has been learned about the model.
    #[host(call)]
    fn deactivate(&self);
}

/// What varies between duplex providers.
#[crate::host(record(
    module = "duplex",
    path = "livekit.agents.llm.duplex",
    name = "DuplexCapabilities",
    dataclass,
    emit,
    extract
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DuplexCapabilities {
    pub user_transcription: bool,
    pub auto_tool_reply_generation: bool,
    #[host(default = false)]
    pub mutable_chat_context: bool,
    #[host(default = false)]
    pub mutable_instructions: bool,
    #[host(default = false)]
    pub mutable_tools: bool,
}

#[crate::host(functions(pid = false))]
pub trait DuplexConfiguration {
    #[host(function)]
    fn duplex_realtime_capabilities(
        caps: DuplexCapabilities,
    ) -> crate::llm::realtime::RealtimeCapabilities;
}

/// One frame of the model's output audio; `start_ms` places it on the model's timeline.
#[crate::host(record(
    module = "duplex",
    path = "livekit.agents.llm.duplex",
    name = "DuplexAudioFrame",
    dataclass,
    emit,
    extract
))]
#[derive(Debug, Clone, PartialEq)]
pub struct DuplexAudio {
    pub frame: AudioFrame,
    #[host(default = None)]
    pub start_ms: Option<i64>,
}

/// A fragment of the model's transcript of its own speech.
#[crate::host(record(
    module = "duplex",
    path = "livekit.agents.llm.duplex",
    name = "DuplexOutputTranscriptDelta",
    dataclass,
    emit,
    extract
))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplexTranscript {
    pub text: String,
    #[host(default = None)]
    pub start_ms: Option<i64>,
    #[host(default = None)]
    pub end_ms: Option<i64>,
}

/// One event of a plugin session: its transcript, or what a realtime session reports.
#[crate::host(record(module = "duplex", extract, transparent))]
#[derive(Debug, Clone, PartialEq)]
pub enum DuplexEvent {
    Transcript(DuplexTranscript),
    Session(Event),
}

/// Caller audio the input dropped to stay current, as the seconds it covered: silence to the model.
#[crate::host(record(
    module = "duplex",
    path = "livekit.agents.llm.duplex",
    name = "DuplexInputGap",
    dataclass,
    emit
))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuplexInputGap {
    pub duration: f64,
}

/// The caller's input is muted, or unmuted, from this point of the audio on.
#[crate::host(record(
    module = "duplex",
    path = "livekit.agents.llm.duplex",
    name = "DuplexInputMute",
    dataclass,
    emit
))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DuplexInputMute {
    pub muted: bool,
}

/// What a plugin pulls from its input, in order: the caller's audio, a gap, a mute or an unmute.
#[crate::host(record(module = "duplex", emit, transparent))]
#[derive(Debug, Clone, PartialEq)]
pub enum DuplexInputItem {
    Audio(AudioFrame),
    Gap(DuplexInputGap),
    Mute(DuplexInputMute),
}

/// A duplex session's input as the engine feeds it: never blocks, and keeps the plugin on current
/// audio.
#[crate::host(native(refused = "expected a native DuplexInput"))]
pub trait DuplexInputQueue: Any + Send + Sync {
    fn push(&self, frame: AudioFrame);
    /// Pre-connect audio: kept in full.
    fn head(&self, frame: AudioFrame);
    /// Measures what the input drops under `cx`'s job.
    fn observe(&self, cx: &crate::telemetry::Cx);
}

/// `_core.DuplexInput`: a duplex session's input. The engine pushes the caller's audio, the plugin
/// pulls it; a plugin more than 250 ms behind skips to the newest audio after a gap.
#[crate::host(class(name = "DuplexInput", native = DuplexInputQueue))]
pub trait DuplexInput: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(env: &Self::Env) -> Self;

    #[host(call)]
    fn push(&self, frame: AudioFrame);

    /// Queued after the audio pushed so far; raises when the plugin stopped pulling.
    #[host(call)]
    fn mute(&self, muted: bool) -> Result<(), ApiError>;

    /// Drops the queued audio: what is pulled next is the newest.
    #[host(call)]
    fn flush(&self) -> Result<(), ApiError>;

    #[host(next)]
    async fn next(&self) -> Result<Option<DuplexInputItem>, ApiError>;
}

/// A `DuplexRealtimeAdapter` (`adapter`) and its model, as the engine opens sessions on it.
#[crate::host(proxy(host = "livekit.agents._bindings.Duplex"))]
pub trait DuplexHost: Send + Sync {
    #[host(call, getter = "adapter.duplex_model.label")]
    fn label(&self) -> Result<String, HostError>;

    #[host(call, getter = "adapter.duplex_model.capabilities")]
    fn capabilities(&self) -> Result<DuplexCapabilities, HostError>;

    /// The gate the adapter or the model declares; `None`: the adaptive gate.
    #[host(call, name = "_core_gate")]
    fn gate(&self) -> Result<Option<Arc<dyn AudioGate>>, HostError>;

    /// A new plugin session.
    #[host(await, name = "_core_session")]
    fn session(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Arc<dyn DuplexSession>, HostError>>;
}

/// A plugin's `DuplexSession`: the engine feeds its input, applies the updates one after the other
/// and pulls the output audio and the events. `_bindings._DuplexSession` holds the plugin's session
/// as `plugin` and the events its callbacks report as `events`.
#[crate::host(proxy(host = "livekit.agents._bindings._DuplexSession"))]
pub trait DuplexSession: Send + Sync {
    /// The input the plugin pulls the caller's audio from.
    #[host(read_once, getter = "plugin._input")]
    fn input(&self) -> &Arc<dyn DuplexInputQueue>;

    /// The whole startup configuration at once (`_update_session`).
    #[host(await, name = "_core_configure")]
    fn configure(
        &self,
        cx: &crate::telemetry::Cx,
        instructions: Option<String>,
        items: Option<Vec<ChatItem>>,
        names: Vec<String>,
    ) -> crate::BoxFuture<'_, Result<(), HostError>>;

    #[host(await, name = "plugin._update_instructions")]
    fn update_instructions(
        &self,
        cx: &crate::telemetry::Cx,
        instructions: String,
    ) -> crate::BoxFuture<'_, Result<(), HostError>>;

    /// New chat items only: the adapter owns the context.
    #[host(await, name = "plugin._append_items")]
    fn append_items(
        &self,
        cx: &crate::telemetry::Cx,
        items: Vec<ChatItem>,
    ) -> crate::BoxFuture<'_, Result<(), HostError>>;

    /// Tool names: the host owns the tool objects.
    #[host(await, name = "_core_update_tools")]
    fn update_tools(
        &self,
        cx: &crate::telemetry::Cx,
        names: Vec<String>,
    ) -> crate::BoxFuture<'_, Result<(), HostError>>;

    #[host(await, name = "_core_generate")]
    fn generate(
        &self,
        cx: &crate::telemetry::Cx,
        instructions: Option<String>,
        tool_choice: Option<ToolChoice>,
    ) -> crate::BoxFuture<'_, Result<(), HostError>>;

    /// The next frame of output audio; `None` once the output ended.
    #[host(await, name = "_core_next_audio")]
    fn next_audio(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Option<DuplexAudio>, HostError>>;

    /// The next event; `None` once the session closed.
    #[host(await, name = "events.get")]
    fn next_event(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> crate::BoxFuture<'_, Result<Option<DuplexEvent>, HostError>>;

    /// Closes the plugin's session, as a failed startup configuration does.
    #[host(await, name = "plugin.aclose")]
    fn close_plugin(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, ()>;

    #[host(await, name = "_core_aclose")]
    fn aclose(&self, cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, ()>;
}

/// `DuplexRealtimeAdapter`'s engine, the session's realtime model when handed to it.
#[crate::host(class(name = "DuplexRealtimeModel", native = crate::Realtime))]
pub trait DuplexRealtimeModel: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// `audio_timeout`: how long to wait for the next frame before the output counts as finished.
    #[host(new)]
    fn new(
        env: &Self::Env,
        host: Arc<dyn DuplexHost>,
        #[host(kw, default = 0.8)] audio_timeout: f64,
    ) -> Result<Self, HostError>;

    /// The realtime capabilities the model runs under.
    #[host(call)]
    fn capabilities(&self) -> crate::llm::realtime::RealtimeCapabilities;
}
