//! `_core.VoiceSession`, the session `voice/agent_session.py`'s `AgentSession` drives, and
//! `_core.SessionOptions`, its options before it runs. Implemented by `lk-agents-runtime`; the
//! decisions behind each command are `lk-agents-voice`'s.

use crate::errors::HostError;
use crate::voice::io::{AudioInput, AudioOutput, TextOutput, VideoInput};
use crate::voice::session_options::{
    Expressive, ResolvedSessionOptions, SessionKwargs, SessionLegs, TurnHandlingKwargs,
    UpdateKwargs,
};
use crate::{BoxFuture, ChatItem, HostRef, Llm, Maybe, Stt, ToolSpec, Tts, Vad};
use std::sync::Arc;

/// A command on a session that cannot take it. Each `Display` is the message a host raises.
#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SessionRefused {
    #[error("the session has not been started")]
    NotStarted,
    #[error("the session is closed")]
    Closed,
    /// The loop dropped its receiver: the command went nowhere.
    #[error("the session loop has ended")]
    LoopEnded,
}

/// `start()` on a session that is not waiting to be started.
#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum OpenRefused {
    #[error("this AgentSession is already started")]
    AlreadyStarted,
    #[error("this AgentSession is closed")]
    Closed,
}

/// A command refused before it was sent: its arguments (`ValueError`) or the session
/// (`RuntimeError`).
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CommandRefused {
    #[error(transparent)]
    #[host(kind = "RuntimeError")]
    Session(#[from] SessionRefused),
    #[error("{0}")]
    Invalid(String),
}

/// `interrupt()` refused by the current speech (`allow_interruptions=False`, no `force`).
#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("This generation handle does not allow interruptions")]
pub struct NotInterruptible;

/// `start()`'s awaited half failed: the loop, or what the host does beside it (the room's
/// connection, the remote session's transport), which raises as itself.
#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, thiserror::Error)]
pub enum StartFailed {
    #[error("the session failed to start: {0}")]
    Driver(String),
    #[error(transparent)]
    #[host(delegate)]
    Host(HostError),
}

/// `aclose(reason=..)` with a reason that is no `CloseReason`.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, thiserror::Error)]
#[error("unknown CloseReason {0:?}")]
pub struct UnknownCloseReason(pub String);

/// `push_video(..)` handed a part that is not an image.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, thiserror::Error)]
#[error(
    "push_video takes an image content part -- `ChatContent.image_frame(..)`, which is what \
     `llm.ImageContent(image=frame)` becomes."
)]
pub struct NotAnImage;

/// Why `VoiceTask.run()` returned no value. A task ended under its awaiter cancels the await, as
/// the close cancels a tool's body.
#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error, crate::Vocabulary)]
pub enum VoiceTaskEnded {
    #[error(
        "VoiceTask.run() must be awaited inside a tool of the agent holding the floor of a live \
         VoiceSession"
    )]
    NotInTool,
    #[error("VoiceTask.run() cannot start while the session's agent is changing or draining")]
    AgentChanging,
    #[error("the session closed before the VoiceTask completed")]
    #[host(kind = "asyncio.CancelledError")]
    Closing,
    #[error("the VoiceTask was cancelled before it completed")]
    #[host(kind = "asyncio.CancelledError")]
    Cancelled,
}

/// `wait_for_idle()`/`wait_for_idle_and_hold()` refused: the session is closing.
#[crate::host(error(kind = "SessionClosing", declare = "RuntimeError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error, crate::Vocabulary)]
pub enum IdleRefused {
    /// The session closed (or its loop ended) first: the reference's `ActivityClosedError`.
    #[error("session is closing")]
    Closing,
}

/// A `commit_user_turn()` superseded by a newer one; the host's future is cancelled.
#[crate::host(error(kind = "asyncio.CancelledError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("superseded by a newer commit_user_turn()")]
pub struct CommitSuperseded;

/// The agent a host `Agent` describes (`AgentActivity._frame_kwargs`), as `update_agent` and
/// `run_voice_task` install it. Each provider is three-state: absent keeps the leg, `None` removes
/// it, an object replaces it. `chat_ctx` replaces when present. The tools are one toolset,
/// replaced together because the runner is per-agent. `turn_handling` is the agent layer.
/// `llm_node`, `tts_node` and `stt_node` are the agent's overrides, each over the leg given beside
/// it: a leg kept from the previous agent keeps that agent's node. `transcription_node` and
/// `realtime_audio_output_node` replace the previous agent's.
#[crate::host(record(module = "voice_session", extract))]
#[derive(Clone)]
pub struct UpdateAgentArgs {
    pub instructions: String,
    #[host(default = None)]
    pub dynamic_instructions: Option<Arc<dyn crate::agent::HostInstructions>>,
    /// The new agent's id.
    pub agent: String,
    /// The new agent's `Agent`, handed back by the events that name it.
    #[host(default = None)]
    pub agent_host: Option<HostRef>,
    #[host(default = None)]
    pub tools: Option<Vec<ToolSpec>>,
    #[host(default = None)]
    pub tools_host: Option<HostRef>,
    /// The agent as its tools are hosted: its definition (retry budgets) and its `AgentHost`.
    pub voice_agent: Arc<dyn crate::agent::AgentFrame>,
    #[host(default = None)]
    pub chat_ctx: Option<Vec<ChatItem>>,
    pub llm: Maybe<Option<Arc<dyn Llm>>>,
    pub tts: Maybe<Option<Arc<dyn Tts>>>,
    pub stt: Maybe<Option<Arc<dyn Stt>>>,
    pub vad: Maybe<Option<Arc<dyn Vad>>>,
    #[host(default = None)]
    pub turn_handling: Option<TurnHandlingKwargs>,
    /// The agent's `min_consecutive_speech_delay` in seconds; `None` defers to the session's.
    #[host(default = None)]
    pub min_consecutive_speech_delay: Option<f64>,
    /// The agent's `use_tts_aligned_transcript`; `None` defers to the session's.
    #[host(default = None)]
    pub use_tts_aligned_transcript: Option<bool>,
    /// The agent's `expressive`; `NOT_GIVEN` defers to the session's.
    pub expressive: Maybe<Expressive>,
    #[host(default = None)]
    pub llm_node: Option<Arc<dyn crate::voice::nodes::NodeFactory>>,
    #[host(default = None)]
    pub tts_node: Option<Arc<dyn crate::voice::nodes::NodeFactory>>,
    #[host(default = None)]
    pub stt_node: Option<Arc<dyn crate::voice::nodes::NodeFactory>>,
    #[host(default = None)]
    pub transcription_node: Option<Arc<dyn crate::voice::nodes::NodeFactory>>,
    #[host(default = None)]
    pub realtime_audio_output_node: Option<Arc<dyn crate::voice::nodes::NodeFactory>>,
}

/// What `_core.VoiceSession` is assembled from: the session's legs, hooks, nodes and options as
/// `VoiceSession(...)` and `start()` hand them over.
#[crate::host(record(module = "voice_session", extract, default))]
#[derive(Clone)]
pub struct VoiceSessionArgs {
    #[host(default = None)]
    pub instructions: Option<String>,
    #[host(default = None)]
    pub dynamic_instructions: Option<Arc<dyn crate::agent::HostInstructions>>,
    #[host(default = "")]
    pub agent: String,
    #[host(default = None)]
    pub agent_host: Option<HostRef>,
    #[host(default = None)]
    pub chat_ctx: Option<Vec<ChatItem>>,
    #[host(default = None)]
    pub storage: Option<Arc<dyn crate::storage::voice::VoiceStorage>>,
    #[host(default = None)]
    pub stt: Option<Arc<dyn Stt>>,
    #[host(default = None)]
    pub vad: Option<Arc<dyn Vad>>,
    #[host(default = None)]
    pub llm: Option<Arc<dyn Llm>>,
    #[host(default = None)]
    pub realtime: Option<Arc<dyn crate::Realtime>>,
    #[host(default = None)]
    pub tts: Option<Arc<dyn Tts>>,
    /// Receives each session event, on the calling loop.
    #[host(default = None)]
    pub on_event: Option<Arc<dyn crate::voice::session_effects::SessionEvents>>,
    #[host(default = None)]
    pub hooks: Option<Arc<dyn crate::voice::session_effects::SessionHooks>>,
    #[host(default = None)]
    pub participant_identity: Option<String>,
    #[host(default = true)]
    pub auto_gain_control: bool,
    #[host(default = 24000)]
    pub input_sample_rate: u32,
    #[host(default = 24000)]
    pub output_sample_rate: u32,
    /// The first agent's own `turn_handling`, `min_consecutive_speech_delay` in seconds and
    /// `use_tts_aligned_transcript`: the agent layer of the stack.
    #[host(default = None)]
    pub agent_turn_handling: Option<TurnHandlingKwargs>,
    #[host(default = None)]
    pub agent_min_consecutive_speech_delay: Option<f64>,
    #[host(default = None)]
    pub agent_use_tts_aligned_transcript: Option<bool>,
    /// The turn detector over the user's audio (`turn_detection`).
    #[host(default = None)]
    pub turn_detector: Option<Arc<dyn crate::EotDetector>>,
    /// The adaptive-interruption detector: an object means the host's gate said yes.
    #[host(default = None)]
    pub interruption_detector: Option<Arc<dyn crate::turn::interruption::InterruptionSource>>,
    #[host(default = true)]
    pub text_output: bool,
    /// `RoomOptions._to_core()`, resolved per leg; without it the scalars above.
    #[host(default = None)]
    pub room_io_config: Option<crate::voice::room_io::RoomIoConfig>,
    #[host(default = None)]
    pub tools: Option<Vec<ToolSpec>>,
    #[host(default = None)]
    pub tools_host: Option<HostRef>,
    /// The first agent as its tools are hosted: its definition and its `AgentHost`.
    #[host(default = None)]
    pub voice_agent: Option<Arc<dyn crate::agent::AgentFrame>>,
    #[host(default = None)]
    pub transcription_node: Option<Arc<dyn crate::voice::nodes::NodeFactory>>,
    #[host(default = None)]
    pub realtime_audio_output_node: Option<Arc<dyn crate::voice::nodes::NodeFactory>>,
    #[host(default = None)]
    pub llm_node: Option<Arc<dyn crate::voice::nodes::NodeFactory>>,
    #[host(default = None)]
    pub tts_node: Option<Arc<dyn crate::voice::nodes::NodeFactory>>,
    #[host(default = None)]
    pub stt_node: Option<Arc<dyn crate::voice::nodes::NodeFactory>>,
    /// The host's own io legs, replacing the room's.
    #[host(default = None)]
    pub host_audio_input: Option<Arc<dyn AudioInput>>,
    #[host(default = None)]
    pub host_audio_output: Option<Arc<dyn AudioOutput>>,
    #[host(default = None)]
    pub host_video_input: Option<Arc<dyn VideoInput>>,
    /// `VoiceActivityVideoSampler(speaking_fps, silent_fps)`'s two rates; `None`: every frame.
    #[host(default = None)]
    pub video_sampler: Option<(f64, f64)>,
    /// A custom `video_sampler` callable: the host pumps and gates; the session reads no camera.
    #[host(default = false)]
    pub host_video_sampler: bool,
    #[host(default = None)]
    pub host_transcription_output: Option<Arc<dyn TextOutput>>,
    #[host(default = false)]
    pub using_default_vad: bool,
    /// The end-of-turn hook's bound, in seconds; `None` is no hook and no wait.
    #[host(default = None)]
    pub hook_bound: Option<f64>,
    #[host(default = 3)]
    pub max_unrecoverable_errors: u32,
    #[host(default = None)]
    pub stt_context: Option<crate::turn::keyterms::SttContextConfig>,
    #[host(default = None)]
    pub keyterm_llm: Option<Arc<dyn Llm>>,
    #[host(default = None)]
    pub keyterm_tools: Option<HostRef>,
    /// The first agent's own `expressive`; `NOT_GIVEN` defers to the session's.
    pub agent_expressive: Maybe<Expressive>,
    /// Prefixes the ids the session mints for chat items; `None` mints a random one.
    #[host(default = None)]
    pub session_id: Option<String>,
    /// Serve the remote session protocol (`lk agent simulate`, the dashboard, `lk agent
    /// console`) on the room's byte streams, or on `console`.
    #[host(default = None)]
    pub remote_session: Option<Arc<dyn crate::worker::remote_session::SessionPort>>,
    /// `lk agent console`'s socket: the session's audio input and output, and the remote
    /// session's transport.
    #[host(default = None)]
    pub console: Option<Arc<dyn crate::voice::console::ConsolePort>>,
    /// The job whose room this is: `start()` connects it beside the opening.
    #[host(default = None)]
    pub job: Option<Arc<dyn crate::worker::job::SessionJob>>,
}

/// `AgentSession(**kwargs)`'s options before the session holds them: the raw kwargs, read once,
/// and the session layer they became (plus every `update_options` so far).
#[crate::host(class(name = "SessionOptions"))]
pub trait SessionOptions: Clone + Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(kwargs: SessionKwargs) -> Self;

    /// `update_options(..)` before the session runs, on the same `merge` the driver uses.
    #[host(call)]
    fn update(&self, update: UpdateKwargs);

    /// The resolved configuration over the host-supplied legs, as the dict `session.options` reads
    /// before `start()`.
    #[host(call)]
    fn resolved(&self, legs: SessionLegs) -> ResolvedSessionOptions;
}

/// `voice/agent_session.py`'s `AgentSession`: assembled at construction, started at `start()`,
/// closed at `aclose()`; every other method is one command to its loop.
#[crate::host(class(name = "VoiceSession"))]
pub trait VoiceSession: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(class(name = "SessionOptions"))]
    type Options: SessionOptions;

    /// The room's microphone and video legs.
    #[host(class(name = "RoomAudioInput"))]
    type MicLeg;
    #[host(class(name = "RoomVideoInput"))]
    type CameraLeg;
    /// A speech `say()` / `generate_reply()` answers: the host's `_core.SpeechHandle`.
    #[host(handle(class = "SpeechHandle"))]
    type Speech: Send + Sync + 'static;

    /// Assemble a session; the loop begins at `start()`. `room` supplies runtime room I/O, or no
    /// RTC attachment; `options` is the session layer `SessionOptions(kwargs)` built (none is an
    /// empty session layer).
    #[host(new)]
    fn new(
        env: &Self::Env,
        room: std::sync::Arc<dyn crate::voice::room_owner::SessionRoom>,
        args: VoiceSessionArgs,
        #[host(default = None)] options: Option<Self::Options>,
    ) -> Result<Self, crate::voice::rtc::RtcError>;

    /// Open the providers and run the loop beside the host event drain. Resolves once the host
    /// has handled the opening events.
    #[host(await)]
    fn start(&self) -> Result<BoxFuture<'static, Result<(), StartFailed>>, OpenRefused>;

    /// `session.usage.model_usage`: the usage of every model the session was billed for.
    #[host(getter)]
    fn usage(&self) -> Vec<crate::metrics::usage::ModelUsage>;

    /// The room's microphone leg, unless `RoomIoConfig(audio_input=False)`, else `None`. A fresh
    /// handle each call over the same leg; safe before or after `start()`.
    #[host(getter)]
    fn audio_input(&self) -> Option<Self::MicLeg>;

    /// The room's video leg, when `RoomIoConfig(video_input=True)` asked for one, else `None`.
    #[host(getter)]
    fn video_input(&self) -> Option<Self::CameraLeg>;

    /// Record the session into `recorder`; `None` detaches. Safe before or after `start()`; the
    /// file's `finish` stays the host's.
    #[host(call)]
    fn attach_recorder(&self, recorder: Option<Arc<dyn crate::recorder::AudioRecorder>>);

    /// `Agent.update_options(stt=.., vad=.., llm=.., tts=..)` on a running agent: an in-place
    /// provider swap, not a handoff. Each leg is three-state; a given one goes under the agent's
    /// node when it overrides one.
    #[host(call)]
    #[allow(clippy::too_many_arguments)]
    fn update_models(
        &self,
        #[host(kw)] llm: Maybe<Option<Arc<dyn Llm>>>,
        tts: Maybe<Option<Arc<dyn Tts>>>,
        stt: Maybe<Option<Arc<dyn Stt>>>,
        vad: Maybe<Option<Arc<dyn Vad>>>,
        #[host(default = None)] llm_node: Option<Arc<dyn crate::voice::nodes::NodeFactory>>,
        #[host(default = None)] tts_node: Option<Arc<dyn crate::voice::nodes::NodeFactory>>,
        #[host(default = None)] stt_node: Option<Arc<dyn crate::voice::nodes::NodeFactory>>,
    ) -> Result<(), SessionRefused>;

    /// The handoff: replace the agent, keep the session. Resolves `True` once the agent is
    /// installed (`on_enter` issued, not returned), `False` when it was refused (closing).
    #[host(await)]
    fn update_agent(
        &self,
        frame: UpdateAgentArgs,
    ) -> Result<BoxFuture<'static, bool>, SessionRefused>;

    /// `VoiceTask.run()`: the task's agent, sent from inside a tool (`origin` is its
    /// `"tool:<call_id>"`). The agent holds the floor until one of its tools completes with a
    /// value the host validated, which this resolves to; the parent comes back without its hooks.
    /// A cancelled await ends the task, and every task it started, and puts the parent back.
    #[host(await)]
    fn run_voice_task(
        &self,
        frame: UpdateAgentArgs,
        #[host(kw, default = None)] origin: Option<String>,
        #[host(default = false)] opens_with_reply: bool,
    ) -> Result<BoxFuture<'static, Result<HostRef, VoiceTaskEnded>>, CommandRefused>;

    /// `session.input.audio = ..` after `start()`; `None` detaches.
    #[host(call)]
    fn set_audio_input(&self, source: Option<Arc<dyn AudioInput>>) -> Result<(), SessionRefused>;

    /// `session.input.video = ..` after `start()`: the video pump reads the new source, or stops
    /// on `None`.
    #[host(call)]
    fn set_video_input(&self, source: Option<Arc<dyn VideoInput>>) -> Result<(), SessionRefused>;

    /// Select the destination for subsequent output segments. Current work retains its admitted
    /// binding, including when the same endpoint object is selected again.
    #[host(call)]
    fn set_audio_output(&self, sink: Option<Arc<dyn AudioOutput>>) -> Result<(), SessionRefused>;

    /// `session.output.transcription = ..` after `start()`: the next reply forwards into the new
    /// leg; one in flight keeps the leg it started with.
    #[host(call)]
    fn set_transcription_output(
        &self,
        sink: Option<Arc<dyn TextOutput>>,
    ) -> Result<(), SessionRefused>;

    /// `Agent.update_tools()` on a running agent. All three parts go together: the agent hosts
    /// its tools.
    #[host(call)]
    fn update_tools(
        &self,
        tools: Option<Vec<ToolSpec>>,
        #[host(kw)] tools_host: Option<HostRef>,
        voice_agent: Option<Arc<dyn crate::agent::AgentFrame>>,
    ) -> Result<(), SessionRefused>;

    /// `AgentSession.update_options(..)` on a running session: merged into its session layer and
    /// re-resolved; `expressive` reaches the next reply, `keyterms` replace the user's list. A turn
    /// detector object is refused: swapping the detector of a running session is not ported.
    #[host(call)]
    fn update_turn_options(&self, update: UpdateKwargs) -> Result<(), CommandRefused>;

    /// The core's accepted configuration; `None` for a session built without options.
    #[host(call)]
    fn resolved_options(&self) -> Option<ResolvedSessionOptions>;

    /// `AgentActivity.push_video(frame)`: one sampled frame, on its way to the model. `False`
    /// when the loop did not take it (not started or closed); a non-image part raises.
    #[host(call)]
    fn push_video(&self, content: crate::ChatContent) -> Result<bool, NotAnImage>;

    /// `AgentSession.generate_reply()`: reply now, with no end of turn behind it. The overrides
    /// apply to this reply only; `tools=`, `tools_host=` and `voice_agent=` go together.
    #[host(call)]
    #[allow(clippy::too_many_arguments)]
    fn generate_reply(
        &self,
        #[host(kw, default = None)] user_input: Option<String>,
        #[host(default = None)] instructions: Option<String>,
        #[host(default = None)] allow_interruptions: Option<bool>,
        #[host(default = None)] tool_choice: Option<crate::ToolChoice>,
        #[host(default = None)] chat_ctx: Option<Vec<ChatItem>>,
        #[host(default = None)] tools: Option<Vec<ToolSpec>>,
        #[host(default = None)] tools_host: Option<HostRef>,
        #[host(default = None)] voice_agent: Option<Arc<dyn crate::agent::AgentFrame>>,
        // Decided by the host: the reference tests a `ContextVar` plus agent/session identity.
        #[host(default = false)] on_enter: bool,
        #[host(default = None)] origin: Option<String>,
    ) -> Result<Self::Speech, CommandRefused>;

    /// `IVRActivity.start()`: `max_silence` is `max_silence_duration` in seconds; `tools` is
    /// `IVRActivity.tools`. A second start is journalled, not run.
    #[host(call)]
    fn ivr_start(
        &self,
        #[host(default = 5.0)] max_silence: f64,
        #[host(default = None)] tools: Option<Vec<ToolSpec>>,
        #[host(kw, default = None)] tools_host: Option<HostRef>,
    ) -> Result<(), CommandRefused>;

    /// `AMD._run`: the answering-machine detector over its config, classifier LLM, tools and
    /// optional STT. It listens once the callee answers in the session's room; the verdict comes
    /// back as `amd_result`. A second attach is refused.
    #[host(call)]
    fn attach_amd(
        &self,
        cfg: crate::voice::amd::AmdConfigKwargs,
        llm: Arc<dyn Llm>,
        tools: Option<HostRef>,
        stt: Option<Arc<dyn Stt>>,
        participant_identity: Option<String>,
    ) -> Result<(), CommandRefused>;

    /// `_aclose_impl`: the hooks' `closing`, the close itself, the turn-detector stream, the
    /// remote session host, the room io, then the session's toolsets. Idempotent: a second call
    /// waits for the first.
    #[host(await)]
    fn aclose(
        &self,
        #[host(kw, default = "user_initiated")] reason: String,
        #[host(default = false)] drain: bool,
    ) -> Result<BoxFuture<'static, ()>, UnknownCloseReason>;

    /// `AgentSession.shutdown(drain=)`: `aclose` without waiting for it.
    #[host(call)]
    fn shutdown(&self, #[host(kw, default = true)] drain: bool);

    /// `RoomIO.set_participant(identity)`: point the audio input and the user transcript at
    /// someone else. `None` unlinks.
    #[host(await)]
    fn set_participant(
        &self,
        identity: Option<String>,
    ) -> BoxFuture<'static, crate::voice::io::Result<()>>;

    /// `RunContext.with_filler(..)` entered for tool `call_id` of `speech_id`: after `delay` s
    /// idle say `text` (or ask `source`), then repeat every `interval`. Returns the scope's number.
    #[host(call)]
    #[allow(clippy::too_many_arguments)]
    fn filler_start(
        &self,
        call_id: String,
        speech_id: String,
        #[host(kw)] delay: f64,
        #[host(default = None)] interval: Option<f64>,
        #[host(default = None)] max_steps: Option<u32>,
        #[host(default = None)] text: Option<String>,
        #[host(default = None)] source: Option<HostRef>,
    ) -> Result<u64, CommandRefused>;

    /// `with_filler`'s exit.
    #[host(call)]
    fn filler_stop(&self, id: u64) -> Result<(), SessionRefused>;

    /// `AgentSession.clear_user_turn()`: drop the open turn and everything accumulated for it.
    #[host(call)]
    fn clear_user_turn(&self) -> Result<(), SessionRefused>;

    /// `Agent.update_instructions()` on the agent that is already driving.
    #[host(call)]
    fn update_instructions(&self, instructions: String) -> Result<(), SessionRefused>;

    /// `input.set_audio_enabled` / `output.set_audio_enabled` /
    /// `output.set_transcription_enabled`: `leg` is `"audio_input"`, `"audio_output"` or
    /// `"transcription"`.
    #[host(call)]
    fn set_io_enabled(&self, leg: String, enabled: bool) -> Result<(), CommandRefused>;

    /// `RunContext.update()` after the first (an `update` outcome), or a detached tool's end.
    #[host(call)]
    fn tool_update(
        &self,
        call_id: String,
        report: crate::llm::tool_executor::ToolReport,
    ) -> Result<(), CommandRefused>;

    /// `Agent.update_chat_ctx()` on a running agent: replace the context.
    #[host(call)]
    fn update_chat_ctx(&self, items: Vec<ChatItem>) -> Result<(), SessionRefused>;

    /// `AgentSession.interrupt(force=)`: the current and queued speeches up to the first
    /// uninterruptible; answers when they are done, or with the refusal.
    #[host(await)]
    fn interrupt(
        &self,
        #[host(kw, default = false)] force: bool,
    ) -> Result<BoxFuture<'static, Result<(), NotInterruptible>>, SessionRefused>;

    /// `AgentSession.drain()`: the driving agent's `on_exit`, then scheduling paused until no
    /// speech is left. Answers `None`, `"activity_changing"` or `"closing"`.
    #[host(await)]
    fn drain(&self) -> Result<BoxFuture<'static, Option<String>>, SessionRefused>;

    /// `EndCallTool`: close the session once the tool's speech (`speech_id`) is done, or once a
    /// realtime model's own reply to the tool result is.
    #[host(call)]
    fn end_call(&self, speech_id: String) -> Result<(), CommandRefused>;

    /// `AgentSession.wait_for_idle()`: answers when the session is idle (no hold but `hold`).
    #[host(await)]
    fn wait_for_idle(
        &self,
        #[host(kw, default = None)] hold: Option<u64>,
    ) -> Result<BoxFuture<'static, Result<(), IdleRefused>>, SessionRefused>;

    /// `RunContext.foreground()`: wait for idle, then hold the floor until `release_idle_hold`.
    /// Answers the hold's id.
    #[host(await)]
    fn wait_for_idle_and_hold(
        &self,
        #[host(kw, default = None)] origin: Option<String>,
    ) -> Result<BoxFuture<'static, Result<u64, IdleRefused>>, CommandRefused>;

    /// `_wait_for_idle_and_hold`'s exit. A hold the loop no longer has is a no-op.
    #[host(call)]
    fn release_idle_hold(&self, hold: u64);

    /// `AgentSession._claim_user_turn()`'s enter: pins the user state to `speaking` and holds
    /// `wait_for_idle` open. Reentrant.
    #[host(call)]
    fn claim_user_turn(&self) -> Result<(), SessionRefused>;

    /// `_claim_user_turn()`'s exit: the last release re-derives the user state.
    #[host(call)]
    fn release_user_turn(&self);

    /// `VoiceSession._run(user_input=)`, answered on the stream: every run event carries `host`,
    /// the host's run recorder.
    #[host(call)]
    fn run(
        &self,
        #[host(kw, default = None)] user_input: Option<String>,
        #[host(default = None)] origin: Option<String>,
        #[host(default = None)] host: Option<HostRef>,
    ) -> Result<(), CommandRefused>;

    /// `AgentSession.say(text)`: speak these words, no LLM. The handle is the promise the loop
    /// fulfils.
    #[host(call)]
    fn say(
        &self,
        text: String,
        #[host(kw, default = None)] allow_interruptions: Option<bool>,
        #[host(default = true)] add_to_chat_ctx: bool,
        #[host(default = None)] origin: Option<String>,
    ) -> Result<Self::Speech, CommandRefused>;

    /// `AgentSession.say(AsyncIterable[str])`: the speech reads `text` to its end.
    #[host(call)]
    fn say_stream(
        &self,
        text: Box<dyn crate::StreamSource<String>>,
        #[host(kw, default = None)] allow_interruptions: Option<bool>,
        #[host(default = true)] add_to_chat_ctx: bool,
        #[host(default = None)] origin: Option<String>,
    ) -> Result<Self::Speech, CommandRefused>;

    /// `AgentSession.say(text, audio=..)`: pre-synthesised frames instead of TTS; `text` is still
    /// committed.
    #[host(call)]
    fn say_audio(
        &self,
        text: String,
        audio: Box<dyn crate::StreamSource<crate::AudioFrame>>,
        #[host(kw, default = None)] allow_interruptions: Option<bool>,
        #[host(default = true)] add_to_chat_ctx: bool,
        #[host(default = None)] origin: Option<String>,
    ) -> Result<Self::Speech, CommandRefused>;

    /// `_start_ivr_detection(transcript)`: a final the navigator counts as if heard.
    #[host(call)]
    fn ivr_final(&self, text: String) -> Result<(), SessionRefused>;

    /// `AMD.started`: the detector listens (the callee answered, no verdict out yet).
    #[host(call)]
    fn amd_listening(&self) -> bool;

    /// `AMD.execute()`'s interrupt and `_resume_authorization()`, after its IVR hand-off.
    #[host(call)]
    fn amd_execute(&self) -> Result<(), SessionRefused>;

    /// `AMD.aclose()`. Idempotent.
    #[host(call)]
    fn amd_close(&self) -> Result<(), SessionRefused>;

    /// `session.keyterms`: the list the recogniser was last handed; `None` before any push.
    #[host(call)]
    fn keyterms(&self) -> Option<Vec<String>>;

    /// `Agent.chat_ctx` of the driving agent; empty before `start()`.
    #[host(call)]
    fn agent_chat_ctx(&self) -> Vec<ChatItem>;

    /// `AgentSession.commit_user_turn(..)`: flush, wait for the final, commit, answer with the
    /// transcript. Runs whether or not awaited; a newer commit supersedes it.
    #[host(await)]
    fn commit_user_turn(
        &self,
        #[host(kw, default = 2.0)] transcript_timeout: f64,
        #[host(default = 2.0)] stt_flush_duration: f64,
        #[host(default = false)] skip_reply: bool,
    ) -> Result<BoxFuture<'static, Result<String, CommitSuperseded>>, SessionRefused>;
}
