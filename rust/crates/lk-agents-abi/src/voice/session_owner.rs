//! The session's lifetime, including resources acquired before the voice loop exists.

use std::sync::Arc;

use crate::{BoxFuture, errors::HostError};

/// `update_options` or `override_options` refused: its arguments or the session, or the host
/// failing to bind a changed provider, raised as itself.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, thiserror::Error)]
pub enum OptionsRefused {
    #[error(transparent)]
    #[host(delegate)]
    Command(#[from] super::session::CommandRefused),
    #[error(transparent)]
    #[host(delegate)]
    Session(#[from] super::session::SessionRefused),
    #[error(transparent)]
    #[host(delegate)]
    Host(#[from] HostError),
}

/// The job a session runs in: its trace is the job's, and its `agent_session` repeats the job's
/// dispatch, worker and agent ids.
#[crate::host(record(module = "session_owner", extract))]
pub struct SessionJobInfo {
    pub id: String,
    #[host(default = None)]
    pub dispatch_id: Option<String>,
    #[host(default = None)]
    pub worker_id: Option<String>,
    #[host(default = None)]
    pub agent_id: Option<String>,
}

#[crate::host(record(module = "session_owner", extract))]
pub struct ProviderBindings {
    pub stt: crate::Maybe<Option<Arc<dyn crate::Stt>>>,
    pub tts: crate::Maybe<Option<Arc<dyn crate::Tts>>>,
    pub vad: crate::Maybe<Option<Arc<dyn crate::Vad>>>,
    #[host(default = None)]
    pub stt_node: Option<Arc<dyn super::nodes::NodeFactory>>,
    #[host(default = None)]
    pub tts_node: Option<Arc<dyn super::nodes::NodeFactory>>,
}

#[derive(Clone)]
pub struct SessionBindings {
    pub room: Arc<dyn super::room_owner::SessionRoom>,
    pub args: super::session::VoiceSessionArgs,
    /// `lk agent console --record`: the job's session records its audio whatever the job's
    /// recording.
    pub record_audio: bool,
}

/// Prepared capabilities cross back as a native object: proxy extraction happens on the
/// host loop at construction, rather than on the executor awaiting a host callback's result.
#[crate::host(native(refused = "expected native session bindings"))]
pub trait SessionBindingsPort: Send + Sync {
    fn bindings(&self) -> SessionBindings;
}

#[crate::host(class(name = "_core_SessionBindings", native = SessionBindingsPort))]
pub trait SessionPreparation: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new(
        room: Arc<dyn super::room_owner::SessionRoom>,
        args: super::session::VoiceSessionArgs,
        #[host(kw, default = false)] record_audio: bool,
    ) -> Self;
}

/// Language-specific SDK adaptation. Resource ordering belongs to the native owner.
#[crate::host(proxy(host = "livekit.agents.voice._session_bindings.SessionHost"))]
pub trait SessionHost: Send + Sync {
    /// The session, as its job holds it.
    #[host(call, getter = "session")]
    fn session(&self) -> Result<crate::HostRef, HostError>;
    /// The room, the agent's frame and the host legs the session opens on. `args.job` is the
    /// job the session runs for, given when it opens on the job's room or the console's.
    #[host(await, name = "prepare")]
    fn prepare(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> BoxFuture<'_, Result<Arc<dyn SessionBindingsPort>, HostError>>;
    #[host(await, name = "started")]
    fn started(&self, cx: &crate::telemetry::Cx) -> BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "closed")]
    fn closed(&self, cx: &crate::telemetry::Cx) -> BoxFuture<'_, ()>;
    /// The running session's bindings of `providers`, the slots whose effective provider changed
    /// (`None`: removed), each under the current agent's node for its leg.
    #[host(call, name = "bind_providers")]
    fn bind_providers(
        &self,
        providers: std::collections::BTreeMap<String, Option<crate::HostRef>>,
    ) -> Result<ProviderBindings, HostError>;
}

#[crate::host(class(name = "_core_SessionOwner"))]
pub trait SessionOwner: Send + Sync + Sized + 'static {
    #[host(env)]
    type Env;
    #[host(class(name = "SessionOptions"))]
    type Options;
    #[host(class(name = "_core_SessionStorage"))]
    type Storage: Send + Sync + 'static;
    #[host(class(name = "VoiceSession"))]
    type Session: Send + Sync + 'static;
    #[host(class(name = "_core_SessionState"))]
    type State: Send + Sync + 'static;

    #[host(new)]
    fn new(env: &Self::Env, options: Self::Options) -> Self;
    /// `job`: the job the session runs in, whose trace its spans join under the span current at
    /// the call, and which alone may read its report. Startup runs in order: the thread's storage
    /// opens; the host prepares the room and legs; a session of the job (on its room, or the
    /// console's) claims the job's primary session and sets the job's recording from its plan;
    /// the start time is set; the job's session opens its audio recording and the job's shutdown
    /// is made to close it; the recording is ready; the toolsets set up and the voice loop starts.
    #[host(await)]
    fn start(
        &self,
        storage: Arc<Self::Storage>,
        host: Arc<dyn SessionHost>,
        #[host(kw, default = None)] job: Option<SessionJobInfo>,
        cx: crate::telemetry::Cx,
    ) -> Result<BoxFuture<'static, Result<(), HostError>>, HostError>;
    #[host(getter)]
    fn session(&self) -> Option<Arc<Self::Session>>;
    /// The session's `agent_session` span, from `start()`: the job's current span while the
    /// session runs.
    #[host(getter)]
    fn span(&self) -> Option<crate::telemetry::record::SpanContext>;
    #[host(getter)]
    fn storage(&self) -> Option<Arc<Self::Storage>>;
    #[host(getter)]
    fn current_agent(&self) -> Option<crate::HostRef>;
    /// The latest state the host's events delivered; before `start()`, an empty history.
    #[host(getter)]
    fn state(&self) -> Arc<Self::State>;
    /// Empty before startup; the final snapshot remains readable after close.
    #[host(getter)]
    fn usage(&self) -> crate::usage::Usage;
    /// No keyterms are applied before a voice driver exists.
    #[host(getter)]
    fn keyterms(&self) -> Vec<String>;
    /// `update_agent(agent, chat_ctx=)`. Before `start()`, `agent` drives once the session
    /// starts, on `chat_ctx` unless `start()` names its own; after, `frame` is the handoff and
    /// `chat_ctx` the context `agent` works on instead of the conversation carried forward.
    #[host(call)]
    fn update_agent(
        &self,
        agent: crate::HostRef,
        frame: Option<super::session::UpdateAgentArgs>,
        #[host(kw, default = None)] chat_ctx: Option<Vec<crate::ChatItem>>,
    ) -> Result<(), super::session::SessionRefused>;
    /// `update_options(..)`: change only the supplied settings. A replaced `stt`, `tts` or `vad`
    /// is installed in place on a running session, without restarting the call or interrupting
    /// speech; a setting an `override_options` layer holds takes the new value once the layer is
    /// released. `interruption` merges over the session's and takes no `mode`, which is set with
    /// `VoiceSession(turn_handling=)`.
    #[host(call)]
    #[allow(clippy::too_many_arguments)]
    fn update_options(
        &self,
        #[host(kw)] stt: crate::Maybe<Option<crate::HostRef>>,
        tts: crate::Maybe<Option<crate::HostRef>>,
        vad: crate::Maybe<Option<crate::HostRef>>,
        turn_detection: crate::Maybe<super::session_options::TurnDetectionKwarg>,
        endpointing_opts: crate::Maybe<super::session_options::EndpointingKwargs>,
        interruption: crate::Maybe<super::session_options::InterruptionKwargs>,
        min_consecutive_speech_delay: crate::Maybe<f64>,
        tts_text_transforms: crate::Maybe<Option<super::session_options::TextTransformSet>>,
        expressive: crate::Maybe<super::session_options::Expressive>,
        keyterms: crate::Maybe<Vec<String>>,
    ) -> Result<(), OptionsRefused>;
    /// `override_options(..)` entered: a layer over the session's settings and providers, each
    /// taking its value from the topmost layer that sets it, named by the id returned, refused as
    /// `update_options` refuses.
    #[host(call)]
    #[allow(clippy::too_many_arguments)]
    fn override_options(
        &self,
        #[host(kw)] stt: crate::Maybe<Option<crate::HostRef>>,
        tts: crate::Maybe<Option<crate::HostRef>>,
        vad: crate::Maybe<Option<crate::HostRef>>,
        turn_detection: crate::Maybe<super::session_options::TurnDetectionKwarg>,
        endpointing_opts: crate::Maybe<super::session_options::EndpointingKwargs>,
        interruption: crate::Maybe<super::session_options::InterruptionKwargs>,
        min_consecutive_speech_delay: crate::Maybe<f64>,
        tts_text_transforms: crate::Maybe<Option<super::session_options::TextTransformSet>>,
    ) -> Result<u64, OptionsRefused>;
    /// The `override_options` block ended, however it ended: its settings are released and the
    /// providers it changed restored. On a closed session the settings still release and the
    /// restore is refused (`SessionClosing`); there is nothing left to restore.
    #[host(call)]
    fn release_override(&self, layer: u64) -> Result<(), OptionsRefused>;
    #[host(call)]
    fn resolved_options(
        &self,
        legs: super::session_options::SessionLegs,
    ) -> super::session_options::ResolvedSessionOptions;
    #[host(await)]
    fn aclose(
        &self,
        reason: String,
        #[host(default = false)] drain: bool,
    ) -> Result<BoxFuture<'static, ()>, super::session::UnknownCloseReason>;
    #[host(call)]
    fn shutdown(
        &self,
        reason: String,
        #[host(default = false)] drain: bool,
    ) -> Result<(), super::session::UnknownCloseReason>;
    /// The recording's final result remains available to the job reporter.
    #[host(await)]
    async fn finish_recording(&self) -> Result<(), crate::recorder::RecordingFailure>;
    /// When the session started (wall clock seconds), once its startup prepared it.
    #[host(getter)]
    fn started_at(&self) -> Option<f64>;
    /// What the session records: its job's plan for it, nothing outside a job or before it starts.
    #[host(getter)]
    fn recording_options(&self) -> crate::telemetry::report::RecordingOptions;
    /// The session's audio recording file (`audio.ogg` in the job's session directory), when it
    /// records audio.
    #[host(getter)]
    fn recording_path(&self) -> Option<String>;
    /// When the audio recording's first frame was written.
    #[host(getter)]
    fn recording_started_at(&self) -> Option<f64>;
    /// The session's report for `job` as of now: `history` is the conversation as the caller
    /// sees it, without item turns; `events` are the report's events. Its options are the resolved
    /// options for `legs` with the recording options under `recording_options`.
    #[host(call)]
    fn report(
        &self,
        job: crate::telemetry::report::ReportJob,
        legs: super::session_options::SessionLegs,
        events: Vec<serde_json::Value>,
        history: Vec<crate::ChatItem>,
    ) -> crate::telemetry::report::SessionReport;
    /// The report once the session closed: its history, item turns and usage are the voice
    /// loop's as it closed (empty for a session that never ran).
    #[host(await)]
    fn final_report(
        &self,
        job: crate::telemetry::report::ReportJob,
        legs: super::session_options::SessionLegs,
        events: Vec<serde_json::Value>,
    ) -> Result<
        BoxFuture<'static, crate::telemetry::report::SessionReport>,
        super::session::SessionRefused,
    >;
    /// Whether the job `job_id` may read the session's report: the job the session first started
    /// in, or any job for a session started outside one.
    #[host(call)]
    fn allows_report(&self, job_id: String) -> bool;
    /// Whether the session's report keeps an event of `kind`: a known kind, its content only when
    /// the session's recording is not redacted.
    #[host(call)]
    fn accepts_event(&self, kind: String) -> bool;
    /// An event of `kind` the report keeps is counted as retained; whether it was accepted.
    #[host(call)]
    fn admit_event(&self, kind: String) -> bool;
}

/// `voice.VoiceSession`'s commands: the owner's settings and reads, and the running session's
/// commands, which refuse before `start()`.
#[crate::host(facade(
    path = "livekit.agents.voice.agent_session",
    extended,
    name = "VoiceSession",
    native = SessionOwner
))]
pub trait VoiceSessionFacade {
    /// The session `start()` started; refused before.
    #[host(delegate)]
    type Running: super::session::VoiceSession;

    /// Empty before startup; the final snapshot remains readable after close.
    #[host(getter)]
    fn usage(&self) -> crate::usage::Usage;
    /// No keyterms are applied before a voice driver exists.
    #[host(getter)]
    fn keyterms(&self) -> Vec<String>;
    /// `update_options(..)`: change only the supplied settings. A replaced `stt`, `tts` or `vad`
    /// is installed in place on a running session, without restarting the call or interrupting
    /// speech; a setting an `override_options` layer holds takes the new value once the layer is
    /// released. `interruption` merges over the session's and takes no `mode`, which is set with
    /// `VoiceSession(turn_handling=)`.
    #[host(
        annotation(
            stt = "..stt.STT | str | None",
            tts = "..tts.TTS | str | None",
            vad = ".vad.VAD | None",
            tts_text_transforms = "collections.abc.Sequence[.session_options.TextTransforms] | None",
            expressive = "bool | .expressive.ExpressiveOptions"
        ),
        resolve(
            stt = "..voice._session_bindings._stt_model",
            tts = "..voice._session_bindings._tts_model"
        )
    )]
    #[allow(clippy::too_many_arguments)]
    fn update_options(
        &self,
        #[host(kw)] stt: crate::Maybe<Option<crate::HostRef>>,
        tts: crate::Maybe<Option<crate::HostRef>>,
        vad: crate::Maybe<Option<crate::HostRef>>,
        turn_detection: crate::Maybe<super::session_options::TurnDetectionKwarg>,
        endpointing_opts: crate::Maybe<super::session_options::EndpointingKwargs>,
        interruption: crate::Maybe<super::session_options::InterruptionKwargs>,
        min_consecutive_speech_delay: crate::Maybe<f64>,
        tts_text_transforms: crate::Maybe<Option<super::session_options::TextTransformSet>>,
        expressive: crate::Maybe<super::session_options::Expressive>,
        keyterms: crate::Maybe<Vec<String>>,
    ) -> Result<(), OptionsRefused>;

    /// `generate_reply()`: reply now, with no end of turn behind it. The overrides apply to this
    /// reply only; `tools` narrows the driving agent's tools to these names (its provider-owned
    /// tools stay), refused on a realtime model as `chat_ctx` is. Sent from the agent's
    /// `on_enter`, it is that hook's reply.
    #[host(
        via = Running,
        not_given,
        origin = "..voice._session_callbacks._origin",
        returns = "..voice.io.SpeechHandle",
        annotation(chat_ctx = "..llm.ChatContext"),
        resolve(chat_ctx = "..voice._session_callbacks._items")
    )]
    fn generate_reply(
        &self,
        #[host(kw, default = None)] user_input: Option<String>,
        #[host(default = None)] instructions: Option<String>,
        #[host(default = None)] tool_choice: Option<crate::ToolChoice>,
        #[host(default = None)] tools: Option<Vec<String>>,
        #[host(default = None)] allow_interruptions: Option<bool>,
        #[host(default = None)] chat_ctx: Option<Vec<crate::ChatItem>>,
    );

    /// `say()`: speak `text` without the LLM, the words given or a stream of them read to its
    /// end. With `audio`, those frames play instead of the TTS and `text`, a `str`, is still
    /// committed. The handle is the promise the session fulfils.
    #[host(
        via = Running,
        not_given,
        origin = "..voice._session_callbacks._origin",
        returns = "..voice.io.SpeechHandle"
    )]
    fn say(
        &self,
        text: crate::OneOrStream<String>,
        #[host(kw, default = None)] audio: Option<Box<dyn crate::StreamOutput<crate::AudioFrame>>>,
        #[host(default = None)] allow_interruptions: Option<bool>,
        #[host(default = true)] add_to_chat_ctx: bool,
    );

    /// `RoomIO.set_participant(identity)`: point the audio input and the user transcript at
    /// someone else. `None` unlinks.
    #[host(via = Running)]
    async fn set_participant(&self, identity: Option<String>) -> crate::voice::io::Result<()>;

    /// `AgentSession.clear_user_turn()`: drop the open turn and everything accumulated for it.
    #[host(via = Running)]
    fn clear_user_turn(&self) -> Result<(), super::session::SessionRefused>;

    /// `AgentSession.interrupt(force=)`: the current and queued speeches up to the first
    /// uninterruptible; answers when they are done, or with the refusal.
    #[host(via = Running, future)]
    async fn interrupt(
        &self,
        #[host(kw, default = false)] force: bool,
    ) -> Result<(), super::session::NotInterruptible>;

    /// `AgentSession.drain()`: the driving agent's `on_exit`, then scheduling paused until no
    /// speech is left. Answers `None`, or why it was refused.
    #[host(via = Running, discard)]
    async fn drain(&self) -> Option<super::session::DrainRefused>;

    /// `AgentSession.wait_for_idle()`: answers when the session is idle (no hold but `hold`).
    #[host(via = Running)]
    async fn wait_for_idle(&self) -> Result<(), super::session::IdleRefused>;

    /// `AgentSession.commit_user_turn(..)`: flush, wait for the final, commit, answer with the
    /// transcript. Runs whether or not awaited; a newer commit supersedes it.
    #[host(via = Running, future)]
    async fn commit_user_turn(
        &self,
        #[host(kw, default = 2.0)] transcript_timeout: f64,
        #[host(default = 2.0)] stt_flush_duration: f64,
        #[host(default = false)] skip_reply: bool,
    ) -> Result<String, super::session::CommitSuperseded>;
}
