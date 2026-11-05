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
    #[host(
        getter,
        facade(module = "session_commands", class = "SessionCommands", target = "_owner")
    )]
    fn usage(&self) -> crate::usage::Usage;
    /// No keyterms are applied before a voice driver exists.
    #[host(
        getter,
        facade(module = "session_commands", class = "SessionCommands", target = "_owner")
    )]
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
    #[host(facade(
        module = "session_commands",
        class = "SessionCommands",
        target = "_owner",
        annotation(
            stt = "..stt.STT | str | None",
            tts = "..tts.TTS | str | None",
            vad = ".vad.VAD | None",
            tts_text_transforms = "collections.abc.Sequence[..voice._session_bindings.TextTransforms] | None",
            expressive = "bool | .expressive.ExpressiveOptions"
        ),
        resolve(
            stt = "..voice._session_bindings._stt_model",
            tts = "..voice._session_bindings._tts_model"
        )
    ))]
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
