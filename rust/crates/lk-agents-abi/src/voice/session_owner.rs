//! The session's lifetime, including resources acquired before the voice loop exists.

use std::sync::Arc;

use crate::{BoxFuture, errors::HostError};

#[crate::host(record(module = "session_owner", emit))]
pub struct SessionOverride {
    pub layer: u64,
    pub providers: std::collections::BTreeMap<String, Option<crate::HostRef>>,
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
    pub recorder: Option<Arc<dyn crate::recorder::AudioRecorder>>,
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
        recorder: Option<Arc<dyn crate::recorder::AudioRecorder>>,
    ) -> Self;
}

/// Language-specific SDK adaptation. Resource ordering belongs to the native owner.
#[crate::host(proxy(host = "livekit.agents.voice._session_host.SessionHost"))]
pub trait SessionHost: Send + Sync {
    #[host(await, name = "prepare")]
    fn prepare(
        &self,
        cx: &crate::telemetry::Cx,
    ) -> BoxFuture<'_, Result<Arc<dyn SessionBindingsPort>, HostError>>;
    #[host(await, name = "started")]
    fn started(&self, cx: &crate::telemetry::Cx) -> BoxFuture<'_, Result<(), HostError>>;
    #[host(await, name = "closed")]
    fn closed(&self, cx: &crate::telemetry::Cx) -> BoxFuture<'_, ()>;
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

    #[host(new)]
    fn new(env: &Self::Env, options: Self::Options) -> Self;
    /// `job`: the job the session runs in, whose trace its spans join under the span current at
    /// the call.
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
    /// The session's `agent_session` span (trace id, span id, sampled), from `start()`: the
    /// job's current span while the session runs.
    #[host(getter)]
    fn span(&self) -> Option<(u128, u64, bool)>;
    #[host(getter)]
    fn storage(&self) -> Option<Arc<Self::Storage>>;
    #[host(getter)]
    fn current_agent(&self) -> Option<crate::HostRef>;
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
    #[host(call)]
    fn update_agent(
        &self,
        agent: crate::HostRef,
        frame: Option<super::session::UpdateAgentArgs>,
    ) -> Result<(), super::session::SessionRefused>;
    #[host(call)]
    fn update_options(
        &self,
        update: super::session_options::UpdateKwargs,
        providers: std::collections::BTreeMap<String, Option<crate::HostRef>>,
        #[host(kw, default = false)] interruption_mode_given: bool,
    ) -> Result<
        std::collections::BTreeMap<String, Option<crate::HostRef>>,
        super::session::CommandRefused,
    >;
    #[host(call)]
    fn override_options(
        &self,
        update: super::session_options::UpdateKwargs,
        providers: std::collections::BTreeMap<String, Option<crate::HostRef>>,
        #[host(kw, default = false)] interruption_mode_given: bool,
    ) -> Result<SessionOverride, super::session::CommandRefused>;
    #[host(call)]
    fn release_override(
        &self,
        layer: u64,
    ) -> std::collections::BTreeMap<String, Option<crate::HostRef>>;
    #[host(call)]
    fn install_providers(
        &self,
        bindings: ProviderBindings,
    ) -> Result<(), super::session::SessionRefused>;
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
}
