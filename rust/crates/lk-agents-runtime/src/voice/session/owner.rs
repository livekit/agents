//! Executes admitted lifetime operations. The owner exists before any room or provider opens;
//! cancelling a host waiter does not abandon startup or finalization.

use super::{SessionOptions, VoiceSession, options::Providers, storage::SessionStorage};
use crate::recording::SessionRecorder;
use crate::telemetry;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::recorder::{AudioRecorder, RecordingFailure, SessionRecorder as _};
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_abi::voice::events::{HostWire, SessionErrorWire};
use lk_agents_abi::voice::session::{
    SessionOptions as _, SessionRefused, UnknownCloseReason, VoiceSession as _,
};
use lk_agents_abi::voice::session_effects::{DeliveredState, SessionEvents};
use lk_agents_abi::voice::session_options::{
    EndpointingKwargs, Expressive, InterruptionKwargs, RecordingOptions, ResolvedSessionOptions,
    SessionLegs, TextTransformSet, TurnDetectionKwarg, UpdateKwargs,
};
use lk_agents_abi::voice::session_owner::SessionOwner as _;
use lk_agents_abi::voice::session_owner::{
    OptionsRefused, SessionBindings, SessionBindingsPort, SessionHost,
};
use lk_agents_abi::voice::session_storage::SessionStorage as _;
use lk_agents_abi::{BoxFuture, HostRef, Maybe};
use lk_agents_net::{env::ProcessEnv, task::TaskSet};
use lk_agents_voice::agent_activity::options::{RECORDING_ALL_OFF, validate_session_update};
use lk_agents_voice::agent_session::lifecycle::{
    CloseOwner, OwnerLifecycle, SessionStartPlans, StartOwner,
};
use parking_lot::Mutex;
use std::sync::{Arc, OnceLock, Weak};
use tokio::sync::{OnceCell, watch};

#[cfg(test)]
#[path = "owner_tests.rs"]
mod tests;

pub struct Prepared(SessionBindings);

impl SessionBindingsPort for Prepared {
    fn bindings(&self) -> SessionBindings {
        self.0.clone()
    }
}

impl lk_agents_abi::voice::session_owner::SessionPreparation for Prepared {
    fn new(
        room: Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>,
        args: lk_agents_abi::voice::session::VoiceSessionArgs,
        record_audio: bool,
    ) -> Self {
        Self(SessionBindings { room, args, record_audio })
    }
}

pub struct SessionOwner(Arc<Inner>);

/// The report's event retention, through its declared operations.
type Retention = lk_agents_net::telemetry::session_report::EventRetention;

struct Inner {
    env: ProcessEnv,
    options: SessionOptions,
    lifecycle: Mutex<OwnerLifecycle>,
    startup: OnceLock<Startup>,
    toolsets: OnceLock<Arc<dyn lk_agents_abi::llm::toolsets::ScopesPort>>,
    session: OnceLock<Arc<VoiceSession>>,
    recorder: OnceLock<Arc<dyn AudioRecorder>>,
    /// What a job's report reads of the session, set as its startup prepares it.
    started_at: OnceLock<f64>,
    recording_options: OnceLock<RecordingOptions>,
    audio: OnceLock<(String, Arc<SessionRecorder>)>,
    /// The report's events: which are kept, and the job allowed to read them.
    retention: Mutex<Retention>,
    tasks: Arc<TaskSet>,
    opened: watch::Sender<Option<Result<(), HostError>>>,
    closed: watch::Sender<bool>,
    recording: OnceCell<Result<(), RecordingFailure>>,
    selection: Mutex<Selection>,
    /// `update_agent(chat_ctx=)` before `start()`: the history a session started without one
    /// opens on.
    start_history: Mutex<Option<Vec<lk_agents_abi::ChatItem>>>,
}

#[derive(Clone)]
struct Startup {
    storage: Arc<SessionStorage>,
    host: Arc<dyn SessionHost>,
    /// The span current at `start()` and the job: where the session's spans go.
    trace: telemetry::Cx,
    job_ids: telemetry::JobIds,
    /// The session's spans, reserved at `start()` so startup work runs under them.
    opening: telemetry::Opening,
}

#[derive(Default)]
struct Selection {
    policy: lk_agents_voice::agent_session::lifecycle::AgentSelection,
    pending: Option<lk_agents_abi::HostRef>,
}

impl Inner {
    fn select(&self, agent: Option<lk_agents_abi::HostRef>) -> u64 {
        let (id, retired) = {
            let mut selection = self.selection.lock();
            let id = selection.policy.begin();
            (id, std::mem::replace(&mut selection.pending, agent))
        };
        drop(retired);
        id
    }
    fn selected(&self, id: u64) {
        let retired = {
            let mut selection = self.selection.lock();
            if selection.policy.settled(id) { selection.pending.take() } else { None }
        };
        drop(retired);
    }
}

/// Native close effects settle resources even if a language event listener raises.
struct Events {
    owner: Weak<Inner>,
    host: Option<Arc<dyn SessionEvents>>,
}

impl SessionEvents for Events {
    fn event(&self, wire: HostWire, state: Arc<dyn DeliveredState>) {
        if let HostWire::Close(close) = &wire
            && let Some(owner) = self.owner.upgrade()
            && let Err(error) = SessionOwner(owner).shutdown(close.reason.to_string(), false)
        {
            log::warn!("the session owner could not begin cleanup: {error}");
        }
        if let Some(host) = &self.host {
            host.event(wire, state);
        }
    }

    fn history_items(
        &self,
        items: Vec<lk_agents_abi::ChatItem>,
    ) -> Result<Vec<lk_agents_abi::HostRef>, HostError> {
        match &self.host {
            Some(host) => host.history_items(items),
            None => Ok(Vec::new()),
        }
    }

    fn error_message(&self, error: &SessionErrorWire) -> Result<String, HostError> {
        match &self.host {
            Some(host) => host.error_message(error),
            None => Ok(error.error.to_string()),
        }
    }
}

impl Inner {
    async fn open(self: &Arc<Self>, startup: Startup) -> Result<(), HostError> {
        let Startup { storage, host, trace, job_ids, mut opening } = startup;
        let ready =
            storage.open_ready().await.map_err(|error| HostError::new(error.to_string()))?;
        // `gen_ai.conversation.id` is the call's thread when it has one, else the job's.
        let trace = match ready.thread() {
            Some(thread) => trace.with_conversation(Some(&thread)),
            None => trace,
        };
        // Startup work runs under `session_start`, reserved before the driver opens it.
        let starting = trace.clone().under::<telemetry::SessionStart>(opening.start.context);
        let bindings = host.prepare(&starting).await?.bindings();
        let mut args = bindings.args;
        self.join_job(host.as_ref(), &mut args, bindings.record_audio)?;
        if let Some(recorder) = self.recorder.get() {
            recorder.ready().await.map_err(|error| HostError::new(error.to_string()))?;
        }
        if args.chat_ctx.is_none() {
            args.chat_ctx = self.start_history.lock().take();
        }
        if let Some(hooks) = args.hooks.clone() {
            // The first agent's activity starts here: its toolsets are set up inside it.
            let activity = telemetry::reserve::<telemetry::StartAgentActivity>(&starting);
            opening.activity = Some(activity);
            let span = starting.under::<telemetry::StartAgentActivity>(activity.context);
            let span = telemetry::start(&span, telemetry::SetupToolsets);
            let scopes = hooks.toolset_scopes().clone();
            let _ = self.toolsets.set(scopes.clone());
            let setup = async {
                scopes.acquire(0, args.session_toolsets.take().unwrap_or_default()).await?;
                if let Some(tools) = &args.tools_host {
                    let scope = args.voice_agent.as_ref().map_or(0, |agent| agent.toolset_scope());
                    scopes.acquire(scope, hooks.toolset_resources(tools)?).await?;
                    args.tools = Some(hooks.tool_infos(tools)?);
                }
                Ok::<_, HostError>(())
            };
            let set_up = setup.await;
            telemetry::end(span, set_up.as_ref());
            set_up?;
        }
        args.on_event =
            Some(Arc::new(Events { owner: Arc::downgrade(self), host: args.on_event.take() }));
        args.storage = ready.voice_store().map(|store| store as Arc<_>);
        if let Some(history) = ready.history() {
            args.chat_ctx = Some(history);
        }
        let initial_agent = args.agent_host.clone();
        let session = Arc::new(
            VoiceSession::new(&self.env, bindings.room, args, Some(self.options.clone()))
                .map_err(|error| HostError::new(error.to_string()))?,
        );
        session.attach_recorder(self.recorder.get().cloned());
        let in_session = trace.clone().under::<telemetry::AgentSession>(opening.session.context);
        session.trace_under(trace, job_ids, opening);
        let _ = self.session.set(session.clone());
        let initial = self.select(initial_agent);
        let started = session.start().map_err(|error| HostError::new(error.to_string()))?.await;
        self.selected(initial);
        started.map_err(|error| match error {
            lk_agents_abi::voice::session::StartFailed::Host(error) => error,
            error => HostError::new(error.to_string()),
        })?;
        host.started(&in_session).await
    }

    /// The session as `args.job`'s, prepared: the job's first session claimed is its primary one,
    /// which records what the job records, serves the remote session protocol and ends the job as
    /// it closes; a later one records nothing and serves no remote session. The console's call
    /// serves it either way. The start time is set, the audio recorder opens on
    /// `<recording_dir>/audio.ogg` when the session records audio, and the job's shutdown closes
    /// the session, in that order.
    fn join_job(
        &self,
        host: &dyn SessionHost,
        args: &mut lk_agents_abi::voice::session::VoiceSessionArgs,
        record_audio: bool,
    ) -> Result<(), HostError> {
        let job = args.job.clone();
        let plan = match &job {
            Some(job) => {
                let primary = job.claim_primary(host.session()?)?;
                let plan = SessionStartPlans::plan_session_start(job.enable_recording()?, !primary);
                job.init_recording(recording_map(&plan.recording))?;
                Some(plan)
            }
            None => None,
        };
        let recording = plan.as_ref().map_or(RECORDING_ALL_OFF, |plan| plan.recording);
        let _ = self.recording_options.set(recording);
        if args.console.is_none() && !plan.as_ref().is_some_and(|plan| plan.hosting) {
            args.remote_session = None;
        }
        if !plan.as_ref().is_some_and(|plan| plan.is_primary) {
            args.job_run = None;
        }
        if args.console.is_some() {
            // The console's call has no job room to connect.
            args.job = None;
        }
        let _ = self.started_at.set(lk_agents_abi::unix_secs(self.env.clock.wall()));
        let Some(job) = job else { return Ok(()) };
        if recording.audio || record_audio {
            let dir = std::path::PathBuf::from(job.recording_dir()?);
            let path = dir.join("audio.ogg").to_string_lossy().into_owned();
            let recorder = SessionRecorder::new(&self.env, path.clone(), 48_000, None, 64_000)
                .map_err(|error| HostError::new(error.to_string()))?;
            let recorder = Arc::new(recorder);
            // Retained before it is ready: even a failed open must be finalized.
            let _ = self.recorder.set(recorder.clone());
            let _ = self.audio.set((path, recorder));
        }
        job.on_shutdown(host.session()?)
    }

    /// Bind the providers whose effective value changed to the running session; before it runs,
    /// the session opens on the options' providers instead.
    fn install(&self, changed: Providers) -> Result<(), OptionsRefused> {
        let (Some(session), Some(startup)) = (self.session.get(), self.startup.get()) else {
            return Ok(());
        };
        if changed.is_empty() {
            return Ok(());
        }
        let bindings = startup.host.bind_providers(changed)?;
        Ok(session.update_models(
            Maybe::NotGiven,
            bindings.tts,
            bindings.stt,
            bindings.vad,
            None,
            bindings.tts_node,
            bindings.stt_node,
        )?)
    }

    async fn finish_recording(&self) -> Result<(), RecordingFailure> {
        let Some(recorder) = self.recorder.get() else { return Ok(()) };
        self.recording.get_or_init(|| recorder.finish()).await.clone()
    }

    async fn cleanup(&self, reason: CloseReason, drain: bool) {
        if let Some(session) = self.session.get() {
            session.close_session(reason, drain).await;
        }
        if let Some(toolsets) = self.toolsets.get()
            && let Err(error) = toolsets.release(None).await
        {
            log::warn!("the session toolsets could not close: {error}");
        }
        if let Some(startup) = self.startup.get() {
            startup.storage.aclose().await;
            startup.host.closed(&telemetry::Cx::NONE).await;
        }
        if let Err(error) = self.finish_recording().await {
            log::warn!("the session recording could not finish: {error}");
        }
    }
}

fn recording_map(options: &RecordingOptions) -> std::collections::BTreeMap<String, bool> {
    let RecordingOptions { audio, traces, logs, transcript, redaction } = *options;
    [
        ("audio", audio),
        ("traces", traces),
        ("logs", logs),
        ("transcript", transcript),
        ("redaction", redaction),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value))
    .collect()
}

fn providers(
    stt: Maybe<Option<HostRef>>,
    tts: Maybe<Option<HostRef>>,
    vad: Maybe<Option<HostRef>>,
) -> Providers {
    [("stt", stt), ("tts", tts), ("vad", vad)]
        .into_iter()
        .filter_map(|(slot, provider)| Some((slot.to_owned(), provider.given()?)))
        .collect()
}

async fn opened(
    mut receiver: watch::Receiver<Option<Result<(), HostError>>>,
) -> Result<(), HostError> {
    loop {
        if let Some(result) = receiver.borrow_and_update().clone() {
            return result;
        }
        receiver.changed().await.map_err(|_| HostError::new("session owner stopped"))?;
    }
}

impl lk_agents_abi::voice::session_owner::SessionOwner for SessionOwner {
    type Env = ProcessEnv;
    type Options = SessionOptions;
    type Storage = SessionStorage;
    type Session = VoiceSession;
    type State = super::delivered::SessionState;

    fn new(env: &ProcessEnv, options: SessionOptions) -> Self {
        Self(Arc::new(Inner {
            env: env.clone(),
            options,
            lifecycle: Mutex::new(OwnerLifecycle::default()),
            startup: OnceLock::new(),
            toolsets: OnceLock::new(),
            session: OnceLock::new(),
            recorder: OnceLock::new(),
            started_at: OnceLock::new(),
            recording_options: OnceLock::new(),
            audio: OnceLock::new(),
            retention: Mutex::new(Retention::process()),
            tasks: Arc::new(TaskSet::default()),
            opened: watch::channel(None).0,
            closed: watch::channel(false).0,
            recording: OnceCell::new(),
            selection: Mutex::new(Selection::default()),
            start_history: Mutex::new(None),
        }))
    }

    fn start(
        &self,
        storage: Arc<SessionStorage>,
        host: Arc<dyn SessionHost>,
        job: Option<lk_agents_abi::voice::session_owner::SessionJobInfo>,
        cx: telemetry::Cx,
    ) -> Result<BoxFuture<'static, Result<(), HostError>>, HostError> {
        let spawner = (self.0.env.spawner)().map_err(HostError::new)?;
        let action = self.0.lifecycle.lock().start();
        match action {
            StartOwner::Closed => return Err(HostError::new("this VoiceSession is closed")),
            StartOwner::Join => {}
            StartOwner::Start => {
                if let Some(job) = &job {
                    // The report belongs to the job the session first started in.
                    self.0.retention.lock().claim(job.id.clone());
                }
                let (job, job_ids) = match job {
                    Some(job) => (
                        Some(telemetry::JobKey(job.id)),
                        telemetry::JobIds {
                            dispatch_id: job.dispatch_id,
                            worker_id: job.worker_id,
                            agent_id: job.agent_id,
                        },
                    ),
                    None => (None, Default::default()),
                };
                let trace = self.0.env.tracer.root().continuing(cx.span()).with_job(job);
                // `agent_session` and `session_start` from `start()`: the room connects and the
                // first agent's toolsets set up under them before the driver opens them.
                let cx = &trace;
                let session = telemetry::reserve::<telemetry::AgentSession>(cx);
                let start = telemetry::reserve::<telemetry::SessionStart>(
                    &cx.clone().under::<telemetry::AgentSession>(session.context),
                );
                let opening = telemetry::Opening { session, start, activity: None };
                let startup = Startup { storage, host, trace, job_ids, opening };
                let _ = self.0.startup.set(startup.clone());
                let inner = self.0.clone();
                self.0.tasks.spawn(
                    spawner.as_ref(),
                    "session startup",
                    Box::pin(async move {
                        let result = inner.open(startup).await;
                        if result.is_err() {
                            // A failed open is terminal and settles every resource acquired so far.
                            let _ = inner.lifecycle.lock().close();
                            inner.cleanup(CloseReason::Error, false).await;
                            inner.closed.send_replace(true);
                        }
                        inner.opened.send_replace(Some(result));
                    }),
                );
            }
        }
        Ok(Box::pin(opened(self.0.opened.subscribe())))
    }

    fn session(&self) -> Option<Arc<VoiceSession>> {
        self.0.session.get().cloned()
    }

    fn span(&self) -> Option<telemetry::SpanContext> {
        Some(self.0.startup.get()?.opening.session.context)
    }

    fn storage(&self) -> Option<Arc<SessionStorage>> {
        self.0.startup.get().map(|startup| startup.storage.clone())
    }

    fn current_agent(&self) -> Option<lk_agents_abi::HostRef> {
        let pending = self.0.selection.lock().pending.clone();
        pending.or_else(|| self.0.session.get().and_then(|session| session.current_agent()))
    }

    fn state(&self) -> Arc<super::delivered::SessionState> {
        self.0.session.get().map(|session| session.delivered()).unwrap_or_default()
    }

    fn usage(&self) -> lk_agents_abi::usage::Usage {
        self.0.session.get().map(|session| session.usage()).unwrap_or_default()
    }

    fn keyterms(&self) -> Vec<String> {
        self.0.session.get().and_then(|session| session.keyterms()).unwrap_or_default()
    }

    fn update_agent(
        &self,
        agent: lk_agents_abi::HostRef,
        frame: Option<lk_agents_abi::voice::session::UpdateAgentArgs>,
        chat_ctx: Option<Vec<lk_agents_abi::ChatItem>>,
    ) -> Result<(), SessionRefused> {
        let Some(session) = self.0.session.get() else {
            if chat_ctx.is_some() {
                *self.0.start_history.lock() = chat_ctx;
            }
            self.0.select(Some(agent));
            return Ok(());
        };
        let spawner = (self.0.env.spawner)().map_err(|_| SessionRefused::Closed)?;
        let mut frame = frame.ok_or(SessionRefused::NotStarted)?;
        if chat_ctx.is_some() {
            frame.chat_ctx = chat_ctx;
        }
        let installed = session.update_agent(frame)?;
        let id = self.0.select(Some(agent));
        let inner = self.0.clone();
        self.0.tasks.spawn(spawner.as_ref(), "agent selection", async move {
            let _ = installed.await;
            inner.selected(id);
        });
        Ok(())
    }

    fn update_options(
        &self,
        stt: Maybe<Option<HostRef>>,
        tts: Maybe<Option<HostRef>>,
        vad: Maybe<Option<HostRef>>,
        turn_detection: Maybe<TurnDetectionKwarg>,
        endpointing_opts: Maybe<EndpointingKwargs>,
        interruption: Maybe<InterruptionKwargs>,
        min_consecutive_speech_delay: Maybe<f64>,
        tts_text_transforms: Maybe<Option<TextTransformSet>>,
        expressive: Maybe<Expressive>,
        keyterms: Maybe<Vec<String>>,
    ) -> Result<(), OptionsRefused> {
        let update = UpdateKwargs {
            endpointing_opts: endpointing_opts.given(),
            turn_detection,
            expressive,
            keyterms: keyterms.given(),
            interruption: interruption.given(),
            min_consecutive_speech_delay: min_consecutive_speech_delay.given(),
            tts_text_transforms,
        };
        validate_session_update(&update)?;
        if let Some(session) = self.0.session.get() {
            session.update_turn_options(update)?;
        } else {
            self.0.options.update(update);
        }
        self.0.install(self.0.options.update_providers(providers(stt, tts, vad)))
    }

    fn override_options(
        &self,
        stt: Maybe<Option<HostRef>>,
        tts: Maybe<Option<HostRef>>,
        vad: Maybe<Option<HostRef>>,
        turn_detection: Maybe<TurnDetectionKwarg>,
        endpointing_opts: Maybe<EndpointingKwargs>,
        interruption: Maybe<InterruptionKwargs>,
        min_consecutive_speech_delay: Maybe<f64>,
        tts_text_transforms: Maybe<Option<TextTransformSet>>,
    ) -> Result<u64, OptionsRefused> {
        let update = UpdateKwargs {
            endpointing_opts: endpointing_opts.given(),
            turn_detection,
            interruption: interruption.given(),
            min_consecutive_speech_delay: min_consecutive_speech_delay.given(),
            tts_text_transforms,
            ..UpdateKwargs::default()
        };
        validate_session_update(&update)?;
        let layer = match self.0.session.get() {
            Some(session) => session.override_options(update)?,
            None => self.0.options.override_options(update),
        };
        let installed =
            self.0.install(self.0.options.override_providers(layer, providers(stt, tts, vad)));
        if let Err(error) = installed {
            // The block never opens: its layer is not left held.
            let _ = self.release_override(layer);
            return Err(error);
        }
        Ok(layer)
    }

    fn release_override(&self, layer: u64) -> Result<(), OptionsRefused> {
        match self.0.session.get() {
            Some(session) => session.release_override(layer),
            None => self.0.options.release_override(layer),
        }
        self.0.install(self.0.options.release_providers(layer))
    }

    fn resolved_options(&self, legs: SessionLegs) -> ResolvedSessionOptions {
        self.0
            .session
            .get()
            .and_then(|session| session.resolved_options())
            .unwrap_or_else(|| self.0.options.resolved(legs))
    }

    fn aclose(
        &self,
        reason: String,
        drain: bool,
    ) -> Result<BoxFuture<'static, ()>, UnknownCloseReason> {
        let reason = CloseReason::from_wire(&reason).ok_or(UnknownCloseReason(reason))?;
        let spawner = (self.0.env.spawner)().map_err(UnknownCloseReason)?;
        let action = self.0.lifecycle.lock().close();
        if let CloseOwner::Close { wait_start } = action {
            let inner = self.0.clone();
            self.0.tasks.spawn(
                spawner.as_ref(),
                "session cleanup",
                Box::pin(async move {
                    if wait_start && opened(inner.opened.subscribe()).await.is_err() {
                        return; // Failed startup already performed cleanup.
                    }
                    inner.cleanup(reason, drain).await;
                    inner.closed.send_replace(true);
                }),
            );
        }
        let mut receiver = self.0.closed.subscribe();
        Ok(Box::pin(async move {
            let _ = receiver.wait_for(|closed| *closed).await;
        }))
    }

    fn shutdown(&self, reason: String, drain: bool) -> Result<(), UnknownCloseReason> {
        // aclose admits and owns the work before returning its wait-only future.
        drop(self.aclose(reason, drain)?);
        Ok(())
    }

    async fn finish_recording(&self) -> Result<(), RecordingFailure> {
        self.0.finish_recording().await
    }

    fn started_at(&self) -> Option<f64> {
        self.0.started_at.get().copied()
    }

    fn recording_options(&self) -> RecordingOptions {
        self.0.recording_options.get().copied().unwrap_or(RECORDING_ALL_OFF)
    }

    fn recording_path(&self) -> Option<String> {
        self.0.audio.get().map(|(path, _)| path.clone())
    }

    fn recording_started_at(&self) -> Option<f64> {
        self.0.audio.get().and_then(|(_, recorder)| recorder.started_at())
    }

    fn allows_report(&self, job_id: String) -> bool {
        self.0.retention.lock().allows_read(&job_id)
    }

    fn accepts_event(&self, kind: String) -> bool {
        let redaction = self.recording_options().redaction;
        self.0.retention.lock().accepts(&kind, redaction)
    }

    fn admit_event(&self, kind: String) -> bool {
        let redaction = self.recording_options().redaction;
        let mut retention = self.0.retention.lock();
        let accepted = retention.accepts(&kind, redaction);
        if accepted {
            retention.admit();
        }
        accepted
    }
}
