use super::{SessionOptions, VoiceSession, options::Providers, storage::SessionStorage};
use crate::recording::SessionRecorder;
use crate::telemetry;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::llm::toolsets::SESSION_TOOLSET_SCOPE;
use lk_agents_abi::recorder::{AudioRecorder, RecordingFailure, SessionRecorder as _};
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_abi::voice::events::{HostWire, SessionErrorWire};
use lk_agents_abi::voice::session::{
    SessionOptions as _, SessionRefused, StartFailed, UnknownCloseReason, VoiceSession as _,
};
use lk_agents_abi::voice::session_effects::{DeliveredState, SessionEvents};
use lk_agents_abi::voice::session_options::{
    Expressive, RecordingOptions, ResolvedSessionOptions, SessionLegs, TurnHandlingKwargs,
    UpdateKwargs,
};
use lk_agents_abi::voice::session_owner::SessionOwner as _;
use lk_agents_abi::voice::session_owner::{
    OptionsRefused, SessionBindings, SessionBindingsPort, SessionHost,
};
use lk_agents_abi::voice::session_storage::SessionStorage as _;
use lk_agents_abi::worker::ipc_host::FrameworkShutdown;
use lk_agents_abi::{BoxFuture, HostRef, Maybe};
use lk_agents_net::signal;
use lk_agents_net::{env::ProcessEnv, task::TaskSet};
use lk_agents_voice::agent_session::lifecycle::{
    CloseOwner, OwnerLifecycle, SessionStartPlans, StartOwner,
};
use lk_agents_voice::agent_session::options::{RECORDING_ALL_OFF, validate_session_update};
use parking_lot::Mutex;
use std::sync::{Arc, OnceLock, Weak};
use tokio::sync::OnceCell;

// Cancelling a host waiter does not abandon startup or finalization.

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

impl SessionOwner {
    pub(crate) fn avatars(&self) -> &crate::avatar::lifecycle::Avatars {
        &self.0.avatars
    }

    /// `session`'s history, item turns and usage reported for `job`, with what the owner holds of
    /// the session now.
    fn assemble(
        &self,
        job: lk_agents_abi::telemetry::report::ReportJob,
        legs: SessionLegs,
        events: Vec<serde_json::Value>,
        session: telemetry::SessionReport,
    ) -> telemetry::SessionReport {
        let timestamp = lk_agents_abi::unix_secs(self.0.env.clock.wall());
        let recording_started_at = self.recording_started_at();
        telemetry::SessionReport {
            job_id: job.job_id,
            room_id: job.room_id,
            room: job.room,
            agent_name: job.agent_name,
            options: lk_agents_core::telemetry::report::options(
                &self.resolved_options(legs),
                &self.recording_options(),
            ),
            events,
            started_at: self.started_at(),
            timestamp,
            duration: recording_started_at.map(|at| timestamp - at),
            // A failed recording's file is truncated: the report offers no audio.
            audio_recording_path: self
                .0
                .audio
                .get()
                .filter(|(_, recorder)| !recorder.failed())
                .map(|(path, _)| path.clone()),
            audio_recording_started_at: recording_started_at,
            sdk_version: job.sdk_version,
            ..session
        }
    }
}

/// The report's event retention, through its declared operations.
type Retention = lk_agents_net::export::session_report::EventRetention;

/// The startup's outcome, once it ran.
type Opened = Option<Result<(), StartFailed>>;

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
    /// The console's call recorded with `--record`: its recording is kept past the job.
    console_recording: OnceLock<bool>,
    /// The report's events: which are kept, and the job allowed to read them.
    retention: Mutex<Retention>,
    tasks: Arc<TaskSet>,
    /// The startup task's: it ends with the task, however it ends.
    opened: signal::Cell<Opened>,
    open_tx: Mutex<Option<signal::Producer<Opened>>>,
    /// The close owner's (the first close, or a failed startup): it ends with its task.
    closed: signal::Cell<bool>,
    close_tx: Mutex<Option<signal::Producer<bool>>>,
    recording: OnceCell<Result<(), RecordingFailure>>,
    selection: Mutex<Selection>,
    /// `update_agent(chat_ctx=)` before `start()`: the history a session started without one
    /// opens on.
    start_history: Mutex<Option<Vec<lk_agents_abi::ChatItem>>>,
    /// The avatars `AvatarSession.start()` attached.
    avatars: crate::avatar::lifecycle::Avatars,
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
        if let HostWire::ItemAdded(added) = &wire
            && let Some(owner) = self.owner.upgrade()
        {
            let at = lk_agents_abi::unix_secs(owner.env.clock.wall());
            crate::avatar::lifecycle::item_added(&SessionOwner(owner), &added.item, at);
        }
        if let HostWire::Close(close) = &wire
            && let Some(owner) = self.owner.upgrade()
            && let Err(error) = SessionOwner(owner).shutdown(Some(close.reason), false)
        {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "the session owner could not begin cleanup: {error}");
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

    fn ended(&self) {
        if let Some(host) = &self.host {
            host.ended();
        }
    }
}

impl Inner {
    /// Admits the close and owns its work; the future waits until the session closed.
    fn close(
        self: &Arc<Self>,
        reason: CloseReason,
        drain: bool,
    ) -> Result<BoxFuture<'static, ()>, HostError> {
        let spawner = (self.env.spawner)().map_err(|e| HostError::new(e.to_string()))?;
        let action = self.lifecycle.lock().close();
        if let CloseOwner::Close { wait_start } = action {
            let inner = self.clone();
            let closed = self.close_tx.lock().take();
            self.tasks.spawn(
                spawner.as_ref(),
                "session cleanup",
                Box::pin(async move {
                    if wait_start && opened(&inner.opened).await.is_err() {
                        return; // Failed startup already performed cleanup.
                    }
                    inner.cleanup(reason, drain).await;
                    if let Some(closed) = closed {
                        closed.finish(true);
                    }
                }),
            );
        }
        let closed = self.closed.clone();
        Ok(Box::pin(async move {
            // Ended: the close owner is gone, so there is nothing left to wait for.
            closed.until(|closed| *closed).await;
        }))
    }

    async fn open(self: &Arc<Self>, startup: Startup) -> Result<(), StartFailed> {
        let Startup { storage, host, trace, job_ids, mut opening } = startup;
        let ready = storage.open_ready(&trace).await?;
        // `gen_ai.conversation.id` is the call's thread.
        let trace = trace.with_conversation(ready.thread());
        // Startup work runs under `session_start`, reserved before the driver opens it.
        let starting = trace.clone().under::<telemetry::SessionStart>(opening.start.context);
        let bindings = host.prepare(&starting).await?.bindings();
        let mut args = bindings.args;
        self.join_job(host.as_ref(), &mut args, bindings.record_audio)?;
        if let Some(recorder) = self.recorder.get() {
            recorder.ready().await?;
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
                    let scope = args
                        .voice_agent
                        .as_ref()
                        .map_or(SESSION_TOOLSET_SCOPE, |agent| agent.toolset_scope());
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
        args.storage = ready.voice_store().map(|store| store as _);
        if let Some(history) = ready.history() {
            args.chat_ctx = Some(history);
        }
        let initial_agent = args.agent_host.clone();
        let job_run = args.job_run.clone();
        let session = Arc::new(
            VoiceSession::new(&self.env, bindings.room, args, Some(self.options.clone()))
                .map_err(|error| HostError::new(error.to_string()))?,
        );
        session.attach_recorder(self.recorder.get().cloned());
        let in_session = trace.clone().under::<telemetry::AgentSession>(opening.session.context);
        if let Some(run) = &job_run {
            run.session_started(in_session.clone());
        }
        session.trace_under(trace, job_ids, opening);
        let _ = self.session.set(session.clone());
        let initial = self.select(initial_agent);
        let started = session.start().map_err(|error| HostError::new(error.to_string()))?.await;
        self.selected(initial);
        started?;
        Ok(host.started(&in_session).await?)
    }

    /// A start that failed: logged once, and the session's reserved spans end with it unless its
    /// driver opened them.
    fn start_failed(&self, error: &StartFailed) {
        let Some(startup) = self.startup.get() else { return };
        let job = startup.trace.job().map(|job| job.0.as_str());
        let thread = startup.storage.supplied_thread();
        log::error!(target: lk_agents_abi::LOG_TARGET, job_id = job, thread_id = thread; "the session could not start: {error}");
        if self.session.get().is_some() {
            return;
        }
        let trace = &startup.trace;
        let supplied = telemetry::SessionTelemetry {
            parent: trace.span(),
            job: trace.job().cloned(),
            conversation: thread.map(str::to_owned),
            job_ids: startup.job_ids.clone(),
            ids: Some(telemetry::seeded_ids(telemetry::random_seed())),
            opening: Some(startup.opening),
        };
        let now = lk_agents_voice::agent_session::Now {
            at: self.env.clock.now(),
            wall: self.env.clock.wall(),
        };
        let agent = startup.storage.agent();
        for record in lk_agents_voice::agent_session::failed_start(supplied, agent, now, error) {
            self.env.tracer.record(record);
        }
    }

    /// The session as `args.job`'s, prepared: the job's first session claimed is its primary one,
    /// which records what the job records, serves the remote session protocol and ends the job as
    /// it closes; a later one records nothing and serves no remote session. The console's call
    /// serves it either way. The start time is set, the audio recorder opens on
    /// `<recording_dir>/audio.ogg` when the session records audio, and the job's shutdown closes
    /// the session, in that order.
    fn join_job(
        self: &Arc<Self>,
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
        let _ = self.console_recording.set(args.console.is_some() && record_audio);
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
        let owner = Arc::downgrade(self);
        let close = move || -> BoxFuture<'static, ()> {
            Box::pin(async move {
                let closing =
                    owner.upgrade().map(|inner| inner.close(CloseReason::JobShutdown, false));
                if let Some(Ok(closed)) = closing {
                    closed.await;
                }
            })
        };
        job.run().add_framework_shutdown(
            "VoiceSession.aclose".into(),
            FrameworkShutdown::Native(Box::new(close)),
        );
        Ok(())
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
            log::warn!(target: lk_agents_abi::LOG_TARGET, "the session toolsets could not close: {error}");
        }
        if let Some(startup) = self.startup.get() {
            startup.storage.aclose().await;
            if let Err(error) = startup.host.closed(&startup.trace).await {
                log::warn!(target: lk_agents_abi::LOG_TARGET, "the session's closed callback raised: {error}");
            }
        }
        if let Err(error) = self.finish_recording().await {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "the session recording could not finish: {error}");
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

async fn opened(opened: &signal::Cell<Opened>) -> Result<(), StartFailed> {
    opened.wait().await.map_err(|_| HostError::new("session owner stopped"))?
}

impl lk_agents_abi::voice::session_owner::SessionOwner for SessionOwner {
    type Env = ProcessEnv;
    type Options = SessionOptions;
    type Storage = SessionStorage;
    type Session = VoiceSession;
    type State = super::delivered::SessionState;

    fn new(env: &ProcessEnv, options: SessionOptions) -> Self {
        let (open_tx, opened) = signal::once();
        let (close_tx, closed) = signal::signal(false);
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
            console_recording: OnceLock::new(),
            retention: Mutex::new(Retention::process()),
            tasks: Arc::new(TaskSet::default()),
            opened,
            open_tx: Mutex::new(Some(open_tx)),
            closed,
            close_tx: Mutex::new(Some(close_tx)),
            recording: OnceCell::new(),
            selection: Mutex::new(Selection::default()),
            start_history: Mutex::new(None),
            avatars: Default::default(),
        }))
    }

    fn start(
        &self,
        storage: Arc<SessionStorage>,
        host: Arc<dyn SessionHost>,
        job: Option<lk_agents_abi::voice::session_owner::SessionJobInfo>,
        cx: telemetry::Cx,
    ) -> Result<BoxFuture<'static, Result<(), StartFailed>>, HostError> {
        let spawner = (self.0.env.spawner)().map_err(|e| HostError::new(e.to_string()))?;
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
                // Reserved here: the room connects and toolsets set up under them before the
                // driver opens them.
                let cx = &trace;
                let session = telemetry::reserve::<telemetry::AgentSession>(cx);
                let start = telemetry::reserve::<telemetry::SessionStart>(
                    &cx.clone().under::<telemetry::AgentSession>(session.context),
                );
                let opening = telemetry::Opening { session, start, activity: None };
                let startup = Startup { storage, host, trace, job_ids, opening };
                let _ = self.0.startup.set(startup.clone());
                let inner = self.0.clone();
                let open_tx = self.0.open_tx.lock().take();
                self.0.tasks.spawn(
                    spawner.as_ref(),
                    "session startup",
                    Box::pin(async move {
                        let result = inner.open(startup).await;
                        if let Err(error) = &result {
                            inner.start_failed(error);
                            // A failed open is terminal and settles every resource acquired so far.
                            let owner = inner.lifecycle.lock().close();
                            let closed = matches!(owner, CloseOwner::Close { .. })
                                .then(|| inner.close_tx.lock().take())
                                .flatten();
                            inner.cleanup(CloseReason::Error, false).await;
                            if let Some(closed) = closed {
                                closed.finish(true);
                            }
                        }
                        if let Some(open_tx) = open_tx {
                            open_tx.resolve(result);
                        }
                    }),
                );
            }
        }
        let inner = self.0.clone();
        Ok(Box::pin(async move { opened(&inner.opened).await }))
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

    fn current_agent(&self) -> Result<lk_agents_abi::HostRef, SessionRefused> {
        let pending = self.0.selection.lock().pending.clone();
        pending
            .or_else(|| self.0.session.get().and_then(|session| session.current_agent()))
            .ok_or(SessionRefused::NotStarted)
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
            match installed.await {
                // Either answer leaves the session's own agent current: the new one, or the one
                // a refused handoff kept.
                true | false => inner.selected(id),
            }
        });
        Ok(())
    }

    fn update_options(
        &self,
        stt: Maybe<Option<HostRef>>,
        tts: Maybe<Option<HostRef>>,
        vad: Maybe<Option<HostRef>>,
        turn_handling: Maybe<TurnHandlingKwargs>,
        min_consecutive_speech_delay: Maybe<f64>,
        expressive: Maybe<Expressive>,
        keyterms: Maybe<Vec<String>>,
    ) -> Result<(), OptionsRefused> {
        let update = UpdateKwargs {
            turn_handling: turn_handling.given(),
            expressive,
            keyterms: keyterms.given(),
            min_consecutive_speech_delay: min_consecutive_speech_delay.given(),
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
        turn_handling: Maybe<TurnHandlingKwargs>,
        min_consecutive_speech_delay: Maybe<f64>,
    ) -> Result<u64, OptionsRefused> {
        let update = UpdateKwargs {
            turn_handling: turn_handling.given(),
            min_consecutive_speech_delay: min_consecutive_speech_delay.given(),
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
        reason: Option<CloseReason>,
        drain: bool,
    ) -> Result<BoxFuture<'static, ()>, UnknownCloseReason> {
        let reason = reason.unwrap_or(CloseReason::UserInitiated);
        self.0.close(reason, drain).map_err(|e| UnknownCloseReason(e.to_string()))
    }

    fn shutdown(&self, reason: Option<CloseReason>, drain: bool) -> Result<(), UnknownCloseReason> {
        // aclose admits and owns the work before returning its wait-only future.
        drop(self.aclose(reason, drain)?);
        Ok(())
    }

    async fn finish_recording(&self) -> Result<(), RecordingFailure> {
        self.0.finish_recording().await
    }

    async fn keep_console_recording(
        &self,
        report: telemetry::SessionReport,
    ) -> Result<Option<String>, RecordingFailure> {
        if self.0.console_recording.get() != Some(&true) {
            return Ok(None);
        }
        use lk_agents_abi::telemetry::report::SessionReportFunctions as _;
        let json = lk_agents_net::export::session_report::ReportFiles::session_report_json(report)
            .map_err(|error| RecordingFailure::Encode(error.to_string()))?;
        let stamp = chrono::DateTime::<chrono::Local>::from(self.0.env.clock.wall());
        let directory = std::path::Path::new(".livekit/sessions")
            .join(stamp.format("%Y-%m-%d-%H%M%S").to_string());
        let audio = self.0.audio.get().map(|(path, _)| std::path::PathBuf::from(path));
        let kept = directory.clone();
        lk_agents_net::blocking_io::blocking_io()
            .run((), move || -> std::io::Result<()> {
                std::fs::create_dir_all(&kept)?;
                std::fs::write(kept.join("session_report.json"), json)?;
                if let Some(audio) = audio.filter(|audio| audio.is_file()) {
                    std::fs::copy(audio, kept.join("audio.ogg"))?;
                }
                Ok(())
            })
            .await
            .map_err(|lost| RecordingFailure::Worker(lost.to_string()))?
            .map_err(|error| RecordingFailure::Io(error.to_string()))?;
        Ok(Some(directory.to_string_lossy().into_owned()))
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

    fn report(
        &self,
        job: lk_agents_abi::telemetry::report::ReportJob,
        legs: SessionLegs,
        events: Vec<serde_json::Value>,
        history: Vec<lk_agents_abi::ChatItem>,
    ) -> telemetry::SessionReport {
        let session = telemetry::SessionReport {
            chat_history: history,
            usage: self.usage(),
            ..telemetry::SessionReport::default()
        };
        self.assemble(job, legs, events, session)
    }

    fn final_report(
        &self,
        job: lk_agents_abi::telemetry::report::ReportJob,
        legs: SessionLegs,
        events: Vec<serde_json::Value>,
    ) -> Result<BoxFuture<'static, telemetry::SessionReport>, SessionRefused> {
        let read = self.0.session.get().map(|session| session.report()).transpose()?;
        let owner = SessionOwner(self.0.clone());
        Ok(Box::pin(async move {
            let session = match read {
                Some(read) => read.await,
                None => telemetry::SessionReport::default(),
            };
            owner.assemble(job, legs, events, session)
        }))
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
