//! Executes admitted lifetime operations. The owner exists before any room or provider opens;
//! cancelling a host waiter does not abandon startup or finalization.

use super::{SessionOptions, VoiceSession, storage::SessionStorage};
use lk_agents_abi::BoxFuture;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::recorder::{AudioRecorder, RecordingFailure};
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_abi::voice::events::{HostWire, SessionErrorWire};
use lk_agents_abi::voice::session::{
    CommandRefused, SessionOptions as _, SessionRefused, UnknownCloseReason, VoiceSession as _,
};
use lk_agents_abi::voice::session_effects::SessionEvents;
use lk_agents_abi::voice::session_options::{ResolvedSessionOptions, SessionLegs, UpdateKwargs};
use lk_agents_abi::voice::session_owner::SessionOwner as _;
use lk_agents_abi::voice::session_owner::{
    ProviderBindings, SessionBindings, SessionBindingsPort, SessionHost, SessionOverride,
};
use lk_agents_abi::voice::session_storage::SessionStorage as _;
use lk_agents_net::{env::ProcessEnv, task::TaskSet};
use lk_agents_voice::session_start::{CloseOwner, OwnerLifecycle, StartOwner};
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
        recorder: Option<Arc<dyn AudioRecorder>>,
    ) -> Self {
        Self(SessionBindings { room, args, recorder })
    }
}

pub struct SessionOwner(Arc<Inner>);

struct Inner {
    env: ProcessEnv,
    options: SessionOptions,
    lifecycle: Mutex<OwnerLifecycle>,
    storage: OnceLock<Arc<SessionStorage>>,
    host: OnceLock<Arc<dyn SessionHost>>,
    toolsets: OnceLock<Arc<dyn lk_agents_abi::llm::toolsets::ScopesPort>>,
    session: OnceLock<Arc<VoiceSession>>,
    recorder: OnceLock<Arc<dyn AudioRecorder>>,
    tasks: Arc<TaskSet>,
    opened: watch::Sender<Option<Result<(), HostError>>>,
    closed: watch::Sender<bool>,
    recording: OnceCell<Result<(), RecordingFailure>>,
    selection: Mutex<Selection>,
}

#[derive(Default)]
struct Selection {
    policy: lk_agents_voice::session_start::AgentSelection,
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
    fn event(&self, wire: HostWire) {
        if let HostWire::Close(close) = &wire
            && let Some(owner) = self.owner.upgrade()
            && let Err(error) = SessionOwner(owner).shutdown(close.reason.to_string(), false)
        {
            log::warn!("the session owner could not begin cleanup: {error}");
        }
        if let Some(host) = &self.host {
            host.event(wire);
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
    async fn open(self: &Arc<Self>) -> Result<(), HostError> {
        let storage = self.storage.get().expect("admitted startup owns storage");
        storage.open().await.map_err(|error| HostError::new(error.to_string()))?;
        let host = self.host.get().expect("admitted startup owns its host");
        let bindings = host.prepare().await?.bindings();
        if let Some(recorder) = bindings.recorder {
            // Retain it before waiting for readiness: even a failed open must be finalized.
            let _ = self.recorder.set(recorder.clone());
            recorder.ready().await.map_err(|error| HostError::new(error.to_string()))?;
        }
        let mut args = bindings.args;
        if let Some(hooks) = &args.hooks {
            let scopes = hooks.toolset_scopes().clone();
            let _ = self.toolsets.set(scopes.clone());
            scopes.acquire(0, args.session_toolsets.take().unwrap_or_default()).await?;
            if let Some(tools) = &args.tools_host {
                let scope = args.voice_agent.as_ref().map_or(0, |agent| agent.toolset_scope());
                scopes.acquire(scope, hooks.toolset_resources(tools)?).await?;
                args.tools = Some(hooks.tool_infos(tools)?);
            }
        }
        args.on_event =
            Some(Arc::new(Events { owner: Arc::downgrade(self), host: args.on_event.take() }));
        args.storage = storage.voice_store().map(|store| store as Arc<_>);
        if let Some(history) = storage.history() {
            args.chat_ctx = Some(history);
        }
        let initial_agent = args.agent_host.clone();
        let session = Arc::new(
            VoiceSession::new(&self.env, bindings.room, args, Some(self.options.clone()))
                .map_err(|error| HostError::new(error.to_string()))?,
        );
        session.attach_recorder(self.recorder.get().cloned());
        let _ = self.session.set(session.clone());
        let initial = self.select(initial_agent);
        let started = session.start().map_err(|error| HostError::new(error.to_string()))?.await;
        self.selected(initial);
        started.map_err(|error| match error {
            lk_agents_abi::voice::session::StartFailed::Host(error) => error,
            error => HostError::new(error.to_string()),
        })?;
        host.started().await
    }

    async fn finish_recording(&self) -> Result<(), RecordingFailure> {
        let Some(recorder) = self.recorder.get() else { return Ok(()) };
        self.recording.get_or_init(|| recorder.finish()).await.clone()
    }

    async fn cleanup(&self, reason: String, drain: bool) {
        if let Some(session) = self.session.get() {
            // The public entry validated the reason before admitting this effect.
            session.aclose(reason, drain).expect("validated close reason").await;
        }
        if let Some(toolsets) = self.toolsets.get()
            && let Err(error) = toolsets.release(None).await
        {
            log::warn!("the session toolsets could not close: {error}");
        }
        if let Some(storage) = self.storage.get() {
            storage.aclose().await;
        }
        if let Some(host) = self.host.get() {
            host.closed().await;
        }
        if let Err(error) = self.finish_recording().await {
            log::warn!("the session recording could not finish: {error}");
        }
    }
}

async fn opened(
    mut receiver: watch::Receiver<Option<Result<(), HostError>>>,
) -> Result<(), HostError> {
    receiver
        .wait_for(Option::is_some)
        .await
        .map_err(|_| HostError::new("session owner stopped"))?;
    receiver.borrow().clone().expect("startup settled")
}

impl lk_agents_abi::voice::session_owner::SessionOwner for SessionOwner {
    type Env = ProcessEnv;
    type Options = SessionOptions;
    type Storage = SessionStorage;
    type Session = VoiceSession;

    fn new(env: &ProcessEnv, options: SessionOptions) -> Self {
        Self(Arc::new(Inner {
            env: env.clone(),
            options,
            lifecycle: Mutex::new(OwnerLifecycle::default()),
            storage: OnceLock::new(),
            host: OnceLock::new(),
            toolsets: OnceLock::new(),
            session: OnceLock::new(),
            recorder: OnceLock::new(),
            tasks: Arc::new(TaskSet::default()),
            opened: watch::channel(None).0,
            closed: watch::channel(false).0,
            recording: OnceCell::new(),
            selection: Mutex::new(Selection::default()),
        }))
    }

    fn start(
        &self,
        storage: Arc<SessionStorage>,
        host: Arc<dyn SessionHost>,
    ) -> Result<BoxFuture<'static, Result<(), HostError>>, HostError> {
        let spawner = (self.0.env.spawner)().map_err(HostError::new)?;
        let action = self.0.lifecycle.lock().start();
        match action {
            StartOwner::Closed => return Err(HostError::new("this VoiceSession is closed")),
            StartOwner::Join => {}
            StartOwner::Start => {
                let _ = self.0.storage.set(storage);
                let _ = self.0.host.set(host);
                let inner = self.0.clone();
                self.0.tasks.spawn(
                    spawner.as_ref(),
                    "session startup",
                    Box::pin(async move {
                        let result = inner.open().await;
                        if result.is_err() {
                            // A failed open is terminal and settles every resource acquired so far.
                            let _ = inner.lifecycle.lock().close();
                            inner.cleanup("error".to_owned(), false).await;
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

    fn storage(&self) -> Option<Arc<SessionStorage>> {
        self.0.storage.get().cloned()
    }

    fn current_agent(&self) -> Option<lk_agents_abi::HostRef> {
        let pending = self.0.selection.lock().pending.clone();
        pending.or_else(|| self.0.session.get().and_then(|session| session.current_agent()))
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
    ) -> Result<(), SessionRefused> {
        let Some(session) = self.0.session.get() else {
            self.0.select(Some(agent));
            return Ok(());
        };
        let spawner = (self.0.env.spawner)().map_err(|_| SessionRefused::Closed)?;
        let frame = frame.ok_or(SessionRefused::NotStarted)?;
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
        update: UpdateKwargs,
        providers: std::collections::BTreeMap<String, Option<lk_agents_abi::HostRef>>,
        interruption_mode_given: bool,
    ) -> Result<std::collections::BTreeMap<String, Option<lk_agents_abi::HostRef>>, CommandRefused>
    {
        lk_agents_voice::agent_activity::options::validate_session_update(
            &update,
            interruption_mode_given,
        )?;
        if let Some(session) = self.0.session.get() {
            session.update_turn_options(update)?;
        } else {
            self.0.options.update(update);
        }
        Ok(self.0.options.update_providers(providers))
    }

    fn override_options(
        &self,
        update: UpdateKwargs,
        providers: std::collections::BTreeMap<String, Option<lk_agents_abi::HostRef>>,
        interruption_mode_given: bool,
    ) -> Result<SessionOverride, CommandRefused> {
        lk_agents_voice::agent_activity::options::validate_session_update(
            &update,
            interruption_mode_given,
        )?;
        let layer = match self.0.session.get() {
            Some(session) => session.override_options(update)?,
            None => self.0.options.override_options(update),
        };
        Ok(SessionOverride {
            layer,
            providers: self.0.options.override_providers(layer, providers),
        })
    }

    fn release_override(
        &self,
        layer: u64,
    ) -> std::collections::BTreeMap<String, Option<lk_agents_abi::HostRef>> {
        match self.0.session.get() {
            Some(session) => session.release_override(layer),
            None => self.0.options.release_override(layer),
        }
        self.0.options.release_providers(layer)
    }

    fn install_providers(&self, bindings: ProviderBindings) -> Result<(), SessionRefused> {
        match self.0.session.get() {
            Some(session) => session.update_models(
                lk_agents_abi::Maybe::NotGiven,
                bindings.tts,
                bindings.stt,
                bindings.vad,
                None,
                bindings.tts_node,
                bindings.stt_node,
            ),
            None => Ok(()),
        }
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
        if CloseReason::from_wire(&reason).is_none() {
            return Err(UnknownCloseReason(reason));
        }
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
}
