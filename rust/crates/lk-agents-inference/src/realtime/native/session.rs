//! The driver: the only file in this crate that awaits anything; no decision is made here.

use std::sync::{Arc, Weak};
use std::time::Duration;

use lk_agents_abi::Clock;
use lk_agents_abi::RealtimeCommand as Command;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::{ApiConnectOptions, ChatContext};
use lk_agents_abi::{Spawner, SpawnerExt};
use lk_agents_net::transport::{TransportError, WsMessage, WsTransport};
use tokio::sync::{Mutex, mpsc};

use crate::headers::BoundIdentity;
use crate::realtime::dialect::{Dialect, TurnDetection};
use crate::realtime::machine::{OptionsPatch, RealtimeMachine, SessionOptions, Step};
use crate::realtime::native::wire::{self, Azure, IdGen, RandomIds, WireShape};
use lk_agents_abi::llm::realtime::{Event, RealtimeCapabilities};
use lk_agents_net::env::WorkerPool;

/// How a connection authenticates.
pub enum Auth {
    /// OpenAI and xAI: `Authorization: Bearer <api_key>`.
    Bearer(String),
    /// Azure: `api-key`, and an Entra token as the bearer.
    Azure { api_key: Option<String>, entra_token: Option<String> },
    /// LiveKit Inference: a fresh gateway JWT per connection, the job's identification headers,
    /// and the provider the gateway routes to.
    Inference {
        api_key: String,
        api_secret: String,
        provider: Option<String>,
        inference_class: Option<String>,
        identity: BoundIdentity,
    },
}

/// Where and how a session connects.
pub struct Connect {
    pub base_url: String,
    pub azure: Option<Azure>,
    pub auth: Auth,
    pub conn: ApiConnectOptions,
}

impl Connect {
    pub fn openai(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Connect {
            base_url: base_url.into(),
            azure: None,
            auth: Auth::Bearer(api_key.into()),
            conn: ApiConnectOptions::default(),
        }
    }

    pub fn shape(&self) -> WireShape {
        WireShape {
            azure: self.azure.is_some(),
            drop_model: matches!(self.auth, Auth::Inference { .. }),
        }
    }

    /// `_create_ws_url_and_headers`.
    fn headers(&self) -> Result<Vec<(String, String)>, ApiError> {
        let ua = ("User-Agent".to_owned(), "LiveKit Agents".to_owned());
        Ok(match &self.auth {
            Auth::Bearer(key) => vec![ua, ("Authorization".to_owned(), format!("Bearer {key}"))],
            Auth::Azure { api_key, entra_token } => {
                let mut h = vec![ua];
                if let Some(token) = entra_token {
                    h.push(("Authorization".to_owned(), format!("Bearer {token}")));
                }
                if let Some(key) = api_key {
                    h.push(("api-key".to_owned(), key.clone()));
                }
                h
            }
            Auth::Inference { api_key, api_secret, provider, inference_class, identity } => {
                // `get_inference_headers()` as the host bound it; unbound, the job-less half.
                let mut h = identity.for_connection().unwrap_or_else(|| {
                    let mut h = crate::headers::Headers::new();
                    h.set("User-Agent", crate::headers::user_agent(env!("CARGO_PKG_VERSION")));
                    if let Some(class) = inference_class {
                        h.set(crate::HEADER_INFERENCE_PRIORITY, class.clone());
                    }
                    h
                });
                let token = lk_agents_net::token::inference_bearer(api_key, api_secret)?;
                h.set("Authorization", format!("Bearer {token}"));
                if let Some(p) = provider {
                    h.set(crate::HEADER_INFERENCE_PROVIDER, p.clone());
                }
                h.into_vec()
            }
        })
    }
}

/// `RealtimeModel`; its sessions are already running when `session()` returns, as in the reference.
/// `E` is the process a host-constructed model runs in (decision 88).
pub struct NativeRealtime {
    dialect: Dialect,
    /// The model's options: what each new session starts from, and what `update_options` edits.
    opts: std::sync::Mutex<SessionOptions>,
    connect: Arc<Connect>,
    transport: Arc<dyn WsTransport>,
    spawner: Arc<dyn Spawner>,
    clock: Arc<dyn Clock>,
    images: Arc<dyn ImageResolver + Send + Sync>,
    workers: WorkerPool,
    ids: Arc<dyn IdGen>,
    /// Whether the caller passed a turn-detection configuration; `can_disable_turn_detection`
    /// reports intent (see [`Dialect::capabilities`]).
    turn_detection_given: bool,
    /// The host class's `label`, stamped on errors and metrics; `None` names the dialect.
    label: Option<String>,
    /// `self._sessions`: the live sessions an `update_options` reaches.
    sessions: std::sync::Mutex<Vec<Weak<super::queue::Ingress>>>,
}

impl NativeRealtime {
    pub fn new(
        dialect: Dialect,
        api_key: impl Into<String>,
        transport: Arc<dyn WsTransport>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
        images: Arc<dyn ImageResolver + Send + Sync>,
        workers: WorkerPool,
    ) -> Self {
        let opts = SessionOptions::from_dialect(&dialect, false);
        let connect = Connect::openai(api_key, dialect.default_base_url);
        NativeRealtime {
            dialect,
            opts: std::sync::Mutex::new(opts),
            connect: Arc::new(connect),
            transport,
            spawner,
            clock,
            images,
            workers,
            ids: Arc::new(RandomIds),
            turn_detection_given: false,
            label: None,
            sessions: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// `turn_detection_given` is `is_given(turn_detection)` before a default was applied.
    pub fn with_options(mut self, opts: SessionOptions, turn_detection_given: bool) -> Self {
        self.opts = std::sync::Mutex::new(opts);
        self.turn_detection_given = turn_detection_given;
        self
    }

    pub fn with_connect(mut self, connect: Connect) -> Self {
        self.connect = Arc::new(connect);
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_ids(mut self, ids: Arc<dyn IdGen>) -> Self {
        self.ids = ids;
        self
    }

    fn options(&self) -> std::sync::MutexGuard<'_, SessionOptions> {
        self.opts.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `get_inference_headers()` of the job the model now serves; every later connection carries
    /// them, with a fresh session id. A no-op off the gateway.
    pub fn bind_identification(&self, headers: crate::headers::Headers) {
        if let Auth::Inference { identity, .. } = &self.connect.auth {
            identity.bind(headers);
        }
    }

    /// `RealtimeModel.update_options`: the model keeps the values for its next sessions, and every
    /// live session diffs them against its own. `transcription` and `noise_reduction` are always
    /// handed on, as the reference does.
    pub fn update_options(&self, mut patch: OptionsPatch) {
        warn_on_half_disabled_turn_taking(&patch.turn_detection);
        let (transcription, noise_reduction) = {
            let mut opts = self.options();
            opts.apply(&patch);
            (opts.transcription.clone(), opts.noise_reduction.clone())
        };
        patch.transcription = lk_agents_abi::Maybe::Given(transcription);
        patch.noise_reduction = lk_agents_abi::Maybe::Given(noise_reduction);
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        sessions.retain(|s| match s.upgrade() {
            Some(tx) => {
                tx.send(Cmd::Options(Box::new(patch.clone())));
                true
            }
            None => false,
        });
    }
}

/// `_warn_on_half_disabled_turn_taking`: the client takes the turns but the server still cuts its
/// own response on speech.
pub fn warn_on_half_disabled_turn_taking(td: &lk_agents_abi::Maybe<Option<TurnDetection>>) {
    if let lk_agents_abi::Maybe::Given(Some(td)) = td {
        let v = td.to_json();
        if v.get("create_response") == Some(&serde_json::Value::Bool(false))
            && v.get("interrupt_response") != Some(&serde_json::Value::Bool(false))
        {
            log::warn!(
                "create_response=False hands turn taking to the client, but the server still \
                 cancels its response on user speech, pass interrupt_response=False as well"
            );
        }
    }
}

/// What a session's pump is told: the core's commands, the model's option changes, and the close.
pub(crate) enum Cmd {
    Core(Command),
    Options(Box<OptionsPatch>),
    Close,
}

impl NativeRealtime {
    /// `provider`: `urlparse(base_url).netloc`.
    pub fn provider(&self) -> String {
        let url = &self.connect.base_url;
        let s = url.split_once("://").map_or(url.as_str(), |(_, r)| r);
        s.split(['/', '?']).next().unwrap_or(s).to_owned()
    }

    pub fn model(&self) -> String {
        self.options().model.clone()
    }

    /// The `model_provider` the metrics name: the host's `provider` (LiveKit Inference reports
    /// `"livekit"`, as its host class does).
    fn metrics_provider(&self) -> String {
        if self.dialect.provider_label == crate::realtime::dialect::INFERENCE.provider_label {
            "livekit".to_owned()
        } else {
            self.provider()
        }
    }

    /// `RealtimeModel.label`.
    pub fn label(&self) -> String {
        self.label
            .clone()
            .unwrap_or_else(|| format!("lk_agents_realtime.{}", self.dialect.provider_label))
    }

    /// `RealtimeCapabilities`, as a dict.
    pub fn capabilities(&self) -> RealtimeCapabilities {
        let opts = self.options();
        self.dialect.capabilities(
            opts.turn_detection.as_ref(),
            self.turn_detection_given,
            opts.audio_output,
            opts.transcription.is_some(),
        )
    }

    /// `session(turn_detection_disabled=...)`. Applied to the session's copy of the options; the
    /// model stays reusable.
    pub async fn session(&self, turn_detection_disabled: bool) -> SessionHandle {
        let mut opts = self.options().clone();
        if turn_detection_disabled && self.dialect.supports_turn_detection_disabled {
            opts.turn_detection = None;
        }
        let caps = self.dialect.capabilities(
            opts.turn_detection.as_ref(),
            self.turn_detection_given,
            opts.audio_output,
            opts.transcription.is_some(),
        );
        let azure = self.connect.azure.as_ref();
        let url = wire::process_base_url(&self.connect.base_url, &opts.model, azure);
        let label = self.label();
        let machine = RealtimeMachine::new(self.dialect.clone(), opts, caps, self.ids.short(""))
            .with_response_timeout(self.connect.conn.read_timeout.map(Duration::from_secs_f64))
            .with_label(label.clone())
            .with_provider(self.metrics_provider())
            .with_wall_anchor(
                self.clock.wall() - Duration::from_nanos(self.clock.now().as_nanos()),
            );

        let (cmd_tx, cmd_rx) = mpsc::channel(super::queue::COMMAND_BACKLOG);
        let (ev_tx, ev_rx) = super::queue::events();
        let shared = Arc::new(Shared {
            machine: Mutex::new(machine),
            caps: std::sync::Mutex::new(caps),
            closing: tokio::sync::Notify::new(),
            cleanup: std::sync::Mutex::new(None),
            retirement: std::sync::Mutex::new(None),
            retirement_wake: Arc::new(tokio::sync::Notify::new()),
            context_ready: tokio::sync::Notify::new(),
            retirement_closing: std::sync::atomic::AtomicBool::new(false),
            workers: self.workers,
        });
        let commands = Arc::new(super::queue::Ingress {
            commands: cmd_tx,
            shared: shared.clone(),
            events: ev_tx.clone(),
            clock: self.clock.clone(),
            label: label.clone(),
        });
        self.sessions.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&commands));

        let pump = Pump {
            shared: shared.clone(),
            url,
            connect: self.connect.clone(),
            shape: self.connect.shape(),
            transport: self.transport.clone(),
            clock: self.clock.clone(),
            provider_label: self.dialect.provider_label,
            label,
            events: ev_tx,
            images: self.images.clone(),
            spawner: self.spawner.clone(),
        };
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        let abort = self.spawner.spawn_unboxed(async move {
            pump.run(cmd_rx).await;
            drop(done_tx);
        });

        SessionHandle {
            shared,
            commands,
            events: Mutex::new(ev_rx),
            abort,
            done: std::sync::Mutex::new(Some(done_rx)),
            clock: self.clock.clone(),
            ids: self.ids.clone(),
        }
    }
}

pub(super) struct Shared {
    /// `aclose` was called: a pump waiting to reconnect stops waiting.
    pub(super) closing: tokio::sync::Notify,
    cleanup: std::sync::Mutex<Option<crate::realtime::ConnectionEffect>>,
    machine: Mutex<RealtimeMachine>,
    retirement: std::sync::Mutex<Option<tokio::sync::watch::Receiver<bool>>>,
    retirement_wake: Arc<tokio::sync::Notify>,
    context_ready: tokio::sync::Notify,
    retirement_closing: std::sync::atomic::AtomicBool,
    /// The machine's, mirrored where a synchronous read reaches it; `update_options` moves it.
    caps: std::sync::Mutex<RealtimeCapabilities>,
    /// Decoding and context retirement run on the process's admitted workers.
    workers: WorkerPool,
}

impl Shared {
    fn record_incomplete_cleanup(&self, generation: u64) -> bool {
        let mut cleanup = self.cleanup.lock().unwrap_or_else(|error| error.into_inner());
        if cleanup.is_some() {
            return false;
        }
        *cleanup = Some(crate::realtime::ConnectionEffect::CleanupIncomplete { generation });
        true
    }

    fn incomplete_cleanup(&self, generation: u64, label: &str) {
        if !self.record_incomplete_cleanup(generation) {
            return;
        }
        log::error!(target: "livekit.agents", connection_generation = generation, model = label; "realtime cleanup did not settle within its deadline");
    }

    fn retirement_unavailable(&self, generation: u64, label: &str, error: &str) {
        if self.record_incomplete_cleanup(generation) {
            log::error!(target: "livekit.agents", connection_generation = generation, model = label, cause = error; "realtime context cleanup service is unavailable");
        }
    }
}

/// A live session. Cheap to clone the way the swapper needs (`Arc<SessionHandle>`), and every
/// mutating method takes `&self`.
pub struct SessionHandle {
    shared: Arc<Shared>,
    commands: Arc<super::queue::Ingress>,
    /// One consumer is the contract.
    events: Mutex<super::queue::EventReceiver>,
    abort: lk_agents_abi::AbortHandle,
    /// Resolves (closed) when the pump has ended.
    done: std::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    clock: Arc<dyn Clock>,
    ids: Arc<dyn IdGen>,
}

impl SessionHandle {
    pub fn capabilities(&self) -> RealtimeCapabilities {
        *self.shared.caps.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Framework cleanup diagnostics stay separate from provider errors and conversation events.
    pub fn cleanup_observation(&self) -> Option<crate::realtime::ConnectionEffect> {
        self.shared.cleanup.lock().unwrap_or_else(|error| error.into_inner()).clone()
    }

    /// The *mirror*: what the server believes.
    pub async fn chat_ctx(&self) -> ChatContext {
        self.shared.machine.lock().await.chat_ctx()
    }

    /// Never blocks and never fails loudly (`send_event` suppresses `ChanClosed`).
    pub fn send(&self, cmd: Command) {
        self.commands.send(Cmd::Core(cmd));
    }

    /// `RealtimeSession.update_options`.
    pub fn update_options(&self, patch: OptionsPatch) {
        self.commands.send(Cmd::Options(Box::new(patch)));
    }

    pub fn push_audio(&self, frame: lk_agents_abi::AudioFrame) {
        self.send(Command::PushAudio(lk_agents_abi::llm::realtime::PushAudio { frame }));
    }

    /// Returns the `req_id` to correlate [`Event::GenerationCreated`] on.
    pub fn generate_reply(&self, instructions: Option<String>) -> String {
        let req_id = self.ids.short("response_create_");
        self.send(Command::GenerateReply(lk_agents_abi::llm::realtime::GenerateReply {
            req_id: req_id.clone(),
            instructions,
            tool_choice: None,
        }));
        req_id
    }

    pub async fn next_event(&self) -> Option<Event> {
        self.events.lock().await.next().await
    }

    /// `aclose`: the response in flight is cancelled and the socket closed with a close frame,
    /// the machine's `close()` failing what still waits; a pump that cannot take the close (it is
    /// connecting) is aborted at the bound.
    pub async fn aclose(&self) {
        self.send(Command::Interrupt(lk_agents_abi::llm::realtime::Interrupt {
            response_id: None,
        }));
        self.commands.send(Cmd::Close);
        self.shared.closing.notify_one();
        let done = self.done.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(done) = done
            && lk_agents_abi::timeout(&*self.clock, lk_agents_abi::HOST_TEARDOWN_TIMEOUT, done)
                .await
                .is_err()
        {
            let generation = self.shared.machine.lock().await.connection_generation();
            self.shared.incomplete_cleanup(generation, &self.commands.label);
        }
        self.abort.abort();
        self.commands.events.finish();
    }
}

struct Pump {
    shared: Arc<Shared>,
    url: String,
    connect: Arc<Connect>,
    shape: WireShape,
    transport: Arc<dyn WsTransport>,
    clock: Arc<dyn Clock>,
    provider_label: &'static str,
    label: String,
    events: super::queue::Delivery,
    images: Arc<dyn ImageResolver + Send + Sync>,
    spawner: Arc<dyn Spawner>,
}

impl Pump {
    /// `_main_task`: connect, run, and reconnect, retrying a failed connection on the connect
    /// options' budget, until a terminal error or a close.
    async fn run(self, commands: mpsc::Receiver<Cmd>) {
        self.run_loop(commands).await;
        self.close().await;
        self.wait_for_retirement().await;
        self.events.finish();
    }

    async fn run_loop(&self, mut commands: mpsc::Receiver<Cmd>) {
        use crate::realtime::connection::{ConnectionEffect, RealtimeConnection};
        let mut connection = RealtimeConnection::new(&self.connect.conn, self.clock.now());
        let mut effects = connection.start(self.clock.now());
        let mut reconnecting = false;
        loop {
            let mut follow_up = Vec::new();
            for effect in effects {
                match effect {
                    ConnectionEffect::Connect { generation } => {
                        let outcome = self
                            .connect_and_run(
                                &mut commands,
                                reconnecting,
                                &mut connection,
                                generation,
                            )
                            .await;
                        reconnecting = true;
                        match outcome {
                            Ok(Outcome::Closed) => {
                                let _ = connection.close(self.clock.now());
                                let _ = connection.released(generation, self.clock.now());
                                return;
                            }
                            Ok(Outcome::Fatal(error)) => {
                                let _ = self.events.send(Event::Error(error));
                                let _ = connection.close(self.clock.now());
                                let _ = connection.released(generation, self.clock.now());
                                self.close().await;
                                return;
                            }
                            Ok(Outcome::ConnectionEffects(effects)) => follow_up.extend(effects),
                            Ok(Outcome::Recycle) => follow_up.extend(connection.failed(
                                generation,
                                self.clock.now(),
                                self.lost(),
                            )),
                            Err(error) => follow_up.extend(connection.failed(
                                generation,
                                self.clock.now(),
                                error,
                            )),
                        }
                        follow_up.extend(connection.released(generation, self.clock.now()));
                    }
                    ConnectionEffect::Retry { error } => self.emit_error(error, true),
                    ConnectionEffect::Failed { error } => {
                        self.emit_error(error, false);
                        self.close().await;
                        return;
                    }
                    ConnectionEffect::Close { .. } => {}
                    effect @ ConnectionEffect::CleanupIncomplete { .. } => {
                        self.observe_cleanup(effect)
                    }
                }
            }
            if !follow_up.is_empty() {
                effects = follow_up;
                continue;
            }
            let Some(deadline) = connection.next_deadline() else {
                return;
            };
            tokio::select! {
                biased;
                () = self.shared.closing.notified() => {
                    let _ = connection.close(self.clock.now());
                    self.close().await;
                    return;
                }
                () = self.clock.wait_until(deadline) => effects = connection.tick(self.clock.now()),
            }
        }
    }

    /// One connection: open, replay on a reconnect, run until it ends. `Err` is the attempt's
    /// failure, retryable or not.
    async fn connect_and_run(
        &self,
        commands: &mut mpsc::Receiver<Cmd>,
        reconnecting: bool,
        connection: &mut crate::realtime::connection::RealtimeConnection,
        generation: u64,
    ) -> Result<Outcome, ApiError> {
        let t0 = self.clock.now();
        let headers = self.connect.headers()?;
        let connect = self.transport.connect(&self.url, headers);
        tokio::pin!(connect);
        let (sender, receiver) = loop {
            let deadline = connection.next_deadline();
            tokio::select! {
                biased;
                () = self.shared.closing.notified() => { self.close().await; return Ok(Outcome::Closed); }
                () = async { match deadline { Some(at) => self.clock.wait_until(at).await, None => std::future::pending().await } } => {
                    let effects = connection.tick(self.clock.now());
                    if effects.iter().any(|effect| !matches!(effect, crate::realtime::connection::ConnectionEffect::CleanupIncomplete { .. })) {
                        return Ok(Outcome::ConnectionEffects(effects));
                    }
                    for effect in effects { self.observe_cleanup(effect); }
                }
                result = &mut connect => match result {
                    Ok(pair) => break pair,
                    Err(error) => return Err(lk_agents_net::transport::connect_error(self.provider_label, error)),
                }
            }
        };
        let acquire_time = self.clock.now().saturating_duration_since(t0);
        let metrics =
            self.shared.machine.lock().await.connection_metrics(self.clock.now(), acquire_time);
        let _ = self.events.send(metrics);

        let step = {
            let mut m = self.shared.machine.lock().await;
            let now = self.clock.now();
            if reconnecting { m.reconnect(now) } else { m.open(now) }
        };
        if !connection.connected(generation, self.clock.now()) {
            self.close_sender(sender, generation).await;
            return Ok(Outcome::Closed);
        }

        let outcome = self.run_ws(commands, sender.clone(), receiver, step, connection).await;
        if self.close_sender(sender, generation).await { outcome } else { Ok(Outcome::Closed) }
    }

    async fn run_ws(
        &self,
        commands: &mut mpsc::Receiver<Cmd>,
        sender: Arc<dyn lk_agents_net::transport::WsSender>,
        mut receiver: Box<dyn lk_agents_net::transport::WsReceiver>,
        initial: Step,
        connection: &mut crate::realtime::connection::RealtimeConnection,
    ) -> Result<Outcome, ApiError> {
        let generation = self.shared.machine.lock().await.connection_generation();
        let (sent, mut receipts) = mpsc::channel(super::writer::RESPONSE_RECEIPTS);
        let (outgoing, writes) = mpsc::channel(super::writer::WRITE_BACKLOG);
        let writer = super::writer::run(
            writes,
            sender,
            self.shape.clone(),
            self.connect.conn,
            self.clock.clone(),
            self.images.clone(),
            self.shared.workers,
            generation,
            sent,
        );
        tokio::pin!(writer);
        self.dispatch(initial, &outgoing)?;
        let mut decoding: Option<
            std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<wire::Incoming, ApiError>> + Send>,
            >,
        > = None;
        let mut context_due = false;
        loop {
            {
                let mut machine = self.shared.machine.lock().await;
                while let Ok(sent) = receipts.try_recv() {
                    machine.response_sent(sent.generation, sent.at, &sent.req_id, sent.is_say);
                }
            }

            // The machine owns every deadline; the driver only asks how long to sleep.
            let deadline = self
                .shared
                .machine
                .lock()
                .await
                .next_deadline()
                .into_iter()
                .chain(connection.next_deadline())
                .min();
            let sleep = deadline.map(|at| self.clock.wait_until(at));
            let context_wake = self.shared.context_ready.notified();
            tokio::pin!(context_wake);
            context_wake.as_mut().enable();
            let continuation_ready = self.shared.machine.lock().await.context_continuation_ready();
            let can_continue = continuation_ready && outgoing.capacity() > 0;
            if can_continue {
                tokio::task::yield_now().await;
            }

            let mut step = tokio::select! {
                biased;
                () = async { match sleep { Some(s) => s.await, None => std::future::pending().await } } => {
                    let effects = connection.tick(self.clock.now());
                    if effects.iter().any(|effect| !matches!(effect, crate::realtime::connection::ConnectionEffect::CleanupIncomplete { .. })) {
                        return Ok(Outcome::ConnectionEffects(effects));
                    }
                    for effect in effects { self.observe_cleanup(effect); }
                    let mut m = self.shared.machine.lock().await;
                    m.tick(self.clock.now())
                },
                result = &mut writer => return result.map(|()| Outcome::Closed),
                () = self.shared.closing.notified() => { self.close().await; return Ok(Outcome::Closed); },
                () = &mut context_wake => continue,
                Some(sent) = receipts.recv() => {
                    self.shared.machine.lock().await.response_sent(sent.generation, sent.at, &sent.req_id, sent.is_say);
                    continue;
                },
                decoded = async { match decoding.as_mut() {
                    Some(work) => work.await,
                    None => std::future::pending().await,
                } } => {
                    decoding = None;
                    let incoming = decoded?;
                    let mut machine = self.shared.machine.lock().await;
                    while let Ok(sent) = receipts.try_recv() { machine.response_sent(sent.generation, sent.at, &sent.req_id, sent.is_say); }
                    machine.incoming_on(generation, self.clock.now(), incoming)
                },
                msg = receiver.recv(), if decoding.is_none() => match msg {
                    Ok(WsMessage::Text(text)) => {
                        if text.len() > 16 * 1024 * 1024 {
                            return Err(ApiError::new("realtime message exceeds 16 MiB").with_retryable(false));
                        }
                        let shape = self.shape.clone();
                        let created_at = self.clock.wall();
                        let workers = self.shared.workers;
                        decoding = Some(Box::pin(async move {
                            let workers = workers().map(lk_agents_media::workers::MediaWorkers::from).map_err(|error| ApiError::new(error).with_retryable(false))?;
                            let worker = workers.try_spawn(lk_agents_media::workers::MediaWorkerClass::Decoder, move || {
                                wire::decode_text(&text, &shape, created_at)
                            }).map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
                            worker.await.map_err(|error| ApiError::new(format!("realtime decoder worker failed: {error}")).with_retryable(false))?
                                .map_err(|error| { let mut api = ApiError::new(error.to_string()).with_retryable(false); api.request_id = error.event_id; api })
                        }));
                        continue;
                    }
                    Ok(WsMessage::Closed) => return Err(self.lost()),
                    Ok(WsMessage::Binary(_) | WsMessage::Other) => continue,
                    Err(TransportError::Timeout) => continue,
                    Err(_) => return Err(self.lost()),
                },
                cmd = commands.recv(), if !context_due || !can_continue => {
                    context_due = true;
                    match cmd {
                    Some(Cmd::Core(cmd)) => {
                        let mut m = self.shared.machine.lock().await;
                        m.command(self.clock.now(), cmd)
                    }
                    Some(Cmd::Options(patch)) => {
                        let mut m = self.shared.machine.lock().await;
                        let step = m.update_options(*patch);
                        *self.shared.caps.lock().unwrap_or_else(|e| e.into_inner()) = m.capabilities();
                        step
                    }
                    Some(Cmd::Close) | None => {
                        self.close().await;
                        return Ok(Outcome::Closed);
                    }
                    }
                },
                capacity = outgoing.reserve(), if continuation_ready && !can_continue => {
                    drop(capacity.map_err(|_| ApiError::new("realtime writer closed during context replay").with_retryable(false))?);
                    continue;
                },
                () = std::future::ready(()), if can_continue => {
                    context_due = false;
                    self.shared.machine.lock().await.continue_context(self.clock.now())
                }

            };

            if let Some(error) = step.fatal.take() {
                let _ = self.dispatch(step, &outgoing);
                return Ok(Outcome::Fatal(error));
            }
            let recycle = step.recycle;
            self.dispatch(step, &outgoing)?;
            if recycle {
                log::debug!("recycling the {} connection", self.provider_label);
                return Ok(Outcome::Recycle);
            }
        }
    }

    /// "connection closed unexpectedly": retryable, so the session reconnects.
    fn lost(&self) -> ApiError {
        ApiError::connection(Some(format!(
            "{} connection closed unexpectedly",
            self.provider_label
        )))
    }

    /// The session loop is over: close any in-progress generation and fail pending replies.
    async fn close(&self) {
        let step = {
            let mut m = self.shared.machine.lock().await;
            let now = self.clock.now();
            m.close(now)
        };
        for ev in step.events {
            if let Err(error) = self.events.send(ev) {
                self.events.fail(self.clock.now(), &self.label, error);
            }
        }
        self.shared.retirement_closing.store(true, std::sync::atomic::Ordering::Release);
        self.wake_retirement();
    }

    async fn wait_for_retirement(&self) {
        let done = self.shared.retirement.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some(mut done) = done {
            let _ = done.wait_for(|settled| *settled).await;
        }
    }

    fn dispatch(
        &self,
        step: Step,
        outgoing: &mpsc::Sender<Vec<crate::realtime::machine::Outgoing>>,
    ) -> Result<(), ApiError> {
        self.wake_retirement();
        if !step.frames.is_empty() {
            outgoing.try_send(step.frames).map_err(|_| {
                ApiError::new("realtime write backlog exceeded").with_retryable(false)
            })?;
        }
        for event in step.events {
            self.events.send(event)?;
        }
        Ok(())
    }

    async fn close_sender(
        &self,
        sender: Arc<dyn lk_agents_net::transport::WsSender>,
        generation: u64,
    ) -> bool {
        let (settled, done) = tokio::sync::oneshot::channel();
        // The executor retains this one resource cleanup even if its caller's budget expires.
        // An incomplete cleanup stops this session rather than accumulating more sockets.
        let _owner = self.spawner.spawn_unboxed(async move {
            sender.close().await;
            let _ = settled.send(());
        });
        if matches!(
            lk_agents_abi::timeout(&*self.clock, lk_agents_abi::HOST_TEARDOWN_TIMEOUT, done).await,
            Ok(Ok(()))
        ) {
            true
        } else {
            self.shared.incomplete_cleanup(generation, &self.label);
            false
        }
    }

    fn observe_cleanup(&self, effect: crate::realtime::connection::ConnectionEffect) {
        if let crate::realtime::connection::ConnectionEffect::CleanupIncomplete { generation } =
            effect
        {
            self.shared.incomplete_cleanup(generation, &self.label);
        }
    }

    fn emit_error(&self, err: ApiError, recoverable: bool) {
        let event = Event::Error(lk_agents_abi::llm::realtime::RealtimeModelError {
            timestamp: self.clock.now(),
            label: self.label.clone(),
            message: err.message.clone(),
            recoverable,
            req_id: None,
            code: None,
            provider_error: Some(err.clone()),
        });
        if self.events.send(event).is_err() {
            self.events.fail(self.clock.now(), &self.label, err);
        }
    }
}

enum Outcome {
    ConnectionEffects(Vec<crate::realtime::connection::ConnectionEffect>),
    Recycle,
    Closed,
    /// The server said the session cannot go on: no reconnect.
    Fatal(lk_agents_abi::llm::realtime::RealtimeModelError),
}

// =============================================================================================
// The `lk-agents-abi` seam
// =============================================================================================

impl lk_agents_abi::Realtime for NativeRealtime {
    fn label(&self) -> String {
        NativeRealtime::label(self)
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        NativeRealtime::capabilities(self)
    }

    fn session(
        &self,
        turn_detection_disabled: bool,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Arc<dyn lk_agents_abi::RealtimeSession>, lk_agents_abi::errors::ProviderError>,
    > {
        Box::pin(async move {
            // Infallible: a connect failure arrives as `Event::Error`, like a reconnect failure.
            Ok(Arc::new(NativeRealtime::session(self, turn_detection_disabled).await)
                as Arc<dyn lk_agents_abi::RealtimeSession>)
        })
    }
}

impl lk_agents_abi::RealtimeSession for SessionHandle {
    fn capabilities(&self) -> RealtimeCapabilities {
        SessionHandle::capabilities(self)
    }

    fn send(&self, cmd: lk_agents_abi::RealtimeCommand) {
        SessionHandle::send(self, cmd);
    }

    fn next_event(&self) -> lk_agents_abi::BoxFuture<'_, Option<Event>> {
        Box::pin(async move { SessionHandle::next_event(self).await })
    }

    fn chat_ctx(&self) -> lk_agents_abi::BoxFuture<'_, ChatContext> {
        Box::pin(async move { SessionHandle::chat_ctx(self).await })
    }

    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move { SessionHandle::aclose(self).await })
    }
}

#[cfg(test)]
mod retirement_tests;

mod retirement;
