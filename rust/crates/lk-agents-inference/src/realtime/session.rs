//! The driver: the only file in this crate that awaits anything; no decision is made here.

use std::sync::{Arc, Weak};
use std::time::Duration;

use lk_agents_abi::errors::ApiError;
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::{ApiConnectOptions, ChatContext};
use lk_agents_abi::{Clock, Elapsed};
use lk_agents_abi::{Spawner, SpawnerExt};
use lk_agents_inference::transport::{TransportError, WsMessage, WsTransport};
use tokio::sync::{Mutex, mpsc};

use crate::headers::BoundIdentity;
use crate::realtime::dialect::{Dialect, TurnDetection};
use crate::realtime::wire::{self, Azure, IdGen, RandomIds, WireShape};
// The trait a native model is taken as (`as_dyn`), named in the binding's registration.
#[cfg(feature = "python")]
use lk_agents_abi::Realtime;
use lk_agents_abi::llm::realtime::{Event, RealtimeCapabilities};
use crate::realtime::machine::{Command, OptionsPatch, RealtimeMachine, SessionOptions, Step};

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
                let token = crate::token::create_access_token(
                    api_key,
                    api_secret,
                    crate::token::DEFAULT_TTL_SECS,
                    crate::token::now_unix(),
                )
                .map_err(|e| ApiError::new(e.to_string()).with_retryable(false))?;
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
pub struct NativeRealtime {
    dialect: Dialect,
    /// The model's options: what each new session starts from, and what `update_options` edits.
    opts: std::sync::Mutex<SessionOptions>,
    connect: Arc<Connect>,
    transport: Arc<dyn WsTransport>,
    spawner: Arc<dyn Spawner>,
    clock: Arc<dyn Clock>,
    images: Arc<dyn ImageResolver + Send + Sync>,
    ids: Arc<dyn IdGen>,
    /// Whether the caller passed a turn-detection configuration; `can_disable_turn_detection`
    /// reports intent (see [`Dialect::capabilities`]).
    turn_detection_given: bool,
    /// The host class's `label`, stamped on errors and metrics; `None` names the dialect.
    label: Option<String>,
    /// `self._sessions`: the live sessions an `update_options` reaches.
    sessions: std::sync::Mutex<Vec<Weak<mpsc::UnboundedSender<Cmd>>>>,
}

impl NativeRealtime {
    pub fn new(
        dialect: Dialect,
        api_key: impl Into<String>,
        transport: Arc<dyn WsTransport>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
        images: Arc<dyn ImageResolver + Send + Sync>,
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
        let (transcription, noise_reduction) = {
            let mut opts = self.options();
            opts.apply(&patch);
            warn_on_half_disabled_turn_taking(&patch.turn_detection);
            (opts.transcription.clone(), opts.noise_reduction.clone())
        };
        patch.transcription = lk_agents_abi::Maybe::Given(transcription);
        patch.noise_reduction = lk_agents_abi::Maybe::Given(noise_reduction);
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        sessions.retain(|s| match s.upgrade() {
            Some(tx) => tx.send(Cmd::Options(patch.clone())).is_ok(),
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
    Options(OptionsPatch),
    Close,
}

/// `_core.NativeRealtimeModel`: a realtime model whose connection, reconnect and conversation
/// mirror are in Rust. Not a `RealtimeModel` subclass (the ABC stays Python); `VoiceSession`
/// takes it as its realtime model, and nothing per-turn crosses. The keyword constructor,
/// `update_options` and `bind_identification` are `realtime::python`'s.
#[lk_agents_abi::host(object(
    name = "NativeRealtimeModel",
    cell = "arc",
    rust_name = PyNativeRealtimeModel,
    pid = false,
    as_dyn(Realtime = as_realtime),
    stub = "def __init__(self, **kwargs: Any) -> None: ...",
    stub = "def update_options(self, **kwargs: Any) -> None: ...",
    stub = "def bind_identification(self, headers: list[tuple[str, str]]) -> None: ..."
))]
impl NativeRealtime {
    /// The model as the session's realtime provider (a native model is not proxied).
    pub fn as_realtime(self: &Arc<Self>) -> Arc<dyn lk_agents_abi::Realtime> {
        Arc::clone(self) as Arc<dyn lk_agents_abi::Realtime>
    }

    /// `provider`: `urlparse(base_url).netloc`.
    #[host(getter)]
    pub fn provider(&self) -> String {
        let url = &self.connect.base_url;
        let s = url.split_once("://").map_or(url.as_str(), |(_, r)| r);
        s.split(['/', '?']).next().unwrap_or(s).to_owned()
    }

    #[host(getter)]
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
    #[host(getter)]
    pub fn label(&self) -> String {
        self.label.clone().unwrap_or_else(|| format!("lk_agents_realtime.{}", self.dialect.provider_label))
    }

    /// `RealtimeCapabilities`, as a dict.
    #[host(method)]
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
        let machine = RealtimeMachine::new(
            self.dialect.clone(),
            opts,
            caps,
            self.ids.clone(),
            self.images.clone(),
        )
        .with_label(label.clone())
        .with_provider(self.metrics_provider())
        .with_wall_anchor(self.clock.wall() - Duration::from_nanos(self.clock.now().as_nanos()));

        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (ev_tx, ev_rx) = mpsc::unbounded_channel();
        let shared = Arc::new(Shared {
            machine: Mutex::new(machine),
            caps: std::sync::Mutex::new(caps),
            closing: tokio::sync::Notify::new(),
        });
        let commands = Arc::new(cmd_tx);
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

struct Shared {
    /// `aclose` was called: a pump waiting to reconnect stops waiting.
    closing: tokio::sync::Notify,
    machine: Mutex<RealtimeMachine>,
    /// The machine's, mirrored where a synchronous read reaches it; `update_options` moves it.
    caps: std::sync::Mutex<RealtimeCapabilities>,
}

/// A live session. Cheap to clone the way the swapper needs (`Arc<SessionHandle>`), and every
/// mutating method takes `&self`.
pub struct SessionHandle {
    shared: Arc<Shared>,
    commands: Arc<mpsc::UnboundedSender<Cmd>>,
    /// One consumer is the contract.
    events: Mutex<mpsc::UnboundedReceiver<Event>>,
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

    /// The *mirror*: what the server believes.
    pub async fn chat_ctx(&self) -> ChatContext {
        self.shared.machine.lock().await.chat_ctx()
    }

    /// Never blocks and never fails loudly (`send_event` suppresses `ChanClosed`).
    pub fn send(&self, cmd: Command) {
        let _ = self.commands.send(Cmd::Core(cmd));
    }

    /// `RealtimeSession.update_options`.
    pub fn update_options(&self, patch: OptionsPatch) {
        let _ = self.commands.send(Cmd::Options(patch));
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
        self.events.lock().await.recv().await
    }

    /// `aclose`: the response in flight is cancelled and the socket closed with a close frame,
    /// the machine's `close()` failing what still waits; a pump that cannot take the close (it is
    /// connecting) is aborted at the bound.
    pub async fn aclose(&self) {
        self.send(Command::Interrupt(lk_agents_abi::llm::realtime::Interrupt { response_id: None }));
        let _ = self.commands.send(Cmd::Close);
        self.shared.closing.notify_one();
        let done = self.done.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(done) = done {
            let _ = lk_agents_abi::timeout(&*self.clock, lk_agents_abi::HOST_TEARDOWN_TIMEOUT, done).await;
        }
        self.abort.abort();
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
    events: mpsc::UnboundedSender<Event>,
}

impl Pump {
    /// `_main_task`: connect, run, and reconnect, retrying a failed connection on the connect
    /// options' budget, until a terminal error or a close.
    async fn run(self, mut commands: mpsc::UnboundedReceiver<Cmd>) {
        let max_retries = self.connect.conn.max_retry;
        let mut num_retries: u32 = 0;
        let mut reconnecting = false;
        loop {
            let err = match self.connect_and_run(&mut commands, reconnecting, &mut num_retries).await {
                Ok(Outcome::Closed) => return,
                Ok(Outcome::Fatal(error)) => {
                    let _ = self.events.send(Event::Error(error));
                    self.close().await;
                    return;
                }
                Ok(Outcome::Recycle) => {
                    reconnecting = true;
                    continue;
                }
                Err(err) => err,
            };
            if max_retries == 0 || !err.retryable || num_retries == max_retries {
                let err = if err.retryable && num_retries == max_retries && max_retries > 0 {
                    ApiError::connection(Some(format!(
                        "{} connection failed after {num_retries} attempts",
                        self.provider_label
                    )))
                } else {
                    err
                };
                self.emit_error(err, false);
                self.close().await;
                return;
            }
            self.emit_error(err, true);
            let interval = self.connect.conn.interval_for_retry(num_retries);
            log::warn!(
                "{} connection failed, retrying in {interval}s (attempt {num_retries}, max {max_retries})",
                self.provider_label
            );
            let at = self.clock.now().saturating_add(Duration::from_secs_f64(interval));
            tokio::select! {
                () = self.clock.wait_until(at) => {}
                () = self.shared.closing.notified() => {
                    self.close().await;
                    return;
                }
            }
            num_retries += 1;
            reconnecting = true;
        }
    }

    /// One connection: open, replay on a reconnect, run until it ends. `Err` is the attempt's
    /// failure, retryable or not.
    async fn connect_and_run(
        &self,
        commands: &mut mpsc::UnboundedReceiver<Cmd>,
        reconnecting: bool,
        num_retries: &mut u32,
    ) -> Result<Outcome, ApiError> {
        let t0 = self.clock.now();
        let headers = self.connect.headers()?;
        // The handshake, bounded on the session clock.
        let timeout = Duration::from_secs_f64(self.connect.conn.timeout.max(0.0));
        let connect = self.transport.connect(&self.url, headers);
        let (sender, receiver) = match lk_agents_abi::timeout(&*self.clock, timeout, connect).await {
            Ok(Ok(pair)) => pair,
            Ok(Err(TransportError::Status { status, message, .. })) => {
                return Err(ApiError::from_http(&message, status, None));
            }
            Ok(Err(_)) => {
                return Err(ApiError::connection(Some(format!(
                    "{} client connection error",
                    self.provider_label
                ))));
            }
            Err(Elapsed) => {
                return Err(ApiError::connection(Some(format!(
                    "{} connection timed out",
                    self.provider_label
                ))));
            }
        };
        let acquire_time = self.clock.now().saturating_duration_since(t0);
        let metrics = self.shared.machine.lock().await.connection_metrics(self.clock.now(), acquire_time);
        let _ = self.events.send(metrics);

        let step = {
            let mut m = self.shared.machine.lock().await;
            let now = self.clock.now();
            if reconnecting { m.reconnect(now) } else { m.open(now) }
        };
        if !self.dispatch(step, sender.as_ref()).await {
            sender.close().await;
            return Err(self.lost());
        }
        if reconnecting {
            *num_retries = 0;
            log::debug!("reconnected to {}", self.provider_label);
        }

        let outcome = self.run_ws(commands, sender.clone(), receiver).await;
        sender.close().await;
        outcome
    }

    async fn run_ws(
        &self,
        commands: &mut mpsc::UnboundedReceiver<Cmd>,
        sender: Arc<dyn lk_agents_inference::transport::WsSender>,
        mut receiver: Box<dyn lk_agents_inference::transport::WsReceiver>,
    ) -> Result<Outcome, ApiError> {
        loop {
            // The machine owns every deadline; the driver only asks how long to sleep.
            let deadline = self.shared.machine.lock().await.next_deadline();
            let sleep = deadline.map(|at| self.clock.wait_until(at));

            let mut step = tokio::select! {
                biased;
                msg = receiver.recv() => match msg {
                    Ok(WsMessage::Text(text)) => {
                        let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&text) else {
                            continue;
                        };
                        self.shape.inbound(&mut value);
                        let incoming = wire::decode(&value);
                        let mut m = self.shared.machine.lock().await;
                        m.incoming(self.clock.now(), incoming)
                    }
                    Ok(WsMessage::Closed) => return Err(self.lost()),
                    Ok(WsMessage::Other) => continue,
                    Err(TransportError::Timeout) => continue,
                    Err(_) => return Err(self.lost()),
                },
                cmd = commands.recv() => match cmd {
                    Some(Cmd::Core(cmd)) => {
                        let mut m = self.shared.machine.lock().await;
                        m.command(self.clock.now(), cmd)
                    }
                    Some(Cmd::Options(patch)) => {
                        let mut m = self.shared.machine.lock().await;
                        let step = m.update_options(patch);
                        *self.shared.caps.lock().unwrap_or_else(|e| e.into_inner()) = m.capabilities();
                        step
                    }
                    Some(Cmd::Close) | None => {
                        self.close().await;
                        return Ok(Outcome::Closed);
                    }
                },
                () = async { match sleep { Some(s) => s.await, None => std::future::pending().await } } => {
                    let mut m = self.shared.machine.lock().await;
                    m.tick(self.clock.now())
                }
            };

            if let Some(error) = step.fatal.take() {
                self.dispatch(step, sender.as_ref()).await;
                return Ok(Outcome::Fatal(error));
            }
            let recycle = step.recycle;
            if !self.dispatch(step, sender.as_ref()).await {
                return Err(self.lost());
            }
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
            let _ = self.events.send(ev);
        }
    }

    /// `false` means the socket died mid-send.
    async fn dispatch(
        &self,
        step: Step,
        sender: &dyn lk_agents_inference::transport::WsSender,
    ) -> bool {
        for mut frame in step.frames {
            self.shape.outbound(&mut frame);
            if sender.send_text(frame.to_string()).await.is_err() {
                return false;
            }
        }
        for ev in step.events {
            let _ = self.events.send(ev);
        }
        true
    }

    fn emit_error(&self, err: ApiError, recoverable: bool) {
        let _ = self.events.send(Event::Error(lk_agents_abi::llm::realtime::RealtimeModelError {
            timestamp: self.clock.now(),
            label: self.label.clone(),
            message: err.message,
            recoverable,
            req_id: None,
            code: None,
        }));
    }
}

enum Outcome {
    Recycle,
    Closed,
    /// The server said the session cannot go on: no reconnect.
    Fatal(lk_agents_abi::llm::realtime::RealtimeModelError),
}

// =============================================================================================
// The `lk-agents-abi` seam
// =============================================================================================

#[async_trait::async_trait]
impl lk_agents_abi::Realtime for NativeRealtime {
    fn label(&self) -> String {
        NativeRealtime::label(self)
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        NativeRealtime::capabilities(self)
    }

    async fn session(
        &self,
        turn_detection_disabled: bool,
    ) -> Result<Arc<dyn lk_agents_abi::RealtimeSession>, lk_agents_abi::errors::ProviderError> {
        // Infallible: a connect failure arrives as `Event::Error`, like a reconnect failure.
        Ok(Arc::new(NativeRealtime::session(self, turn_detection_disabled).await))
    }
}

#[async_trait::async_trait]
impl lk_agents_abi::RealtimeSession for SessionHandle {
    fn capabilities(&self) -> RealtimeCapabilities {
        SessionHandle::capabilities(self)
    }

    fn send(&self, cmd: lk_agents_abi::RealtimeCommand) {
        SessionHandle::send(self, cmd);
    }

    async fn next_event(&self) -> Option<Event> {
        SessionHandle::next_event(self).await
    }

    async fn chat_ctx(&self) -> ChatContext {
        SessionHandle::chat_ctx(self).await
    }

    async fn aclose(&self) {
        SessionHandle::aclose(self).await
    }
}
