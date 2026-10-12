//! The driver: the only file in this crate that awaits anything; no decision is made here.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::{Spawner, SpawnerExt};
use lk_agents_abi::{Clock, Elapsed};
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_inference::transport::{
    TransportError, WsMessage, WsTransport, connect_error, to_ws_url,
};
use lk_agents_abi::ChatContext;
use tokio::sync::{Mutex, mpsc};

use crate::realtime::dialect::Dialect;
// The trait a native model is taken as (`as_dyn`), named in the binding's registration.
#[cfg(feature = "python")]
use lk_agents_abi::Realtime;
use lk_agents_abi::llm::realtime::{Event, RealtimeCapabilities};
use crate::realtime::machine::{Command, RealtimeMachine, SessionOptions, Step};
use crate::realtime::wire::{self, IdGen, RandomIds};

/// `DEFAULT_API_CONNECT_OPTIONS.timeout`.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// `RealtimeModel`; its sessions are already running when `session()` returns, as in the reference.
pub struct NativeRealtime {
    dialect: Dialect,
    opts: SessionOptions,
    api_key: String,
    base_url: String,
    transport: Arc<dyn WsTransport>,
    spawner: Arc<dyn Spawner>,
    clock: Arc<dyn Clock>,
    images: Arc<dyn ImageResolver + Send + Sync>,
    ids: Arc<dyn IdGen>,
    /// Whether the caller passed a turn-detection configuration; `can_disable_turn_detection`
    /// reports intent (see [`Dialect::capabilities`]).
    turn_detection_given: bool,
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
        let base_url = dialect.default_base_url.to_owned();
        NativeRealtime {
            dialect,
            opts,
            api_key: api_key.into(),
            base_url,
            transport,
            spawner,
            clock,
            images,
            ids: Arc::new(RandomIds),
            turn_detection_given: false,
        }
    }

    pub fn with_options(mut self, opts: SessionOptions) -> Self {
        self.opts = opts;
        self.turn_detection_given = true;
        self
    }

    pub fn with_ids(mut self, ids: Arc<dyn IdGen>) -> Self {
        self.ids = ids;
        self
    }
}

/// `_core.NativeRealtimeModel`: a realtime model whose connection, reconnect and conversation
/// mirror are in Rust. Not a `RealtimeModel` subclass (the ABC stays Python); `AgentSession`
/// recognises it by identity, and nothing per-turn crosses.
#[lk_agents_abi::host(object(name = "NativeRealtimeModel", cell = "arc", pid = false, as_dyn(Realtime = as_realtime)))]
impl NativeRealtime {
    /// The keyword constructor: the dialect's defaults, then the caller's overrides -- a dialect's
    /// default model and voice are a pair, and mixing them with OpenAI's is a 400 on the first
    /// turn. `dialect` is `"openai"` or `"xai"`.
    #[allow(clippy::too_many_arguments)]
    #[host(new(
        pid = true,
        kw_only = "base_url",
        defaults(speed = 1.0, audio_output = true, turn_detection = true)
    ))]
    pub fn for_host(
        dialect: Dialect,
        api_key: String,
        base_url: Option<String>,
        model: Option<String>,
        voice: Option<String>,
        instructions: Option<String>,
        speed: f64,
        audio_output: bool,
        turn_detection: bool,
        #[host(ambient)] spawner: Arc<dyn Spawner>,
    ) -> Self {
        let mut opts = SessionOptions::from_dialect(&dialect, !turn_detection);
        if let Some(model) = model {
            opts.model = model;
        }
        if let Some(voice) = voice {
            opts.voice = voice;
        }
        opts.instructions = instructions;
        opts.speed = speed;
        opts.audio_output = audio_output;
        let mut native = NativeRealtime::new(
            dialect,
            api_key,
            Arc::new(crate::native::TungsteniteTransport),
            spawner,
            // The wall clock, not a session's `Timebase`: a model outlives any one session and
            // `max_session_duration` is measured against real time.
            Arc::new(lk_agents_abi::RealClock::default()),
            // Empty: images reaching a realtime session are already serialised by the host, and
            // resolving again would encode the same frame twice.
            Arc::new(lk_agents_abi::llm::provider_format::PreResolved::from_pairs(Vec::<(String, lk_agents_abi::llm::provider_format::SerializedImage)>::new())),
        );
        if let Some(url) = base_url {
            native.base_url = url;
        }
        // Called unconditionally: `with_options` also records that the caller had an opinion on
        // turn detection, which `can_disable_turn_detection` reports.
        native.with_options(opts)
    }

    /// The model as the session's realtime provider (a native model is not proxied).
    pub fn as_realtime(self: &Arc<Self>) -> Arc<dyn lk_agents_abi::Realtime> {
        Arc::clone(self) as Arc<dyn lk_agents_abi::Realtime>
    }

    /// `provider`: `urlparse(base_url).netloc`. `"api.x.ai"` for the xAI default.
    #[host(getter)]
    pub fn provider(&self) -> &str {
        let s = self.base_url.split_once("://").map_or(self.base_url.as_str(), |(_, r)| r);
        s.split(['/', '?']).next().unwrap_or(s)
    }

    #[host(getter)]
    pub fn model(&self) -> &str {
        &self.opts.model
    }

    /// `RealtimeModel.label`.
    #[host(getter)]
    pub fn label(&self) -> String {
        format!("lk_agents_realtime.{}", self.dialect.provider_label)
    }

    /// `RealtimeCapabilities`, as a dict.
    #[host(method)]
    pub fn capabilities(&self) -> RealtimeCapabilities {
        self.dialect.capabilities(
            self.opts.turn_detection.as_ref(),
            self.turn_detection_given,
            self.opts.audio_output,
            self.opts.transcription_model.is_some(),
        )
    }

    /// `session(turn_detection_disabled=...)`. Applied to the session's copy of the options; the
    /// model stays reusable.
    pub async fn session(&self, turn_detection_disabled: bool) -> SessionHandle {
        let mut opts = self.opts.clone();
        if turn_detection_disabled && self.dialect.supports_turn_detection_disabled {
            opts.turn_detection = None;
        }
        let caps = self.dialect.capabilities(
            opts.turn_detection.as_ref(),
            self.turn_detection_given,
            opts.audio_output,
            opts.transcription_model.is_some(),
        );
        let url = wire::process_base_url(&to_ws_url(&self.base_url), &opts.model);
        let machine = RealtimeMachine::new(
            self.dialect.clone(),
            opts,
            caps,
            self.ids.clone(),
            self.images.clone(),
        );

        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (ev_tx, ev_rx) = mpsc::unbounded_channel();
        let shared = Arc::new(Shared { machine: Mutex::new(machine), caps });

        let pump = Pump {
            shared: shared.clone(),
            url,
            headers: vec![
                ("User-Agent".to_owned(), "LiveKit Agents".to_owned()),
                ("Authorization".to_owned(), format!("Bearer {}", self.api_key)),
            ],
            transport: self.transport.clone(),
            clock: self.clock.clone(),
            provider_label: self.dialect.provider_label,
            events: ev_tx,
        };
        let abort = self.spawner.spawn_unboxed(pump.run(cmd_rx));

        SessionHandle {
            shared,
            commands: cmd_tx,
            events: Mutex::new(ev_rx),
            abort,
            ids: self.ids.clone(),
        }
    }
}

struct Shared {
    machine: Mutex<RealtimeMachine>,
    caps: RealtimeCapabilities,
}

/// A live session. Cheap to clone the way the swapper needs (`Arc<SessionHandle>`), and every
/// mutating method takes `&self`.
pub struct SessionHandle {
    shared: Arc<Shared>,
    commands: mpsc::UnboundedSender<Command>,
    /// One consumer is the contract.
    events: Mutex<mpsc::UnboundedReceiver<Event>>,
    abort: lk_agents_abi::AbortHandle,
    ids: Arc<dyn IdGen>,
}

impl SessionHandle {
    pub fn capabilities(&self) -> RealtimeCapabilities {
        self.shared.caps
    }

    /// The *mirror*: what the server believes.
    pub async fn chat_ctx(&self) -> ChatContext {
        self.shared.machine.lock().await.chat_ctx()
    }

    /// Never blocks and never fails loudly (`send_event` suppresses `ChanClosed`).
    pub fn send(&self, cmd: Command) {
        let _ = self.commands.send(cmd);
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

    pub async fn aclose(&self) {
        // Dropping the command sender ends the pump through the machine's `close()`.
        let _ = self.commands.send(Command::Interrupt(lk_agents_abi::llm::realtime::Interrupt { response_id: None }));
        self.abort.abort();
    }
}

struct Pump {
    shared: Arc<Shared>,
    url: String,
    headers: Vec<(String, String)>,
    transport: Arc<dyn WsTransport>,
    clock: Arc<dyn Clock>,
    provider_label: &'static str,
    events: mpsc::UnboundedSender<Event>,
}

impl Pump {
    /// `_main_task`: connect, run, and reconnect until a fatal error or a close.
    async fn run(self, mut commands: mpsc::UnboundedReceiver<Command>) {
        let mut reconnecting = false;
        loop {
            let t0 = self.clock.now();
            // The handshake, bounded on the session clock.
            let connect = self.transport.connect(&self.url, self.headers.clone());
            let conn = match lk_agents_abi::timeout(&*self.clock, CONNECT_TIMEOUT, connect).await {
                Ok(conn) => conn,
                Err(Elapsed) => Err(TransportError::Timeout),
            };
            let (sender, receiver) = match conn {
                Ok(pair) => pair,
                Err(e) => {
                    self.emit_error(connect_error(self.provider_label, e), false).await;
                    return;
                }
            };
            let _ = self.events.send(Event::ConnectionAcquired {
                acquire_time: self.clock.now().saturating_duration_since(t0),
            });

            let step = {
                let mut m = self.shared.machine.lock().await;
                let now = self.clock.now();
                if reconnecting { m.reconnect(now) } else { m.open(now) }
            };
            if !self.dispatch(step, sender.as_ref()).await {
                return;
            }

            match self.run_ws(&mut commands, sender.clone(), receiver).await {
                Outcome::Recycle => {
                    reconnecting = true;
                    sender.close().await;
                }
                Outcome::Closed => {
                    sender.close().await;
                    let step = {
                        let mut m = self.shared.machine.lock().await;
                        let now = self.clock.now();
                        m.close(now)
                    };
                    for ev in step.events {
                        let _ = self.events.send(ev);
                    }
                    return;
                }
                Outcome::Fatal(err) => {
                    sender.close().await;
                    self.emit_error(err, false).await;
                    return;
                }
            }
        }
    }

    async fn run_ws(
        &self,
        commands: &mut mpsc::UnboundedReceiver<Command>,
        sender: Arc<dyn lk_agents_inference::transport::WsSender>,
        mut receiver: Box<dyn lk_agents_inference::transport::WsReceiver>,
    ) -> Outcome {
        loop {
            // The machine owns every deadline; the driver only asks how long to sleep.
            let deadline = self.shared.machine.lock().await.next_deadline();
            let sleep = deadline.map(|at| self.clock.wait_until(at));

            let step = tokio::select! {
                biased;
                msg = receiver.recv() => match msg {
                    Ok(WsMessage::Text(text)) => {
                        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
                            continue;
                        };
                        let incoming = wire::decode(&value);
                        let mut m = self.shared.machine.lock().await;
                        m.incoming(self.clock.now(), incoming)
                    }
                    // "this will trigger a reconnection"
                    Ok(WsMessage::Closed) => return Outcome::Recycle,
                    Ok(WsMessage::Other) => continue,
                    Err(TransportError::Timeout) => continue,
                    Err(e) => return Outcome::Fatal(connect_error(self.provider_label, e)),
                },
                cmd = commands.recv() => match cmd {
                    Some(cmd) => {
                        let mut m = self.shared.machine.lock().await;
                        m.command(self.clock.now(), cmd)
                    }
                    None => return Outcome::Closed,
                },
                () = async { match sleep { Some(s) => s.await, None => std::future::pending().await } } => {
                    let mut m = self.shared.machine.lock().await;
                    m.tick(self.clock.now())
                }
            };

            if let Some(err) = &step.fatal {
                let err = err.clone();
                self.dispatch(step, sender.as_ref()).await;
                return Outcome::Fatal(err);
            }
            let recycle = step.recycle;
            if !self.dispatch(step, sender.as_ref()).await {
                return Outcome::Recycle;
            }
            if recycle {
                return Outcome::Recycle;
            }
        }
    }

    /// `false` means the socket died mid-send.
    async fn dispatch(
        &self,
        step: Step,
        sender: &dyn lk_agents_inference::transport::WsSender,
    ) -> bool {
        for frame in step.frames {
            if sender.send_text(frame.to_string()).await.is_err() {
                return false;
            }
        }
        for ev in step.events {
            let _ = self.events.send(ev);
        }
        true
    }

    async fn emit_error(&self, err: ApiError, recoverable: bool) {
        let _ = self.events.send(Event::Error(lk_agents_abi::llm::realtime::RealtimeModelError {
            timestamp: self.clock.now(),
            label: format!("lk_agents_realtime.{}", self.provider_label),
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
    Fatal(ApiError),
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
