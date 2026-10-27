//! The realtime model: the production native session (`lk_agents_inference::realtime`, the one a
//! LiveKit Inference realtime model runs on) over a socket whose far end is [`server::Server`],
//! the OpenAI Realtime server as the real one answers (`corpus/probes/`) and the corpus shows it.
//!
//! So what reaches the driver is what the production wire machine makes of real server traffic:
//! its reconnects, its stale-response bookkeeping, its transcripts and its errors are under test
//! with the driver, not stood in for. The server answers the client on its own clock, with the
//! delays the corpus measured; what it does of its own accord -- hears the person start and stop
//! (its VAD), drops the socket, errs -- is a `RealtimeEvent` the case sends.
//!
//! The socket is a resource (`Kind::Realtime`), one per session the driver opens, and every
//! response the client has been told of is one too (`Kind::Generation`), until its
//! `response.done`. Audio deltas are tagged with the generation and its chunk. What the server
//! refused, or answered from the wrong state, is recorded as a `Breach` on the socket.
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use lk_agents_abi::{
    ApiConnectOptions, ChatContext, Clock, Realtime, RealtimeCommand, RealtimeSession, Spawner,
    errors::ProviderError,
    llm::realtime::{Event, RealtimeCapabilities},
};
use lk_agents_inference::{
    realtime::{Connect, NativeRealtime, SeqIds, SessionOptions, TurnDetection, INFERENCE, session::Auth},
    testkit::NoImages,
    transport::{TransportError, WsMessage, WsReceiver, WsSender, WsTransport},
};

use super::{Control, Registry, Slot};
use crate::{
    input::{Input, RtEvent},
    record::{Detail, Kind, Tag},
};

pub mod server;

pub use server::{Exo, Server, Vad};

/// The model a session names in its metrics.
pub const MODEL: &str = "openai/gpt-realtime";

/// How a realtime setup's server is configured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServerSetup {
    /// Server-side turn detection, and which; `None` is turn detection explicitly off.
    pub vad: Option<Vad>,
    /// What the server's delays and choices are drawn from.
    pub seed: u64,
    /// How long a connection takes to open.
    pub connect: Duration,
    /// How often a response calls a tool, in percent; `None` is the corpus's rate.
    pub tool_calls: Option<u8>,
    /// The model answers in text (`modalities=["text"]`).
    pub text_only: bool,
}

pub struct FakeRealtime {
    registry: Arc<Registry>,
    setup: ServerSetup,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    /// Sessions opened, for a per-session seed.
    sessions: AtomicU64,
}

impl FakeRealtime {
    pub fn new(
        registry: Arc<Registry>,
        setup: ServerSetup,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
    ) -> Arc<Self> {
        Arc::new(FakeRealtime { registry, setup, clock, spawner, sessions: AtomicU64::new(0) })
    }

    /// The production model, over this socket.
    fn model(&self, transport: Arc<dyn WsTransport>) -> NativeRealtime {
        let mut opts = SessionOptions::from_dialect(&INFERENCE, false);
        opts.model = MODEL.into();
        opts.turn_detection = match self.setup.vad {
            Some(Vad::Semantic) => opts.turn_detection,
            Some(Vad::Server) => Some(TurnDetection::ServerVad {
                threshold: 0.5,
                prefix_padding_ms: 300,
                silence_duration_ms: 500,
                create_response: true,
                interrupt_response: true,
            }),
            None => None,
        };
        // Server detection left at its default is the driver's to switch off when it detects
        // turns itself; switched off by the developer, it stays off.
        opts.audio_output = !self.setup.text_only;
        let given = self.setup.vad.is_none();
        // The model's own clock, as a model built by the host runs on (`RealClock` from its
        // construction): never the session's.
        let clock = Arc::new(ModelClock { session: self.clock.clone(), offset: MODEL_BUILT_BEFORE });
        NativeRealtime::new(INFERENCE, "fake-key", transport, self.spawner.clone(), clock, Arc::new(NoImages))
        .with_options(opts, given)
        .with_connect(Connect {
            base_url: "https://realtime.fake/v1".into(),
            azure: None,
            auth: Auth::Bearer("fake-key".into()),
            conn: ApiConnectOptions::default(),
        })
        .with_ids(Arc::new(SeqIds::default()))
    }
}

#[async_trait::async_trait]
impl Realtime for FakeRealtime {
    fn label(&self) -> String {
        "fake::realtime".into()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        self.model(Arc::new(Refused)).capabilities()
    }

    async fn session(&self, disabled: bool) -> Result<Arc<dyn RealtimeSession>, ProviderError> {
        let nth = self.sessions.fetch_add(1, Ordering::SeqCst);
        let socket = Arc::new(Socket {
            tool_calls: self.setup.tool_calls,
            seed: self.setup.seed ^ (nth << 32),
            connect: self.setup.connect,
            clock: self.clock.clone(),
            registry: self.registry.clone(),
            slot: OnceLock::new(),
            state: Mutex::new(State::default()),
            wake: tokio::sync::Notify::new(),
            up: AtomicBool::new(true),
        });
        let slot = self.registry.register(
            Kind::Realtime,
            Detail::None,
            Control::Realtime(socket.clone()),
        );
        let _ = socket.slot.set(slot);
        let model = self.model(Arc::new(Transport(socket.clone())));
        let inner = lk_agents_abi::Realtime::session(&model, disabled).await?;
        Ok(Arc::new(Session { inner, socket }))
    }
}

/// How long before the session the model was built (at the entrypoint, or in a prewarm).
const MODEL_BUILT_BEFORE: Duration = Duration::from_secs(3_600);

/// The session's time, counted from the model's construction `offset` earlier: what a model's
/// own clock reads.
struct ModelClock {
    session: Arc<dyn Clock>,
    offset: Duration,
}

#[async_trait::async_trait]
impl Clock for ModelClock {
    fn now(&self) -> lk_agents_abi::Ts {
        self.session.now().saturating_add(self.offset)
    }

    async fn wait_until(&self, at: lk_agents_abi::Ts) {
        self.session.wait_until(at.saturating_sub(self.offset)).await
    }

    fn wall(&self) -> std::time::SystemTime {
        self.session.wall()
    }
}

/// A transport no session connects through: what the capabilities are read from.
struct Refused;

#[async_trait::async_trait]
impl WsTransport for Refused {
    async fn connect(
        &self,
        _: &str,
        _: Vec<(String, String)>,
    ) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), TransportError> {
        Err(TransportError::Closed)
    }
}

/// The session the driver holds: the production one, and the socket's end of life.
struct Session {
    inner: Arc<dyn RealtimeSession>,
    socket: Arc<Socket>,
}

#[async_trait::async_trait]
impl RealtimeSession for Session {
    fn capabilities(&self) -> RealtimeCapabilities {
        self.inner.capabilities()
    }

    fn send(&self, cmd: RealtimeCommand) {
        self.inner.send(cmd);
    }

    async fn next_event(&self) -> Option<Event> {
        self.inner.next_event().await
    }

    async fn chat_ctx(&self) -> ChatContext {
        self.inner.chat_ctx().await
    }

    /// The production close, then the session is over for the server: a connection the client
    /// left open (no close frame) keeps the socket unclosed, as a server still billing it would.
    async fn aclose(&self) {
        self.inner.aclose().await;
        self.socket.end_session();
    }
}

/// One session's socket: the server behind it and the connection now open, if any.
pub struct Socket {
    seed: u64,
    tool_calls: Option<u8>,
    connect: Duration,
    clock: Arc<dyn Clock>,
    registry: Arc<Registry>,
    slot: OnceLock<Slot>,
    state: Mutex<State>,
    /// Told whenever the server may have something new to say.
    wake: tokio::sync::Notify,
    /// The server takes connections; a drop sets it down until the case brings it back.
    up: AtomicBool,
}

#[derive(Default)]
struct State {
    server: Option<Server>,
    /// The connection open now; a frame for another one is refused.
    connection: u64,
    open: bool,
    /// Generations the client knows of and has not seen end, by source.
    generations: HashMap<u32, Slot>,
    /// The session is closed for good.
    closed: bool,
}

impl Socket {
    /// The connection ends: every generation on it with it. For good on a close.
    fn hang_up(&self, closed: bool) {
        let mut state = self.state.lock().unwrap();
        state.open = false;
        state.closed |= closed;
        state.server = None;
        for (_, slot) in state.generations.drain() {
            self.registry.mark_closed(slot);
        }
        if closed && let Some(slot) = self.slot.get() {
            self.registry.mark_closed(*slot);
        }
        drop(state);
        self.wake.notify_one();
    }

    fn now(&self) -> lk_agents_abi::Ts {
        self.clock.now()
    }

    /// The session closed. The socket is closed for good only if the client closed its
    /// connection: one still open was dropped without a close frame.
    fn end_session(&self) {
        let open = {
            let mut state = self.state.lock().unwrap();
            state.closed = true;
            state.open
        };
        if !open && let Some(slot) = self.slot.get() {
            self.registry.mark_closed(*slot);
        }
        self.wake.notify_one();
    }
}

/// What the server's deliveries mean to the run's record.
struct Hooks<'a> {
    socket: &'a Socket,
    generations: &'a mut HashMap<u32, Slot>,
}

impl server::Hooks for Hooks<'_> {
    fn created(&mut self, _: &str) -> u32 {
        let slot = self.socket.registry.register(Kind::Generation, Detail::None, Control::Generation);
        self.generations.insert(slot.nth, slot);
        slot.nth
    }

    fn done(&mut self, source: u32) {
        if let Some(slot) = self.generations.remove(&source) {
            self.socket.registry.mark_closed(slot);
        }
    }

    fn produced(&mut self, source: u32, chunk: u16) {
        self.socket.registry.record_generated(source, chunk);
    }

    fn breach(&mut self, breach: crate::record::Breach) {
        if let Some(slot) = self.socket.slot.get() {
            self.socket.registry.record_breach(slot.nth, breach);
        }
    }

    fn heard(&mut self, pcm: &[u8]) {
        if let Some(slot) = self.socket.slot.get() {
            let (pairs, _) = pcm.as_chunks::<2>();
            let tags = pairs.iter().map(|b| Tag::decode(i16::from_le_bytes(*b))).collect();
            self.socket.registry.record_forwarded(*slot, tags);
        }
    }
}

/// One server-side event onto a registered socket. True when a real server could do it now.
pub(super) fn deliver(socket: &Socket, what: &Input) -> bool {
    let Input::RealtimeEvent { event, .. } = what else { return false };
    let accepted = match event {
        RtEvent::Disconnected => {
            let open = socket.state.lock().unwrap().open;
            if open {
                socket.up.store(false, Ordering::SeqCst);
                socket.hang_up(false);
            }
            open
        }
        RtEvent::Reconnected => !socket.up.swap(true, Ordering::SeqCst),
        other => {
            let exo = match other {
                RtEvent::SpeechStarted => Exo::SpeechStarted,
                RtEvent::SpeechStopped => Exo::SpeechStopped,
                RtEvent::StrayDelta => Exo::StrayDelta,
                RtEvent::SecondGeneration => Exo::ServerResponse,
                RtEvent::GenerationDone => Exo::FinishNow,
                RtEvent::ServerError => Exo::ServerError,
                RtEvent::FatalError => Exo::FatalError,
                RtEvent::ResponseFailed => Exo::ResponseFailed,
                RtEvent::TranscriptionFailed => Exo::TranscriptionFailed,
                RtEvent::Disconnected | RtEvent::Reconnected => unreachable!("answered above"),
            };
            let now = socket.now();
            let mut state = socket.state.lock().unwrap();
            match state.server.as_mut() {
                Some(server) => server.exo(now, exo),
                None => false,
            }
        }
    };
    socket.wake.notify_one();
    accepted
}

struct Transport(Arc<Socket>);

#[async_trait::async_trait]
impl WsTransport for Transport {
    async fn connect(
        &self,
        _: &str,
        _: Vec<(String, String)>,
    ) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), TransportError> {
        let socket = &self.0;
        if !socket.connect.is_zero() {
            tokio::time::sleep(socket.connect).await;
        }
        if !socket.up.load(Ordering::SeqCst) {
            return Err(TransportError::Status {
                status: 503,
                message: "Service Unavailable".into(),
                headers: Vec::new(),
                body: None,
            });
        }
        let connection = {
            let mut state = socket.state.lock().unwrap();
            if state.closed {
                return Err(TransportError::Closed);
            }
            state.connection += 1;
            state.open = true;
            let mut server =
                Server::new(socket.seed ^ state.connection)
                    .with_tool_calls(socket.tool_calls)
                    // Unique across the session's sockets too, as a real server's ids are.
                    .with_ids(format!("{}s{}c", socket.slot.get().map_or(0, |s| s.nth), state.connection));
            server.connect(socket.now());
            state.server = Some(server);
            state.connection
        };
        socket.wake.notify_one();
        Ok((
            Arc::new(Sender { socket: socket.clone(), connection }),
            Box::new(Receiver { socket: socket.clone(), connection }),
        ))
    }
}

struct Sender {
    socket: Arc<Socket>,
    connection: u64,
}

#[async_trait::async_trait]
impl WsSender for Sender {
    async fn send_text(&self, text: String) -> Result<(), TransportError> {
        let socket = &self.socket;
        let now = socket.now();
        {
            let mut guard = socket.state.lock().unwrap();
            let state = &mut *guard;
            if !state.open || state.connection != self.connection {
                return Err(TransportError::Closed);
            }
            let Ok(frame) = serde_json::from_str::<serde_json::Value>(&text) else {
                return Ok(());
            };
            let server = state.server.as_mut().expect("an open connection has a server");
            let mut hooks = Hooks { socket, generations: &mut state.generations };
            server.client(now, &frame, &mut hooks);
        }
        socket.wake.notify_one();
        Ok(())
    }

    async fn send_binary(&self, _: Vec<u8>) -> Result<(), TransportError> {
        Ok(())
    }

    fn is_closed(&self) -> bool {
        let state = self.socket.state.lock().unwrap();
        !state.open || state.connection != self.connection
    }

    async fn close(&self) {
        let held = {
            let state = self.socket.state.lock().unwrap();
            let mine = state.open && state.connection == self.connection;
            mine.then(|| state.server.as_ref().map(Server::held).unwrap_or_default())
        };
        if let Some(held) = held {
            if let Some(slot) = self.socket.slot.get() {
                self.socket.registry.record_held(slot.nth, held);
            }
            self.socket.hang_up(false);
        }
    }
}

struct Receiver {
    socket: Arc<Socket>,
    connection: u64,
}

#[async_trait::async_trait]
impl WsReceiver for Receiver {
    async fn recv(&mut self) -> Result<WsMessage, TransportError> {
        let socket = self.socket.clone();
        loop {
            let due = {
                let mut guard = socket.state.lock().unwrap();
                let state = &mut *guard;
                if !state.open || state.connection != self.connection {
                    return Ok(WsMessage::Closed);
                }
                let now = socket.now();
                let server = state.server.as_mut().expect("an open connection has a server");
                let mut hooks = Hooks { socket: &socket, generations: &mut state.generations };
                if let Some(frame) = server.pop(now, &mut hooks) {
                    return Ok(WsMessage::Text(frame.to_string()));
                }
                server.next_due()
            };
            match due {
                Some(at) => {
                    tokio::select! {
                        () = socket.clock.wait_until(at) => {}
                        () = socket.wake.notified() => {}
                    }
                }
                None => socket.wake.notified().await,
            }
        }
    }
}
