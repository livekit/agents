//! The realtime model: the production realtime core (`lk_agents_inference::realtime`, the
//! conversation machine and connection policy a LiveKit Inference realtime model runs on), driven
//! the way the runtime's session pump drives it, over [`server::Server`], the OpenAI Realtime
//! server as the real one answers (`corpus/probes/`) and the corpus shows it.
//!
//! So what reaches the driver is what the production core makes of real server traffic: its
//! reconnects, its stale-response bookkeeping, its transcripts and its errors are under test with
//! the driver, not stood in for. The pump is inference's (`lk_agents_inference::realtime::native::session`)
//! without its I/O: the socket is the server's typed records, the person's audio reaches it in the
//! writer's 100 ms batches, and the server answers on its own clock with the delays the corpus
//! measured. What the server does of its own accord -- hears the person start and stop (its VAD),
//! drops the socket, errs -- is a `RealtimeEvent` the case sends.
//!
//! The socket is a resource (`Kind::Realtime`), one per session the driver opens, and every
//! response the client has been told of is one too (`Kind::Generation`), until its
//! `response.done`. Audio deltas are tagged with the generation and its chunk. What the server
//! refused, or answered from the wrong state, is recorded as a `Breach` on the socket.
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use lk_agents_abi::{
    AbortHandle, ApiConnectOptions, AudioFrame, ChatContext, Clock, Realtime, RealtimeCommand,
    RealtimeSession, Spawner, SpawnerExt, Ts,
    errors::{ApiError, ProviderError},
    llm::realtime::{Event, Interrupt, RealtimeCapabilities, RealtimeModelError},
    utils::byte_stream::AudioByteStream,
};
use lk_agents_inference::realtime::{
    ConnectionEffect, INFERENCE, Outgoing, RealtimeConnection, RealtimeMachine, SessionOptions,
    Step, TurnDetection,
};
use lk_agents_net::transport::{TransportError, connect_error};

use super::{Control, Registry, SAMPLE_RATE, Slot};
use crate::{
    input::{Input, RtEvent},
    record::{Detail, Kind, Tag},
};

pub mod duplex;
pub mod fallback;
pub mod server;

pub use duplex::FakeDuplex;
pub use fallback::FallbackRealtime;
pub use server::{Exo, Server, Vad};

/// The model a session names in its metrics.
pub const MODEL: &str = "openai/gpt-realtime";

/// `RealtimeModel.label` of a LiveKit Inference model: on every error and metric.
const LABEL: &str = "lk_agents_realtime.LiveKit Inference Realtime";

/// How long before the session the model was built (at the entrypoint, or in a prewarm): the
/// core runs on the model's own clock (`RealClock` from its construction), never the session's.
const MODEL_BUILT_BEFORE: Duration = Duration::from_secs(3_600);

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

    /// The model's options, as a LiveKit Inference realtime model built for this setup has them.
    fn options(&self) -> SessionOptions {
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
        opts.audio_output = !self.setup.text_only;
        opts
    }

    /// Server detection left at its default is the driver's to switch off when it detects turns
    /// itself; switched off by the developer, it stays off.
    fn capabilities_of(&self, opts: &SessionOptions) -> RealtimeCapabilities {
        let given = self.setup.vad.is_none();
        let transcription = opts.transcription.is_some();
        INFERENCE.capabilities(
            opts.turn_detection.as_ref(),
            given,
            opts.audio_output,
            transcription,
        )
    }
}

#[async_trait::async_trait]
impl Realtime for FakeRealtime {
    fn label(&self) -> String {
        "fake::realtime".into()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        self.capabilities_of(&self.options())
    }

    /// Running when it returns, as the runtime's session is: a connect failure arrives as an
    /// `Event::Error`, like a reconnect failure.
    async fn session(&self, disabled: bool) -> Result<Arc<dyn RealtimeSession>, ProviderError> {
        let nth = self.sessions.fetch_add(1, Ordering::SeqCst);
        let mut opts = self.options();
        if disabled && INFERENCE.supports_turn_detection_disabled {
            opts.turn_detection = None;
        }
        let caps = self.capabilities_of(&opts);
        let conn = ApiConnectOptions::default();
        let born = self.clock.now().saturating_add(MODEL_BUILT_BEFORE);
        let machine = RealtimeMachine::new(INFERENCE, opts, caps, "1".into())
            .with_response_timeout(conn.read_timeout.map(Duration::from_secs_f64))
            .with_label(LABEL.into())
            .with_provider("livekit".into())
            .with_wall_anchor(self.clock.wall() - Duration::from_nanos(born.as_nanos()));
        let mut connection = RealtimeConnection::new(&conn, born);
        let effects = connection.start(born).into();
        let socket = Arc::new(Socket {
            tool_calls: self.setup.tool_calls,
            seed: self.setup.seed ^ (nth << 32),
            connect: self.setup.connect,
            clock: self.clock.clone(),
            registry: self.registry.clone(),
            slot: OnceLock::new(),
            state: Mutex::new(State {
                machine,
                connection,
                link: Link::Waiting,
                effects,
                reconnecting: false,
                commands: VecDeque::new(),
                closing: false,
                events: VecDeque::new(),
                audio: batches(),
                generations: HashMap::new(),
                connections: 0,
                closed: false,
            }),
            wake: tokio::sync::Notify::new(),
            ready: tokio::sync::Notify::new(),
            up: AtomicBool::new(true),
        });
        let slot =
            self.registry.register(Kind::Realtime, Detail::None, Control::Realtime(socket.clone()));
        let _ = socket.slot.set(slot);
        let pump = socket.clone();
        let abort = self.spawner.spawn_unboxed(async move { pump.run().await });
        Ok(Arc::new(Session { socket, abort }))
    }
}

/// The writer's batching of the person's audio: whole 100 ms appends, the rest held back.
fn batches() -> AudioByteStream {
    AudioByteStream::new(SAMPLE_RATE, 1, None, false)
}

/// The session the driver holds.
struct Session {
    socket: Arc<Socket>,
    abort: AbortHandle,
}

#[async_trait::async_trait]
impl RealtimeSession for Session {
    fn capabilities(&self) -> RealtimeCapabilities {
        self.socket.state.lock().unwrap().machine.capabilities()
    }

    /// Never blocks, never fails loudly: queued for the pump, as on the runtime's channel.
    fn send(&self, cmd: RealtimeCommand) {
        self.socket.state.lock().unwrap().commands.push_back(cmd);
        self.socket.wake.notify_one();
    }

    async fn next_event(&self) -> Option<Event> {
        self.socket
            .until(|state| match state.events.pop_front() {
                Some(event) => Some(Some(event)),
                None => matches!(state.link, Link::Over).then_some(None),
            })
            .await
    }

    /// The mirror: what the server believes.
    async fn chat_ctx(&self) -> ChatContext {
        self.socket.state.lock().unwrap().machine.chat_ctx()
    }

    /// The runtime's close: the response in flight interrupted and the close signalled, the pump
    /// given the teardown bound to settle; then the session is over for the server. A connection
    /// the pump left open (no close frame) keeps the socket unclosed, as a server still billing
    /// it would.
    async fn aclose(&self) {
        let socket = &self.socket;
        {
            let mut state = socket.state.lock().unwrap();
            state.commands.push_back(RealtimeCommand::Interrupt(Interrupt { response_id: None }));
            state.closing = true;
        }
        socket.wake.notify_one();
        let over = socket.until(|state| matches!(state.link, Link::Over).then_some(()));
        let _ = lk_agents_abi::timeout(&*socket.clock, lk_agents_abi::HOST_TEARDOWN_TIMEOUT, over)
            .await;
        self.abort.abort();
        let open = {
            let mut state = socket.state.lock().unwrap();
            state.closed = true;
            let open = matches!(state.link, Link::Open { server: Some(_), .. });
            state.link = Link::Over;
            open
        };
        socket.ready.notify_waiters();
        if !open && let Some(slot) = socket.slot.get() {
            socket.registry.mark_closed(*slot);
        }
    }
}

/// Where the session's connection stands.
enum Link {
    /// Between attempts: on the connection policy's next deadline, or its next effect.
    Waiting,
    /// An attempt in flight, opening at `opens_at` (the run's clock); `started` on the model's.
    Connecting { generation: u64, started: Ts, opens_at: Ts },
    /// A connection, and the server behind it until the socket drops (`None`).
    Open { generation: u64, server: Option<Server> },
    /// The pump ended: no event comes after the queued ones.
    Over,
}

/// What one connection ended in, as the runtime's `connect_and_run` reports it.
enum Outcome {
    /// The session closed.
    Closed,
    /// The server said the session cannot go on.
    Fatal(RealtimeModelError),
    /// The connection policy's deadline ended the attempt.
    Effects(Vec<ConnectionEffect>),
    /// The core asked for a fresh connection.
    Recycle,
    /// The attempt failed, or the socket was lost.
    Failed(ApiError),
}

struct State {
    machine: RealtimeMachine,
    connection: RealtimeConnection,
    link: Link,
    /// Connection effects not yet acted on, in the order the runtime's loop takes them.
    effects: VecDeque<ConnectionEffect>,
    /// A connection was attempted before: the next one replays the conversation.
    reconnecting: bool,
    /// The core's commands, as the runtime's channel holds them for the pump.
    commands: VecDeque<RealtimeCommand>,
    /// `aclose` was called.
    closing: bool,
    events: VecDeque<Event>,
    audio: AudioByteStream,
    /// Generations the client knows of and has not seen end, by source.
    generations: HashMap<u32, Slot>,
    /// Connections opened, for each one's server.
    connections: u64,
    /// The session is closed for good: the server takes no connection.
    closed: bool,
}

/// When the pump runs next.
enum Wake {
    At(Ts),
    Idle,
    Over,
}

/// Whether connection effects end the attempt in flight: anything but a cleanup diagnostic.
fn ends_attempt(effects: &[ConnectionEffect]) -> bool {
    effects.iter().any(|e| !matches!(e, ConnectionEffect::CleanupIncomplete { .. }))
}

/// One session's socket: the core and the server behind it.
pub struct Socket {
    seed: u64,
    tool_calls: Option<u8>,
    connect: Duration,
    clock: Arc<dyn Clock>,
    registry: Arc<Registry>,
    slot: OnceLock<Slot>,
    state: Mutex<State>,
    /// Told whenever the pump may have something new to do.
    wake: tokio::sync::Notify,
    /// Told whenever an event is queued or the pump ended.
    ready: tokio::sync::Notify,
    /// The server takes connections; a drop sets it down until the case brings it back.
    up: AtomicBool,
}

impl Socket {
    fn now(&self) -> Ts {
        self.clock.now()
    }

    fn model_now(&self) -> Ts {
        self.now().saturating_add(MODEL_BUILT_BEFORE)
    }

    /// The first value `read` finds in the state, waiting for the pump between reads.
    async fn until<T>(&self, mut read: impl FnMut(&mut State) -> Option<T>) -> T {
        loop {
            let ready = self.ready.notified();
            tokio::pin!(ready);
            ready.as_mut().enable();
            if let Some(value) = read(&mut self.state.lock().unwrap()) {
                return value;
            }
            ready.await;
        }
    }

    async fn run(self: Arc<Self>) {
        loop {
            let wake = {
                let mut state = self.state.lock().unwrap();
                self.advance(&mut state)
            };
            self.ready.notify_waiters();
            match wake {
                Wake::Over => return,
                Wake::At(at) => {
                    tokio::select! {
                        () = self.clock.wait_until(at) => {}
                        () = self.wake.notified() => {}
                    }
                }
                Wake::Idle => self.wake.notified().await,
            }
        }
    }

    /// Everything the pump can do now; then when it runs next.
    fn advance(&self, st: &mut State) -> Wake {
        loop {
            let (now, m) = (self.now(), self.model_now());
            // The connection policy's next deadline, on the run's clock.
            let policy =
                st.connection.next_deadline().map(|at| at.saturating_sub(MODEL_BUILT_BEFORE));
            match st.link {
                Link::Over => return Wake::Over,
                Link::Waiting => {
                    if let Some(effect) = st.effects.pop_front() {
                        self.effect(st, effect);
                    } else if st.closing {
                        let _ = st.connection.close(m);
                        self.end(st, true);
                    } else {
                        match policy {
                            None => self.end(st, false),
                            Some(at) if at <= now => {
                                let effects = st.connection.tick(m);
                                st.effects.extend(effects);
                            }
                            Some(at) => return Wake::At(at),
                        }
                    }
                }
                Link::Connecting { generation, started, opens_at } => {
                    if st.closing {
                        self.ended(st, generation, Outcome::Closed);
                    } else if policy.is_some_and(|at| at <= now) {
                        let effects = st.connection.tick(m);
                        if ends_attempt(&effects) {
                            self.ended(st, generation, Outcome::Effects(effects));
                        }
                    } else if now >= opens_at {
                        self.opened(st, generation, started);
                    } else {
                        return Wake::At(policy.map_or(opens_at, |at| at.min(opens_at)));
                    }
                }
                Link::Open { generation, ref server } => {
                    let lost = server.is_none();
                    let server_due = server.as_ref().and_then(Server::next_due);
                    let deadline = st
                        .machine
                        .next_deadline()
                        .map(|at| at.saturating_sub(MODEL_BUILT_BEFORE))
                        .into_iter()
                        .chain(policy)
                        .min();
                    // The runtime's biased order: the deadline, the close, the socket, the
                    // commands, the context work: retired histories go to a cleanup worker,
                    // a context page continues only once the machine can take it.
                    if deadline.is_some_and(|at| at <= now) {
                        let effects = st.connection.tick(m);
                        if ends_attempt(&effects) {
                            self.ended(st, generation, Outcome::Effects(effects));
                        } else {
                            let step = st.machine.tick(m);
                            self.stepped(st, generation, step);
                        }
                    } else if st.closing {
                        self.ended(st, generation, Outcome::Closed);
                    } else if lost {
                        self.ended(st, generation, Outcome::Failed(lost_connection()));
                    } else if server_due.is_some_and(|at| at <= now) {
                        let wall = self.clock.wall();
                        let State { link, generations, machine, .. } = &mut *st;
                        let Link::Open { server: Some(server), .. } = link else { unreachable!() };
                        let mut hooks = Hooks { socket: self, generations };
                        let step = server.pop(now, wall, &mut hooks).map(|delivered| {
                            machine.incoming_on(machine.connection_generation(), m, delivered.event)
                        });
                        if let Some(step) = step {
                            self.stepped(st, generation, step);
                        }
                    } else if let Some(cmd) = st.commands.pop_front() {
                        let step = st.machine.command(m, cmd);
                        self.stepped(st, generation, step);
                    } else if let Some(retired) = st.machine.take_context_retirement() {
                        // an admitted cleanup worker reclaims the retired history
                        drop(retired);
                    } else if st.machine.context_continuation_ready() {
                        let step = st.machine.continue_context(m);
                        self.stepped(st, generation, step);
                    } else {
                        let at = deadline.into_iter().chain(server_due).min();
                        return at.map_or(Wake::Idle, Wake::At);
                    }
                }
            }
        }
    }

    /// One connection effect, as the runtime's loop acts on it.
    fn effect(&self, st: &mut State, effect: ConnectionEffect) {
        match effect {
            ConnectionEffect::Connect { generation } => {
                let opens_at = self.now().saturating_add(self.connect);
                st.link = Link::Connecting { generation, started: self.model_now(), opens_at };
            }
            ConnectionEffect::Retry { error } => self.emit_error(st, error, true),
            ConnectionEffect::Failed { error } => {
                self.emit_error(st, error, false);
                self.end(st, true);
            }
            ConnectionEffect::Close { .. } | ConnectionEffect::CleanupIncomplete { .. } => {}
        }
    }

    /// The attempt's socket opens now: or the server is down, or the session closed for good.
    fn opened(&self, st: &mut State, generation: u64, started: Ts) {
        let refused = if st.closed {
            Some(TransportError::Closed)
        } else if !self.up.load(Ordering::SeqCst) {
            Some(TransportError::Status {
                status: 503,
                message: "Service Unavailable".into(),
                headers: Vec::new(),
                body: None,
            })
        } else {
            None
        };
        if let Some(error) = refused {
            let error = connect_error(INFERENCE.provider_label, error);
            return self.ended(st, generation, Outcome::Failed(error));
        }
        let (now, m) = (self.now(), self.model_now());
        st.connections += 1;
        let ids = format!("{}s{}c", self.slot.get().map_or(0, |s| s.nth), st.connections);
        let mut server = Server::new(self.seed ^ st.connections)
            .with_tool_calls(self.tool_calls)
            // Unique across the session's sockets too, as a real server's ids are.
            .with_ids(ids);
        server.connect(now);
        st.link = Link::Open { generation, server: Some(server) };
        st.audio = batches();
        let acquired = m.saturating_duration_since(started);
        let metrics = st.machine.connection_metrics(m, acquired);
        st.events.push_back(metrics);
        let step = if st.reconnecting { st.machine.reconnect(m) } else { st.machine.open(m) };
        if !st.connection.connected(generation, m) {
            return self.ended(st, generation, Outcome::Closed);
        }
        self.stepped(st, generation, step);
    }

    /// What the core decided: its frames onto the socket, its events to the driver.
    fn stepped(&self, st: &mut State, generation: u64, mut step: Step) {
        let fatal = step.fatal.take();
        let recycle = step.recycle;
        self.dispatch(st, step);
        if let Some(error) = fatal {
            self.ended(st, generation, Outcome::Fatal(error));
        } else if recycle {
            self.ended(st, generation, Outcome::Recycle);
        }
    }

    fn dispatch(&self, st: &mut State, step: Step) {
        let (now, m) = (self.now(), self.model_now());
        let State { machine, link, audio, generations, events, .. } = st;
        if let Link::Open { server: Some(server), .. } = link {
            // The machine's own connection generation, not the policy's attempt's.
            let generation = machine.connection_generation();
            let mut hooks = Hooks { socket: self, generations };
            for frame in step.frames {
                match frame {
                    Outgoing::AudioAppend { frame } => {
                        for chunk in audio.push(&frame.data) {
                            let samples = chunk.samples_per_channel;
                            let frame = AudioFrame::new(chunk.data, SAMPLE_RATE, 1, samples);
                            server.client(now, &Outgoing::AudioAppend { frame }, &mut hooks);
                        }
                    }
                    Outgoing::AudioClear => {
                        *audio = batches();
                        server.client(now, &Outgoing::AudioClear, &mut hooks);
                    }
                    frame => {
                        server.client(now, &frame, &mut hooks);
                        // The writer's receipt: the request is on the wire.
                        match &frame {
                            Outgoing::ResponseCreate { req_id, .. } => {
                                machine.response_sent(generation, m, req_id, false);
                            }
                            Outgoing::ForceMessage { req_id, .. } => {
                                machine.response_sent(generation, m, req_id, true);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        events.extend(step.events);
    }

    /// One connection ended, as `connect_and_run` returned: the socket closed with a close frame
    /// if it was still up, and what the connection policy makes of the outcome.
    fn ended(&self, st: &mut State, generation: u64, outcome: Outcome) {
        let m = self.model_now();
        if let Link::Open { server: Some(server), .. } = &st.link {
            if let Some(slot) = self.slot.get() {
                self.registry.record_held(slot.nth, server.held());
            }
            self.hang_up(st);
        }
        st.link = Link::Waiting;
        st.reconnecting = true;
        let follow_up = match outcome {
            Outcome::Closed => {
                let _ = st.connection.close(m);
                let _ = st.connection.released(generation, m);
                // A close the pump took ends the session; a refused handover ends it silently.
                let close = st.closing;
                return self.end(st, close);
            }
            Outcome::Fatal(error) => {
                st.events.push_back(Event::Error(error));
                let _ = st.connection.close(m);
                let _ = st.connection.released(generation, m);
                return self.end(st, true);
            }
            Outcome::Effects(effects) => effects,
            Outcome::Recycle => st.connection.failed(generation, m, lost_connection()),
            Outcome::Failed(error) => st.connection.failed(generation, m, error),
        };
        st.effects.extend(follow_up);
        let released = st.connection.released(generation, m);
        st.effects.extend(released);
    }

    /// The pump ends; `close`: the core closes, failing what still waits.
    fn end(&self, st: &mut State, close: bool) {
        if close {
            let step = st.machine.close(self.model_now());
            st.events.extend(step.events);
        }
        st.link = Link::Over;
    }

    fn emit_error(&self, st: &mut State, error: ApiError, recoverable: bool) {
        st.events.push_back(Event::Error(RealtimeModelError {
            timestamp: self.model_now(),
            label: LABEL.into(),
            message: error.message.clone(),
            recoverable,
            req_id: None,
            code: None,
            provider_error: Some(error),
        }));
    }

    /// The connection ends, every generation on it with it.
    fn hang_up(&self, st: &mut State) {
        if let Link::Open { server, .. } = &mut st.link {
            *server = None;
        }
        for (_, slot) in st.generations.drain() {
            self.registry.mark_closed(slot);
        }
        self.wake.notify_one();
    }
}

/// "connection closed unexpectedly": retryable, so the session reconnects.
fn lost_connection() -> ApiError {
    let message = format!("{} connection closed unexpectedly", INFERENCE.provider_label);
    ApiError::connection(Some(message))
}

/// What the server's deliveries mean to the run's record.
struct Hooks<'a> {
    socket: &'a Socket,
    generations: &'a mut HashMap<u32, Slot>,
}

impl server::Hooks for Hooks<'_> {
    fn created(&mut self, _: &str) -> u32 {
        let slot =
            self.socket.registry.register(Kind::Generation, Detail::None, Control::Generation);
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
            let mut state = socket.state.lock().unwrap();
            let open = matches!(state.link, Link::Open { server: Some(_), .. });
            if open {
                socket.up.store(false, Ordering::SeqCst);
                socket.hang_up(&mut state);
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
            match &mut socket.state.lock().unwrap().link {
                Link::Open { server: Some(server), .. } => server.exo(now, exo),
                _ => false,
            }
        }
    };
    socket.wake.notify_one();
    accepted
}
