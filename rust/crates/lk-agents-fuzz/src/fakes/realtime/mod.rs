//! The realtime model at the `Realtime` boundary: a LiveKit Inference realtime session as the
//! driver sees it, over [`server::Server`], the realtime server as the corpus shows the real one
//! doing (`corpus/calibration.json`, `corpus/probes/` in lk-agents-test-support).
//!
//! The session reports what the server does as the driver's `Event`s, the way the native session
//! does: the response it was asked for or opened on its own, its message, words and audio, its
//! end and metrics; the items the server added; the person's speech and transcript; errors, and
//! the connection's loss, retries and return. It keeps what the native session keeps -- the
//! conversation the server is being brought to, the replies asked and not yet answered, the
//! response it told the driver of -- so a reconnect restores the conversation and asks again for
//! what was not answered, a reply cancelled before it was created never reaches the driver, and
//! what a response sends after its end is told to nobody. Inference's own tests hold the native
//! session's machine to the recorded server; this fake reproduces its outcomes, its timings and
//! its failure shapes.
//!
//! The socket is a resource (`Kind::Realtime`), one per session the driver opens, and every
//! response the session was told of is one too (`Kind::Generation`), until its end. Audio deltas
//! are tagged with the generation and its chunk. What the server refused, or answered from the
//! wrong state, is recorded as a `Breach` on the socket. What the server does of its own accord --
//! hears the person start and stop (its VAD), drops the socket, errs -- is a `RealtimeEvent` the
//! case sends.
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use lk_agents_abi::{
    AbortHandle, ApiConnectOptions, BoxFuture, ChatContext, ChatItem, Clock, Realtime,
    RealtimeCommand, RealtimeSession, Spawner, SpawnerExt, ToolChoice, Ts,
    errors::{ApiError, ProviderError, TimeoutPhase},
    llm::realtime::{
        Event, GenerateReplyFailed, GenerationDone, GenerationStatus, InputSpeechStopped,
        RealtimeCapabilities, RealtimeError, SessionReconnected, TokenUsage,
    },
};

use super::{Control, Registry, SAMPLE_RATE, Slot};
use crate::{
    input::{Input, RtEvent},
    record::{Detail, Kind, Tag},
};

pub mod duplex;
mod events;
pub mod fallback;
pub mod server;

pub use duplex::FakeDuplex;
use events::{message_done, modalities};
pub use fallback::FallbackRealtime;
pub use server::{Exo, Server, Vad};

/// The model a session names in its metrics.
pub const MODEL: &str = "openai/gpt-realtime";

/// The session's input transcription model.
const TRANSCRIPTION_MODEL: &str = "gpt-4o-mini-transcribe";

/// `RealtimeModel.label` of a LiveKit Inference model: on every error and metric.
const LABEL: &str = "lk_agents_realtime.LiveKit Inference Realtime";

/// What the session's errors call the provider.
const PROVIDER: &str = "LiveKit Inference Realtime";

/// How long before the session the model was built (at the entrypoint, or in a prewarm): its
/// errors are stamped on the model's own clock (`RealClock` from its construction).
const MODEL_BUILT_BEFORE: Duration = Duration::from_secs(3_600);

/// The samples an audio delta carries.
const DELTA_SAMPLES: usize = (server::DELTA_MS * SAMPLE_RATE as u64 / 1000) as usize;

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
}

/// What a LiveKit Inference realtime model reports: server detection left at its default is the
/// driver's to switch off when it detects turns itself; switched off by the developer, it stays
/// off. The session transcribes the person and never speaks a `say` itself.
fn capabilities(vad: Option<Vad>, configured: bool, text_only: bool) -> RealtimeCapabilities {
    RealtimeCapabilities {
        message_truncation: true,
        turn_detection: vad.is_some(),
        user_transcription: true,
        auto_tool_reply_generation: false,
        audio_output: !text_only,
        manual_function_calls: true,
        can_disable_turn_detection: configured,
        mutable_chat_context: true,
        mutable_instructions: true,
        mutable_tools: true,
        per_response_tool_choice: true,
        supports_say: false,
        auto_turn_reply_generation: vad.is_some(),
    }
}

impl Realtime for FakeRealtime {
    fn label(&self) -> String {
        "fake::realtime".into()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        let vad = self.setup.vad;
        capabilities(vad, vad.is_some(), self.setup.text_only)
    }

    /// Running when it returns, as the runtime's session is: a connect failure arrives as an
    /// `Event::Error`, like a reconnect failure.
    fn session(
        &self,
        disabled: bool,
    ) -> BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
        Box::pin(async move {
            let nth = self.sessions.fetch_add(1, Ordering::SeqCst);
            let vad = self.setup.vad.filter(|_| !disabled);
            let caps = capabilities(vad, self.setup.vad.is_some(), self.setup.text_only);
            let now = self.clock.now();
            let opts = ApiConnectOptions::default();
            let socket = Arc::new(Socket {
                setup: ServerSetup { vad, ..self.setup },
                seed: self.setup.seed ^ (nth << 32),
                caps,
                opts,
                clock: self.clock.clone(),
                registry: self.registry.clone(),
                slot: OnceLock::new(),
                state: Mutex::new(State::new(attempt(now, &opts, self.setup.connect))),
                wake: tokio::sync::Notify::new(),
                ready: tokio::sync::Notify::new(),
                up: AtomicBool::new(true),
            });
            let slot = self.registry.register(
                Kind::Realtime,
                Detail::None,
                Control::Realtime(socket.clone()),
            );
            let _ = socket.slot.set(slot);
            let pump = socket.clone();
            let abort = self.spawner.spawn_unboxed(async move { pump.run().await });
            Ok(Arc::new(Session { socket, abort }) as Arc<dyn RealtimeSession>)
        })
    }
}

/// A connection attempt from `now`: open after the setup's delay, unless the connect bound ends it.
fn attempt(now: Ts, opts: &ApiConnectOptions, connect: Duration) -> Link {
    let bound = opts.connect_timeout.map(Duration::from_secs_f64);
    Link::Connecting {
        started: now,
        opens_at: now.saturating_add(connect),
        gives_up_at: bound.filter(|b| connect >= *b).map(|b| now.saturating_add(b)),
    }
}

/// The session the driver holds.
struct Session {
    socket: Arc<Socket>,
    abort: AbortHandle,
}

impl RealtimeSession for Session {
    fn capabilities(&self) -> RealtimeCapabilities {
        self.socket.caps
    }

    /// Never blocks, never fails loudly: queued for the pump, as on the runtime's channel.
    fn send(&self, cmd: RealtimeCommand) {
        self.socket.state.lock().unwrap().commands.push_back(cmd);
        self.socket.wake.notify_one();
    }

    fn next_event(&self) -> BoxFuture<'_, Option<Event>> {
        Box::pin(self.socket.until(|state| match state.events.pop_front() {
            Some(event) => Some(Some(event)),
            None => matches!(state.link, Link::Over).then_some(None),
        }))
    }

    /// The mirror: what the server is being brought to.
    fn chat_ctx(&self) -> BoxFuture<'_, ChatContext> {
        Box::pin(async move { ChatContext::new(self.socket.state.lock().unwrap().desired.clone()) })
    }

    /// The runtime's close: the pump given the teardown bound to settle; then the session is over.
    fn aclose(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            let socket = &self.socket;
            socket.state.lock().unwrap().closing = true;
            socket.wake.notify_one();
            let over = socket.until(|state| matches!(state.link, Link::Over).then_some(()));
            let _ =
                lk_agents_abi::timeout(&*socket.clock, lk_agents_abi::HOST_TEARDOWN_TIMEOUT, over)
                    .await;
            self.abort.abort();
            {
                let mut state = socket.state.lock().unwrap();
                socket.hang_up(&mut state);
                state.link = Link::Over;
            }
            socket.ready.notify_waiters();
            if let Some(slot) = socket.slot.get() {
                socket.registry.mark_closed(*slot);
            }
        })
    }
}

/// Where the session's connection stands.
enum Link {
    /// An attempt in flight, opening at `opens_at` unless the connect bound gives up first.
    Connecting { started: Ts, opens_at: Ts, gives_up_at: Option<Ts> },
    /// A connection, and the server behind it until the socket drops (`None`).
    Open { server: Option<Server> },
    /// Between attempts: the next at `at`.
    Retrying { at: Ts },
    /// The session ended: no event comes after the queued ones.
    Over,
}

/// The response the session told the driver of: its messages, and whether each is still open and
/// has settled its modalities.
struct Generation {
    r: u32,
    id: String,
    created_at: Ts,
    first_token: Option<Ts>,
    messages: Vec<(String, bool, bool)>,
}

/// A reply asked for and not yet answered.
struct Asked {
    deadline: Option<Ts>,
    instructions: Option<String>,
    tool_choice: Option<ToolChoice>,
}

struct State {
    link: Link,
    /// The driver's commands, as the runtime's channel holds them for the pump.
    commands: VecDeque<RealtimeCommand>,
    /// `aclose` was called.
    closing: bool,
    events: VecDeque<Event>,
    /// The conversation the server is being brought to.
    desired: Vec<ChatItem>,
    instructions: String,
    tools: Vec<String>,
    asked: BTreeMap<String, Asked>,
    /// Replies cancelled before they were created; `settled` among them: already failed.
    discarded: BTreeSet<String>,
    settled: BTreeSet<String>,
    generation: Option<Generation>,
    /// Assistant messages heard as audio on this connection: what a truncation may cut.
    spoken: BTreeSet<String>,
    /// The person's audio sent since the last commit or clear, in whole 100 ms batches, and what
    /// is left of the last.
    pushed: Duration,
    remainder: Duration,
    user_speaking: bool,
    /// When the server heard each turn start, as wall seconds.
    speech_started: HashMap<String, f64>,
    /// Each turn's transcript so far.
    transcripts: HashMap<String, String>,
    /// Failed attempts since the connection was last open.
    failures: u32,
    /// The session was open once: the next connection restores the conversation.
    reconnecting: bool,
    connections: u64,
    /// Responses of this connection the run registered, by index.
    generations: HashMap<u32, Slot>,
}

impl State {
    fn new(link: Link) -> Self {
        State {
            link,
            commands: VecDeque::new(),
            closing: false,
            events: VecDeque::new(),
            desired: Vec::new(),
            instructions: String::new(),
            tools: Vec::new(),
            asked: BTreeMap::new(),
            discarded: BTreeSet::new(),
            settled: BTreeSet::new(),
            generation: None,
            spoken: BTreeSet::new(),
            pushed: Duration::ZERO,
            remainder: Duration::ZERO,
            user_speaking: false,
            speech_started: HashMap::new(),
            transcripts: HashMap::new(),
            failures: 0,
            reconnecting: false,
            connections: 0,
            generations: HashMap::new(),
        }
    }

    fn server(&mut self) -> Option<&mut Server> {
        match &mut self.link {
            Link::Open { server } => server.as_mut(),
            _ => None,
        }
    }

    /// The response the driver was told of ends unfinished: its messages close, it is cancelled.
    fn close_generation(&mut self, audio: bool) {
        let Some(g) = self.generation.take() else { return };
        for (id, open, settled) in &g.messages {
            if !settled {
                self.events.push_back(modalities(id, &g.id, audio));
            }
            if *open {
                self.events.push_back(message_done(id, &g.id));
            }
        }
        self.events.push_back(Event::GenerationDone(GenerationDone {
            response_id: g.id,
            status: GenerationStatus::Cancelled,
            usage: TokenUsage::default(),
        }));
    }
}

/// One session's socket: the session's state and the server behind it.
pub struct Socket {
    setup: ServerSetup,
    seed: u64,
    caps: RealtimeCapabilities,
    opts: ApiConnectOptions,
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

    fn wall_secs(&self) -> f64 {
        let wall = self.clock.wall();
        wall.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64()
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
                None => return,
                Some(Some(at)) => {
                    tokio::select! {
                        biased;
                        () = self.clock.wait_until(at) => {}
                        () = self.wake.notified() => {}
                    }
                }
                Some(None) => self.wake.notified().await,
            }
        }
    }

    /// Everything the session does now; then when it runs next (`None`: it is over).
    fn advance(&self, st: &mut State) -> Option<Option<Ts>> {
        loop {
            let now = self.now();
            let reply_due = st.asked.values().filter_map(|a| a.deadline).min();
            if reply_due.is_some_and(|at| at <= now) {
                self.time_out_replies(st, now);
                continue;
            }
            match st.link {
                Link::Over => return None,
                Link::Retrying { at } => {
                    if st.closing {
                        self.end(st);
                    } else if at <= now {
                        st.link = attempt(now, &self.opts, self.setup.connect);
                    } else {
                        return Some(Some(reply_due.map_or(at, |r| r.min(at))));
                    }
                }
                Link::Connecting { started, opens_at, gives_up_at } => {
                    if st.closing {
                        self.end(st);
                    } else if gives_up_at.is_some_and(|at| at <= now) {
                        self.failed(st, ApiError::timed_out(TimeoutPhase::Connect));
                    } else if opens_at <= now {
                        if self.up.load(Ordering::SeqCst) {
                            self.opened(st, started);
                        } else {
                            self.failed(st, ApiError::from_http("Service Unavailable", 503, None));
                        }
                    } else {
                        let at = gives_up_at.map_or(opens_at, |g| g.min(opens_at));
                        return Some(Some(reply_due.map_or(at, |r| r.min(at))));
                    }
                }
                Link::Open { ref server } => {
                    let server_due = server.as_ref().and_then(Server::next_due);
                    let lost = server.is_none();
                    // The runtime's biased order: the close, the socket, the commands.
                    if st.closing {
                        self.hang_up(st);
                        self.end(st);
                    } else if lost {
                        self.lost(st);
                    } else if server_due.is_some_and(|at| at <= now) {
                        let happening = st.server().and_then(|s| s.pop(now));
                        if let Some(happening) = happening {
                            self.told(st, now, happening);
                        }
                        self.record_breaches(st);
                    } else if let Some(cmd) = st.commands.pop_front() {
                        self.command(st, now, cmd);
                        self.record_breaches(st);
                    } else {
                        let at = server_due.into_iter().chain(reply_due).min();
                        return Some(at);
                    }
                }
            }
        }
    }

    fn record_breaches(&self, st: &mut State) {
        let breaches = st.server().map(Server::take_breaches).unwrap_or_default();
        if let Some(slot) = self.slot.get() {
            for breach in breaches {
                self.registry.record_breach(slot.nth, breach);
            }
        }
    }

    // --- the connection --------------------------------------------------------------------

    /// The attempt's socket opens now: the session configures the server, and a reconnect
    /// restores the conversation and asks again for what was not answered.
    fn opened(&self, st: &mut State, started: Ts) {
        let now = self.now();
        st.connections += 1;
        st.failures = 0;
        let ids = format!("{}s{}c", self.slot.get().map_or(0, |s| s.nth), st.connections);
        let mut server = Server::new(self.seed ^ st.connections, ids);
        server.configure(self.setup.vad, !self.setup.text_only, self.setup.tool_calls);
        server.instructions(&st.instructions);
        server.tools(st.tools.clone());
        st.link = Link::Open { server: Some(server) };
        st.events.push_back(self.connection_metrics(now.saturating_duration_since(started)));
        if !std::mem::replace(&mut st.reconnecting, true) {
            return;
        }
        // The new connection's VAD heard no onset: the person's speech the old one opened is over
        // as far as the session can tell. Its buffer holds none of the old one's audio.
        if std::mem::take(&mut st.user_speaking) {
            st.events.push_back(Event::InputSpeechStopped(InputSpeechStopped {
                user_transcription_enabled: true,
            }));
        }
        st.pushed = Duration::ZERO;
        st.remainder = Duration::ZERO;
        let desired = st.desired.clone();
        let server = st.server().expect("just opened");
        let mut previous = None;
        for item in &desired {
            server.create(now, item, previous.take());
            previous = Some(item.id().to_owned());
        }
        for req_id in std::mem::take(&mut st.discarded) {
            if !st.settled.remove(&req_id) {
                let error = RealtimeError::new("generate_reply cancelled");
                st.events
                    .push_back(Event::GenerateReplyFailed(GenerateReplyFailed { req_id, error }));
            }
        }
        st.spoken.clear();
        st.close_generation(!self.setup.text_only);
        // A reply asked for and not yet created is asked for again.
        let again: Vec<(String, Option<String>, Option<ToolChoice>)> = st
            .asked
            .iter_mut()
            .rev()
            .map(|(id, a)| {
                a.deadline = None;
                (id.clone(), a.instructions.clone(), a.tool_choice.clone())
            })
            .collect();
        let server = st.server().expect("just opened");
        for (req_id, instructions, tool_choice) in again {
            server.respond_to(now, req_id, instructions.as_deref(), tool_choice);
        }
        st.events.push_back(Event::SessionReconnected(SessionReconnected {}));
    }

    /// The socket dropped: the session reports it and tries again at once.
    fn lost(&self, st: &mut State) {
        let message = format!("{PROVIDER} connection closed unexpectedly");
        self.emit_error(st, ApiError::connection(Some(message)), true);
        st.link = Link::Retrying { at: self.now() };
    }

    /// An attempt failed: retried after the policy's interval, or the session ends.
    fn failed(&self, st: &mut State, error: ApiError) {
        st.failures += 1;
        if st.failures < self.opts.max_retry {
            let interval = self.opts.interval_for_retry(st.failures - 1);
            self.emit_error(st, error, true);
            st.link =
                Link::Retrying { at: self.now().saturating_add(Duration::from_secs_f64(interval)) };
        } else {
            self.emit_error(st, error, false);
            self.end(st);
        }
    }

    /// The connection ends, every generation on it with it: what its server held is recorded.
    fn hang_up(&self, st: &mut State) {
        if let Link::Open { server } = &mut st.link
            && let Some(server) = server.take()
            && let Some(slot) = self.slot.get()
        {
            self.registry.record_held(slot.nth, server.held());
        }
        for (_, slot) in st.generations.drain() {
            self.registry.mark_closed(slot);
        }
    }

    /// The session ends: what still waits fails.
    fn end(&self, st: &mut State) {
        st.close_generation(!self.setup.text_only);
        for (req_id, _) in std::mem::take(&mut st.asked) {
            let error = RealtimeError::new("realtime session closed");
            st.events.push_back(Event::GenerateReplyFailed(GenerateReplyFailed { req_id, error }));
        }
        st.link = Link::Over;
    }

    fn time_out_replies(&self, st: &mut State, now: Ts) {
        let expired: Vec<String> = st
            .asked
            .iter()
            .filter(|(_, a)| a.deadline.is_some_and(|at| at <= now))
            .map(|(id, _)| id.clone())
            .collect();
        for req_id in expired {
            st.asked.remove(&req_id);
            st.discarded.insert(req_id.clone());
            st.settled.insert(req_id.clone());
            let error = RealtimeError::new("response creation timed out");
            st.events.push_back(Event::GenerateReplyFailed(GenerateReplyFailed { req_id, error }));
        }
    }

    // --- the driver's commands -------------------------------------------------------------

    fn command(&self, st: &mut State, now: Ts, cmd: RealtimeCommand) {
        match cmd {
            RealtimeCommand::UpdateInstructions(u) => {
                st.server().expect("open").instructions(&u.instructions);
                st.instructions = u.instructions;
            }
            RealtimeCommand::UpdateTools(u) => {
                st.tools = u.tools.iter().map(|t| t.name.clone()).collect();
                let tools = st.tools.clone();
                st.server().expect("open").tools(tools);
            }
            RealtimeCommand::UpdateChatCtx(u) => {
                let items = u.chat_ctx.items;
                let server = st.server().expect("open");
                // What the conversation loses goes; what it gains is created in its place; a
                // message the driver records as cut is replaced by its text.
                for id in server.ids() {
                    if !items.iter().any(|i| i.id() == id) {
                        server.delete(&id);
                    }
                }
                let mut previous = None;
                for item in &items {
                    let unheld_output = matches!(item, ChatItem::FunctionCallOutput(o) if !server.holds_call(&o.call_id));
                    if !server.holds(item.id()) && !unheld_output {
                        server.create(now, item, previous.clone());
                    } else if matches!(item, ChatItem::Message(m) if m.interrupted) {
                        server.as_text(item.id());
                    }
                    if server.holds(item.id()) {
                        previous = Some(item.id().to_owned());
                    }
                }
                st.desired = items;
            }
            RealtimeCommand::PushItem(p) => {
                let item = *p.item;
                let server = st.server().expect("open");
                // The output of a call the server does not hold answers nothing: not sent.
                if let ChatItem::FunctionCallOutput(o) = &item
                    && !server.holds_call(&o.call_id)
                {
                    return;
                }
                if !server.holds(item.id()) {
                    server.create(now, &item, None);
                    st.desired.push(item);
                }
            }
            RealtimeCommand::PushAudio(p) => {
                let frame = p.frame;
                if let Some(slot) = self.slot.get() {
                    let (pairs, _) = frame.data.as_chunks::<2>();
                    let tags = pairs.iter().map(|b| Tag::decode(i16::from_le_bytes(*b))).collect();
                    self.registry.record_forwarded(*slot, tags);
                }
                if frame.sample_rate == 0 {
                    return self.emit_error(
                        st,
                        ApiError::connection(Some("invalid audio sample rate".into())),
                        false,
                    );
                }
                st.remainder += Duration::from_secs_f64(
                    f64::from(frame.samples_per_channel) / f64::from(frame.sample_rate),
                );
                let batch = Duration::from_millis(100);
                while st.remainder >= batch {
                    st.remainder -= batch;
                    st.pushed += batch;
                    st.server().expect("open").append_audio(100);
                }
            }
            RealtimeCommand::CommitAudio(_) => {
                // "OpenAI requires at least 100ms of audio": less is not sent.
                if st.pushed > Duration::from_millis(100) {
                    st.pushed = Duration::ZERO;
                    st.server().expect("open").commit_audio(now);
                }
            }
            RealtimeCommand::ClearAudio(_) => {
                st.pushed = Duration::ZERO;
                st.remainder = Duration::ZERO;
                st.server().expect("open").clear_audio();
            }
            RealtimeCommand::GenerateReply(g) => {
                if st.asked.contains_key(&g.req_id) || st.discarded.contains(&g.req_id) {
                    return;
                }
                // The server replaces the session's instructions with a response's own.
                let instructions =
                    g.instructions.map(|extra| format!("{}\n{extra}", st.instructions));
                let deadline =
                    self.opts.read_timeout.map(|t| now.saturating_add(Duration::from_secs_f64(t)));
                let asked = Asked {
                    deadline,
                    instructions: instructions.clone(),
                    tool_choice: g.tool_choice.clone(),
                };
                st.asked.insert(g.req_id.clone(), asked);
                st.server().expect("open").respond_to(
                    now,
                    g.req_id,
                    instructions.as_deref(),
                    g.tool_choice,
                );
            }
            RealtimeCommand::Say(s) => {
                let error = RealtimeError::new(format!(
                    "{PROVIDER} does not implement say(). use a TTS model instead"
                ));
                st.events.push_back(Event::GenerateReplyFailed(GenerateReplyFailed {
                    req_id: s.req_id,
                    error,
                }));
            }
            RealtimeCommand::CancelReply(c) => {
                // Nothing is cancelled now: the reply is cancelled by id when it is created.
                if st.asked.remove(&c.req_id).is_some() {
                    st.discarded.insert(c.req_id);
                }
            }
            RealtimeCommand::Interrupt(i) => {
                if st.generation.is_some() {
                    st.server().expect("open").cancel(now, i.response_id.as_deref());
                }
            }
            RealtimeCommand::Truncate(t) => {
                // A message gone from the conversation, or held as text (a reconnect restored
                // it), has nothing to cut. One heard not at all is replaced by its text.
                let held = st.desired.iter().any(|i| i.id() == t.message_id);
                if !t.has_audio || !held || !st.spoken.contains(&t.message_id) {
                    return;
                }
                let server = st.server().expect("open");
                if t.audio_end_ms > 0 {
                    server.truncate(now, &t.message_id, t.audio_end_ms);
                } else {
                    server.as_text(&t.message_id);
                }
            }
        }
    }
}

/// One server-side event onto a registered socket. True when a real server could do it now.
pub(super) fn deliver(socket: &Socket, what: &Input) -> bool {
    let Input::RealtimeEvent { event, .. } = what else { return false };
    let now = socket.now();
    let accepted = {
        let mut state = socket.state.lock().unwrap();
        match event {
            RtEvent::Disconnected => {
                let open = state.server().is_some();
                if open {
                    socket.up.store(false, Ordering::SeqCst);
                    socket.hang_up(&mut state);
                }
                open
            }
            RtEvent::Reconnected => !socket.up.swap(true, Ordering::SeqCst),
            // A delta naming no response the session knows: it is dropped, nothing is told.
            RtEvent::StrayDelta => state.server().is_some(),
            other => {
                let exo = match other {
                    RtEvent::SpeechStarted => Exo::SpeechStarted,
                    RtEvent::SpeechStopped => Exo::SpeechStopped,
                    RtEvent::SecondGeneration => Exo::ServerResponse,
                    RtEvent::GenerationDone => Exo::FinishNow,
                    RtEvent::ServerError => Exo::ServerError,
                    RtEvent::FatalError => Exo::FatalError,
                    RtEvent::ResponseFailed => Exo::ResponseFailed,
                    RtEvent::TranscriptionFailed => Exo::TranscriptionFailed,
                    RtEvent::Disconnected | RtEvent::Reconnected | RtEvent::StrayDelta => {
                        unreachable!("answered above")
                    }
                };
                state.server().is_some_and(|server| server.exo(now, exo))
            }
        }
    };
    socket.wake.notify_one();
    accepted
}
