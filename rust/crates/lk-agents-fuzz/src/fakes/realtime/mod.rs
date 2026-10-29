//! The realtime model at the `Realtime` port the voice session consumes: a provider fake, its
//! session a thin pass between the driver and [`server::Server`], the realtime server as the
//! corpus shows the real one doing (`corpus/calibration.json`, `corpus/probes/` in
//! lk-agents-test-support).
//!
//! The driver's commands reach the server as it sends them, and what the server does reaches the
//! driver as the port's `Event`s: the response it was asked for or opened on its own, its message,
//! words and audio, its end and metrics; the items the server added; the person's speech and
//! transcript; errors. It carries no session logic: no connection retries, no reconnect restore,
//! no reply deadlines, no fallback. Inference's own tests hold the native session and the
//! fallback adapter (decisions 70, 113). What the provider promises at the port it keeps: a reply
//! cancelled before it was created is failed, never answered; a reply still waiting when the
//! session ends is failed; a connection slower than the client's bound times out; a connection
//! drop is a recoverable error, and the reconnect a server holding the conversation as the driver
//! gave it (as text), asked again for the replies still waiting; a connection that never comes
//! back within the client's retries ends the session; a fallback setup's fatal error is a swap to
//! a fresh model.
//!
//! The session is a resource (`Kind::Realtime`), one per session the driver opens, and every
//! response it told the driver of is one too (`Kind::Generation`), until its end. Audio deltas
//! are tagged with the generation and its chunk. What the server refused, or answered from the
//! wrong state, is recorded as a `Breach` on the session. What the server does of its own accord
//! -- hears the person start and stop (its VAD), drops the connection, errs -- is a
//! `RealtimeEvent` the case sends.
use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use lk_agents_abi::{
    ApiConnectOptions, BoxFuture, ChatContext, ChatItem, Clock, OwnedTask, Realtime,
    RealtimeCommand, RealtimeSession, Spawner, SpawnerExt, Ts,
    errors::{ApiError, ProviderError, TimeoutPhase},
    llm::realtime::{
        Event, GenerateReplyFailed, InputSpeechStopped, RealtimeCapabilities, RealtimeError,
        SessionReconnected,
    },
};

use super::{Control, Registry, SAMPLE_RATE, Slot};
use crate::{
    input::{Input, RtEvent},
    record::{Detail, Kind, Tag},
};

pub mod duplex;
mod events;
pub mod server;

pub use duplex::FakeDuplex;
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

/// How long a fallback adapter takes to close the failed model's session and open the next one.
const SWAP_MS: u64 = 50;

/// The samples an audio delta carries.
const DELTA_SAMPLES: usize = (server::DELTA_MS * SAMPLE_RATE as u64 / 1000) as usize;

/// How a realtime setup's server is configured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServerSetup {
    /// Server-side turn detection, and which; `None` is turn detection explicitly off.
    pub vad: Option<Vad>,
    /// What the server's delays and choices are drawn from.
    pub seed: u64,
    /// How long the session takes to connect.
    pub connect: Duration,
    /// How often a response calls a tool, in percent; `None` is the corpus's rate.
    pub tool_calls: Option<u8>,
    /// The model answers in text (`modalities=["text"]`).
    pub text_only: bool,
    /// A fallback adapter's model: a fatal error swaps to a fresh model rather than ending.
    pub fallback: bool,
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

    /// Running when it returns, as the real session is: it connects in the background.
    fn session(
        &self,
        disabled: bool,
    ) -> BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
        Box::pin(async move {
            let nth = self.sessions.fetch_add(1, Ordering::SeqCst);
            let vad = self.setup.vad.filter(|_| !disabled);
            let setup = ServerSetup { vad, seed: self.setup.seed ^ (nth << 32), ..self.setup };
            let caps = capabilities(vad, self.setup.vad.is_some(), self.setup.text_only);
            let now = self.clock.now();
            // A connection slower than the client's bound is reported as timed out.
            let bound = ApiConnectOptions::default().connect_timeout.map(Duration::from_secs_f64);
            let gives_up = bound
                .filter(|b| setup.connect >= *b)
                .map(|b| (now.saturating_add(b), ApiError::timed_out(TimeoutPhase::Connect)));
            let socket = Arc::new(Socket {
                setup,
                caps,
                clock: self.clock.clone(),
                registry: self.registry.clone(),
                slot: OnceLock::new(),
                state: Mutex::new(State::new(now.saturating_add(setup.connect), gives_up)),
                wake: tokio::sync::Notify::new(),
                ready: tokio::sync::Notify::new(),
            });
            let slot = self.registry.register(
                Kind::Realtime,
                Detail::None,
                Control::Realtime(socket.clone()),
            );
            let _ = socket.slot.set(slot);
            socket.state.lock().unwrap().server = socket.new_server(1);
            let pump = socket.clone();
            let pump = self.spawner.spawn(async move { pump.run().await });
            Ok(Arc::new(Session { socket, pump }) as Arc<dyn RealtimeSession>)
        })
    }
}

/// The session the driver holds.
struct Session {
    socket: Arc<Socket>,
    /// Delivers the server's timed happenings; cancelled at the close or with the session.
    pump: OwnedTask,
}

impl RealtimeSession for Session {
    fn capabilities(&self) -> RealtimeCapabilities {
        self.socket.caps
    }

    /// Never blocks: queued for the pump, which sends it once the connection is up.
    fn send(&self, cmd: RealtimeCommand) {
        self.socket.state.lock().unwrap().commands.push_back(cmd);
        self.socket.wake.notify_one();
    }

    fn next_event(&self) -> BoxFuture<'_, Option<Event>> {
        Box::pin(self.socket.until(|st| match st.events.pop_front() {
            Some(event) => Some(Some(event)),
            None => st.over.then_some(None),
        }))
    }

    /// The conversation the server is being brought to.
    fn chat_ctx(&self) -> BoxFuture<'_, ChatContext> {
        Box::pin(async move { ChatContext::new(self.socket.state.lock().unwrap().desired.clone()) })
    }

    fn aclose(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            self.pump.cancel();
            {
                let mut st = self.socket.state.lock().unwrap();
                if !st.over {
                    self.socket.hang_up(&mut st);
                    st.over = true;
                }
            }
            self.socket.ready.notify_waiters();
            if let Some(slot) = self.socket.slot.get() {
                self.socket.registry.mark_closed(*slot);
            }
        })
    }
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

struct State {
    server: Server,
    /// Connecting until then: commands wait.
    opens_at: Option<Ts>,
    /// The connection dropped; commands wait for its return.
    down: bool,
    /// A fallback setup's model failed: the next opening is the swap to a fresh one.
    swapping: bool,
    /// When the client gives up connecting, or reconnecting, and the error it reports.
    gives_up: Option<(Ts, ApiError)>,
    /// The session ended: no event comes after the queued ones.
    over: bool,
    /// Commands waiting for the connection.
    commands: VecDeque<RealtimeCommand>,
    events: VecDeque<Event>,
    /// The conversation the server is being brought to.
    desired: Vec<ChatItem>,
    instructions: String,
    tools: Vec<String>,
    /// Replies asked and not yet created; those cancelled before they were.
    asked: BTreeSet<String>,
    cancelled: BTreeSet<String>,
    generation: Option<Generation>,
    /// When the server heard each turn start, as wall seconds, and each turn's transcript so far.
    speech_started: HashMap<String, f64>,
    transcripts: HashMap<String, String>,
    /// Servers connected: the first, then reconnects and a fallback setup's swaps.
    connections: u64,
    /// Responses the run registered, by index on the current server.
    generations: HashMap<u32, Slot>,
}

impl State {
    fn new(opens_at: Ts, gives_up: Option<(Ts, ApiError)>) -> Self {
        State {
            server: Server::new(0, String::new()),
            opens_at: Some(opens_at),
            down: false,
            swapping: false,
            gives_up,
            over: false,
            commands: VecDeque::new(),
            events: VecDeque::new(),
            desired: Vec::new(),
            instructions: String::new(),
            tools: Vec::new(),
            asked: BTreeSet::new(),
            cancelled: BTreeSet::new(),
            generation: None,
            speech_started: HashMap::new(),
            transcripts: HashMap::new(),
            connections: 1,
            generations: HashMap::new(),
        }
    }

    /// The response the driver was told of ends unfinished: its messages close, it is cancelled.
    fn close_generation(&mut self, audio: bool) {
        let Some(g) = self.generation.take() else { return };
        for (id, open, settled) in &g.messages {
            if !settled {
                self.events.push_back(events::modalities(id, &g.id, audio));
            }
            if *open {
                self.events.push_back(events::message_done(id, &g.id));
            }
        }
        self.events.push_back(events::cancelled(g.id));
    }
}

/// One session: its state and the server behind it.
pub struct Socket {
    setup: ServerSetup,
    caps: RealtimeCapabilities,
    clock: Arc<dyn Clock>,
    registry: Arc<Registry>,
    slot: OnceLock<Slot>,
    state: Mutex<State>,
    /// Told whenever the pump may have something new to do.
    wake: tokio::sync::Notify,
    /// Told whenever an event is queued or the session ended.
    ready: tokio::sync::Notify,
}

impl Socket {
    fn now(&self) -> Ts {
        self.clock.now()
    }

    fn wall_secs(&self) -> f64 {
        let wall = self.clock.wall();
        wall.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64()
    }

    /// How long the client keeps reconnecting: an attempt per retry, each taking the connection
    /// time, with the retry intervals between them.
    fn retry_budget(&self) -> Duration {
        let opts = ApiConnectOptions::default();
        let intervals: f64 =
            (0..opts.max_retry.saturating_sub(1)).map(|i| opts.interval_for_retry(i)).sum();
        Duration::from_secs_f64(intervals) + self.setup.connect * opts.max_retry
    }

    fn new_server(&self, connection: u64) -> Server {
        let ids = format!("{}s{connection}c", self.slot.get().map_or(0, |s| s.nth));
        let mut server = Server::new(self.setup.seed ^ connection, ids);
        server.configure(self.setup.vad, !self.setup.text_only, self.setup.tool_calls);
        server
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
            if st.over {
                return None;
            }
            let now = self.now();
            let gives_up_at = st.gives_up.as_ref().map(|(at, _)| *at);
            if gives_up_at.is_some_and(|at| at <= now) {
                let (_, error) = st.gives_up.take().expect("due");
                self.emit_error(st, error, self.setup.fallback);
                self.fatal(st);
                continue;
            }
            if let Some(at) = st.opens_at {
                if at > now {
                    return Some(Some(gives_up_at.map_or(at, |g| g.min(at))));
                }
                st.opens_at = None;
                if std::mem::take(&mut st.swapping) {
                    st.connections += 1;
                    st.server = self.new_server(st.connections);
                    st.desired.clear();
                    st.commands.clear();
                    st.events.push_back(Event::SessionSwapped { regenerate: true });
                } else {
                    st.events.push_back(self.connection_metrics(self.setup.connect));
                }
            }
            if st.down {
                return Some(gives_up_at);
            }
            // What the server did before what the driver sends now.
            let due = st.server.next_due();
            if due.is_some_and(|at| at <= now) {
                if let Some(happening) = st.server.pop(now) {
                    self.told(st, now, happening);
                }
            } else if let Some(cmd) = st.commands.pop_front() {
                self.command(st, now, cmd);
            } else {
                return Some(due);
            }
            self.record_breaches(st);
        }
    }

    fn record_breaches(&self, st: &mut State) {
        if let Some(slot) = self.slot.get() {
            for breach in st.server.take_breaches() {
                self.registry.record_breach(slot.nth, breach);
            }
        }
    }

    /// The server is left: what it held is recorded, and its responses end with it.
    fn hang_up(&self, st: &mut State) {
        if let Some(slot) = self.slot.get() {
            self.registry.record_held(slot.nth, st.server.held());
        }
        for (_, slot) in st.generations.drain() {
            self.registry.mark_closed(slot);
        }
    }

    /// A fatal error: a fallback setup swaps to a fresh model, which the driver gives its frame;
    /// otherwise the session ends, failing the replies still waiting.
    fn fatal(&self, st: &mut State) {
        let audio = !self.setup.text_only;
        self.hang_up(st);
        st.close_generation(audio);
        if self.setup.fallback {
            // The error is told first; the healthy model is up once the old session closed and
            // the new one opened, and what the old one was sent meanwhile is dropped.
            let opens_at = server::later(self.now(), SWAP_MS);
            (st.opens_at, st.down, st.gives_up, st.swapping) = (Some(opens_at), false, None, true);
            st.asked.clear();
            st.cancelled.clear();
            return;
        }
        for req_id in std::mem::take(&mut st.asked) {
            let error = RealtimeError::new("realtime session closed");
            st.events.push_back(Event::GenerateReplyFailed(GenerateReplyFailed { req_id, error }));
        }
        st.over = true;
    }

    // --- the driver's commands -------------------------------------------------------------

    fn command(&self, st: &mut State, now: Ts, cmd: RealtimeCommand) {
        match cmd {
            RealtimeCommand::UpdateInstructions(u) => {
                st.server.instructions(&u.instructions);
                st.instructions = u.instructions;
            }
            RealtimeCommand::UpdateTools(u) => {
                st.tools = u.tools.iter().map(|t| t.name.clone()).collect();
                st.server.tools(st.tools.clone());
            }
            RealtimeCommand::UpdateChatCtx(u) => {
                let items = u.chat_ctx.items;
                let server = &mut st.server;
                // What the conversation loses goes; what it gains is created in its place; a
                // message the driver records as cut is held as its text.
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
                // The output of a call the server does not hold answers nothing.
                if let ChatItem::FunctionCallOutput(o) = &item
                    && !st.server.holds_call(&o.call_id)
                {
                    return;
                }
                if !st.server.holds(item.id()) {
                    st.server.create(now, &item, None);
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
                if frame.sample_rate > 0 {
                    let ms = u64::from(frame.samples_per_channel) * 1000
                        / u64::from(frame.sample_rate);
                    st.server.append_audio(ms);
                }
            }
            // A commit of less than the provider's 100 ms minimum is not made.
            RealtimeCommand::CommitAudio(_) if st.server.buffered_ms() < 100 => {}
            RealtimeCommand::CommitAudio(_) => st.server.commit_audio(now),
            RealtimeCommand::ClearAudio(_) => st.server.clear_audio(),
            RealtimeCommand::GenerateReply(g) => {
                st.asked.insert(g.req_id.clone());
                st.server.respond_to(now, g.req_id, g.instructions.as_deref(), g.tool_choice);
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
            // The reply is cancelled by id when it is created.
            RealtimeCommand::CancelReply(c) => {
                if st.asked.remove(&c.req_id) {
                    st.cancelled.insert(c.req_id);
                }
            }
            RealtimeCommand::Interrupt(i) => {
                if st.generation.is_some() {
                    st.server.cancel(now, i.response_id.as_deref());
                }
            }
            RealtimeCommand::Truncate(t) => {
                // A message the server does not hold, or holds no audio of (a reconnect restored
                // it as text), has nothing to cut.
                if !t.has_audio || st.server.audio_ms(&t.message_id).is_none_or(|ms| ms == 0) {
                    return;
                }
                if t.audio_end_ms > 0 {
                    st.server.truncate(now, &t.message_id, t.audio_end_ms);
                } else {
                    st.server.as_text(&t.message_id);
                }
            }
        }
    }
}

/// One server-side event onto a registered session. True when a real server could do it now.
pub(super) fn deliver(socket: &Socket, what: &Input) -> bool {
    let Input::RealtimeEvent { event, .. } = what else { return false };
    let now = socket.now();
    let accepted = {
        let mut guard = socket.state.lock().unwrap();
        let st = &mut *guard;
        let connected = !st.over && st.opens_at.is_none();
        match event {
            RtEvent::Disconnected if connected && !st.down => {
                // The connection's loss: its server, the response in flight and the person's turn
                // go with it, and the session says it will reconnect.
                st.down = true;
                if st.server.speaking() {
                    st.events.push_back(Event::InputSpeechStopped(InputSpeechStopped {
                        user_transcription_enabled: true,
                    }));
                }
                socket.hang_up(st);
                st.server = Server::new(0, String::new());
                st.close_generation(!socket.setup.text_only);
                let message = format!("{PROVIDER} connection closed unexpectedly");
                socket.emit_error(st, ApiError::connection(Some(message)), true);
                // A server that never takes the connection again outlasts the client's retries.
                let unavailable = ApiError::from_http("Service Unavailable", 503, None);
                st.gives_up = Some((now.saturating_add(socket.retry_budget()), unavailable));
                true
            }
            RtEvent::Reconnected if st.down => {
                // A fresh server holding the conversation as the driver last gave it, as text;
                // the replies still waiting are asked of it again.
                (st.down, st.gives_up) = (false, None);
                st.connections += 1;
                st.server = socket.new_server(st.connections);
                let mut previous = None;
                for item in &st.desired {
                    st.server.create(now, item, previous.replace(item.id().to_owned()));
                }
                for req_id in st.asked.clone() {
                    st.server.respond_to(now, req_id, None, None);
                }
                st.events.push_back(Event::SessionReconnected(SessionReconnected {}));
                true
            }
            RtEvent::Disconnected | RtEvent::Reconnected => false,
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
                    RtEvent::Disconnected | RtEvent::Reconnected => unreachable!("answered above"),
                };
                connected && !st.down && st.server.exo(now, exo)
            }
        }
    };
    socket.wake.notify_one();
    socket.ready.notify_waiters();
    accepted
}
