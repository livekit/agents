use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use lk_agents_abi::{
    ApiConnectOptions, BoxFuture, ChatContext, Clock, OwnedTask, Realtime, RealtimeCommand,
    RealtimeSession, Spawner, SpawnerExt, Ts,
    errors::{ApiError, ProviderError, Timeout},
    llm::realtime::{Event, RealtimeCapabilities},
};

use super::{Control, Registry, SAMPLE_RATE, Slot};
use crate::{
    input::{Input, RtEvent},
    record::{Detail, Kind},
};

// The production sans-I/O machine owns client bookkeeping. This adapter executes its typed
// effects against an independent simulated provider, without serialization or socket I/O.

use lk_agents_abi::llm::realtime_options::{ServerVad, TurnDetection};
use lk_agents_inference::realtime::{Dialect, RealtimeMachine, SessionOptions};

#[cfg(test)]
mod client_tests;
pub mod duplex;
mod events;
mod outgoing;
pub mod server;

pub use duplex::FakeDuplex;
pub use server::{Exo, Server, Vad};

/// The model a session names in its metrics.
pub const MODEL: &str = "openai/gpt-realtime";

/// `RealtimeModel.label` of a LiveKit Inference model: on every error and metric.
const LABEL: &str = "lk_agents_realtime.LiveKit Inference Realtime";

const PROVIDER: &str = "LiveKit Inference Realtime";

/// How long before the session the model was built: its errors are stamped on the model's own
/// clock (`RealClock` from its construction).
const MODEL_BUILT_BEFORE: Duration = Duration::from_secs(3_600);

/// How long a fallback adapter takes to close the failed model's session and open the next one.
const SWAP_MS: u64 = 50;

const DELTA_SAMPLES: usize = (server::DELTA_MS * SAMPLE_RATE as u64 / 1000) as usize;

/// Delays a case sets rather than drawing them from the model's seed, in ms.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RealtimeDelays {
    /// The server creates a reply this long after its `response.create`.
    pub created: Option<u64>,
    /// Delay only the first response acknowledgement, so later requests can recover normally.
    pub first_created: Option<u64>,
    /// The server ends a reply (`response.done`) this long after its last audio.
    pub done: Option<u64>,
    /// Delay between an applied conversation write and its acknowledgement.
    pub context_ack: Option<u64>,
    /// A user turn's transcription completion, independent of response generation.
    pub transcription: Option<u64>,
    /// Override the first turn only, allowing a later turn's final to arrive first.
    pub first_transcription: Option<u64>,
    /// First partial transcript, independent of its final; subsequent deltas follow in order.
    pub transcription_partial: Option<u64>,
    /// A duplex model's utterance of more than two words has its last words' text arrive this
    /// long after their sound.
    pub late_words: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServerSetup {
    pub provider: lk_agents_abi::llm::realtime::RealtimeProvider,
    /// Server-side turn detection; `None` is turn detection explicitly off.
    pub vad: Option<Vad>,
    pub seed: u64,
    pub connect: Duration,
    /// How often a response calls a tool, in percent; `None` is the measured rate.
    pub tool_calls: Option<u8>,
    /// `modalities=["text"]`.
    pub text_only: bool,
    /// A fallback adapter's model: a fatal error swaps to a fresh model rather than ending.
    pub fallback: bool,
    pub timing: crate::fakes::Timing,
    pub delays: RealtimeDelays,
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

    fn machine(&self, vad: Option<Vad>, caps: RealtimeCapabilities, seed: u64) -> RealtimeMachine {
        let setup = ServerSetup { vad, ..self.setup.clone() };
        client_machine(&setup, caps, self.clock.as_ref(), format!("dst{seed}_"))
    }
}

fn client_machine(
    setup: &ServerSetup,
    caps: RealtimeCapabilities,
    clock: &dyn Clock,
    id_prefix: String,
) -> RealtimeMachine {
    let dialect = Dialect::of(setup.provider);
    let mut opts = SessionOptions::from_dialect(&dialect, setup.vad.is_none());
    if matches!(setup.vad, Some(Vad::Server)) {
        opts.turn_detection = Some(TurnDetection::Server(ServerVad {
            threshold: Some(0.5),
            prefix_padding_ms: Some(300),
            silence_duration_ms: Some(500),
            idle_timeout_ms: None,
            create_response: Some(true),
            interrupt_response: Some(true),
        }));
    }
    opts.model = MODEL.into();
    opts.audio_output = !setup.text_only;
    RealtimeMachine::new(dialect, opts, caps, id_prefix)
        .with_response_timeout(
            ApiConnectOptions::default().read_timeout.map(Duration::from_secs_f64),
        )
        .with_label(LABEL.into())
        .with_provider("livekit".into())
        .with_wall_anchor(clock.wall() - Duration::from_nanos(clock.now().as_nanos()))
}

/// A LiveKit Inference realtime model's capabilities: server detection left at its default is the
/// driver's to switch off; switched off by the developer, it stays off.
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
        supports_overlapping_speech: false,
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
        _: lk_agents_abi::telemetry::Cx,
        disabled: bool,
        _: Option<lk_agents_abi::HostRef>,
    ) -> BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
        Box::pin(async move {
            let nth = self.sessions.fetch_add(1, Ordering::SeqCst);
            let vad = self.setup.vad.filter(|_| !disabled);
            let seed = self.setup.seed ^ (nth << 32);
            let setup = ServerSetup { vad, seed, ..self.setup.clone() };
            let caps = capabilities(vad, self.setup.vad.is_some(), self.setup.text_only);
            let now = self.clock.now();
            // A connection slower than the client's bound is reported as timed out.
            let bound = ApiConnectOptions::default().connect_timeout.map(Duration::from_secs_f64);
            let gives_up = bound
                .filter(|b| setup.connect >= *b)
                .map(|b| (now.saturating_add(b), ApiError::timed_out(Timeout::Connect)));
            let opened = now.saturating_add(setup.connect);
            let socket = Arc::new(Socket {
                setup,
                caps,
                clock: self.clock.clone(),
                registry: self.registry.clone(),
                slot: OnceLock::new(),
                state: Mutex::new(State::new(opened, gives_up, self.machine(vad, caps, seed))),
                wake: tokio::sync::Notify::new(),
                ready: tokio::sync::Notify::new(),
            });
            let slot = self.registry.register(
                Kind::Realtime,
                Detail::None,
                Control::Realtime(socket.clone()),
            );
            assert!(socket.slot.set(slot).is_ok(), "realtime socket registered twice");
            socket.state.lock().unwrap().server = socket.new_server(1);
            let pump = socket.clone();
            let pump = self.spawner.spawn(async move { pump.run().await });
            Ok(Arc::new(Session { socket, pump }) as Arc<dyn RealtimeSession>)
        })
    }
}

struct Session {
    socket: Arc<Socket>,
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

    fn head(&self, frame: lk_agents_abi::AudioFrame) {
        self.send(RealtimeCommand::PushAudio(lk_agents_abi::PushAudio { frame }));
    }

    fn next_event(&self) -> BoxFuture<'_, Option<Event>> {
        Box::pin(self.socket.until(|st| match st.events.pop_front() {
            Some(event) => Some(Some(event)),
            None => st.over.then_some(None),
        }))
    }

    fn chat_ctx(&self) -> BoxFuture<'_, ChatContext> {
        Box::pin(async move { self.socket.state.lock().unwrap().machine.chat_ctx() })
    }

    fn aclose(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            self.pump.cancel();
            {
                let mut st = self.socket.state.lock().unwrap();
                if !st.over {
                    self.socket.hang_up(&mut st);
                    let step = st.machine.close(self.socket.now());
                    st.events.extend(step.events);
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

/// The response the driver was told of, and per message whether it is open and has settled its
/// modalities.
struct State {
    server: Server,
    machine: RealtimeMachine,
    opens_at: Option<Ts>,
    down: bool,
    swapping: bool,
    gives_up: Option<(Ts, ApiError)>,
    over: bool,
    commands: VecDeque<RealtimeCommand>,
    events: VecDeque<Event>,
    connections: u64,
    /// Observation identities only; response lifecycle belongs to the production machine.
    generations: HashMap<u32, Slot>,
}

impl State {
    fn new(opens_at: Ts, gives_up: Option<(Ts, ApiError)>, machine: RealtimeMachine) -> Self {
        Self {
            server: Server::new(0, String::new()),
            machine,
            opens_at: Some(opens_at),
            down: false,
            swapping: false,
            gives_up,
            over: false,
            commands: VecDeque::new(),
            events: VecDeque::new(),
            connections: 1,
            generations: HashMap::new(),
        }
    }
}

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

    /// When the server next does something on its own clock (a cancelled response's end, the
    /// rest of a reply), if it is still serving.
    pub(crate) fn next_due(&self) -> Option<Ts> {
        let st = self.state.lock().unwrap();
        if st.over || st.down {
            None
        } else {
            st.server.next_due().into_iter().chain(st.machine.next_deadline()).min()
        }
    }

    /// How long the client keeps reconnecting: each retry's connect time plus the intervals.
    fn retry_budget(&self) -> Duration {
        let opts = ApiConnectOptions::default();
        let intervals: f64 =
            (0..opts.max_retry.saturating_sub(1)).map(|i| opts.interval_for_retry(i)).sum();
        Duration::from_secs_f64(intervals) + self.setup.connect * opts.max_retry
    }

    fn new_server(&self, connection: u64) -> Server {
        let ids = format!("{}s{connection}c", self.slot.get().map_or(0, |s| s.nth));
        let mut server = Server::new(self.setup.seed ^ connection, ids);
        server.provider = self.setup.provider;
        server.configure(
            self.setup.vad,
            !self.setup.text_only,
            self.setup.tool_calls,
            self.setup.timing.clone(),
            self.setup.delays,
        );
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

    /// Runs everything due now; returns when it runs next (`None`: it is over).
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
                    st.commands.clear();
                    st.events.push_back(Event::SessionSwapped { regenerate: true });
                } else {
                    st.events.push_back(st.machine.connection_metrics(now, self.setup.connect));
                }
                let step = st.machine.open(now);
                self.settle(st, now, step);
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
                let step = st.machine.command(now, cmd);
                self.settle(st, now, step);
            } else if st.machine.next_deadline().is_some_and(|at| at <= now) {
                let step = st.machine.tick(now);
                self.settle(st, now, step);
            } else if st.machine.context_continuation_ready() {
                let step = st.machine.continue_context(now);
                self.settle(st, now, step);
            } else {
                return Some(due.into_iter().chain(st.machine.next_deadline()).min());
            }
            // Retirement changes ownership only, not protocol decisions.
            while let Some(retired) = st.machine.take_context_retirement() {
                drop(retired);
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

    /// Records what the server held; its responses end with it.
    fn hang_up(&self, st: &mut State) {
        if let Some(slot) = self.slot.get() {
            self.registry.record_held(slot.nth, st.server.held());
        }
        for (_, slot) in st.generations.drain() {
            self.registry.mark_closed(slot);
        }
    }

    /// A fallback setup swaps to a fresh model; otherwise the session ends, failing the replies
    /// still waiting.
    fn fatal(&self, st: &mut State) {
        self.hang_up(st);
        let step = st.machine.close(self.now());
        st.events.extend(step.events);
        if self.setup.fallback {
            // What the old model was sent during the swap is dropped.
            let opens_at = server::later(self.now(), SWAP_MS);
            (st.opens_at, st.down, st.gives_up, st.swapping) = (Some(opens_at), false, None, true);
            st.machine = client_machine(
                &self.setup,
                self.caps,
                self.clock.as_ref(),
                format!("swap{}_", st.connections + 1),
            );
            return;
        }

        st.over = true;
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
                // The server, the response in flight and the person's turn go with the connection.
                st.down = true;
                socket.hang_up(st);
                st.server = Server::new(0, String::new());
                let message = format!("{PROVIDER} connection closed unexpectedly");
                socket.emit_error(st, ApiError::connection(Some(message)), true);
                // A server that never takes the connection again outlasts the client's retries.
                let unavailable = ApiError::from_http("Service Unavailable", 503, None);
                st.gives_up = Some((now.saturating_add(socket.retry_budget()), unavailable));
                true
            }
            RtEvent::Reconnected if st.down => {
                // A fresh server holds the last given conversation as text; waiting replies are
                // asked again.
                (st.down, st.gives_up) = (false, None);
                st.connections += 1;
                st.server = socket.new_server(st.connections);
                let step = st.machine.reconnect(now);
                socket.settle(st, now, step);
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
                    RtEvent::OldestTranscriptionFailed => Exo::OldestTranscriptionFailed,
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
