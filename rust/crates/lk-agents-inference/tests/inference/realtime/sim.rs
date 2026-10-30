//! The realtime core over the simulated server, on virtual time: a LiveKit Inference realtime
//! session's conversation machine and connection policy (`RealtimeMachine`, `RealtimeConnection`)
//! stepped in the order the native session's pump steps them, against [`Server`] as the real
//! one was measured. The socket is the server's typed records; the person's audio reaches it in
//! the writer's 100 ms batches; the server answers on its own clock.
//!
//! A test sends commands and server-side happenings ([`Exo`], a dropped socket) at instants of
//! its choosing and reads what the driver would have been told ([`Sim::events`]) and what the
//! server refused or answered from the wrong state ([`Sim::breaches`]). What the machine gets
//! wrong about the server shows here; what the driver does with its events is the DST's.
use std::{
    collections::{BTreeSet, VecDeque},
    time::{Duration, SystemTime},
};

use lk_agents_abi::{
    ApiConnectOptions, AudioFrame, PushAudio, Ts,
    errors::ApiError,
    llm::realtime::{Event, RealtimeModelError},
};
use lk_agents_inference::realtime::{
    ConnectionEffect, INFERENCE, Outgoing, RealtimeConnection, RealtimeMachine, SessionOptions,
    Step, TurnDetection,
};
use lk_agents_media::AudioByteStream;
use lk_agents_net::transport::{TransportError, connect_error};

pub use lk_agents_abi::RealtimeCommand as Command;

use super::server::{Breach, Exo, Hooks, SAMPLE_RATE, Server, Vad};

/// The wall clock the session's metrics and items are stamped against: a call in 2026.
const WALL: Duration = Duration::from_secs(1_793_000_000);

/// `RealtimeModel.label` of a LiveKit Inference model: on every error and metric.
const LABEL: &str = "lk_agents_realtime.LiveKit Inference Realtime";

/// How a session's server is configured.
#[derive(Clone, Copy, Debug)]
pub struct Setup {
    /// Server-side turn detection, and which; `None` is turn detection off.
    pub vad: Option<Vad>,
    /// What the server's delays and choices are drawn from.
    pub seed: u64,
    /// How long a connection takes to open.
    pub connect: Duration,
    /// How often a response calls a tool, in percent; `None` is the measured rate (8 in 94).
    pub tool_calls: Option<u8>,
}

impl Default for Setup {
    fn default() -> Self {
        Setup { vad: None, seed: 0, connect: Duration::ZERO, tool_calls: None }
    }
}

/// Where the session's connection stands.
enum Link {
    /// Between attempts: on the connection policy's next deadline, or its next effect.
    Waiting,
    /// An attempt in flight, opening at `opens_at`.
    Connecting { generation: u64, started: Ts, opens_at: Ts },
    /// A connection, and the server behind it until the socket drops (`None`).
    Open { generation: u64, server: Option<Box<Server>> },
    /// The pump ended: no event comes after the ones already told.
    Over,
}

/// What one connection ended in, as the native session's `connect_and_run` reports it.
enum Outcome {
    Closed,
    Fatal(RealtimeModelError),
    Effects(Vec<ConnectionEffect>),
    Recycle,
    Failed(ApiError),
}

pub struct Sim {
    setup: Setup,
    now: Ts,
    machine: RealtimeMachine,
    connection: RealtimeConnection,
    link: Link,
    effects: VecDeque<ConnectionEffect>,
    /// A connection was attempted before: the next one replays the conversation.
    reconnecting: bool,
    commands: VecDeque<Command>,
    closing: bool,
    /// The server takes connections; a drop sets it down until it is restored.
    up: bool,
    audio: AudioByteStream,
    connections: u64,
    events: Vec<(Ts, Event)>,
    record: Record,
}

/// What the server's deliveries and refusals tell the test.
#[derive(Default)]
struct Record {
    breaches: Vec<Breach>,
    responses: u32,
    /// The responses created and not done yet.
    live: BTreeSet<u32>,
    /// A response was created before the previous one was done.
    overlapped: bool,
}

impl Hooks for Record {
    fn created(&mut self, _: &str) -> u32 {
        self.responses += 1;
        self.overlapped |= !self.live.is_empty();
        self.live.insert(self.responses);
        self.responses
    }
    fn done(&mut self, source: u32) {
        self.live.remove(&source);
    }
    fn produced(&mut self, _: u32, _: u16) {}
    fn heard(&mut self, _: &[u8]) {}
    fn breach(&mut self, breach: Breach) {
        self.breaches.push(breach);
    }
}

impl Sim {
    /// A session as a LiveKit Inference realtime model opens one, connecting at once.
    pub fn new(setup: Setup) -> Sim {
        let mut opts = SessionOptions::from_dialect(&INFERENCE, false);
        opts.turn_detection = match setup.vad {
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
        let caps = INFERENCE.capabilities(
            opts.turn_detection.as_ref(),
            setup.vad.is_none(),
            opts.audio_output,
            opts.transcription.is_some(),
        );
        let conn = ApiConnectOptions::default();
        let now = Ts::ZERO;
        let machine = RealtimeMachine::new(INFERENCE, opts, caps, "1".into())
            .with_response_timeout(conn.read_timeout.map(Duration::from_secs_f64))
            .with_label(LABEL.into())
            .with_provider("livekit".into())
            .with_wall_anchor(SystemTime::UNIX_EPOCH + WALL);
        let mut connection = RealtimeConnection::new(&conn, now);
        let effects = connection.start(now).into();
        let mut sim = Sim {
            setup,
            now,
            machine,
            connection,
            link: Link::Waiting,
            effects,
            reconnecting: false,
            commands: VecDeque::new(),
            closing: false,
            up: true,
            audio: batches(),
            connections: 0,
            events: Vec::new(),
            record: Record::default(),
        };
        sim.step();
        sim
    }

    /// A command, as the driver sends it now.
    pub fn send(&mut self, cmd: Command) {
        self.commands.push_back(cmd);
        self.step();
    }

    /// The person's audio, `ms` of it in 10 ms frames.
    pub fn speak(&mut self, ms: u64) {
        let samples = SAMPLE_RATE / 100;
        for _ in 0..ms / 10 {
            let frame = AudioFrame::new(vec![1; samples as usize * 2], SAMPLE_RATE, 1, samples);
            self.send(Command::PushAudio(PushAudio { frame }));
        }
    }

    /// What the server does of its own accord now; false when a real server could not.
    pub fn exo(&mut self, exo: Exo) -> bool {
        let now = self.now;
        let done = match &mut self.link {
            Link::Open { server: Some(server), .. } => server.exo(now, exo),
            _ => false,
        };
        self.step();
        done
    }

    /// The server drops the socket and refuses connections until [`Self::restore`].
    pub fn drop_socket(&mut self) {
        if let Link::Open { server, .. } = &mut self.link
            && server.is_some()
        {
            *server = None;
            self.up = false;
            // Its responses end with it.
            self.record.live.clear();
        }
        self.step();
    }

    pub fn restore(&mut self) {
        self.up = true;
        self.step();
    }

    /// `ms` of virtual time, the pump doing everything that comes due.
    pub fn advance(&mut self, ms: u64) {
        let until = self.now.saturating_add(Duration::from_millis(ms));
        loop {
            match self.step() {
                Some(at) if at <= until => self.now = at,
                _ => break,
            }
        }
        self.now = until;
        self.step();
    }

    pub fn events(&self) -> &[(Ts, Event)] {
        &self.events
    }

    /// What the server refused of the client, or answered from the wrong state.
    pub fn breaches(&self) -> &[Breach] {
        &self.record.breaches
    }

    /// The server created a response before the previous one's `response.done`.
    pub fn overlapped(&self) -> bool {
        self.record.overlapped
    }

    /// The refusals among them.
    pub fn refused(&self) -> Vec<&Breach> {
        self.breaches().iter().filter(|b| matches!(b, Breach::Refused { .. })).collect()
    }

    /// Everything the pump can do now; then when it runs next.
    fn step(&mut self) -> Option<Ts> {
        let now = self.now;
        loop {
            let policy = self.connection.next_deadline();
            match self.link {
                Link::Over => return None,
                Link::Waiting => {
                    if let Some(effect) = self.effects.pop_front() {
                        self.effect(effect);
                    } else if self.closing {
                        let _ = self.connection.close(now);
                        self.end(true);
                    } else {
                        match policy {
                            None => self.end(false),
                            Some(at) if at <= now => {
                                let effects = self.connection.tick(now);
                                self.effects.extend(effects);
                            }
                            Some(at) => return Some(at),
                        }
                    }
                }
                Link::Connecting { generation, started, opens_at } => {
                    if self.closing {
                        self.ended(generation, Outcome::Closed);
                    } else if policy.is_some_and(|at| at <= now) {
                        let effects = self.connection.tick(now);
                        if ends_attempt(&effects) {
                            self.ended(generation, Outcome::Effects(effects));
                        }
                    } else if now >= opens_at {
                        self.opened(generation, started);
                    } else {
                        return Some(policy.map_or(opens_at, |at| at.min(opens_at)));
                    }
                }
                Link::Open { generation, ref server } => {
                    let lost = server.is_none();
                    let server_due = server.as_deref().and_then(Server::next_due);
                    let deadline = self.machine.next_deadline().into_iter().chain(policy).min();
                    // The pump's biased order: the deadline, the close, the socket, the commands,
                    // the context work.
                    if deadline.is_some_and(|at| at <= now) {
                        let effects = self.connection.tick(now);
                        if ends_attempt(&effects) {
                            self.ended(generation, Outcome::Effects(effects));
                        } else {
                            let step = self.machine.tick(now);
                            self.stepped(generation, step);
                        }
                    } else if self.closing {
                        self.ended(generation, Outcome::Closed);
                    } else if lost {
                        self.ended(generation, Outcome::Failed(lost_connection()));
                    } else if server_due.is_some_and(|at| at <= now) {
                        let wall =
                            SystemTime::UNIX_EPOCH + WALL + Duration::from_nanos(now.as_nanos());
                        let Link::Open { server: Some(server), .. } = &mut self.link else {
                            unreachable!()
                        };
                        let step = server.pop(now, wall, &mut self.record).map(|event| {
                            let at = self.machine.connection_generation();
                            self.machine.incoming_on(at, now, event)
                        });
                        if let Some(step) = step {
                            self.stepped(generation, step);
                        }
                    } else if let Some(cmd) = self.commands.pop_front() {
                        let step = self.machine.command(now, cmd);
                        self.stepped(generation, step);
                    } else if let Some(retired) = self.machine.take_context_retirement() {
                        drop(retired);
                    } else if self.machine.context_continuation_ready() {
                        let step = self.machine.continue_context(now);
                        self.stepped(generation, step);
                    } else {
                        return deadline.into_iter().chain(server_due).min();
                    }
                }
            }
        }
    }

    /// One connection effect, as the pump acts on it.
    fn effect(&mut self, effect: ConnectionEffect) {
        match effect {
            ConnectionEffect::Connect { generation } => {
                let opens_at = self.now.saturating_add(self.setup.connect);
                self.link = Link::Connecting { generation, started: self.now, opens_at };
            }
            ConnectionEffect::Retry { error } => self.emit_error(error, true),
            ConnectionEffect::Failed { error } => {
                self.emit_error(error, false);
                self.end(true);
            }
            ConnectionEffect::Close { .. } | ConnectionEffect::CleanupIncomplete { .. } => {}
        }
    }

    /// The attempt's socket opens now, or the server is down.
    fn opened(&mut self, generation: u64, started: Ts) {
        if !self.up {
            let refused = TransportError::Status {
                status: 503,
                message: "Service Unavailable".into(),
                headers: Vec::new(),
                body: None,
            };
            let error = connect_error(INFERENCE.provider_label, refused);
            return self.ended(generation, Outcome::Failed(error));
        }
        let now = self.now;
        self.connections += 1;
        let mut server = Server::new(self.setup.seed ^ self.connections)
            .with_tool_calls(self.setup.tool_calls)
            .with_ids(format!("{}c", self.connections));
        server.connect(now);
        self.link = Link::Open { generation, server: Some(Box::new(server)) };
        self.audio = batches();
        let metrics = self.machine.connection_metrics(now, now.saturating_duration_since(started));
        self.events.push((now, metrics));
        let step =
            if self.reconnecting { self.machine.reconnect(now) } else { self.machine.open(now) };
        if !self.connection.connected(generation, now) {
            return self.ended(generation, Outcome::Closed);
        }
        self.stepped(generation, step);
    }

    /// What the core decided: its frames onto the socket, its events to the driver.
    fn stepped(&mut self, generation: u64, mut step: Step) {
        let fatal = step.fatal.take();
        let recycle = step.recycle;
        self.dispatch(step);
        if let Some(error) = fatal {
            self.ended(generation, Outcome::Fatal(error));
        } else if recycle {
            self.ended(generation, Outcome::Recycle);
        }
    }

    fn dispatch(&mut self, step: Step) {
        let now = self.now;
        if let Link::Open { server: Some(server), .. } = &mut self.link {
            let generation = self.machine.connection_generation();
            for frame in step.frames {
                match frame {
                    Outgoing::AudioAppend { frame } => {
                        for chunk in self.audio.push(&frame.data) {
                            let samples = chunk.samples_per_channel;
                            let frame = AudioFrame::new(chunk.data, SAMPLE_RATE, 1, samples);
                            server.client(now, &Outgoing::AudioAppend { frame }, &mut self.record);
                        }
                    }
                    Outgoing::AudioClear => {
                        self.audio = batches();
                        server.client(now, &Outgoing::AudioClear, &mut self.record);
                    }
                    frame => {
                        server.client(now, &frame, &mut self.record);
                        // The writer's receipt: the request is on the wire.
                        match &frame {
                            Outgoing::ResponseCreate { req_id, .. } => {
                                self.machine.response_sent(generation, now, req_id, false);
                            }
                            Outgoing::ForceMessage { req_id, .. } => {
                                self.machine.response_sent(generation, now, req_id, true);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        self.events.extend(step.events.into_iter().map(|e| (now, e)));
    }

    /// One connection ended: the socket closed if it was still up, and what the connection policy
    /// makes of the outcome.
    fn ended(&mut self, generation: u64, outcome: Outcome) {
        let now = self.now;
        self.link = Link::Waiting;
        self.reconnecting = true;
        let follow_up = match outcome {
            Outcome::Closed => {
                let _ = self.connection.close(now);
                let _ = self.connection.released(generation, now);
                return self.end(self.closing);
            }
            Outcome::Fatal(error) => {
                self.events.push((now, Event::Error(error)));
                let _ = self.connection.close(now);
                let _ = self.connection.released(generation, now);
                return self.end(true);
            }
            Outcome::Effects(effects) => effects,
            Outcome::Recycle => self.connection.failed(generation, now, lost_connection()),
            Outcome::Failed(error) => self.connection.failed(generation, now, error),
        };
        self.effects.extend(follow_up);
        let released = self.connection.released(generation, now);
        self.effects.extend(released);
    }

    /// The pump ends; `close`: the core closes, failing what still waits.
    fn end(&mut self, close: bool) {
        if close {
            let step = self.machine.close(self.now);
            let now = self.now;
            self.events.extend(step.events.into_iter().map(|e| (now, e)));
        }
        self.link = Link::Over;
    }

    fn emit_error(&mut self, error: ApiError, recoverable: bool) {
        let event = Event::Error(RealtimeModelError {
            timestamp: self.now,
            label: LABEL.into(),
            message: error.message.clone(),
            recoverable,
            req_id: None,
            code: None,
            provider_error: Some(error),
        });
        self.events.push((self.now, event));
    }
}

/// The writer's batching of the person's audio: whole 100 ms appends, the rest held back.
fn batches() -> AudioByteStream {
    AudioByteStream::new(SAMPLE_RATE, 1, None, false)
}

/// Whether connection effects end the attempt in flight: anything but a cleanup diagnostic.
fn ends_attempt(effects: &[ConnectionEffect]) -> bool {
    effects.iter().any(|e| !matches!(e, ConnectionEffect::CleanupIncomplete { .. }))
}

/// "connection closed unexpectedly": retryable, so the session reconnects.
fn lost_connection() -> ApiError {
    let message = format!("{} connection closed unexpectedly", INFERENCE.provider_label);
    ApiError::connection(Some(message))
}
