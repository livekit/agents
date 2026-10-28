//! The session's actor: the loop, its input queue and timers, and the handle a host and its peer
//! systems hold.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::VoiceSession;
use crate::agent_session::SessionShared;
use crate::agent_session::commands::{
    Answer, Close, Command, HostRecorder, IvrCommand, Origin, RequestId, Sent, SpeechCall,
};
use crate::agent_session::effects::{Out, Timer};
use crate::agent_session::host_events::HostEvent;
use crate::agent_session::time::Now;
use crate::audio::ChannelEnd;
use crate::events::SessionEvent;
use crate::generation::GenEvent;
use crate::room_io::StreamOpened;
use crate::speech_handle::SpeechHandle;
use crate::trace::{
    CloseAct, CloseStep, CoreStep, PlayoutStep, RoomStep, SpeechStep, Step, Journal,
};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::{FrameId, SpeechEvent, TimerSet, Ts, VadEvent};
use lk_agents_rtc::RoomEvent;
use std::sync::{Arc, Mutex};

/// Only what a host could read, in loop order; never machine state.
#[derive(Debug, Clone)]
pub enum SystemTap {
    Session(SessionEvent),
    Vad(lk_agents_abi::VadEvent),
    Stt(lk_agents_abi::SpeechEvent),
    Ivr(IvrCommand),
    CloseRequested,
}

/// One unbounded channel per subscriber; a departed one is forgotten on publish.
#[derive(Clone, Default)]
pub struct WatchHub {
    taps: Arc<Mutex<Vec<tokio::sync::mpsc::UnboundedSender<SystemTap>>>>,
}

impl WatchHub {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn subscribe(&self) -> tokio::sync::mpsc::UnboundedReceiver<SystemTap> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.taps.lock().unwrap_or_else(|p| p.into_inner()).push(tx);
        rx
    }

    /// Cheap with none: the common session runs no system.
    pub fn publish(&self, item: SystemTap) {
        let mut taps = self.taps.lock().unwrap_or_else(|p| p.into_inner());
        taps.retain(|tap| tap.send(item.clone()).is_ok());
    }

    pub fn is_empty(&self) -> bool {
        self.taps.lock().unwrap_or_else(|p| p.into_inner()).is_empty()
    }
}

impl std::fmt::Debug for WatchHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WatchHub")
            .field("taps", &self.taps.lock().map(|t| t.len()).unwrap_or(0))
            .finish()
    }
}

/// Everything a peer of the host may touch, and nothing else.
#[derive(Clone)]
pub struct PeerHandle {
    /// The loop's queue: commands, and the audio taps' installs.
    pub queue: Tx,
    pub(super) events: tokio::sync::mpsc::UnboundedSender<HostEvent>,
    pub journal: Journal,
    pub time: crate::agent_session::Timebase,
    pub spawner: Arc<dyn lk_agents_abi::Spawner>,
    pub(super) watch: WatchHub,
    pub(super) resources: lk_agents_abi::resources::Resources,
    /// Flips at `begin_closing`: a system's own drain ends on it like the driver's do.
    pub closing: tokio::sync::watch::Receiver<bool>,
}

impl PeerHandle {
    pub fn subscribe(&self) -> tokio::sync::mpsc::UnboundedReceiver<SystemTap> {
        self.watch.subscribe()
    }

    pub fn send(&self, origin: Origin, cmd: impl Into<Command>) -> Result<(), QueueClosed> {
        self.queue.send_from(origin, cmd.into())
    }

    pub fn journal(&self, step: Step) {
        self.journal.record(step);
    }

    pub fn emit(&self, ev: SessionEvent) {
        if self.journal.is_on() {
            self.journal(Step::Emitted(ev.journal_form()));
        }
        let _ = self.events.send(HostEvent::Session(ev));
    }

    /// The microphone's own frames, never the gate's silence. A second tap replaces the first.
    pub fn install_audio_tap(
        &self,
        sink: Arc<dyn lk_agents_abi::StreamSink<lk_agents_abi::AudioFrame>>,
    ) -> lk_agents_abi::resources::ResourceId {
        let sink = self.resources.register_audio_sink(sink);
        let channel = crate::audio::AudioChannel::Amd;
        let _ = self.queue.send(crate::data_plane::ForwarderCommand::Install { channel, sink });
        sink
    }

    /// Only if this tap is still the one in place.
    pub fn remove_audio_tap(&self, tap: lk_agents_abi::resources::ResourceId) {
        let _ = self.queue.send(crate::data_plane::ForwarderCommand::Remove(tap));
    }
}

impl std::fmt::Debug for PeerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerHandle").field("time", &self.time).field("watch", &self.watch).finish()
    }
}

pub struct SessionHandle {
    pub api: Tx,
    /// Every event, in the order the driver emitted it.
    pub events: Rx<HostEvent>,
    /// Each awaited command's answer, once, by its call id.
    pub answers: Rx<(RequestId, Answer)>,
    /// The last keyterm list pushed to the recogniser; `None` before any push.
    pub keyterms: tokio::sync::watch::Receiver<Option<Vec<String>>>,
    /// The driving agent's context and state, as last published.
    pub activity: tokio::sync::watch::Receiver<crate::agent_session::ActivitySnapshot>,
    /// Whether a new speech would be refused now.
    pub scheduling: tokio::sync::watch::Receiver<crate::agent_session::SchedulingGate>,
    /// What a system runs on; the host starts the systems it configures on this after `start()`.
    pub peer: PeerHandle,
    pub ivr_detection: bool,
}

pub(super) type Rx<T> = tokio::sync::mpsc::UnboundedReceiver<T>;

/// The loop's one queue: every source sends on it, so the loop receives one [`Event`] at a time
/// in arrival order and merges it with its timers by a stated rule; the trace is a function of the
/// event sequence. What each source sends becomes an event by its `From` impl below.
#[derive(Clone)]
pub struct Tx(tokio::sync::mpsc::UnboundedSender<Event>);

impl Tx {
    pub(super) fn new(inner: tokio::sync::mpsc::UnboundedSender<Event>) -> Self {
        Tx(inner)
    }

    /// A queue nobody reads: what a handle made outside a session sends on.
    pub fn disconnected() -> Self {
        Tx(tokio::sync::mpsc::unbounded_channel().0)
    }

    pub fn send(&self, event: impl Into<Event>) -> Result<(), QueueClosed> {
        self.0.send(event.into()).map_err(|_| QueueClosed)
    }

    pub fn send_from(&self, origin: Origin, command: Command) -> Result<(), QueueClosed> {
        self.send(Input::Host(Sent { origin, command }))
    }

    pub fn is_closed(&self) -> bool {
        self.0.is_closed()
    }

    /// Resolves when the loop has dropped its receiver.
    pub async fn closed(&self) {
        self.0.closed().await
    }
}

impl std::fmt::Debug for Tx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Tx")
    }
}

/// What each source sends, as the loop receives it.
macro_rules! queue_events {
    ($($ty:ty => |$v:pat_param| $event:expr;)*) => {$(
        impl From<$ty> for Event {
            fn from($v: $ty) -> Event {
                $event
            }
        }
    )*};
}
queue_events! {
    Input => |input| Event::Input(input);
    crate::data_plane::ForwarderCommand => |command| Event::Forwarder(command);
    Command => |command| Event::Input(Input::Host(Sent { origin: Origin::Host, command }));
    crate::audio::Tagged<SpeechEvent> => |ev| Event::Input(Input::Stt(ev));
    VadEvent => |ev| Event::Input(Input::Vad(ev));
    RoomEvent => |ev| Event::Input(Input::Room(ev));
    crate::room_io::RoomIoEvent => |ev| Event::Input(Input::RoomIo(ev));
    GenEvent => |ev| Event::Input(Input::Gen(ev));
    crate::agent_activity::realtime::RealtimeWake => |wake| Event::Input(Input::Realtime(wake));
    lk_agents_abi::turn::detector::LegEvent<lk_agents_abi::EotPrediction> =>
        |ev| Event::Input(eot_input(ev));
    lk_agents_abi::turn::detector::LegEvent<crate::audio_recognition::OverlapVerdict> =>
        |ev| Event::Input(Input::Overlap(ev));
    ChannelEnd => |end| Event::Input(Input::ChannelEnded(end));
    StreamOpened => |opened| Event::Input(Input::StreamOpened(opened));
    lk_agents_rtc::VideoFrameRef => |frame| Event::Input(Input::Video(frame));
}

/// `Closed` is the pump's own end.
fn eot_input(ev: lk_agents_abi::turn::detector::LegEvent<lk_agents_abi::EotPrediction>) -> Input {
    use lk_agents_abi::turn::detector::{LegEvent, ReqId};
    match ev {
        LegEvent::Verdict { req, verdict } => Input::Eot(req.unwrap_or(ReqId(0)), Ok(verdict)),
        LegEvent::Error { req, recoverable: _, err } => {
            Input::Eot(req.unwrap_or(ReqId(0)), Err(err))
        }
        LegEvent::Closed => Input::ChannelEnded(ChannelEnd {
            leg: "eot",
            cause: crate::trace::ChannelEndCause::SourceEnded,
            sink: None,
            closed: None,
            error: None,
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueClosed;

/// Timers are the loop's own.
#[derive(lk_agents_abi::Vocabulary)]
pub enum Input {
    Host(Sent),
    Stt(crate::audio::Tagged<SpeechEvent>),
    Vad(VadEvent),
    Room(RoomEvent),
    /// Forwarded from the loop so they share one arrival order.
    RoomIo(crate::room_io::RoomIoEvent),
    Gen(GenEvent),
    Realtime(crate::agent_activity::realtime::RealtimeWake),
    Eot(
        lk_agents_abi::turn::detector::ReqId,
        Result<lk_agents_abi::EotPrediction, lk_agents_abi::errors::ProviderError>,
    ),
    Overlap(lk_agents_abi::turn::detector::LegEvent<crate::audio_recognition::OverlapVerdict>),
    ChannelEnded(ChannelEnd),
    /// The STT stream failed an attempt it retries.
    SttRetried(lk_agents_abi::errors::ProviderError),
    StreamOpened(crate::room_io::StreamOpened),
    Video(lk_agents_rtc::VideoFrameRef),
    /// The data plane's reports. The source's first frame forwarded (`input_started_at`).
    InputStarted {
        at: Ts,
    },
    /// Silence substituted per channel since the previous gate change, at a gate change.
    GateSilenced(Vec<(crate::audio::AudioChannel, u64)>),
    InputCensus(crate::data_plane::InputCensus),
}

/// What the queue owner hands [`Runtime::handle`](crate::runtime::Runtime::handle): an event in
/// arrival order, or time reaching [`Driver::wake_at`].
pub enum Event {
    /// For the decision core.
    Input(Input),
    /// For the data plane, from the decision core.
    Forwarder(crate::data_plane::ForwarderCommand),
    Wake,
}

/// What handling an [`Event`] reads and writes, for the DST's partial-order sampling: two events
/// race when they touch one object and one of them writes it. A pure function of the event, read
/// off the router below; where a handler's reach is unclear it writes more, never less, since a
/// false independence is what loses an interleaving's weight.
#[cfg(feature = "dst")]
pub mod objects {
    use super::{Event, Input};
    use crate::agent_activity::realtime::RealtimeWake;
    use crate::agent_activity::scheduling::SpeechId;
    use crate::agent_session::commands::{
        ActivityCommand, Command, Sent, SpeechCommand, TurnCommand,
    };
    use crate::data_plane::ForwarderCommand;
    use crate::generation::GenEvent;
    use lk_agents_abi::resources::ResourceId;

    /// One piece of the session's state.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum Object {
        /// Everything: the close, a handoff, a timer.
        All,
        /// One speech: its handle, steps and playout.
        Speech(SpeechId),
        /// The scheduler: the queue, the current speech, its authorization, the agent's state.
        Floor,
        /// The user's turn: recognition, endpointing, the user's state.
        Turn,
        /// The running agent's conversation record.
        Chat,
        /// The running agent's configuration: instructions, tools, models, options.
        Agent,
        /// The provider legs the core holds and which streams are live.
        Legs,
        /// One recognition stream.
        Stream(ResourceId),
        /// The data plane.
        Forwarder,
        /// The realtime socket's machine.
        Realtime,
        /// The room link.
        Room,
        /// The host hooks in flight.
        Hooks,
        /// The tool calls in flight.
        Tools,
        /// The close's phase.
        Close,
        /// Unrecoverable errors counted toward the close.
        Errors,
        Keyterms,
        Dtmf,
        Recorder,
        Video,
        Run,
        Fillers,
        /// The session's span tree, which every step on a turn, a speech or the conversation
        /// also writes.
        Spans,
        /// The idle waiters and holds.
        Waiters,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Access {
        pub object: Object,
        pub write: bool,
    }

    /// Two access sets race: one object in both, written by at least one (POS*: reads commute).
    pub fn races(a: &[Access], b: &[Access]) -> bool {
        a.iter().any(|x| {
            b.iter().any(|y| {
                (x.write || y.write)
                    && (x.object == y.object || x.object == Object::All || y.object == Object::All)
            })
        })
    }

    use Object::*;

    fn w(objects: &[Object]) -> Vec<Access> {
        objects.iter().map(|&object| Access { object, write: true }).collect()
    }

    fn with_reads(mut accesses: Vec<Access>, reads: &[Object]) -> Vec<Access> {
        accesses.extend(reads.iter().map(|&object| Access { object, write: false }));
        accesses
    }

    impl Event {
        pub fn accesses(&self) -> Vec<Access> {
            let mut accesses = match self {
                Event::Wake => w(&[All]),
                Event::Forwarder(ForwarderCommand::ReportCensus) => {
                    with_reads(Vec::new(), &[Forwarder])
                }
                Event::Forwarder(_) => w(&[Forwarder]),
                Event::Input(input) => input_accesses(input),
            };
            let spans = |a: &Access| {
                a.write
                    && matches!(
                        a.object,
                        Speech(_) | Floor | Turn | Chat | Agent | Hooks | Tools | Close | Realtime
                    )
            };
            if accesses.iter().any(spans) {
                accesses.push(Access { object: Spans, write: true });
            }
            accesses
        }
    }

    /// A speech's own event: it writes the speech and reads the scheduler that holds it.
    fn speech(id: SpeechId, writes: &[Object]) -> Vec<Access> {
        let mut accesses = w(&[Speech(id)]);
        accesses.extend(w(writes));
        with_reads(accesses, &[Floor, Agent])
    }

    fn input_accesses(input: &Input) -> Vec<Access> {
        let recognition = |stream: Option<ResourceId>| {
            let mut accesses = w(&[Turn, Floor, Chat]);
            accesses.extend(stream.map(|s| Access { object: Stream(s), write: true }));
            with_reads(accesses, &[Legs, Agent])
        };
        match input {
            Input::Host(Sent { command, .. }) => command_accesses(command),
            Input::Stt(tagged) => {
                if tagged.item.kind == lk_agents_abi::SpeechEventType::RecognitionUsage {
                    with_reads(w(&[Turn]), &[Legs])
                } else {
                    recognition(Some(tagged.stream))
                }
            }
            Input::Vad(_) | Input::Eot(..) | Input::Overlap(_) => recognition(None),
            Input::Room(lk_agents_rtc::RoomEvent::Close { .. }) => w(&[All]),
            Input::Room(_) => Vec::new(),
            Input::RoomIo(ev) => match ev {
                crate::room_io::RoomIoEvent::ParticipantUnlinked(_) => w(&[All]),
                crate::room_io::RoomIoEvent::ParticipantLinked(_)
                | crate::room_io::RoomIoEvent::KeyPressed => w(&[Turn, Room]),
                crate::room_io::RoomIoEvent::TextInput(_)
                | crate::room_io::RoomIoEvent::DeleteRoomRequested(_) => w(&[Room]),
            },
            Input::Gen(ev) => gen_accesses(ev),
            Input::Realtime(wake) => {
                let mut accesses = w(&[Realtime, Floor, Turn, Chat, Tools]);
                match wake {
                    RealtimeWake::ToolsDone { speech, .. } | RealtimeWake::Heard { speech, .. } => {
                        accesses.push(Access { object: Speech(*speech), write: true })
                    }
                    RealtimeWake::Closed | RealtimeWake::Ended => {
                        accesses.push(Access { object: Close, write: true })
                    }
                    RealtimeWake::Event(_) => {}
                }
                with_reads(accesses, &[Agent])
            }
            Input::ChannelEnded(_) => with_reads(w(&[Legs, Turn, Floor, Errors, Close]), &[Agent]),
            Input::SttRetried(_) => w(&[Errors, Close]),
            Input::StreamOpened(_) => with_reads(w(&[Legs, Keyterms]), &[Close]),
            // A sampled frame mints the ids an item would, and pushes it to the model.
            Input::Video(_) => with_reads(w(&[Video, Chat, Realtime]), &[Turn, Close]),
            Input::InputStarted { .. } => w(&[Legs]),
            Input::GateSilenced(_) => Vec::new(),
            Input::InputCensus(_) => with_reads(Vec::new(), &[Close]),
        }
    }

    fn gen_accesses(ev: &GenEvent) -> Vec<Access> {
        match ev {
            GenEvent::FirstFrame { speech: id } | GenEvent::PlaybackStarted { speech: id } => {
                speech(*id, &[Floor, Turn])
            }
            GenEvent::Done { speech: id, .. }
            | GenEvent::StepSpoken { speech: id, .. }
            | GenEvent::Failed { speech: id, .. } => speech(*id, &[Floor, Turn, Chat, Tools]),
            GenEvent::ToolsFinished { speech: id, .. } => speech(*id, &[Floor, Chat, Tools, Hooks]),
            GenEvent::ToolStep { speech: id, .. } => speech(*id, &[Floor]),
            GenEvent::ChatFinished { speech: id, error, .. } => {
                speech(*id, if error.is_some() { &[Floor] } else { &[] })
            }
            GenEvent::LaunchReleased { speech: id } => {
                with_reads(Vec::new(), &[Speech(*id), Floor])
            }
            GenEvent::ToolCallStarted { speech: id, .. } => speech(*id, &[Tools, Fillers]),
            GenEvent::ToolCallEnded { speech: id, .. } => {
                speech(*id, &[Tools, Floor, Chat, Fillers])
            }
            GenEvent::ProviderFailed { speech: id, .. } => speech(*id, &[Errors, Close]),
            GenEvent::KeytermAnswer { .. } => with_reads(w(&[Keyterms]), &[Legs]),
            // A forwarded event may open or close a span.
            GenEvent::Emit(_) => w(&[Spans]),
            GenEvent::Trace(_) => Vec::new(),
        }
    }

    fn command_accesses(command: &Command) -> Vec<Access> {
        match command {
            Command::Speech(cmd) => match cmd {
                SpeechCommand::Say { .. } | SpeechCommand::GenerateReply { .. } => {
                    with_reads(w(&[Floor, Chat]), &[Agent, Turn, Close])
                }
                SpeechCommand::Interrupt { .. } => w(&[Floor, Turn]),
                SpeechCommand::InterruptSpeech { speech, .. }
                | SpeechCommand::SetAllowInterruptions { speech, .. } => {
                    w(&[Speech(*speech), Floor])
                }
                SpeechCommand::CommitUserTurn { .. } => w(&[Turn, Floor, Chat]),
                SpeechCommand::WaitForIdle { .. } => with_reads(w(&[Waiters]), &[Floor, Turn]),
                SpeechCommand::WaitForIdleAndHold { .. }
                | SpeechCommand::ReleaseIdleHold { .. } => {
                    with_reads(w(&[Waiters, Floor]), &[Turn])
                }
                SpeechCommand::ClaimUserTurn | SpeechCommand::ReleaseUserTurn => {
                    w(&[Turn, Floor, Waiters])
                }
                SpeechCommand::EndCall { speech } => w(&[Speech(*speech), Close]),
                SpeechCommand::SetRunOutput { speech, .. } => w(&[Speech(*speech), Run]),
                SpeechCommand::ToolUpdate { .. } => w(&[Tools, Floor, Chat]),
                SpeechCommand::FillerStart { speech, .. } => {
                    with_reads(w(&[Fillers]), &[Speech(*speech), Floor])
                }
                SpeechCommand::FillerStop { .. } => w(&[Fillers]),
                SpeechCommand::PauseAuthorization | SpeechCommand::ResumeAuthorization => {
                    w(&[Floor])
                }
            },
            Command::Activity(cmd) => match cmd {
                ActivityCommand::Update(_) | ActivityCommand::Drain { .. } => w(&[All]),
                ActivityCommand::UpdateInstructions(_)
                | ActivityCommand::UpdateTools(_)
                | ActivityCommand::UpdateExpressive(_) => w(&[Agent, Chat]),
                ActivityCommand::SetAudioInput(_) => w(&[Legs, Turn]),
                ActivityCommand::SetVideoInput(_) => w(&[Video]),
                ActivityCommand::SetIoEnabled { .. } => w(&[Legs, Turn, Floor]),
                ActivityCommand::UpdateChatCtx(_) => w(&[Chat, Realtime]),
                ActivityCommand::SetAudioOutput(_) => w(&[Legs, Floor]),
                ActivityCommand::SetTranscriptionOutput(_) => w(&[Legs]),
                ActivityCommand::UpdateModels { .. } => w(&[Legs, Agent, Realtime]),
            },
            Command::Turn(cmd) => match cmd {
                TurnCommand::Commit { .. } | TurnCommand::Clear => w(&[Turn, Floor, Chat]),
                TurnCommand::Configure(_) => w(&[Agent, Turn, Floor]),
                TurnCommand::SkipReplies(_) => w(&[Turn]),
            },
            Command::Video(_) => w(&[Video, Chat, Realtime]),
            Command::Close(_) => w(&[All]),
            Command::HookDone { .. } => w(&[Hooks, Floor, Turn, Chat, Tools]),
            Command::HostMetrics(_) => Vec::new(),
            Command::Ivr(crate::agent_session::IvrCommand::AddTools(_)) => w(&[Agent]),
            Command::Ivr(_) => Vec::new(),
            Command::SetStaticKeyterms(_) => w(&[Keyterms]),
            Command::Recorder(_) => w(&[Recorder]),
            Command::Dtmf(_) => w(&[Dtmf, Floor]),
            Command::Run(_) => w(&[Run, Floor, Chat, Agent]),
            Command::AmdSpan(_) => w(&[Spans]),
        }
    }
}

/// The session's decision core: handed every [`Input`] in arrival order by the
/// [`Runtime`](crate::runtime::Runtime), it merges them with its timers and orchestrates the
/// components in sequence: the [`AgentSession`], its running
/// [`AgentActivity`](crate::agent_activity::AgentActivity), and the [`AgentInput`] legs. Every
/// decision is journalled, dated on the session clock.
pub struct Driver {
    pub(crate) session: VoiceSession,
    pub(crate) activity: crate::agent_activity::AgentActivity,
    pub(crate) input: crate::agent_activity::io::AgentInput,
    /// Machines never see it; [`Out`] carries it for them.
    pub(crate) current: FrameId,
    pub(crate) timers: TimerSet<Timer>,
    /// What the spawned tasks share with the loop, the session clock among it.
    pub(crate) shared: Arc<SessionShared>,
    pub(crate) events: tokio::sync::mpsc::UnboundedSender<HostEvent>,
    pub(super) answers: tokio::sync::mpsc::UnboundedSender<(RequestId, Answer)>,
    pub(crate) journal: Journal,
    /// The core knows no subscriber.
    pub(crate) watch: WatchHub,
    /// Where the spawned work reports back, and the data plane's commands go, in one order.
    pub(crate) queue: Tx,
    /// Decides ownership (a tool's hold is its run's).
    pub(crate) command_origin: Origin,
    /// The time of the step being handled, or of the last one.
    pub(crate) step: crate::agent_session::time::StepClock,
    pub(crate) stopped: bool,
    pub(super) opened: bool,
    /// The activity changed this cycle: published once, as the cycle's effects are done.
    pub(crate) activity_changed: std::cell::Cell<bool>,
}

impl Driver {
    /// Only while performing an effect, when no cycle exists.
    pub(crate) fn journal(&self, step: Step) {
        self.journal.record(step);
    }

    /// An event sent to the host where no cycle exists, journalled first.
    pub(crate) fn emit_now(&self, ev: SessionEvent) {
        if self.journal.is_on() {
            self.journal(Step::Emitted(ev.journal_form()));
        }
        let _ = self.events.send(HostEvent::Session(ev));
    }

    pub(crate) fn out(&self) -> Out {
        Out::new(self.current, self.journal.clone())
    }

    /// Performed with its events on the host stream, so `start()` returns after them. Idempotent.
    pub fn open(&mut self, now: Now) {
        if std::mem::replace(&mut self.opened, true) {
            return;
        }
        self.step.set(now);
        let now = now.at;
        // The AEC warmup is *not* armed here but at the first `speaking` edge; armed now, it would
        // refuse every interruption for the session.
        self.publish_gate();
        // The away deadline runs from start: both are silent until somebody speaks.
        {
            let mut out = self.out();
            let mut spans = Vec::new();
            let wall = self.step.wall_secs_at(now);
            let agent = self.activity.agent.clone();
            self.session.trace.open(wall, &agent, &mut spans);
            spans.into_iter().for_each(|ev| out.machine().emit(ev));
            // `start()`'s `_update_agent_state("listening")`: the host's first state edge.
            self.set_agent_state(now, AgentState::Listening, &mut out);
            let closing = self.session.closing_past_edge();
            out.machine().append(self.session.states.reconcile_away_timer(now, closing));
            if let Some(recorder) = self.session.start_recorder.take() {
                self.attach_recorder(now, recorder, &mut out);
            }
            self.issue_on_enter(now, &mut out);
            let mut spans = Vec::new();
            self.session.trace.opened(self.step.wall_secs_at(now), &mut spans);
            spans.into_iter().for_each(|ev| out.machine().emit(ev));
            self.execute(out);
        }
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Answer an awaited command; `None` is a command nobody awaits.
    pub(crate) fn answer(&self, call: Option<RequestId>, answer: Answer) {
        if let Some(call) = call {
            let _ = self.answers.send((call, answer));
        }
    }

    /// The handle of a speech the loop creates now: the one the host already holds for its
    /// `say()` / `generate_reply()`, else a new one.
    pub(crate) fn new_speech(
        &self,
        host: Option<&Arc<SpeechHandle>>,
        id: SpeechId,
        allow_interruptions: bool,
    ) -> Arc<SpeechHandle> {
        match host {
            Some(handle) => {
                handle.create(id, allow_interruptions);
                Arc::clone(handle)
            }
            None => Arc::new(SpeechHandle::new(id, allow_interruptions, self.shared.queue.clone())),
        }
    }

    /// The host's call is answered once its speech exists, in the cycle that announces it.
    pub(crate) fn answer_speech(&self, call: Option<SpeechCall>) {
        if let Some(call) = call {
            self.answer(Some(call.id), Answer::Speech);
        }
    }

    /// The loop stopped. What was queued and never delivered is refused (a census is still
    /// journalled), then the session's last event goes out, then every waiter and task the loop
    /// leaves is released: nothing settles after `close`.
    pub(crate) fn finish(&mut self, undelivered: Vec<Input>) {
        let mut out = self.out();
        for input in undelivered {
            match input {
                // Only a host command carries a promise somebody awaits. What it hands over is
                // released with what the core held.
                Input::Host(Sent { origin: _, command }) => {
                    self.session.providers.extend(command.resources());
                    self.refuse_command(command, &mut out)
                }
                Input::InputCensus(census) => self.on_input_census(census, &mut out),
                Input::StreamOpened(StreamOpened { outcome: Ok(stream), .. }) => {
                    self.discard_open(stream)
                }
                _ => {}
            }
        }
        // Only the refusals are sent: nothing else takes effect once the loop stopped.
        for (call, answer) in out.take_answers() {
            self.answer(Some(call), answer);
        }
        self.run_close_at_end();
        let mut out = self.out();
        if let Some((reason, error)) =
            self.session.closing.as_ref().map(|c| (c.reason, c.error.clone()))
        {
            out.machine().emit(SessionEvent::Close { reason, error });
        }
        self.execute(out);
        self.settle_waiters_at_end();
        self.release_all_providers();
    }

    /// When the core next wants [`Event::Wake`]: its earliest deadline, or the last step's instant
    /// when a watched task is already due.
    pub fn wake_at(&self) -> Option<Ts> {
        self.timers.next_deadline()
    }

    /// Whether `timer` is armed, this cycle's own arms and cancels counted.
    pub(crate) fn armed(&self, timer: Timer, out: &mut Out) -> bool {
        out.arms(timer).unwrap_or_else(|| self.timers.is_armed(timer))
    }

    /// Cancels `timer` if it is armed: `true` when it was.
    pub(crate) fn disarm(&self, timer: Timer, out: &mut Out) -> bool {
        let armed = self.armed(timer, out);
        if armed {
            out.machine().cancel(timer);
        }
        armed
    }

    /// One step: `input` at `now` (`None`: time reached [`Self::wake_at`]), then everything it
    /// made due in the same instant.
    pub(crate) fn handle(&mut self, input: Option<Input>, now: Now) {
        self.step.set(now);
        let now = now.at;
        let mut out = self.out();
        match input {
            None => {
                if self.timers.next_deadline().is_some_and(|at| at <= now) {
                    self.on_timers(now, &mut out);
                }
            }
            Some(Input::Stt(ev)) => self.on_stt(now, ev, &mut out),
            Some(Input::Vad(ev)) => self.on_vad(now, ev, &mut out),
            Some(Input::Room(ev)) => self.on_rtc(now, ev, &mut out),
            Some(Input::Host(Sent { origin, command })) => {
                out.trace(CoreStep::CommandReceived {
                    origin: origin.clone(),
                    kind: command.kind(),
                });
                self.command_origin = origin;
                self.session.providers.extend(command.resources());
                // Past the close's hard edge no command takes effect: its promise is answered.
                // A `Close` joins; a hook's answer is still read.
                let late = self.session.closing_past_edge()
                    && !matches!(command, Command::Close(_) | Command::HookDone { .. });
                if late {
                    self.refuse_command(command, &mut out);
                    self.pump_schedule(now, &mut out);
                    self.execute(out);
                    self.publish_gate();
                    return;
                }
                self.on_api(now, command, &mut out);
            }
            Some(Input::RoomIo(ev)) => self.on_room_io(now, ev, &mut out),
            Some(Input::Gen(ev)) => self.on_generation(now, ev, &mut out),
            Some(Input::Realtime(ev)) => self.on_realtime(now, ev, &mut out),
            Some(Input::Eot(req, p)) => self.on_eot(now, req, p, &mut out),
            Some(Input::Overlap(ev)) => self.on_overlap_verdict(now, ev, &mut out),
            Some(Input::ChannelEnded(end)) => self.on_leg_ended(now, end, &mut out),
            Some(Input::SttRetried(error)) => {
                self.report_error(now, crate::events::ErrorSource::Stt, true, error, None, &mut out);
            }
            Some(Input::StreamOpened(opened)) => self.on_leg_opened(now, opened, &mut out),
            Some(Input::Video(frame)) => self.on_video_frame(now, frame, &mut out),
            Some(Input::InputStarted { at }) => self.input.first_input_at = Some(at),
            Some(Input::GateSilenced(silenced)) => {
                for (leg, frames) in silenced {
                    out.trace(PlayoutStep::GateSilenced { leg, frames });
                }
            }
            Some(Input::InputCensus(census)) => self.on_input_census(census, &mut out),
        }
        // Scheduled in the same cycle as the wake that changed it.
        self.pump_schedule(now, &mut out);
        self.execute(out);
        // The run's settle check, after the wake's own effects; a re-prompt is scheduled in
        // this cycle.
        let mut out = self.out();
        if self.run_check(now, &mut out) {
            self.pump_schedule(now, &mut out);
        }
        self.settle_waiters(now, &mut out);
        // What the waiters started (a background tool's deferred reply, a filler's say) is scheduled
        // in this cycle too, or it waits, the floor free, for some unrelated wake.
        self.pump_schedule(now, &mut out);
        self.release_providers(&mut out);
        self.execute(out);
        self.publish_gate();
    }

    /// **Already done**, not dropped: an unattached handle would answer `done() == false` forever.
    pub(crate) fn settle_undelivered(&mut self, what: &str, call: Option<SpeechCall>) {
        let id = self.session.ids.next_speech_id();
        let allow_interruptions = self.session.config.effective.turn.interruption.enabled();
        let handle = self.new_speech(call.as_ref().map(|c| &c.speech), id, allow_interruptions);
        handle.mark_done(Some(lk_agents_abi::errors::ProviderError::Local(
            lk_agents_abi::errors::ApiError::new(format!(
                "the session loop ended before {what}() ran"
            )),
        )));
        // The host stream still exists, so the speech is announced and ended on it like every other.
        let kind = if what == "say" {
            crate::agent_activity::scheduling::SpeechKind::Say
        } else {
            crate::agent_activity::scheduling::SpeechKind::GenerateReply
        };
        let announce =
            SessionEvent::SpeechCreated { speech: Arc::clone(&handle), kind, user_initiated: true };
        self.emit_now(announce);
        // Born past the close's edge: the scheduling task's refusal and the close's release.
        self.journal(Step::Speech(SpeechStep::ReplyCreated { speech: id }));
        self.journal(Step::Speech(SpeechStep::ScheduleRejected {
            speech: id,
            reason: crate::agent_activity::scheduling::Rejected::Stopped,
        }));
        self.journal(Step::Close(CloseStep::Performed(CloseAct::Released { speech: id })));
        let done =
            SessionEvent::SpeechDone { speech: id, outcome: crate::events::SpeechOutcome::Failed };
        self.emit_now(done);
        self.answer_speech(call);
    }


    pub(super) fn on_rtc(&mut self, now: Ts, ev: RoomEvent, out: &mut Out) {
        out.trace(RoomStep::Event(crate::trace::RoomEventKind::of(&ev)));
        match ev {
            // The room going away is the job ending (`CloseReason.JOB_SHUTDOWN`, no drain).
            RoomEvent::Close { .. } => {
                let reason = lk_agents_abi::voice::events::CloseReason::JobShutdown;
                self.begin_closing(now, Close { reason, drain: false, error: None }, out)
            }
            // The departure arrives as the link's own edge (`ParticipantUnlinked`); the rest is
            // `RoomIo`'s.
            RoomEvent::TrackSubscribed { .. }
            | RoomEvent::TrackUnsubscribed { .. }
            | RoomEvent::TrackPublished { .. }
            | RoomEvent::TrackUnpublished { .. }
            | RoomEvent::LocalTrackPublished { .. }
            | RoomEvent::ParticipantConnected { .. }
            | RoomEvent::ParticipantActive { .. }
            | RoomEvent::ParticipantAttributesChanged { .. }
            | RoomEvent::ParticipantDisconnected { .. }
            | RoomEvent::ConnectionStateChanged { .. }
            | RoomEvent::SipDtmf { .. } => {}
        }
    }

    pub(super) fn on_api(&mut self, now: Ts, cmd: Command, out: &mut Out) {
        match cmd {
            Command::Speech(cmd) => self.on_speech(now, cmd, out),
            Command::Activity(cmd) => self.on_frame(now, cmd, out),
            Command::Turn(cmd) => self.on_turn(now, cmd, out),
            Command::Video(image) => {
                self.push_video(image, out);
            }
            Command::Recorder(crate::agent_session::RecorderCommand::Attach(HostRecorder(
                recorder,
            ))) => self.attach_recorder(now, recorder, out),
            Command::Recorder(crate::agent_session::RecorderCommand::Detach) => {
                self.detach_recorder(now, out)
            }
            Command::Close(close) => self.begin_closing(now, close, out),
            Command::HookDone { id, outcome } => self.on_hook_done(now, id, outcome, out),
            Command::Run(spec) => self.on_run(now, *spec, out),
            Command::AmdSpan(verdict) => {
                let mut spans = Vec::new();
                let wall = self.step.wall_secs_at(now);
                self.session.trace.amd(wall, verdict.as_ref(), &mut spans);
                spans.into_iter().for_each(|ev| out.machine().emit(ev));
            }
            Command::Dtmf(cmd) => self.dtmf_step(cmd.into(), out),
            Command::HostMetrics(record) => {
                out.machine().emit(crate::events::SessionEvent::MetricsCollected {
                    metrics: record,
                    speech: None,
                });
            }
            Command::SetStaticKeyterms(terms) => {
                self.keyterms_step(
                    now,
                    crate::keyterm_detection::KeytermIn::StaticTerms(terms),
                    out,
                );
            }
            // Addressed to the IVR navigator: journalled and relayed on the feed, acted on by
            // nobody: except `AddTools`, since the core owns what a request is offered.
            Command::Ivr(IvrCommand::AddTools(tools)) => {
                let added = self.session.extend_tools(tools);
                out.trace(crate::trace::ToolStep::SessionToolsExtended { added });
            }
            Command::Ivr(command) => {
                self.watch.publish(SystemTap::Ivr(command));
            }
        }
    }

    /// The next timer due by `now` that `pick` accepts, fired. One an earlier fire of this step
    /// cancelled or moved is not due: its cancel effect runs after the step, and asyncio's
    /// `_run_once` likewise skips a handle cancelled earlier in the same tick.
    pub(crate) fn pop_due(
        &mut self,
        now: Ts,
        out: &mut Out,
        pick: impl Fn(Timer) -> bool,
    ) -> Option<Timer> {
        for touched in out.timers_touched() {
            self.timers.cancel(touched);
        }
        let timer = self.timers.pop_due_where(now, pick)?;
        out.trace(CoreStep::TimerFired(timer.into(), now));
        Some(timer)
    }

    /// Every due timer but the scheduler's, which the scheduling pass fires wherever it runs.
    pub(super) fn on_timers(&mut self, now: Ts, out: &mut Out) {
        let pick = |t| !matches!(t, Timer::InterruptionTimeout(_) | Timer::ConsecutiveSpeechDelay);
        while let Some(timer) = self.pop_due(now, out, pick) {
            match timer {
                // A close that moved on ends the pass: the edge cancels the turn's timers, and
                // whatever else is due fires on the next wake, after the phase's effects.
                Timer::Close => {
                    if self.close_tick(now, out) {
                        return;
                    }
                }
                Timer::Commit => self.perform_pending_commit(now, out),
                Timer::Hook(id) => self.hook_timed_out(now, id, out),
                // `PausedSpeech` owns this one.
                Timer::FalseInterruption => self.on_false_interruption_timeout(now, out),
                // `_on_aec_warmup_expired`: the one place `aec_warmup_armed` goes back to `false`
                // before close.
                Timer::AecWarmup => {
                    let closing = self.session.closing_past_edge();
                    self.session.aec.expire(closing);
                }
                Timer::UserAway => {
                    out.machine().append(self.session.states.on_user_away(now));
                }
                Timer::RecorderFlush => self.flush_recorder(now, true, out),
                Timer::KeytermTimeout => self.on_keyterm_timeout(now, out),
                Timer::DtmfDebounce => {
                    self.dtmf_step(crate::workflows::dtmf::DtmfTaskEvent::Debounced, out)
                }
                Timer::EndCallReply => self.end_call_reply_timed_out(now, out),
                // A realtime request the server never answered: the machine settles it.
                Timer::RealtimeRequest(key) => self.realtime_request_timed_out(key, out),
                Timer::Filler(id) => self.on_filler_timer(now, id, out),
                Timer::EotPredictionTimeout
                | Timer::EndpointingDelay
                | Timer::TranscriptionTimeout
                | Timer::BackchannelBoundary
                | Timer::UserTurnDuration => {
                    let eff = self.activity.audio_recognition.on_timer(now, timer);
                    out.machine().append(eff);
                }
                // Fired by the scheduling pass.
                Timer::InterruptionTimeout(_) | Timer::ConsecutiveSpeechDelay => {}
            }
        }
    }

    /// Sent to the data plane and journalled only when the verdict changed. Through
    /// [`Driver::journal`]: no cycle exists here.
    pub(super) fn publish_gate(&mut self) {
        self.shared.launch.publish(self.launch_hold());
        let session = &self.session;
        let inputs = crate::audio::GateInputs {
            agent_state: session.states.agent_state(),
            aec_warmup_remaining: session.aec.remaining(),
            aec_warmup_armed: session.aec.armed(),
            uninterruptible_speech_active: self
                .current_handle()
                .is_some_and(|h| !h.done() && !h.interrupted() && !h.allow_interruptions()),
            discard_audio_if_uninterruptible: session
                .config
                .effective
                .turn
                .interruption
                .discard_audio_if_uninterruptible(),
        };
        let discard = crate::audio::audio_gate(&inputs);
        if self.input.gate_discard.replace(discard) != Some(discard) {
            let _ = self.queue.send(crate::data_plane::ForwarderCommand::SetGateDiscard(discard));
            self.journal(Step::Playout(PlayoutStep::GatePublished {
                stt_real: !discard,
                realtime_real: !discard,
                vad_real: true,
                amd_real: true,
            }));
        }
    }

    /// Interruptible speeches only (an uninterruptible greeting plays over the user by design); a
    /// speech whose first frame is out is never held again.
    pub(super) fn launch_hold(&self) -> std::collections::BTreeSet<SpeechId> {
        // Read off the gate, not the scheduling task's current speech, which a block or
        // supersession can move away from a speech still parked at launch.
        if !self.activity.audio_recognition.user_speaking() {
            return std::collections::BTreeSet::new();
        }
        self.shared
            .launch
            .launching()
            .into_iter()
            .filter(|speech| {
                self.activity.speech_scheduler.handle(*speech).is_some_and(|h| h.interruptible())
            })
            .collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DriverError {
    /// **Moved, never stringified.**
    #[error("provider: {0}")]
    Provider(#[from] ProviderError),

    #[error("resample: {0}")]
    Resample(String),

    #[error("the session has no {0}")]
    Missing(&'static str),
}

/// The IVR navigator, when the session detects IVRs; the AMD is the host's to start.
pub fn start_ivr(peer: &PeerHandle, ivr_detection: bool) {
    crate::ivr::IvrSystem::spawn(peer.clone(), ivr_detection);
}
