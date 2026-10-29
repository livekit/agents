//! `_core.VoiceSession` as the host holds it: assembled from the host's arguments, started,
//! commanded and closed. It owns what runs beside the session's loop (the host event drain, hook
//! answers, handoffs, a streamed `say`'s feed); the decisions behind each command are voice's.

pub mod builder;
pub mod calls;
mod class;
mod options;
pub mod shared;
#[cfg(test)]
mod tests;
pub mod time;
pub mod transcript_sync;

pub use builder::*;
pub use calls::*;
pub use options::SessionOptions;
pub use shared::*;
pub use time::Timebase;

use crate::voice::nodes::Nodes;
use crate::voice::session::transcript_sync::TranscriptSynchronizer;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::resources::Resources;
use lk_agents_abi::voice::events::{CloseReason, HostWire};
use lk_agents_abi::voice::room_owner::RoomIo;
use lk_agents_abi::voice::session::{
    CommandRefused, OpenRefused, SessionRefused, StartFailed, UpdateAgentArgs,
};
use lk_agents_abi::voice::session_effects::{FillerReply, SessionEvents, SessionHooks};
use lk_agents_abi::voice::session_options::SessionConnOptions;
use lk_agents_abi::worker::remote_session::{SessionHosting, SessionPort};
use lk_agents_abi::{
    BoxFuture, HOST_TEARDOWN_TIMEOUT, HostRef, Llm, Maybe, Spawner, Stt, Tts, Vad,
};
use lk_agents_net::task::{TaskId, TaskSet};
use lk_agents_voice::agent_activity::options::{ConfigStack, LegSet};
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::lifecycle::{
    CloseAction, OpenAction, RecorderTarget, SessionLifecycle,
};
use lk_agents_voice::agent_session::nodes::OutputNodes;
use lk_agents_voice::agent_session::{AgentUpdate, Answer, Close, Command, DriverError, HookCall, HostEvent, HostRecorder, Origin, RecorderCommand, RequestId, SayText, SessionView, SpeechCall, SpeechCommand};
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::generation::Instructions;
use crate::voice::speech_handle::{SayStream, SpeechHandle, Speeches};
use lk_agents_voice::tool_executor::Toolset;
use lk_agents_voice::trace::{CoreStep, Step};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

/// The host's session handle: assembled at construction, opened once, commanded, and closed.
/// Startup owns the runtime resources; accepted core effects supply the readable views.
pub struct VoiceSession {
    /// `Arc`, so `open`'s future can write the running loop back after the borrow ended.
    stage: Arc<Mutex<Stage>>,
    view: tokio::sync::watch::Receiver<Arc<SessionView>>,
    /// Every speech's live handle, from its host call or creation to the session's end.
    speeches: Arc<Speeches>,
    room_io: Arc<dyn RoomIo>,
    /// The transcript synchroniser, held only so `close_io` can close it.
    synchronizer: Option<Arc<TranscriptSynchronizer>>,
    spawner: Arc<dyn Spawner>,
    /// What runs beside the loop: its drain, hook answers, handoffs, fed `say`s, a `shutdown`.
    tasks: Arc<TaskSet>,
    /// The driver's clock: `close` on a virtual clock moves it.
    time: Timebase,
    /// The last `with_filler` scope numbered by `filler_start`.
    last_filler: AtomicU64,
    /// Every `Agent` by its id, as the host installed it: the events naming an agent hand it
    /// back.
    agents: HostAgents,
    /// The monotonic-to-wall pair every `Ts`-carrying event converts against, read once so events
    /// never disagree about when `Ts::ZERO` was.
    anchor: lk_agents_abi::WallAnchor,
    host: HostParts,
    /// Every provider handed over is registered here; the driver performs against it.
    resources: Resources,
    /// `conn_options=`: every leg handed over, at start or later, is bound to its budget.
    conn: SessionConnOptions,
}

/// What a deterministic simulation needs from a session it drives, installed before `start()`.
#[cfg(feature = "dst")]
pub struct Probe {
    /// Every journal step, in order, from the first the session wrote.
    pub step: Box<dyn FnMut(&Step) + Send>,
    /// Every host event, typed, as the drain takes it and before the host sees its wire form.
    pub host_event: HostEventProbe,
}

#[cfg(feature = "dst")]
pub type HostEventProbe = Box<dyn FnMut(&HostEvent) + Send>;

/// Decided before assembly but opened at `open`, on the runtime (opening spawns).
type InterruptionPortFactory = Box<
    dyn FnOnce(
            lk_agents_abi::WallAnchor,
        )
            -> Arc<dyn lk_agents_voice::agent_activity::false_interruption::InterruptionPort>
        + Send,
>;

/// The host's last line against a runtime that stopped ticking (above the close machine's bounds).
const CLOSE_WATCHDOG: Duration = Duration::from_secs(30);
/// One frame of virtual time per real millisecond while a virtual-clock close runs.
const CLOSE_TICK: Duration = Duration::from_millis(20);
const CLOSE_TICK_BUDGET: u32 = 30_000;

enum Stage {
    Ready(Box<Ready>),
    Opening(Option<Arc<dyn SessionHosting>>),
    Running(Running),
    Closed(Option<Arc<dyn SessionHosting>>),
}

impl Stage {
    fn lifecycle(&self) -> SessionLifecycle {
        match self {
            Self::Ready(_) => SessionLifecycle::Ready,
            Self::Opening(_) => SessionLifecycle::Opening,
            Self::Running(_) => SessionLifecycle::Running,
            Self::Closed(_) => SessionLifecycle::Closed,
        }
    }
}

struct Ready {
    builder: DriverBuilder,
    /// The opening agent's `tts_node` and `stt_node`, run over the opening legs.
    nodes: Nodes,
    interruption: Option<InterruptionPortFactory>,
    #[cfg(feature = "dst")]
    probe: Option<HostEventProbe>,
}

struct Running {
    remote: Option<Arc<dyn SessionHosting>>,
    api: Tx,
    /// The replies of the calls the host awaits, until the core answers them.
    calls: Arc<Calls>,
    /// `session.keyterms`, as the driver last pushed them.
    keyterms: tokio::sync::watch::Receiver<Option<Vec<String>>>,
    /// `Agent.chat_ctx`: the driving agent's view, as the driver last published it.
    chat_ctx: tokio::sync::watch::Receiver<Arc<lk_agents_abi::ChatContext>>,
    peer: PeerHandle,
    /// One per session.
    amd: Option<Arc<dyn lk_agents_abi::voice::amd::AmdPort>>,
    /// The loop and the host event drain, cancelled once the close has drained.
    loop_tasks: Vec<TaskId>,
    /// Resolves when the host's drain has handed over the last event, so `close` handlers are
    /// queued before the close completes.
    drained: Option<tokio::sync::oneshot::Receiver<()>>,
    /// The loop's id counter: a speech call's handle has its id before the loop creates it.
    speech_ids: Arc<AtomicU64>,
    /// The executor's, for every TTS swapped in.
    tts_adapter: TtsAdapter,
    /// The executor's, for every STT swapped in.
    stt_adapter: SttAdapter,
}

impl Running {
    /// A host call's id, from the session's counter.
    fn next_call(&self) -> RequestId {
        RequestId(self.speech_ids.fetch_add(1, Ordering::Relaxed))
    }
}

/// Live inputs handed to the runtime when a remote session host is opened.
#[derive(Clone)]
pub struct RemoteSessionConfig {
    pub port: Arc<dyn SessionPort>,
    pub room: Option<Arc<dyn lk_agents_abi::voice::rtc::RtcRoom>>,
    pub console: Option<Arc<dyn lk_agents_abi::voice::console::ConsolePort>>,
}

/// What only a host-built session carries.
#[derive(Default)]
struct HostParts {
    events: Option<Arc<dyn SessionEvents>>,
    /// Without it the hook calls are the embedder's to answer (`Command::HookDone`).
    hooks: Option<Arc<dyn SessionHooks>>,
    /// Set once the close sequence ran; a second `aclose()` waits for the first.
    closed: Arc<tokio::sync::OnceCell<()>>,
    /// Handoffs in call order: one waits for the previous one's toolset setup.
    handoffs: Arc<tokio::sync::Mutex<()>>,
    /// The turn-detector stream, closed once the loop is gone (the driver's legs never owned it).
    eot: Option<Arc<dyn lk_agents_abi::EotDetector>>,
    /// `resolved_options()` answers `None` for a session built with no options object.
    has_options: bool,
    /// Where `AMD` awaits the callee: the call's room, `None` on an offline (console) session.
    call_room: Option<Arc<dyn lk_agents_abi::voice::rtc::RtcRoom>>,
    /// The remote session protocol this session serves, if it does.
    remote: Option<RemoteSessionConfig>,
    /// The job whose room the session runs in.
    job: Option<Arc<dyn lk_agents_abi::worker::job::SessionJob>>,
    /// `session.usage`: every metrics record the host is handed, folded per model.
    usage: Arc<Mutex<lk_agents_core::telemetry::usage::ModelUsageCollector>>,
}

/// The providers a host call hands over, registered.
struct ModelRefs {
    llm: Maybe<Option<lk_agents_abi::resources::LlmRef>>,
    tts: Maybe<Option<lk_agents_abi::resources::TtsRef>>,
    stt: Maybe<Option<lk_agents_abi::resources::SttRef>>,
    vad: Maybe<Option<lk_agents_abi::resources::VadRef>>,
}

type Leg<T> = Maybe<Option<Arc<T>>>;

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl VoiceSession {
    pub fn time(&self) -> &Timebase {
        &self.time
    }

    /// The scheduler supplied when the session was assembled.
    pub fn spawner(&self) -> Arc<dyn Spawner> {
        self.spawner.clone()
    }

    pub fn legs(&self) -> LegSet {
        self.view.borrow().config.legs
    }

    pub fn stack(&self) -> ConfigStack {
        self.view.borrow().config.stack
    }

    /// Every provider and leg the session still holds.
    #[cfg(feature = "dst")]
    pub fn resources(&self) -> &Resources {
        &self.resources
    }

    /// Every step the driver journals, from the first, to `observer`. Before `start()`.
    #[cfg(any(test, feature = "dst"))]
    pub fn observe(&self, observer: impl FnMut(&Step) + Send + 'static) {
        if let Stage::Ready(ready) = &mut *self.lock() {
            ready.builder.journal = lk_agents_voice::trace::Journal::observed(observer);
        }
    }

    #[cfg(feature = "dst")]
    pub fn probe(&self, Probe { step, host_event }: Probe) {
        if let Stage::Ready(ready) = &mut *self.lock() {
            ready.builder.journal = lk_agents_voice::trace::Journal::observed(step);
            ready.probe = Some(host_event);
        }
    }

    /// Refuses synchronously when there is nothing to open; does not wait for a participant.
    /// The loop and `drain` run in the session's task set, so `close` can cancel both.
    pub fn open(
        &self,
        mut executor: Box<dyn SessionExecutor>,
        remote: Option<Arc<dyn SessionHosting>>,
        drain: impl FnOnce(SessionRunner) -> BoxFuture<'static, ()> + Send + 'static,
    ) -> Result<impl Future<Output = Result<(), DriverError>> + Send + 'static, OpenRefused> {
        let ready = {
            let mut stage = self.lock();
            match stage.lifecycle().open()? {
                OpenAction::Begin => {
                    let Stage::Ready(ready) =
                        std::mem::replace(&mut *stage, Stage::Opening(remote.clone()))
                    else {
                        unreachable!("opening consumes ready resources")
                    };
                    *ready
                }
                OpenAction::InstallRunning => unreachable!("an open request begins startup"),
            }
        };

        let room_io = Arc::clone(&self.room_io);
        let (spawner, tasks) = (Arc::clone(&self.spawner), Arc::clone(&self.tasks));
        let stage = Arc::clone(&self.stage);
        let anchor = self.anchor;
        let time = self.time.clone();
        let speeches = Arc::clone(&self.speeches);
        Ok(async move {
            #[cfg(feature = "dst")]
            let Ready { mut builder, nodes, interruption, probe } = ready;
            #[cfg(not(feature = "dst"))]
            let Ready { mut builder, nodes, interruption } = ready;
            if let Some(open_port) = interruption {
                builder.interruption = Some(open_port(anchor));
            }

            // The room's own IO first; its events join the driver's queue, one stream, one order.
            room_io.clone().start();
            builder.room_io_events = Some(room_io.clone());

            let tts_adapter = executor.tts_adapter();
            let stt_adapter = executor.stt_adapter();
            prepare_opening_legs(&mut builder, &nodes, &tts_adapter, &stt_adapter)?;
            let (cores, handle) = builder.start(&mut *executor).await?;
            let SessionHandle {
                api,
                events,
                keyterms,
                chat_ctx,
                peer,
                ivr_detection: _,
                speech_ids,
                calls,
            } = handle;
            let mut owner = executor.own(cores);
            // The opening cycle: `open` resolves only once the drain took its events, so a host's
            // `start()` sees `listening` dispatched first.
            owner.open(time.read());
            let (opening_tx, opening_rx) = tokio::sync::oneshot::channel();
            let opening = Opening { left: events.len(), delivered: Some(opening_tx) };

            let mut loop_tasks = Vec::new();
            if let Some(run) = owner.run() {
                loop_tasks.push(tasks.spawn(spawner.as_ref(), "voice session loop", run));
            }
            // Drained whether or not anybody listens: the queue is unbounded.
            let (drained_tx, drained) = tokio::sync::oneshot::channel();
            let runner = SessionRunner {
                events,
                speeches,
                room_io: Arc::clone(&room_io),
                api: api.clone(),
                drained: Some(drained_tx),
                opening,
                #[cfg(feature = "dst")]
                host_event: probe,
            };
            loop_tasks.push(tasks.spawn(spawner.as_ref(), "host event drain", drain(runner)));

            {
                let mut stage = lock(&stage);
                match stage.lifecycle().opened() {
                    OpenAction::InstallRunning => {
                        *stage = Stage::Running(Running {
                            api,
                            calls,
                            keyterms,
                            chat_ctx,
                            peer,
                            amd: None,
                            remote,
                            loop_tasks,
                            drained: Some(drained),
                            speech_ids,
                            tts_adapter,
                            stt_adapter,
                        });
                    }
                    OpenAction::Begin => {
                        unreachable!("startup completion installs running resources")
                    }
                }
            }
            // A drain that returned without reading drops the sender: nothing to wait for.
            let _ = opening_rx.await;
            Ok(())
        })
    }

    /// Opens the providers through `executor` and runs it beside the host event drain: one event
    /// per call, in driver order; a hook call handed to the host loop in that order, then awaited
    /// on its own task. The job's room connects beside the opening; the remote session starts
    /// once both are done, so what it queued meanwhile goes out on a connected room.
    pub fn start_with(
        &self,
        executor: Box<dyn SessionExecutor>,
    ) -> Result<BoxFuture<'static, Result<(), StartFailed>>, OpenRefused> {
        let job = self.host.job.clone();
        let remote = self.host.remote.clone().map(|config| {
            crate::voice::remote_session::hosting(config, self.time.clone(), self.spawner.clone())
        });
        let drain = HostDrain {
            events: self.host.events.clone(),
            hooks: self.host.hooks.clone(),
            remote: remote.clone(),
            usage: Arc::clone(&self.host.usage),
            anchor: self.anchor,
            agents: self.agents.clone(),
            spawner: Arc::clone(&self.spawner),
            tasks: Arc::clone(&self.tasks),
        };
        let opening =
            self.open(executor, remote.clone(), move |runner| Box::pin(drain.run(runner)))?;
        Ok(Box::pin(async move {
            let connect = async {
                match &job {
                    Some(job) => job.connect().await,
                    None => Ok(()),
                }
            };
            let (opened, connected) = tokio::join!(opening, connect);
            opened.map_err(|e| StartFailed::Driver(e.to_string()))?;
            connected.map_err(StartFailed::Host)?;
            if let Some(remote) = remote {
                remote.start().await.map_err(StartFailed::Host)?;
            }
            Ok(())
        }))
    }

    /// `None` detaches. Safe before or after `open`.
    pub fn attach_recorder(
        &self,
        recorder: Option<Arc<dyn lk_agents_abi::recorder::AudioRecorder>>,
    ) {
        let mut stage = self.lock();
        match (stage.lifecycle().recorder_target(), &mut *stage) {
            (RecorderTarget::Builder, Stage::Ready(ready)) => ready.builder.recorder = recorder,
            (RecorderTarget::Running, Stage::Running(running)) => {
                let cmd = match recorder {
                    Some(tap) => RecorderCommand::Attach(HostRecorder(tap)),
                    None => RecorderCommand::Detach,
                };
                let _ = running.api.send(Command::Recorder(cmd));
            }
            _ => {}
        }
    }

    /// `false`: the loop did not take it (not started or closed), and the image is dropped.
    pub fn push_video(&self, image: lk_agents_abi::ImageContent) -> bool {
        self.running(|running| Ok(running.api.send(Command::Video(image)).is_ok())).unwrap_or(false)
    }

    /// Refused before `start()` and after the close.
    fn running<R>(
        &self,
        f: impl FnOnce(&mut Running) -> Result<R, SessionRefused>,
    ) -> Result<R, SessionRefused> {
        let mut stage = self.lock();
        stage.lifecycle().running()?;
        let Stage::Running(running) = &mut *stage else { unreachable!("running resources") };
        f(running)
    }

    fn remote_host(&self) -> Option<Arc<dyn SessionHosting>> {
        match &*self.lock() {
            Stage::Opening(remote) | Stage::Closed(remote) => remote.clone(),
            Stage::Ready(_) => None,
            Stage::Running(running) => running.remote.clone(),
        }
    }

    /// The live handle of speech `id`, whenever the session held it.
    pub fn speech(&self, id: SpeechId) -> Option<Arc<SpeechHandle>> {
        self.speeches.get(id)
    }

    pub fn api(&self) -> Result<Tx, SessionRefused> {
        self.running(|r| Ok(r.api.clone()))
    }

    /// A call to await: its id goes in the command, which carries its reply; `pick` takes its
    /// answer.
    pub fn call<T>(
        &self,
        pick: fn(Answer) -> Option<T>,
    ) -> Result<(Call, Awaited<T>), SessionRefused> {
        self.running(|r| Ok(open(r.next_call(), pick)))
    }

    /// A `say()` / `generate_reply()`: the handle the host gets now, with its id and the
    /// interruptibility it is created with (`allow_interruptions`, else the options' as they
    /// stand), and the call that brings it to the loop.
    pub fn speech_call(
        &self,
        allow_interruptions: Option<bool>,
    ) -> Result<(Arc<SpeechHandle>, SpeechCall, Call), SessionRefused> {
        let allow = allow_interruptions
            .unwrap_or_else(|| self.stack().resolve(&self.legs()).turn.interruption.enabled());
        self.running(|r| {
            let id = SpeechId(r.speech_ids.fetch_add(1, Ordering::Relaxed));
            let speech = SpeechHandle::pending(r.api.clone(), id, allow);
            self.speeches.register(Arc::clone(&speech));
            let reply_to = ReplyTo::Speech(SpeechReply(Arc::clone(&speech)));
            let call = Call { id: r.next_call(), reply_to };
            Ok((speech, SpeechCall { id: call.id, speech: id }, call))
        })
    }

    pub fn send(&self, cmd: impl Into<Command>) -> Result<(), SessionRefused> {
        self.send_from(Origin::Host, cmd)
    }

    pub fn send_from(&self, origin: Origin, cmd: impl Into<Command>) -> Result<(), SessionRefused> {
        self.send_command(origin, cmd.into(), None)
    }

    /// The command `make` builds around a new call's id, sent with its reply; `pick` takes the
    /// answer.
    pub fn ask<T, C: Into<Command>>(
        &self,
        origin: Origin,
        pick: fn(Answer) -> Option<T>,
        make: impl FnOnce(RequestId) -> C,
    ) -> Result<Awaited<T>, SessionRefused> {
        let (call, answer) = self.call(pick)?;
        self.send_awaited(origin, make(call.id), call)?;
        Ok(answer)
    }

    /// A command naming `call`, sent with its reply.
    pub fn send_awaited(
        &self,
        origin: Origin,
        cmd: impl Into<Command>,
        call: Call,
    ) -> Result<(), SessionRefused> {
        self.send_command(origin, cmd.into(), Some(call))
    }

    /// A command the loop never takes releases the providers it hands over.
    fn send_command(
        &self,
        origin: Origin,
        cmd: Command,
        call: Option<Call>,
    ) -> Result<(), SessionRefused> {
        let handed = cmd.resources();
        let sent = self.running(|r| {
            r.calls.send_command(&r.api, origin, cmd, call).map_err(|_| SessionRefused::LoopEnded)
        });
        if sent.is_err() {
            handed.into_iter().for_each(|id| self.resources.release(id));
        }
        sent
    }

    /// `AMD._run`. One per session; the verdict comes back as `amd_result`.
    pub fn attach_amd(&self, setup: lk_agents_voice::amd::AmdSetup) -> Result<(), SessionRefused> {
        self.running(|r| {
            if r.amd.is_some() {
                r.peer.journal(Step::Core(CoreStep::PortFailure("amd: already attached")));
            } else {
                r.amd = Some(crate::voice::amd::spawn(r.peer.clone(), setup));
            }
            Ok(())
        })
    }

    /// `_aclose_impl`: resolves once the loop returned and the drain handed over the last event.
    /// The stage stays `Running` meanwhile (hooks and tools still answer through it). Idempotent.
    pub fn close(
        &self,
        reason: CloseReason,
        drain: bool,
        error: Option<lk_agents_voice::events::ErrorEvent>,
    ) -> impl Future<Output = ()> + Send + 'static {
        let running = {
            let mut stage = self.lock();
            match (stage.lifecycle().close(), &mut *stage) {
                (CloseAction::Running, Stage::Running(running)) => Some((
                    running.api.clone(),
                    running.drained.take(),
                    running.loop_tasks.clone(),
                    running.remote.clone(),
                )),
                (CloseAction::Opening, Stage::Opening(remote)) => {
                    *stage = Stage::Closed(remote.clone());
                    None
                }
                (CloseAction::None, _) => None,
                _ => unreachable!("the close acts on the stage it was decided on"),
            }
        };
        let (stage, tasks) = (Arc::clone(&self.stage), Arc::clone(&self.tasks));
        let time = self.time.clone();
        async move {
            let Some((api, drained, loop_tasks, remote)) = running else { return };
            let _ = api.send(Command::Close(Close { reason, drain, error }));
            // Every close phase is bounded on the session clock; this watchdog is a backstop. On a
            // virtual clock the host stopped advancing, so the clock is moved here.
            if time.is_virtual() {
                advance_until(&time, api.closed()).await;
            } else {
                let _ = time.timeout(CLOSE_WATCHDOG, api.closed()).await;
            }
            // `Closed` **before** awaiting the drain: `Running` holds the `PeerHandle`, a sender on
            // the host stream, and the drain only ends when every sender is gone.
            {
                let mut stage = lock(&stage);
                match stage.lifecycle().closed() {
                    CloseAction::Closed => *stage = Stage::Closed(remote),
                    _ => unreachable!("close completion installs closed resources"),
                }
            }
            // Then the drain: the host's `close` is queued before this future completes.
            if let Some(drained) = drained {
                let _ = time.timeout(HOST_TEARDOWN_TIMEOUT, drained).await;
            }
            loop_tasks.into_iter().for_each(|task| tasks.cancel(task));
        }
    }

    /// After the host's `close` handlers ran. The synchroniser first, or its rotation tasks publish
    /// into a closed stream.
    pub fn close_io(&self) -> impl Future<Output = ()> + Send + 'static {
        let room_io = Arc::clone(&self.room_io);
        let synchronizer = self.synchronizer.clone();
        let time = self.time.clone();
        async move {
            if let Some(sync) = &synchronizer {
                let _ = time.timeout(HOST_TEARDOWN_TIMEOUT, sync.aclose()).await;
            }
            let _ = time.timeout(HOST_TEARDOWN_TIMEOUT, room_io.aclose()).await;
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Stage> {
        lock(&self.stage)
    }

    /// Given present, given `None` clears, not-given keeps. Each provider given is wrapped, under
    /// the agent's node when it overrides one, and registered: the command carries its reference.
    fn swap(
        &self,
        llm: Leg<dyn Llm>,
        tts: Leg<dyn Tts>,
        stt: Leg<dyn Stt>,
        vad: Leg<dyn Vad>,
        nodes: &Nodes,
    ) -> ModelRefs {
        fn map<T: ?Sized, R>(leg: Leg<T>, f: impl FnOnce(Arc<T>) -> R) -> Maybe<Option<R>> {
            match leg {
                Maybe::Given(given) => Maybe::Given(given.map(f)),
                Maybe::NotGiven => Maybe::NotGiven,
            }
        }
        let clock = self.time.clock();
        let r = &self.resources;
        // Before start nothing reaches the loop (the command is refused and the leg released).
        let adapters = self.running(|r| Ok((r.tts_adapter.clone(), r.stt_adapter.clone()))).ok();
        let vad_now = match &vad {
            Maybe::Given(given) => given.clone(),
            Maybe::NotGiven => {
                let id = self.view.borrow().vad.as_ref().map(|v| v.id);
                id.and_then(|id| self.resources.vad(id))
            }
        };
        ModelRefs {
            llm: map(llm, |l| r.register_llm(nodes.llm(llm_leg(l, self.conn.llm)))),
            tts: map(tts, |t| {
                let t = match &adapters {
                    Some((adapter, _)) => tts_leg(t, adapter, clock.clone(), self.conn.tts),
                    None => t,
                };
                r.register_tts(nodes.tts(t))
            }),
            stt: map(stt, |s| {
                let s = match &adapters {
                    Some((_, adapter)) => {
                        stt_leg(s, adapter, vad_now, clock.clone(), self.conn.stt)
                    }
                    None => s,
                };
                r.register_stt(nodes.stt(s))
            }),
            vad: map(vad, |v| r.register_vad(v)),
        }
    }

    /// The agent `frame` describes, its providers registered, as `update_agent` installs it.
    pub fn frame(&self, frame: UpdateAgentArgs) -> AgentUpdate {
        let UpdateAgentArgs {
            instructions,
            dynamic_instructions,
            agent,
            agent_host,
            tools,
            tools_host,
            voice_agent,
            chat_ctx,
            llm,
            tts,
            stt,
            vad,
            turn_handling,
            min_consecutive_speech_delay,
            use_tts_aligned_transcript,
            expressive,
            llm_node,
            tts_node,
            stt_node,
            transcription_node,
            realtime_audio_output_node,
        } = frame;
        self.agents.install(&agent, agent_host);
        let options = lk_agents_voice::turn::agent_layer(
            turn_handling.as_ref(),
            min_consecutive_speech_delay,
            use_tts_aligned_transcript,
        );
        let nodes = Nodes { llm: llm_node, tts: tts_node, stt: stt_node };
        let ModelRefs { llm, tts, stt, vad } = self.swap(llm, tts, stt, vad, &nodes);
        AgentUpdate {
            agent,
            instructions: Instructions::plain(instructions),
            dynamic_instructions,
            toolset: Toolset {
                specs: tools.unwrap_or_default(),
                host: tools_host,
                agent: Some(voice_agent),
            },
            chat_ctx,
            llm,
            tts,
            stt,
            vad,
            options,
            expressive: expressive.given(),
            output_nodes: OutputNodes {
                transcription: transcription_node,
                realtime_audio: realtime_audio_output_node,
            },
            call: None,
        }
    }

    /// `update`'s toolsets set up first, in call order: the turn is taken now. Then the command
    /// `wrap` makes of it is sent as `origin`.
    fn send_frame(
        &self,
        origin: Origin,
        mut update: AgentUpdate,
        call: Call,
        wrap: impl FnOnce(AgentUpdate) -> Command + Send + 'static,
    ) -> Result<(), SessionRefused> {
        let (Some(hooks), Some(tools)) = (self.host.hooks.clone(), update.toolset.host.clone())
        else {
            return self.send_awaited(origin, wrap(update), call);
        };
        let (api, calls) = self.running(|r| Ok((r.api.clone(), Arc::clone(&r.calls))))?;
        let resources = self.resources.clone();
        // The turn is queued for now, in call order: the lock's waiter list is entered by this
        // first poll (a tokio lock takes the real waker at the next one). Only the lock is polled
        // here; the setup's host bridge runs in the task, with its own waker.
        let mut turn = Box::pin(Arc::clone(&self.host.handoffs).lock_owned());
        let held =
            turn.as_mut().poll(&mut std::task::Context::from_waker(std::task::Waker::noop()));
        let handoff = async move {
            let _turn = match held {
                std::task::Poll::Ready(guard) => guard,
                std::task::Poll::Pending => turn.await,
            };
            let set_up = hooks.setup_tools(&tools).await.unwrap_or_else(|e| {
                hooks.hook_failed("setup_tools", e.host_ref.clone());
                Vec::new()
            });
            let toolset = std::mem::take(&mut update.toolset);
            update.toolset = Toolset::set_up(set_up, toolset.host, toolset.agent);
            let cmd = wrap(update);
            let handed = cmd.resources();
            if calls.send_command(&api, origin, cmd, Some(call)).is_err() {
                handed.into_iter().for_each(|id| resources.release(id));
            }
        };
        self.tasks.spawn(self.spawner.as_ref(), "handoff", handoff);
        Ok(())
    }

    fn speak(
        &self,
        (text, stream): (SayText, Option<SayStream>),
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<String>,
    ) -> Result<Arc<SpeechHandle>, CommandRefused> {
        let origin = origin_of(origin)?;
        let (speech, call, reply) = self.speech_call(allow_interruptions)?;
        // The handle holds what `say()` streams until its speech's task takes it.
        if let Some(stream) = stream {
            speech.hold_say(stream);
        }
        let call = Some(call);
        self.send_awaited(
            origin,
            SpeechCommand::Say { text, allow_interruptions, add_to_chat_ctx, call },
            reply,
        )?;
        Ok(speech)
    }

    /// A `say` of streamed words: the speech and its feeding end.
    pub fn say_text_channel(
        &self,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<String>,
    ) -> Result<(Arc<SpeechHandle>, Pusher<String>), CommandRefused> {
        // Backpressure against a runaway producer, not a buffer.
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let said = (SayText::Stream, Some(SayStream::Text(rx)));
        let speech = self.speak(said, allow_interruptions, add_to_chat_ctx, origin)?;
        Ok((speech, Pusher(Mutex::new(Some(tx)))))
    }

    /// A `say` of pre-synthesised frames, `text` still committed: the speech and its feeding end.
    pub fn say_audio_channel(
        &self,
        text: String,
        allow_interruptions: Option<bool>,
        add_to_chat_ctx: bool,
        origin: Option<String>,
    ) -> Result<(Arc<SpeechHandle>, Pusher<lk_agents_abi::AudioFrame>), CommandRefused> {
        // Deeper than the text channel: at 20 ms a frame, 64 would be 1.3 s of audio.
        let (tx, rx) = tokio::sync::mpsc::channel(512);
        let said = (SayText::Audio { text }, Some(SayStream::Audio(rx)));
        let speech = self.speak(said, allow_interruptions, add_to_chat_ctx, origin)?;
        Ok((speech, Pusher(Mutex::new(Some(tx)))))
    }

    /// `source` read into `pusher` as one of the session's tasks.
    fn feed<T: Send + 'static>(
        &self,
        pusher: Pusher<T>,
        source: Box<dyn lk_agents_abi::StreamSource<T>>,
    ) {
        let clock = self.time.clock();
        self.tasks.spawn(self.spawner.as_ref(), "say feed", pusher.feed(source, clock));
    }

    /// DTMF collection, run by the driver. Answers once, with the outcome (`kind`: `complete`,
    /// `incomplete` or `stopped`).
    pub fn dtmf_start(
        &self,
        num_digits: usize,
        ask_for_confirmation: bool,
        dtmf_input_timeout: f64,
        dtmf_stop_event: String,
    ) -> Result<BoxFuture<'static, lk_agents_abi::voice::workflows::DtmfOutcomeWire>, CommandRefused>
    {
        use lk_agents_voice::agent_session::DtmfCommand;
        use lk_agents_voice::workflows::dtmf::{DtmfCollector, DtmfEvent, DtmfOutcome};
        let stop = DtmfEvent::parse(&dtmf_stop_event).ok_or_else(|| {
            CommandRefused::Invalid(format!("{dtmf_stop_event:?} is not a DTMF key"))
        })?;
        let timeout = Duration::try_from_secs_f64(dtmf_input_timeout)
            .map_err(|e| CommandRefused::Invalid(format!("dtmf_input_timeout: {e}")))?;
        let collector = DtmfCollector::new(num_digits, ask_for_confirmation, timeout, stop)
            .map_err(|e| CommandRefused::Invalid(e.to_string()))?;
        let rx = self.ask(Origin::Host, Answer::dtmf, |call| DtmfCommand::Start {
            collector,
            call: Some(call),
        })?;
        Ok(Box::pin(async move { rx.await.unwrap_or(DtmfOutcome::Stopped).wire() }))
    }

    /// Whether the last DTMF collection is inside its debounced reply.
    pub fn dtmf_reply_running(&self) -> bool {
        self.view.borrow().dtmf_reply_running
    }

    /// A metrics record only a plugin can measure, emitted as `metrics_collected`. Engine-computed
    /// kinds are refused (they would count twice).
    pub fn host_metrics(
        &self,
        record: lk_agents_abi::metrics::base::AgentMetrics,
    ) -> Result<(), CommandRefused> {
        use lk_agents_abi::metrics::base::AgentMetrics as M;
        match &record {
            M::Llm(_) | M::Tts(_) | M::Stt(_) | M::Eou(_) => {
                return Err(CommandRefused::Invalid(format!(
                    "host_metrics: {} records are the engine's own; handing one in would count it twice",
                    record.type_literal()
                )));
            }
            M::Vad(_)
            | M::EotInference(_)
            | M::RealtimeModel(_)
            | M::Interruption(_)
            | M::Avatar(_) => {}
        }
        Ok(self.send(Command::HostMetrics(Box::new(record)))?)
    }
}

/// For a host that stopped moving time (awaiting `aclose()`). Paced on real time (1 ms per step) so
/// host tasks can run between steps.
async fn advance_until(time: &Timebase, done: impl Future<Output = ()>) {
    let pacer = lk_agents_net::clock::RealClock::new();
    let mut done = std::pin::pin!(done);
    for _ in 0..=CLOSE_TICK_BUDGET {
        let tick = lk_agents_abi::Clock::wait_until(
            &pacer,
            lk_agents_abi::Clock::now(&pacer).saturating_add(Duration::from_millis(1)),
        );
        tokio::select! {
            biased;
            () = &mut done => return,
            () = tick => time.advance(CLOSE_TICK),
        }
    }
}

fn origin_of(origin: Option<String>) -> Result<Origin, CommandRefused> {
    origin.map_or(Ok(Origin::Host), |o| o.parse().map_err(CommandRefused::Invalid))
}

/// The `session_usage_updated` a `metrics_collected` record brings, once folded into `usage`. The
/// end-of-utterance and avatar records are reported beside the providers' (no usage follows them).
fn session_usage_updated(
    usage: &mut lk_agents_core::telemetry::usage::ModelUsageCollector,
    wire: &HostWire,
) -> Option<HostWire> {
    use lk_agents_abi::metrics::base::AgentMetrics;
    use lk_agents_abi::metrics::usage::UsageCollector as _;
    let HostWire::MetricsCollected(m) = wire else { return None };
    if matches!(m.metrics, AgentMetrics::Eou(_) | AgentMetrics::Avatar(_)) {
        return None;
    }
    usage.collect(&m.metrics);
    let model_usage = usage.flatten();
    Some(HostWire::SessionUsageUpdated(lk_agents_abi::voice::events::SessionUsageWire {
        model_usage,
    }))
}

/// One stream, in emission order, so ordering between items, handoffs and completions needs no
/// barrier. Dropping the runner releases a waiting [`close`](VoiceSession::close).
pub struct SessionRunner {
    events: tokio::sync::mpsc::UnboundedReceiver<HostEvent>,
    /// The host's objects for the speeches the events name.
    speeches: Arc<Speeches>,
    room_io: Arc<dyn RoomIo>,
    api: Tx,
    drained: Option<tokio::sync::oneshot::Sender<()>>,
    opening: Opening,
    #[cfg(feature = "dst")]
    host_event: Option<HostEventProbe>,
}

struct Opening {
    left: usize,
    delivered: Option<tokio::sync::oneshot::Sender<()>>,
}

impl SessionRunner {
    /// `None` once the loop has returned. The room's transcript leg is fed before the event is
    /// handed back so one utterance's segments land in order.
    pub async fn next_host_event(&mut self) -> Option<HostEvent> {
        // Asked again: the event handed back last was handled by the host.
        if self.opening.left == 0
            && let Some(delivered) = self.opening.delivered.take()
        {
            let _ = delivered.send(());
        }
        let ev = self.events.recv().await?;
        self.opening.left = self.opening.left.saturating_sub(1);
        #[cfg(feature = "dst")]
        if let Some(observe) = &mut self.host_event {
            observe(&ev);
        }
        // What the room I/O mirrors: the agent's state edge and the user's transcript.
        match &ev {
            HostEvent::Session(SessionEvent::AgentStateChanged { new, .. }) => {
                self.room_io.clone().on_agent_state_changed(new.as_str())
            }
            HostEvent::Session(SessionEvent::UserInputTranscribed(t)) => {
                let text = t.ev.best_alternative().map(|a| a.text.as_str()).unwrap_or_default();
                let _ = self.room_io.on_user_input_transcribed(text, t.is_final).await;
            }
            _ => {}
        }
        Some(ev)
    }

    pub fn api(&self) -> &Tx {
        &self.api
    }
}

impl Drop for SessionRunner {
    fn drop(&mut self) {
        if let Some(drained) = self.drained.take() {
            let _ = drained.send(());
        }
    }
}

/// The host's side of the event stream: each event serialised to the host, each hook call answered.
struct HostDrain {
    events: Option<Arc<dyn SessionEvents>>,
    hooks: Option<Arc<dyn SessionHooks>>,
    remote: Option<Arc<dyn SessionHosting>>,
    usage: Arc<Mutex<lk_agents_core::telemetry::usage::ModelUsageCollector>>,
    anchor: lk_agents_abi::WallAnchor,
    agents: HostAgents,
    spawner: Arc<dyn Spawner>,
    tasks: Arc<TaskSet>,
}

impl HostDrain {
    async fn run(self, mut runner: SessionRunner) {
        let mut answering = Vec::new();
        while let Some(ev) = runner.next_host_event().await {
            match ev {
                HostEvent::Hook(call) => {
                    if let (Some(remote), HookCall::FunctionToolsExecuted { event, .. }) =
                        (&self.remote, &call)
                    {
                        remote.forward_tools_executed(event);
                    }
                    let Some(hooks) = &self.hooks else { continue };
                    let answer = answer_hook(
                        Arc::clone(hooks),
                        call,
                        self.agents.clone(),
                        runner.api().clone(),
                        Arc::clone(&self.spawner),
                        Arc::clone(&self.tasks),
                    );
                    // The host call is made by this first poll, in event order; the rest of the
                    // answer runs on its own task.
                    let mut answer = Box::pin(answer);
                    let first =
                        std::future::poll_fn(|cx| std::task::Poll::Ready(answer.as_mut().poll(cx)))
                            .await;
                    if first.is_pending() {
                        answering.retain(|task| self.tasks.contains(*task));
                        answering.push(self.tasks.spawn(
                            self.spawner.as_ref(),
                            "hook answer",
                            answer,
                        ));
                    }
                }
                ev => {
                    let speeches = Arc::clone(&runner.speeches);
                    let Some(wire) = ev.wire(self.anchor, &|id| speeches.object(id)) else {
                        continue;
                    };
                    // `_on_metrics_collected`: a record is folded into the session's usage, then
                    // `session_usage_updated` follows it.
                    let updated = session_usage_updated(&mut lock(&self.usage), &wire);
                    for wire in std::iter::once(wire).chain(updated) {
                        self.deliver(wire);
                    }
                }
            }
        }
        answering.into_iter().for_each(|task| self.tasks.cancel(task));
        // Dropping the runner tells `aclose()` the host has seen everything.
        drop(runner);
        // However the session closed, its remote session host closes with it.
        if let Some(remote) = self.remote {
            remote.aclose().await;
        }
    }

    fn deliver(&self, wire: HostWire) {
        match (&self.remote, &self.events, &wire) {
            (Some(remote), Some(events), HostWire::Error(error)) => {
                // The host's record of the error, else the error's own text.
                let message = events.error_message(error).unwrap_or_else(|e| {
                    log::warn!(target: "livekit.agents", error = e.to_string().as_str(); "error record unreadable");
                    error.error.to_string()
                });
                remote.forward_error(message);
            }
            (Some(remote), ..) => remote.forward(&wire),
            (None, ..) => {}
        }
        if let Some(events) = &self.events {
            events.event(wire);
        }
    }
}

/// Runs the hook `call` asks for through `hooks` and answers it. Calls the host first, before any
/// other await, so a caller polling it once has handed the call to the host loop in event order.
async fn answer_hook(
    hooks: Arc<dyn SessionHooks>,
    call: HookCall,
    agents: HostAgents,
    api: Tx,
    spawner: Arc<dyn Spawner>,
    tasks: Arc<TaskSet>,
) {
    use lk_agents_voice::agent::TurnEdit;
    use lk_agents_voice::agent_session::{HookId, HookOutcome};
    let id = call.id();
    let kind = call.kind();
    let outcome: Result<HookOutcome, HostError> = match call {
        HookCall::OnEnter { agent, .. } => match agents.get(&agent) {
            Some(agent) => hooks.on_enter(&agent).await.map(|()| HookOutcome::Returned),
            None => Ok(HookOutcome::Failed),
        },
        HookCall::OnExit { agent, .. } => match agents.get(&agent) {
            Some(agent) => {
                let outcome = hooks.on_exit(&agent).await.map(|()| HookOutcome::Returned);
                // The toolsets close after the answer, and outlive the hook's cancellation.
                let hooks = Arc::clone(&hooks);
                let close = async move { hooks.close_toolsets(Some(&agent)).await };
                tasks.spawn(spawner.as_ref(), "agent toolsets close", close);
                outcome
            }
            None => Ok(HookOutcome::Failed),
        },
        HookCall::OnUserTurnCompleted {
            agent,
            chat_ctx,
            new_transcript,
            transcript_confidence,
            ..
        } => match agents.get(&agent) {
            Some(agent) => hooks
                .on_user_turn_completed(&agent, chat_ctx, &new_transcript, transcript_confidence)
                .await
                .map(|(user_text, items, stop)| {
                    let mut edits = Vec::new();
                    if stop {
                        edits.push(TurnEdit::StopResponse);
                    }
                    if let Some(user_text) = user_text.filter(|t| *t != new_transcript) {
                        edits.push(TurnEdit::UpdateUserMessage(user_text));
                    }
                    if let Some(items) = items {
                        edits.push(TurnEdit::ReplaceItems(items));
                    }
                    HookOutcome::Turn(edits)
                }),
            None => Ok(HookOutcome::Failed),
        },
        HookCall::OnUserTurnExceeded { agent, event, .. } => match agents.get(&agent) {
            Some(agent) => {
                hooks.on_user_turn_exceeded(&agent, &event).await.map(|()| HookOutcome::Returned)
            }
            None => Ok(HookOutcome::Failed),
        },
        HookCall::FunctionToolsExecuted { event, .. } => {
            use lk_agents_abi::ChatItem;
            let calls = event.function_calls.into_iter().map(ChatItem::FunctionCall).collect();
            let outputs =
                event.function_call_outputs.into_iter().map(ChatItem::FunctionCallOutput).collect();
            hooks
                .function_tools_executed(calls, outputs)
                .await
                .map(|reply_required| HookOutcome::FunctionToolsExecuted { reply_required })
        }
        // Unanswered: the reference's templates.
        HookCall::FillerSource { source: None, .. } => {
            Ok(HookOutcome::Filler { text: None, spoke: false })
        }
        HookCall::FillerSource { source: Some(source), step, .. } => {
            hooks.filler_source(&source, step).await.map(|reply| match reply {
                FillerReply::Text(text) => HookOutcome::Filler { text: Some(text), spoke: false },
                FillerReply::Spoke => HookOutcome::Filler { text: None, spoke: true },
                FillerReply::Declined => HookOutcome::Filler { text: None, spoke: false },
            })
        }
    };
    let outcome = outcome.unwrap_or_else(|e| {
        hooks.hook_failed(kind.as_str(), e.host_ref.clone());
        HookOutcome::Failed
    });
    let _ = api.send_from(Origin::Host, Command::HookDone { id: HookId(id.0), outcome });
}

/// The host's `Agent` objects by agent id. The hooks and the handoff record name an agent by
/// id; the host reads the object it installed.
#[derive(Clone, Default)]
struct HostAgents(Arc<Mutex<std::collections::HashMap<String, HostRef>>>);

impl HostAgents {
    fn install(&self, agent: &str, host: Option<HostRef>) {
        if let Some(host) = host {
            lock(&self.0).insert(agent.to_owned(), host);
        }
    }

    fn get(&self, agent: &str) -> Option<HostRef> {
        lock(&self.0).get(agent).cloned()
    }
}

/// The feeding end of a streamed `say`: what the speech reads, chunk by chunk. Dropping it, or
/// [`end`](Self::end), finishes the speech.
pub struct Pusher<T>(Mutex<Option<tokio::sync::mpsc::Sender<T>>>);

impl<T: Send + 'static> Pusher<T> {
    /// Queue one item. A full channel waits rather than dropping (every chunk is spoken, in
    /// order). `false` once the speech is gone.
    pub async fn push(&self, item: T) -> bool {
        let tx = lock(&self.0).clone();
        match tx {
            Some(tx) => tx.send(item).await.is_ok(),
            None => false,
        }
    }

    /// No more items. Idempotent.
    pub fn end(&self) {
        lock(&self.0).take();
    }

    /// Resolves once the speech stopped reading.
    async fn closed(&self) {
        let tx = lock(&self.0).clone();
        if let Some(tx) = tx {
            tx.closed().await;
        }
    }

    /// `source` read into the speech to its end, or until the speech is gone; then closed. A
    /// failing source ends the speech with what it gave.
    async fn feed(
        self,
        mut source: Box<dyn lk_agents_abi::StreamSource<T>>,
        clock: Arc<dyn lk_agents_abi::Clock>,
    ) {
        loop {
            let next = tokio::select! {
                biased;
                next = source.next() => next,
                () = self.closed() => break,
            };
            match next {
                Ok(Some(item)) => {
                    if !self.push(item).await {
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    log::error!("say(): the stream it speaks raised: {e}");
                    break;
                }
            }
        }
        self.end();
        source.close(clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
    }
}
